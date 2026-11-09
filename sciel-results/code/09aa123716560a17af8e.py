"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

# Precompute all K4s
_N = 16
_K4S = list(combinations(range(_N), 4))
_K4_EDGES = []
for quad in _K4S:
    edges = []
    for a in range(4):
        for b in range(a + 1, 4):
            i, j = quad[a], quad[b]
            edges.append((i, j))
    _K4_EDGES.append(edges)

# Precompute for each edge, which K4s contain it
_EDGE_TO_K4S = [[] for _ in range(_N * (_N - 1) // 2)]
def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (_N - 1) - i * (i + 1) // 2 + (j - i - 1)

for k4_idx, edges in enumerate(_K4_EDGES):
    for (i, j) in edges:
        _EDGE_TO_K4S[_edge_idx(i, j)].append(k4_idx)

# For each K4, precompute the 6 edge indices
_K4_EDGE_IDXS = []
for edges in _K4_EDGES:
    _K4_EDGE_IDXS.append([_edge_idx(i, j) for (i, j) in edges])

_NUM_K4S = len(_K4S)
_NUM_EDGES = _N * (_N - 1) // 2  # 120

def _distinct_count(c0, c1, c2, c3, c4, c5):
    """Count distinct values among 6 colors."""
    distinct = 1
    if c1 != c0: distinct += 1
    if c2 != c0 and c2 != c1: distinct += 1
    if c3 != c0 and c3 != c1 and c3 != c2: distinct += 1
    if c4 != c0 and c4 != c1 and c4 != c2 and c4 != c3: distinct += 1
    if c5 != c0 and c5 != c1 and c5 != c2 and c5 != c3 and c5 != c4: distinct += 1
    return distinct

def _count_bad_k4s(colors):
    """Count K4s with fewer than 5 distinct colors. colors is a numpy array of edge colors."""
    bad = 0
    for k4_idx in range(_NUM_K4S):
        eidxs = _K4_EDGE_IDXS[k4_idx]
        c0 = colors[eidxs[0]]
        c1 = colors[eidxs[1]]
        c2 = colors[eidxs[2]]
        c3 = colors[eidxs[3]]
        c4 = colors[eidxs[4]]
        c5 = colors[eidxs[5]]
        if _distinct_count(c0, c1, c2, c3, c4, c5) < 5:
            bad += 1
    return bad

def _count_colors_used(colors):
    return len(set(colors.tolist()))

def _compute_delta_bad(c, edge_idx, new_color):
    """Compute change in bad K4 count if we set edge_idx to new_color."""
    affected_k4s = _EDGE_TO_K4S[edge_idx]
    old_color = c[edge_idx]
    
    delta_bad = 0
    for k4_idx in affected_k4s:
        eidxs = _K4_EDGE_IDXS[k4_idx]
        c0 = c[eidxs[0]]
        c1 = c[eidxs[1]]
        c2 = c[eidxs[2]]
        c3 = c[eidxs[3]]
        c4 = c[eidxs[4]]
        c5 = c[eidxs[5]]
        
        distinct_before = _distinct_count(c0, c1, c2, c3, c4, c5)
        bad_before = 1 if distinct_before < 5 else 0
        
        # Find position of edge_idx in this K4
        pos = -1
        for p in range(6):
            if eidxs[p] == edge_idx:
                pos = p
                break
        
        # Simulate new colors
        if pos == 0:
            distinct_after = _distinct_count(new_color, c1, c2, c3, c4, c5)
        elif pos == 1:
            distinct_after = _distinct_count(c0, new_color, c2, c3, c4, c5)
        elif pos == 2:
            distinct_after = _distinct_count(c0, c1, new_color, c3, c4, c5)
        elif pos == 3:
            distinct_after = _distinct_count(c0, c1, c2, new_color, c4, c5)
        elif pos == 4:
            distinct_after = _distinct_count(c0, c1, c2, c3, new_color, c5)
        else:
            distinct_after = _distinct_count(c0, c1, c2, c3, c4, new_color)
        
        bad_after = 1 if distinct_after < 5 else 0
        delta_bad += bad_after - bad_before
    
    return delta_bad

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Start with the known valid 17-coloring
    colors = np.array([(i + j) % 17 for i in range(_N) for j in range(i + 1, _N)], dtype=np.int32)
    
    best_colors = colors.copy()
    best_score = _count_colors_used(best_colors)  # 17
    
    # Strategy: local search to minimize bad K4s, then minimize colors used
    # For each target color count, try to find a valid coloring
    
    rng = random.Random(42)
    
    for target_colors in [15, 14, 13]:
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        for init_iter in range(50):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with target_colors
            c = np.array([rng.randint(0, target_colors - 1) for _ in range(_NUM_EDGES)], dtype=np.int32)
            
            bad = _count_bad_k4s(c)
            
            # Local search with simulated annealing
            temperature = 0.5
            cooling = 0.9995
            max_iters = 100000
            
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                if bad == 0:
                    # Valid coloring!
                    actual_colors = _count_colors_used(c)
                    if actual_colors < best_score:
                        best_score = actual_colors
                        best_colors = c.copy()
                    break
                
                # Pick a random edge and try changing it
                edge_idx = rng.randint(0, _NUM_EDGES - 1)
                old_color = c[edge_idx]
                
                # Try a new random color
                new_color = rng.randint(0, target_colors - 1)
                if new_color == old_color:
                    continue
                
                delta_bad = _compute_delta_bad(c, edge_idx, new_color)
                
                if delta_bad <= 0:
                    c[edge_idx] = new_color
                    bad += delta_bad
                else:
                    # Accept worse with probability based on temperature
                    if rng.random() < np.exp(-delta_bad / temperature):
                        c[edge_idx] = new_color
                        bad += delta_bad
                
                temperature *= cooling
            
            if bad == 0:
                actual_colors = _count_colors_used(c)
                if actual_colors < best_score:
                    best_score = actual_colors
                    best_colors = c.copy()
                # Found valid for this target, move to next
                break
    
    # Final check: ensure best_colors is valid
    if _count_bad_k4s(best_colors) > 0:
        # Fallback to the original 17-coloring
        return [(i + j) % 17 for i in range(_N) for j in range(i + 1, _N)]
    
    # Return the best coloring found
    return best_colors.tolist()
# EVOLVE-BLOCK-END
