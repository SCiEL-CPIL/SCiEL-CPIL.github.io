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

# Precompute for each K4 and each position in that K4, the other 5 edge indices
# This allows O(1) distinct count updates
_K4_OTHER_EDGES = []  # _K4_OTHER_EDGES[k4_idx][pos] = tuple of the other 5 edge indices
for k4_idx in range(_NUM_K4S):
    eidxs = _K4_EDGE_IDXS[k4_idx]
    others = []
    for pos in range(6):
        others.append(tuple(eidxs[p] for p in range(6) if p != pos))
    _K4_OTHER_EDGES.append(others)

# Precompute for each edge_idx, the list of (k4_idx, pos) pairs
_EDGE_TO_K4_POS = [[] for _ in range(_NUM_EDGES)]
for k4_idx in range(_NUM_K4S):
    eidxs = _K4_EDGE_IDXS[k4_idx]
    for pos in range(6):
        _EDGE_TO_K4_POS[eidxs[pos]].append((k4_idx, pos))

def _distinct_count6(c0, c1, c2, c3, c4, c5):
    """Count distinct values among 6 colors - optimized."""
    d = 1
    if c1 != c0: d += 1
    if c2 != c0 and c2 != c1: d += 1
    if c3 != c0 and c3 != c1 and c3 != c2: d += 1
    if c4 != c0 and c4 != c1 and c4 != c2 and c4 != c3: d += 1
    if c5 != c0 and c5 != c1 and c5 != c2 and c5 != c3 and c5 != c4: d += 1
    return d

def _count_bad_k4s(colors):
    """Count K4s with fewer than 5 distinct colors."""
    bad = 0
    for k4_idx in range(_NUM_K4S):
        eidxs = _K4_EDGE_IDXS[k4_idx]
        if _distinct_count6(colors[eidxs[0]], colors[eidxs[1]], colors[eidxs[2]],
                           colors[eidxs[3]], colors[eidxs[4]], colors[eidxs[5]]) < 5:
            bad += 1
    return bad

def _count_colors_used(colors):
    return len(set(colors.tolist()))

def _compute_delta_bad(c, edge_idx, new_color):
    """Compute change in bad K4 count if we set edge_idx to new_color - O(1) per affected K4."""
    delta_bad = 0
    for k4_idx, pos in _EDGE_TO_K4_POS[edge_idx]:
        eidxs = _K4_EDGE_IDXS[k4_idx]
        others = _K4_OTHER_EDGES[k4_idx][pos]
        old_color = c[eidxs[pos]]
        
        # Before
        d_before = _distinct_count6(c[eidxs[0]], c[eidxs[1]], c[eidxs[2]],
                                    c[eidxs[3]], c[eidxs[4]], c[eidxs[5]])
        bad_before = 1 if d_before < 5 else 0
        
        # After: replace position pos with new_color
        if pos == 0:
            d_after = _distinct_count6(new_color, c[others[0]], c[others[1]], c[others[2]], c[others[3]], c[others[4]])
        elif pos == 1:
            d_after = _distinct_count6(c[others[0]], new_color, c[others[1]], c[others[2]], c[others[3]], c[others[4]])
        elif pos == 2:
            d_after = _distinct_count6(c[others[0]], c[others[1]], new_color, c[others[2]], c[others[3]], c[others[4]])
        elif pos == 3:
            d_after = _distinct_count6(c[others[0]], c[others[1]], c[others[2]], new_color, c[others[3]], c[others[4]])
        elif pos == 4:
            d_after = _distinct_count6(c[others[0]], c[others[1]], c[others[2]], c[others[3]], new_color, c[others[4]])
        else:
            d_after = _distinct_count6(c[others[0]], c[others[1]], c[others[2]], c[others[3]], c[others[4]], new_color)
        
        bad_after = 1 if d_after < 5 else 0
        delta_bad += bad_after - bad_before
    
    return delta_bad

def _anneal(c, bad, rng, deadline, target_colors, t_start, cooling_rate):
    """Run simulated annealing, return (c, bad, converged)."""
    temperature = t_start
    iters_since_improve = 0
    best_bad_local = bad
    
    while time.time() < deadline:
        if bad == 0:
            return c, bad, True
        
        # Pick a random edge
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
            if bad < best_bad_local:
                best_bad_local = bad
                iters_since_improve = 0
            else:
                iters_since_improve += 1
        else:
            # Accept worse with probability based on temperature
            if rng.random() < np.exp(-delta_bad / temperature):
                c[edge_idx] = new_color
                bad += delta_bad
                iters_since_improve += 1
            else:
                iters_since_improve += 1
        
        temperature *= cooling_rate
        if temperature < 0.01:
            break
    
    return c, bad, (bad == 0)

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Start with the known valid 17-coloring
    colors = np.array([(i + j) % 17 for i in range(_N) for j in range(i + 1, _N)], dtype=np.int32)
    
    best_colors = colors.copy()
    best_score = _count_colors_used(best_colors)  # 17
    
    rng = random.Random(42)
    
    for target_colors in [13, 14, 15]:
        if time.time() > deadline - 15:
            break
        
        # Try multiple random initializations with different strategies
        for init_iter in range(200):
            if time.time() > deadline - 5:
                break
            
            # Different initialization strategies
            if init_iter % 3 == 0:
                # Pure random
                c = np.array([rng.randint(0, target_colors - 1) for _ in range(_NUM_EDGES)], dtype=np.int32)
            elif init_iter % 3 == 1:
                # Start from 17-coloring and remap
                c = np.array([(i + j) % 17 for i in range(_N) for j in range(i + 1, _N)], dtype=np.int32)
                # Remap colors to fit in target_colors
                color_map = {old: rng.randint(0, target_colors - 1) for old in range(17)}
                c = np.array([color_map[old] for old in c], dtype=np.int32)
            else:
                # Start from 17-coloring with random perturbations
                c = np.array([(i + j) % 17 for i in range(_N) for j in range(i + 1, _N)], dtype=np.int32)
                for _ in range(20):
                    eidx = rng.randint(0, _NUM_EDGES - 1)
                    c[eidx] = rng.randint(0, target_colors - 1)
            
            bad = _count_bad_k4s(c)
            
            # Run simulated annealing with varying parameters
            t_start = 0.3 + rng.random() * 0.7
            cooling_rate = 0.999 + rng.random() * 0.001
            
            c, bad, converged = _anneal(c, bad, rng, deadline - 5, target_colors, t_start, cooling_rate)
            
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
