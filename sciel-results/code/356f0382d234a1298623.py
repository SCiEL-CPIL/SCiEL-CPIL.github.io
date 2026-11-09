"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

_N = 16
_NUM_EDGES = _N * (_N - 1) // 2  # 120

# Precompute all K4s and their edge indices
def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (_N - 1) - i * (i + 1) // 2 + (j - i - 1)

_K4S = list(combinations(range(_N), 4))
_K4_EDGE_IDXS = []
for quad in _K4S:
    edges = []
    for a in range(4):
        for b in range(a + 1, 4):
            i, j = quad[a], quad[b]
            edges.append(_edge_idx(i, j))
    _K4_EDGE_IDXS.append(edges)

_NUM_K4S = len(_K4S)  # 1820

# For each edge, precompute which K4s contain it and at which position
_EDGE_TO_K4_POS = [[] for _ in range(_NUM_EDGES)]
for k4_idx in range(_NUM_K4S):
    eidxs = _K4_EDGE_IDXS[k4_idx]
    for pos, eidx in enumerate(eidxs):
        _EDGE_TO_K4_POS[eidx].append((k4_idx, pos))

# Precompute popcount for 16-bit values
_PC16 = [bin(i).count('1') for i in range(65536)]

def _popcount64(x):
    """Fast popcount for 64-bit integer."""
    count = 0
    while x:
        count += _PC16[x & 0xFFFF]
        x >>= 16
    return count

def _distinct_count_fast(c0, c1, c2, c3, c4, c5):
    """Count distinct colors among 6 values using bitmask."""
    lo = 0
    hi = 0
    for c in (c0, c1, c2, c3, c4, c5):
        if c < 64:
            lo |= (1 << c)
        else:
            hi |= (1 << (c - 64))
    return _popcount64(lo) + _popcount64(hi)

def _count_bad_k4s(colors):
    """Count K4s with fewer than 5 distinct colors."""
    bad = 0
    for k4_idx in range(_NUM_K4S):
        eidxs = _K4_EDGE_IDXS[k4_idx]
        c0 = colors[eidxs[0]]
        c1 = colors[eidxs[1]]
        c2 = colors[eidxs[2]]
        c3 = colors[eidxs[3]]
        c4 = colors[eidxs[4]]
        c5 = colors[eidxs[5]]
        if _distinct_count_fast(c0, c1, c2, c3, c4, c5) < 5:
            bad += 1
    return bad

def _count_colors_used(colors):
    return len(set(colors.tolist()))

def _compute_delta_bad(c, edge_idx, new_color):
    """Compute change in bad K4 count if we set edge_idx to new_color."""
    old_color = c[edge_idx]
    if old_color == new_color:
        return 0
    
    delta_bad = 0
    for k4_idx, pos in _EDGE_TO_K4_POS[edge_idx]:
        eidxs = _K4_EDGE_IDXS[k4_idx]
        
        c0 = c[eidxs[0]]
        c1 = c[eidxs[1]]
        c2 = c[eidxs[2]]
        c3 = c[eidxs[3]]
        c4 = c[eidxs[4]]
        c5 = c[eidxs[5]]
        
        distinct_before = _distinct_count_fast(c0, c1, c2, c3, c4, c5)
        bad_before = 1 if distinct_before < 5 else 0
        
        if pos == 0:
            distinct_after = _distinct_count_fast(new_color, c1, c2, c3, c4, c5)
        elif pos == 1:
            distinct_after = _distinct_count_fast(c0, new_color, c2, c3, c4, c5)
        elif pos == 2:
            distinct_after = _distinct_count_fast(c0, c1, new_color, c3, c4, c5)
        elif pos == 3:
            distinct_after = _distinct_count_fast(c0, c1, c2, new_color, c4, c5)
        elif pos == 4:
            distinct_after = _distinct_count_fast(c0, c1, c2, c3, new_color, c5)
        else:
            distinct_after = _distinct_count_fast(c0, c1, c2, c3, c4, new_color)
        
        bad_after = 1 if distinct_after < 5 else 0
        delta_bad += bad_after - bad_before
    
    return delta_bad

def _greedy_init(c, target_colors, rng):
    """Greedy initialization: assign colors one by one to minimize bad K4s."""
    edge_order = list(range(_NUM_EDGES))
    rng.shuffle(edge_order)
    for ei in edge_order:
        best_c = 0
        best_delta = float('inf')
        for trial_color in range(target_colors):
            c[ei] = trial_color
            delta = 0
            for k4_idx, pos in _EDGE_TO_K4_POS[ei]:
                eidxs = _K4_EDGE_IDXS[k4_idx]
                cols = [c[eidxs[p]] for p in range(6)]
                if _distinct_count_fast(*cols) < 5:
                    delta += 1
            if delta < best_delta:
                best_delta = delta
                best_c = trial_color
        c[ei] = best_c
    return c

def _local_search(c, target_colors, rng, deadline, max_iters=50000):
    """Simulated annealing local search to minimize bad K4s."""
    temperature = 0.5
    cooling = 0.9995
    no_improve_count = 0
    current_bad = _count_bad_k4s(c)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        if current_bad == 0:
            break
        
        # Pick a random edge
        edge_idx = rng.randint(0, _NUM_EDGES - 1)
        old_color = c[edge_idx]
        
        # Try a new random color
        new_color = rng.randint(0, target_colors - 1)
        if new_color == old_color:
            continue
        
        delta_bad = _compute_delta_bad(c, edge_idx, new_color)
        
        if delta_bad < 0:
            c[edge_idx] = new_color
            current_bad += delta_bad
            no_improve_count = 0
        elif delta_bad == 0:
            if rng.random() < 0.5:
                c[edge_idx] = new_color
        else:
            if rng.random() < np.exp(-delta_bad / temperature):
                c[edge_idx] = new_color
                current_bad += delta_bad
            else:
                no_improve_count += 1
                if no_improve_count > 1000:
                    temperature *= 0.9
                    no_improve_count = 0
        
        temperature *= cooling
    
    return c, current_bad

def _polish_reduce_colors(c, current_max, deadline):
    """Polish: try to reduce the maximum color used by reassigning edges."""
    c = c.copy()
    
    while time.time() < deadline:
        colors_in_use = set(c.tolist())
        if not colors_in_use:
            break
        target_max = max(colors_in_use)
        
        # Try to remove target_max by reassigning edges with that color
        found_improvement = False
        for idx in range(_NUM_EDGES):
            if time.time() > deadline:
                break
            if c[idx] != target_max:
                continue
            
            old_color = c[idx]
            # Try to assign a different color (preferably lower)
            for new_color in range(0, target_max):
                if new_color == old_color:
                    continue
                
                # Apply change
                c[idx] = new_color
                
                # Check if still valid
                if _count_bad_k4s(c) == 0:
                    # Valid! We reduced the max color
                    found_improvement = True
                    break
                else:
                    c[idx] = old_color
            
            if found_improvement:
                break
        
        if not found_improvement:
            break
    
    return c

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Start with the known valid 17-coloring
    colors_17 = np.array([(i + j) % 17 for i in range(_N) for j in range(i + 1, _N)], dtype=np.int32)
    best_colors = colors_17.copy()
    best_score = _count_colors_used(best_colors)  # 17
    
    rng = random.Random(42)
    
    # Try to reduce colors from 16 down to 13
    for target_colors in [16, 15, 14, 13]:
        if time.time() > deadline - 20:
            break
        
        # Try multiple initializations
        for init_iter in range(20):
            if time.time() > deadline - 10:
                break
            
            # Initialization strategy:
            if init_iter == 0:
                c = colors_17 % target_colors
            elif init_iter == 1:
                # Greedy initialization: assign colors one by one
                c = np.zeros(_NUM_EDGES, dtype=np.int32)
                c = _greedy_init(c, target_colors, rng)
            else:
                c = np.array([rng.randint(0, target_colors - 1) for _ in range(_NUM_EDGES)], dtype=np.int32)
            
            # Local search with simulated annealing
            c, current_bad = _local_search(c, target_colors, rng, deadline - 10)
            
            if current_bad == 0:
                # Polish to reduce colors further
                c = _polish_reduce_colors(c, target_colors - 1, deadline - 5)
                
                actual_colors = _count_colors_used(c)
                if actual_colors < best_score:
                    best_score = actual_colors
                    best_colors = c.copy()
                break
    
    # Final check: ensure best_colors is valid
    if _count_bad_k4s(best_colors) > 0:
        return [(i + j) % 17 for i in range(_N) for j in range(i + 1, _N)]
    
    return best_colors.tolist()
# EVOLVE-BLOCK-END
