"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2
NUM_K4 = 1820

# Pre-compute edge indices
EDGE_IDX = {}
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        EDGE_IDX[(i, j)] = idx
        idx += 1

# Pre-compute K4s and their edge indices
K4_EDGES = []
for quad in combinations(range(N), 4):
    a, b, c, d = quad
    edges = [
        EDGE_IDX[(a, b)], EDGE_IDX[(a, c)], EDGE_IDX[(a, d)],
        EDGE_IDX[(b, c)], EDGE_IDX[(b, d)], EDGE_IDX[(c, d)]
    ]
    K4_EDGES.append(tuple(edges))

# Pre-compute for each edge, which K4s it belongs to
EDGE_TO_K4 = [[] for _ in range(NUM_EDGES)]
for k_idx, edges in enumerate(K4_EDGES):
    for e in edges:
        EDGE_TO_K4[e].append(k_idx)

# Precompute K4 edge arrays as numpy for fast access
K4_EDGES_ARR = np.array(K4_EDGES, dtype=np.int32)  # shape (1820, 6)

def _count_distinct_6(c0, c1, c2, c3, c4, c5):
    return len({c0, c1, c2, c3, c4, c5})

def _compute_violations(colors):
    """Compute violation status and count using numpy."""
    colors = np.asarray(colors, dtype=np.int32)
    k4_colors = colors[K4_EDGES_ARR]
    k4_sorted = np.sort(k4_colors, axis=1)
    diffs = np.diff(k4_sorted, axis=1)
    distinct = 1 + np.sum(diffs != 0, axis=1)
    violations = distinct < 5
    return violations, int(np.sum(violations))

def _try_structured_k(k, deadline):
    """Try structured colorings based on vertex labels mod k."""
    best_cand = None
    best_viol = float('inf')
    
    # Try (i+j) % k
    cand = np.array([(i + j) % k for i in range(N) for j in range(i + 1, N)], dtype=np.int32)
    viol, tv = _compute_violations(cand)
    if tv < best_viol:
        best_viol = tv
        best_cand = cand.copy()
    
    # Try (i-j) % k
    cand = np.array([(i - j) % k for i in range(N) for j in range(i + 1, N)], dtype=np.int32)
    viol, tv = _compute_violations(cand)
    if tv < best_viol:
        best_viol = tv
        best_cand = cand.copy()
    
    # Try (i*2+j) % k
    cand = np.array([(i * 2 + j) % k for i in range(N) for j in range(i + 1, N)], dtype=np.int32)
    viol, tv = _compute_violations(cand)
    if tv < best_viol:
        best_viol = tv
        best_cand = cand.copy()
    
    # Try (i+j+1) % k
    cand = np.array([(i + j + 1) % k for i in range(N) for j in range(i + 1, N)], dtype=np.int32)
    viol, tv = _compute_violations(cand)
    if tv < best_viol:
        best_viol = tv
        best_cand = cand.copy()
    
    # Try (i*j) % k
    cand = np.array([(i * j) % k for i in range(N) for j in range(i + 1, N)], dtype=np.int32)
    viol, tv = _compute_violations(cand)
    if tv < best_viol:
        best_viol = tv
        best_cand = cand.copy()
    
    return best_cand, best_viol

def _evaluate_move(current, e_idx, new_c, violations_arr):
    """Evaluate changing edge e_idx to new_c. Returns (delta, affected_k4s)."""
    delta = 0
    affected = EDGE_TO_K4[e_idx]
    for k_idx in affected:
        edges = K4_EDGES[k_idx]
        c0 = int(current[edges[0]]) if edges[0] != e_idx else new_c
        c1 = int(current[edges[1]]) if edges[1] != e_idx else new_c
        c2 = int(current[edges[2]]) if edges[2] != e_idx else new_c
        c3 = int(current[edges[3]]) if edges[3] != e_idx else new_c
        c4 = int(current[edges[4]]) if edges[4] != e_idx else new_c
        c5 = int(current[edges[5]]) if edges[5] != e_idx else new_c
        
        new_distinct = _count_distinct_6(c0, c1, c2, c3, c4, c5)
        new_status = new_distinct < 5
        old_status = violations_arr[k_idx]
        
        if old_status and not new_status:
            delta -= 1
        elif not old_status and new_status:
            delta += 1
    return delta, affected

def _apply_move(current, e_idx, new_c, color_freq, violations_arr):
    """Apply a move and update state."""
    current[e_idx] = new_c
    color_freq[new_c] += 1
    affected = EDGE_TO_K4[e_idx]
    for k_idx in affected:
        edges = K4_EDGES[k_idx]
        violations_arr[k_idx] = _count_distinct_6(
            int(current[edges[0]]), int(current[edges[1]]),
            int(current[edges[2]]), int(current[edges[3]]),
            int(current[edges[4]]), int(current[edges[5]])
        ) < 5

def _revert_move(current, e_idx, old_c, color_freq, violations_arr):
    """Revert a move."""
    current[e_idx] = old_c
    color_freq[old_c] += 1
    affected = EDGE_TO_K4[e_idx]
    for k_idx in affected:
        edges = K4_EDGES[k_idx]
        violations_arr[k_idx] = _count_distinct_6(
            int(current[edges[0]]), int(current[edges[1]]),
            int(current[edges[2]]), int(current[edges[3]]),
            int(current[edges[4]]), int(current[edges[5]])
        ) < 5

def _reduce_colors(current, violations_arr, deadline):
    """Aggressively reduce number of colors while maintaining validity."""
    MAX_COLOR = 30
    color_freq = np.zeros(MAX_COLOR, dtype=np.int32)
    for c in current:
        if c < MAX_COLOR:
            color_freq[c] += 1
    current_colors = int(np.sum(color_freq > 0))
    
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        
        # Find the rarest color
        nonzero_freq = color_freq[color_freq > 0]
        if len(nonzero_freq) == 0:
            break
        min_freq = int(np.min(nonzero_freq))
        rare_colors = np.where(color_freq == min_freq)[0]
        
        for old_c in rare_colors:
            if time.time() > deadline - 3:
                break
            if color_freq[old_c] == 0:
                continue
            
            edges_with_old = np.where(current == old_c)[0]
            if len(edges_with_old) == 0:
                continue
            
            # Try to reassign each edge with old_c to another color
            for e_idx in edges_with_old:
                if color_freq[old_c] == 0:
                    break
                
                e_idx = int(e_idx)
                # Try all other used colors
                used_colors = np.where((color_freq > 0) & (np.arange(MAX_COLOR) != old_c))[0]
                for new_c in used_colors:
                    new_c = int(new_c)
                    delta, affected = _evaluate_move(current, e_idx, new_c, violations_arr)
                    
                    # Check if this move reduces color count without creating violations
                    new_color_count = current_colors
                    if color_freq[old_c] == 1:
                        new_color_count -= 1
                    if color_freq[new_c] == 0:
                        new_color_count += 1
                    
                    if delta == 0 and new_color_count < current_colors:
                        # Apply move
                        _apply_move(current, e_idx, new_c, color_freq, violations_arr)
                        current_colors = new_color_count
                        improved = True
                        break
                    elif delta < 0:
                        # This move fixes violations, might be worth trying
                        _apply_move(current, e_idx, new_c, color_freq, violations_arr)
                        improved = True
                        break
    
    return current, current_colors

def _construct():
    deadline = time.time() + 140
    random.seed(42)
    np.random.seed(42)
    
    # Try multiple initial constructions
    best = None
    best_violations = float('inf')
    best_violations_arr = None
    
    # Try structured constructions for various k
    for k in range(13, 20):
        if time.time() > deadline - 15:
            break
        cand, tv = _try_structured_k(k, deadline)
        if cand is not None and tv < best_violations:
            best_violations = tv
            best = cand
            best_violations_arr, _ = _compute_violations(cand)
        if best_violations == 0:
            break
    
    # Also try random starts with fewer colors - focus on 13-15
    for trial in range(500):
        if time.time() > deadline - 10:
            break
        max_c = random.choice([13, 13, 13, 14, 14, 15])
        cand = np.array([random.randint(0, max_c - 1) for _ in range(NUM_EDGES)], dtype=np.int32)
        viol, tv = _compute_violations(cand)
        if tv < best_violations:
            best_violations = tv
            best = cand
            best_violations_arr = viol.copy()
        if tv == 0:
            break
    
    if best is None:
        return [(i + j) % 17 for i in range(N) for j in range(i + 1, N)]
    
    current = best.copy()
    current_violations = best_violations
    current_violations_arr = best_violations_arr.copy()
    
    # Track color frequencies
    MAX_COLOR = 20
    color_freq = np.zeros(MAX_COLOR, dtype=np.int32)
    for c in current:
        if c < MAX_COLOR:
            color_freq[c] += 1
    current_colors = int(np.sum(color_freq > 0))
    
    # Phase 1: Reduce violations using simulated annealing
    temp = 2.0
    cooling = 0.9995
    min_temp = 0.1
    
    while time.time() < deadline - 15 and current_violations > 0:
        # Pick a violated K4
        violated_k4s = np.where(current_violations_arr)[0]
        if len(violated_k4s) == 0:
            current_violations = 0
            break
            
        k_idx = int(random.choice(violated_k4s))
        edges = K4_EDGES[k_idx]
        e_idx = int(random.choice(edges))
        old_c = int(current[e_idx])
        
        # Determine max color to try based on current usage
        max_color_try = max(15, current_colors + 2)
        
        # Try to find a color that fixes this violation
        improved = False
        for attempt in range(30):
            new_c = random.randint(0, max_color_try - 1)
            if new_c == old_c:
                continue
            
            delta, affected = _evaluate_move(current, e_idx, new_c, current_violations_arr)
            
            if delta < 0:
                # This move reduces violations
                _apply_move(current, e_idx, new_c, color_freq, current_violations_arr)
                current_violations += delta
                improved = True
                break
            elif delta == 0:
                # Neutral move, accept with some probability
                if random.random() < 0.4:
                    _apply_move(current, e_idx, new_c, color_freq, current_violations_arr)
                    improved = True
                    break
            else:
                # Accept with probability exp(-delta / temp)
                if delta > 0:
                    prob = np.exp(-delta / max(temp, 0.01))
                    if random.random() < prob:
                        _apply_move(current, e_idx, new_c, color_freq, current_violations_arr)
                        current_violations += delta
                        improved = True
                        break
        
        if not improved:
            # Try a random edge change
            e_idx = random.randint(0, NUM_EDGES - 1)
            old_c = int(current[e_idx])
            new_c = random.randint(0, max_color_try - 1)
            if new_c != old_c:
                delta, affected = _evaluate_move(current, e_idx, new_c, current_violations_arr)
                if delta <= 0 or random.random() < np.exp(-max(delta, 0) / max(temp, 0.01)):
                    _apply_move(current, e_idx, new_c, color_freq, current_violations_arr)
                    current_violations += delta
        
        temp *= cooling
        if temp < min_temp:
            temp = min_temp
    
    # Phase 2: Reduce colors
    if current_violations == 0:
        current, current_colors = _reduce_colors(current, current_violations_arr, deadline)
    
    # Phase 3: Final polish - try to reduce colors further
    if time.time() < deadline - 3 and current_violations == 0:
        MAX_COLOR = 20
        color_freq = np.zeros(MAX_COLOR, dtype=np.int32)
        for c in current:
            if c < MAX_COLOR:
                color_freq[c] += 1
        
        # Try systematic color reduction
        for _ in range(5000):
            if time.time() > deadline - 3:
                break
            
            # Find rarest color
            nonzero_freq = color_freq[color_freq > 0]
            if len(nonzero_freq) == 0:
                break
            min_freq = int(np.min(nonzero_freq))
            if min_freq > 3:
                break
                
            rare_colors = np.where(color_freq == min_freq)[0]
            old_c = int(random.choice(rare_colors))
            
            if color_freq[old_c] == 0:
                continue
            
            edges_with_old = np.where(current == old_c)[0]
            if len(edges_with_old) == 0:
                continue
            
            e_idx = int(random.choice(edges_with_old))
            
            # Try reassigning to a common color
            used_colors = np.where(color_freq > 0)[0]
            used_colors = used_colors[used_colors != old_c]
            if len(used_colors) == 0:
                continue
            
            # Prefer colors with higher frequency
            freqs = color_freq[used_colors]
            if len(freqs) > 0:
                probs = freqs / freqs.sum()
                new_c = int(np.random.choice(used_colors, p=probs))
            else:
                new_c = int(random.choice(used_colors))
            
            delta, affected = _evaluate_move(current, e_idx, new_c, current_violations_arr)
            
            # Only accept if it doesn't create violations and reduces color count
            new_color_count = int(np.sum(color_freq > 0))
            if color_freq[old_c] == 1:
                new_color_count -= 1
            if color_freq[new_c] == 0:
                new_color_count += 1
            
            if delta == 0 and new_color_count < int(np.sum(color_freq > 0)):
                _apply_move(current, e_idx, new_c, color_freq, current_violations_arr)
    
    # Final: count distinct colors used
    return [int(c) for c in current]

def construct():
    return _construct()
# EVOLVE-BLOCK-END
