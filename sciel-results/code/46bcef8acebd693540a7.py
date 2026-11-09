# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

_N = 16
_NUM_EDGES = _N * (_N - 1) // 2
_K4S = list(combinations(range(_N), 4))
_NUM_K4S = len(_K4S)

# Precompute edge indices for each K4
def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (_N - 1) - i * (i + 1) // 2 + (j - i - 1)

_K4_EDGE_IDXS = []
_EDGE_TO_K4S = [[] for _ in range(_NUM_EDGES)]

for k4_idx, quad in enumerate(_K4S):
    edges = []
    for a in range(4):
        for b in range(a + 1, 4):
            i, j = quad[a], quad[b]
            idx = _edge_idx(i, j)
            edges.append(idx)
            _EDGE_TO_K4S[idx].append(k4_idx)
    _K4_EDGE_IDXS.append(edges)

# Convert to numpy arrays for faster access
_K4_EDGE_IDXS_ARR = np.array(_K4_EDGE_IDXS, dtype=np.int32)

def _count_distinct_6(c0, c1, c2, c3, c4, c5):
    # Fast distinct count for 6 items
    d = 1
    if c1 != c0: d += 1
    if c2 != c0 and c2 != c1: d += 1
    if c3 != c0 and c3 != c1 and c3 != c2: d += 1
    if c4 != c0 and c4 != c1 and c4 != c2 and c4 != c3: d += 1
    if c5 != c0 and c5 != c1 and c5 != c2 and c5 != c3 and c5 != c4: d += 1
    return d

def _evaluate(colors):
    """Count bad K4s and return (bad_count, bad_flags_array)"""
    bad_count = 0
    bad_flags = np.zeros(_NUM_K4S, dtype=bool)
    for k4_idx in range(_NUM_K4S):
        eidxs = _K4_EDGE_IDXS[k4_idx]
        c0 = colors[eidxs[0]]
        c1 = colors[eidxs[1]]
        c2 = colors[eidxs[2]]
        c3 = colors[eidxs[3]]
        c4 = colors[eidxs[4]]
        c5 = colors[eidxs[5]]
        d = _count_distinct_6(c0, c1, c2, c3, c4, c5)
        if d < 5:
            bad_count += 1
            bad_flags[k4_idx] = True
    return bad_count, bad_flags

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    rng = random.Random(42)
    
    best_valid_colors = None
    best_num_colors = 1000
    
    # Try different target color counts
    # We want to minimize colors used.
    # Start with a random initialization using K colors, try to make it valid.
    # If valid, record it. Try smaller K.
    
    # We will iterate K from 13 to 15 (and maybe 16 if stuck)
    # For each K, run local search.
    
    for K in [13, 14, 15, 16]:
        if time.time() > deadline:
            break
            
        # Initialize with random colors in [0, K-1]
        colors = np.array([rng.randint(0, K - 1) for _ in range(_NUM_EDGES)], dtype=np.int32)
        bad_count, bad_flags = _evaluate(colors)
        
        # Local Search
        # We want to reduce bad_count to 0.
        # Move: change color of one edge.
        
        T = 1.0
        T_min = 0.01
        cooling = 0.9999
        
        # If we find a valid solution with K colors, we can break and try K-1?
        # No, we are iterating K in increasing order? No, 13, 14, 15.
        # If we find valid with 13, that's optimal (since lower bound is 13).
        # So we should try 13 first. If found, return.
        # If not found in time for 13, try 14, etc.
        
        # However, if we start with 13 colors and can't fix it, we might get stuck.
        # We can also try to start with more colors and reduce?
        # The task says "Minimize the number of colours used".
        # So finding ANY valid coloring with 13 colors is best.
        
        # Let's run a decent number of iterations for each K.
        
        max_iters = 20000
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            if bad_count == 0:
                # Valid coloring with K colors
                # Check if we can use fewer colors?
                # The colors are in [0, K-1]. The actual number used might be less.
                actual_colors = len(set(colors.tolist()))
                if actual_colors < best_num_colors:
                    best_num_colors = actual_colors
                    best_valid_colors = colors.copy()
                # If K=13, we can't do better (lower bound).
                if K == 13:
                    return best_valid_colors.tolist()
                break # Try next K? Or continue to see if we can reduce colors within this K?
                # Actually, if we have a valid coloring with K=14, we might be able to recolor it to use fewer colors?
                # That's a separate problem. Let's just record it and try K=13 in next outer loop?
                # No, the outer loop tries K=13 first. If it fails, it tries K=14.
                # So if we find valid with K=14, we store it. Then we don't try K=13 again?
                # Wait, the loop is `for K in [13, 14, 15]`.
                # If K=13 fails to find a solution in `max_iters`, it moves to K=14.
                # If K=14 finds a solution, it breaks the inner loop.
                # Then it moves to K=15.
                # We should probably not waste time on K=15 if we already have a K=14 solution,
                # unless we think we can find K=13 later? No, we already tried K=13.
                # So we can break the outer loop if we found a solution?
                # No, we want the MINIMUM colors.
                # If we found a solution with 14 colors, we should check if we can find one with 13.
                # But we already spent time on 13.
                # Maybe we should allocate time to 13, then 14, then 15.
                # If we find a solution in 14, we save it.
                # If we find a solution in 15, we save it only if it uses fewer colors than the 14 one?
                # Unlikely.
                
                # So:
                # 1. Try K=13. If success, return.
                # 2. Try K=14. If success, save.
                # 3. Try K=15. If success, save if better.
                # 4. Return best saved.
                
                break # Break inner loop for this K
                
            # Select edge to change
            if bad_count > 0 and rng.random() < 0.8:
                # Pick a bad K4, then an edge in it
                bad_k4s = np.where(bad_flags)[0]
                k4_idx = bad_k4s[rng.randint(0, len(bad_k4s) - 1)]
                edges_in_k4 = _K4_EDGE_IDXS[k4_idx]
                edge_idx = edges_in_k4[rng.randint(0, 5)]
            else:
                edge_idx = rng.randint(0, _NUM_EDGES - 1)
                
            old_color = colors[edge_idx]
            new_color = rng.randint(0, K - 1)
            
            if old_color == new_color:
                continue
                
            # Calculate delta
            delta = 0
            for k4_idx in _EDGE_TO_K4S[edge_idx]:
                eidxs = _K4_EDGE_IDXS[k4_idx]
                
                # Get current colors
                c0 = colors[eidxs[0]]
                c1 = colors[eidxs[1]]
                c2 = colors[eidxs[2]]
                c3 = colors[eidxs[3]]
                c4 = colors[eidxs[4]]
                c5 = colors[eidxs[5]]
                
                was_bad = bad_flags[k4_idx]
                
                # Determine which position in the tuple is edge_idx
                # We can just construct the new tuple
                # This is a bit slow but manageable.
                # Optimization: precompute position?
                # Let's just create the new list
                new_colors_list = [c0, c1, c2, c3, c4, c5]
                # Find position
                # eidxs is a list of edge indices.
                # We need to replace the one corresponding to edge_idx.
                # Since eidxs are unique in a K4, we can check equality.
                pos = -1
                for p, e in enumerate(eidxs):
                    if e == edge_idx:
                        pos = p
                        break
                
                if pos != -1:
                    new_colors_list[pos] = new_color
                    d_new = _count_distinct_6(*new_colors_list)
                    is_bad = d_new < 5
                    
                    if was_bad and not is_bad:
                        delta -= 1
                    elif not was_bad and is_bad:
                        delta += 1
            
            # Acceptance
            if delta <= 0:
                accept = True
            else:
                accept = rng.random() < np.exp(-delta / T)
                
            if accept:
                colors[edge_idx] = new_color
                bad_count += delta
                # Update bad_flags
                for k4_idx in _EDGE_TO_K4S[edge_idx]:
                    eidxs = _K4_EDGE_IDXS[k4_idx]
                    c0 = colors[eidxs[0]]
                    c1 = colors[eidxs[1]]
                    c2 = colors[eidxs[2]]
                    c3 = colors[eidxs[3]]
                    c4 = colors[eidxs[4]]
                    c5 = colors[eidxs[5]]
                    d = _count_distinct_6(c0, c1, c2, c3, c4, c5)
                    bad_flags[k4_idx] = (d < 5)
                    
            T *= cooling
            if T < T_min:
                T = T_min
                
        # End of local search for this K
        # If we found a valid solution, it's stored in best_valid_colors
        # If K was 13 and we found one, we returned.
        # If K was 14 or 15, we continue to next K?
        # Actually, if we found a valid solution with K=14, we should probably stop trying K=15
        # because 15 colors is worse.
        # But what if K=14 solution used 15 colors? (Impossible, colors are in 0..13).
        # So if we have a valid solution with K=14, it uses <= 14 colors.
        # A solution with K=15 uses <= 15 colors.
        # So the K=14 solution is always at least as good.
        # So we can break the outer loop if we found a valid solution.
        
        if best_valid_colors is not None and best_num_colors <= K:
             # We found a solution with K colors.
             # Since we iterate K=13, 14, 15, and we break if found,
             # this means we found the smallest K that worked in this run.
             # Wait, if K=13 failed, we go to K=14. If K=14 succeeds, we break.
             # So best_num_colors will be <= 14.
             # We can return.
             pass # We will return at the end or break here?
             # Let's break the outer loop.
             break

    if best_valid_colors is not None:
        return best_valid_colors.tolist()
        
    # Fallback: return any valid coloring, even if many colors
    # Use a simple modular coloring which is likely valid with enough colors
    # K=17 is known to be safe for small n?
    # Let's try K=17
    colors = np.array([(i + j) % 17 for i in range(_N) for j in range(i + 1, _N)], dtype=np.int32)
    bad_count, _ = _evaluate(colors)
    if bad_count == 0:
        return colors.tolist()
        
    # If even that fails (unlikely), return random
    return np.array([rng.randint(0, 19) for _ in range(_NUM_EDGES)], dtype=np.int32).tolist()
# EVOLVE-BLOCK-END
