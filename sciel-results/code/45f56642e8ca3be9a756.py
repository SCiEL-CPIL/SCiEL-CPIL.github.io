import time
import numpy as np
import random

# Precompute all nonconstant 4-term progressions as a numpy array
_ALL_APS = []
for a in range(1, 125):  # a+3d <= 128 => a <= 128-3
    max_d = (128 - a) // 3
    for d in range(1, max_d + 1):
        _ALL_APS.append((a-1, a+d-1, a+2*d-1, a+3*d-1))

_ALL_APS_ARR = np.array(_ALL_APS, dtype=np.int32)
_AP_COUNT = len(_ALL_APS_ARR)

def _count_violations(colors_np):
    """
    Vectorized count of 4-term progressions with fewer than 3 distinct colors.
    colors_np: shape (128,) int array
    Returns: number of violated progressions (int)
    """
    # Get colors for each position in each AP: shape (_AP_COUNT, 4)
    c = colors_np[_ALL_APS_ARR]  # shape (AP_COUNT, 4)
    
    # For each AP, check if it has < 3 distinct colors
    # A 4-element set has < 3 distinct colors iff:
    # - all 4 same (1 color), or
    # - exactly 2 colors (e.g., 3+1 or 2+2)
    
    # Method: for each AP, compute number of distinct colors
    # Use pairwise comparison: count how many elements differ from the first
    # Actually, let's use a different approach:
    # An AP is violated iff it uses at most 2 colors.
    # It uses at most 2 colors iff there exist 2 values v1, v2 such that all 4 are in {v1, v2}.
    # Equivalently: the number of distinct values <= 2.
    
    # Efficient check: 
    # distinct_count >= 3 iff NOT (all same OR exactly two values)
    # all same: c[:,0]==c[:,1] and c[:,1]==c[:,2] and c[:,2]==c[:,3]
    # exactly two: (c[:,0]==c[:,1] and c[:,2]!=c[:,0] and c[:,2]==c[:,3]) or
    #              (c[:,0]!=c[:,1] and c[:,1]==c[:,2] and c[:,2]==c[:,3]) or
    #              (c[:,0]==c[:,2] and c[:,1]!=c[:,0] and c[:,3]!=c[:,0]) or
    #              ... this gets complicated.
    
    # Simpler: use the fact that for 4 elements, distinct < 3 means:
    # Let a,b,c,d be the 4 colors.
    # distinct < 3 iff (a==b and c==d) or (a==c and b==d) or (a==d and b==c) or (a==b and a==c) or ...
    # Actually easiest: count pairs that are equal. If >= 3 pairs equal in a specific way...
    
    # Cleanest vectorized approach:
    # For each AP, compute the number of distinct colors by checking all pairs.
    # n_distinct = 4 - (number of "redundant" elements)
    # Alternative: an AP has < 3 colors iff it can be covered by 2 color classes.
    
    # Let me use: an AP is valid iff it has at least 3 distinct colors.
    # Invalid iff max frequency >= 3 (so only 1 or 2 colors total) OR the multiset has exactly 2 distinct values.
    # Actually: < 3 distinct means 1 or 2 distinct values.
    
    # 1 distinct: all 4 equal
    all_equal = (c[:,0] == c[:,1]) & (c[:,1] == c[:,2]) & (c[:,2] == c[:,3])
    
    # 2 distinct: not all equal, but only 2 values present.
    # This happens when the multiset has a pattern like (x,x,x,y) or (x,x,y,y) or (x,y,x,y) etc.
    # Equivalently: NOT all_equal AND (there exist two values covering all 4)
    # A simpler test: the number of distinct values in {c0,c1,c2,c3} is 2.
    # This is equivalent to: (c0==c1 and c2==c3 and c0!=c2) OR
    #                        (c0==c2 and c1==c3 and c0!=c1) OR
    #                        (c0==c3 and c1==c2 and c0!=c1) OR
    #                        (c0==c1 and c0==c2 and c3!=c0) OR
    #                        (c0==c1 and c0==c3 and c2!=c0) OR
    #                        (c0==c2 and c0==c3 and c1!=c0) OR
    #                        (c0==c3 and c1==c2 and c0!=c1) -- already covered
    #                        (c1==c2 and c1==c3 and c0!=c1) OR
    #                        ... this is getting messy.
    
    # Better: use the identity that for 4 elements, the number of distinct values is 2
    # iff the sum of pairwise equality indicators has a specific structure.
    # Actually, let's just directly compute:
    # distinct_2 = (~all_equal) & (
    #     ((c[:,0]==c[:,1]) & (c[:,0]==c[:,2]) & (c[:,3]!=c[:,0])) |
    #     ((c[:,0]==c[:,1]) & (c[:,0]==c[:,3]) & (c[:,2]!=c[:,0])) |
    #     ((c[:,0]==c[:,2]) & (c[:,0]==c[:,3]) & (c[:,1]!=c[:,0])) |
    #     ((c[:,0]==c[:,3]) & (c[:,1]==c[:,2]) & (c[:,0]!=c[:,1])) |
    #     ((c[:,1]==c[:,2]) & (c[:,1]==c[:,3]) & (c[:,0]!=c[:,1])) |
    #     ((c[:,0]==c[:,1]) & (c[:,2]==c[:,3]) & (c[:,0]!=c[:,2]))
    # )
    
    # Let me simplify by checking all 6 pairs and using the fact that
    # 2 distinct values means exactly one of these patterns holds.
    
    # Pattern A: three of one, one of another
    pat_a1 = (c[:,0]==c[:,1]) & (c[:,0]==c[:,2]) & (c[:,3]!=c[:,0])
    pat_a2 = (c[:,0]==c[:,1]) & (c[:,0]==c[:,3]) & (c[:,2]!=c[:,0])
    pat_a3 = (c[:,0]==c[:,2]) & (c[:,0]==c[:,3]) & (c[:,1]!=c[:,0])
    pat_a4 = (c[:,1]==c[:,2]) & (c[:,1]==c[:,3]) & (c[:,0]!=c[:,1])
    
    # Pattern B: two of each
    pat_b1 = (c[:,0]==c[:,1]) & (c[:,2]==c[:,3]) & (c[:,0]!=c[:,2])
    pat_b2 = (c[:,0]==c[:,2]) & (c[:,1]==c[:,3]) & (c[:,0]!=c[:,1])
    pat_b3 = (c[:,0]==c[:,3]) & (c[:,1]==c[:,2]) & (c[:,0]!=c[:,1])
    
    two_distinct = pat_a1 | pat_a2 | pat_a3 | pat_a4 | pat_b1 | pat_b2 | pat_b3
    
    violated = all_equal | two_distinct
    return int(np.sum(violated))

def _count_distinct_colors(colors_np):
    """Count the number of distinct color labels used."""
    return len(np.unique(colors_np))

def construct():
    start = time.time()
    DEADLINE = 85.0  # safety margin before 90s
    
    n = 128
    
    # Start with a random initial coloring using k colors
    # We'll try to find a valid coloring with as few colors as possible.
    # Use simulated annealing to minimize violations, then minimize color count.
    
    best_colors = None
    best_score = float('inf')
    best_ncolors = n
    
    # Try different initial color counts
    for k in [3, 4, 5, 6, 8, 12, 16, 24, 32]:
        if time.time() > DEADLINE:
            break
        
        # Multiple restarts for each k
        max_restarts = 5 if k <= 6 else 3
        
        for restart in range(max_restarts):
            if time.time() > DEADLINE:
                break
            
            # Initialize with random colors in [0, k)
            colors = np.random.randint(0, k, size=n, dtype=np.int32)
            
            # Simulated annealing
            T = 10.0
            T_min = 0.01
            cooling = 0.995
            max_iters = 500000
            iter_count = 0
            
            current_violations = _count_violations(colors)
            current_ncolors = _count_distinct_colors(colors)
            current_score = current_violations * 1000 + current_ncolors
            
            best_local_violations = current_violations
            best_local_ncolors = current_ncolors
            best_local_colors = colors.copy()
            
            # If already valid, record and try to reduce colors
            if current_violations == 0:
                best_local_violations = 0
                best_local_ncolors = current_ncolors
                best_local_colors = colors.copy()
            
            while T > T_min and iter_count < max_iters:
                if time.time() > DEADLINE:
                    break
                
                # Pick a random position and change its color
                pos = np.random.randint(0, n)
                old_color = colors[pos]
                new_color = np.random.randint(0, k)
                
                if new_color == old_color:
                    continue
                
                colors[pos] = new_color
                new_violations = _count_violations(colors)
                new_ncolors = _count_distinct_colors(colors)
                new_score = new_violations * 1000 + new_ncolors
                
                delta = new_score - current_score
                
                if delta <= 0 or np.random.random() < np.exp(-delta / T):
                    # Accept
                    current_violations = new_violations
                    current_ncolors = new_ncolors
                    current_score = new_score
                    
                    if current_violations < best_local_violations or \
                       (current_violations == best_local_violations and current_ncolors < best_local_ncolors):
                        best_local_violations = current_violations
                        best_local_ncolors = current_ncolors
                        best_local_colors = colors.copy()
                else:
                    # Reject
                    colors[pos] = old_color
                
                T *= cooling
                iter_count += 1
                
                # Early termination if we found a valid coloring with few colors
                if best_local_violations == 0 and best_local_ncolors <= k:
                    break
            
            # Update global best
            if best_local_violations < best_score or \
               (best_local_violations == best_score and best_local_ncolors < best_ncolors):
                best_score = best_local_violations
                best_ncolors = best_local_ncolors
                best_colors = best_local_colors.copy()
            
            # If we found a valid coloring, try to reduce colors further
            if best_local_violations == 0:
                break  # Found valid with k colors, try smaller k in next iteration? No, we're going up.
                # Actually we want to minimize colors, so if valid with k, we should try k-1, k-2, etc.
                # But our loop goes 3,4,5,6,... so if valid with k=5, we should have found it with k=4 first.
                # Unless the search didn't converge. Let's continue to try more restarts.
    
    # If we found a valid coloring, return it
    if best_colors is not None and best_score == 0:
        # Remap colors to consecutive labels 0..k-1
        unique_colors = np.unique(best_colors)
        color_map = {c: i for i, c in enumerate(unique_colors)}
        result = np.array([color_map[c] for c in best_colors], dtype=np.int32)
        return result.tolist()
    
    # Fallback: try a greedy approach with more colors
    if time.time() < DEADLINE - 5:
        # Greedy: assign colors to minimize violations
        colors = np.zeros(n, dtype=np.int32)
        for pos in range(n):
            best_c = 0
            best_v = float('inf')
            for c in range(32):
                colors[pos] = c
                v = _count_violations(colors)
                if v < best_v:
                    best_v = v
                    best_c = c
                if v == 0:
                    break
            colors[pos] = best_c
        
        if _count_violations(colors) == 0:
            unique_colors = np.unique(colors)
            color_map = {c: i for i, c in enumerate(unique_colors)}
            result = np.array([color_map[c] for c in colors], dtype=np.int32)
            return result.tolist()
    
    # Last resort: assign unique colors to all positions (always valid)
    return list(range(n))
