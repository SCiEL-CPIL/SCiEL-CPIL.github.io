import numpy as np
import time
import random

def construct():
    n = 128
    deadline = time.time() + 55.0  # 55s safety margin within 60s limit
    
    # Precompute all non-constant 4-APs as index arrays
    # AP: (a, a+d, a+2d, a+3d) for a in 1..n, d >= 1, a+3d <= n
    ap_indices = []
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            # Convert to 0-indexed
            ap_indices.append(np.array([a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1], dtype=np.int32))
    
    if not ap_indices:
        return [0] * n
    
    # Stack into a matrix: shape (num_aps, 4)
    ap_matrix = np.array(ap_indices, dtype=np.int32)
    num_aps = ap_matrix.shape[0]
    
    # For each AP, precompute the 4 vertex indices
    # We'll use this to check if an AP has < 3 distinct colors
    
    # Helper: count distinct colors in each AP given a coloring
    # colors: shape (n,) with values in 0..k-1
    # Returns: boolean array of shape (num_aps,) where True means "bad" (< 3 distinct)
    def count_bad_aps(colors):
        # colors: (n,)
        # ap_colors: (num_aps, 4)
        ap_colors = colors[ap_matrix]
        
        # For each AP, check if it has < 3 distinct colors
        # A 4-element set has < 3 distinct iff:
        # - All 4 same (1 distinct)
        # - 3 same, 1 different (2 distinct)
        # - 2 pairs of same (2 distinct)
        
        # Let's check: is it "bad"?
        # Bad if: (c0==c1==c2==c3) OR (exactly 2 distinct)
        
        # Compute pairwise equalities
        c0 = ap_colors[:, 0]
        c1 = ap_colors[:, 1]
        c2 = ap_colors[:, 2]
        c3 = ap_colors[:, 3]
        
        # All same
        all_same = (c0 == c1) & (c1 == c2) & (c2 == c3)
        
        # Exactly 2 distinct: 
        # Case 1: c0==c1, c2==c3, c0!=c2 (two pairs)
        # Case 2: c0==c2, c1==c3, c0!=c1 (cross pairs)
        # Case 3: c0==c3, c1==c2, c0!=c1 (other cross)
        # Case 4: three same, one different:
        #   c0==c1==c2 != c3
        #   c0==c1==c3 != c2
        #   c0==c2==c3 != c1
        #   c1==c2==c3 != c0
        
        # Simpler: count distinct by checking if any pair differs
        # Actually, let's just compute the number of distinct values efficiently
        
        # For each AP, the number of distinct values is:
        # 1 if all same
        # 2 if not all same but some specific patterns
        # 3 or 4 otherwise
        
        # Let's use: distinct_count = 1 + (c0!=c1) + (c0!=c2 and c1!=c2) + (c0!=c3 and c1!=c3 and c2!=c3)
        # This is the standard way to count distinct in a small set
        
        d1 = (c0 != c1).astype(np.int8)
        d2 = (c0 != c2) & (c1 != c2)
        d3 = (c0 != c3) & (c1 != c3) & (c2 != c3)
        
        distinct_count = 1 + d1 + d2.astype(np.int8) + d3.astype(np.int8)
        
        # Bad if distinct_count < 3
        return distinct_count < 3
    
    # Initialize best result
    best_colors = None
    best_bad_count = num_aps + 1
    
    # Try small number of colors first
    for k in [3, 4, 5]:
        if time.time() > deadline:
            break
            
        # Run multiple restarts
        for restart in range(500):
            if time.time() > deadline:
                break
                
            # Random initialization
            colors = np.random.randint(0, k, size=n, dtype=np.int32)
            bad_mask = count_bad_aps(colors)
            bad_count = np.sum(bad_mask)
            
            if bad_count < best_bad_count:
                best_bad_count = bad_count
                best_colors = colors.copy()
                
            if bad_count == 0:
                return colors.tolist()
            
            # Local search: hill climbing with random kicks
            stuck = 0
            while bad_count > 0 and time.time() < deadline:
                # Find a bad AP
                bad_indices = np.where(bad_mask)[0]
                if len(bad_indices) == 0:
                    break
                    
                # Pick a random bad AP
                ap_idx = bad_indices[random.randint(0, len(bad_indices) - 1)]
                ap = ap_matrix[ap_idx]
                
                # Try changing each vertex in this AP to each color
                best_delta = 0
                best_move = None
                
                for v in ap:
                    old_color = colors[v]
                    for new_color in range(k):
                        if new_color == old_color:
                            continue
                        
                        # Apply change
                        colors[v] = new_color
                        new_bad_mask = count_bad_aps(colors)
                        new_bad_count = np.sum(new_bad_mask)
                        delta = new_bad_count - bad_count
                        
                        # Revert
                        colors[v] = old_color
                        
                        if delta < best_delta:
                            best_delta = delta
                            best_move = (v, new_color)
                            if delta < 0:
                                # Keep going to find best
                                pass
                
                if best_move is not None and best_delta < 0:
                    v, new_color = best_move
                    colors[v] = new_color
                    bad_mask = count_bad_aps(colors)
                    bad_count = np.sum(bad_mask)
                    stuck = 0
                    
                    if bad_count < best_bad_count:
                        best_bad_count = bad_count
                        best_colors = colors.copy()
                        
                    if bad_count == 0:
                        return colors.tolist()
                else:
                    # No improving move found, random kick
                    stuck += 1
                    if stuck > 50:
                        v = ap[random.randint(0, 3)]
                        new_color = random.randint(0, k - 1)
                        colors[v] = new_color
                        bad_mask = count_bad_aps(colors)
                        bad_count = np.sum(bad_mask)
                        stuck = 0
                        
                        if bad_count < best_bad_count:
                            best_bad_count = bad_count
                            best_colors = colors.copy()

    # Return best found
    if best_colors is not None:
        return best_colors.tolist()
    else:
        return [0] * n
