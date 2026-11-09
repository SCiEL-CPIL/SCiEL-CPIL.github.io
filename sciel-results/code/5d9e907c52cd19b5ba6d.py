import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s margin to be safe
    
    n = 128
    
    # 1. Precompute all 4-term APs
    aps_list = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            # Indices are 0-based
            aps_list.append((a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d))
    
    aps = np.array(aps_list, dtype=np.int32)
    num_aps = len(aps)
    
    # Map index to list of AP indices containing it (for potential incremental updates, 
    # but we will use full vectorized recalc for simplicity and speed given small n)
    
    def count_bad_colors(colors):
        """
        Count APs with <= 2 distinct colors using vectorization.
        colors: numpy array of shape (n,)
        Returns: integer count of bad APs
        """
        # Extract colors for each AP
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        
        # An AP is bad if distinct count <= 2
        # Distinct count <= 2 is equivalent to:
        # NOT (distinct count >= 3)
        # Distinct count >= 3 if:
        # (c0 != c1 and c0 != c2 and c0 != c3) OR 
        # (c0 == c1 and (c2 != c3 or (c2 != c0 and c3 != c0))) ... this is getting complex.
        
        # Simpler boolean logic for "at least 3 distinct":
        # If we have at least 3 distinct, then it's NOT the case that all pairs are from a set of size 2.
        # Let's just compute the number of distinct elements for each AP.
        # Since k is small (3-6), we can check:
        # A set {c0, c1, c2, c3} has size <= 2 iff:
        # (c0 == c1 == c2 == c3) OR 
        # (c0 == c1 and c2 == c3 and c0 != c2) OR
        # (c0 == c2 and c1 == c3 and c0 != c1) OR
        # (c0 == c3 and c1 == c2 and c0 != c1) OR
        # (c0 == c1 == c2 and c0 != c3) OR
        # (c0 == c1 == c3 and c0 != c2) OR
        # (c0 == c2 == c3 and c0 != c1) OR
        # (c1 == c2 == c3 and c1 != c0)
        
        # Vectorized check for "bad" (<= 2 distinct):
        # bad if:
        # (c0 == c1) and (c0 == c2) and (c0 == c3)  # All same
        # OR (c0 == c1) and (c2 == c3) and (c0 != c2) # AABB
        # OR (c0 == c2) and (c1 == c3) and (c0 != c1) # ABAB
        # OR (c0 == c3) and (c1 == c2) and (c0 != c1) # ABBA
        # OR (c0 == c1) and (c1 == c2) and (c0 != c3) # AAAB
        # OR (c0 == c1) and (c1 == c3) and (c0 != c2) # AABA
        # OR (c0 == c2) and (c2 == c3) and (c0 != c1) # ABBB (wait, c0=c2=c3, c1 diff) -> ABBA? No.
        
        # Let's use a more robust vectorized distinct count logic.
        # distinct_count = 1 + (c0!=c1) + (c0!=c2 and c1!=c2) + ... 
        # Actually, for 4 elements, distinct count is:
        # 1 if all same
        # 2 if exactly 2 values
        # 3 if exactly 3 values
        # 4 if all distinct
        
        # We want to count if distinct < 3.
        # distinct < 3 is equivalent to:
        # (c0 == c1) and (c0 == c2) and (c0 == c3)
        # OR
        # ( (c0 == c1) and (c2 == c3) and (c0 != c2) )
        # OR
        # ( (c0 == c2) and (c1 == c3) and (c0 != c1) )
        # OR
        # ( (c0 == c3) and (c1 == c2) and (c0 != c1) )
        # OR
        # ( (c0 == c1) and (c1 == c2) and (c0 != c3) )
        # OR
        # ( (c0 == c1) and (c1 == c3) and (c0 != c2) )
        # OR
        # ( (c0 == c2) and (c2 == c3) and (c0 != c1) )
        # OR
        # ( (c1 == c2) and (c2 == c3) and (c1 != c0) )
        
        # Let's simplify:
        # bad = (
        #   (c0 == c1) & (c0 == c2) & (c0 == c3) |
        #   (c0 == c1) & (c2 == c3) & (c0 != c2) |
        #   (c0 == c2) & (c1 == c3) & (c0 != c1) |
        #   (c0 == c3) & (c1 == c2) & (c0 != c1) |
        #   (c0 == c1) & (c1 == c2) & (c0 != c3) |
        #   (c0 == c1) & (c1 == c3) & (c0 != c2) |
        #   (c0 == c2) & (c2 == c3) & (c0 != c1) |
        #   (c1 == c2) & (c2 == c3) & (c1 != c0)
        # )
        
        # Wait, (c0==c1, c1==c2, c0!=c3) covers AAAB.
        # (c0==c1, c1==c3, c0!=c2) covers AABA.
        # (c0==c2, c2==c3, c0!=c1) covers ABBB? No, c0=c2=c3, c1!=c0. This is XYYY.
        # (c1==c2, c2==c3, c1!=c0) covers XXYY? No, c1=c2=c3, c0!=c1. This is XXXY.
        
        # Let's just calculate distinct count directly using numpy unique? No, that's slow.
        # Let's use the property: distinct < 3 iff (c0,c1,c2,c3) can be mapped to a set of size 1 or 2.
        
        # Alternative:
        # is_bad = (
        #   (c0 == c1) & (c0 == c2) & (c0 == c3) |
        #   (c0 == c1) & (c2 == c3) & (c0 != c2) |
        #   (c0 == c2) & (c1 == c3) & (c0 != c1) |
        #   (c0 == c3) & (c1 == c2) & (c0 != c1) |
        #   (c0 == c1) & (c1 == c2) & (c0 != c3) |
        #   (c0 == c1) & (c1 == c3) & (c0 != c2) |
        #   (c0 == c2) & (c2 == c3) & (c0 != c1) |
        #   (c1 == c2) & (c2 == c3) & (c1 != c0)
        # )
        
        # Let's verify coverage:
        # All same: 1st term.
        # 2 distinct:
        #   Pattern AABB: c0=c1, c2=c3, c0!=c2. 2nd term.
        #   Pattern ABAB: c0=c2, c1=c3, c0!=c1. 3rd term.
        #   Pattern ABBA: c0=c3, c1=c2, c0!=c1. 4th term.
        #   Pattern AAAB: c0=c1=c2, c3!=c0. 5th term (c0=c1, c1=c2, c0!=c3).
        #   Pattern AABA: c0=c1=c3, c2!=c0. 6th term (c0=c1, c1=c3, c0!=c2).
        #   Pattern ABBB: c0=c2=c3, c1!=c0. 7th term (c0=c2, c2=c3, c0!=c1).
        #   Pattern BAAA: c1=c2=c3, c0!=c1. 8th term (c1=c2, c2=c3, c1!=c0).
        
        # This covers all cases of <= 2 distinct colors.
        
        bad_mask = (
            (c0 == c1) & (c0 == c2) & (c0 == c3) |
            (c0 == c1) & (c2 == c3) & (c0 != c2) |
            (c0 == c2) & (c1 == c3) & (c0 != c1) |
            (c0 == c3) & (c1 == c2) & (c0 != c1) |
            (c0 == c1) & (c1 == c2) & (c0 != c3) |
            (c0 == c1) & (c1 == c3) & (c0 != c2) |
            (c0 == c2) & (c2 == c3) & (c0 != c1) |
            (c1 == c2) & (c2 == c3) & (c1 != c0)
        )
        
        return np.sum(bad_mask)

    # 2. Get baseline valid solution using the seed's greedy method
    # This ensures we have a valid solution to return if optimization fails.
    def greedy_solve():
        colors = []
        for x in range(1, 129):
            found = False
            for color in range(128):
                # Check constraint 1: All 4-term APs ending at x
                # (x-1-j*d) for j in 1,2,3 must be in colors
                # x is 1-based in loop, colors is 0-based list.
                # Index in colors: x-1.
                # AP: a, a+d, a+2d, a+3d.
                # If a+3d = x-1 (0-based index), then a = x-1-3d.
                # Positions: x-1-3d, x-1-2d, x-1-d, x-1.
                # Colors at these positions must have >= 3 distinct values.
                
                valid = True
                max_d1 = (x - 1) // 3
                for d in range(1, max_d1 + 1):
                    idx3 = x - 1
                    idx2 = x - 1 - d
                    idx1 = x - 1 - 2 * d
                    idx0 = x - 1 - 3 * d
                    if idx0 < 0:
                        continue
                    c3 = colors[idx3]
                    c2 = colors[idx2]
                    c1 = colors[idx1]
                    c0 = colors[idx0]
                    # Check if {c0, c1, c2, c3, color} has >= 3 distinct?
                    # No, the AP is (idx0, idx1, idx2, idx3).
                    # The new color is for idx3? No, we are assigning color to x-1.
                    # So the AP is (x-1-3d, x-1-2d, x-1-d, x-1).
                    # The colors are colors[idx0], colors[idx1], colors[idx2], color.
                    distinct_count = len({colors[idx0], colors[idx1], colors[idx2], color})
                    if distinct_count < 3:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check constraint 2: APs where x is not the last element?
                # The seed code checks:
                # all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1, (x-1)//2+1))
                # This checks APs (x-1-2d, x-1-d, x-1, x-1+d)?
                # If we assign color to x-1, we form APs where x-1 is the 2nd or 3rd element?
                # Actually, the seed's logic is a specific greedy that works.
                # Let's just replicate the seed's logic exactly to get a valid baseline.
                
                valid2 = True
                max_d2 = (x - 1) // 2
                for d in range(1, max_d2 + 1):
                    # AP: (x-1-2d, x-1-d, x-1, x-1+d)
                    # We don't know x-1+d yet.
                    # The seed checks: any(colors[x-1-j*d] != color for j in (1,2))
                    # j=1: colors[x-1-d] != color
                    # j=2: colors[x-1-2d] != color
                    # This ensures that the first 3 elements of the AP (x-1-2d, x-1-d, x-1) are not all the same color?
                    # If colors[x-1-d] == color AND colors[x-1-2d] == color, then the first 3 are (color, color, color).
                    # Then the 4th element (future) can only be 1 color to keep distinct <= 2?
                    # If future is color, distinct=1. If future is other, distinct=2.
                    # So if first 3 are same, we are forced to have distinct <= 2 regardless of future?
                    # Yes. So we must prevent first 3 from being same.
                    # This is a valid necessary condition.
                    
                    if colors[x - 1 - d] == color and colors[x - 1 - 2 * d] == color:
                        valid2 = False
                        break
                
                if valid2:
                    colors.append(color)
                    found = True
                    break
            if not found:
                # Should not happen with 128 colors
                colors.append(0)
        return colors

    baseline_colors = greedy_solve()
    baseline_k = len(set(baseline_colors))
    
    best_sol = baseline_colors
    best_val = baseline_k
    
    # 3. Try to improve with k < baseline_k
    # We want to minimize the number of labels used.
    # We will try k = 3, 4, ... up to baseline_k - 1.
    
    # We use Simulated Annealing for each k.
    # If we find a valid solution with k colors, we return it (since it's better than baseline).
    
    for k in range(3, baseline_k):
        if time.time() > deadline - 5:
            break
            
        # Try to find a solution with exactly k colors (or fewer, but we restrict palette to 0..k-1)
        # If a solution uses fewer than k colors, it's still valid for "minimize labels", 
        # but our SA is restricted to palette 0..k-1. 
        # If it finds a solution using only 2 colors, that would be invalid (needs >=3 distinct in APs).
        # So we are looking for a valid coloring using subset of {0..k-1}.
        
        found = False
        # Multiple restarts
        num_restarts = 20
        for _ in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            colors = np.random.randint(0, k, n)
            current_cost = count_bad_colors(colors)
            
            if current_cost == 0:
                # Check distinct count
                distinct_used = len(set(colors))
                if distinct_used <= k:
                    best_sol = list(colors)
                    best_val = distinct_used
                    found = True
                    break
            
            # Simulated Annealing
            T = 1.0
            max_iters = 5000
            no_improve = 0
            
            for _ in range(max_iters):
                if time.time() > deadline - 5:
                    break
                    
                if current_cost == 0:
                    distinct_used = len(set(colors))
                    best_sol = list(colors)
                    best_val = distinct_used
                    found = True
                    break
                    
                i = np.random.randint(n)
                old_color = colors[i]
                new_color = np.random.randint(0, k)
                if new_color == old_color:
                    continue
                
                colors[i] = new_color
                new_cost = count_bad_colors(colors)
                delta = new_cost - current_cost
                
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_cost = new_cost
                    no_improve = 0
                else:
                    colors[i] = old_color
                    no_improve += 1
                    if no_improve > 200:
                        break
                        
                T *= 0.99
                if T < 0.01:
                    T = 0.01
            
            if found:
                break

        if found:
            # We found a solution with k (or fewer) colors.
            # Since we iterate k from 3 upwards, the first valid k is the minimum.
            # However, it's possible we found a solution with k=3 in the k=4 loop?
            # No, if k=3 loop failed, k=4 loop might find a solution using only 3 colors?
            # If it uses only 3 colors, it's a valid 3-coloring.
            # We should check if best_val < k.
            # If best_val is 3, we can break immediately.
            if best_val <= 3:
                break
            # If best_val is 4, and we are in k=4 loop, we have a valid 4-coloring.
            # We should continue to see if k=3 was possible? 
            # No, we already tried k=3. If k=3 failed, we assume no 3-coloring exists or we missed it.
            # Given time constraints, we accept the best found so far.
            break

    # If we didn't find anything better, return baseline
    return best_sol

