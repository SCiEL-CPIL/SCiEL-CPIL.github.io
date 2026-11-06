import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    
    # Precompute all non-constant 4-term APs
    # Each AP is a tuple of 4 indices (a, a+d, a+2d, a+3d)
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Convert to numpy array for vectorized operations
    # Shape: (num_aps, 4)
    if not aps:
        return list(range(n))
        
    ap_array = np.array(aps, dtype=np.int32)
    num_aps = len(aps)
    
    def count_violations(colors):
        """
        Vectorized count of 4-APs with fewer than 3 distinct colors.
        colors: numpy array of shape (n,)
        """
        # Get colors for each position in each AP
        # c0, c1, c2, c3 for each AP
        c0 = colors[ap_array[:, 0]]
        c1 = colors[ap_array[:, 1]]
        c2 = colors[ap_array[:, 2]]
        c3 = colors[ap_array[:, 3]]
        
        # An AP is valid if it has at least 3 distinct colors.
        # It is INVALID (violation) if it has 1 or 2 distinct colors.
        # 
        # Case 1: All 4 same (1 distinct)
        # c0 == c1 == c2 == c3
        all_same = (c0 == c1) & (c1 == c2) & (c2 == c3)
        
        # Case 2: Exactly 2 distinct colors.
        # This happens if the set {c0, c1, c2, c3} has size 2.
        # We can check this by seeing if the number of distinct values is 2.
        # A faster way: 
        # If not all same, it has 2 distinct iff:
        # (c0 == c1 and c2 == c3 and c0 != c2) OR
        # (c0 == c2 and c1 == c3 and c0 != c1) OR
        # (c0 == c3 and c1 == c2 and c0 != c1) OR
        # (c0 == c1 and c0 == c2 and c1 != c3) -> wait, this is 2 distinct? No, c0,c1,c2 same, c3 diff. Set size 2.
        # Actually, simpler: 
        # Distinct count is 2 iff:
        # (c0 == c1) and (c2 == c3) and (c0 != c2)  -> Pattern AABA? No, AA BB.
        # Let's just compute distinct counts per row. 
        # Since n is small (4), we can just do:
        # num_distinct = 4 - (c0==c1) - (c0==c2) - (c0==c3) - (c1==c2) - (c1==c3) - (c2==c3)
        # This formula works for set size?
        # If all same: 4 - 6 = -2. (Should be 1).
        # The formula $4 - \sum_{i<j} [c_i == c_j]$ gives the number of distinct elements ONLY if we assume something?
        # No, for a set of size k, the number of equal pairs is $\binom{k}{2}$.
        # So $4 - \binom{k}{2}$? No.
        # Let's use a different approach.
        # We want to know if distinct count < 3.
        # This is equivalent to: distinct count == 1 OR distinct count == 2.
        
        # Distinct count == 1: all_same
        # Distinct count == 2: 
        # This is true if NOT all_same AND (c0 == c1 or c0 == c2 or c0 == c3) AND (c1 == c2 or c1 == c3) AND (c2 == c3) ... this is getting complex.
        
        # Alternative: 
        # An AP is invalid if it is NOT the case that it has at least 3 distinct colors.
        # It has at least 3 distinct colors if:
        # (c0 != c1 and c0 != c2 and c0 != c3) -> c0 is unique among others? No, c1,c2,c3 could be same. 
        # If c0 is different from c1, c2, c3, then we have at least 2 colors. If c1,c2,c3 are not all same, we have 3.
        
        # Let's just calculate the number of distinct colors for each AP using a trick or simple logic.
        # Given small size (4), we can use:
        # distinct = 1 + (c0 != c1) + (c1 != c2) + (c2 != c3) ... no.
        
        # Let's use the property: 
        # If we sort the 4 values, distinct count is 1 + count of changes.
        # But sorting 4 elements vectorized is hard.
        
        # Let's stick to boolean logic for "at least 3 distinct".
        # It fails (violation) if:
        # 1. All 4 are same.
        # 2. They form a pattern with only 2 colors.
        #    Patterns with 2 colors (up to permutation):
        #    AAAA (1 color)
        #    AAAB (2 colors)
        #    AABB (2 colors)
        #    ABAB (2 colors)
        #    ABBA (2 colors)
        #    AABA (2 colors)
        #    ABAA (2 colors)
        #    
        #    Basically, any combination where the set of values has size 1 or 2.
        
        # We can check if size is 1 or 2 by checking if:
        # (c0 == c1 and c2 == c3 and c0 == c2) -> All same
        # OR
        # (c0 == c1 and c0 == c2 and c0 == c3) -> All same
        # ...
        
        # Easier: 
        # Let's compute the number of distinct values by checking pairs.
        # Let $E$ be the number of equal pairs among the 4.
        # If distinct=1, E=6.
        # If distinct=2, E can be:
        #   3+1 split: 3 pairs in group of 3, 0 in group of 1? No.
        #   Group sizes: 2,2 -> pairs: 1+1=2.
        #   Group sizes: 3,1 -> pairs: 3+0=3.
        # If distinct=3, E can be:
        #   Group sizes: 2,1,1 -> pairs: 1+0+0=1.
        # If distinct=4, E=0.
        
        # So:
        # Distinct=1 <=> E=6
        # Distinct=2 <=> E in {2, 3}
        # Distinct=3 <=> E=1
        # Distinct=4 <=> E=0
        
        # We want to count violations (Distinct < 3).
        # So we want to count APs where E >= 2.
        
        eq_01 = (c0 == c1)
        eq_02 = (c0 == c2)
        eq_03 = (c0 == c3)
        eq_12 = (c1 == c2)
        eq_13 = (c1 == c3)
        eq_23 = (c2 == c3)
        
        E = eq_01 + eq_02 + eq_03 + eq_12 + eq_13 + eq_23
        
        violations = (E >= 2).sum()
        return int(violations)

    def get_distinct_count(colors):
        return len(set(colors.tolist()))

    # Initial valid coloring using many colors
    # Use a simple greedy to get a valid start
    def generate_initial_valid():
        colors = np.zeros(n, dtype=np.int32)
        # Assign random colors from a large palette, then fix violations greedily?
        # Or just use the seed's greedy logic but with a large palette.
        # Let's use a simple randomized greedy with a large palette (e.g., 20)
        palette = list(range(20))
        random.shuffle(palette)
        
        # Reuse the logic from the seed but simpler
        aps_ending = [[] for _ in range(n)]
        for i in range(n):
            for d in range(1, i // 3 + 1):
                a = i - 3 * d
                if a >= 0:
                    aps_ending[i].append((a, d))
                    
        for i in range(n):
            valid_colors = []
            for c in palette:
                ok = True
                for a, d in aps_ending[i]:
                    vals = {colors[a], colors[a+d], colors[a+2*d], c}
                    if len(vals) < 3:
                        ok = False
                        break
                if ok:
                    valid_colors.append(c)
            if valid_colors:
                colors[i] = random.choice(valid_colors)
            else:
                # Fallback: assign a new unique color (should not happen with palette 20)
                colors[i] = 100 + i
                
        # Verify
        if count_violations(colors) == 0:
            return colors
        return None

    best_colors = None
    best_distinct = n
    
    # Try to find a good initial valid coloring
    for _ in range(10):
        init = generate_initial_valid()
        if init is not None:
            distinct = get_distinct_count(init)
            if distinct < best_distinct:
                best_distinct = distinct
                best_colors = init.copy()
            if time.time() > deadline - 10:
                break

    if best_colors is None:
        # Fallback to all distinct
        return list(range(n))

    # Simulated Annealing to merge colors
    T = 1.0
    cooling_rate = 0.9995
    
    current_colors = best_colors.copy()
    current_violations = count_violations(current_colors)
    current_distinct = get_distinct_count(current_colors)
    
    # We want to minimize distinct count, keeping violations 0.
    # Strategy: 
    # 1. If violations > 0, we must reduce violations first.
    # 2. If violations == 0, we try to merge colors.
    
    # Actually, let's just run SA on the objective: 
    # Objective = Violations * 1000 + Distinct_Count
    # This prioritizes validity, then minimizes colors.
    
    def objective(colors):
        v = count_violations(colors)
        d = get_distinct_count(colors)
        return v * 1000 + d

    current_obj = objective(current_colors)
    
    # SA Loop
    steps = 0
    while time.time() < deadline - 5:
        steps += 1
        
        # Get list of currently used colors
        used_colors = list(set(current_colors.tolist()))
        
        if len(used_colors) < 2:
            break
            
        # Pick two random colors to merge
        c_target, c_source = random.sample(used_colors, 2)
        
        # Create new state
        new_colors = current_colors.copy()
        mask = new_colors == c_source
        new_colors[mask] = c_target
        
        new_obj = objective(new_colors)
        delta = new_obj - current_obj
        
        if delta <= 0 or random.random() < np.exp(-delta / T):
            current_colors = new_colors
            current_obj = new_obj
            current_violations = count_violations(current_colors)
            current_distinct = get_distinct_count(current_colors)
            
            if current_violations == 0 and current_distinct < best_distinct:
                best_distinct = current_distinct
                best_colors = current_colors.copy()
                
        T *= cooling_rate
        
        # If we've reached 2 colors and 0 violations, we can stop? 
        # No, we want to see if we can get to 1? 
        # "Every nonconstant four-term progression ... must use AT LEAST three distinct colors."
        # If we use 1 color, every AP uses 1 color. Violation.
        # If we use 2 colors, every AP uses at most 2 colors. Violation.
        # So minimum colors is at least 3.
        if best_distinct == 3:
            break

    # Final check and return
    if best_colors is not None and count_violations(best_colors) == 0:
        return best_colors.tolist()
    
    # Fallback
    return list(range(n))
