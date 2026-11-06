import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline, 10s margin
    
    n = 128
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) where all in [0, n-1]
    ap_list = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    ap_array = np.array(ap_list, dtype=np.int32)
    num_aps = len(ap_array)
    
    # Helper: count violations for a given coloring
    def count_violations(colors):
        """Return number of 4-term APs with < 3 distinct colors."""
        c = np.array(colors, dtype=np.int8)
        # For each AP, check if it uses < 3 distinct colors
        # c[ap_array] gives shape (num_aps, 4)
        vals = c[ap_array]  # (num_aps, 4)
        # Count distinct colors: an AP has < 3 distinct if it's monochromatic or bichromatic
        # Monochromatic: all 4 same
        mono = (vals[:, 0] == vals[:, 1]) & (vals[:, 1] == vals[:, 2]) & (vals[:, 2] == vals[:, 3])
        # Bichromatic: exactly 2 distinct colors
        # Check: not mono, and at most 2 distinct
        # A 4-tuple has exactly 2 distinct if it's not all same and (sorted has at most 2 unique)
        # Simpler: check if it's bichromatic by checking patterns
        # Actually, let's just count distinct per row
        # Use a trick: for each row, count how many are equal to first, second, etc.
        # Faster: an AP has < 3 distinct iff it is NOT (all 4 different) AND NOT (exactly 3 different)
        # Actually, < 3 distinct means 1 or 2 distinct.
        # 1 distinct: all same
        # 2 distinct: exactly 2 values appear
        # Let's compute: for each AP, the number of distinct colors
        # We can do: distinct = 4 - (number of "collisions") but that's complex
        # Alternative: use the fact that < 3 distinct means the set has size 1 or 2
        # Let's just check: is it mono? if yes, violation.
        # If not mono, check if it's bichromatic: 
        # A 4-tuple is bichromatic iff it's not mono and one of these patterns:
        # (x,x,x,y), (x,x,y,x), (x,x,y,y), (x,y,x,x), (x,y,x,y), (x,y,y,x), (x,y,y,y)
        # But with only 2 colors, any non-mono 4-tuple is bichromatic!
        # Wait no - with 3+ colors, a non-mono 4-tuple could have 3 or 4 distinct.
        # So: violation = mono OR (exactly 2 distinct)
        # To check exactly 2 distinct: not mono, and at least one pair is equal, and not all pairs different
        # Simpler approach: for each row, check if all 4 are same (mono) or if the set has size 2.
        # Let's use: for a row (a,b,c,d), it has < 3 distinct iff:
        #   (a==b==c==d) OR 
        #   (a==b and c==d and a!=c) OR  # AABB
        #   (a==c and b==d and a!=b) OR  # ABAB
        #   (a==d and b==c and a!=b) OR  # ABBA
        #   (a==b and a!=c and a!=d and c==d)  # AA BB but that's AABB already
        #   Actually, let me think differently.
        #   2 distinct means exactly 2 values. The patterns for 4 positions with exactly 2 values:
        #   Count of value1 can be 1,2,3 (not 0 or 4 since not mono)
        #   This is getting complex. Let me just use numpy to count distinct.
        
        # Efficient way: for each row, check if it's mono
        # Then for non-mono, check if it has exactly 2 distinct
        # A row has exactly 2 distinct iff: it's not mono, and (min != max) and (the row has at most 2 unique values)
        # Let's just do: sort each row and check
        # But sorting 4 elements per row for ~1000 rows is fine
        
        # Actually, simplest vectorized approach:
        # A 4-tuple has < 3 distinct iff it is NOT (all four pairwise distinct in a 3+ way)
        # Let me just count: for each row, number of distinct = ?
        # Use: distinct_count = 4 - sum over pairs of (indicator they're equal) ... no that's not right.
        
        # Let me just do it with a loop-free approach:
        # For each AP, the number of distinct colors is at least 3 iff:
        # Not mono AND not bichromatic
        # Mono: all 4 equal
        # Bichromatic: exactly 2 distinct values
        # A 4-tuple is bichromatic iff: it's not mono AND (there exist i,j,k,l such that...)
        # 
        # OK let me just use: for each row, check if it has 1 or 2 distinct values
        # I'll compute: for each row, is it mono? Then for non-mono, check if max-min pattern gives 2 distinct
        # 
        # Vectorized check for exactly 2 distinct in a 4-tuple:
        # A 4-tuple (a,b,c,d) has exactly 2 distinct iff:
        # (a==b) and (c==d) and (a!=c)  -> AABB
        # (a==c) and (b==d) and (a!=b)  -> ABAB  
        # (a==d) and (b==c) and (a!=b)  -> ABBA
        # (a==b) and (a==c) and (b!=d)  -> AAAB
        # (a==b) and (a==d) and (b!=c)  -> AABA
        # (a==c) and (a==d) and (b!=c)  -> ABAA
        # (b==c) and (b==d) and (a!=c)  -> ABBA... wait that's same as above
        # (b==d) and (c==d) and (a!=b)  -> ABAB... 
        # (a==b) and (c!=a) and (c!=d) and (a!=d) ... this is getting messy
        
        # Let me just use a different approach: 
        # For each row, compute the number of distinct values by checking all 6 pairs
        # A 4-tuple has exactly k distinct values. 
        # < 3 distinct means 1 or 2 distinct.
        # 1 distinct: all 6 pairs equal
        # 2 distinct: not all 6 pairs equal, but the 4 values take only 2 distinct values
        
        # I'll just use: violation if mono OR (bichromatic)
        # Bichromatic check: the 4 values, when you take the set, has size 2
        # Equivalent: not mono, and (a==b or a==c or a==d) and (b==c or b==d or c==d) and not all same
        # Actually: a 4-tuple has exactly 2 distinct iff it's not mono and at least one of the following:
        # The first element equals at least one other, AND the last element equals at least one other, AND not all same
        # This is still complex.
        
        # Let me just do it the simple way with numpy:
        # For each row, count distinct by: 
        #   distinct = 1 + (b!=a) + (c!=a and c!=b) + (d!=a and d!=b and d!=c)
        # This gives the number of distinct values!
        
        a, b, c, d = vals[:, 0], vals[:, 1], vals[:, 2], vals[:, 3]
        distinct_count = (
            1 + 
            (b != a).astype(np.int8) + 
            ((c != a) & (c != b)).astype(np.int8) + 
            ((d != a) & (d != b) & (d != c)).astype(np.int8)
        )
        violations = (distinct_count < 3).sum()
        return int(violations)
    
    # Start with random 3-color assignment
    best_colors = None
    best_violations = float('inf')
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Random initial coloring with 3 colors
        colors = np.random.randint(0, 3, size=n, dtype=np.int8)
        violations = count_violations(colors)
        
        if violations == 0:
            # Found a valid 3-coloring!
            # Convert to list of ints in 0..127
            return [int(c) for c in colors]
        
        # Simulated annealing
        if violations < best_violations:
            best_violations = violations
            best_colors = colors.copy()
        
        temp = 1.0
        cooling = 0.9995
        max_iter = 5000
        for i in range(max_iter):
            if time.time() > deadline:
                break
            temp *= cooling
            if temp < 0.01:
                break
            
            # Pick a random position to change
            pos = np.random.randint(0, n)
            old_color = colors[pos]
            new_color = np.random.randint(0, 3)
            if new_color == old_color:
                continue
            
            colors[pos] = new_color
            new_violations = count_violations(colors)
            
            delta = new_violations - violations
            if delta <= 0 or random.random() < np.exp(-delta / temp):
                violations = new_violations
                if violations < best_violations:
                    best_violations = violations
                    best_colors = colors.copy()
                    if violations == 0:
                        return [int(c) for c in colors]
            else:
                colors[pos] = old_color
        
        if best_violations == 0:
            return [int(c) for c in best_colors]
    
    # If we didn't find a 3-coloring, try with more colors or return best effort
    # Return best_colors mapped to 0..127
    if best_colors is not None:
        return [int(c) for c in best_colors]
    
    # Fallback: sequential coloring (should always be valid)
    colors = []
    for x in range(1, 129):
        for color in range(128):
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                if x - 1 - 3 * d >= 0:
                    ap_colors = set()
                    for j in range(4):
                        idx = x - 1 - j * d
                        if idx >= 0:
                            ap_colors.add(colors[idx])
                        else:
                            valid = False
                            break
                    if not valid:
                        break
                    if len(ap_colors | {color}) < 3:
                        valid = False
                        break
            if valid:
                colors.append(color)
                break
        if not colors or len(colors) != x:
            colors.append(x - 1)  # fallback
    return colors
