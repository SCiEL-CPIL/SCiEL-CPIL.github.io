import time
import itertools

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    def count_distinct(arr):
        return len(set(arr))
    
    def is_valid_4tp(colors, n):
        """Check if every nonconstant 4-term AP uses >=3 distinct colors."""
        for a in range(n):
            for d in range(1, (n - 1 - a) // 3 + 1):
                c = {colors[a], colors[a+d], colors[a+2*d], colors[a+3*d]}
                if len(c) < 3:
                    return False
        return True
    
    def is_valid_4tp_fast(colors, n):
        """Faster check using vectorized approach."""
        import numpy as np
        c = np.array(colors)
        # For each (a, d), check the 4-term AP
        for a in range(n):
            max_d = (n - 1 - a) // 3
            for d in range(1, max_d + 1):
                vals = c[a:a+4*d:d]
                if len(set(vals.tolist())) < 3:
                    return False
        return True
    
    # Strategy 1: Try modular constructions i mod m
    # For i in 1..128, color = i mod m (gives values 0..m-1)
    # Check m = 3, 4, 5, ...
    
    best_colors = None
    best_count = 128  # worst case: all distinct
    
    # Try modular m from 3 upward
    for m in range(3, 128):
        if time.time() > deadline:
            break
        colors = [(i) % m for i in range(1, 129)]
        distinct = len(set(colors))
        if distinct >= best_count:
            # Can't improve, skip verification (but m>best_count means distinct=m>best_count)
            if m > best_count:
                continue
        if is_valid_4tp(colors, 128):
            best_colors = colors
            best_count = distinct
            if best_count == 3:
                break  # can't do better than 3
    
    # Strategy 2: If modular didn't find 3, try to verify if 3 colors is possible
    # by checking if the greedy approach can achieve it
    if best_count > 3 and time.time() < deadline - 5:
        # Try greedy with 3 colors
        colors = [-1] * 128
        colors[0] = 0
        colors[1] = 1
        colors[2] = 2
        success = True
        for x in range(3, 128):
            if time.time() > deadline - 2:
                success = False
                break
            # Try each of the 3 colors
            placed = False
            for color in range(3):
                valid = True
                # Check all 4-term APs where x is the last term: x = a+3d
                for d in range(1, (x + 1) // 3):
                    a = x - 3 * d
                    if a >= 0:
                        ap_colors = {colors[a], colors[a+d], colors[a+2*d], color}
                        if len(ap_colors) < 3:
                            valid = False
                            break
                if not valid:
                    continue
                # Check all 4-term APs where x is the second term: x = a+d
                for d in range(1, (x + 1) // 4):
                    a = x - d
                    if a >= 0 and a + 3*d < 128:
                        # We can only check if the other terms are already placed
                        # Actually for forward-looking, we just need to ensure
                        # that the AP won't be forced to have <3 colors
                        # For now, only check completed APs
                        pass
                if valid:
                    colors[x] = color
                    placed = True
                    break
            if not placed:
                success = False
                break
        if success and all(c >= 0 for c in colors):
            if is_valid_4tp(colors, 128):
                if len(set(colors)) < best_count:
                    best_colors = colors
                    best_count = len(set(colors))
    
    if best_colors is None:
        # Fallback: all distinct
        best_colors = list(range(128))
    
    return best_colors
