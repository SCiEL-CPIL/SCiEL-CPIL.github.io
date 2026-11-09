import random
import time

def count_violations(colors, n=128):
    """Count the number of 4-APs that use fewer than 3 distinct colors."""
    violations = 0
    for a in range(1, n + 1):
        # d is the difference
        # a, a+d, a+2d, a+3d must be <= n
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            c1 = colors[a - 1]
            c2 = colors[a + d - 1]
            c3 = colors[a + 2 * d - 1]
            c4 = colors[a + 3 * d - 1]
            distinct = len({c1, c2, c3, c4})
            if distinct < 3:
                violations += 1
    return violations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    n = 128
    
    # Try to find a 3-coloring
    best_colors = None
    best_violations = float('inf')
    
    # Initial random 3-coloring
    random.seed(42)
    colors = [random.randint(0, 2) for _ in range(n)]
    
    # Local search
    for iteration in range(100000):
        if time.time() > deadline:
            break
            
        v = count_violations(colors, n)
        if v < best_violations:
            best_violations = v
            best_colors = colors[:]
            if v == 0:
                return colors
        
        # Pick a random position to change
        pos = random.randint(0, n - 1)
        old_color = colors[pos]
        
        # Try changing to a different color (within 0..2)
        new_color = (old_color + 1) % 3
        
        colors[pos] = new_color
        v_new = count_violations(colors, n)
        
        # If it didn't improve or got worse, revert (or accept with prob, but simple hill climb is fine for small N)
        if v_new > v:
            colors[pos] = old_color
        elif v_new == v:
            # Keep or revert, random walk
            if random.random() < 0.5:
                colors[pos] = old_color
    
    if best_colors is not None and best_violations == 0:
        return best_colors

    # If 3 colors failed, try 4 colors
    # The seed program likely produces a valid 4-coloring.
    # We can use a simple greedy for 4 colors as fallback, or continue search.
    # But 3 is the theoretical minimum for this specific constraint on N=128?
    # Actually, let's just return the best found or a known valid pattern.
    
    # Fallback: Greedy 4-coloring similar to seed but optimized
    colors = [0] * n
    for x in range(1, n + 1):
        for color in range(4):
            valid = True
            # Check all 4-APs ending at x or involving x?
            # Simpler: just check all 4-APs containing x after assignment.
            # This is O(N^2) per step, so O(N^3) total. 128^3 is ~2M, fast.
            
            # Actually, the seed logic is:
            # 1. All 4-APs ending at x must have >=3 colors.
            # 2. All 3-APs ending at x must not allow a 4th color to make it 2-color?
            # The seed logic was specific.
            
            # Let's just use the local search result if it's valid, otherwise a simple valid 4-coloring.
            # A known valid 4-coloring for 4-APs with >=3 colors is easy to find.
            # For example, c(x) = x % 4?
            # Check d=4: a, a, a, a -> 1 color. Fail.
            
            # Let's use the local search result even if not 0, but we need a VALID solution.
            # If local search didn't find 0 violations, we must ensure validity.
            # The seed program is guaranteed valid.
            # Let's run a fast greedy for 4 colors.
            
            valid = True
            # Check all 4-APs that include x
            for a in range(1, x + 1):
                for d in range(1, (n - a) // 3 + 1):
                    # AP: a, a+d, a+2d, a+3d
                    # If any of these is x, we need to check the set
                    if a == x or a+d == x or a+2*d == x or a+3*d == x:
                        c_vals = []
                        for k in range(4):
                            idx = a + k*d
                            if idx == x:
                                c_vals.append(color)
                            else:
                                c_vals.append(colors[idx-1])
                        if len(set(c_vals)) < 3:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                colors[x-1] = color
                break
        else:
            # Should not happen with 4 colors
            colors[x-1] = 0

    return colors

# EVOLVE-BLOCK-START
def construct():
    import random
    import time
    
    start_time = time.time()
    deadline = start_time + 55  # 5s margin
    
    n = 128
    
    def count_violations(colors):
        violations = 0
        for a in range(1, n + 1):
            max_d = (n - a) // 3
            for d in range(1, max_d + 1):
                c1 = colors[a - 1]
                c2 = colors[a + d - 1]
                c3 = colors[a + 2 * d - 1]
                c4 = colors[a + 3 * d - 1]
                distinct = len({c1, c2, c3, c4})
                if distinct < 3:
                    violations += 1
        return violations

    # Try 3 colors
    random.seed(123)
    best_colors = None
    best_v = float('inf')
    
    # Multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        colors = [random.randint(0, 2) for _ in range(n)]
        v = count_violations(colors)
        
        for step in range(2000):
            if time.time() > deadline:
                break
            if v < best_v:
                best_v = v
                best_colors = colors[:]
                if v == 0:
                    return colors
            
            pos = random.randint(0, n - 1)
            old_c = colors[pos]
            new_c = (old_c + 1) % 3
            colors[pos] = new_c
            v_new = count_violations(colors)
            
            if v_new > v:
                colors[pos] = old_c
            elif v_new == v:
                if random.random() < 0.5:
                    colors[pos] = old_c
            # else v_new < v, keep
    
    if best_colors is not None and best_v == 0:
        return best_colors

    # Fallback to 4 colors greedy
    colors = [0] * n
    for x in range(1, n + 1):
        for color in range(4):
            valid = True
            for a in range(1, x + 1):
                max_d = (n - a) // 3
                for d in range(1, max_d + 1):
                    # Check if x is in this AP
                    # a + k*d == x for k in 0..3
                    # k = (x-a)/d must be integer in 0..3
                    if (x - a) % d == 0:
                        k = (x - a) // d
                        if 0 <= k <= 3:
                            c_vals = []
                            for j in range(4):
                                idx = a + j * d
                                if idx == x:
                                    c_vals.append(color)
                                else:
                                    c_vals.append(colors[idx - 1])
                            if len(set(c_vals)) < 3:
                                valid = False
                                break
                if not valid:
                    break
            if valid:
                colors[x - 1] = color
                break
    return colors
# EVOLVE-BLOCK-END
