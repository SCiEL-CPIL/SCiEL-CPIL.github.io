import time
import random
import numpy as np

def construct():
    deadline = time.time() + 85  # safety margin
    
    def count_violations(colors, n=128):
        """Count 4-term APs that use fewer than 3 distinct colors."""
        violations = 0
        c = colors
        for a in range(n):
            for d in range(1, (n - 1 - a) // 3 + 1):
                a1, a2, a3, a4 = a, a+d, a+2*d, a+3*d
                if a4 >= n:
                    break
                colors_used = {c[a1], c[a2], c[a3], c[a4]}
                if len(colors_used) < 3:
                    violations += 1
        return violations
    
    def is_valid(colors, n=128):
        c = colors
        for a in range(n):
            for d in range(1, (n - 1 - a) // 3 + 1):
                a4 = a + 3*d
                if a4 >= n:
                    break
                if len({c[a], c[a+d], c[a+2*d], c[a+3*d]}) < 3:
                    return False
        return True
    
    # Try with 3 colors first
    n = 128
    best_colors = None
    best_num_colors = 999
    
    # First try a greedy approach with 3 colors
    for num_colors in [3, 4, 5]:
        if time.time() > deadline:
            break
        
        # Try multiple random initializations with repair
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            # Random initial assignment
            colors = list(np.random.randint(0, num_colors, n))
            
            # Repair loop: fix violations
            max_iter = 5000
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Find a violating 4-term AP
                found_violation = False
                for a in range(n):
                    for d in range(1, (n - 1 - a) // 3 + 1):
                        a4 = a + 3*d
                        if a4 >= n:
                            break
                        c1, c2, c3, c4 = colors[a], colors[a+d], colors[a+2*d], colors[a+3*d]
                        if len({c1, c2, c3, c4}) < 3:
                            # Try to fix by changing one of the positions
                            # Pick the position that's most "redundant"
                            positions = [a, a+d, a+2*d, a+3*d]
                            pos_colors = [c1, c2, c3, c4]
                            
                            # Try changing each position to a different color
                            best_fix = None
                            for pi, pos in enumerate(positions):
                                for new_c in range(num_colors):
                                    if new_c == pos_colors[pi]:
                                        continue
                                    # Check if this fix helps
                                    test_colors = colors[:]
                                    test_colors[pos] = new_c
                                    # Quick check: does this AP now have >= 3 colors?
                                    tc = [test_colors[p] for p in positions]
                                    if len(set(tc)) >= 3:
                                        # Check if we create new violations nearby
                                        # Simple heuristic: just accept
                                        best_fix = (pos, new_c)
                                        break
                                if best_fix:
                                    break
                            
                            if best_fix:
                                colors[best_fix[0]] = best_fix[1]
                                found_violation = True
                                break
                    if found_violation:
                        break
                
                if not found_violation:
                    break
            
            if is_valid(colors, n):
                used = len(set(colors))
                if used < best_num_colors:
                    best_num_colors = used
                    best_colors = colors[:]
                break  # Found valid, stop trying more attempts for this num_colors
    
    # If we found a valid solution, return it
    if best_colors is not None:
        # Map to 0..127 range (they're already in 0..num_colors-1 which is fine)
        return best_colors
    
    # Fallback: use the greedy seed approach
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                    for d in range(1, (x-1)//3+1) if x-1-3*d >= 0)
                and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                        for d in range(1, (x-1)//2+1) if x-1-2*d >= 0)):
                colors.append(color)
                break
    return colors
