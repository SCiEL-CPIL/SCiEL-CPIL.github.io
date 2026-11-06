import random
import time

def construct():
    start_time = time.time()
    DEADLINE = 80  # safety margin before 90s
    
    N = 128
    
    # Precompute all nonconstant 4-term APs: (a, a+d, a+2d, a+3d) with 1-indexed positions
    # Convert to 0-indexed: positions 0..127
    aps = []
    for a in range(N):
        for d in range(1, N):
            if a + 3 * d < N:
                aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    
    def count_violations(colors):
        """Count number of 4-term APs that use fewer than 3 distinct colors."""
        violations = 0
        for (a, b, c, d) in aps:
            ca, cb, cc, cd = colors[a], colors[b], colors[c], colors[d]
            # Count distinct colors
            if ca == cb == cc == cd:
                violations += 1
            elif ca == cb == cc or ca == cb == cd or ca == cc == cd or cb == cc == cd:
                violations += 1
            elif ca == cb and cc == cd and ca != cc:
                violations += 1
            elif ca == cc and cb == cd and ca != cb:
                violations += 1
            elif ca == cd and cb == cc and ca != cb:
                violations += 1
            # Two pairs of same color, different from each other - that's 2 colors, violation
            # Already covered above
        return violations
    
    def count_violations_fast(colors):
        """Faster violation count using bitmasks or direct check."""
        violations = 0
        for (a, b, c, d) in aps:
            ca, cb, cc, cd = colors[a], colors[b], colors[c], colors[d]
            # Check if all 4 use at most 2 colors
            if ca == cb == cc == cd:
                violations += 1
            elif ca == cb == cc and ca != cd:
                violations += 1
            elif ca == cb == cd and ca != cc:
                violations += 1
            elif ca == cc == cd and ca != cb:
                violations += 1
            elif cb == cc == cd and cb != ca:
                violations += 1
            elif ca == cb and cc == cd and ca != cc:
                violations += 1
            elif ca == cc and cb == cd and ca != cb:
                violations += 1
            elif ca == cd and cb == cc and ca != cb:
                violations += 1
        return violations
    
    best_colors = None
    best_num_colors = N  # start with worst case
    
    # Try different numbers of colors, starting from small
    for num_colors in range(2, 13):
        if time.time() - start_time > DEADLINE:
            break
        
        # Try multiple random restarts for each num_colors
        for restart in range(50):
            if time.time() - start_time > DEADLINE:
                break
            
            # Random initial coloring
            colors = [random.randint(0, num_colors - 1) for _ in range(N)]
            violations = count_violations_fast(colors)
            
            if violations == 0:
                best_colors = colors[:]
                best_num_colors = num_colors
                return best_colors
            
            # Local search: try to fix violations
            improved = True
            max_iters = 5000
            for it in range(max_iters):
                if time.time() - start_time > DEADLINE:
                    break
                
                if violations == 0:
                    best_colors = colors[:]
                    best_num_colors = num_colors
                    return best_colors
                
                # Pick a random violating AP and try to fix it
                # Find a random violating AP
                found_violation = False
                for _ in range(100):
                    idx = random.randint(0, num_aps - 1)
                    a, b, c, d = aps[idx]
                    ca, cb, cc, cd = colors[a], colors[b], colors[c], colors[d]
                    # Check if this is a violation
                    is_violation = False
                    if ca == cb == cc == cd:
                        is_violation = True
                    elif ca == cb == cc and ca != cd:
                        is_violation = True
                    elif ca == cb == cd and ca != cc:
                        is_violation = True
                    elif ca == cc == cd and ca != cb:
                        is_violation = True
                    elif cb == cc == cd and cb != ca:
                        is_violation = True
                    elif ca == cb and cc == cd and ca != cc:
                        is_violation = True
                    elif ca == cc and cb == cd and ca != cb:
                        is_violation = True
                    elif ca == cd and cb == cc and ca != cb:
                        is_violation = True
                    
                    if is_violation:
                        found_violation = True
                        break
                
                if not found_violation:
                    # No violation found in random sampling, but violations > 0?
                    # Just break and try restart
                    break
                
                # Try changing one of the 4 positions to a different color
                positions = [a, b, c, d]
                random.shuffle(positions)
                made_change = False
                for pos in positions:
                    old_color = colors[pos]
                    new_color = (old_color + 1) % num_colors
                    if new_color == old_color:
                        continue
                    
                    # Compute change in violations
                    # This is expensive; let's just try it and check
                    colors[pos] = new_color
                    new_violations = count_violations_fast(colors)
                    
                    if new_violations < violations:
                        violations = new_violations
                        made_change = True
                        break
                    else:
                        colors[pos] = old_color
                
                if not made_change:
                    # Try all possible new colors for each position
                    made_change = False
                    for pos in positions:
                        old_color = colors[pos]
                        for new_color in range(num_colors):
                            if new_color == old_color:
                                continue
                            colors[pos] = new_color
                            new_violations = count_violations_fast(colors)
                            if new_violations < violations:
                                violations = new_violations
                                made_change = True
                                break
                            else:
                                colors[pos] = old_color
                        if made_change:
                            break
                    
                    if not made_change:
                        # Stuck, break out
                        break
    
    # If we found a valid coloring, return it
    if best_colors is not None:
        return best_colors
    
    # Fallback: use the greedy approach from the seed
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                    for d in range(1, (x-1)//3 + 1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                        for d in range(1, (x-1)//2 + 1))):
                colors.append(color)
                break
    return colors
