import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit, 10s buffer
    
    # Try increasing number of colors
    for K in range(3, 20):
        # Try multiple random restarts for this K
        for restart in range(100):
            if time.time() > deadline:
                break
            colors = [-1] * 128
            valid = True
            for x in range(1, 129):
                # Get available colors
                # Check constraints with previous positions
                # A 4-AP is a, a+d, a+2d, a+3d
                # If we assign colors[x-1] = c, we must check all 4-APs containing x
                # that have other positions assigned.
                
                # Optimization: Precompute 4-APs?
                # Or check on the fly.
                
                # For each d, check if x is part of a 4-AP
                # x can be a, a+d, a+2d, a+3d
                # 1. x = a: a, a+d, a+2d, a+3d. Others: x+d, x+2d, x+3d.
                # 2. x = a+d: a, a+d, a+2d, a+3d. Others: x-d, x+d, x+2d.
                # 3. x = a+2d: a, a+d, a+2d, a+3d. Others: x-2d, x-d, x+d.
                # 4. x = a+3d: a, a+d, a+2d, a+3d. Others: x-3d, x-2d, x-d.
                
                # We need to ensure that for any such 4-AP, if all 4 are assigned,
                # the number of distinct colors is >= 3.
                
                # In greedy, we only check if assigning c creates a violation
                # with already assigned neighbors.
                
                # If a 4-AP has 3 assigned and 1 unassigned, we don't check it yet.
                # But if we assign the 4th, we check.
                
                # So for current x, we check all 4-APs where x is the last one assigned?
                # No, we just check all 4-APs containing x where the other 3 are assigned.
                
                possible_colors = []
                for c in range(K):
                    ok = True
                    # Check all d such that x is in a 4-AP
                    # Max d for x=1 is 0. For x=128, d up to 42.
                    # d can be up to (x-1)//3 for x=a.
                    # d can be up to (x-1)//1 for x=a+3d? No, a >= 1.
                    # x = a+3d => a = x-3d >= 1 => d <= (x-1)//3.
                    # x = a => a+d, a+2d, a+3d <= 128 => 3d <= 128-x => d <= (128-x)//3.
                    
                    # So d ranges from 1 to min((x-1)//3, (128-x)//3)
                    # Wait, x can be a, a+d, a+2d, a+3d.
                    # Case 1: x=a. d <= (128-x)//3.
                    # Case 2: x=a+d. a=x-d >= 1 => d <= x-1. a+3d = x+2d <= 128 => 2d <= 128-x => d <= (128-x)//2.
                    # Case 3: x=a+2d. a=x-2d >= 1 => 2d <= x-1 => d <= (x-1)//2. a+3d = x+d <= 128 => d <= 128-x.
                    # Case 4: x=a+3d. a=x-3d >= 1 => 3d <= x-1 => d <= (x-1)//3.
                    
                    # Combined: d must be such that the 4 terms are in [1, 128].
                    # Max d is roughly 42.
                    
                    for d in range(1, 43):
                        # Check 4 configurations
                        # 1. x, x+d, x+2d, x+3d
                        if x + 3*d <= 128:
                            idxs = [x-1, x+d-1, x+2*d-1, x+3*d-1]
                            # Check if other 3 are assigned
                            assigned_vals = [colors[i] for i in idxs if i != x-1]
                            if all(v != -1 for v in assigned_vals):
                                # Check if {c} U assigned_vals has size >= 3
                                if len(set([c] + assigned_vals)) < 3:
                                    ok = False
                                    break
                        # 2. x-2d, x-d, x, x+d
                        if x - 2*d >= 1 and x + d <= 128:
                            idxs = [x-2*d-1, x-d-1, x-1, x+d-1]
                            assigned_vals = [colors[i] for i in idxs if i != x-1]
                            if all(v != -1 for v in assigned_vals):
                                if len(set([c] + assigned_vals)) < 3:
                                    ok = False
                                    break
                        # 3. x-d, x, x+d, x+2d
                        if x - d >= 1 and x + 2*d <= 128:
                            idxs = [x-d-1, x-1, x+d-1, x+2*d-1]
                            assigned_vals = [colors[i] for i in idxs if i != x-1]
                            if all(v != -1 for v in assigned_vals):
                                if len(set([c] + assigned_vals)) < 3:
                                    ok = False
                                    break
                        # 4. x-3d, x-2d, x-d, x
                        if x - 3*d >= 1:
                            idxs = [x-3*d-1, x-2*d-1, x-d-1, x-1]
                            assigned_vals = [colors[i] for i in idxs if i != x-1]
                            if all(v != -1 for v in assigned_vals):
                                if len(set([c] + assigned_vals)) < 3:
                                    ok = False
                                    break
                        if not ok:
                            break
                    if ok:
                        possible_colors.append(c)
                
                if not possible_colors:
                    valid = False
                    break
                # Pick a random color from possible
                colors[x-1] = random.choice(possible_colors)
            
            if valid:
                # Verify full solution
                # Check all 4-APs
                is_valid = True
                for a in range(1, 129):
                    for d in range(1, (128-a)//3 + 1):
                        idxs = [a-1, a+d-1, a+2*d-1, a+3*d-1]
                        vals = [colors[i] for i in idxs]
                        if len(set(vals)) < 3:
                            is_valid = False
                            break
                    if not is_valid:
                        break
                if is_valid:
                    return colors
        
        if time.time() > deadline:
            break
    
    # Fallback: Return seed-like solution or best found
    # If we didn't find a valid one, we must return something.
    # The problem requires a valid coloring.
    # If we fail, we might return an invalid one? "invalid result ... scores 0".
    # So we should try to return a valid one.
    # The greedy without randomization (seed) is valid?
    # The seed checks all constraints. It is valid.
    # So I should fall back to the seed logic if random fails.
    
    colors=[]
    for x in range(1,129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors
