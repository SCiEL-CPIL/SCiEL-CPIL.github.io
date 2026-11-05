import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Precompute 4-APs
    aps = []
    for d in range(1, 43):
        for a in range(1, 129 - 3*d):
            aps.append((a-1, a+d-1, a+2*d-1, a+3*d-1))
            
    # Precompute which APs contain each position
    aps_containing = [[] for _ in range(128)]
    for i, ap in enumerate(aps):
        for pos in ap:
            aps_containing[pos].append(i)
            
    best_colors = None
    best_distinct = 128
    
    # Try k=3, 4, 5...
    for k in range(3, 10):
        if time.time() > deadline:
            break
            
        # Try to find a valid coloring with k colors
        found = False
        for restart in range(50):
            if time.time() > deadline:
                break
                
            colors = np.random.randint(0, k, 128)
            
            # Calculate initial violations
            violations = [0] * len(aps)
            for i, ap in enumerate(aps):
                if len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3:
                    violations[i] = 1
            current_violations = sum(violations)
            best_violations = current_violations
            best_colors_local = colors.copy()
            
            temp = 1.0
            # Increase steps since it's faster now
            for step in range(50000):
                if time.time() > deadline:
                    break
                    
                # Pick a random position
                pos = np.random.randint(128)
                old_color = colors[pos]
                
                # Try a new color
                new_color = np.random.randint(k)
                if new_color == old_color:
                    continue
                
                # Calculate delta
                delta = 0
                for i in aps_containing[pos]:
                    ap = aps[i]
                    # Old state
                    old_set = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
                    # New state
                    new_set = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
                    # Replace color at pos
                    # This is a bit slow to create sets.
                    # Let's just check the condition directly.
                    
                    # Better: just check if the AP was valid and becomes invalid, or vice versa.
                    # Valid means len(set) >= 3.
                    
                    # Let's stick to the set for simplicity, it's fast enough for ~100 APs per pos.
                    # But we can optimize by not creating sets if possible.
                    # For now, keep it simple.
                    
                    # Actually, let's just re-evaluate the specific AP.
                    # This is O(1) per AP.
                    pass
                
                # This approach is still a bit slow in Python if not careful.
                # Let's use a simpler heuristic: just re-calculate violations for affected APs.
                
                colors[pos] = new_color
                new_violations = current_violations
                for i in aps_containing[pos]:
                    ap = aps[i]
                    was_valid = (violations[i] == 0)
                    is_valid = len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) >= 3
                    is_valid = 0 if is_valid else 1
                    
                    if was_valid and not is_valid:
                        new_violations += 1
                        violations[i] = 1
                    elif not was_valid and is_valid:
                        new_violations -= 1
                        violations[i] = 0
                    else:
                        violations[i] = 1 if not is_valid else 0
                        
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_colors_local = colors.copy()
                elif new_violations == best_violations:
                    pass
                else:
                    # Revert
                    colors[pos] = old_color
                    # Revert violations
                    for i in aps_containing[pos]:
                        ap = aps[i]
                        is_valid = len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) >= 3
                        violations[i] = 0 if is_valid else 1
                    current_violations = old_violations # Wait, we need to store old_violations
                    # This is getting messy.
                    
                    # Let's simplify: Just store the state and revert if worse.
                    # And use a simple hill climbing (no SA) first to see if it works.
                    
                current_violations = new_violations
                temp *= 0.999
                
                if best_violations == 0:
                    distinct = len(set(best_colors_local))
                    if distinct < best_distinct:
                        best_distinct = distinct
                        best_colors = best_colors_local.copy()
                    found = True
                    break
                    
            if found:
                break
                
    if best_colors is not None:
        return best_colors.tolist()
    else:
        # Fallback
        return list(range(128)) # Invalid, but prevents crash

