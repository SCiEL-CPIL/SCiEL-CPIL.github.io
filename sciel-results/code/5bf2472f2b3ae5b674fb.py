import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    # Precompute 4-term progressions
    aps = []
    for d in range(1, 43):
        max_start = 128 - 3 * d
        for a in range(max_start + 1):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    
    def get_violations(colors):
        # Returns list of (ap_index, missing_color)
        violations = []
        for idx, p in enumerate(aps):
            c0 = colors[p[0]]
            c1 = colors[p[1]]
            c2 = colors[p[2]]
            c3 = colors[p[3]]
            if c0 != 0 and c1 != 0 and c2 != 0 and c3 != 0:
                violations.append((idx, 0))
            if c0 != 1 and c1 != 1 and c2 != 1 and c3 != 1:
                violations.append((idx, 1))
            if c0 != 2 and c1 != 2 and c2 != 2 and c3 != 2:
                violations.append((idx, 2))
        return violations

    best_colors = None
    best_violations = float('inf')
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        colors = [random.randint(0, 2) for _ in range(128)]
        violations = get_violations(colors)
        current_violations = len(violations)
        
        if current_violations == 0:
            return colors
        
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
        
        # Greedy fix loop
        no_improve = 0
        while time.time() < deadline and no_improve < 200:
            if not violations:
                return colors
                
            # Pick a random violation
            ap_idx, missing_color = random.choice(violations)
            p = aps[ap_idx]
            
            # Pick a random position in this AP to flip to missing_color
            pos = random.choice(p)
            
            old_color = colors[pos]
            if old_color == missing_color:
                # This shouldn't happen if violation is correct, but just in case
                continue
                
            colors[pos] = missing_color
            
            # Recalculate violations
            new_violations = get_violations(colors)
            new_count = len(new_violations)
            
            if new_count < current_violations:
                current_violations = new_count
                violations = new_violations
                no_improve = 0
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
            else:
                # Revert
                colors[pos] = old_color
                no_improve += 1
                
                # Random restart within the loop if stuck
                if no_improve > 50:
                    # Perturb
                    for _ in range(5):
                        j = random.randint(0, 127)
                        colors[j] = random.randint(0, 2)
                    violations = get_violations(colors)
                    current_violations = len(violations)
                    no_improve = 0
                    if current_violations == 0:
                        return colors
                    if current_violations < best_violations:
                        best_violations = current_violations
                        best_colors = colors[:]

    if best_colors is not None and best_violations == 0:
        return best_colors
    
    # Fallback to seed
    def seed_construct():
        colors = []
        for x in range(1, 129):
            for color in range(128):
                if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                    and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                    colors.append(color)
                    break
        return colors

    return seed_construct()
