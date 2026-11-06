import random
import time

def construct():
    deadline = time.time() + 85  # 85 second safety margin (limit is 90)
    
    # Try with increasing number of colors
    for num_colors in [3, 4, 5, 6]:
        palette = list(range(num_colors))
        result = _try_construct(palette, deadline)
        if result is not None:
            return result
    
    # Fallback: return a valid coloring with many colors (e.g., identity)
    # This ensures we return *something* valid, even if not optimal
    # Identity coloring: color[i] = i. 
    # Does identity satisfy "at least 3 colors in every 4-AP"?
    # AP: a, a+d, a+2d, a+3d. Colors: a, a+d, a+2d, a+3d.
    # These are distinct integers. So yes, 4 distinct colors.
    # So identity is always valid.
    return list(range(128))

def _try_construct(palette, deadline):
    """Try to construct a valid coloring using the given palette using Local Search."""
    n = 128
    num_colors = len(palette)
    
    # Precompute all 4-APs
    aps = []
    for d in range(1, n//3 + 1):
        for a in range(1, n - 3*d + 1):
            # 0-indexed indices
            idx0 = a - 1
            idx1 = a - 1 + d
            idx2 = a - 1 + 2*d
            idx3 = a - 1 + 3*d
            aps.append((idx0, idx1, idx2, idx3))
            
    def count_violations(colors):
        """Count the number of 4-APs that use fewer than 3 distinct colors."""
        v = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            distinct = len({c0, c1, c2, c3})
            if distinct < 3:
                v += 1
        return v

    def delta_violation(pos, old_color, new_color):
        """Calculate change in violations if colors[pos] changes from old_color to new_color.
        Only considers APs containing pos."""
        delta = 0
        for ap in aps:
            if pos not in ap:
                continue
            # Get current colors for this AP
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            
            # Check if this AP is currently a violation
            was_violation = len({c0, c1, c2, c3}) < 3
            
            # Apply change
            if ap[0] == pos: c0 = new_color
            elif ap[1] == pos: c1 = new_color
            elif ap[2] == pos: c2 = new_color
            elif ap[3] == pos: c3 = new_color
            
            # Check if it is a violation now
            is_violation = len({c0, c1, c2, c3}) < 3
            
            if was_violation and not is_violation:
                delta -= 1
            elif not was_violation and is_violation:
                delta += 1
        
        return delta

    # Local Search
    best_colors = None
    best_violations = float('inf')
    
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        # Initialize randomly
        colors = [random.choice(palette) for _ in range(n)]
        current_violations = count_violations(colors)
        
        if current_violations == 0:
            return colors[:]
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            
        no_improvement_steps = 0
        max_no_improvement = 500  # Escape local minima
        
        while time.time() < deadline:
            if current_violations == 0:
                return colors[:]
                
            if no_improvement_steps > max_no_improvement:
                # Perturb: change a few random positions
                for _ in range(5):
                    pos = random.randint(0, n-1)
                    old_c = colors[pos]
                    new_c = random.choice(palette)
                    if new_c != old_c:
                        d = delta_violation(pos, old_c, new_c)
                        colors[pos] = new_c
                        current_violations += d
                no_improvement_steps = 0
                continue
            
            # Pick a random position
            pos = random.randint(0, n-1)
            old_color = colors[pos]
            
            best_new_color = old_color
            best_delta = 0
            
            for c in palette:
                if c == old_color:
                    continue
                d = delta_violation(pos, old_color, c)
                if d < best_delta:
                    best_delta = d
                    best_new_color = c
                elif d == best_delta and random.random() < 0.1:
                    # Random tie-breaking to avoid cycles
                    best_new_color = c
                    best_delta = d
                    
            if best_new_color != old_color:
                colors[pos] = best_new_color
                current_violations += best_delta
                
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                    no_improvement_steps = 0
                else:
                    no_improvement_steps += 1
            else:
                no_improvement_steps += 1
                
    if best_violations == 0:
        return best_colors
    return None

def _verify(colors):
    """Verify that the coloring satisfies all constraints."""
    n = len(colors)
    for d in range(1, n//3 + 1):
        for a in range(1, n - 3*d + 1):
            idx0 = a - 1
            idx1 = a - 1 + d
            idx2 = a - 1 + 2*d
            idx3 = a - 1 + 3*d
            distinct = len({colors[idx0], colors[idx1], colors[idx2], colors[idx3]})
            if distinct < 3:
                return False
    return True
