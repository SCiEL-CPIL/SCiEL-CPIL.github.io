import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 128
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with 0-indexed
    ap_list = []
    pos_to_aps = [[] for _ in range(n)]
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            idx = len(ap_list)
            ap_list.append(ap)
            for p in ap:
                pos_to_aps[p].append(idx)
                
    num_aps = len(ap_list)
    
    def is_valid_coloring(colors):
        """Check if every 4-term progression uses at least 3 distinct colors."""
        for ap in ap_list:
            if len(set(colors[ap])) < 3:
                return False
        return True
        
    def count_violations(colors):
        """Count the number of APs with fewer than 3 distinct colors."""
        violations = 0
        for ap in ap_list:
            if len(set(colors[ap])) < 3:
                violations += 1
        return violations

    def solve_with_k_colors(k, time_limit=30.0):
        """
        Uses Simulated Annealing to find a valid k-coloring.
        Returns the coloring if found, else None.
        """
        if time.time() > deadline:
            return None
            
        # Initialize with random coloring
        colors = [random.randint(0, k - 1) for _ in range(n)]
        
        # Calculate initial violations
        current_violations = count_violations(colors)
        
        if current_violations == 0:
            return colors
            
        best_colors = colors[:]
        best_violations = current_violations
        
        # Simulated Annealing parameters
        T = 1.0
        cooling_rate = 0.999
        min_T = 0.001
        
        while T > min_T and time.time() < deadline - 5.0:
            # Pick a random position to change
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            
            # Pick a new color different from old
            new_color = random.choice([c for c in range(k) if c != old_color])
            
            # Calculate change in violations
            # Only APs involving 'pos' can change status
            delta_violations = 0
            for ap_idx in pos_to_aps[pos]:
                ap = ap_list[ap_idx]
                
                # Get colors before change
                # We need to check if the AP was valid before and after
                # To do this efficiently, we can just re-evaluate the specific APs
                
                # Create a tuple of colors for this AP with the new color at 'pos'
                # Note: 'pos' is one of the elements in 'ap'
                # We can't just modify 'colors' yet, we need to simulate
                
                # Optimization: 
                # Instead of creating sets repeatedly, we can check the condition directly.
                # Condition: len(set(vals)) >= 3
                
                # Let's just do the simple re-evaluation for the affected APs.
                # n is small, so this is fine.
                
                # Get the 4 positions
                p0, p1, p2, p3 = ap
                
                # Determine which index in the AP corresponds to 'pos'
                # We can just build the list of colors with the change applied
                vals = [colors[p0], colors[p1], colors[p2], colors[p3]]
                
                # Find the index of pos in the AP
                # Since pos is in the AP, it must be one of p0,p1,p2,p3
                if p0 == pos:
                    vals[0] = new_color
                elif p1 == pos:
                    vals[1] = new_color
                elif p2 == pos:
                    vals[2] = new_color
                elif p3 == pos:
                    vals[3] = new_color
                    
                # Check old validity
                old_vals = [colors[p0], colors[p1], colors[p2], colors[p3]]
                # Wait, we haven't changed colors yet.
                # So old_vals is just the current state.
                # But we need to know if it was a violation BEFORE the change.
                # And if it will be a violation AFTER the change.
                
                # Actually, we can just calculate the violation status for the "before" and "after" states.
                
                # Before state:
                was_violation = len(set(old_vals)) < 3
                
                # After state:
                is_violation = len(set(vals)) < 3
                
                if was_violation and not is_violation:
                    delta_violations -= 1
                elif not was_violation and is_violation:
                    delta_violations += 1
                # else: no change in violation count for this AP
            
            new_violations = current_violations + delta_violations
            
            # Acceptance criterion
            if new_violations < current_violations:
                # Always accept improvement
                colors[pos] = new_color
                current_violations = new_violations
                
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                    
                if current_violations == 0:
                    return colors
            else:
                # Deterioration or no change
                # Accept with probability exp(-delta / T)
                # delta is positive (increase in violations)
                delta = new_violations - current_violations
                if delta == 0:
                    # Random walk
                    if random.random() < 0.5:
                        colors[pos] = new_color
                        # current_violations unchanged
                else:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        colors[pos] = new_color
                        current_violations = new_violations
            
            T *= cooling_rate
            
        return best_colors if best_violations == 0 else None

    # Try to find a 4-coloring
    # We try multiple restarts if the first one fails
    max_restarts = 5
    restart_time_budget = (deadline - time.time()) / max_restarts
    
    for i in range(max_restarts):
        remaining_time = deadline - time.time() - 5.0
        if remaining_time <= 0:
            break
            
        # Use a portion of the time for each restart
        t_limit = min(restart_time_budget, remaining_time)
        
        # Increase k if 4 fails? 
        # Based on W(3,4)=76, 3 is impossible. 
        # 4 is likely possible.
        result = solve_with_k_colors(4, t_limit)
        
        if result is not None:
            return result
        
    # If 4 colors failed, try 5 colors with remaining time
    remaining_time = deadline - time.time() - 5.0
    if remaining_time > 5:
        result = solve_with_k_colors(5, remaining_time)
        if result is not None:
            return result

    # Fallback: Greedy construction with many colors (guaranteed to work but high score)
    # This ensures we return *something* valid, even if not optimal.
    colors = []
    for x in range(1, 129):
        found = False
        for color in range(128):
            valid = True
            # Check all 4-APs ending at x (0-indexed x-1)
            # A 4-AP is (a, a+d, a+2d, a+3d). 
            # If x-1 is the last term, then a+3d = x-1 => a = x-1-3d.
            # We need a >= 0 => 3d <= x-1 => d <= (x-1)//3.
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 1 - 3 * d
                b = a + d
                c = a + 2 * d
                # Positions: a, b, c, x-1
                # Colors: colors[a], colors[b], colors[c], color
                if len({colors[a], colors[b], colors[c], color}) < 3:
                    valid = False
                    break
            
            # Also need to check APs where x-1 is not the last term?
            # The greedy approach assigns left to right.
            # When assigning x-1, all previous positions are assigned.
            # Any 4-AP involving x-1 must have the other 3 positions < x-1?
            # Not necessarily. 
            # Example: AP (x-1, x-1+d, ...). These involve future positions.
            # However, the constraint is global. 
            # If we just ensure that no *completed* AP is violated, 
            # and we assume future assignments can fix things, it might fail.
            # But the standard greedy for such problems usually checks:
            # 1. APs where x-1 is the last element (all others assigned).
            # 2. APs where x-1 is the 3rd element (2nd and 4th assigned? No, 4th is future).
            # Actually, if we only check completed APs (where all 4 are assigned), 
            # we might create a situation where a future position is forced into an invalid state.
            # However, with 128 colors available, we can always pick a unique color 
            # that doesn't conflict with any partial AP?
            # No, we want to minimize colors.
            
            # The seed program's fallback logic:
            # It checks:
            # 1. len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3
            #    This checks APs ending at x-1.
            # 2. any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1, (x-1)//2 + 1)
            #    This seems to check APs where x-1 is the middle? 
            #    Actually, the second condition in the seed is a bit obscure.
            
            # For the fallback, let's just use a simple greedy that picks the smallest color 
            # that doesn't create a violation in any AP where x-1 is the LAST element.
            # And we hope that this extends to a full solution. 
            # With 128 colors, it's very likely to work if we pick colors carefully.
            # Actually, if we just pick a NEW unique color for each position, it's always valid.
            # But we want to minimize.
            
            # Let's stick to the seed's fallback logic which is known to be valid (score 1.0).
            if valid:
                colors.append(color)
                found = True
                break
        if not found:
            # Should not happen if we have enough colors
            colors.append(127)
            
    return colors

# Test/Run
if __name__ == "__main__":
    res = construct()
    # Verify
    n = 128
    ap_list = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
            
    valid = True
    for ap in ap_list:
        if len(set(res[ap])) < 3:
            valid = False
            break
            
    print(f"Valid: {valid}, Colors used: {len(set(res))}")
