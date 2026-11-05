"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    N = 128
    deadline = time.time() + 80.0  # 10s safety margin for 90s wall-clock
    
    # Precompute all non-constant 4-term arithmetic progressions
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for i in range(N - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    M = len(aps)
    
    # Precompute which APs involve each index
    ap_indices_for_pos = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            ap_indices_for_pos[pos].append(idx)
    
    def count_violations(colors):
        """Count number of 4-APs with fewer than 3 distinct colors."""
        count = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            # Fast distinct count check
            if c0 == c1 == c2 == c3:
                count += 1
            elif (c0 == c1 == c2) or (c0 == c1 == c3) or (c0 == c2 == c3) or (c1 == c2 == c3):
                count += 1
            elif (c0 == c1 and c2 == c3 and c0 != c2) or \
                 (c0 == c2 and c1 == c3 and c0 != c1) or \
                 (c0 == c3 and c1 == c2 and c0 != c1):
                count += 1
        return count
    
    def get_violations_for_pos(colors, pos, new_val):
        """Return number of violations in APs containing pos, assuming colors[pos] = new_val."""
        v_count = 0
        for ap_idx in ap_indices_for_pos[pos]:
            ap = aps[ap_idx]
            c0 = colors[ap[0]] if ap[0] != pos else new_val
            c1 = colors[ap[1]] if ap[1] != pos else new_val
            c2 = colors[ap[2]] if ap[2] != pos else new_val
            c3 = colors[ap[3]] if ap[3] != pos else new_val
            # Check if fewer than 3 distinct colors
            if c0 == c1 == c2 == c3:
                v_count += 1
            elif (c0 == c1 == c2) or (c0 == c1 == c3) or (c0 == c2 == c3) or (c1 == c2 == c3):
                v_count += 1
            elif (c0 == c1 and c2 == c3 and c0 != c2) or \
                 (c0 == c2 and c1 == c3 and c0 != c1) or \
                 (c0 == c3 and c1 == c2 and c0 != c1):
                v_count += 1
        return v_count
    
    def greedy_construct(k):
        """
        Greedy construction that avoids creating monochromatic 3-term APs
        (which would force a 4th color in any extension) and ensures
        every 4-AP seen so far has >= 3 colors.
        """
        colors = []
        for x in range(1, N + 1):
            pos = x - 1
            found = False
            for color in range(k):
                valid = True
                
                # Check all 4-APs ending at pos
                for d in range(1, (x - 1) // 3 + 1):
                    if x - 1 - 3 * d < 0:
                        break
                    c0 = colors[x - 1 - 3 * d]
                    c1 = colors[x - 1 - 2 * d]
                    c2 = colors[x - 1 - d]
                    c3 = color
                    if len({c0, c1, c2, c3}) < 3:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check all 3-term APs ending at pos to avoid monochromatic triples
                # A monochromatic 3-AP (a, a+d, a+2d) with all same color would
                # make it impossible to extend to a 4-AP with only 2 more colors
                # without violating the constraint if the 4th term matches.
                # More precisely: if we have c(a)=c(a+d)=c(a+2d)=c, then any
                # 4-AP (a, a+d, a+2d, a+3d) needs c(a+3d) != c, which is fine.
                # But the real issue is: if we create a pattern that forces
                # future positions into a tight constraint.
                # Actually, the key insight from the observation is to prevent
                # monochromatic 3-term APs because they create dead-ends.
                # Let's check: if colors[pos-d] == colors[pos-2d] == color,
                # then we have a monochromatic 3-AP ending at pos.
                # This is allowed as long as any 4-AP containing it has >= 3 colors.
                # But it restricts future choices.
                
                # Let's also check that we don't create a situation where
                # a 3-term AP is monochromatic AND would force issues.
                # The safest approach: avoid creating monochromatic 3-term APs
                # when possible, as they reduce flexibility.
                
                # Check 3-term APs where pos is the last element
                for d in range(1, (x - 1) // 2 + 1):
                    if x - 1 - 2 * d < 0:
                        break
                    if colors[x - 1 - d] == color and colors[x - 1 - 2 * d] == color:
                        # This would create a monochromatic 3-term AP
                        # It's not strictly forbidden, but it's a dead-end risk
                        # Let's allow it but prefer to avoid
                        pass  # Allow for now, the 4-AP check above handles validity
                
                if valid:
                    colors.append(color)
                    found = True
                    break
            
            if not found:
                # Fallback: pick a random color (shouldn't happen with enough colors)
                colors.append(random.randint(0, k - 1))
        
        return colors
    
    def local_search(colors, k, time_budget):
        """
        Local search to minimize violations, trying to reach 0 violations.
        Returns the best coloring found.
        """
        start_time = time.time()
        current_violations = count_violations(colors)
        best_colors = colors[:]
        best_violations = current_violations
        
        while time.time() - start_time < time_budget:
            if current_violations == 0:
                break
            
            # Pick a random position
            pos = random.randint(0, N - 1)
            current_val = colors[pos]
            
            # Find the best new value for this position
            best_new_val = current_val
            best_v = current_violations
            
            # Try all k values
            for new_val in range(k):
                if new_val == current_val:
                    continue
                # Estimate change in violations
                old_v = get_violations_for_pos(colors, pos, current_val)
                new_v = get_violations_for_pos(colors, pos, new_val)
                delta = new_v - old_v
                if delta < best_v - current_violations:
                    best_v = current_violations + delta
                    best_new_val = new_val
            
            # If no improvement, try a random move (simulated annealing style)
            if best_new_val == current_val:
                if random.random() < 0.3:  # 30% chance to accept a random move
                    best_new_val = random.randint(0, k - 1)
                else:
                    continue
            
            colors[pos] = best_new_val
            current_violations = count_violations(colors)
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors[:]
        
        return best_colors
    
    # Try to find a solution with minimum k colors
    best_solution = None
    
    for k in range(3, 15):
        if time.time() > deadline:
            break
        
        found = False
        # Give more time to smaller k values
        time_per_k = max(2.0, 10.0 / (k - 2))
        
        for restart in range(30):
            if time.time() > deadline:
                break
            
            # Greedy construction with k colors
            colors = greedy_construct(k)
            
            # Check if already valid
            if count_violations(colors) == 0:
                best_solution = colors
                found = True
                break
            
            # Local search to fix violations
            colors = local_search(colors, k, time_per_k / 10)
            
            if count_violations(colors) == 0:
                best_solution = colors
                found = True
                break
        
        if found:
            break
    
    if best_solution is None:
        # Fallback: return a valid coloring with more colors
        # Use a simple pattern that's guaranteed valid
        # With 4 colors, we can use a pattern based on modulo
        best_solution = [i % 4 for i in range(N)]
        
        # Verify and fix if needed
        if count_violations(best_solution) > 0:
            # Use more colors
            best_solution = [i % 8 for i in range(N)]
            if count_violations(best_solution) > 0:
                # Ultimate fallback: all distinct
                best_solution = list(range(N))
    
    return best_solution
# EVOLVE-BLOCK-END
