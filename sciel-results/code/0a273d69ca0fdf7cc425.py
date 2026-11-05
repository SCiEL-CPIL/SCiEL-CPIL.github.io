import time
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit to be safe
    
    n = 128
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) 0-indexed
    ap_list = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
            
    # Precompute all 3-term APs for monochromatic avoidance
    ap3_list = []
    for d in range(1, n // 2 + 1):
        for a in range(0, n - 2 * d):
            ap3_list.append((a, a + d, a + 2 * d))
    
    # pos_to_aps: for each position, list of 4-term AP indices it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    # pos_to_aps3: for each position, list of 3-term AP indices it belongs to
    pos_to_aps3 = [[] for _ in range(n)]
    for idx, ap in enumerate(ap3_list):
        for pos in ap:
            pos_to_aps3[pos].append(idx)
    
    # Count how many APs each position is in (for variable ordering)
    ap_count_per_pos = [len(pos_to_aps[i]) for i in range(n)]
    
    def get_conflicting_colors(pos, color, colors):
        """Check if assigning 'color' to 'pos' violates any 4-term AP constraint."""
        for ap_idx in pos_to_aps[pos]:
            ap = ap_list[ap_idx]
            a, b, c, d = ap
            vals = [colors[a], colors[b], colors[c], colors[d]]
            # Set pos to color temporarily
            for i, p in enumerate(ap):
                if p == pos:
                    vals[i] = color
                    break
            assigned = [v for v in vals if v is not None]
            
            if len(assigned) == 4:
                if len(set(assigned)) < 3:
                    return True  # Conflict
            elif len(assigned) == 3:
                if len(set(assigned)) == 1:
                    return True  # Conflict
        return False
    
    def get_conflicting_3ap(pos, color, colors):
        """Check if assigning 'color' to 'pos' creates a monochromatic 3-term AP."""
        for ap_idx in pos_to_aps3[pos]:
            ap = ap3_list[ap_idx]
            a, b, c = ap
            vals = [colors[a], colors[b], colors[c]]
            # Set pos to color temporarily
            for i, p in enumerate(ap):
                if p == pos:
                    vals[i] = color
                    break
            assigned = [v for v in vals if v is not None]
            if len(assigned) == 3:
                if len(set(assigned)) == 1:
                    return True  # Monochromatic 3-term AP
        return False
    
    # Greedy solution with forward checking + 3-term AP avoidance
    greedy_colors = [None] * n
    used_colors = set()
    
    for idx in range(n):
        best_color = None
        best_score = -1
        
        for color in range(128):
            # Check if assigning color to idx violates any 4-term AP constraint
            if get_conflicting_colors(idx, color, greedy_colors):
                continue
            # Check if assigning color creates monochromatic 3-term AP
            if get_conflicting_3ap(idx, color, greedy_colors):
                continue
            
            # Score: prefer colors that keep more options open
            score = 0
            for ap_idx in pos_to_aps[idx]:
                ap = ap_list[ap_idx]
                a, b, c, d = ap
                vals = [greedy_colors[a], greedy_colors[b], greedy_colors[c], greedy_colors[d]]
                for i, p in enumerate(ap):
                    if p == idx:
                        vals[i] = color
                        break
                assigned = [v for v in vals if v is not None]
                if len(assigned) == 3 and len(set(assigned)) == 1:
                    score -= 100  # Very bad
                elif len(assigned) == 4 and len(set(assigned)) < 3:
                    score -= 100  # Very bad
                elif len(assigned) == 3:
                    score += len(set(assigned))
            
            # Prefer already-used colors to minimize total colors
            if color in used_colors:
                score += 50
            
            if score > best_score:
                best_score = score
                best_color = color
        
        if best_color is None:
            # Fallback: relax 3-term AP constraint
            for color in range(128):
                if get_conflicting_colors(idx, color, greedy_colors):
                    continue
                greedy_colors[idx] = color
                if not get_conflicting_3ap(idx, color, greedy_colors):
                    best_color = color
                    break
                greedy_colors[idx] = None
            if best_color is None:
                # Last resort: any non-conflicting color
                for color in range(128):
                    greedy_colors[idx] = color
                    if not get_conflicting_colors(idx, color, greedy_colors):
                        best_color = color
                        break
                    greedy_colors[idx] = None
            if best_color is None:
                best_color = 0
        
        greedy_colors[idx] = best_color
        used_colors.add(best_color)
    
    num_greedy_colors = len(used_colors)
    
    # Validate the greedy solution
    def count_violations(colors_arr):
        violations = 0
        for ap in ap_list:
            a, b, c, d = ap
            vals = [colors_arr[a], colors_arr[b], colors_arr[c], colors_arr[d]]
            if len(set(vals)) < 3:
                violations += 1
        return violations
    
    greedy_ints = [c if c is not None else 0 for c in greedy_colors]
    if count_violations(greedy_ints) == 0:
        # If greedy uses 3 colors, it's likely optimal
        if num_greedy_colors <= 3:
            return greedy_ints
    else:
        # Greedy failed, use 0 as fallback for now
        greedy_ints = [0] * n
        num_greedy_colors = 1
    
    # Try backtracking for K=3 with proper constraint propagation
    colors = [None] * n
    
    # Variable ordering: most constrained first (highest degree)
    var_order = sorted(range(n), key=lambda i: -ap_count_per_pos[i])
    
    # Track which variables are assigned
    assigned = [False] * n
    assigned_count = 0
    
    def is_feasible(pos, color):
        """Check if assigning color to pos is feasible."""
        if assigned[pos]:
            return colors[pos] == color
        return not get_conflicting_colors(pos, color, colors)
    
    def propagate(pos, color):
        """Assign pos=color and propagate constraints. Returns False if infeasible."""
        nonlocal assigned_count
        
        # Check if assignment is valid
        if not is_feasible(pos, color):
            return False
        
        # Assign
        colors[pos] = color
        assigned[pos] = True
        assigned_count += 1
        
        # Propagate to neighbors
        for ap_idx in pos_to_aps[pos]:
            ap = ap_list[ap_idx]
            a, b, c, d = ap
            vals = [colors[a], colors[b], colors[c], colors[d]]
            assigned_count_in_ap = sum(1 for v in vals if v is not None)
            
            if assigned_count_in_ap == 4:
                # Check if AP is satisfied
                if len(set(vals)) < 3:
                    # Conflict: need to unassign
                    colors[pos] = None
                    assigned[pos] = False
                    assigned_count -= 1
                    return False
            elif assigned_count_in_ap == 3:
                # Check if all 3 assigned are same color
                assigned_vals = [v for v in vals if v is not None]
                if len(set(assigned_vals)) == 1:
                    # The unassigned position must be different, but even then only 2 colors
                    # This is a conflict
                    colors[pos] = None
                    assigned[pos] = False
                    assigned_count -= 1
                    return False
        
        return True
    
    def unpropagate(pos):
        """Unassign pos."""
        nonlocal assigned_count
        colors[pos] = None
        assigned[pos] = False
        assigned_count -= 1
    
    def backtrack(depth):
        if time.time() > deadline:
            return False
        
        if assigned_count == n:
            return True
        
        # Find next unassigned variable using MRV (most constrained)
        best_pos = -1
        best_domain_size = 1000
        
        for pos in var_order:
            if assigned[pos]:
                continue
            # Count valid colors
            valid_count = 0
            for color in range(3):
                if is_feasible(pos, color):
                    valid_count += 1
                    if valid_count < best_domain_size:
                        best_domain_size = valid_count
                        best_pos = pos
            if valid_count == 0:
                return False  # Dead end
            if best_domain_size == 1:
                break
        
        if best_pos == -1:
            return False
        
        # Try colors
        for color in range(3):
            if not is_feasible(best_pos, color):
                continue
            
            if propagate(best_pos, color):
                if backtrack(depth + 1):
                    return True
            unpropagate(best_pos)
        
        return False
    
    # Try to solve with 3 colors
    if backtrack(0):
        return colors
    
    # If 3 colors failed, try local search to improve greedy
    best_solution = greedy_ints[:]
    best_color_count = num_greedy_colors
    
    # Local search: try to reduce number of colors
    for iteration in range(200):
        if time.time() > deadline - 5:
            break
        
        improved = False
        for i in range(n):
            for new_color in range(best_color_count):
                if new_color == best_solution[i]:
                    continue
                
                # Try this change
                old = best_solution[i]
                best_solution[i] = new_color
                
                # Check if still valid
                if count_violations(best_solution) == 0:
                    # Check if we reduced color count
                    used = set(best_solution)
                    if len(used) < best_color_count:
                        best_color_count = len(used)
                        improved = True
                        break
                    # Keep the change even if same count (might help later)
                    break
                else:
                    best_solution[i] = old
            
            if improved:
                break
        
        if not improved:
            break
    
    # Final check
    if count_violations(best_solution) == 0:
        return best_solution
    
    # Return original greedy
    return greedy_ints
