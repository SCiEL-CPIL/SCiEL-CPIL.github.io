import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe
    
    n = 128
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) 0-indexed
    ap_list = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # pos_to_aps: for each position, list of AP indices it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    # Greedy solution first (uses smallest valid color)
    greedy_colors = []
    for x in range(1, n + 1):
        idx = x - 1
        best_color = -1
        for color in range(128):
            valid = True
            # Check 4-term APs ending at idx
            for d in range(1, (idx // 3) + 1):
                a = idx - 3 * d
                c1 = greedy_colors[a]
                c2 = greedy_colors[a + d]
                c3 = greedy_colors[a + 2 * d]
                if len({color, c1, c2, c3}) < 3:
                    valid = False
                    break
            if not valid:
                continue
            # Check 3-term APs ending at idx (necessary condition)
            for d in range(1, (idx // 2) + 1):
                a = idx - 2 * d
                c1 = greedy_colors[a]
                c2 = greedy_colors[a + d]
                if color == c1 and c1 == c2:
                    valid = False
                    break
            if valid:
                best_color = color
                break
        if best_color == -1:
            best_color = 0
        greedy_colors.append(best_color)
        
    num_greedy_colors = len(set(greedy_colors))
    
    # If greedy uses 3 colors, it's likely optimal (min is 3).
    if num_greedy_colors <= 3:
        return greedy_colors
    
    # Local search approach: start with random 3-coloring, then fix violations
    import random
    
    def count_violations(colors):
        violations = 0
        for ap in ap_list:
            a, b, c, d_pos = ap
            vals = {colors[a], colors[b], colors[c], colors[d_pos]}
            if len(vals) < 3:
                violations += 1
        return violations
    
    def local_search(max_iters=50000):
        best_colors = None
        best_violations = float('inf')
        
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Random initial coloring with 3 colors
            colors = [random.randint(0, 2) for _ in range(n)]
            violations = count_violations(colors)
            
            if violations == 0:
                return colors
            
            if violations < best_violations:
                best_violations = violations
                best_colors = colors[:]
            
            # Simulated annealing / hill climbing
            current_violations = violations
            for i in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Pick a random position to change
                pos = random.randint(0, n - 1)
                old_color = colors[pos]
                new_color = random.randint(0, 2)
                
                if new_color == old_color:
                    continue
                
                # Calculate change in violations
                delta = 0
                for ap_idx in pos_to_aps[pos]:
                    a, b, c, d_pos = ap_list[ap_idx]
                    # Before change
                    old_vals = {colors[a], colors[b], colors[c], colors[d_pos]}
                    old_viol = 1 if len(old_vals) < 3 else 0
                    
                    # After change
                    new_colors = colors[:]
                    new_colors[pos] = new_color
                    new_vals = {new_colors[a], new_colors[b], new_colors[c], new_colors[d_pos]}
                    new_viol = 1 if len(new_vals) < 3 else 0
                    
                    delta += new_viol - old_viol
                
                colors[pos] = new_color
                new_violations = current_violations + delta
                
                # Accept if better or with some probability
                if new_violations < current_violations:
                    current_violations = new_violations
                    if new_violations == 0:
                        return colors
                    if new_violations < best_violations:
                        best_violations = new_violations
                        best_colors = colors[:]
                elif new_violations == current_violations:
                    # Accept with some probability to escape local minima
                    if random.random() < 0.3:
                        current_violations = new_violations
                else:
                    # Reject, undo
                    colors[pos] = old_color
            
            # If we found a perfect solution, return it
            if current_violations == 0:
                return colors
    
    result = local_search()
    if result is not None:
        return result
    
    # If local search failed, try backtracking with constraint propagation
    colors = [-1] * n
    
    def is_valid_assignment(pos, color):
        # Check constraints involving 'pos'
        for ap_idx in pos_to_aps[pos]:
            a, b, c, d_pos = ap_list[ap_idx]
            vals = [colors[a], colors[b], colors[c], colors[d_pos]]
            assigned_vals = [v for v in vals if v != -1]
            if len(assigned_vals) == 4:
                if len(set(assigned_vals)) < 3:
                    return False
            elif len(assigned_vals) == 3:
                if len(set(assigned_vals)) == 1:
                    return False
        return True

    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
            
        # Try colors 0, 1, 2
        for color in range(3):
            colors[idx] = color
            if is_valid_assignment(idx, color):
                if backtrack(idx + 1):
                    return True
            colors[idx] = -1
            
        return False

    # Try to solve with 3 colors
    if backtrack(0):
        return colors
    
    # If 3 colors failed, return greedy
    return greedy_colors
