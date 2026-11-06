import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Target n values in descending order
    target_ns = [26, 25, 24, 23, 22, 21, 20]
    
    # Allocate time budgets
    # Total 110s
    # n=26 gets 70s
    # n=25 gets 30s
    # n=24 gets 5s
    # n<24 gets 1s each (should be instant)
    budgets = {
        26: 70.0,
        25: 30.0,
        24: 5.0
    }
    
    last_valid = {'n': 1, 'colors': []} # Trivial valid
    
    for n in target_ns:
        if time.time() > deadline - 2.0:
            break
            
        t_budget = budgets.get(n, 1.0)
        
        # Ensure we don't exceed global deadline
        t_remaining = deadline - time.time()
        if t_budget > t_remaining:
            t_budget = t_remaining - 0.5
            
        if t_budget <= 0:
            continue
            
        result = solve_n(n, t_budget)
        if result is not None:
            return result
        else:
            # If no solution found for this n, we can't return an invalid one.
            # We continue to smaller n.
            pass
            
    # If we fail to find a solution for any n in the list (unlikely for small n),
    # return the trivial n=1 solution or a small valid one.
    # n=1 is always valid.
    return last_valid

def solve_n(n, time_limit):
    num_edges = n * (n - 1) // 2
    if num_edges == 0:
        return {'n': n, 'colors': []}
        
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Initialize random colors
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    # Precompute neighbors for each color
    # neighbors[c][v] is bitmask of neighbors in color c
    neighbors = [[0] * n for _ in range(5)]
    
    # Build initial neighbors
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)
        
    def calc_violations():
        v = 0
        for c in range(5):
            for u in range(n):
                nu = neighbors[c][u]
                for w in range(u + 1, n):
                    common = nu & neighbors[c][w]
                    k = common.bit_count()
                    if k >= 2:
                        v += k * (k - 1) // 2
        return v
        
    current_violations = calc_violations()
    if current_violations == 0:
        return {'n': n, 'colors': colors}
        
    best_violations = current_violations
    best_colors = colors[:]
    
    start_time = time.time()
    temperature = 1.0
    cooling_rate = 0.9995
    
    # Best Choice Simulated Annealing
    while time.time() < start_time + time_limit:
        # Pick random edge
        e_idx = random.randint(0, num_edges - 1)
        i, j = edges[e_idx]
        old_color = colors[e_idx]
        
        best_delta = float('inf')
        best_new_color = -1
        
        # Try all 4 other colors
        for new_color in range(5):
            if new_color == old_color:
                continue
                
            # Apply change tentatively
            neighbors[old_color][i] &= ~(1 << j)
            neighbors[old_color][j] &= ~(1 << i)
            neighbors[new_color][i] |= (1 << j)
            neighbors[new_color][j] |= (1 << i)
            
            # Calculate violations
            v = calc_violations()
            delta = v - current_violations
            
            # Revert tentatively
            neighbors[new_color][i] &= ~(1 << j)
            neighbors[new_color][j] &= ~(1 << i)
            neighbors[old_color][i] |= (1 << j)
            neighbors[old_color][j] |= (1 << i)
            
            if delta < best_delta:
                best_delta = delta
                best_new_color = new_color
                
        # Apply the best choice
        new_color = best_new_color
        neighbors[old_color][i] &= ~(1 << j)
        neighbors[old_color][j] &= ~(1 << i)
        neighbors[new_color][i] |= (1 << j)
        neighbors[new_color][j] |= (1 << i)
        colors[e_idx] = new_color
        current_violations += best_delta
        
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            if best_violations == 0:
                return {'n': n, 'colors': colors}
        
        # Cooling
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
            
    # Check if best found is valid
    if best_violations == 0:
        return {'n': n, 'colors': best_colors}
        
    return None
