import time
import random
import sys
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Helper: count violations using bitsets
    def count_violations(n, colors):
        neighbors = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
        
        cost = 0
        for c in range(5):
            for u in range(n):
                nu = neighbors[c][u]
                if nu == 0:
                    continue
                for v in range(u + 1, n):
                    nv = neighbors[c][v]
                    common = nu & nv
                    if common & (common - 1):
                        cost += 1
        return cost

    def get_seed_solution():
        n = 11
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = min(j - i, n - (j - i))
                colors.append(d - 1)
        return n, colors

    best_n, best_colors = get_seed_solution()

    def solve_sa(n, time_limit, initial_colors=None):
        """
        Simulated Annealing with incremental cost updates.
        If initial_colors is provided, use it as the starting point.
        """
        total_edges = n * (n - 1) // 2
        
        if initial_colors is not None:
            colors = initial_colors[:]
        else:
            colors = [random.randint(0, 4) for _ in range(total_edges)]
        
        color_matrix = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                color_matrix[i][j] = c
                color_matrix[j][i] = c
                idx += 1
        
        neighbors = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
        
        def get_cost():
            cost = 0
            for c in range(5):
                for u in range(n):
                    nu = neighbors[c][u]
                    if nu == 0:
                        continue
                    for v in range(u + 1, n):
                        nv = neighbors[c][v]
                        common = nu & nv
                        if common & (common - 1):
                            cost += 1
            return cost
        
        current_cost = get_cost()
        if current_cost == 0:
            return colors
        
        search_start = time.time()
        stagnation_counter = 0
        best_cost_so_far = current_cost
        best_colors_snapshot = colors[:]
        
        # SA parameters
        T = 10.0
        cooling_rate = 0.995
        min_T = 0.01
        kick_threshold = 50
        
        while time.time() - search_start < time_limit:
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            c_old = color_matrix[i][j]
            
            best_new_color = -1
            best_delta = 0
            
            for c_new in range(5):
                if c_new == c_old:
                    continue
                
                delta = 0
                
                # c_old removals
                for w in range(n):
                    if w == i or w == j:
                        continue
                    if neighbors[c_old][j] & (1 << w):
                        k = (neighbors[c_old][i] & neighbors[c_old][w]).bit_count()
                        delta -= (k - 1)
                    if neighbors[c_old][i] & (1 << w):
                        k = (neighbors[c_old][j] & neighbors[c_old][w]).bit_count()
                        delta -= (k - 1)
                
                # c_new additions
                for w in range(n):
                    if w == i or w == j:
                        continue
                    if neighbors[c_new][j] & (1 << w):
                        k = (neighbors[c_new][i] & neighbors[c_new][w]).bit_count()
                        delta += k
                    if neighbors[c_new][i] & (1 << w):
                        k = (neighbors[c_new][j] & neighbors[c_new][w]).bit_count()
                        delta += k
                        
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = c_new
            
            # Acceptance criterion (Metropolis)
            if best_new_color != -1:
                if best_delta < 0:
                    accept = True
                else:
                    # Accept worse move with probability exp(-delta / T)
                    if T > 0:
                        accept = random.random() < (2.718281828 ** (-best_delta / T))
                    else:
                        accept = False
                
                if accept:
                    # Apply move
                    neighbors[c_old][i] &= ~(1 << j)
                    neighbors[c_old][j] &= ~(1 << i)
                    neighbors[best_new_color][i] |= (1 << j)
                    neighbors[best_new_color][j] |= (1 << i)
                    
                    color_matrix[i][j] = best_new_color
                    color_matrix[j][i] = best_new_color
                    
                    idx = i * (2 * n - i - 1) // 2 + j - i - 1
                    colors[idx] = best_new_color
                    
                    current_cost += best_delta
                    stagnation_counter = 0
                    
                    if current_cost < best_cost_so_far:
                        best_cost_so_far = current_cost
                        best_colors_snapshot = colors[:]
                        
                    if current_cost == 0:
                        return colors
                else:
                    stagnation_counter += 1
            else:
                stagnation_counter += 1

            # Cooling
            T *= cooling_rate
            if T < min_T:
                T = min_T
            
            # Perturbation if stuck
            if stagnation_counter > kick_threshold:
                # Make a few random moves
                for _ in range(10):
                    ri = random.randint(0, n - 1)
                    rj = random.randint(0, n - 1)
                    if ri == rj:
                        continue
                    if ri > rj:
                        ri, rj = rj, ri
                    rc_old = color_matrix[ri][rj]
                    rc_new = random.randint(0, 4)
                    if rc_new == rc_old:
                        continue
                    
                    neighbors[rc_old][ri] &= ~(1 << rj)
                    neighbors[rc_old][rj] &= ~(1 << ri)
                    neighbors[rc_new][ri] |= (1 << rj)
                    neighbors[rc_new][rj] |= (1 << ri)
                    
                    color_matrix[ri][rj] = rc_new
                    color_matrix[rj][ri] = rc_new
                    ridx = ri * (2 * n - ri - 1) // 2 + rj - ri - 1
                    colors[ridx] = rc_new
                    
                    # Recalculate cost (expensive but rare)
                    current_cost = get_cost()
                    break
                stagnation_counter = 0
                # Reset temperature to help escape
                T = max(T, 1.0)

        return None

    # Strategy: Try to find valid colorings for n=26 down to 20.
    # First, try to find a solution for n=26. If found, try to extend it to n=27.
    # If n=27 fails, return the best n found.
    
    # Phase 1: Find a valid coloring for n=26
    # We use multiple restarts to increase the chance of finding a valid coloring.
    # The time budget is split among restarts.
    
    n_target = 26
    found_26 = False
    solution_26 = None
    
    # Give 80% of the time to finding n=26
    time_for_26 = (deadline - time.time()) * 0.8
    num_restarts = max(1, int(time_for_26 / 10.0))
    
    for restart in range(num_restarts):
        remaining_time = deadline - time.time() - 5.0
        if remaining_time < 5.0:
            break
        
        t_limit = min(10.0, remaining_time)
        sol = solve_sa(n_target, t_limit)
        
        if sol is not None:
            if count_violations(n_target, sol) == 0:
                best_n = n_target
                best_colors = sol
                found_26 = True
                solution_26 = sol
                break
    
    # Phase 2: If we found n=26, try to extend to n=27
    if found_26:
        # We have a valid coloring for n=26.
        # Try to extend it to n=27 by adding a new vertex 26.
        # The new vertex needs to be connected to all existing 26 vertices.
        # We can try different colorings for the new edges and see if we can fix the rest.
        
        # Start with the existing coloring and add new edges with random colors.
        # Then run SA on the n=27 instance.
        
        n_target_27 = 27
        # Create an initial coloring for n=27 by extending the n=26 solution.
        # The first 26*25/2 edges are the same.
        # The new edges are (26, 0), (26, 1), ..., (26, 25).
        
        initial_27 = solution_26[:]
        for j in range(26):
            initial_27.append(random.randint(0, 4))
        
        # Run SA on n=27 with this initial coloring.
        # Give the remaining time to this attempt.
        remaining_time = deadline - time.time() - 5.0
        if remaining_time > 5.0:
            sol_27 = solve_sa(n_target_27, remaining_time, initial_27)
            
            if sol_27 is not None:
                if count_violations(n_target_27, sol_27) == 0:
                    best_n = n_target_27
                    best_colors = sol_27
    
    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    res = construct()
    print(f"n={res['n']}")
