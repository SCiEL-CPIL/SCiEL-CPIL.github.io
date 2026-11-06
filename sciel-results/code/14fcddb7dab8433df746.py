import time
import random
import sys

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

    def solve_local_search(n, time_limit):
        """
        Local search with incremental cost updates.
        Returns a valid coloring list or None if not found within time_limit.
        """
        total_edges = n * (n - 1) // 2
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
        temperature = 1.0
        cooling_rate = 0.99995
        min_temperature = 0.01

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
            best_cost = current_cost
            best_delta = 0

            # Evaluate all possible new colors
            for c_new in range(5):
                if c_new == c_old:
                    continue
                
                delta = 0
                
                # c_old removals
                for w in range(n):
                    if w == i or w == j:
                        continue
                    if neighbors[c_old][j] & (1 << w):
                        k = bin(neighbors[c_old][i] & neighbors[c_old][w]).count('1')
                        delta -= (k - 1)
                    if neighbors[c_old][i] & (1 << w):
                        k = bin(neighbors[c_old][j] & neighbors[c_old][w]).count('1')
                        delta -= (k - 1)
                
                # c_new additions
                for w in range(n):
                    if w == i or w == j:
                        continue
                    if neighbors[c_new][j] & (1 << w):
                        k = bin(neighbors[c_new][i] & neighbors[c_new][w]).count('1')
                        delta += k
                    if neighbors[c_new][i] & (1 << w):
                        k = bin(neighbors[c_new][j] & neighbors[c_new][w]).count('1')
                        delta += k
                        
                new_cost = current_cost + delta
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_new_color = c_new
                    best_delta = delta
            
            # Simulated annealing acceptance
            if best_new_color != -1:
                # Accept if it improves or with probability based on temperature
                if best_delta < 0 or random.random() < pow(2.718281828, -best_delta / temperature):
                    neighbors[c_old][i] &= ~(1 << j)
                    neighbors[c_old][j] &= ~(1 << i)
                    neighbors[best_new_color][i] |= (1 << j)
                    neighbors[best_new_color][j] |= (1 << i)
                    
                    color_matrix[i][j] = best_new_color
                    color_matrix[j][i] = best_new_color
                    
                    idx = i * (2 * n - i - 1) // 2 + j - i - 1
                    colors[idx] = best_new_color
                    
                    current_cost = best_cost
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
            
            # Cool down
            temperature = max(min_temperature, temperature * cooling_rate)
            
            # Perturbation if stuck
            if stagnation_counter > 100:
                # Make a random move even if it increases cost
                for _ in range(5):
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
                temperature = 1.0  # Reset temperature after perturbation

        return None

    # Strategy: Try to find valid colorings for n=27 down to 23.
    # Give more time to larger n values
    targets = [27, 26, 25, 24, 23, 22, 21, 20]
    
    for n in targets:
        if time.time() > deadline - 5:
            break
        
        # Allocate time based on n
        if n >= 27:
            t_limit = min(30.0, deadline - time.time() - 2)
        elif n >= 26:
            t_limit = min(25.0, deadline - time.time() - 2)
        else:
            t_limit = min(10.0, deadline - time.time() - 2)
        
        if t_limit < 1.0:
            break
            
        sol = solve_local_search(n, t_limit)
        
        if sol is not None:
            if count_violations(n, sol) == 0:
                best_n = n
                best_colors = sol
                break
            else:
                pass
                
    # If we didn't find anything better than n=11, ensure we have a valid coloring
    if best_n == 11 and count_violations(best_n, best_colors) != 0:
        # Generate a valid coloring for n=11
        sol = solve_local_search(11, 5.0)
        if sol is not None and count_violations(11, sol) == 0:
            best_n = 11
            best_colors = sol
        else:
            # Use the seed solution which is guaranteed valid
            best_n, best_colors = get_seed_solution()
                
    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    res = construct()
    print(f"n={res['n']}")
