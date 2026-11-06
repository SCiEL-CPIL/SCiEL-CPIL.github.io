import random
import time
import numpy as np

N = 64
TOTAL_POINTS = N * N

def construct():
    # Precompute coordinates
    coords = [(x, y) for x in range(N) for y in range(N)]
    
    # Precompute squared distances matrix
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    xs = np.array(xs)
    ys = np.array(ys)
    
    diff_x = xs[:, None] - xs[None, :]
    diff_y = ys[:, None] - ys[None, :]
    dist_matrix = diff_x**2 + diff_y**2
    
    MAX_DIST = 2 * (N - 1) ** 2

    def calc_cost(indices):
        n = len(indices)
        if n < 3:
            return 0
        cost = 0
        for k in indices:
            counts = [0] * (MAX_DIST + 1)
            for i in indices:
                if i != k:
                    d = dist_matrix[k, i]
                    counts[d] += 1
            for c in counts:
                if c > 1:
                    cost += c * (c - 1) // 2
        return cost

    def solve():
        best_indices = []
        best_size = 0
        start_time = time.time()
        
        while time.time() - start_time < 290:
            # Run SA
            current_indices = random.sample(range(TOTAL_POINTS), 30)
            current_cost = calc_cost(current_indices)
            current_energy = current_cost * 1000 - len(current_indices)
            
            best_iter_indices = current_indices[:]
            best_iter_energy = current_energy
            
            T = 500.0
            T_min = 1.0
            cooling_rate = 0.9995
            steps = 50000
            
            for step in range(steps):
                if time.time() - start_time > 285:
                    break
                if T < T_min:
                    break
                    
                move_type = random.randint(0, 2)
                new_indices = current_indices[:]
                
                if move_type == 0: # Add
                    if len(new_indices) < TOTAL_POINTS:
                        set_current = set(current_indices)
                        while True:
                            cand = random.randint(0, TOTAL_POINTS - 1)
                            if cand not in set_current:
                                break
                        new_indices.append(cand)
                elif move_type == 1: # Remove
                    if len(new_indices) > 0:
                        idx_to_remove = random.randint(0, len(new_indices) - 1)
                        new_indices.pop(idx_to_remove)
                else: # Swap
                    if len(new_indices) > 0:
                        idx_to_swap = random.randint(0, len(new_indices) - 1)
                        set_current = set(current_indices)
                        while True:
                            cand = random.randint(0, TOTAL_POINTS - 1)
                            if cand not in set_current:
                                break
                        new_indices[idx_to_swap] = cand
                
                new_cost = calc_cost(new_indices)
                new_energy = new_cost * 1000 - len(new_indices)
                
                delta = new_energy - current_energy
                
                if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                    current_indices = new_indices
                    current_cost = new_cost
                    current_energy = new_energy
                    
                    if current_energy < best_iter_energy:
                        best_iter_energy = current_energy
                        best_iter_indices = current_indices[:]
                
                T *= cooling_rate
            
            # Check if best_iter_indices is valid (cost 0)
            final_cost = calc_cost(best_iter_indices)
            if final_cost == 0:
                if len(best_iter_indices) > best_size:
                    best_size = len(best_iter_indices)
                    best_indices = best_iter_indices[:]
            else:
                # Fallback: Repair by removing points
                repair_indices = best_iter_indices[:]
                for _ in range(100):
                    if calc_cost(repair_indices) == 0:
                        break
                    if repair_indices:
                        repair_indices.pop(random.randint(0, len(repair_indices) - 1))
                
                if calc_cost(repair_indices) == 0:
                    if len(repair_indices) > best_size:
                        best_size = len(repair_indices)
                        best_indices = repair_indices[:]
        
        return [coords[i] for i in best_indices]

    return solve()

# Run construct to get the result
# The function construct is the entry point.
