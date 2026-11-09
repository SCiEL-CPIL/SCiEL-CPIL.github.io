import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try to improve n
    for n in range(12, 20):
        if time.time() > deadline - 10:
            break
            
        num_edges = n * (n - 1) // 2
        found = False
        
        # Try a few random restarts of Hill Climbing
        for restart in range(20):
            if time.time() > deadline - 15:
                break
                
            colors = np.random.randint(0, 5, num_edges)
            
            # Precompute indices
            # idx(u, v)
            
            def get_idx(u, v):
                if u > v: u, v = v, u
                return u * (2 * n - u - 1) // 2 + (v - u - 1)
            
            # Helper to compute energy
            def calc_energy(col):
                M = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
                for i in range(n):
                    for j in range(i + 1, n):
                        c = col[get_idx(i, j)]
                        M[c][i, j] = 1
                        M[c][j, i] = 1
                energy = 0
                for c in range(5):
                    M2 = M[c] @ M[c]
                    tri = np.triu(M2, 1)
                    energy += np.sum(tri * (tri - 1) // 2)
                return energy // 2

            current_energy = calc_energy(colors)
            best_local_energy = current_energy
            best_local_colors = colors.copy()
            
            # Hill Climbing
            for step in range(5000):
                if time.time() > deadline - 15:
                    break
                    
                e = np.random.randint(0, num_edges)
                old_c = colors[e]
                new_c = (old_c + random.randint(1, 4)) % 5
                colors[e] = new_c
                
                new_energy = calc_energy(colors)
                
                if new_energy < current_energy:
                    current_energy = new_energy
                    if current_energy < best_local_energy:
                        best_local_energy = current_energy
                        best_local_colors = colors.copy()
                        if current_energy == 0:
                            found = True
                            break
                else:
                    # Reject
                    colors[e] = old_c
                    
            if found:
                best_n = n
                best_colors = best_local_colors.tolist()
                break
        else:
            # If we didn't find a solution for this n, we likely won't for higher n
            # But we keep the best valid n we had
            break

    return {'n': best_n, 'colors': best_colors}
