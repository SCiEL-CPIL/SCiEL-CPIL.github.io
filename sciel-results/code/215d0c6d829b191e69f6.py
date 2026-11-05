import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # Target n=26 based on known constructions
    n = 26
    num_colors = 5
    
    # Fallback solution for n=11
    def get_fallback():
        n_seed = 11
        colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
        return {'n': n_seed, 'colors': colors_seed}

    # Data structures
    # col[i][j] is the color of edge (i,j)
    col = [[0]*n for _ in range(n)]
    # adj[c][v] is a bitmask of neighbors of v in color c
    adj = [[0]*n for _ in range(num_colors)]
    
    def build_adj():
        for c in range(num_colors):
            for v in range(n):
                adj[c][v] = 0
        for i in range(n):
            for j in range(i+1, n):
                c = col[i][j]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                
    def random_init():
        for i in range(n):
            for j in range(i+1, n):
                c = random.randint(0, num_colors-1)
                col[i][j] = c
                col[j][i] = c
        build_adj()

    def calc_penalty():
        total = 0
        for c in range(num_colors):
            for u in range(n):
                mask_u = adj[c][u]
                # Optimization: only check v > u to avoid double counting pairs
                # But the penalty is defined on unordered pairs {u,v}
                for v in range(u+1, n):
                    common = mask_u & adj[c][v]
                    k = common.bit_count()
                    if k >= 2:
                        total += k * (k - 1) // 2
        return total

    def try_solve_sa(restart_id):
        random_init()
        current_penalty = calc_penalty()
        
        if current_penalty == 0:
            return True
        
        best_penalty = current_penalty
        best_col = [row[:] for row in col]
        
        T = 10.0
        cooling = 0.999
        min_T = 0.1
        
        no_improve = 0
        
        while time.time() < deadline:
            if current_penalty == 0:
                return True
                
            # Pick random edge
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
                
            old_c = col[i][j]
            new_c = random.randint(0, num_colors-1)
            if new_c == old_c:
                continue
            
            # Apply change
            col[i][j] = new_c
            col[j][i] = new_c
            
            # Update adjacencies
            # Remove from old_c
            adj[old_c][i] &= ~(1 << j)
            adj[old_c][j] &= ~(1 << i)
            # Add to new_c
            adj[new_c][i] |= (1 << j)
            adj[new_c][j] |= (1 << i)
            
            new_penalty = calc_penalty()
            
            delta = new_penalty - current_penalty
            
            if delta < 0:
                current_penalty = new_penalty
                no_improve = 0
                if current_penalty < best_penalty:
                    best_penalty = current_penalty
                    best_col = [row[:] for row in col]
            else:
                # Accept with probability exp(-delta/T)
                if random.random() < math.exp(-delta / T):
                    current_penalty = new_penalty
                    no_improve = 0
                else:
                    # Revert
                    col[i][j] = old_c
                    col[j][i] = old_c
                    adj[old_c][i] |= (1 << j)
                    adj[old_c][j] |= (1 << i)
                    adj[new_c][i] &= ~(1 << j)
                    adj[new_c][j] &= ~(1 << i)
                    no_improve += 1
            
            T *= cooling
            if T < min_T:
                T = min_T
                
            # Restart local search if stuck too long
            if no_improve > 5000:
                break
                
        # Check if best found is valid
        if best_penalty == 0:
            # Copy best_col to col
            for i in range(n):
                for j in range(n):
                    col[i][j] = best_col[i][j]
            return True
            
        return False

    # Try to find a solution for n=26
    # Multiple restarts if needed
    max_restarts = 10
    for r in range(max_restarts):
        if try_solve_sa(r):
            # Verify
            if calc_penalty() == 0:
                res_colors = []
                for i in range(n):
                    for j in range(i+1, n):
                        res_colors.append(col[i][j])
                return {'n': n, 'colors': res_colors}
        
        if time.time() > deadline - 5:
            break

    # If we failed to find n=26, return fallback n=11
    return get_fallback()
