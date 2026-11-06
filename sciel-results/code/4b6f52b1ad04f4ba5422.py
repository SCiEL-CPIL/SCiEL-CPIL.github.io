import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Fallback solution (n=11) from seed
    def get_seed_solution(n):
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
    
    # Try to find larger n using Restarted Simulated Annealing
    # We know n=26 is possible.
    # We will try n from 26 down to 12.
    
    best_solution = get_seed_solution(11)
    
    for n in range(26, 11, -1):
        # Allocate time based on n
        # Larger n needs more time, but we want to try 26 first.
        # If we have 100s left, give 30s to 26, 20s to 25, etc.
        time_remaining = deadline - time.time()
        if time_remaining < 5:
            break
            
        # Heuristic: 26 is hard, 25 is hard, ...
        # Give 26 a big chunk, but allow restarts.
        time_limit_for_n = min(30, time_remaining * 0.5) 
        if n == 26:
             time_limit_for_n = min(40, time_remaining * 0.6)
        elif n == 25:
             time_limit_for_n = min(20, time_remaining * 0.4)
        else:
             time_limit_for_n = min(10, time_remaining * 0.2)
             
        if time_limit_for_n < 2:
            continue

        solution = find_valid_coloring_sa_restarted(n, time_limit=time_limit_for_n)
        if solution is not None:
            return solution
            
    return best_solution

def find_valid_coloring_sa_restarted(n, time_limit=10):
    """
    Uses Restarted Simulated Annealing to find a 5-coloring of K_n with no C4 in any color.
    """
    num_edges = n * (n - 1) // 2
    if num_edges == 0:
        return {'n': n, 'colors': []}
        
    start_t = time.time()
    
    # Precompute edge indices
    edge_index = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[i][j] = idx
            edge_index[j][i] = idx
            idx += 1
            
    # We will run multiple short SA instances
    # Each instance runs for a fraction of the time, then restarts with new random init.
    # This helps escape local minima.
    
    while time.time() < start_t + time_limit:
        # Check if we have enough time for a short run (e.g., 1s)
        if time.time() > start_t + time_limit - 0.5:
            break
            
        solution = _run_sa_once(n, edge_index, time_limit=1.0)
        if solution is not None:
            return solution
            
    return None

def _run_sa_once(n, edge_index, time_limit=1.0):
    num_edges = n * (n - 1) // 2
    
    # Initialize random coloring
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    # Bitset adjacency
    adj_bit = [[0] * n for _ in range(5)]
    
    def rebuild_adj_bit():
        for c in range(5):
            for u in range(n):
                adj_bit[c][u] = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[edge_index[i][j]]
                adj_bit[c][i] |= (1 << j)
                adj_bit[c][j] |= (1 << i)
                
    rebuild_adj_bit()
    
    def count_c4s_fast():
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = bin(adj_bit[c][u] & adj_bit[c][v]).count('1')
                    if common >= 2:
                        total += common * (common - 1) // 2
        return total

    energy = count_c4s_fast()
    if energy == 0:
        return {'n': n, 'colors': colors}

    # SA Parameters for short run
    T = 5.0
    cooling = 0.99
    min_T = 0.1
    start_t = time.time()
    
    def delta_energy_remove(i, j, c):
        delta = 0
        Nj = [k for k in range(n) if k != i and (adj_bit[c][j] & (1 << k))]
        Ni = [k for k in range(n) if k != j and (adj_bit[c][i] & (1 << k))]
        
        for k in Nj:
            old_common = bin(adj_bit[c][i] & adj_bit[c][k]).count('1')
            new_common = old_common - 1
            delta -= (old_common * (old_common - 1) // 2) - (new_common * (new_common - 1) // 2)
            
        for k in Ni:
            old_common = bin(adj_bit[c][j] & adj_bit[c][k]).count('1')
            new_common = old_common - 1
            delta -= (old_common * (old_common - 1) // 2) - (new_common * (new_common - 1) // 2)
        return delta

    def delta_energy_add(i, j, c):
        delta = 0
        Nj = [k for k in range(n) if k != i and (adj_bit[c][j] & (1 << k))]
        Ni = [k for k in range(n) if k != j and (adj_bit[c][i] & (1 << k))]
        
        for k in Nj:
            old_common = bin(adj_bit[c][i] & adj_bit[c][k]).count('1')
            new_common = old_common + 1
            delta += (new_common * (new_common - 1) // 2) - (old_common * (old_common - 1) // 2)
            
        for k in Ni:
            old_common = bin(adj_bit[c][j] & adj_bit[c][k]).count('1')
            new_common = old_common + 1
            delta += (new_common * (new_common - 1) // 2) - (old_common * (old_common - 1) // 2)
        return delta

    while time.time() < start_t + time_limit:
        if T < min_T:
            T = min_T
            
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
            
        idx = edge_index[i][j]
        c1 = colors[idx]
        c2 = random.randint(0, 4)
        if c1 == c2:
            continue
            
        dE = delta_energy_remove(i, j, c1) + delta_energy_add(i, j, c2)
        
        if dE <= 0 or random.random() < math.exp(-dE / T):
            colors[idx] = c2
            adj_bit[c1][i] &= ~(1 << j)
            adj_bit[c1][j] &= ~(1 << i)
            adj_bit[c2][i] |= (1 << j)
            adj_bit[c2][j] |= (1 << i)
            energy += dE
            
            if energy == 0:
                return {'n': n, 'colors': colors}
                
        T *= cooling
        
    return None
# EVOLVE-BLOCK-END
