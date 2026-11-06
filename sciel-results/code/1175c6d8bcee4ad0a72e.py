"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s deadline, 5s safety margin for overhead/return
    
    N = 256
    
    # Build conflict sets
    squares = set()
    s = 1
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1
    
    # conflict[a] = set of b in [1,N] such that a+b is a perfect square
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for sq in squares:
            b = sq - a
            if 1 <= b <= N:
                conflict[a].add(b)
    
    # Numbers where 2a is a perfect square are forbidden (self-conflict)
    forbidden = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            forbidden.add(a)
    
    valid_numbers = [x for x in range(1, N + 1) if x not in forbidden]
    
    # Use bitmask for speed
    M = len(valid_numbers)
    num_to_idx = {x: i for i, x in enumerate(valid_numbers)}
    idx_to_num = valid_numbers
    
    # Build adjacency as bitmasks
    adj = [0] * M
    for i, a in enumerate(valid_numbers):
        for b in conflict[a]:
            if b in num_to_idx:
                j = num_to_idx[b]
                if j != i:  # exclude self
                    adj[i] |= (1 << j)
    
    # Compute degrees
    degrees = [bin(adj[i]).count('1') for i in range(M)]
    
    def greedy_init(order):
        selected = 0
        for i in order:
            if (adj[i] & selected) == 0:
                selected |= (1 << i)
        return selected
    
    # Initial solution: Greedy with lowest degree first
    degree_order = sorted(range(M), key=lambda i: degrees[i])
    best_mask = greedy_init(degree_order)
    best_size = bin(best_mask).count('1')
    
    # Simulated Annealing
    def simulated_annealing(start_mask, start_size, steps, t_start, t_end):
        current_mask = start_mask
        current_size = start_size
        t = t_start
        cooling_rate = (t_end / t_start) ** (1.0 / steps)
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Pick a random vertex
            v = random.randint(0, M - 1)
            
            if (current_mask >> v) & 1:
                # v is in set, try removing it
                trial_mask = current_mask & ~(1 << v)
                trial_size = current_size - 1
                delta_e = 1  # Energy increases by 1 (size decreases by 1)
            else:
                # v is not in set, try adding it
                # Find neighbors in current set
                neighbors_in_set = adj[v] & current_mask
                if neighbors_in_set == 0:
                    trial_mask = current_mask | (1 << v)
                    trial_size = current_size + 1
                    delta_e = -1
                else:
                    # Remove all neighbors, add v
                    trial_mask = (current_mask & ~neighbors_in_set) | (1 << v)
                    trial_size = current_size - bin(neighbors_in_set).count('1') + 1
                    delta_e = bin(neighbors_in_set).count('1') - 1
            
            # Acceptance criterion
            if delta_e <= 0:
                current_mask = trial_mask
                current_size = trial_size
            else:
                # Accept with probability exp(-delta_e / t)
                # delta_e is positive, so -delta_e/t is negative
                if random.random() < math.exp(-delta_e / t):
                    current_mask = trial_mask
                    current_size = trial_size
            
            t *= cooling_rate
            
        return current_mask, current_size

    # Run multiple SA runs with different initializations and temperatures
    random.seed(42)
    
    # 1. SA from greedy lowest degree
    mask, size = simulated_annealing(best_mask, best_size, 50000, 50.0, 0.1)
    if size > best_size:
        best_mask = mask
        best_size = size

    # 2. SA from random greedy restarts
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        random_order = list(range(M))
        random.shuffle(random_order)
        init_mask = greedy_init(random_order)
        init_size = bin(init_mask).count('1')
        
        # Use higher temp for exploration
        mask, size = simulated_annealing(init_mask, init_size, 30000, 100.0, 0.1)
        if size > best_size:
            best_mask = mask
            best_size = size

    # 3. Final polish: SA with low temp from best found
    if time.time() < deadline - 2:
        mask, size = simulated_annealing(best_mask, best_size, 20000, 10.0, 0.01)
        if size > best_size:
            best_mask = mask
            best_size = size

    # Convert back to actual numbers
    result = [idx_to_num[i] for i in range(M) if (best_mask >> i) & 1]
    
    if not result:
        result = [x for x in range(1, N + 1, 3) if x not in forbidden]
    
    return sorted(result)
# EVOLVE-BLOCK-END
