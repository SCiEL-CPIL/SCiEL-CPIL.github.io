"""
Memetic Genetic Algorithm (GA) for QAPLIB wil100 with optimized delta computation.
Strategy:
1. Population-based search with vectorized O(n^2) swap delta computation for fast local search.
2. Cycle Crossover + multi-point swap mutation.
3. Tournament selection with elitism.
4. Optimized time allocation: more generations with lighter local search, plus a final polish phase.
"""
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _hungarian_initial(a, b, mode='row'):
    """
    Generate an initial permutation using the Hungarian algorithm on a surrogate cost matrix.
    mode: 'row' or 'col'
    """
    n = len(a)
    if mode == 'row':
        a_sums = a.sum(axis=1)
        b_sums = b.sum(axis=1)
        C = -np.outer(a_sums, b_sums)
    elif mode == 'col':
        a_sums = a.sum(axis=0)
        b_sums = b.sum(axis=0)
        C = -np.outer(a_sums, b_sums)
    else:
        C = -np.outer(a.sum(axis=1), b.sum(axis=1))
        
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p

def _compute_all_swap_deltas(a, b, p):
    """
    Compute the cost delta for swapping every pair of positions (i, j) 
    in the current permutation p.
    
    Returns a matrix D where D[i, j] is the change in cost if we swap 
    positions i and j in p.
    """
    C = b[np.ix_(p, p)]
    # Precompute symmetric part of a for reuse
    a_sym = a + a.T
    
    S1 = a.T @ C
    T1 = a @ C
    
    D_S1 = np.diag(S1)
    D_T1 = np.diag(T1)
    
    # Efficient computation using broadcasting
    Term1 = S1 + S1.T - D_S1[:, None] - D_S1[None, :]
    Term2 = T1 + T1.T - D_T1[:, None] - D_T1[None, :]
    
    Raw = Term1 + Term2
    
    # Error correction term
    Error = -2.0 * a_sym * C
    
    Delta = Raw - Error
    
    # Zero out diagonal
    np.fill_diagonal(Delta, 0.0)
    
    return Delta

def _local_search(a, b, p, max_iters=50, time_limit=None, start_time=None):
    """
    Steepest descent 2-opt local search with early termination.
    """
    if time_limit is None:
        time_limit = float('inf')
    if start_time is None:
        start_time = time.time()
        
    current_cost = _qap_cost(a, b, p)
    n = len(p)
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        deltas = _compute_all_swap_deltas(a, b, p)
        
        # Find the best improving move
        min_idx = np.argmin(deltas)
        min_delta = deltas.flat[min_idx]
        
        if min_delta >= 0:
            break # Local minimum reached
            
        # Find indices of the best move
        i, j = np.unravel_index(min_idx, deltas.shape)
        
        # Apply swap
        p[i], p[j] = p[j], p[i]
        current_cost += min_delta
        
    return p, current_cost

def _cycle_crossover(p1, p2):
    """
    Cycle Crossover (CX).
    """
    n = len(p1)
    child = np.full(n, -1, dtype=np.int64)
    
    # Find cycles
    # A cycle is a sequence of positions where p1[i] is at position p2.index(p1[i])?
    # No, standard CX:
    # 1. Start at position 0.
    # 2. Value v = p1[0].
    # 3. Find position j in p2 where p2[j] == v.
    # 4. The cycle includes positions 0, j, and the next position in p1 at j, etc.
    
    visited = np.zeros(n, dtype=bool)
    
    # We need to map values to positions in p2 for O(1) lookup
    pos_in_p2 = np.empty(n, dtype=np.int64)
    for i in range(n):
        pos_in_p2[p2[i]] = i
        
    # Identify cycles
    # A cycle is defined by following the mapping: i -> pos_in_p2[p1[i]]
    # Actually, let's just iterate through positions.
    
    # Standard CX algorithm:
    # 1. Pick a random start position s.
    # 2. The cycle consists of positions s, j1, j2... such that:
    #    - p1[s] is at position j1 in p2.
    #    - p1[j1] is at position j2 in p2.
    #    - ... until we return to s.
    # 3. Copy these positions from p1 to child.
    # 4. Fill the rest from p2.
    
    s = random.randint(0, n - 1)
    current = s
    while not visited[current]:
        visited[current] = True
        child[current] = p1[current]
        # Move to the position in p2 that holds the same value as p1[current]
        val = p1[current]
        current = pos_in_p2[val]
        
    # Fill the rest from p2
    for i in range(n):
        if child[i] == -1:
            child[i] = p2[i]
            
    return child

def _mutate(p, num_swaps):
    """
    Apply random swaps to the permutation.
    """
    n = len(p)
    for _ in range(num_swaps):
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i != j:
            p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    total_deadline = 115.0 # 5s buffer
    start_time = time.time()
    
    # Population parameters - reduced pop size for more generations
    pop_size = 40
    elite_size = 4
    mutation_swaps = 2
    
    # Initialize population
    population = []
    costs = []
    
    # Add Hungarian inits with moderate local search
    for mode in ['row', 'col']:
        p = _hungarian_initial(a, b, mode)
        p, c = _local_search(a, b, p, max_iters=15, time_limit=2.0, start_time=start_time)
        population.append(p.copy())
        costs.append(c)
        
    # Fill rest with random + lighter local search
    while len(population) < pop_size:
        p = np.random.permutation(n)
        p, c = _local_search(a, b, p, max_iters=8, time_limit=1.0, start_time=start_time)
        population.append(p.copy())
        costs.append(c)
        
    best_idx = np.argmin(costs)
    best_p = population[best_idx].copy()
    best_cost = costs[best_idx]
    
    # Main GA Loop - more generations with lighter local search
    gen = 0
    while time.time() - start_time < total_deadline - 10.0:
        # Sort population by cost
        indices = np.argsort(costs)
        population = [population[i] for i in indices]
        costs = [costs[i] for i in indices]
        
        # Update global best
        if costs[0] < best_cost:
            best_cost = costs[0]
            best_p = population[0].copy()
            
        # Elitism: Keep the best 'elite_size'
        new_population = [population[i].copy() for i in range(elite_size)]
        
        # Generate offspring to fill the rest of the population
        while len(new_population) < pop_size:
            # Tournament selection
            candidates = random.sample(range(len(population)), 3)
            best_cand = min(candidates, key=lambda x: costs[x])
            p1 = population[best_cand]
            
            candidates = random.sample(range(len(population)), 3)
            best_cand = min(candidates, key=lambda x: costs[x])
            p2 = population[best_cand]
            
            # Crossover
            child = _cycle_crossover(p1, p2)
            
            # Mutation
            child = _mutate(child, mutation_swaps)
            
            # Check time remaining
            time_left = total_deadline - (time.time() - start_time)
            if time_left < 1.0:
                break
            
            # Lighter local search: fewer iterations, tighter time limit
            child, c = _local_search(a, b, child, max_iters=12, time_limit=0.3, start_time=time.time())
            
            new_population.append(child)
            
        population = new_population
        costs = [_qap_cost(a, b, p) for p in population]
        
        gen += 1
    
    # Final polish phase: intensive local search on the best solution
    time_left = total_deadline - (time.time() - start_time)
    if time_left > 5.0:
        best_p, best_cost = _local_search(a, b, best_p, max_iters=100, time_limit=time_left - 2.0, start_time=start_time)
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
