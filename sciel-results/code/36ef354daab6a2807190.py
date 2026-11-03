"""
Memetic Genetic Algorithm (GA) for QAPLIB wil100.
Strategy:
1. Population-based search to maintain diversity and avoid getting stuck in a single basin of attraction.
2. Vectorized O(n^2) computation of cost deltas for ALL possible 2-opt swaps, enabling fast steepest-descent local search.
3. Genetic operators:
   - Crossover: Cycle Crossover (CX). Preserves subtour structures from parents, which is often effective for permutation problems.
   - Mutation: Multi-point random swap mutation.
4. Selection: Tournament selection.
5. Elitism: The best individual from the previous generation is always carried over to the next.
6. Time management:
   - Strict deadline guard.
   - Fixed number of generations, but each generation's local search is time-boxed or limited to a few iterations to allow for population updates.
   
Improvements from inspirations:
- Enhanced Hungarian initialization using both row and column sums for better structural coherence.
- Integrated Simulated Annealing (SA) as a secondary local search mechanism to escape local minima more effectively than pure steepest descent.
- Used vectorized swap delta computation for both GA local search and SA.
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
    Compute the cost delta for swapping every pair of facilities (i, j) 
    in the current permutation p.
    
    Returns a matrix D where D[i, j] is the change in cost if we swap 
    positions i and j in p.
    """
    C = b[np.ix_(p, p)]
    S1 = a.T @ C
    T1 = a @ C
    
    D_S1 = np.diag(S1)
    D_T1 = np.diag(T1)
    
    Term1 = S1 - D_S1[:, None] - D_S1[None, :] + S1.T
    Term2 = T1 - D_T1[:, None] - D_T1[None, :] + T1.T
    
    Raw = Term1 + Term2
    
    # Error term: -2 * (a[i,j] c[i,j] + a[j,i] c[j,i])
    # c[i,j] = b[p[i], p[j]]
    # Since C is symmetric, c[j,i] = c[i,j].
    # So Error[i,j] = -2 * (a[i,j] + a[j,i]) * C[i,j]
    Error = -2.0 * (a + a.T) * C
    
    Delta = Raw - Error
    
    # Zero out diagonal
    np.fill_diagonal(Delta, 0.0)
    
    return Delta

def _local_search(a, b, p, max_iters=50, time_limit=None, start_time=None, use_sa=False, T_start=1000.0, T_end=1.0):
    """
    Steepest descent 2-opt local search with optional Simulated Annealing.
    """
    if time_limit is None:
        time_limit = float('inf')
    if start_time is None:
        start_time = time.time()
        
    current_cost = _qap_cost(a, b, p)
    n = len(p)
    best_p = p.copy()
    best_cost = current_cost
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        deltas = _compute_all_swap_deltas(a, b, p)
        
        # Find the best improving move
        min_delta = np.min(deltas)
        
        if use_sa:
            # Simulated Annealing: accept worse moves with some probability
            elapsed = time.time() - start_time
            frac = min(1.0, elapsed / time_limit) if time_limit < float('inf') else 0.0
            T = T_start * (T_end / T_start) ** frac
            
            if min_delta < 0 or (T > 1e-10 and np.random.rand() < np.exp(-min_delta / T)):
                i, j = np.unravel_index(np.argmin(deltas), deltas.shape)
                p[i], p[j] = p[j], p[i]
                current_cost += int(min_delta)
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            else:
                # Try a random move if the best move is not accepted
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i != j:
                    delta_ij = deltas[i, j]
                    if delta_ij < 0 or (T > 1e-10 and np.random.rand() < np.exp(-delta_ij / T)):
                        p[i], p[j] = p[j], p[i]
                        current_cost += int(delta_ij)
                        
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_p = p.copy()
        else:
            # Pure steepest descent
            if min_delta >= 0:
                break # Local minimum reached
                
            # Find indices of the best move
            i, j = np.unravel_index(np.argmin(deltas), deltas.shape)
            
            # Apply swap
            p[i], p[j] = p[j], p[i]
            current_cost += int(min_delta)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    return best_p, best_cost

def _cycle_crossover(p1, p2):
    """
    Cycle Crossover (CX).
    """
    n = len(p1)
    child = np.full(n, -1, dtype=np.int64)
    
    visited = np.zeros(n, dtype=bool)
    
    # We need to map values to positions in p2 for O(1) lookup
    pos_in_p2 = np.empty(n, dtype=np.int64)
    for i in range(n):
        pos_in_p2[p2[i]] = i
        
    # Standard CX algorithm
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
    
    # Population parameters
    pop_size = 30
    elite_size = 5
    mutation_swaps = 3
    
    # Initialize population
    population = []
    costs = []
    
    # Add Hungarian inits
    for mode in ['row', 'col']:
        p = _hungarian_initial(a, b, mode)
        p, c = _local_search(a, b, p, max_iters=30, time_limit=total_deadline, start_time=start_time)
        population.append(p.copy())
        costs.append(c)
        
    # Fill rest with random
    while len(population) < pop_size:
        p = np.random.permutation(n)
        p, c = _local_search(a, b, p, max_iters=20, time_limit=total_deadline, start_time=start_time)
        population.append(p.copy())
        costs.append(c)
        
    best_idx = np.argmin(costs)
    best_p = population[best_idx].copy()
    best_cost = costs[best_idx]
    
    # Main GA Loop
    gen = 0
    while time.time() - start_time < total_deadline - 5.0:
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
            def select():
                # Pick 3 random individuals, choose the best
                candidates = random.sample(range(len(population)), 3)
                best_cand = min(candidates, key=lambda x: costs[x])
                return population[best_cand]
            
            p1 = select()
            p2 = select()
            
            # Crossover
            child = _cycle_crossover(p1, p2)
            
            # Mutation
            child = _mutate(child, mutation_swaps)
            
            # Local Search (polish) with occasional SA
            time_left = total_deadline - (time.time() - start_time)
            if time_left < 1.0:
                break
            
            # Use SA for every 3rd offspring to escape local minima
            use_sa = (len(new_population) % 3 == 0)
            child, c = _local_search(a, b, child, max_iters=40, time_limit=1.0, start_time=time.time(), use_sa=use_sa)
            
            new_population.append(child)
            
        population = new_population
        # Recalculate costs for the new population
        costs = [_qap_cost(a, b, p) for p in population]
        
        gen += 1
        
    # Final polish with pure steepest descent on the best solution
    # This ensures we end at a local minimum, which is often better than a random point near one.
    time_left = total_deadline - (time.time() - start_time)
    if time_left > 2.0:
        best_p, best_cost = _local_search(a, b, best_p, max_iters=5000, time_limit=time_left, start_time=time.time(), use_sa=False)
    
    return best_p.tolist()
