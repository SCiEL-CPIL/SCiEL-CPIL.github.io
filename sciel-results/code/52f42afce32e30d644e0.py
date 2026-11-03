"""
Memetic Genetic Algorithm (GA) for QAPLIB wil100.
Strategy:
1. Population-based search to maintain diversity and avoid getting stuck in a single basin of attraction.
2. Vectorized O(n^2) computation of cost deltas for ALL possible 2-opt swaps, enabling fast steepest-descent local search.
3. Genetic operators:
   - Crossover: Cycle Crossover (CX). Preserves subtour structures from parents, which is often effective for permutation problems.
   - Mutation: Multi-point random swap mutation. The mutation rate adapts or is fixed to ensure sufficient exploration.
4. Selection: Tournament selection.
5. Elitism: The best individual from the previous generation is always carried over to the next.
6. Time management:
   - Strict deadline guard.
   - Fixed number of generations, but each generation's local search is time-boxed or limited to a few iterations to allow for population updates.
   
Why this is different:
- Current program (ILS): Single trajectory (or few independent ones) relying on perturbation to escape local minima.
- New program (GA): Maintains a population of N solutions. Crossover combines information from different parts of the search space. Mutation introduces random noise. This parallel exploration is structurally different from sequential ILS.
"""
import numpy as np
import time
import random

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _greedy_initial(a, b):
    """
    Create an initial permutation using a greedy heuristic.
    Matches facilities with high flow to locations with low distance.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy = np.argsort(-a_row_sums, kind="stable")
    # Sort locations by ascending distance sum
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
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

def _local_search(a, b, p, max_iters=50, time_limit=None, start_time=None):
    """
    Steepest descent 2-opt local search.
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
        min_delta = np.min(deltas)
        
        if min_delta >= 0:
            break # Local minimum reached
            
        # Find indices of the best move
        i, j = np.unravel_index(np.argmin(deltas), deltas.shape)
        
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
    
    # Population parameters
    pop_size = 50
    elite_size = 5
    mutation_swaps = 3 # Fixed for simplicity, could adapt
    
    # Initialize population
    # One greedy, rest random
    population = []
    costs = []
    
    p0 = _greedy_initial(a, b)
    p0, c0 = _local_search(a, b, p0, max_iters=10, time_limit=total_deadline, start_time=start_time)
    population.append(p0.copy())
    costs.append(c0)
    
    for _ in range(pop_size - 1):
        p = np.random.permutation(n)
        p, c = _local_search(a, b, p, max_iters=10, time_limit=total_deadline, start_time=start_time)
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
            
            # Local Search (polish)
            # Limit time per individual to ensure we complete the generation
            # Rough estimate: 50 individuals * 2s each = 100s. Too much.
            # We need to be smarter.
            # Let's check time remaining.
            time_left = total_deadline - (time.time() - start_time)
            if time_left < 1.0:
                break
            
            # Allocate a small time budget for local search per individual
            # e.g., 0.5s max? Or just a few iterations.
            # The vectorized delta is fast. Let's do up to 20 iterations or 0.5s.
            child, c = _local_search(a, b, child, max_iters=20, time_limit=0.5, start_time=time.time())
            
            new_population.append(child)
            
        population = new_population
        # Recalculate costs for the new population (we have them for elites, need for offspring)
        # Actually, _local_search returns the cost. But we didn't store it in the list above properly for the new ones?
        # Let's just recalculate all costs to be safe, it's O(n^2) which is fast.
        costs = [_qap_cost(a, b, p) for p in population]
        
        gen += 1
        
    return best_p.tolist()

# EVOLVE-BLOCK-END
