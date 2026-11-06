"""
Memetic Algorithm with Multi-Start Random Keys and Iterated Local Search (ILS)
for Flow Shop Scheduling.

Strategy:
1. Initialize a population of diverse permutations using Random Keys (sort by random values).
2. For each individual, apply a strong local search (Or-opt moves: move blocks of 1, 2, or 3 jobs).
3. Maintain a pool of the best K solutions found so far.
4. If local search stagnates (no improvement for X iterations), restart from a random perturbation
   of a solution in the pool (Iterated Local Search style).
5. Continue until time limit.

This differs from the current program's single-trajectory Hill Climbing/SA by explicitly
managing a population of elite solutions and using block moves (Or-opt) which are often
more effective for flow shop than simple 2-opt reversals or single-job insertions.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _calc_makespan(processing_times, perm):
    """
    Calculate makespan for a given permutation.
    Optimized for speed with local variable binding.
    """
    n_machines = processing_times.shape[1]
    if len(perm) == 0:
        return 0.0
    
    completion = np.zeros(n_machines)
    pt = processing_times
    comp = completion
    nm = n_machines
    
    for j in perm:
        row = pt[j]
        prev_c = 0.0
        for m in range(nm):
            if m == 0:
                new_c = comp[0] + row[0]
            else:
                if comp[m] > prev_c:
                    new_c = comp[m] + row[m]
                else:
                    new_c = prev_c + row[m]
            comp[m] = new_c
            prev_c = new_c
            
    return float(comp[-1])

def _or_opt_move(perm, i, j, block_size):
    """
    Generate a new permutation by moving a block of size 'block_size' 
    from position i to position j.
    Returns the new permutation list.
    """
    if block_size <= 0 or i < 0 or i + block_size > len(perm):
        return perm
    
    # Extract block
    block = perm[i:i+block_size]
    
    # Remove block
    new_perm = perm[:i] + perm[i+block_size:]
    
    # Adjust j if the removal happened before j
    if j > i:
        j -= block_size
    # Clamp j
    if j < 0:
        j = 0
    if j > len(new_perm):
        j = len(new_perm)
        
    # Insert block at j
    # Note: Standard Or-opt usually inserts such that the block occupies positions j..j+block_size-1
    # We insert before index j in the reduced list.
    return new_perm[:j] + block + new_perm[j:]

def _local_search_or_opt(processing_times, perm, deadline, max_stagnation=100):
    """
    Local search using Or-opt moves (blocks of size 1, 2, 3).
    Uses a first-improvement strategy to be fast.
    Returns the best permutation found.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm
    
    best_perm = list(perm)
    best_ms = _calc_makespan(processing_times, best_perm)
    
    stagnation_count = 0
    block_sizes = [1, 2, 3]
    
    # Adaptive: if n_jobs is large, we might need to sample positions to fit in time
    # But Or-opt is O(N^2 * N) if we check all. 
    # For N=100, N^2 = 10,000 moves. Makespan calc is O(N*M).
    # Let's try to be efficient.
    
    while time.time() < deadline - 2.0:
        improved = False
        # Randomize the order of moves to avoid bias
        # We will iterate through positions and block sizes
        # To save time, we can shuffle the positions to check
        
        positions = list(range(n_jobs))
        random.shuffle(positions)
        
        for i in positions:
            if time.time() > deadline - 2.0:
                break
            
            for bs in block_sizes:
                if i + bs > n_jobs:
                    continue
                
                # Determine valid insertion positions
                # We want to move block from i to j
                # j can be anywhere except where it would result in the same permutation
                # Simplest: try all j in 0..n_jobs-1
                
                # To speed up, we can just pick a few random j's if N is large
                if n_jobs > 50:
                    # Sample 5 random insertion points
                    js = [random.randint(0, n_jobs - 1) for _ in range(5)]
                else:
                    js = range(n_jobs)
                
                for j in js:
                    # Skip if j is effectively the same position
                    # If i == j, or if the block is moved to an adjacent spot that results in same order?
                    # Or-opt move definition: remove block, insert at j.
                    # If j == i, no change.
                    # If j == i+1, block moves 1 step right? No, indices shift.
                    # Let's just calculate and compare.
                    
                    new_perm = _or_opt_move(best_perm, i, j, bs)
                    
                    # Check if it's actually a change
                    if new_perm == best_perm:
                        continue
                        
                    ms = _calc_makespan(processing_times, new_perm)
                    
                    if ms < best_ms - 1e-9:
                        best_ms = ms
                        best_perm = new_perm
                        improved = True
                        # Reset stagnation
                        stagnation_count = 0
                        # Break out to restart the neighborhood scan from the new best
                        # This is "First Improvement" within the pass, but we restart the pass
                        break
                
                if improved:
                    break
            
            if improved:
                break
        
        if not improved:
            stagnation_count += 1
            # If we have stagnated for a while, break out of local search
            # and let the main loop decide (restart or move to next individual)
            if stagnation_count > max_stagnation:
                break
                
    return best_perm

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # 1. Generate Initial Population
    # Use Random Keys: generate random numbers, sort indices by them.
    # This creates diverse permutations.
    
    pop_size = 10
    # If N is very small, pop_size can be larger
    if n_jobs < 20:
        pop_size = 20
    elif n_jobs > 100:
        pop_size = 5
    
    # Elite pool: list of (makespan, permutation)
    elite_pool = []
    
    # Add NEH solution as a strong start
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    neh_perm = list(sorted_jobs)
    neh_ms = _calc_makespan(processing_times, neh_perm)
    elite_pool.append((neh_ms, neh_perm))
    
    # Generate random keys permutations
    for _ in range(pop_size):
        keys = np.random.random(n_jobs)
        perm = list(np.argsort(keys))
        ms = _calc_makespan(processing_times, perm)
        elite_pool.append((ms, perm))
    
    # Sort elite pool by makespan
    elite_pool.sort(key=lambda x: x[0])
    
    # Keep only top K elites
    max_elites = min(5, len(elite_pool))
    elite_pool = elite_pool[:max_elites]
    
    best_global_ms, best_global_perm = elite_pool[0]
    
    # 2. Main Loop: Iterated Local Search on Elites
    # We will iteratively pick an elite, run local search, and update the pool.
    
    while time.time() < deadline - 5.0:
        # Pick an elite to improve
        # Weighted random selection: better elites have higher chance, but we also want diversity
        # Let's just cycle through or pick randomly from top 3
        if len(elite_pool) > 0:
            idx = random.randint(0, min(2, len(elite_pool) - 1))
            current_ms, current_perm = elite_pool[idx]
        else:
            break
            
        # Run Local Search
        # Give it a time slice? Or run to convergence?
        # Running to convergence might take too long if it gets stuck in a large basin.
        # Let's cap the local search time per call to ensure we explore multiple starts.
        ls_deadline = min(deadline - 5.0, time.time() + 10.0)
        
        improved_perm = _local_search_or_opt(processing_times, current_perm, ls_deadline)
        improved_ms = _calc_makespan(processing_times, improved_perm)
        
        # Update Global Best
        if improved_ms < best_global_ms:
            best_global_ms = improved_ms
            best_global_perm = improved_perm
        
        # Update Elite Pool
        # Insert new solution
        elite_pool.append((improved_ms, improved_perm))
        elite_pool.sort(key=lambda x: x[0])
        
        # Trim pool
        if len(elite_pool) > max_elites:
            # Remove the worst
            elite_pool = elite_pool[:max_elites]
            
        # If the local search didn't improve the specific elite significantly,
        # we might want to perturb it for the next iteration if we pick it again.
        # But since we pick randomly, diversity is maintained by the initial pool.
        
        # Optional: If we are stagnating globally (no improvement in best_global for a while),
        # inject a new random individual into the pool to maintain diversity.
        # For simplicity, we rely on the initial diverse pool and the fact that 
        # local search modifies the permutation. If the pool converges, we might stop improving.
        # Let's add a diversity injection every few iterations if time permits.
        
        # Check if all elites are identical (or very similar)
        # If so, inject a random one.
        if len(elite_pool) > 1:
            # Simple check: are the first two permutations identical?
            if elite_pool[0][1] == elite_pool[1][1]:
                # Inject random
                keys = np.random.random(n_jobs)
                new_perm = list(np.argsort(keys))
                new_ms = _calc_makespan(processing_times, new_perm)
                elite_pool.append((new_ms, new_perm))
                elite_pool.sort(key=lambda x: x[0])
                elite_pool = elite_pool[:max_elites]

    return best_global_perm
# EVOLVE-BLOCK-END
