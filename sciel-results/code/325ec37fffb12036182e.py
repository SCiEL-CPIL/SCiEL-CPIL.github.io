"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(pt, order):
    """Compute makespan for a given job order on a flow shop."""
    n_jobs = len(order)
    n_machines = pt.shape[1]
    if n_jobs == 0:
        return 0.0
    # C[i][j] = completion time of job i on machine j
    # We can compute this in O(n_jobs * n_machines)
    # Using a 1D array for the current machine times
    machine_times = np.zeros(n_machines, dtype=np.float64)
    for job_idx in order:
        row = pt[job_idx]
        for m in range(n_machines):
            if m == 0:
                machine_times[m] += row[m]
            else:
                # Completion on machine m = max(completion on machine m-1 for this job, machine m availability) + processing time
                # Since we process jobs sequentially in the order, and machines are sequential:
                # C(job, m) = max(C(job, m-1), C(prev_job, m)) + p(job, m)
                # In our iteration:
                # C(job, m-1) is the previous value of machine_times[m-1] before update? No.
                # Let's trace:
                # We update machine_times[m] in place.
                # For job j, machine 0: C = p_j0. machine_times[0] += p_j0.
                # For job j, machine 1: C = max(C_j0, machine_times[1]) + p_j1.
                # But machine_times[1] currently holds C_{prev_job, 1}.
                # And C_j0 is the new machine_times[0]? No, machine_times[0] was just updated.
                # So C_j0 = machine_times[0].
                # So C_j1 = max(machine_times[0], machine_times[1]) + p_j1.
                # Then machine_times[1] = C_j1.
                # This is correct.
                machine_times[m] = max(machine_times[m-1], machine_times[m]) + row[m]
    return machine_times[-1]


def _neh_order(pt, reverse=False, randomize=False, rng=None):
    """
    NEH heuristic for flow shop.
    If reverse=True, use the reverse NEH (start from the job with largest sum, but insert differently? 
    Actually, standard NEH builds the sequence. Reverse NEH is often just applying NEH to the reversed problem 
    or using a different insertion criterion. A common variant is to consider both forward and backward insertions.
    Here, we'll implement standard NEH. If reverse is True, we might just reverse the final order or use a different tie-break.
    For diversity, we can also randomize the selection of the next job if there are ties or by adding noise.
    """
    n_jobs = pt.shape[0]
    n_machines = pt.shape[1]
    
    # Calculate total processing time for each job
    totals = pt.sum(axis=1)
    
    # Sort jobs in descending order of total processing time
    if randomize and rng is not None:
        # Add small noise to break ties and allow different orders
        totals_noisy = totals + rng.uniform(0, 0.1, size=n_jobs)
        sorted_jobs = np.argsort(-totals_noisy)
    else:
        sorted_jobs = np.argsort(-totals)
        
    if reverse:
        sorted_jobs = sorted_jobs[::-1]
        
    # Initialize the sequence with the first job
    seq = [sorted_jobs[0]]
    
    # Insert remaining jobs one by one
    for i in range(1, n_jobs):
        job = sorted_jobs[i]
        # Find the best position to insert the job
        best_pos = 0
        best_makespan = np.inf
        
        # Try all possible insertion positions
        for pos in range(len(seq) + 1):
            # Create a new sequence with the job inserted at pos
            new_seq = seq[:pos] + [job] + seq[pos:]
            # Calculate makespan for this new sequence
            ms = _makespan(pt, new_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
                
        # Insert the job at the best position
        seq.insert(best_pos, job)
        
    return seq


def _swap_mutation(order, rate=0.1, rng=None):
    """Apply swap mutation to the order."""
    n = len(order)
    mutated = order.copy()
    for i in range(n):
        if rng.random() < rate:
            j = rng.randint(0, n-1)
            mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


def _order_crossover(parent1, parent2, rng):
    """
    Order Crossover (OX) for permutations.
    """
    n = len(parent1)
    if n < 3:
        return parent1.copy()
        
    # Select two cut points
    c1, c2 = sorted(rng.sample(range(n), 2))
    
    # Initialize the child
    child = [None] * n
    
    # Copy the segment from parent1
    for i in range(c1, c2 + 1):
        child[i] = parent1[i]
        
    # Fill the rest with elements from parent2 in order, skipping those already in child
    pos = (c2 + 1) % n
    for val in parent2:
        if val not in child:
            child[pos] = val
            pos = (pos + 1) % n
            
    return child


def order_jobs(processing_times):
    # Ensure processing_times is a numpy array
    pt = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = pt.shape
    
    # Handle edge cases
    if n_jobs <= 1:
        return list(range(n_jobs))
        
    # Set a deadline
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initialize random number generator
    rng = np.random.default_rng()
    
    # Generate initial population using diverse NEH variants
    population = []
    
    # 1. Standard NEH
    pop1 = _neh_order(pt, reverse=False, randomize=False, rng=None)
    population.append(pop1)
    
    # 2. Reverse NEH
    pop2 = _neh_order(pt, reverse=True, randomize=False, rng=None)
    population.append(pop2)
    
    # 3. Randomized NEH
    for _ in range(3):
        pop_rand = _neh_order(pt, reverse=False, randomize=True, rng=rng)
        population.append(pop_rand)
        
    # 4. Random permutations
    for _ in range(5):
        pop_random = list(rng.permutation(n_jobs))
        population.append(pop_random)
        
    # Evaluate initial population
    best_order = None
    best_makespan = np.inf
    
    for order in population:
        ms = _makespan(pt, order)
        if ms < best_makespan:
            best_makespan = ms
            best_order = order.copy()
            
    # Genetic Algorithm loop
    pop_size = len(population)
    mutation_rate = 0.1
    elite_count = 2  # Keep the best 2 solutions
    
    while time.time() < deadline:
        # Selection and Crossover
        new_population = []
        
        # Elitism: Keep the best solutions
        # Sort population by makespan
        population_sorted = sorted(population, key=lambda x: _makespan(pt, x))
        for i in range(min(elite_count, pop_size)):
            new_population.append(population_sorted[i].copy())
            
        # Generate new individuals
        while len(new_population) < pop_size:
            # Select parents using tournament selection
            p1 = rng.choice(population)
            p2 = rng.choice(population)
            
            # Crossover
            child = _order_crossover(p1, p2, rng)
            
            # Mutation
            child = _swap_mutation(child, mutation_rate, rng)
            
            new_population.append(child)
            
        population = new_population
        
        # Evaluate new population and update best
        for order in population:
            ms = _makespan(pt, order)
            if ms < best_makespan:
                best_makespan = ms
                best_order = order.copy()
                
        # Adaptive mutation rate: increase if no improvement
        # For simplicity, we keep it constant, but could adjust based on stagnation
        
    # Final local search polish on the best order found
    # Try all pairwise swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_jobs):
            if time.time() > deadline:
                break
            for j in range(i + 1, n_jobs):
                # Swap jobs i and j
                best_order[i], best_order[j] = best_order[j], best_order[i]
                new_ms = _makespan(pt, best_order)
                if new_ms < best_makespan:
                    best_makespan = new_ms
                    improved = True
                else:
                    # Swap back
                    best_order[i], best_order[j] = best_order[j], best_order[i]
                    
    return list(best_order)
# EVOLVE-BLOCK-END
