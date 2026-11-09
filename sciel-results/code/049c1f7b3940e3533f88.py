"""Combined program: fast NEH variants, efficient local search, and a small GA with OX/insertion mutation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

def _makespan(order, P):
    """Compute makespan for a given job order on processing_times matrix P."""
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = P.shape[1]
    
    if n_machines == 0:
        return 0.0
        
    # Use list-based approach for small n_machines to reduce NumPy overhead
    if n_machines <= 20:
        first_job = order[0]
        row = P[first_job]
        curr_row = [0.0] * n_machines
        curr_row[0] = row[0]
        for m in range(1, n_machines):
            curr_row[m] = curr_row[m-1] + row[m]
            
        for idx in range(1, n_jobs):
            j = order[idx]
            row = P[j]
            new_row = [0.0] * n_machines
            new_row[0] = curr_row[0] + row[0]
            for m in range(1, n_machines):
                prev_val = curr_row[m]
                left_val = new_row[m-1]
                new_row[m] = (prev_val if prev_val > left_val else left_val) + row[m]
            curr_row = new_row
        
        return curr_row[-1]
    else:
        # Fallback to numpy for large n_machines
        first_job = order[0]
        curr_row = np.cumsum(P[first_job])
        for idx in range(1, n_jobs):
            j = order[idx]
            prev_row = curr_row
            new_row = np.empty(n_machines)
            new_row[0] = prev_row[0] + P[j, 0]
            if n_machines > 1:
                for m in range(1, n_machines):
                    new_row[m] = max(prev_row[m], new_row[m-1]) + P[j, m]
            curr_row = new_row
        return curr_row[-1]

def _neh_heuristic(P, sort_key='total', seed=None):
    """
    NEH heuristic for flow shop scheduling.
    """
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Step 1: Sort by specified key
    if sort_key == 'total':
        totals = P.sum(axis=1)
        if seed is not None:
            rng = np.random.RandomState(seed)
            noise = rng.uniform(0, 1e-6, n_jobs)
            totals = totals + noise
        sorted_indices = np.argsort(-totals)
    elif sort_key == 'first_last':
        key_vals = P[:, 0] + P[:, -1]
        if seed is not None:
            rng = np.random.RandomState(seed)
            noise = rng.uniform(0, 1e-6, n_jobs)
            key_vals = key_vals + noise
        sorted_indices = np.argsort(-key_vals)
    elif sort_key == 'random':
        if seed is not None:
            rng = np.random.RandomState(seed)
            sorted_indices = rng.permutation(n_jobs)
        else:
            sorted_indices = np.random.permutation(n_jobs)
    else:
        totals = P.sum(axis=1)
        sorted_indices = np.argsort(-totals)
        
    # Start with the first job
    current_seq = [int(sorted_indices[0])]
    
    # Insert remaining jobs one by one
    for k in range(1, n_jobs):
        job = int(sorted_indices[k])
        best_pos = -1
        best_makespan = float('inf')
        
        # Try all possible insertion positions (0 to len(current_seq))
        for pos in range(len(current_seq) + 1):
            # Create new sequence
            new_seq = current_seq[:pos] + [job] + current_seq[pos:]
            # Calculate makespan
            ms = _makespan(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        if best_pos == -1:
            current_seq.append(job)
        else:
            current_seq.insert(best_pos, job)
        
    return current_seq

def _local_search(order, P, deadline=None, max_passes=10):
    """
    Local search using adjacent swaps (2-opt) and then insertions if time permits.
    """
    n_jobs = len(order)
    if n_jobs < 2:
        return order
    
    current_order = list(order)
    current_ms = _makespan(current_order, P)
    
    # Phase 1: Adjacent Swaps
    for _pass in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(n_jobs - 1):
            if deadline is not None and time.time() > deadline:
                break
            
            # Try swapping adjacent jobs i and i+1
            new_order = current_order.copy()
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            
            new_ms = _makespan(new_order, P)
            if new_ms < current_ms:
                current_order = new_order
                current_ms = new_ms
                improved = True
        
        if not improved:
            break
            
    # Phase 2: Insertion Moves (if time permits)
    if deadline is not None and time.time() < deadline - 2.0:
        # Do a few passes of best-insertion
        for _pass in range(2): # Limit passes to keep it fast
            if deadline is not None and time.time() > deadline:
                break
            improved = False
            for i in range(n_jobs):
                if deadline is not None and time.time() > deadline:
                    break
                job = current_order[i]
                # Remove job
                current_order.remove(job)
                best_pos = i
                best_ms = current_ms # Initial ms is for the sequence with job at i
                
                # Try inserting at all positions
                # Note: We removed job, so length is n_jobs-1. Positions 0 to n_jobs-1
                for pos in range(len(current_order) + 1):
                    if deadline is not None and pos % 5 == 0:
                        if time.time() > deadline:
                            break
                    new_order = current_order[:pos] + [job] + current_order[pos:]
                    new_ms = _makespan(new_order, P)
                    if new_ms < best_ms:
                        best_ms = new_ms
                        best_pos = pos
                
                # Re-insert at best position
                current_order.insert(best_pos, job)
                if best_ms < current_ms:
                    current_ms = best_ms
                    improved = True
            
            if not improved:
                break

    return current_order

def _ox_crossover(p1, p2, rng):
    """Order Crossover (OX1)."""
    n = len(p1)
    if n < 2:
        return list(p1)
    
    # Choose two cut points
    i, j = sorted(rng.sample(range(n), 2))
    
    # Initialize child with None
    child = [None] * n
    
    # Copy segment from p1
    for k in range(i, j + 1):
        child[k] = p1[k]
    
    # Fill rest from p2 in order, skipping already placed
    placed = set(child[i:j+1])
    fill_idx = i
    for val in p2:
        if val not in placed:
            while child[fill_idx] is not None:
                fill_idx += 1
                if fill_idx >= n:
                    fill_idx = 0
            child[fill_idx] = val
            fill_idx += 1
            if fill_idx >= n:
                fill_idx = 0
            if fill_idx == i: # Full circle
                 # Should not happen if logic is correct, but break to avoid infinite loop
                 break
                 
    return child

def _mutate_insertion(perm, rng):
    """Insertion mutation: remove a job and insert at a random position."""
    n = len(perm)
    if n < 2:
        return list(perm)
    
    i = rng.randint(n)
    job = perm[i]
    new_perm = perm[:i] + perm[i+1:]
    j = rng.randint(len(new_perm) + 1)
    new_perm = new_perm[:j] + [job] + new_perm[j:]
    return new_perm

def _mutate_swap(perm, rng):
    """Swap mutation: swap two random jobs."""
    n = len(perm)
    if n < 2:
        return list(perm)
    
    i, j = sorted(rng.sample(range(n), 2))
    new_perm = list(perm)
    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
    return new_perm

def _genetic_algorithm(population, P, deadline, rng):
    """
    Simple Genetic Algorithm with OX and insertion/swap mutation.
    """
    n_jobs = P.shape[0]
    pop_size = len(population)
    if pop_size < 2:
        return population[0]
    
    # Evaluate initial population
    evals = [(perm, _makespan(perm, P)) for perm in population]
    
    # Sort by makespan
    evals.sort(key=lambda x: x[1])
    
    # Elitism: keep top 2
    elite = [evals[0][0], evals[1][0]]
    
    while time.time() < deadline:
        # Create new population
        new_pop = [list(elite[0]), list(elite[1])]
        
        # Fill rest with offspring
        while len(new_pop) < pop_size:
            # Tournament selection
            def select():
                idxs = rng.sample(range(len(evals)), 2)
                if evals[idxs[0]][1] < evals[idxs[1]][1]:
                    return evals[idxs[0]][0]
                else:
                    return evals[idxs[1]][0]
            
            p1 = select()
            p2 = select()
            
            child = _ox_crossover(p1, p2, rng)
            
            # Mutation
            if rng.random() < 0.3:
                child = _mutate_insertion(child, rng)
            elif rng.random() < 0.3:
                child = _mutate_swap(child, rng)
            
            new_pop.append(child)
        
        # Evaluate new population
        evals = [(perm, _makespan(perm, P)) for perm in new_pop]
        evals.sort(key=lambda x: x[1])
        
        # Update elites
        if evals[0][1] < _makespan(elite[0], P):
            elite[0] = list(evals[0][0])
        if evals[1][1] < _makespan(elite[1], P):
            elite[1] = list(evals[1][0])
            
        # Sort elites by fitness
        if _makespan(elite[0], P) > _makespan(elite[1], P):
            elite[0], elite[1] = elite[1], elite[0]
            
        # Check for convergence or time
        if time.time() > deadline - 5.0:
            break
            
    return elite[0]

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    Uses diverse NEH heuristics, local search, and a small GA.
    """
    P = np.asarray(processing_times)
    if P.size == 0:
        return []
        
    n_jobs = P.shape[0]
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Set a deadline with a safety margin
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Generate multiple candidates using different NEH variants
    candidates = []
    
    # 1. Standard NEH
    seq1 = _neh_heuristic(P, sort_key='total')
    candidates.append(seq1)
    
    # 2. NEH with first/last machine focus
    if time.time() < deadline - 10.0:
        seq2 = _neh_heuristic(P, sort_key='first_last')
        candidates.append(seq2)
        
    # 3. Randomized NEH
    if time.time() < deadline - 20.0 and n_jobs <= 200:
        for seed in [42, 123, 999]:
            if time.time() > deadline - 20.0:
                break
            seq_r = _neh_heuristic(P, sort_key='random', seed=seed)
            candidates.append(seq_r)

    # Find the best candidate
    best_seq = candidates[0]
    best_ms = _makespan(best_seq, P)
    
    for seq in candidates[1:]:
        if len(seq) < n_jobs:
            remaining = [j for j in range(n_jobs) if j not in seq]
            seq.extend(remaining)
        ms = _makespan(seq, P)
        if ms < best_ms:
            best_ms = ms
            best_seq = seq

    # Ensure best_seq is a valid permutation
    if len(best_seq) < n_jobs:
        remaining = [j for j in range(n_jobs) if j not in best_seq]
        best_seq.extend(remaining)
        
    # Apply local search if there is time
    if time.time() < deadline - 10.0:
        ls_deadline = deadline - 5.0
        best_seq = _local_search(best_seq, P, deadline=ls_deadline)
        best_ms = _makespan(best_seq, P)
    
    # Run GA if there is still time
    if time.time() < deadline - 15.0 and n_jobs <= 500:
        # Build population from best seq and its variants
        pop = [list(best_seq)]
        
        # Add some diverse individuals
        # Random permutations
        for _ in range(2):
            pop.append(np.random.permutation(n_jobs).tolist())
            
        # Perturbed best seq
        for _ in range(3):
            perturbed = list(best_seq)
            for _ in range(3):
                perturbed = _mutate_insertion(perturbed, np.random)
            pop.append(perturbed)
            
        # Ensure unique and valid
        seen = set()
        unique_pop = []
        for p in pop:
            if len(p) == n_jobs and sorted(p) == list(range(n_jobs)):
                key = tuple(p)
                if key not in seen:
                    seen.add(key)
                    unique_pop.append(p)
        
        if len(unique_pop) >= 2:
            ga_deadline = deadline - 2.0
            ga_best = _genetic_algorithm(unique_pop, P, ga_deadline, np.random)
            ga_ms = _makespan(ga_best, P)
            if ga_ms < best_ms:
                best_seq = ga_best
                best_ms = ga_ms
                
    return list(best_seq)

# EVOLVE-BLOCK-END
