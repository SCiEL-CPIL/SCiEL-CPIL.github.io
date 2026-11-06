"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math
import heapq

def _calc_makespan(processing_times, order):
    """
    Calculate the makespan for a given job order efficiently.
    processing_times: np.ndarray [n_jobs, n_machines]
    order: list of job indices
    """
    n_machines = processing_times.shape[1]
    if len(order) == 0:
        return 0.0
    
    # completion[m] is the time when machine m finishes the current job
    completion = np.zeros(n_machines, dtype=np.float64)
    
    for job in order:
        pt = processing_times[job]
        # Update completion times for each machine
        # C[m] = max(C[m], C[m-1]) + pt[m]
        # We need to be careful with in-place updates or use a temp variable
        prev_completion = 0.0
        for m in range(n_machines):
            # The machine cannot start before the previous machine finished the same job
            # AND before the machine finished the previous job
            start_time = max(completion[m], prev_completion)
            completion[m] = start_time + pt[m]
            prev_completion = completion[m]
            
    return float(completion[-1])

def _neh_initial(processing_times, rng=None, randomize=False):
    """
    NEH (Nawaz, Enscore, Ham) heuristic.
    Sort jobs by total processing time descending.
    Insert each job into the position that minimizes makespan.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    if rng is None:
        rng = random.Random(42)
        
    # Calculate total processing time for each job
    total_times = processing_times.sum(axis=1)
    
    # Sort indices by total time descending
    order = list(np.argsort(-total_times))
    
    # Optional randomization for diversity
    if randomize and n_jobs > 2:
        # Swap a few adjacent elements to create diversity
        for _ in range(min(3, n_jobs // 5)):
            i = rng.randint(0, n_jobs - 2)
            if rng.random() < 0.5:
                order[i], order[i+1] = order[i+1], order[i]
                
    seq = [order[0]]
    for job in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        # Try inserting job at each position in current seq
        for pos in range(len(seq) + 1):
            # Create trial sequence
            trial = seq[:pos] + [job] + seq[pos:]
            ms = _calc_makespan(processing_times, trial)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
        
    return seq

def _dp_exact_small(processing_times, max_jobs=15):
    """
    Exact Dynamic Programming for small instances.
    State: (mask, last_job) -> min makespan
    This is O(2^n * n^2) which is feasible for n <= 15.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs > max_jobs:
        return None, float('inf')
        
    # dp[mask][last] = min makespan for subset 'mask' ending with job 'last'
    # We use a dictionary or array. For n=15, 2^15 * 15 = 491k states.
    # Each state transition is O(n). Total O(2^n * n^2) ~ 7M ops. Very fast.
    
    INF = float('inf')
    # dp[mask][last]
    # To save memory, we can use a dict or a 2D array of size [1<<n][n]
    # 1<<15 is 32768. 32768 * 15 = ~500k floats. ~4MB. OK.
    
    size = 1 << n_jobs
    dp = np.full((size, n_jobs), INF, dtype=np.float64)
    parent = np.full((size, n_jobs), -1, dtype=np.int32)
    
    # Initialize single job states
    for j in range(n_jobs):
        mask = 1 << j
        # Makespan is just the sum of processing times for that job
        dp[mask][j] = processing_times[j].sum()
        
    # Iterate over all masks
    for mask in range(1, size):
        # Find set bits in mask
        # For each last job j in mask
        for j in range(n_jobs):
            if not (mask & (1 << j)):
                continue
            current_cost = dp[mask][j]
            if current_cost == INF:
                continue
                
            # Try adding a new job k not in mask
            remaining_mask = (size - 1) ^ mask
            if remaining_mask == 0:
                continue
                
            # Iterate through set bits of remaining_mask
            r = remaining_mask
            while r:
                lsb = r & (-r)
                k = int(math.log2(lsb))
                new_mask = mask | (1 << k)
                
                # Calculate new makespan
                # The completion time vector for 'mask' ending in 'j' is not stored.
                # We only stored the scalar makespan (C_last).
                # This is a problem: Flow shop makespan depends on the whole completion vector,
                # not just the last machine's completion time.
                # Wait, for flow shop, C_last is the makespan.
                # But to extend, we need the completion times on ALL machines.
                # Standard DP for Flow Shop requires storing the completion vector or using a different state representation.
                # Actually, the state (mask, last) is NOT sufficient for Flow Shop TSP-like DP because
                # the completion time on machine m depends on the history, not just the last job.
                # Unlike TSP where cost is additive, here the cost is determined by the max of sums.
                # So simple DP on (mask, last) does NOT work for Flow Shop Makespan.
                #
                # Correction: The "Exact DP" mentioned in the prompt's inspiration might refer to something else
                # or it was a flawed assumption in the "inspiration" code that got a score.
                # Let's check the inspiration code again.
                # Inspiration 1 had `_dp_exact_small`? No, it had `_makespan`, `_compute_prefix...`
                # The prompt says: "Exact DP for small n provides a globally informed starting point... Measured: 1.196 -> 1.213"
                # But the code snippet for inspiration 1 was cut off.
                # Let's look at the "CURRENT PROGRAM". It does NOT have exact DP.
                # The "INSPIRATIONS" text mentions "Exact DP for small n" in the knowledge section.
                # However, standard DP on subsets for Flow Shop is NOT O(2^n * n^2) with just last job.
                # It is NP-hard.
                # Maybe the "Exact DP" was actually a different method, like Branch and Bound?
                # Or maybe it stored the completion vector? Storing the vector makes the state space continuous/huge.
                #
                # Let's assume the "Exact DP" in the knowledge is a misnomer or refers to a specific
                # small-n solver that works differently, OR it's a trick.
                # Given the risk, I will NOT implement a flawed DP.
                # Instead, I will rely on the strong local search and SA from the current program,
                # but enhanced with the "incremental evaluation" idea from Inspiration 1 if possible,
                # or just optimize the current local search.
                
                # Let's stick to the robust local search + SA approach.
                # I will skip this DP function as it's likely incorrect for general flow shop
                # unless it's very small and uses a different state (e.g. storing the full vector in a hash map,
                # which is still exponential in a bad way).
                
                pass
            r ^= lsb
            
    return None, INF

def _insertion_local_search(processing_times, seq, time_limit, deadline=None):
    """
    Local search using insertion moves.
    Remove a job, try inserting it at all other positions, keep best.
    Optimized with early termination and incremental checks if possible.
    """
    n_jobs = len(seq)
    if n_jobs <= 1:
        return seq, _calc_makespan(processing_times, seq)
        
    current_ms = _calc_makespan(processing_times, seq)
    best_ms = current_ms
    best_seq = list(seq)
    
    start_time = time.time()
    
    # We iterate until no improvement or time limit
    while True:
        if time.time() > deadline:
            break
            
        improved = False
        # First Improvement or Best Improvement?
        # Best Improvement is better for convergence but slower per pass.
        # For small n, Best Improvement is fine.
        
        best_move_delta = 0
        best_move_i = -1
        best_move_j = -1
        
        for i in range(n_jobs):
            job = best_seq[i]
            # Remove job
            seq_reduced = best_seq[:i] + best_seq[i+1:]
            
            # Try inserting at all positions in reduced sequence
            for pos in range(n_jobs - 1):
                if pos == i:
                    continue # Same position, skip? No, i is removed, so pos in reduced is different.
                    # If we remove index i, the new list has length n-1.
                    # Inserting at pos in the new list.
                    # If pos == i, it's the same relative position?
                    # Actually, if we remove i and insert at i, it's the same sequence.
                    # So we can skip pos == i.
                    # But careful with indices.
                    # If i=0, remove 0. Insert at 0 is same.
                    # If i=1, remove 1. Insert at 1 is same as original?
                    # Original: [A, B, C]. Remove B (i=1) -> [A, C]. Insert B at 1 -> [A, B, C]. Yes.
                    # So skip pos == i.
                    continue

                trial = seq_reduced[:pos] + [job] + seq_reduced[pos:]
                ms = _calc_makespan(processing_times, trial)
                delta = ms - best_ms
                
                if delta < best_move_delta - 1e-9:
                    best_move_delta = delta
                    best_move_i = i
                    best_move_j = pos
                    
        if best_move_i != -1:
            # Apply best move
            job = best_seq[best_move_i]
            seq_reduced = best_seq[:best_move_i] + best_seq[best_move_i+1:]
            best_seq = seq_reduced[:best_move_j] + [job] + seq_reduced[best_move_j:]
            best_ms += best_move_delta
            improved = True
            
        if not improved:
            break
            
    return best_seq, best_ms

def _simulated_annealing(processing_times, seq, time_limit, deadline=None, rng=None):
    """
    Simulated Annealing to escape local minima.
    """
    if rng is None:
        rng = random.Random()
        
    n_jobs = len(seq)
    if n_jobs <= 1:
        return seq, _calc_makespan(processing_times, seq)
        
    current_seq = list(seq)
    current_ms = _calc_makespan(processing_times, current_seq)
    best_seq = list(current_seq)
    best_ms = current_ms
    
    # Initial temperature: roughly 10% of makespan
    temperature = max(current_ms * 0.1, 1.0)
    cooling_rate = 0.995
    
    start_time = time.time()
    
    while time.time() < deadline and temperature > 1e-5:
        # Generate a neighbor
        # Random swap of two jobs
        i = rng.randint(0, n_jobs - 1)
        j = rng.randint(0, n_jobs - 1)
        if i == j:
            continue
            
        new_seq = list(current_seq)
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        new_ms = _calc_makespan(processing_times, new_seq)
        
        delta = new_ms - current_ms
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or rng.random() < math.exp(-delta / temperature):
            current_seq = new_seq
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_seq = list(current_seq)
                
        temperature *= cooling_rate
        
    return best_seq, best_ms

def order_jobs(processing_times):
    start_time = time.time()
    # Safety margin: 120s limit, leave 5s for safety
    deadline = start_time + 115.0
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    rng = random.Random(42)
    
    best_perm = None
    best_ms = float('inf')
    
    # Phase 1: Generate diverse initial solutions using NEH variants
    # We will run a few NEH variants to find a good starting point
    
    # Variant 1: Standard NEH
    if time.time() < deadline - 10.0:
        perm1 = _neh_initial(processing_times, rng, randomize=False)
        ms1 = _calc_makespan(processing_times, perm1)
        if ms1 < best_ms:
            best_ms = ms1
            best_perm = perm1
            
    # Variant 2: NEH with slight randomization
    if time.time() < deadline - 10.0:
        perm2 = _neh_initial(processing_times, rng, randomize=True)
        ms2 = _calc_makespan(processing_times, perm2)
        if ms2 < best_ms:
            best_ms = ms2
            best_perm = perm2
            
    # Variant 3: NEH with different random seed
    if time.time() < deadline - 10.0:
        rng3 = random.Random(12345)
        perm3 = _neh_initial(processing_times, rng3, randomize=True)
        ms3 = _calc_makespan(processing_times, perm3)
        if ms3 < best_ms:
            best_ms = ms3
            best_perm = perm3
            
    # Phase 2: Local Search (Insertion)
    # This is the most critical part for improvement
    if time.time() < deadline - 5.0 and best_perm is not None:
        ls_deadline = deadline - 2.0
        best_perm, best_ms = _insertion_local_search(processing_times, best_perm, time_limit=10.0, deadline=ls_deadline)
        
    # Phase 3: Simulated Annealing for diversification
    # If we have time left, use SA to try to jump out of local minima
    if time.time() < deadline - 2.0 and best_perm is not None:
        sa_deadline = deadline - 1.0
        rng_sa = random.Random(99999)
        sa_perm, sa_ms = _simulated_annealing(processing_times, best_perm, time_limit=5.0, deadline=sa_deadline, rng=rng_sa)
        if sa_ms < best_ms:
            best_ms = sa_ms
            best_perm = sa_perm
            
            # Polish the SA result with local search if time permits
            if time.time() < deadline - 0.5:
                best_perm, best_ms = _insertion_local_search(processing_times, best_perm, time_limit=2.0, deadline=deadline - 0.1)
                
    return best_perm

# EVOLVE-BLOCK-END
