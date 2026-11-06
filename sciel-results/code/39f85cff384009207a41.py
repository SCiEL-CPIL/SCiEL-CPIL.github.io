"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(order, processing_times):
    """Compute makespan for a given job order. O(n*m)."""
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return 0.0
    if len(order) == 0:
        return 0.0
    # completion[m] = completion time of the last processed job on machine m
    completion = np.zeros(n_machines)
    for j in order:
        row = processing_times[j]
        prev = completion[0]
        for m in range(n_machines):
            prev = prev + row[m]
            completion[m] = prev
    return completion[-1]


def _neh_construct(processing_times):
    """
    NEH (Nawaz, Enscore, Ham) heuristic.
    1. Sort jobs by decreasing total processing time.
    2. Start with the first job.
    3. Insert each subsequent job into the position that minimizes the makespan.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    seq = [int(sorted_jobs[0])]
    for k in range(1, n_jobs):
        job = int(sorted_jobs[k])
        best_pos = 0
        best_ms = np.inf
        for pos in range(len(seq) + 1):
            # Create a temporary sequence
            # To avoid list copy overhead, we can just evaluate
            # But for simplicity and correctness first, we use a helper or inline logic
            # Since N is likely small enough for O(N^2) construction with O(NM) eval, 
            # total construction is O(N^3 M). For N=100, M=10, this is 10^6 * 10 = 10^7 ops. Acceptable.
            
            # Optimization: We can evaluate insertion without full list creation if we track completion times,
            # but simple list insertion + _makespan is easier to implement correctly.
            new_seq = seq[:pos] + [job] + seq[pos:]
            ms = _makespan(new_seq, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        seq.insert(best_pos, job)
        
    return seq


def _hill_climb_adjacent_swaps(seq, processing_times, time_limit, rng):
    """
    Hill Climbing using adjacent swaps.
    Neighborhood: Swap job at i with job at i+1.
    """
    n = len(seq)
    if n <= 1:
        return seq, _makespan(seq, processing_times)
    
    cur_seq = np.array(seq, dtype=int)
    cur_ms = _makespan(cur_seq, processing_times)
    best_seq = cur_seq.copy()
    best_ms = cur_ms
    
    start_time = time.time()
    
    # We will perform iterations until time runs out.
    # Each iteration tries all adjacent swaps, picks the best improving move.
    # If no improving move is found, we break (local optimum).
    # To avoid getting stuck, we could add restarts, but let's stick to one deep search first 
    # or simple random restarts if time permits.
    
    while time.time() - start_time < time_limit:
        improved = False
        best_move_ms = cur_ms
        best_move_i = -1
        
        for i in range(n - 1):
            # Swap i and i+1
            cur_seq[i], cur_seq[i+1] = cur_seq[i+1], cur_seq[i]
            ms = _makespan(cur_seq, processing_times)
            # Undo swap
            cur_seq[i], cur_seq[i+1] = cur_seq[i+1], cur_seq[i]
            
            if ms < best_move_ms - 1e-9:
                best_move_ms = ms
                best_move_i = i
        
        if best_move_i != -1:
            # Apply the best improving move
            cur_seq[best_move_i], cur_seq[best_move_i+1] = cur_seq[best_move_i+1], cur_seq[best_move_i]
            cur_ms = best_move_ms
            improved = True
            
            if cur_ms < best_ms - 1e-9:
                best_ms = cur_ms
                best_seq = cur_seq.copy()
        else:
            # No improving move found, we are at a local optimum
            break
            
    return best_seq.tolist(), best_ms


def order_jobs(processing_times):
    processing_times = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_machines == 0:
        return list(range(n_jobs))
    if n_jobs == 1:
        return [0]

    deadline = time.time() + 110.0  # safety margin under 120s
    rng = np.random.default_rng(12345)

    # Initial solution: NEH heuristic
    best_seq = _neh_construct(processing_times)
    best_ms = _makespan(best_seq, processing_times)
    
    # Run Hill Climbing for the remaining time
    # We will run multiple restarts if time allows to escape local optima.
    # But first, refine the initial solution.
    
    remaining_time = deadline - time.time()
    if remaining_time > 0:
        # Split time: 50% for initial refinement, 50% for restarts?
        # Or just run hill climbing until time is up?
        # Hill climbing stops at local optimum. So we should do random restarts.
        
        # Strategy: 
        # 1. Refine NEH solution.
        # 2. If time left, generate random perturbations and refine them.
        
        # Refine initial
        t_refine = min(remaining_time * 0.5, 10.0) # Cap initial refine to 10s to allow restarts
        if t_refine > 0:
            best_seq, best_ms = _hill_climb_adjacent_swaps(best_seq, processing_times, t_refine, rng)
            
        # Restart loop
        while time.time() < deadline - 5.0: # Leave 5s safety margin
            # Generate a random perturbation of the current best
            # Or generate a completely new random solution?
            # Perturbation is usually better.
            # Let's do a random swap perturbation.
            perturbed_seq = best_seq.copy()
            for _ in range(3): # 3 random swaps
                i = rng.integers(0, n_jobs)
                j = rng.integers(0, n_jobs)
                perturbed_seq[i], perturbed_seq[j] = perturbed_seq[j], perturbed_seq[i]
            
            # Refine the perturbed solution
            # Allocate small time for each restart
            t_restart = min(1.0, (deadline - 5.0 - time.time()))
            if t_restart <= 0:
                break
                
            seq, ms = _hill_climb_adjacent_swaps(perturbed_seq, processing_times, t_restart, rng)
            
            if ms < best_ms - 1e-9:
                best_ms = ms
                best_seq = seq

    return list(best_seq)
# EVOLVE-BLOCK-END
