"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _calc_makespan(processing_times, perm):
    """
    Calculate makespan for a given permutation efficiently.
    Uses a tight Python loop which is often faster than numpy overhead for small N.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return 0.0
        
    # completion[m] is the completion time of the last job on machine m
    completion = np.zeros(n_machines)
    
    for j in perm:
        row = processing_times[j]
        prev_completion = 0.0
        for m in range(n_machines):
            if m == 0:
                new_c = completion[0] + row[0]
            else:
                if completion[m] > prev_completion:
                    new_c = completion[m] + row[m]
                else:
                    new_c = prev_completion + row[m]
            
            completion[m] = new_c
            prev_completion = new_c
            
    return float(completion[-1])

def _v_shape_heuristic(processing_times):
    """
    V-shaped heuristic for initial solution.
    Jobs are sorted by total processing time.
    Jobs with high totals are placed at the beginning or end, alternating.
    """
    n_jobs = processing_times.shape[0]
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    left = []
    right = []
    # Alternate placing jobs in left and right lists
    # Usually, we place the most critical jobs at the ends.
    # A common V-shape construction:
    # 1. Sort jobs by total processing time (descending).
    # 2. For each job, decide whether to put it at the front or back of the current sequence.
    #    Simple heuristic: alternate, or use NEH-like insertion but restricted to ends?
    #    Actually, the classic V-shape (Pinedo) suggests that the optimal sequence is V-shaped:
    #    first part non-decreasing, second part non-increasing in some metric.
    #    A simpler constructive heuristic often used:
    #    Sort by total processing time.
    #    Place jobs alternately at the beginning and end of the sequence.
    
    seq = []
    left_idx = 0
    right_idx = n_jobs - 1
    
    # We will build the sequence from the outside in.
    # Let's just use a simple alternating insertion into a list.
    # But to be efficient, we can build two lists and concatenate.
    
    # Let's try: Sort descending. 
    # Place 1st at start. 2nd at end. 3rd at start (after 1st). 4th at end (before 2nd).
    # This creates a V-shape if the totals are roughly monotonic.
    
    # Actually, a better V-shape heuristic is to sort by total processing time,
    # then use a rule to assign to left or right.
    # Let's stick to a robust NEH-like but with a bias towards ends?
    # No, the prompt asks for "V-shaped heuristic".
    # Let's implement the standard V-shape:
    # 1. Calculate total processing time for each job.
    # 2. Sort jobs in descending order of total processing time.
    # 3. Construct the sequence by taking jobs from the sorted list and placing them
    #    alternately at the beginning and the end of the current sequence.
    
    current_seq = []
    left_add = 0
    right_add = 0
    # We need to maintain the actual list.
    # To make it O(N^2) which is fine for small N, we can just insert.
    # But for large N, we should be careful.
    # Let's use a list and insert at 0 or end.
    
    for idx, job in enumerate(sorted_jobs):
        if idx % 2 == 0:
            current_seq.insert(0, job)
        else:
            current_seq.append(job)
            
    return current_seq

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # 1. Initial Construction: V-shaped Heuristic
    current_perm = _v_shape_heuristic(processing_times)
    current_ms = _calc_makespan(processing_times, current_perm)
    
    best_perm = current_perm[:]
    best_ms = current_ms
    
    # 2. Tabu Search
    # Parameters
    tabu_tenure = max(10, n_jobs // 2)  # Dynamic tenure
    max_iterations = 100000
    no_improve_count = 0
    patience = 2000  # Restart if no improvement for this many steps
    
    # Tabu list: store (i, j) pairs of swaps or moves
    # To prevent cycling, we store the move that generated the current solution.
    # For a swap of positions i and j, the reverse move is also (i, j).
    # We use a dictionary or set of tuples.
    # To be more effective, we can use a list of last K moves.
    
    tabu_moves = []
    # To allow quick lookup, we could use a set, but we need to expire old ones.
    # A list of (move, expiration_step) is standard.
    
    step = 0
    last_improvement_step = 0
    
    while time.time() < deadline and step < max_iterations:
        # Check for stagnation
        if step - last_improvement_step > patience:
            # Diversification: random restart or large perturbation
            # Let's do a random restart but keep the best found so far
            # Or just perturb the current solution significantly.
            # Perturbation: randomly swap 5% of jobs
            n_swaps = max(1, n_jobs // 20)
            for _ in range(n_swaps):
                i = random.randint(0, n_jobs - 1)
                j = random.randint(0, n_jobs - 1)
                if i != j:
                    current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
            current_ms = _calc_makespan(processing_times, current_perm)
            # Reset tabu list
            tabu_moves = []
            last_improvement_step = step
            continue
        
        # Generate neighbors
        # We will evaluate a sample of swaps (2-opt is too expensive for full scan in TS usually)
        # Or we can do best-improvement tabu search with a limited neighborhood.
        # For speed, let's sample random swaps.
        # If n_jobs is small, we could do full, but sampling is safer for time.
        
        best_candidate_perm = None
        best_candidate_ms = float('inf')
        best_candidate_move = None
        
        n_samples = 50  # Number of random swaps to try per iteration
        
        for _ in range(n_samples):
            i = random.randint(0, n_jobs - 1)
            j = random.randint(0, n_jobs - 1)
            if i == j:
                continue
            
            move = (min(i, j), max(i, j))
            
            # Check tabu
            is_tabu = False
            for m, exp in tabu_moves:
                if m == move and step < exp:
                    is_tabu = True
                    break
            
            # Aspiration criterion: if the move leads to a solution better than best so far, allow it
            if is_tabu and current_ms >= best_ms:
                continue
            
            # Apply move
            candidate_perm = current_perm[:]
            candidate_perm[i], candidate_perm[j] = candidate_perm[j], candidate_perm[i]
            
            candidate_ms = _calc_makespan(processing_times, candidate_perm)
            
            if candidate_ms < best_candidate_ms:
                best_candidate_ms = candidate_ms
                best_candidate_perm = candidate_perm
                best_candidate_move = move
        
        # Accept the best candidate
        if best_candidate_perm is not None:
            # Update tabu list
            tabu_moves.append((best_candidate_move, step + tabu_tenure))
            # Clean up old tabu moves
            while tabu_moves and tabu_moves[0][1] <= step:
                tabu_moves.pop(0)
            
            # Check for improvement
            if best_candidate_ms < current_ms:
                current_ms = best_candidate_ms
                current_perm = best_candidate_perm
                last_improvement_step = step
                
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_perm = current_perm[:]
            else:
                # Non-improving move (allowed by tabu search to escape local minima)
                current_ms = best_candidate_ms
                current_perm = best_candidate_perm
        
        step += 1
        
        # Dynamic tenure adjustment
        if step % 100 == 0:
            # If stagnating, increase tenure to force diversification?
            # Or decrease to allow more moves?
            # Let's keep it simple.
            pass

    return best_perm
# EVOLVE-BLOCK-END
