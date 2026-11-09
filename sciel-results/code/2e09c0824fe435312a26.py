"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(perm, pt):
    """
    Compute the makespan for a given permutation.
    Uses the standard flow shop recurrence:
    C[j, k] = max(C[j-1, k], C[j, k-1]) + p[j, k]
    """
    if len(perm) == 0:
        return 0.0
    n = len(perm)
    m = pt.shape[1]
    
    # C_prev[k] is the completion time of the previous job on machine k
    C_prev = np.zeros(m)
    
    for j in perm:
        C_curr = np.zeros(m)
        for k in range(m):
            if k == 0:
                C_curr[0] = C_prev[0] + pt[j, 0]
            else:
                C_curr[k] = max(C_prev[k], C_curr[k-1]) + pt[j, k]
        C_prev = C_curr
        
    return C_prev[-1]

def _fast_makespan_delta(perm, pt, i, j):
    """
    Calculate the new makespan after swapping positions i and j in perm.
    This is an O(n*m) operation, but often faster than full recomputation if we can short-circuit,
    though for simplicity and robustness in Python, we might just recompute.
    However, to make Tabu Search fast enough, we need a better approach or very tight loops.
    
    Actually, a full O(n*m) recompute in pure Python is ~1000 operations for n=100, m=10.
    10k iterations * 1000 ops = 10M ops. Python can do ~10M-50M simple ops/sec.
    So 10k iterations might take 1-5 seconds. This is acceptable for a 110s budget.
    
    Let's implement a vectorized-ish recompute to ensure speed.
    """
    n = len(perm)
    m = pt.shape[1]
    
    # We need to compute the makespan of the new permutation.
    # Instead of creating a new list (copy overhead), we can swap in place, compute, and swap back.
    # But since we are evaluating many candidates, we should be careful.
    
    # Let's just do a fast loop.
    C_prev = np.zeros(m)
    
    # We iterate through the permutation, but with i and j swapped.
    # To avoid list copying, we can map indices.
    
    # Actually, creating a new list is O(n). The makespan calc is O(n*m).
    # Let's just do the standard loop but optimized.
    
    # Pre-extract the processing times of the swapped jobs to avoid indexing pt[j, k] repeatedly?
    # No, pt is a numpy array, indexing is fast.
    
    # Let's write a tight loop.
    for idx in range(n):
        # Determine which job is at this position
        if idx == i:
            job = perm[j]
        elif idx == j:
            job = perm[i]
        else:
            job = perm[idx]
            
        C_curr = np.zeros(m)
        # Unroll first machine
        C_curr[0] = C_prev[0] + pt[job, 0]
        
        for k in range(1, m):
            # C_curr[k] = max(C_prev[k], C_curr[k-1]) + pt[job, k]
            val1 = C_prev[k]
            val2 = C_curr[k-1]
            if val1 > val2:
                C_curr[k] = val1 + pt[job, k]
            else:
                C_curr[k] = val2 + pt[job, k]
        C_prev = C_curr
        
    return C_prev[-1]

def _neh_heuristic(pt):
    """
    NEH heuristic for flow shop.
    """
    n, m = pt.shape
    if n <= 1:
        return list(range(n))
        
    totals = pt.sum(axis=1)
    order = np.argsort(-totals)
    
    current_perm = [order[0]]
    
    for idx in range(1, n):
        job = order[idx]
        best_pos = 0
        best_mk = float('inf')
        
        for pos in range(len(current_perm) + 1):
            new_perm = current_perm[:pos] + [job] + current_perm[pos:]
            mk = _makespan(new_perm, pt)
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
                
        current_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
        
    return current_perm

def _tabu_search(perm, pt, deadline):
    """
    Tabu Search for flow shop scheduling.
    Neighborhood: Swap any two jobs.
    Tabu List: Store (i, j) pairs or (job_i, job_j) pairs.
    Aspiration Criterion: Accept tabu move if it improves the global best.
    """
    n = len(perm)
    if n <= 1:
        return perm
    
    current_perm = list(perm)
    current_mk = _makespan(current_perm, pt)
    
    best_perm = current_perm[:]
    best_mk = current_mk
    
    # Tabu list parameters
    tabu_tenure = max(5, n // 5)  # Dynamic tenure
    tabu_list = {}  # key: (job_a, job_b), value: iteration when it expires
    
    # To speed up neighbor generation, we can pre-calculate some things, 
    # but swap neighborhood is n*(n-1)/2. For n=100, that's ~5000 neighbors.
    # Evaluating 5000 neighbors * O(n*m) makespan.
    # If n=100, m=10, makespan is ~1000 ops. 5000 * 1000 = 5M ops per iteration.
    # Python: ~1-5 seconds per iteration.
    # 110s / 3s = ~30 iterations. This might be too slow if we do FULL scan.
    
    # Optimization: Use a random subset of neighbors or a faster evaluation?
    # Let's try a hybrid: Best-improvement on a random subset of neighbors to keep it fast,
    # or First-improvement on a random shuffle of neighbors.
    # First-improvement is usually faster per iteration and allows more iterations.
    
    # Let's use First-Improvement with a random order of moves.
    # This allows us to escape local optima more easily if we have time.
    
    no_improve_count = 0
    max_no_improve = 500  # Restart if stuck
    
    while time.time() < deadline:
        # Generate a random permutation of all possible swap moves
        # For n=100, 4950 moves. Shuffling is O(n^2).
        # We can just pick random moves and check if they are tabu.
        
        improved = False
        # Try up to 1000 random moves per iteration to find an improvement
        # If we find one, we apply it and break to next iteration.
        
        # To ensure we don't miss the best move, we could do a limited scan.
        # Let's do a "First Improvement" strategy on a random sample of neighbors.
        
        # Actually, for small n, full scan is fine. For large n, random sample.
        if n < 30:
            moves = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(moves)
        else:
            # Sample 1000 random moves
            moves = []
            for _ in range(1000):
                i, j = random.sample(range(n), 2)
                if i > j: i, j = j, i
                moves.append((i, j))
        
        for i, j in moves:
            # Check tabu status
            # We store moves as (job_at_i, job_at_j) to be invariant to position changes?
            # Or (i, j) positions?
            # Standard Tabu Search often uses (i, j) positions or (job, job) pairs.
            # Using (job, job) is more robust to position shifts.
            job_a = current_perm[i]
            job_b = current_perm[j]
            move_key = (job_a, job_b) if job_a < job_b else (job_b, job_a)
            
            current_iter = len(tabu_list) # Simple counter proxy, or use actual iter count
            # Let's use a global iteration counter
            if move_key in tabu_list and tabu_list[move_key] > 0:
                # It is tabu
                # Aspiration: if this move leads to a new global best, accept it
                # But we don't know the makespan yet without calculating.
                # We can calculate, and if it's < best_mk, accept even if tabu.
                pass # Proceed to calculate, check aspiration later
            
            # Calculate new makespan
            new_mk = _fast_makespan_delta(current_perm, pt, i, j)
            
            # Check Aspiration
            is_tabu = (move_key in tabu_list and tabu_list[move_key] > 0)
            
            if new_mk < current_mk:
                # Improvement
                if not is_tabu or new_mk < best_mk:
                    # Accept
                    current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
                    current_mk = new_mk
                    
                    if current_mk < best_mk:
                        best_mk = current_mk
                        best_perm = current_perm[:]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        
                    # Update Tabu List
                    # Add this move to tabu list for 'tabu_tenure' iterations
                    # We need a global iteration counter. Let's just use a simple dict update.
                    # To avoid complexity, we'll store the expiration time as "current_time + tenure"
                    # But "iterations" is better. Let's maintain a counter.
                    # Since we are in a loop, let's just increment a counter outside.
                    
                    # For simplicity in this block, let's use a list of moves to be tabu for next T iterations.
                    # We'll manage the tabu list outside this inner loop.
                    
                    # Update tabu list: mark this move as tabu
                    # We need to know the current iteration number.
                    # Let's restructure: use a global `iter_count`.
                    
                    # Since I can't easily pass iter_count here without refactoring,
                    # I will use a simple heuristic: store the move in a list, and check age.
                    # Actually, let's just use a dictionary {move_key: iter_count + tenure}
                    # I'll define iter_count in the outer scope.
                    
                    # Let's assume `iter_count` is available.
                    # I will modify the code structure to include `iter_count`.
                    
                    # For now, let's just break to next outer iteration after one move.
                    # This is "First Improvement" strategy.
                    improved = True
                    break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Diversification: random perturbation
                # Swap 2-5 random jobs
                k = random.randint(2, min(5, n))
                indices = random.sample(range(n), k)
                # Shuffle them? Or just swap pairs?
                # Let's do a random permutation of these k positions
                jobs_to_swap = [current_perm[idx] for idx in indices]
                random.shuffle(jobs_to_swap)
                for idx, job in zip(indices, jobs_to_swap):
                    current_perm[idx] = job
                
                current_mk = _makespan(current_perm, pt)
                no_improve_count = 0
                # Clear tabu list to allow exploration
                tabu_list.clear()
        
        # Advance iteration counter (pseudo-code, need to implement properly)
        # The tabu_list logic above is a bit loose. Let's refine the main loop.

    return best_perm

def order_jobs(processing_times):
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))
        
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Initial Solution via NEH
    current_perm = _neh_heuristic(processing_times)
    current_mk = _makespan(current_perm, processing_times)
    best_perm = current_perm[:]
    best_mk = current_mk
    
    # 2. Tabu Search
    # We need a proper Tabu Search implementation with a global iteration counter.
    
    # Let's rewrite the tabu search logic inline to manage state correctly.
    
    tabu_tenure = max(5, n // 5)
    tabu_list = {} # key: (job_a, job_b), value: iter_to_expire
    iter_count = 0
    no_improve_count = 0
    max_no_improve = 1000 # Allow more iterations before restart
    
    while time.time() < deadline - 1.0:
        iter_count += 1
        
        # Generate moves
        if n < 30:
            moves = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(moves)
        else:
            # Sample moves to keep it fast
            num_samples = min(2000, n * (n-1) // 2)
            moves = []
            for _ in range(num_samples):
                i, j = random.sample(range(n), 2)
                if i > j: i, j = j, i
                moves.append((i, j))
        
        best_move_found = False
        best_new_mk = float('inf')
        best_move_i, best_move_j = -1, -1
        best_move_key = None
        
        # First Improvement Strategy:
        # Find the first non-tabu improving move, or a tabu move that beats global best.
        
        for i, j in moves:
            job_a = current_perm[i]
            job_b = current_perm[j]
            move_key = (job_a, job_b) if job_a < job_b else (job_b, job_a)
            
            is_tabu = (move_key in tabu_list and tabu_list[move_key] > iter_count)
            
            # Calculate makespan
            new_mk = _fast_makespan_delta(current_perm, pt, i, j)
            
            # Acceptance Criteria
            if new_mk < current_mk:
                if not is_tabu:
                    # Non-tabu improvement
                    best_move_found = True
                    best_new_mk = new_mk
                    best_move_i, best_move_j = i, j
                    best_move_key = move_key
                    break # First improvement
                elif new_mk < best_mk:
                    # Aspiration: Tabu move but improves global best
                    best_move_found = True
                    best_new_mk = new_mk
                    best_move_i, best_move_j = i, j
                    best_move_key = move_key
                    break # First improvement (aspiration)
            elif new_mk < best_mk and not is_tabu:
                # Even if it doesn't improve current, if it improves global best and is not tabu?
                # Standard Tabu usually requires improvement over current to accept, 
                # unless it's a specific variant. Let's stick to current improvement for stability.
                pass
        
        if best_move_found:
            # Apply move
            current_perm[best_move_i], current_perm[best_move_j] = current_perm[best_move_j], current_perm[best_move_i]
            current_mk = best_new_mk
            
            if current_mk < best_mk:
                best_mk = current_mk
                best_perm = current_perm[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Update Tabu List
            tabu_list[best_move_key] = iter_count + tabu_tenure
        else:
            # No improving move found in this iteration
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Diversification: Random Perturbation
                # Swap 2-5 random jobs
                k = random.randint(2, min(5, n))
                indices = random.sample(range(n), k)
                jobs_to_swap = [current_perm[idx] for idx in indices]
                random.shuffle(jobs_to_swap)
                for idx, job in zip(indices, jobs_to_swap):
                    current_perm[idx] = job
                
                current_mk = _makespan(current_perm, processing_times)
                no_improve_count = 0
                # Clear tabu list to allow exploration of this new region
                tabu_list.clear()
                
                # Optionally, if the perturbed solution is worse than best, 
                # we might want to revert to best and perturb again?
                # Or just continue from here.
                # Let's continue from here.

    return list(best_perm)

# EVOLVE-BLOCK-END
