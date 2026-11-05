"""Flow-shop permutation via fast heuristics, VND, and Simulated Annealing with Tabu."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(pt, pt_rows, perm):
    """
    Compute makespan for a given permutation.
    Optimized for small number of machines.
    """
    n_jobs, n_machines = pt.shape
    if n_machines == 0 or n_jobs == 0:
        return 0.0
    
    # Use Python lists for speed in the inner loop
    # prev_comp[m] is the completion time of the previous job on machine m
    prev_comp = [0.0] * n_machines
    
    for j in perm:
        row = pt_rows[j]
        # curr_comp[m] is the completion time of current job on machine m
        # We can compute this in-place if we are careful, but we need the previous 
        # job's completion times for the 'max' part.
        # Actually, we need prev_comp[m] and curr_comp[m-1].
        # We can store curr_comp in a temporary list or reuse.
        
        # Optimization: 
        # C_jm = max(C_{j-1,m}, C_{j,m-1}) + P_jm
        # We iterate m from 0 to M-1.
        # We need C_{j,m-1} (which is the last calculated value for current job)
        # and C_{j-1,m} (which is prev_comp[m]).
        
        curr_val = 0.0
        for m in range(n_machines):
            # C_{j,m-1} is curr_val (0 for m=0)
            # C_{j-1,m} is prev_comp[m]
            if m == 0:
                curr_val = prev_comp[0] + row[0]
            else:
                # max(prev_comp[m], curr_val)
                if prev_comp[m] > curr_val:
                    curr_val = prev_comp[m] + row[m]
                else:
                    curr_val = curr_val + row[m]
            
            # Store current completion time for this machine
            # We will update prev_comp after the job is done? No, we need 
            # to keep prev_comp intact until the whole job is processed?
            # Actually, C_{j,m} depends on C_{j-1,m}. 
            # So we need the OLD prev_comp[m] for the max calculation.
            # So we cannot overwrite prev_comp[m] until we've used it for this job.
            # But we only use prev_comp[m] once per job (for machine m).
            # So we can update a new list or just store the result.
            
        # Wait, the loop above calculates curr_val sequentially.
        # We need to update prev_comp to be the completion times of THIS job 
        # for the NEXT job.
        # So we need a new array for curr_comp to update prev_comp at the end?
        # Or we can store the results in a temporary list.
        
        # Let's refine:
        # prev_comp holds C_{j-1, 0..M-1}
        # We calculate C_{j, 0..M-1} into a temp list or variable.
        # Then set prev_comp = C_j.
        
        # Let's rewrite the inner loop properly.
        
    # The previous implementation in the prompt was:
    # prev = [0.0] * n_machines
    # for j in perm:
    #   row = pt_rows[j]
    #   curr = [0.0] * n_machines
    #   curr[0] = prev[0] + row[0]
    #   for m in range(1, n_machines):
    #       val = prev[m] if prev[m] > curr[m-1] else curr[m-1]
    #       curr[m] = val + row[m]
    #   prev = curr
    # return prev[-1]
    
    # This is correct. Let's optimize it by avoiding list creation if possible?
    # We can use two lists and swap them.
    
    n_jobs, n_machines = pt.shape
    if n_machines == 0 or n_jobs == 0:
        return 0.0
        
    prev = [0.0] * n_machines
    curr = [0.0] * n_machines
    
    for j in perm:
        row = pt_rows[j]
        curr[0] = prev[0] + row[0]
        for m in range(1, n_machines):
            # max(prev[m], curr[m-1])
            p = prev[m]
            c = curr[m-1]
            if p > c:
                curr[m] = p + row[m]
            else:
                curr[m] = c + row[m]
        
        # Swap
        prev, curr = curr, prev
        
    return prev[-1]


def _neh_initial(pt):
    """NEH heuristic."""
    n_jobs, n_machines = pt.shape
    pt_rows = [list(pt[j]) for j in range(n_jobs)]
    totals = pt.sum(axis=1)
    order = np.argsort(-totals)
    
    perm = [int(order[0])]
    for i in range(1, n_jobs):
        job = int(order[i])
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            test_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(pt, pt_rows, test_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
    return perm


def _palmer_initial(pt):
    """Palmer's heuristic."""
    n_jobs, n_machines = pt.shape
    n = n_machines
    scores = np.zeros(n_jobs)
    for j in range(n_jobs):
        s = 0
        for m in range(n):
            if m < n // 2:
                s += (n - 1 - m) * pt[j, m]
            else:
                s -= m * pt[j, m]
        scores[j] = s
    return list(np.argsort(-scores))


def _vnd_local_search(pt, pt_rows, perm, deadline, max_block_size=3):
    """
    Variable Neighborhood Descent.
    Neighborhoods: 1-insertion, 2-insertion, ..., max_block_size-insertion.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm[:], _makespan(pt, pt_rows, perm)
    
    current_perm = perm[:]
    current_ms = _makespan(pt, pt_rows, current_perm)
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        
        # Try each neighborhood structure
        for k in range(1, max_block_size + 1):
            if time.time() > deadline:
                break
            
            best_found = False
            best_perm = current_perm[:]
            best_ms = current_ms
            
            # Generate neighbors by moving a block of size k
            # To avoid O(N^2 * k) complexity explosion, we can limit checks or use random sampling if N is large.
            # For standard sizes (N<100), full scan is okay if optimized.
            # However, full scan of all blocks and all positions is O(N^2).
            # Number of blocks: N-k+1. Positions: N-k+1. Total ~ N^2.
            # If N=100, 10,000 moves. Each makespan O(N*M). 10,000 * 100 * 10 = 10M ops.
            # This might be too slow for Python in a loop if repeated many times.
            
            # Optimization: Only check a subset of moves or use a faster heuristic for move evaluation?
            # Let's stick to full scan for small N, but break early if no improvement found after some checks?
            # Or just do 1-insertion (k=1) which is the most common.
            
            if k == 1:
                # 1-insertion
                for i in range(n_jobs):
                    if time.time() > deadline:
                        break
                    job = current_perm[i]
                    # Remove job
                    temp_perm = current_perm[:i] + current_perm[i+1:]
                    # Insert at all positions
                    for j in range(len(temp_perm) + 1):
                        if j == i:
                            continue
                        new_perm = temp_perm[:j] + [job] + temp_perm[j:]
                        new_ms = _makespan(pt, pt_rows, new_perm)
                        if new_ms < best_ms:
                            best_ms = new_ms
                            best_perm = new_perm
            else:
                # k-insertion (block move)
                # This is much more expensive. We will only do this if N is small or time allows.
                # For simplicity and speed, we might skip k>1 if N > 50.
                if n_jobs > 50:
                    continue
                
                for i in range(n_jobs - k + 1):
                    if time.time() > deadline:
                        break
                    block = current_perm[i:i+k]
                    # Remove block
                    temp_perm = current_perm[:i] + current_perm[i+k:]
                    # Insert block at all positions
                    for j in range(len(temp_perm) - k + 2): # Can insert before or after existing blocks
                         # Positions for a block of size k in a list of size N-k
                         # Valid indices: 0 to (N-k)
                         # Inserting at j means new_perm = temp[:j] + block + temp[j:]
                         # We must ensure the block doesn't overlap with original position?
                         # Actually, "Or-Opt" style moves usually remove a block and insert elsewhere.
                         # We skip positions that would result in the same permutation.
                         
                         # The original position was i.
                         # If we insert at i, it's the same.
                         # If we insert at i+1, it's the same (block shifted right by 1? No, removed then inserted).
                         # If we remove indices i..i+k-1, the list shrinks.
                         # Inserting at j in the new list.
                         # If j == i, we put it back.
                         # If j == i+1, we put it after the gap?
                         # Let's just check all j and filter out duplicates if necessary, or rely on makespan comparison.
                         
                         new_perm = temp_perm[:j] + block + temp_perm[j:]
                         
                         # Skip if same permutation
                         # Simple check: if new_perm == current_perm, skip
                         # But creating new_perm is expensive.
                         # We can check: if j == i or (j == i+1 and ...)? 
                         # Actually, if we remove block at i, and insert at i, it's identical.
                         # If we insert at i+1, it's also identical? 
                         # Example: [A, B, C, D], remove B (i=1, k=1). Temp: [A, C, D].
                         # Insert B at 1: [A, B, C, D] (Same)
                         # Insert B at 2: [A, C, B, D] (Different)
                         # So j=i is the only trivial duplicate for k=1?
                         # For k=2: [A, B, C, D], remove BC (i=1, k=2). Temp: [A, D].
                         # Insert BC at 1: [A, BC, D] -> [A, B, C, D] (Same)
                         # Insert BC at 0: [BC, A, D]
                         # Insert BC at 2: [A, D, BC]
                         # So j=i is the trivial duplicate.
                         
                         if j == i:
                             continue
                             
                         new_ms = _makespan(pt, pt_rows, new_perm)
                         if new_ms < best_ms:
                             best_ms = new_ms
                             best_perm = new_perm
                             
            if best_ms < current_ms:
                current_perm = best_perm
                current_ms = best_ms
                improved = True
                # Restart the neighborhood loop with the new best
                break
        
    return current_perm, current_ms


def _simulated_annealing(pt, pt_rows, perm, deadline, max_iter=20000):
    """
    Simulated Annealing with Tabu List.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm[:], _makespan(pt, pt_rows, perm)
    
    current_perm = perm[:]
    current_ms = _makespan(pt, pt_rows, current_perm)
    best_perm = current_perm[:]
    best_ms = current_ms
    
    T = max(current_ms * 0.1, 1.0)
    T_min = max(current_ms * 0.001, 0.1)
    cooling = 0.99
    
    # Tabu list: store recent permutations (as tuples) to avoid cycling
    tabu_list = []
    tabu_size = 50
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Generate neighbor: pick random job to move
        i = random.randint(0, n_jobs - 1)
        job = current_perm[i]
        temp_perm = current_perm[:i] + current_perm[i+1:]
        
        # Pick random insertion position
        j = random.randint(0, len(temp_perm))
        if j == i:
            continue  # No change
        
        new_perm = temp_perm[:j] + [job] + temp_perm[j:]
        
        # Check tabu
        new_key = tuple(new_perm)
        if new_key in tabu_list:
            # If it's not the best solution found so far, skip
            if new_ms >= best_ms: 
                 continue
            # If it is better than best, accept even if tabu
        
        new_ms = _makespan(pt, pt_rows, new_perm)
        
        delta = new_ms - current_ms
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_ms = new_ms
            current_perm = new_perm
            
            # Update tabu
            tabu_list.append(new_key)
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)
                
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm[:]
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_perm, best_ms


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    """
    pt = np.asarray(processing_times)
    n_jobs, n_machines = pt.shape
    
    # Deadline guard: 145s to leave 5s margin before 150s kill
    start_time = time.time()
    deadline = start_time + 145
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))
    
    rng = np.random.RandomState(42)
    
    # Pre-extract job rows as Python lists for faster access in makespan computation
    pt_rows = [list(pt[j]) for j in range(n_jobs)]
    
    # Generate diverse initial solutions
    candidates = []
    
    # 1. NEH
    candidates.append(_neh_initial(pt))
    
    # 2. Palmer
    candidates.append(_palmer_initial(pt))
    
    # 3. LPT
    totals = pt.sum(axis=1)
    candidates.append(list(np.argsort(-totals)))
    
    # 4. SPT
    candidates.append(list(np.argsort(totals)))
    
    # 5. Sort by first machine descending
    candidates.append(list(np.argsort(-pt[:, 0])))
    
    # 6. Sort by last machine descending
    candidates.append(list(np.argsort(-pt[:, -1])))
    
    # 7. Random permutations
    for _ in range(5):
        candidates.append(list(rng.permutation(n_jobs)))
    
    best_perm = candidates[0]
    best_ms = _makespan(pt, pt_rows, best_perm)
    
    # Run VND local search on each candidate
    # Allocate time: 50% for local search, 50% for SA
    ls_deadline = start_time + (deadline - start_time) * 0.5
    
    for cand in candidates:
        if time.time() > ls_deadline:
            break
        
        perm, ms = _vnd_local_search(pt, pt_rows, cand, ls_deadline, max_block_size=2)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    # Run Simulated Annealing from best solution found
    sa_deadline = deadline - 2
    if time.time() < sa_deadline:
        perm, ms = _simulated_annealing(pt, pt_rows, best_perm, sa_deadline, max_iter=50000)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    # Final quick VND if time remains
    if time.time() < deadline - 1:
        perm, ms = _vnd_local_search(pt, pt_rows, best_perm, deadline - 1, max_block_size=1)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    return best_perm

# EVOLVE-BLOCK-END
