"""Flow-shop permutation via TSP-like construction, Or-opt moves, and multi-start search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(pt, perm):
    """Compute makespan for a given permutation using DP."""
    n_jobs, n_machines = pt.shape
    if n_machines == 0 or n_jobs == 0:
        return 0.0
    
    # Use pure Python lists for speed on small M
    prev = [0.0] * n_machines
    for j in perm:
        curr = [0.0] * n_machines
        curr[0] = prev[0] + pt[j, 0]
        for m in range(1, n_machines):
            val = prev[m] if prev[m] > curr[m-1] else curr[m-1]
            curr[m] = val + pt[j, m]
        prev = curr
    return prev[-1]

def _compute_edge_weights(pt):
    """
    Pre-compute an N x N matrix of incremental makespan contributions.
    edge[i][j] approximates the cost of placing job j immediately after job i.
    """
    n_jobs, n_machines = pt.shape
    # edge[i][j] = makespan of [i, j] - makespan of [i]
    # Actually, a better metric for TSP-like construction is the marginal increase.
    # Let's define cost(i, j) as the completion time of job j on the last machine
    # given that job i was the previous job.
    # C(i, j) = max(C_i_machines, C_j_machines_prev) + p_j_machines... this is complex.
    # Simpler: Just use the sum of processing times or a specific heuristic.
    # The inspiration says "approximated by the incremental makespan contribution".
    # Let's compute the actual makespan of the pair [i, j] minus the makespan of [i].
    # This captures the interaction.
    
    edge = np.zeros((n_jobs, n_jobs))
    
    # Pre-compute makespan of single jobs (which is just sum of processing times)
    single_ms = pt.sum(axis=1)
    
    for i in range(n_jobs):
        for j in range(n_jobs):
            if i == j:
                edge[i][j] = float('inf')
                continue
            # Makespan of sequence [i, j]
            # Job i: finishes on machine m at sum(pt[i, :m+1])
            # Job j: 
            #   Machine 0: pt[i,0] + pt[j,0]
            #   Machine m: max(finish_i_m, finish_j_m-1) + pt[j,m]
            
            # We can compute this quickly
            finish_i = np.cumsum(pt[i])
            finish_j = np.zeros(n_machines)
            finish_j[0] = finish_i[0] + pt[j, 0]
            for m in range(1, n_machines):
                finish_j[m] = max(finish_i[m], finish_j[m-1]) + pt[j, m]
            
            ms_pair = finish_j[-1]
            # The "cost" of inserting j after i is the increase in makespan
            # compared to just having i.
            edge[i][j] = ms_pair - single_ms[i]
            
    return edge

def _tsp_nearest_neighbor(pt, edge_weights, start_job):
    """
    Construct a permutation using nearest neighbor on the edge weight graph.
    """
    n_jobs = pt.shape[0]
    visited = [False] * n_jobs
    perm = [start_job]
    visited[start_job] = True
    
    for _ in range(n_jobs - 1):
        last_job = perm[-1]
        best_next = -1
        best_cost = float('inf')
        
        for j in range(n_jobs):
            if not visited[j]:
                cost = edge_weights[last_job][j]
                if cost < best_cost:
                    best_cost = cost
                    best_next = j
        
        if best_next == -1:
            break
            
        perm.append(best_next)
        visited[best_next] = True
        
    return perm

def _neh_initial(pt):
    """NEH heuristic: sort by total processing time descending, insert each job at best position."""
    n_jobs, n_machines = pt.shape
    totals = pt.sum(axis=1)
    order = np.argsort(-totals)
    
    perm = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            test_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(pt, test_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
    return perm

def _palmer_initial(pt):
    """Palmer's heuristic: sort by (n-1)*sum_first_half - sum_second_half."""
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

def _lpt_initial(pt):
    """LPT: sort by total processing time descending."""
    totals = pt.sum(axis=1)
    return list(np.argsort(-totals))

def _spt_initial(pt):
    """SPT: sort by total processing time ascending."""
    totals = pt.sum(axis=1)
    return list(np.argsort(totals))

def _random_initial(n_jobs, rng):
    """Random permutation."""
    return list(rng.permutation(n_jobs))

def _or_opt_local_search(pt, perm, deadline, max_evals=50000):
    """
    Local search with Or-opt moves (moving a segment of 1, 2, or 3 jobs).
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm[:], _makespan(pt, perm)
    
    current_perm = perm[:]
    current_ms = _makespan(pt, current_perm)
    best_perm = current_perm[:]
    best_ms = current_ms
    
    eval_count = 0
    
    while eval_count < max_evals:
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try Or-opt moves: remove segment of length L (1, 2, 3) and insert elsewhere
        for L in [1, 2, 3]:
            if time.time() > deadline or eval_count >= max_evals:
                break
                
            for i in range(n_jobs - L + 1):
                if time.time() > deadline or eval_count >= max_evals:
                    break
                    
                # Segment to move
                segment = current_perm[i:i+L]
                
                # Try inserting at all valid positions
                # We remove the segment first, so the list shrinks by L
                # Then we insert into the remaining list
                # Valid insertion positions in the new list: 0 to len(new_list)
                
                # Optimization: skip positions that result in the same permutation
                # This is tricky, so we just check if the resulting permutation is different
                # or just rely on makespan comparison.
                
                for j in range(n_jobs - L + 1):
                    # j is the position in the original list where the segment would be inserted
                    # But since we are removing first, let's construct the new list
                    # Actually, it's easier to just generate the new permutation and check
                    # if it's the same as current.
                    
                    # Construct new perm
                    # Remove segment
                    remaining = current_perm[:i] + current_perm[i+L:]
                    
                    # Determine insertion index in 'remaining'
                    # If j < i, insert at j
                    # If j >= i + L, insert at j - L
                    # If i <= j < i + L, it's the same position (or adjacent), skip to avoid no-op or duplicate
                    # Let's just try all insertions in 'remaining' and skip if resulting perm == current_perm
                    # This is O(N) check, which is fine.
                    
                    # Actually, a simpler way:
                    # Just iterate over all possible insertion positions in the reduced list
                    # and check if the resulting permutation is different from current_perm.
                    
                    # To be efficient, let's just try moving the segment to a different location.
                    # If we remove from i, and insert at j in the remaining list:
                    # If j < i, the segment moves left.
                    # If j >= i, the segment moves right (adjusted for removal).
                    
                    # Let's just generate the new perm and compare.
                    # For each potential insertion index k in [0, len(remaining)]:
                    for k in range(len(remaining) + 1):
                        # Skip if this results in the same permutation
                        # This happens if k == i (inserting back at start of removed section)
                        # or k == i + L (inserting back at end) -- wait, indices are in 'remaining'
                        # If we removed i..i+L-1, then:
                        # Inserting at k < i is moving left.
                        # Inserting at k >= i is moving right.
                        # If k == i, we are inserting right after the previous element, which is the original position.
                        # If k == i + L, we are inserting after the next element, which is also original?
                        # No, if we remove i..i+L-1, the element at i in 'remaining' is original i+L.
                        # So inserting at k=i puts it before original i+L.
                        # Inserting at k=i+L puts it after original i+L.
                        
                        # To avoid no-ops:
                        # If k == i, it's the same position.
                        # If k == i + L, it's also effectively the same relative position?
                        # Let's just check if new_perm == current_perm.
                        
                        new_perm = remaining[:k] + segment + remaining[k:]
                        
                        # Quick check: if new_perm is same as current_perm, skip
                        # This is O(N).
                        # We can optimize by checking just the moved part, but for now O(N) is ok for small N.
                        # Actually, let's just compute makespan. If it's the same, delta is 0, no improvement.
                        # So we don't need to explicitly skip no-ops for correctness, just for speed.
                        # But computing makespan is O(N*M). Checking equality is O(N).
                        # So checking equality first is better.
                        
                        # Simple check: if k == i or k == i + L (approx), it might be a no-op.
                        # Let's just do the equality check.
                        # But doing it for every k is expensive.
                        # Let's just compute makespan.
                        
                        new_ms = _makespan(pt, new_perm)
                        eval_count += 1
                        
                        if new_ms < current_ms:
                            current_ms = new_ms
                            current_perm = new_perm
                            improved = True
                            if current_ms < best_ms:
                                best_ms = current_ms
                                best_perm = current_perm[:]
                            # First improvement: break out of loops?
                            # Or-opt often benefits from continuing to find better moves in the same pass?
                            # Or restart?
                            # Let's break out of k loop, and continue to next i?
                            # Or break all and restart while?
                            # Let's break out of k, i, L and restart while for a fresh pass.
                            # This is a common strategy.
                            break
                        # If no improvement, continue
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                continue
            break
            
        if not improved:
            break
    
    return best_perm, best_ms

def _simulated_annealing(pt, perm, deadline, max_iter=20000):
    """
    Simulated annealing with Or-opt moves.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm[:], _makespan(pt, perm)
    
    current_perm = perm[:]
    current_ms = _makespan(pt, current_perm)
    best_perm = current_perm[:]
    best_ms = current_ms
    
    T = max(current_ms * 0.1, 1.0)
    T_min = max(current_ms * 0.001, 0.1)
    cooling = 0.995
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Generate neighbor: Or-opt move
        L = random.randint(1, min(3, n_jobs))
        i = random.randint(0, n_jobs - L)
        segment = current_perm[i:i+L]
        remaining = current_perm[:i] + current_perm[i+L:]
        
        # Pick random insertion position
        k = random.randint(0, len(remaining))
        
        # Avoid no-op
        if k == i or k == i + L:
            # This might be a no-op, but let's just try
            pass
            
        new_perm = remaining[:k] + segment + remaining[k:]
        
        # If it's the same, skip
        if new_perm == current_perm:
            continue
            
        new_ms = _makespan(pt, new_perm)
        
        delta = new_ms - current_ms
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_ms = new_ms
            current_perm = new_perm
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm[:]
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_perm, best_ms

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using TSP-like construction,
    Or-opt local search, and simulated annealing.
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
    
    # Pre-compute edge weights for TSP-like construction
    # This is O(N^2 * M), which is acceptable for small N
    edge_weights = _compute_edge_weights(pt)
    
    candidates = []
    
    # 1. TSP Nearest Neighbor starting from the job with highest total processing time
    totals = pt.sum(axis=1)
    start_job = int(np.argmax(totals))
    candidates.append(_tsp_nearest_neighbor(pt, edge_weights, start_job))
    
    # 2. NEH (best single heuristic)
    candidates.append(_neh_initial(pt))
    
    # 3. Palmer's heuristic
    candidates.append(_palmer_initial(pt))
    
    # 4. LPT
    candidates.append(_lpt_initial(pt))
    
    # 5. SPT
    candidates.append(_spt_initial(pt))
    
    # 6. Sort by first machine descending
    candidates.append(list(np.argsort(-pt[:, 0])))
    
    # 7. Sort by last machine descending
    candidates.append(list(np.argsort(-pt[:, -1])))
    
    # 8. Random permutations
    for _ in range(4):
        candidates.append(_random_initial(n_jobs, rng))
    
    best_perm = candidates[0]
    best_ms = _makespan(pt, best_perm)
    
    # Run Or-opt local search on each candidate
    for cand in candidates:
        if time.time() > deadline - 10:  # Leave 10s for final SA
            break
        
        perm, ms = _or_opt_local_search(pt, cand, deadline - 10)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    # Run simulated annealing from best solution found
    if time.time() < deadline - 5:
        perm, ms = _simulated_annealing(pt, best_perm, deadline - 5, max_iter=20000)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    # If we still have time, run SA from a few other good starts
    if time.time() < deadline - 2:
        for cand in candidates[:5]:
            if time.time() > deadline - 2:
                break
            perm, ms = _simulated_annealing(pt, cand, deadline - 2, max_iter=5000)
            if ms < best_ms:
                best_ms = ms
                best_perm = perm
    
    return best_perm

# EVOLVE-BLOCK-END
