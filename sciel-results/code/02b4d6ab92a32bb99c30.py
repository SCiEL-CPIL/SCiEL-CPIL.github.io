"""Flow-shop permutation via multi-start heuristics + cost-guided insertion + 2-opt local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan_fast(pt_list, perm):
    """
    Fast makespan calculation using pure Python lists.
    pt_list: list of lists, shape [n_jobs][n_machines].
    perm: list of job indices.
    """
    n_machines = len(pt_list[0]) if pt_list else 0
    n = len(perm)
    if n == 0:
        return 0.0
    if n_machines == 0:
        return 0.0
        
    prev = [0.0] * n_machines
    
    for job in perm:
        row = pt_list[job]
        curr = [0.0] * n_machines
        curr[0] = prev[0] + row[0]
        for m in range(1, n_machines):
            val = prev[m] if prev[m] > curr[m-1] else curr[m-1]
            curr[m] = val + row[m]
        prev = curr
    return prev[-1]


def _build_cost_matrix(pt_list, n_jobs):
    """
    Pre-compute a heuristic cost matrix to guide insertion moves.
    cost[i][j] approximates the marginal cost of inserting job i after job j.
    Lower cost is better.
    """
    n_machines = len(pt_list[0]) if pt_list else 0
    costs = [[0.0] * n_jobs for _ in range(n_jobs)]
    for i in range(n_jobs):
        row_i = pt_list[i]
        for j in range(n_jobs):
            if i == j:
                costs[i][j] = float('inf')
                continue
            row_j = pt_list[j]
            # Heuristic: sum of max(0, pt[i][m] - pt[j][m]) for m>=1 plus pt[i][0]
            # This captures the "bottleneck" effect of placing i after j
            c = row_i[0]
            for m in range(1, n_machines):
                diff = row_i[m] - row_j[m]
                if diff > 0:
                    c += diff
            costs[i][j] = c
    return costs


def _neh_initial(pt):
    """NEH heuristic."""
    n_jobs, n_machines = pt.shape
    totals = pt.sum(axis=1)
    order = list(np.argsort(-totals))
    perm = [order[0]]
    pt_list = pt.tolist()
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            test = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_fast(pt_list, test)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
    return perm


def _palmer_initial(pt):
    """Palmer's dispatching rule."""
    n_jobs, n_machines = pt.shape
    n = n_machines
    scores = np.zeros(n_jobs)
    for j in range(n_jobs):
        s = 0.0
        for m in range(n):
            if m < n // 2:
                s += (n - 1 - m) * pt[j, m]
            else:
                s -= m * pt[j, m]
        scores[j] = s
    return list(np.argsort(-scores))


def _lpt_initial(pt):
    totals = pt.sum(axis=1)
    return list(np.argsort(-totals))


def _bottleneck_initial(pt):
    """Sort by total processing on the machine with largest total load, descending."""
    n_jobs, n_machines = pt.shape
    machine_loads = pt.sum(axis=0)
    bottleneck = int(np.argmax(machine_loads))
    return list(np.argsort(-pt[:, bottleneck]))


def _random_initial(n_jobs, seed):
    rng = random.Random(seed)
    perm = list(range(n_jobs))
    rng.shuffle(perm)
    return perm


def _local_search_hybrid(pt_list, perm, deadline, cost_matrix):
    """
    Hybrid local search using Insertion (guided by cost matrix) and 2-Opt (segment reversal).
    """
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(pt_list, perm)

    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms

    # Adaptive parameters
    # For small n, we can be more exhaustive. For large n, we need to be selective.
    max_rounds = 10000
    
    for _round in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        
        # 1. Try Insertion moves, guided by cost matrix
        # We generate a list of potential moves, estimate their cost, and try the best ones first.
        # To save time, we don't evaluate ALL moves if n is large.
        
        # Generate candidate moves
        # Move (i, j): remove job at i, insert at j.
        # Cost estimate: cost_matrix[job_i][predecessor_of_j]
        
        candidates = []
        for i in range(n):
            job = current[i]
            for j in range(n):
                if j == i or j == i + 1:
                    continue
                # Predecessor of position j in the new sequence
                # If j == 0, pred is effectively "start" (we can use 0 cost or a dummy)
                # If j > 0, pred is current[j-1]
                if j == 0:
                    est_cost = cost_matrix[job][job] # Dummy, will be high
                    # Actually, inserting at 0 means no predecessor in the list, but it becomes the first.
                    # The cost heuristic is designed for "after j". 
                    # Let's just use a high estimate or skip sorting for j=0?
                    # Better: just append and sort.
                    est_cost = 0.0 # Start with optimistic
                else:
                    pred = current[j-1]
                    est_cost = cost_matrix[job][pred]
                
                candidates.append((est_cost, i, j))
        
        # Sort candidates by estimated cost (lowest first)
        candidates.sort(key=lambda x: x[0])
        
        # Evaluate top K candidates
        # K depends on n. For n=50, K=50 is 50 evaluations. For n=100, K=20.
        K = min(50, max(5, n // 2))
        
        best_move_found = False
        best_move_ms = current_ms
        best_move_i = -1
        best_move_j = -1
        
        for idx in range(min(K, len(candidates))):
            if time.time() > deadline:
                break
            _, i, j = candidates[idx]
            job = current[i]
            rest = current[:i] + current[i+1:]
            # Adjust j if i < j, because removing i shifts positions
            if i < j:
                j -= 1
            test = rest[:j] + [job] + rest[j:]
            ms = _makespan_fast(pt_list, test)
            if ms < best_move_ms - 1e-9:
                best_move_ms = ms
                best_move_i = i
                best_move_j = j
                best_move_found = True
                # We could break here if we want first-improvement, but best-improvement is often better
                # However, for speed, if we find a good one, we might stop early.
                # Let's stick to evaluating all K to find the best among them.
        
        if best_move_found:
            # Apply the best move found
            i, j = best_move_i, best_move_j
            job = current[i]
            rest = current[:i] + current[i+1:]
            if i < j:
                j -= 1
            current = rest[:j] + [job] + rest[j:]
            current_ms = best_move_ms
            if current_ms < best_ms - 1e-9:
                best_ms = current_ms
                best = list(current)
            improved = True
            continue # Restart loop with new current
        
        # 2. If no insertion move improved, try 2-Opt moves
        # 2-Opt: Reverse segment [i, j]
        # We try a subset of 2-opt moves to save time.
        # For efficiency, we try adjacent swaps first (which is 2-opt of length 2), then larger.
        
        # Simple 2-Opt search: iterate i, j
        # To avoid O(n^2) full evaluations, we can limit j range or use random sampling.
        # Given n is likely small (<100), O(n^2) is feasible if we are careful.
        # n=100 -> 10000 moves. 10000 makespans is fast in Python if optimized? 
        # _makespan_fast is O(n*m). If m=10, 100*10=1000 ops per makespan. 10000*1000 = 10^7 ops. 
        # This is doable in <1s in C, but Python might take 1-2s. 
        # So we should limit the number of 2-opt moves per round or use time checks.
        
        two_opt_improved = False
        best_two_opt_ms = current_ms
        best_two_opt_i = -1
        best_two_opt_j = -1
        
        # Limit 2-opt search to a subset of moves to keep it fast
        # We'll try all i, but limit j to i+1 to i+10 or so? 
        # Or just do a full sweep if n is small.
        
        if n <= 50:
            # Full 2-opt sweep
            for i in range(n - 1):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    # Reverse segment i..j
                    segment = current[i:j+1]
                    segment.reverse()
                    test = current[:i] + segment + current[j+1:]
                    ms = _makespan_fast(pt_list, test)
                    if ms < best_two_opt_ms - 1e-9:
                        best_two_opt_ms = ms
                        best_two_opt_i = i
                        best_two_opt_j = j
                        two_opt_improved = True
        else:
            # Sampled 2-opt for larger n
            # Try random segments
            for _ in range(100):
                if time.time() > deadline:
                    break
                i = random.randint(0, n - 2)
                j = random.randint(i + 1, n - 1)
                segment = current[i:j+1]
                segment.reverse()
                test = current[:i] + segment + current[j+1:]
                ms = _makespan_fast(pt_list, test)
                if ms < best_two_opt_ms - 1e-9:
                    best_two_opt_ms = ms
                    best_two_opt_i = i
                    best_two_opt_j = j
                    two_opt_improved = True

        if two_opt_improved:
            i, j = best_two_opt_i, best_two_opt_j
            segment = current[i:j+1]
            segment.reverse()
            current = current[:i] + segment + current[j+1:]
            current_ms = best_two_opt_ms
            if current_ms < best_ms - 1e-9:
                best_ms = current_ms
                best = list(current)
            improved = True
            continue # Restart
        
        # If no improvement in either neighborhood, we are at a local optimum
        if not improved:
            break
            
    return best, best_ms


def _simulated_annealing(pt_list, perm, deadline, max_iter=50000, seed=0):
    """Simulated annealing with insertion moves."""
    rng = random.Random(seed)
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(pt_list, perm)
    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms

    T = max(current_ms * 0.05, 1.0)
    T_min = max(current_ms * 0.0005, 0.01)
    cooling = 0.999

    for _ in range(max_iter):
        if time.time() > deadline:
            break
        i = rng.randint(0, n - 1)
        job = current[i]
        rest = current[:i] + current[i+1:]
        j = rng.randint(0, n - 1)
        if j == i or j == i + 1:
            j = (j + 1) % n
        test = rest[:j] + [job] + rest[j:]
        new_ms = _makespan_fast(pt_list, test)
        delta = new_ms - current_ms
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current = test
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best = list(current)
        T *= cooling
        if T < T_min:
            T = T_min
    return best, best_ms


def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = pt.shape
    start = time.time()
    deadline = start + 145.0  # Safety margin

    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))

    pt_list = pt.tolist()
    
    # Pre-compute cost matrix for guided search
    cost_matrix = _build_cost_matrix(pt_list, n_jobs)

    # Build diverse initial solutions
    candidates = [
        _neh_initial(pt),
        _palmer_initial(pt),
        _lpt_initial(pt),
        _bottleneck_initial(pt),
    ]
    # Add a few random starts
    for s in range(3):
        candidates.append(_random_initial(n_jobs, 100 + s))

    # Evaluate all, pick best
    best_ms = float('inf')
    best_perm = candidates[0]
    for cand in candidates:
        ms = _makespan_fast(pt_list, cand)
        if ms < best_ms:
            best_ms = ms
            best_perm = cand

    # Run hybrid local search on the best candidate
    # Allocate 70% of time to local search on the best candidate
    ls_deadline = start + (deadline - start) * 0.7
    perm, ms = _local_search_hybrid(pt_list, best_perm, ls_deadline, cost_matrix)
    if ms < best_ms:
        best_ms = ms
        best_perm = perm

    # If time remains, try local search on other good candidates
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        for cand in candidates[1:]:
            if time.time() > deadline - 10.0:
                break
            ms0 = _makespan_fast(pt_list, cand)
            if ms0 > best_ms * 1.1:
                continue
            # Run a shorter local search
            sub_deadline = min(deadline - 2.0, time.time() + remaining_time / 3)
            perm, ms = _local_search_hybrid(pt_list, cand, sub_deadline, cost_matrix)
            if ms < best_ms:
                best_ms = ms
                best_perm = perm

    # Simulated Annealing if time permits
    if time.time() < deadline - 15.0:
        sa_deadline = deadline - 5.0
        perm, ms = _simulated_annealing(pt_list, best_perm, sa_deadline, max_iter=200000, seed=7)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Final polish with local search
    if time.time() < deadline - 1.0:
        perm, ms = _local_search_hybrid(pt_list, best_perm, deadline - 0.5, cost_matrix)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    return best_perm
# EVOLVE-BLOCK-END
