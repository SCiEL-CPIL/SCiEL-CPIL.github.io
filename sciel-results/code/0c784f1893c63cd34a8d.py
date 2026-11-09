"""Flow-shop makespan minimizer: NEH + Simulated Annealing with insertion local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan(perm, pt):
    """
    Calculate makespan for a permutation using the standard recurrence.
    perm: 1-D array of job indices (length n_jobs)
    pt: (n_jobs, n_machines) processing times
    Returns: float makespan
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    if n_jobs == 0:
        return 0.0
    if n_jobs == 1:
        return float(pt[perm[0]].sum())
    
    p = pt[perm]  # (n_jobs, n_machines)
    c = np.zeros(n_jobs, dtype=np.float64)
    
    for m in range(n_machines):
        c[0] = p[0, m]
        for j in range(1, n_jobs):
            c[j] = max(c[j], c[j-1]) + p[j, m]
    
    return float(c[-1])


def _neh(processing_times):
    """
    NEH heuristic: order jobs by total processing time desc, insert each optimally.
    """
    n_jobs, n_machines = processing_times.shape
    order = list(np.argsort(-processing_times.sum(axis=1)))
    
    perm = [order[0]]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_cost = float('inf')
        
        for pos in range(i + 1):
            cand = perm[:pos] + [job] + perm[pos:]
            cost = _makespan(np.array(cand, dtype=np.int64), processing_times)
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return np.array(perm, dtype=np.int64)


def _insertion_local_search(perm, pt, deadline, margin):
    """
    Local search using insertion moves: remove job at i and insert at j.
    """
    n_jobs = len(perm)
    best = perm.copy()
    best_cost = _makespan(best, pt)
    improved = True
    
    while improved:
        improved = False
        
        for i in range(n_jobs):
            if time.monotonic() > deadline - margin:
                return best, best_cost
            for j in range(n_jobs):
                if j == i:
                    continue
                cand = list(best)
                job = cand.pop(i)
                insert_pos = j if j <= i else j - 1
                cand.insert(insert_pos, job)
                cand_arr = np.array(cand, dtype=np.int64)
                cost = _makespan(cand_arr, pt)
                if cost < best_cost - 1e-12:
                    best_cost = cost
                    best = cand_arr
                    improved = True
                    break
            if improved:
                break
    
    return best, best_cost


def _swap_local_search(perm, pt, deadline, margin):
    """
    Local search using swap moves: exchange two positions i, j.
    """
    n_jobs = len(perm)
    best = perm.copy()
    best_cost = _makespan(best, pt)
    improved = True
    
    while improved:
        improved = False
        
        for i in range(n_jobs - 1):
            if time.monotonic() > deadline - margin:
                return best, best_cost
            for j in range(i + 1, n_jobs):
                cand = best.copy()
                cand[i], cand[j] = cand[j], cand[i]
                cost = _makespan(cand, pt)
                if cost < best_cost - 1e-12:
                    best_cost = cost
                    best = cand
                    improved = True
                    break
            if improved:
                break
    
    return best, best_cost


def _two_opt_local_search(perm, pt, deadline, margin):
    """
    Local search using 2-opt (segment reversal) moves.
    """
    n_jobs = len(perm)
    best = perm.copy()
    best_cost = _makespan(best, pt)
    improved = True
    
    while improved:
        improved = False
        
        for i in range(n_jobs - 1):
            if time.monotonic() > deadline - margin:
                return best, best_cost
            for j in range(i + 1, n_jobs):
                cand = best.copy()
                cand[i:j + 1] = cand[i:j + 1][::-1]
                cost = _makespan(cand, pt)
                if cost < best_cost - 1e-12:
                    best_cost = cost
                    best = cand
                    improved = True
                    break
            if improved:
                break
    
    return best, best_cost


def _simulated_annealing(start_perm, pt, deadline, margin, rng):
    """
    Simulated annealing with insertion perturbations and local search polish.
    """
    n_jobs = len(start_perm)
    n_machines = pt.shape[1]
    
    current = start_perm.copy()
    current_cost = _makespan(current, pt)
    best = current.copy()
    best_cost = current_cost
    
    # Cooling schedule
    t0 = time.monotonic()
    total_time = deadline - t0
    margin_time = margin
    
    # Parameters
    T_initial = max(best_cost * 0.1, 1.0)
    T_final = max(best_cost * 0.001, 0.01)
    alpha = 0.995  # cooling rate
    
    T = T_initial
    
    while time.monotonic() < deadline - margin_time:
        if T < T_final:
            break
        
        # Perturbation: random insertion move
        i = rng.integers(0, n_jobs)
        j = rng.integers(0, n_jobs)
        if i == j:
            continue
        
        cand = list(current)
        job = cand.pop(i)
        insert_pos = j if j <= i else j - 1
        cand.insert(insert_pos, job)
        cand_arr = np.array(cand, dtype=np.int64)
        cand_cost = _makespan(cand_arr, pt)
        
        delta = cand_cost - current_cost
        
        if delta < 0:
            # Accept improvement
            current = cand_arr
            current_cost = cand_cost
            if current_cost < best_cost:
                best = current.copy()
                best_cost = current_cost
        else:
            # Accept worsening with probability exp(-delta/T)
            if rng.random() < np.exp(-delta / T):
                current = cand_arr
                current_cost = cand_cost
        
        # Periodically polish with local search
        if int((deadline - time.monotonic()) * 10) % 50 == 0:
            if time.monotonic() < deadline - margin_time - 2.0:
                polished, polished_cost = _insertion_local_search(current, pt, deadline, margin_time + 2.0)
                if polished_cost < best_cost:
                    best = polished
                    best_cost = polished_cost
                current = polished
                current_cost = polished_cost
        
        T *= alpha
    
    return best, best_cost


def order_jobs(processing_times):
    processing_times = np.asarray(processing_times, dtype=float)
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]

    t0 = time.monotonic()
    deadline = t0 + 145.0  # Safety margin
    margin = 5.0

    # 1. Constructive Heuristic: NEH
    best_perm = _neh(processing_times)
    best_cost = _makespan(best_perm, processing_times)
    
    # 2. Local Search: insertion first, then swap, then 2-opt
    best_perm, best_cost = _insertion_local_search(best_perm, processing_times, deadline, margin)
    best_perm, best_cost = _swap_local_search(best_perm, processing_times, deadline, margin)
    best_perm, best_cost = _two_opt_local_search(best_perm, processing_times, deadline, margin)
    
    # 3. Try other initial solutions
    # SPT on first machine
    spt_perm = np.argsort(processing_times[:, 0])
    spt_cost = _makespan(spt_perm, processing_times)
    if spt_cost < best_cost:
        best_perm = spt_perm
        best_cost = spt_cost
        best_perm, best_cost = _insertion_local_search(best_perm, processing_times, deadline, margin)
        best_perm, best_cost = _swap_local_search(best_perm, processing_times, deadline, margin)
    
    # SPT on last machine
    spt_last_perm = np.argsort(processing_times[:, -1])
    spt_last_cost = _makespan(spt_last_perm, processing_times)
    if spt_last_cost < best_cost:
        best_perm = spt_last_perm
        best_cost = spt_last_cost
        best_perm, best_cost = _insertion_local_search(best_perm, processing_times, deadline, margin)
        best_perm, best_cost = _swap_local_search(best_perm, processing_times, deadline, margin)
    
    # 4. Simulated Annealing with multiple restarts
    rng = np.random.default_rng(42)
    
    # SA from current best
    if time.monotonic() < deadline - margin - 10.0:
        sa_best, sa_cost = _simulated_annealing(best_perm, processing_times, deadline, margin, rng)
        if sa_cost < best_cost:
            best_perm = sa_best
            best_cost = sa_cost
    
    # Random restarts with SA
    for _ in range(3):
        if time.monotonic() > deadline - margin - 15.0:
            break
        rand_perm = rng.permutation(n_jobs)
        rand_cost = _makespan(rand_perm, processing_times)
        if rand_cost < best_cost * 1.5:  # Only if not too bad
            # Polish the random solution
            polished, polished_cost = _insertion_local_search(rand_perm, processing_times, deadline, margin)
            if polished_cost < best_cost:
                best_perm = polished
                best_cost = polished_cost
            
            # SA from the polished random solution
            if time.monotonic() < deadline - margin - 10.0:
                sa_best, sa_cost = _simulated_annealing(polished, processing_times, deadline, margin, rng)
                if sa_cost < best_cost:
                    best_perm = sa_best
                    best_cost = sa_cost
    
    # Final polish
    if time.monotonic() < deadline - margin:
        best_perm, best_cost = _insertion_local_search(best_perm, processing_times, deadline, margin)
        best_perm, best_cost = _swap_local_search(best_perm, processing_times, deadline, margin)
    
    return list(best_perm)
# EVOLVE-BLOCK-END
