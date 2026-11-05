"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, P):
    """Compute makespan for a given permutation (vectorized-ish)."""
    n = len(perm)
    m = P.shape[1]
    # C[j, k] = completion time of job j on machine k
    # We process jobs in perm order
    C = np.zeros((n, m))
    for i, job in enumerate(perm):
        for k in range(m):
            if k == 0:
                C[i, k] = C[i - 1, k] + P[job, k] if i > 0 else P[job, k]
            else:
                C[i, k] = max(C[i - 1, k], C[i, k - 1]) + P[job, k]
    return C[n - 1, m - 1]


def _makespan_vec(perm, P):
    """Faster makespan using numpy where possible."""
    n = len(perm)
    m = P.shape[1]
    if n == 0:
        return 0.0
    # Build completion matrix row by row
    C_prev = np.zeros(m)
    C_curr = np.zeros(m)
    for job in perm:
        row = P[job]
        C_curr[0] = C_prev[0] + row[0]
        if m > 1:
            C_curr[1:] = np.maximum(C_prev[1:], C_curr[:-1]) + row[1:]
        C_prev = C_curr.copy()
    return C_prev[m - 1]


def _neh_construct(P):
    """NEH heuristic for flow-shop scheduling."""
    n = P.shape[0]
    m = P.shape[1]
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    # Sort jobs by total processing time descending
    totals = P.sum(axis=1)
    order = np.argsort(-totals)
    
    # Build permutation incrementally
    perm = [order[0]]
    for idx in range(1, n):
        job = order[idx]
        # Try all positions in current perm
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_vec(candidate, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return perm


def _two_opt(perm, P):
    """2-opt local search: swap any two jobs, keep if improves."""
    n = len(perm)
    if n <= 1:
        return perm, _makespan_vec(perm, P)
    
    best = list(perm)
    best_ms = _makespan_vec(best, P)
    improved = True
    max_iter = 5000  # safety
    
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        for i in range(n):
            for j in range(i + 1, n):
                if i == j:
                    continue
                # Swap
                trial = best[:]
                trial[i], trial[j] = trial[j], trial[i]
                ms = _makespan_vec(trial, P)
                if ms < best_ms - 1e-9:
                    best = trial
                    best_ms = ms
                    improved = True
    return best, best_ms


def _perturb(perm, strength=2):
    """Random perturbation: swap 'strength' random pairs."""
    n = len(perm)
    if n <= 1:
        return list(perm)
    p = list(perm)
    for _ in range(strength):
        i, j = np.random.choice(n, 2, replace=False)
        p[i], p[j] = p[j], p[i]
    return p


def order_jobs(processing_times):
    P = np.asarray(processing_times, dtype=np.float64)
    n, m = P.shape
    
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    deadline = time.time() + 140.0  # 10s safety margin
    rng = np.random.default_rng()
    
    best_perm = None
    best_ms = float('inf')
    
    # Phase 1: NEH construction
    perm0 = _neh_construct(P)
    ms0 = _makespan_vec(perm0, P)
    best_perm = list(perm0)
    best_ms = ms0
    
    # Phase 2: Local search on NEH result
    if time.time() < deadline:
        refined, ms_ref = _two_opt(perm0, P)
        if ms_ref < best_ms:
            best_perm = refined
            best_ms = ms_ref
    
    # Phase 3: Iterated local search with restarts
    if time.time() < deadline:
        current = list(best_perm)
        current_ms = best_ms
        no_improve = 0
        max_no_improve = 200
        
        while time.time() < deadline and no_improve < max_no_improve:
            # Perturb
            strength = min(3, max(1, len(current) // 5))
            perturbed = _perturb(current, strength)
            
            # Local search
            trial, trial_ms = _two_opt(perturbed, P)
            
            if trial_ms < current_ms - 1e-9:
                current = trial
                current_ms = trial_ms
                no_improve = 0
            else:
                no_improve += 1
            
            if trial_ms < best_ms - 1e-9:
                best_perm = list(trial)
                best_ms = trial_ms
        
        # Phase 4: Random restarts if time remains
        if time.time() < deadline - 5.0:
            for restart in range(50):
                if time.time() > deadline - 2.0:
                    break
                # Random initial permutation
                rand_perm = list(rng.permutation(n))
                refined, ms = _two_opt(rand_perm, P)
                if ms < best_ms - 1e-9:
                    best_perm = refined
                    best_ms = ms
    
    return best_perm
# EVOLVE-BLOCK-END
