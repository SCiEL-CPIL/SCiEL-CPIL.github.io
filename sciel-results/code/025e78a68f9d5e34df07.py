"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def _makespan(perm, pt):
    """Compute makespan for a given permutation using the flow-shop recurrence."""
    n = len(perm)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    comp = np.zeros(m, dtype=float)
    for j in range(n):
        job = perm[j]
        for mi in range(m):
            prev = comp[mi - 1] if mi > 0 else 0.0
            comp[mi] = max(comp[mi], prev) + pt[job, mi]
    return comp[-1]

def _neh(processing_times):
    """NEH heuristic for flow shop scheduling."""
    n, m = processing_times.shape
    total_pt = processing_times.sum(axis=1)
    order = np.argsort(-total_pt)
    
    # Build permutation by inserting jobs one by one in best position
    perm = [order[0]]
    for k in range(1, n):
        job = order[k]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            trial = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(trial, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return np.array(perm, dtype=int)

def _two_opt(perm, pt, max_iterations=100):
    """2-opt local search: try all pairs of positions and swap if improves."""
    n = len(perm)
    if n < 2:
        return perm.copy()
    
    current_ms = _makespan(perm, pt)
    best_perm = perm.copy()
    best_ms = current_ms
    improved = True
    iters = 0
    
    while improved and iters < max_iterations:
        improved = False
        iters += 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                trial = best_perm.copy()
                trial[i], trial[j] = trial[j], trial[i]
                ms = _makespan(trial, pt)
                if ms < best_ms - 1e-9:
                    best_ms = ms
                    best_perm = trial
                    improved = True
    
    return best_perm

def _or_opt(perm, pt):
    """Or-opt: try moving a segment of length 1, 2, or 3 to a different position."""
    n = len(perm)
    if n < 2:
        return perm.copy()
    
    best_perm = perm.copy()
    best_ms = _makespan(best_perm, pt)
    improved = True
    
    while improved:
        improved = False
        for seg_len in [1, 2, 3]:
            if n < seg_len + 1:
                continue
            for i in range(n - seg_len + 1):
                segment = best_perm[i:i+seg_len].copy()
                for j in range(n - seg_len + 1):
                    if j == i:
                        continue
                    # Remove segment from position i
                    if j > i:
                        j = j - seg_len  # adjust for removal
                    if j == i:
                        continue
                    # Insert at position j
                    trial = np.concatenate([best_perm[:i], best_perm[i+seg_len:j+seg_len] if j > i else best_perm[i+seg_len:j], segment, best_perm[j+seg_len:] if j > i else best_perm[j:]])
                    # Simpler: rebuild
                    trial = list(best_perm)
                    seg = trial[i:i+seg_len]
                    del trial[i:i+seg_len]
                    # Insert at j (adjusted)
                    if j > i:
                        j = j  # already adjusted
                    trial[j:j] = seg
                    trial = np.array(trial, dtype=int)
                    ms = _makespan(trial, pt)
                    if ms < best_ms - 1e-9:
                        best_ms = ms
                        best_perm = trial
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return best_perm

def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on a flow-shop instance."""
    start_time = time.time()
    effective_deadline = 140.0  # safety margin from 150s limit
    
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))
    
    rng = np.random.default_rng(42)
    
    # Initialize with NEH
    best_perm = _neh(processing_times)
    best_ms = _makespan(best_perm, processing_times)
    
    # Maintain a set of diverse solutions
    solutions = [best_perm.copy()]
    
    # Time budget for main loop
    while time.time() < effective_deadline:
        # Generate a new candidate from one of the existing solutions
        base_idx = rng.integers(0, len(solutions))
        base = solutions[base_idx].copy()
        
        # Apply a random perturbation
        if n >= 2:
            pert_type = rng.integers(0, 3)
            if pert_type == 0:
                # Swap two positions
                i, j = rng.integers(0, n, 2)
                if i != j:
                    base[i], base[j] = base[j], base[i]
            elif pert_type == 1:
                # Insert a job at a random position
                i = rng.integers(0, n)
                j = rng.integers(0, n)
                if i != j:
                    gene = base[i]
                    if i < j:
                        base[i:j] = base[i+1:j+1]
                        base[j] = gene
                    else:
                        base[j+1:i+1] = base[j:i]
                        base[i] = gene
            else:
                # Reverse a segment
                i, j = rng.integers(0, n, 2)
                if i > j:
                    i, j = j, i
                if j - i >= 1:
                    base[i:j+1] = base[i:j+1][::-1]
        
        # Local search on the perturbed solution
        candidate = _two_opt(base, processing_times, max_iterations=50)
        candidate_ms = _makespan(candidate, processing_times)
        
        # Accept if better than best
        if candidate_ms < best_ms - 1e-9:
            best_ms = candidate_ms
            best_perm = candidate.copy()
            solutions.append(candidate.copy())
            if len(solutions) > 20:
                solutions = solutions[-20:]
        else:
            # Sometimes add diverse solutions even if not better, to maintain diversity
            if rng.random() < 0.1 and len(solutions) < 15:
                solutions.append(candidate.copy())
    
    return list(best_perm)
# EVOLVE-BLOCK-END
