"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation using NumPy."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    
    # Vectorized DP: compute row by row, using vectorized ops within each row
    pt_matrix = processing_times[np.array(perm)]  # shape (n_jobs, n_machines)
    
    prev_row = np.zeros(n_machines, dtype=np.float64)
    for j in range(n_jobs):
        row = pt_matrix[j]
        if j == 0:
            curr_row = np.cumsum(row)
        else:
            curr_row = np.maximum(prev_row, np.concatenate(([prev_row[-1] if n_machines > 1 else 0.0], prev_row[:-1])) if n_machines > 1 else prev_row) + row
            # Actually simpler: for m >= 1, C[j][m] = max(C[j-1][m], C[j][m-1]) + pt[j][m]
            # C[j][0] = C[j-1][0] + pt[j][0]
            curr_row = np.empty(n_machines, dtype=np.float64)
            curr_row[0] = prev_row[0] + row[0]
            if n_machines > 1:
                for m in range(1, n_machines):
                    curr_row[m] = max(prev_row[m], curr_row[m-1]) + row[m]
        prev_row = curr_row
    
    return float(prev_row[-1])


def _makespan_from_prev(perm, processing_times, prev_perm=None, prev_ms=None):
    """Compute makespan, optionally using previous permutation for incremental update."""
    if prev_perm is None or prev_ms is None:
        return _makespan(perm, processing_times)
    
    # For now, just recompute; incremental would require tracking changes
    return _makespan(perm, processing_times)


def _makespan_from_prev(perm, processing_times, prev_perm=None, prev_ms=None):
    """Compute makespan, optionally using previous permutation for incremental update."""
    if prev_perm is None or prev_ms is None:
        return _makespan(perm, processing_times)
    
    # For now, just recompute; incremental would require tracking changes
    return _makespan(perm, processing_times)


def _neh_order(processing_times):
    """Generate NEH heuristic ordering."""
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))
    
    # Forward NEH: insert jobs one by one at best position
    partial = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(partial) + 1):
            candidate = partial[:pos] + [job] + partial[pos:]
            ms = _makespan(candidate, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        partial = partial[:best_pos] + [job] + partial[best_pos:]
    
    return partial


def _neh_order_perturbed(processing_times, rng, perturbation_strength=0.3):
    """Generate NEH ordering with perturbed job order."""
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    
    # Create perturbed order: mix sorted order with random
    base_order = list(np.argsort(-totals))
    perturbed_order = base_order.copy()
    
    # Apply random swaps to perturb
    n_swaps = max(1, int(n_jobs * perturbation_strength))
    for _ in range(n_swaps):
        i = rng.randint(0, n_jobs - 1)
        j = rng.randint(0, n_jobs - 1)
        perturbed_order[i], perturbed_order[j] = perturbed_order[j], perturbed_order[i]
    
    # Forward NEH: insert jobs one by one at best position
    partial = [perturbed_order[0]]
    for i in range(1, n_jobs):
        job = perturbed_order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(partial) + 1):
            candidate = partial[:pos] + [job] + partial[pos:]
            ms = _makespan(candidate, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        partial = partial[:best_pos] + [job] + partial[best_pos:]
    
    return partial


def _insertion_local_search(perm, processing_times, deadline, max_passes=100):
    """Local search using insertion moves (move job from one position to another)."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return list(perm), _makespan(perm, processing_times)
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    
    for _ in range(max_passes):
        if time.time() >= deadline:
            break
        
        improved = False
        for i in range(n_jobs):
            if time.time() >= deadline:
                break
            
            # Try moving job i to each other position
            job = best_perm[i]
            for j in range(n_jobs):
                if i == j:
                    continue
                
                # Create new permutation by removing job from i and inserting at j
                new_perm = best_perm[:i] + best_perm[i+1:]
                insert_pos = j if j < i else j - 1
                new_perm = new_perm[:insert_pos] + [job] + new_perm[insert_pos:]
                
                ms = _makespan(new_perm, processing_times)
                if ms < best_ms - 1e-10:
                    best_perm = new_perm
                    best_ms = ms
                    improved = True
                    break  # Restart scan after improvement
        
        if not improved:
            break
    
    return best_perm, best_ms


def _swap_local_search(perm, processing_times, deadline, max_passes=50):
    """Local search using adjacent swap moves."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return list(perm), _makespan(perm, processing_times)
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    
    for _ in range(max_passes):
        if time.time() >= deadline:
            break
        
        improved = False
        for i in range(n_jobs - 1):
            if time.time() >= deadline:
                break
            
            # Try swapping adjacent jobs i and i+1
            new_perm = list(best_perm)
            new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
            ms = _makespan(new_perm, processing_times)
            if ms < best_ms - 1e-10:
                best_perm = new_perm
                best_ms = ms
                improved = True
        
        if not improved:
            break
    
    return best_perm, best_ms


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin: 5s before 120s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    
    rng = random.Random(42)
    
    # Baseline: longest processing time first
    baseline_perm = list(np.argsort(-processing_times.sum(axis=1)))
    best_perm = baseline_perm
    best_ms = _makespan(best_perm, processing_times)
    
    # Try NEH heuristic
    if time.time() < deadline:
        neh_perm = _neh_order(processing_times)
        neh_ms = _makespan(neh_perm, processing_times)
        if neh_ms < best_ms:
            best_perm = neh_perm
            best_ms = neh_ms
    
    # Multi-start: try perturbed NEH orders
    if time.time() < deadline:
        n_starts = max(2, min(10, int((deadline - time.time()) / 0.5)))
        for _ in range(n_starts):
            if time.time() >= deadline:
                break
            
            perturbed = _neh_order_perturbed(processing_times, rng, perturbation_strength=0.2)
            p_ms = _makespan(perturbed, processing_times)
            if p_ms < best_ms:
                best_perm = perturbed
                best_ms = p_ms
    
    # Local search: insertion moves first (more powerful), then swaps
    if time.time() < deadline and n_jobs > 1:
        remaining = deadline - time.time()
        
        # Phase 1: Insertion local search
        if remaining > 5.0:
            insert_deadline = start_time + remaining - 2.0  # Leave 2s for swap phase
            polished, polished_ms = _insertion_local_search(
                best_perm, processing_times, insert_deadline, max_passes=50
            )
            if polished_ms < best_ms:
                best_perm = polished
                best_ms = polished_ms
        
        # Phase 2: Swap local search for fine-tuning
        if time.time() < deadline:
            swap_deadline = deadline
            polished, polished_ms = _swap_local_search(
                best_perm, processing_times, swap_deadline, max_passes=100
            )
            if polished_ms < best_ms:
                best_perm = polished
                best_ms = polished_ms
        
        # Phase 3: Another round of insertion to refine further
        if time.time() < deadline:
            polished, polished_ms = _insertion_local_search(
                best_perm, processing_times, deadline, max_passes=30
            )
            if polished_ms < best_ms:
                best_perm = polished
                best_ms = polished_ms
    
    return list(best_perm)
# EVOLVE-BLOCK-END
