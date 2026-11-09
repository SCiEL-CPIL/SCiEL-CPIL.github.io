# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan_vec(processing_times, perm):
    """Compute makespan using vectorized NumPy."""
    n = len(perm)
    m = processing_times.shape[1]
    if n == 0 or m == 0:
        return 0.0
    P = processing_times[perm]
    C = np.zeros((n, m))
    C[0] = np.cumsum(P[0])
    for j in range(1, n):
        prev = C[j-1]
        cur = np.empty(m)
        cur[0] = prev[0] + P[j, 0]
        for k in range(1, m):
            cur[k] = max(cur[k-1], prev[k]) + P[j, k]
        C[j] = cur
    return C[-1, -1]

def _makespan_fast(processing_times, perm):
    """Fast makespan computation using list-based approach for small m."""
    n = len(perm)
    m = processing_times.shape[1]
    if n == 0 or m == 0:
        return 0.0
    P = processing_times
    prev_row = [0.0] * m
    curr_row = [0.0] * m
    for j in range(n):
        job = perm[j]
        if j == 0:
            s = 0.0
            for k in range(m):
                s += P[job, k]
                curr_row[k] = s
        else:
            c0 = prev_row[0]
            for k in range(m):
                val = P[job, k]
                if k == 0:
                    curr_row[0] = c0 + val
                else:
                    cm1 = curr_row[k-1]
                    pk = prev_row[k]
                    curr_row[k] = (cm1 if cm1 > pk else pk) + val
        prev_row, curr_row = curr_row, prev_row
    return prev_row[-1]

def _insertion_cost(processing_times, perm, job, pos):
    """Compute makespan of inserting `job` at position `pos` in `perm`."""
    n = len(perm)
    m = processing_times.shape[1]
    if m == 0:
        return 0.0
    P = processing_times
    prev_row = [0.0] * m
    curr_row = [0.0] * m
    for j in range(n + 1):
        if j < pos:
            job_j = perm[j]
        else:
            job_j = job if j == pos else perm[j - 1]
        if j == 0:
            s = 0.0
            for k in range(m):
                s += P[job_j, k]
                curr_row[k] = s
        else:
            c0 = prev_row[0]
            for k in range(m):
                val = P[job_j, k]
                if k == 0:
                    curr_row[0] = c0 + val
                else:
                    cm1 = curr_row[k-1]
                    pk = prev_row[k]
                    curr_row[k] = (cm1 if cm1 > pk else pk) + val
        prev_row, curr_row = curr_row, prev_row
    return prev_row[-1]

def _neh_standard(processing_times, rng=None):
    """Standard NEH: sort by total processing time descending, insert at best position."""
    n, m = processing_times.shape
    if n == 0:
        return []
    totals = processing_times.sum(axis=1)
    if rng is not None:
        noise = rng.uniform(0, 1e-6, size=n)
        order = np.argsort(-totals + noise)
    else:
        order = np.argsort(-totals)
    perm = []
    for job in order:
        best_pos = 0
        best_ms = np.inf
        for pos in range(len(perm) + 1):
            ms = _insertion_cost(processing_times, perm, job, pos)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    return perm

def _neh_first_last(processing_times, rng=None):
    """NEH variant: first job is the one with max total, then insert best."""
    n, m = processing_times.shape
    if n == 0:
        return []
    totals = processing_times.sum(axis=1)
    if rng is not None:
        noise = rng.uniform(0, 1e-6, size=n)
        totals_noisy = totals + noise
    else:
        totals_noisy = totals
    first_job = int(np.argmax(totals_noisy))
    perm = [first_job]
    remaining = [j for j in range(n) if j != first_job]
    if rng is not None:
        noise = rng.uniform(0, 1e-6, size=len(remaining))
        rem_order = sorted(remaining, key=lambda j: -(totals_noisy[j] + noise[remaining.index(j)]))
    else:
        rem_order = sorted(remaining, key=lambda j: -totals[j])
    for job in rem_order:
        best_pos = 0
        best_ms = np.inf
        for pos in range(len(perm) + 1):
            ms = _insertion_cost(processing_times, perm, job, pos)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    return perm

def _neh_last_machine(processing_times, rng=None):
    """NEH variant: sort by last machine processing time descending."""
    n, m = processing_times.shape
    if n == 0:
        return []
    if m == 0:
        return list(range(n))
    last_col = processing_times[:, m - 1]
    if rng is not None:
        noise = rng.uniform(0, 1e-6, size=n)
        order = np.argsort(-last_col + noise)
    else:
        order = np.argsort(-last_col)
    perm = []
    for job in order:
        best_pos = 0
        best_ms = np.inf
        for pos in range(len(perm) + 1):
            ms = _insertion_cost(processing_times, perm, job, pos)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    return perm

def _random_perm(processing_times, rng):
    """Random permutation."""
    n = processing_times.shape[0]
    return list(rng.permutation(n))

def _aco_construct(processing_times, pheromones, rng, alpha=1.0, beta=2.0):
    """
    Construct a permutation using Ant Colony Optimization probabilistic selection.
    At each step, select the next job with probability proportional to
    (pheromone^(alpha) * (1/processing_time)^(beta)).
    """
    n, m = processing_times.shape
    if n == 0:
        return []
    
    # Compute heuristic values: inverse of total processing time
    totals = processing_times.sum(axis=1)
    # Avoid division by zero
    totals_safe = np.where(totals > 0, totals, 1.0)
    heuristic = 1.0 / totals_safe
    
    remaining = list(range(n))
    perm = []
    
    for _ in range(n):
        if not remaining:
            break
        
        # Compute selection probabilities
        probs = []
        for job in remaining:
            tau = pheromones[job] if pheromones[job] > 1e-10 else 1e-10
            eta = heuristic[job]
            prob = (tau ** alpha) * (eta ** beta)
            probs.append(prob)
        
        total_prob = sum(probs)
        if total_prob <= 0:
            # Fall back to random
            idx = rng.integers(0, len(remaining))
            job = remaining[idx]
        else:
            # Normalize and sample
            probs = [p / total_prob for p in probs]
            cumulative = []
            cum = 0.0
            for p in probs:
                cum += p
                cumulative.append(cum)
            r = rng.random()
            idx = 0
            for i, c in enumerate(cumulative):
                if r <= c:
                    idx = i
                    break
            else:
                idx = len(remaining) - 1
            job = remaining[idx]
        
        perm.append(job)
        remaining.remove(job)
    
    return perm

def _vnd_polish(processing_times, perm, deadline, rng=None):
    """Variable Neighborhood Descent with 1-3 insertions and swaps."""
    n = len(perm)
    if n < 2:
        return list(perm), _makespan_fast(processing_times, perm)
    
    current_perm = list(perm)
    current_ms = _makespan_fast(processing_times, current_perm)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1-Insertion
        for i in range(n):
            if time.time() >= deadline: break
            job = current_perm[i]
            remaining = current_perm[:i] + current_perm[i+1:]
            best_pos = i
            best_ms = current_ms
            for j in range(len(remaining) + 1):
                if j == i or j == i+1: continue
                ms = _insertion_cost(processing_times, remaining, job, j)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = j
            if best_ms < current_ms:
                current_perm = remaining[:best_pos] + [job] + remaining[best_pos:]
                current_ms = best_ms
                improved = True
                break
        
        if improved or time.time() >= deadline: continue
        
        # 2-Insertion
        for i in range(n-1):
            if time.time() >= deadline: break
            block = current_perm[i:i+2]
            remaining = current_perm[:i] + current_perm[i+2:]
            best_pos = i
            best_ms = current_ms
            for j in range(len(remaining) + 1):
                if j == i or j == i+1 or j == i+2: continue
                new_perm = remaining[:j] + block + remaining[j:]
                ms = _makespan_fast(processing_times, new_perm)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = j
            if best_ms < current_ms:
                current_perm = remaining[:best_pos] + block + remaining[best_pos:]
                current_ms = best_ms
                improved = True
                break
        
        if improved or time.time() >= deadline: continue
        
        # 3-Insertion
        if n >= 3:
            for i in range(n-2):
                if time.time() >= deadline: break
                block = current_perm[i:i+3]
                remaining = current_perm[:i] + current_perm[i+3:]
                best_pos = i
                best_ms = current_ms
                for j in range(len(remaining) + 1):
                    if j == i or j == i+1 or j == i+2 or j == i+3: continue
                    new_perm = remaining[:j] + block + remaining[j:]
                    ms = _makespan_fast(processing_times, new_perm)
                    if ms < best_ms:
                        best_ms = ms
                        best_pos = j
                if best_ms < current_ms:
                    current_perm = remaining[:best_pos] + block + remaining[best_pos:]
                    current_ms = best_ms
                    improved = True
                    break
            
            if improved or time.time() >= deadline: continue

        # Swaps
        for i in range(n-1):
            if time.time() >= deadline: break
            for j in range(i+1, n):
                new_perm = current_perm[:]
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                ms = _makespan_fast(processing_times, new_perm)
                if ms < current_ms:
                    current_perm = new_perm
                    current_ms = ms
                    improved = True
                    break
            if improved: break

    return current_perm, current_ms

def _simulated_annealing(processing_times, perm, deadline, rng=None):
    """Simulated Annealing with 2-opt block-reversal and insertion moves."""
    n = len(perm)
    if n < 2:
        return list(perm), _makespan_fast(processing_times, perm)
    
    if rng is None:
        rng = np.random.default_rng(42)
    
    current_perm = list(perm)
    current_ms = _makespan_fast(processing_times, current_perm)
    best_perm = list(current_perm)
    best_ms = current_ms
    
    avg_processing = np.mean(processing_times)
    T0 = max(current_ms * 0.05, avg_processing * n * 0.1, 1.0)
    T_end = T0 * 1e-6
    cooling_rate = 0.995
    
    T = T0
    steps_since_best = 0
    max_stall = 500
    
    while time.time() < deadline and T > T_end:
        if rng.random() < 0.6:
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i == j:
                continue
            job = current_perm[i]
            remaining = current_perm[:i] + current_perm[i+1:]
            if j > i:
                j -= 1
            new_perm = remaining[:j] + [job] + remaining[j:]
        else:
            i = rng.integers(0, n - 1)
            j = rng.integers(i + 1, n)
            new_perm = current_perm[:i] + current_perm[i:j+1][::-1] + current_perm[j+1:]
        
        new_ms = _makespan_fast(processing_times, new_perm)
        delta = new_ms - current_ms
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_perm = new_perm
            current_ms = new_ms
            if current_ms < best_ms:
                best_perm = list(current_perm)
                best_ms = current_ms
                steps_since_best = 0
            else:
                steps_since_best += 1
        
        T *= cooling_rate
        
        if steps_since_best > max_stall:
            T = max(T, T0 * 0.01)
            steps_since_best = 0
    
    return best_perm, best_ms

def _aco_optimize(processing_times, initial_perm, deadline, rng, n_ants=10, alpha=1.0, beta=2.0, rho=0.5):
    """
    Ant Colony Optimization for flow-shop scheduling.
    Uses pheromone trails to guide construction, with greedy insertion for local improvement.
    """
    n, m = processing_times.shape
    if n < 2:
        return list(initial_perm), _makespan_fast(processing_times, initial_perm)
    
    # Initialize pheromones
    # Use inverse of processing times as initial pheromone
    totals = processing_times.sum(axis=1)
    totals_safe = np.where(totals > 0, totals, 1.0)
    pheromones = 1.0 / totals_safe
    pheromones = pheromones / pheromones.sum() * n  # Normalize
    
    best_perm = list(initial_perm)
    best_ms = _makespan_fast(processing_times, best_perm)
    
    # Global best pheromone update value
    Q = 1.0
    
    while time.time() < deadline:
        # Each ant constructs a solution
        for ant in range(n_ants):
            if time.time() >= deadline:
                break
            
            # Construct permutation using ACO
            perm = _aco_construct(processing_times, pheromones, rng, alpha, beta)
            
            # Local improvement with greedy insertion (VND-lite)
            # Do a quick 1-insertion polish
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(n):
                    if time.time() >= deadline:
                        break
                    job = perm[i]
                    remaining = perm[:i] + perm[i+1:]
                    best_pos = i
                    best_ms_local = _makespan_fast(processing_times, perm)
                    for j in range(len(remaining) + 1):
                        if j == i or j == i+1:
                            continue
                        ms = _insertion_cost(processing_times, remaining, job, j)
                        if ms < best_ms_local:
                            best_ms_local = ms
                            best_pos = j
                    if best_pos != i:
                        perm = remaining[:best_pos] + [job] + remaining[best_pos:]
                        improved = True
                        break
            
            ms = _makespan_fast(processing_times, perm)
            
            if ms < best_ms:
                best_ms = ms
                best_perm = list(perm)
        
        # Update pheromones
        if time.time() >= deadline:
            break
        
        # Evaporate
        pheromones *= (1.0 - rho)
        
        # Deposit pheromone on best solution
        # Amount deposited is proportional to 1/makespan
        deposit = Q / max(best_ms, 1e-10)
        for pos, job in enumerate(best_perm):
            # Deposit on the job itself (simplified: deposit on job index)
            pheromones[job] += deposit
        
        # Normalize pheromones to prevent numerical issues
        pheromones = np.maximum(pheromones, 1e-10)
        pheromones = pheromones / pheromones.sum() * n
    
    return best_perm, best_ms

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    
    Args:
        processing_times: np.ndarray of shape (jobs, machines)
        
    Returns:
        List of job indices in the order they should be processed.
    """
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    n, m = processing_times.shape
    if n == 0:
        return []
    if n == 1:
        return [0]
    if m == 0:
        return list(range(n))
    
    rng = np.random.default_rng(12345)
    
    # Generate multiple candidate permutations from different heuristics
    candidates = []
    
    # Standard NEH (deterministic)
    candidates.append(_neh_standard(processing_times))
    
    # First-last NEH (deterministic)
    candidates.append(_neh_first_last(processing_times))
    
    # Last-machine NEH (deterministic)
    candidates.append(_neh_last_machine(processing_times))
    
    # Randomized NEH variants
    for i in range(4):
        if time.time() > deadline:
            break
        candidates.append(_neh_standard(processing_times, rng))
    
    for i in range(4):
        if time.time() > deadline:
            break
        candidates.append(_neh_first_last(processing_times, rng))
    
    # A few random starts
    for i in range(2):
        if time.time() > deadline:
            break
        candidates.append(_random_perm(processing_times, rng))
    
    # Evaluate all candidates and find the best
    best_perm = candidates[0]
    best_ms = _makespan_fast(processing_times, best_perm)
    
    for perm in candidates[1:]:
        if time.time() > deadline:
            break
        ms = _makespan_fast(processing_times, perm)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    # VND Polish on the best candidate
    if time.time() < deadline - 2:
        improved_perm, improved_ms = _vnd_polish(processing_times, best_perm, deadline)
        if improved_ms < best_ms:
            best_perm = improved_perm
            best_ms = improved_ms
    
    # ACO Optimization to explore the space
    if time.time() < deadline - 1:
        improved_perm, improved_ms = _aco_optimize(processing_times, best_perm, deadline, rng)
        if improved_ms < best_ms:
            best_perm = improved_perm
            best_ms = improved_ms
    
    # Simulated Annealing to escape local optima
    if time.time() < deadline - 1:
        improved_perm, improved_ms = _simulated_annealing(processing_times, best_perm, deadline, rng)
        if improved_ms < best_ms:
            best_perm = improved_perm
            best_ms = improved_ms
    
    # Final VND Polish
    if time.time() < deadline - 0.5:
        improved_perm, improved_ms = _vnd_polish(processing_times, best_perm, deadline)
        if improved_ms < best_ms:
            best_perm = improved_perm
            best_ms = improved_ms
    
    return list(best_perm)
# EVOLVE-BLOCK-END
