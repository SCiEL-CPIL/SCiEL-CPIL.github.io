<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _calc_makespan(processing_times, perm):
    """
    Vectorized makespan calculation using NumPy.
    O(n_jobs * n_machines) but with much lower constant factor than pure Python.
    """
    n_jobs = len(perm)
    if n_jobs == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    if n_machines == 1:
        return float(processing_times[perm, 0].sum())
    
    pt = processing_times[perm]  # shape (n_jobs, n_machines)
    
    # completion[j, m] = completion time of job j on machine m
    # completion[j, 0] = sum of pt[0..j, 0]
    # completion[j, m] = max(completion[j-1, m], completion[j, m-1]) + pt[j, m]
    completion = np.zeros((n_jobs, n_machines), dtype=np.float64)
    completion[:, 0] = np.cumsum(pt[:, 0])
    for m in range(1, n_machines):
        # completion[j, m] = max(completion[j-1, m], completion[j, m-1]) + pt[j, m]
        prev_col = completion[:n_jobs-1, m]  # completion[j-1, m] for j=1..n-1
        curr_col = completion[:, m-1]        # completion[j, m-1] for j=0..n-1
        # For j=0: completion[0, m] = max(0, completion[0, m-1]) + pt[0, m]
        # For j>=1: completion[j, m] = max(completion[j-1, m], completion[j, m-1]) + pt[j, m]
        # We can compute this iteratively or use a trick:
        # completion[j, m] = max(completion[j-1, m], completion[j, m-1]) + pt[j, m]
        # This is a sequential dependency along j, so we need a loop over j
        # But we can vectorize over m for each j, or just do the j-loop in Python
        # Actually for small n_machines this is fine. Let's do it properly.
        # Alternative: use the fact that this is a prefix-max operation
        # completion[j,m] = max(completion[j-1,m], completion[j,m-1]) + pt[j,m]
        # This can be rewritten as:
        # Let A[j] = completion[j, m-1], B[j] = completion[j-1, m]
        # completion[j, m] = max(B[j], A[j]) + pt[j, m]
        # But B depends on the previous computation of column m, so it's sequential in j.
        # We'll use a simple loop over j for this column.
        for j in range(n_jobs):
            if j == 0:
                completion[j, m] = completion[j, m-1] + pt[j, m]
            else:
                c_prev_machine = completion[j-1, m]
                c_prev_job = completion[j, m-1]
                completion[j, m] = max(c_prev_machine, c_prev_job) + pt[j, m]
    
    return float(completion[-1, -1])


def _calc_makespan_fast(processing_times, perm):
    """
    Fast makespan using a tighter loop. For very small n_machines,
    this avoids NumPy overhead.
    """
    n_jobs = len(perm)
    if n_jobs == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    if n_machines == 1:
        return float(processing_times[perm, 0].sum())
    
    # Use a 1D array for current completion times
    completion = np.zeros(n_machines, dtype=np.float64)
    pt = processing_times
    
    for job in perm:
        row = pt[job]
        prev = 0.0
        for m in range(n_machines):
            if m == 0:
                completion[m] += row[m]
            else:
                if completion[m] < prev:
                    completion[m] = prev
                completion[m] += row[m]
            prev = completion[m]
    
    return float(completion[-1])


def _neh_heuristic(processing_times, rng=None):
    """
    NEH heuristic: Sort jobs by total processing time descending,
    then insert each job into the position that minimizes makespan.
    Optional rng for tie-breaking diversity.
    """
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))
    
    if rng is not None and n_jobs > 1:
        # Add slight randomness to break ties or perturb order slightly
        # Swap a random pair to create diversity
        i, j = rng.sample(range(n_jobs), 2)
        order[i], order[j] = order[j], order[i]
    
    sequence = [order[0]]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(sequence) + 1):
            new_seq = sequence[:pos] + [job] + sequence[pos:]
            ms = _calc_makespan_fast(processing_times, new_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        sequence.insert(best_pos, job)
    
    return sequence


def _random_permutation(n_jobs):
    """Generate a random permutation of job indices."""
    perm = list(range(n_jobs))
    random.shuffle(perm)
    return perm


def _reverse_permutation(sequence):
    """Return the reversed sequence."""
    return list(reversed(sequence))


def _shifted_permutation(sequence, shift):
    """Return the sequence rotated by shift positions."""
    n = len(sequence)
    if shift % n == 0:
        return list(sequence)
    return list(sequence[shift % n:]) + list(sequence[:shift % n])


def _swap_pair(sequence, i, j):
    """Return a new sequence with elements at i and j swapped."""
    new_seq = list(sequence)
    new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
    return new_seq


def _insertion_move(sequence, i, j):
    """
    Remove element at position i and insert it at position j.
    If j > i, adjust for the shift caused by removal.
    """
    new_seq = list(sequence)
    job = new_seq.pop(i)
    if j > i:
        insert_idx = j - 1
    else:
        insert_idx = j
    new_seq.insert(insert_idx, job)
    return new_seq


def _generate_candidate(processing_times, n_jobs, strategy, rng_seed=None):
    """
    Generate a single candidate permutation based on the given strategy.
    strategy: 'neh', 'random', 'reversed_neh', 'shifted_neh', 'random_swap_neh'
    """
    if strategy == 'neh':
        return _neh_heuristic(processing_times)
    elif strategy == 'random':
        return _random_permutation(n_jobs)
    elif strategy == 'reversed_neh':
        return _reverse_permutation(_neh_heuristic(processing_times))
    elif strategy == 'shifted_neh':
        base = _neh_heuristic(processing_times)
        shift = random.randint(1, max(1, n_jobs - 1))
        return _shifted_permutation(base, shift)
    elif strategy == 'random_swap_neh':
        base = _neh_heuristic(processing_times)
        if n_jobs >= 2:
            i, j = random.sample(range(n_jobs), 2)
            return _swap_pair(base, i, j)
        return base
    else:
        return _random_permutation(n_jobs)


def _local_search(processing_times, sequence, deadline):
    """
    Best-improvement insertion local search with full restart.
    Evaluates all insertion moves and picks the best one each iteration.
    """
    n_jobs = len(sequence)
    current_seq = list(sequence)
    current_makespan = _calc_makespan_fast(processing_times, current_seq)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_gain = 0.0
        best_i = -1
        best_j = -1
        
        for i in range(n_jobs):
            if time.time() >= deadline:
                break
            job = current_seq[i]
            for j in range(n_jobs):
                if j == i:
                    continue
                if time.time() >= deadline:
                    break
                
                new_seq = _insertion_move(current_seq, i, j)
                new_makespan = _calc_makespan_fast(processing_times, new_seq)
                
                gain = current_makespan - new_makespan
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_j = j
        
        if best_i >= 0 and best_gain > 1e-10:
            current_seq = _insertion_move(current_seq, best_i, best_j)
            current_makespan -= best_gain
            improved = True
    
    return current_seq, current_makespan


def _simulated_annealing(processing_times, sequence, deadline, rng=None):
    """
    Simulated annealing with insertion perturbations for better neighborhood connectivity.
    """
    n_jobs = len(sequence)
    if n_jobs < 2:
        return list(sequence), _calc_makespan_fast(processing_times, sequence)
    
    if rng is None:
        rng = random.Random()
    
    current_seq = list(sequence)
    current_makespan = _calc_makespan_fast(processing_times, current_seq)
    
    best_seq = list(current_seq)
    best_makespan = current_makespan
    
    T = current_makespan * 0.1
    cooling_rate = 0.999
    min_T = 1e-6
    no_improve = 0
    patience = 5000
    
    while time.time() < deadline and T > min_T:
        # Use insertion moves for better connectivity
        i, j = rng.sample(range(n_jobs), 2)
        if j == i:
            continue
        new_seq = _insertion_move(current_seq, i, j)
        new_makespan = _calc_makespan_fast(processing_times, new_seq)
        
        delta = new_makespan - current_makespan
        
        if delta < 0 or (T > min_T and rng.random() < np.exp(-delta / max(T, 1e-10))):
            current_seq = new_seq
            current_makespan = new_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_seq = list(current_seq)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        T *= cooling_rate
        
        if no_improve > patience:
            break
    
    return best_seq, best_makespan


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    rng = random.Random(42)
    
    # Phase 0: Generate diverse NEH-based candidates
    candidates = []
    
    # Base NEH
    base_neh = _neh_heuristic(processing_times, rng=None)
    candidates.append(base_neh)
    
    # Multiple NEH restarts with perturbations
    n_restarts = min(8, max(3, n_jobs // 5))
    for r in range(n_restarts):
        if time.time() >= deadline - 10.0:
            break
        # Perturb the NEH order
        perturbed_order = list(np.argsort(-processing_times.sum(axis=1)))
        # Apply random swaps to create diversity
        n_swaps = max(1, n_jobs // 10)
        for _ in range(n_swaps):
            i, j = rng.sample(range(n_jobs), 2)
            perturbed_order[i], perturbed_order[j] = perturbed_order[j], perturbed_order[i]
        # Run NEH insertion with this perturbed order
        sequence = [perturbed_order[0]]
        for i in range(1, n_jobs):
            job = perturbed_order[i]
            best_pos = 0
            best_makespan = float('inf')
            for pos in range(len(sequence) + 1):
                new_seq = sequence[:pos] + [job] + sequence[pos:]
                ms = _calc_makespan_fast(processing_times, new_seq)
                if ms < best_makespan:
                    best_makespan = ms
                    best_pos = pos
            sequence.insert(best_pos, job)
        candidates.append(sequence)
    
    # Add a few random candidates for diversity
    for _ in range(3):
        if time.time() >= deadline - 10.0:
            break
        candidates.append(_random_permutation(n_jobs))
    
    # Evaluate all candidates and keep the best
    best_seq = None
    best_makespan = float('inf')
    
    for cand in candidates:
        if time.time() >= deadline:
            break
        ms = _calc_makespan_fast(processing_times, cand)
        if ms < best_makespan:
            best_makespan = ms
            best_seq = list(cand)
    
    if best_seq is None:
        best_seq = list(range(n_jobs))
        best_makespan = _calc_makespan_fast(processing_times, best_seq)
    
    # Phase 1: Local search on the best candidate
    if time.time() < deadline - 10.0:
        ls_deadline = min(deadline - 10.0, time.time() + 40.0)
        ls_seq, ls_makespan = _local_search(processing_times, best_seq, ls_deadline)
        if ls_makespan < best_makespan:
            best_seq = ls_seq
            best_makespan = ls_makespan
    
    # Phase 2: Simulated annealing if time permits
    if time.time() < deadline - 15.0:
        sa_deadline = min(deadline - 5.0, time.time() + 50.0)
        sa_seq, sa_makespan = _simulated_annealing(processing_times, best_seq, sa_deadline, rng=rng)
        if sa_makespan < best_makespan:
            best_seq = sa_seq
            best_makespan = sa_makespan
    
    # Phase 3: Final local search polish
    if time.time() < deadline - 2.0:
        final_seq, final_makespan = _local_search(processing_times, best_seq, deadline)
        if final_makespan < best_makespan:
            best_seq = final_seq
            best_makespan = final_makespan
    
    return best_seq


# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _calc_makespan_fast(processing_times, perm):
    """
    Fast makespan calculation. Uses NumPy for the sequential dependency
    along jobs for each machine column, which is much faster than pure Python loops.
    """
    n_jobs = len(perm)
    if n_jobs == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    if n_machines == 1:
        return float(processing_times[perm, 0].sum())
    
    pt = processing_times[perm]  # shape (n_jobs, n_machines)
    
    # completion[j, m] = completion time of job j on machine m
    # completion[j, 0] = cumsum(pt[:, 0])
    # completion[j, m] = max(completion[j-1, m], completion[j, m-1]) + pt[j, m]
    completion = np.empty((n_jobs, n_machines), dtype=np.float64)
    completion[:, 0] = np.cumsum(pt[:, 0])
    
    for m in range(1, n_machines):
        # Sequential dependency: completion[j, m] depends on completion[j-1, m]
        # We must iterate over j, but can use NumPy for the max+add per step
        # For small n_machines this loop is fine; the inner work is O(1)
        col_prev = completion[:, m - 1]  # completion[j, m-1] for all j
        col_curr = completion[:, m]
        col_curr[0] = col_prev[0] + pt[0, m]
        for j in range(1, n_jobs):
            # completion[j, m] = max(completion[j-1, m], completion[j, m-1]) + pt[j, m]
            cm = completion[j-1, m]
            cj = col_prev[j]
            col_curr[j] = (cm if cm > cj else cj) + pt[j, m]
    
    return float(completion[-1, -1])


def _calc_makespan_from_completion(processing_times, perm):
    """
    Compute full completion matrix for a permutation.
    Returns (completion_matrix, makespan).
    """
    n_jobs = len(perm)
    if n_jobs == 0:
        return np.zeros((0, processing_times.shape[1])), 0.0
    n_machines = processing_times.shape[1]
    if n_machines == 1:
        comp = np.cumsum(processing_times[perm, 0]).reshape(-1, 1)
        return comp, float(comp[-1, 0])
    
    pt = processing_times[perm]
    completion = np.empty((n_jobs, n_machines), dtype=np.float64)
    completion[:, 0] = np.cumsum(pt[:, 0])
    
    for m in range(1, n_machines):
        col_prev = completion[:, m - 1]
        col_curr = completion[:, m]
        col_curr[0] = col_prev[0] + pt[0, m]
        for j in range(1, n_jobs):
            cm = completion[j-1, m]
            cj = col_prev[j]
            col_curr[j] = (cm if cm > cj else cj) + pt[j, m]
    
    return completion, float(completion[-1, -1])


def _neh_heuristic(processing_times, rng=None):
    """
    NEH heuristic: Sort jobs by total processing time descending,
    then insert each job into the position that minimizes makespan.
    """
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))
    
    if rng is not None and n_jobs > 1:
        i, j = rng.sample(range(n_jobs), 2)
        order[i], order[j] = order[j], order[i]
    
    sequence = [order[0]]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(sequence) + 1):
            new_seq = sequence[:pos] + [job] + sequence[pos:]
            ms = _calc_makespan_fast(processing_times, new_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        sequence.insert(best_pos, job)
    
    return sequence


def _random_permutation(n_jobs):
    """Generate a random permutation of job indices."""
    perm = list(range(n_jobs))
    random.shuffle(perm)
    return perm


def _first_improvement_ls(processing_times, sequence, deadline, rng=None):
    """
    First-improvement insertion local search.
    For each job position i, try inserting it at each position j (in random order),
    and accept the first improvement found.
    """
    n_jobs = len(sequence)
    if n_jobs < 2:
        return list(sequence), _calc_makespan_fast(processing_times, sequence)
    
    if rng is None:
        rng = random.Random()
    
    current_seq = list(sequence)
    current_makespan = _calc_makespan_fast(processing_times, current_seq)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try each job position
        for i in range(n_jobs):
            if time.time() >= deadline:
                break
            job = current_seq[i]
            # Try all insertion positions in random order
            positions = list(range(n_jobs))
            positions.remove(i)
            rng.shuffle(positions)
            
            for j in positions:
                if time.time() >= deadline:
                    break
                # Build new sequence: remove job from i, insert at j
                # If j > i, after removal the insert index is j-1
                new_seq = current_seq[:i] + current_seq[i+1:]
                insert_idx = j if j <= i else j - 1
                new_seq = new_seq[:insert_idx] + [job] + new_seq[insert_idx:]
                
                new_makespan = _calc_makespan_fast(processing_times, new_seq)
                
                if new_makespan < current_makespan - 1e-10:
                    current_seq = new_seq
                    current_makespan = new_makespan
                    improved = True
                    break  # Move to next job position
    
    return current_seq, current_makespan


def _tabu_search(processing_times, sequence, deadline, rng=None, tabu_len=15):
    """
    Tabu search with insertion moves.
    Maintains a tabu list of recent (i, j) moves.
    Aspiration criterion: accept tabu move if it improves the best solution.
    """
    n_jobs = len(sequence)
    if n_jobs < 2:
        return list(sequence), _calc_makespan_fast(processing_times, sequence)
    
    if rng is None:
        rng = random.Random()
    
    current_seq = list(sequence)
    current_makespan = _calc_makespan_fast(processing_times, current_seq)
    
    best_seq = list(current_seq)
    best_makespan = current_makespan
    
    # Tabu list: set of (i, j) pairs that are forbidden
    tabu = set()
    tabu_counter = {}  # (i,j) -> iteration when it became tabu
    
    no_improve = 0
    patience = 2000
    max_iter = 100000
    
    for iteration in range(max_iter):
        if time.time() >= deadline:
            break
        
        # Remove expired tabu entries
        current_iter = iteration
        expired = []
        for key, exp_iter in tabu_counter.items():
            if current_iter - exp_iter >= tabu_len:
                expired.append(key)
        for key in expired:
            tabu.discard(key)
            del tabu_counter[key]
        
        # Evaluate all non-tabu moves, pick the best
        best_gain = float('-inf')
        best_i = -1
        best_j = -1
        best_new_makespan = float('inf')
        
        for i in range(n_jobs):
            if time.time() >= deadline:
                break
            job = current_seq[i]
            for j in range(n_jobs):
                if j == i:
                    continue
                move = (i, j)
                is_tabu = move in tabu
                
                # Build new sequence
                new_seq = current_seq[:i] + current_seq[i+1:]
                insert_idx = j if j <= i else j - 1
                new_seq = new_seq[:insert_idx] + [job] + new_seq[insert_idx:]
                
                new_makespan = _calc_makespan_fast(processing_times, new_seq)
                gain = current_makespan - new_makespan
                
                # Aspiration: accept if improves best even if tabu
                if not is_tabu and gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_j = j
                    best_new_makespan = new_makespan
                elif is_tabu and new_makespan < best_makespan:
                    best_gain = gain
                    best_i = i
                    best_j = j
                    best_new_makespan = new_makespan
        
        if best_i < 0 or best_new_makespan >= current_makespan - 1e-10:
            no_improve += 1
            if no_improve > patience:
                break
            # Random perturbation to escape
            i, j = rng.sample(range(n_jobs), 2)
            new_seq = current_seq[:i] + current_seq[i+1:]
            insert_idx = j if j <= i else j - 1
            new_seq = new_seq[:insert_idx] + [current_seq[i]] + new_seq[insert_idx:]
            current_seq = new_seq
            current_makespan = _calc_makespan_fast(processing_times, current_seq)
            tabu.add((i, j))
            tabu_counter[(i, j)] = iteration
            continue
        
        # Apply best move
        new_seq = current_seq[:best_i] + current_seq[best_i+1:]
        insert_idx = best_j if best_j <= best_i else best_j - 1
        new_seq = new_seq[:insert_idx] + [current_seq[best_i]] + new_seq[insert_idx:]
        current_seq = new_seq
        current_makespan = best_new_makespan
        
        # Update tabu
        tabu.add((best_i, best_j))
        tabu_counter[(best_i, best_j)] = iteration
        
        if current_makespan < best_makespan:
            best_makespan = current_makespan
            best_seq = list(current_seq)
            no_improve = 0
        else:
            no_improve += 1
    
    return best_seq, best_makespan


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    rng = random.Random(42)
    
    # Phase 0: Generate diverse NEH-based candidates
    candidates = []
    
    # Base NEH
    base_neh = _neh_heuristic(processing_times, rng=None)
    candidates.append(base_neh)
    
    # Multiple NEH restarts with perturbations
    n_restarts = min(10, max(4, n_jobs // 4))
    for r in range(n_restarts):
        if time.time() >= deadline - 10.0:
            break
        # Perturb the NEH order
        perturbed_order = list(np.argsort(-processing_times.sum(axis=1)))
        # Apply random swaps to create diversity
        n_swaps = max(1, n_jobs // 8)
        for _ in range(n_swaps):
            i, j = rng.sample(range(n_jobs), 2)
            perturbed_order[i], perturbed_order[j] = perturbed_order[j], perturbed_order[i]
        # Run NEH insertion with this perturbed order
        sequence = [perturbed_order[0]]
        for i in range(1, n_jobs):
            job = perturbed_order[i]
            best_pos = 0
            best_makespan = float('inf')
            for pos in range(len(sequence) + 1):
                new_seq = sequence[:pos] + [job] + sequence[pos:]
                ms = _calc_makespan_fast(processing_times, new_seq)
                if ms < best_makespan:
                    best_makespan = ms
                    best_pos = pos
            sequence.insert(best_pos, job)
        candidates.append(sequence)
    
    # Add a few random candidates for diversity
    for _ in range(3):
        if time.time() >= deadline - 10.0:
            break
        candidates.append(_random_permutation(n_jobs))
    
    # Evaluate all candidates and keep the best
    best_seq = None
    best_makespan = float('inf')
    
    for cand in candidates:
        if time.time() >= deadline:
            break
        ms = _calc_makespan_fast(processing_times, cand)
        if ms < best_makespan:
            best_makespan = ms
            best_seq = list(cand)
    
    if best_seq is None:
        best_seq = list(range(n_jobs))
        best_makespan = _calc_makespan_fast(processing_times, best_seq)
    
    # Phase 1: First-improvement local search on the best candidate
    if time.time() < deadline - 15.0:
        ls_deadline = min(deadline - 15.0, time.time() + 30.0)
        ls_seq, ls_makespan = _first_improvement_ls(processing_times, best_seq, ls_deadline, rng=rng)
        if ls_makespan < best_makespan:
            best_seq = ls_seq
            best_makespan = ls_makespan
    
    # Phase 2: Tabu search to escape local optima
    if time.time() < deadline - 20.0:
        ts_deadline = min(deadline - 20.0, time.time() + 60.0)
        ts_seq, ts_makespan = _tabu_search(processing_times, best_seq, ts_deadline, rng=rng, tabu_len=12)
        if ts_makespan < best_makespan:
            best_seq = ts_seq
            best_makespan = ts_makespan
    
    # Phase 3: Final first-improvement local search polish
    if time.time() < deadline - 3.0:
        final_seq, final_makespan = _first_improvement_ls(processing_times, best_seq, deadline - 1.0, rng=rng)
        if final_makespan < best_makespan:
            best_seq = final_seq
            best_makespan = final_makespan
    
    return best_seq


# EVOLVE-BLOCK-END
>>>>>>> REPLACE
