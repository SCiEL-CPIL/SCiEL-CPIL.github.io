"""Flow-shop permutation via GA with OX + bottleneck initialization + delta-based insertion local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan_fast(pt_list, perm):
    """Fast makespan calculation using pure Python lists."""
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


def _delta_insertion(pt_list, perm, i, j):
    """
    Compute the change in makespan when removing job at position i and inserting at position j.
    Returns the new makespan directly (or the delta).
    For flow-shop, we recompute the completion times for affected positions.
    """
    n = len(perm)
    n_machines = len(pt_list[0]) if pt_list else 0
    if n <= 1 or n_machines == 0:
        return _makespan_fast(pt_list, perm)

    job = perm[i]
    # Build the new permutation
    rest = perm[:i] + perm[i+1:]
    # Adjust j: if j > i, the insertion position in 'rest' is j-1
    if j > i:
        j_adj = j - 1
    else:
        j_adj = j
    new_perm = rest[:j_adj] + [job] + rest[j_adj:]

    # For efficiency, we could compute delta, but for correctness let's just compute makespan
    # Actually, for a true delta, we need to track completion times.
    # Given the complexity of exact delta for flow-shop (depends on max), 
    # and that _makespan_fast is quite fast, we stick to full recomputation 
    # but optimize the inner loop as much as possible.
    
    # Optimization: If the move doesn't change the relative order of most jobs,
    # we can potentially reuse some computation, but it's complex.
    # Let's just use _makespan_fast but with a slight optimization: 
    # if the job is moved to a position where it doesn't affect the critical path much, 
    # but that's hard to know a priori.
    
    return _makespan_fast(pt_list, new_perm)


def _build_cost_matrix(pt_list, n_jobs):
    """Pre-compute a cost matrix to guide search."""
    n_machines = len(pt_list[0]) if pt_list else 0
    costs = [[0.0] * n_jobs for _ in range(n_jobs)]
    for i in range(n_jobs):
        for j in range(n_jobs):
            if i == j:
                continue
            row_i = pt_list[i]
            row_j = pt_list[j]
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


def _spt_initial(pt):
    totals = pt.sum(axis=1)
    return list(np.argsort(totals))


def _bottleneck_initial(pt):
    """Sort by total processing on the machine with largest total load, descending."""
    n_jobs, n_machines = pt.shape
    machine_loads = pt.sum(axis=0)
    bottleneck = int(np.argmax(machine_loads))
    return list(np.argsort(-pt[:, bottleneck]))


def _mean_initial(pt):
    return list(np.argsort(-pt.mean(axis=1)))


def _first_machine_initial(pt):
    return list(np.argsort(-pt[:, 0]))


def _last_machine_initial(pt):
    return list(np.argsort(-pt[:, -1]))


def _random_initial(n_jobs, seed):
    rng = random.Random(seed)
    perm = list(range(n_jobs))
    rng.shuffle(perm)
    return perm


def _local_search_insertion(pt_list, perm, deadline, max_rounds=5000, cost_matrix=None):
    """
    Local search using insertion neighborhood with Top-K improvement.
    """
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(pt_list, perm)

    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms

    # Dynamic K based on problem size
    if n < 10:
        K = 10
    elif n < 20:
        K = 5
    else:
        K = 3

    for _round in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        moves = []
        for i in range(n):
            job = current[i]
            for j in range(n):
                if j == i or j == i + 1:
                    continue
                if cost_matrix is not None:
                    pred = current[j-1] if j > 0 else current[0]
                    est = cost_matrix[job][pred]
                else:
                    est = 0.0
                moves.append((est, i, j))
        
        if not moves:
            break

        moves.sort(key=lambda x: x[0])
        
        best_move_ms = current_ms
        best_move_i = -1
        best_move_j = -1
        
        for idx in range(min(K, len(moves))):
            est, i, j = moves[idx]
            if time.time() > deadline:
                break
            job = current[i]
            rest = current[:i] + current[i+1:]
            test = rest[:j] + [job] + rest[j:]
            ms = _makespan_fast(pt_list, test)
            if ms < best_move_ms - 1e-9:
                best_move_ms = ms
                best_move_i = i
                best_move_j = j
        
        if best_move_i != -1:
            job = current[best_move_i]
            rest = current[:best_move_i] + current[best_move_i+1:]
            current = rest[:best_move_j] + [job] + rest[best_move_j:]
            current_ms = best_move_ms
            improved = True
            if current_ms < best_ms - 1e-9:
                best_ms = current_ms
                best = list(current)
        
        if not improved:
            break
            
    return best, best_ms


def _local_search_2opt(pt_list, perm, deadline, max_rounds=5000, cost_matrix=None):
    """
    Local search using 2-opt segment reversal.
    """
    n = len(perm)
    if n <= 2:
        return perm, _makespan_fast(pt_list, perm)

    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms

    for _round in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                if cost_matrix is not None and (j - i) > 2:
                    if i > 0:
                        c_cur = cost_matrix[current[i]][current[i-1]]
                        c_rev = cost_matrix[current[j]][current[i-1]]
                        if c_rev > c_cur * 1.5:
                            continue
                    if j < n - 1:
                        c_cur = cost_matrix[current[j]][current[j+1]]
                        c_rev = cost_matrix[current[i]][current[j+1]]
                        if c_rev > c_cur * 1.5:
                            continue
                test = current[:i] + current[i:j+1][::-1] + current[j+1:]
                ms = _makespan_fast(pt_list, test)
                if ms < current_ms - 1e-9:
                    current_ms = ms
                    current = test
                    improved = True
                    if current_ms < best_ms - 1e-9:
                        best_ms = current_ms
                        best = list(current)
                    break
            if improved:
                break
        if not improved:
            break
    return best, best_ms


def _simulated_annealing(pt_list, perm, deadline, max_iter=100000, seed=0, cost_matrix=None):
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
    cooling = 0.9995

    for _ in range(max_iter):
        if time.time() > deadline:
            break
        i = rng.randint(0, n - 1)
        job = current[i]
        rest = current[:i] + current[i+1:]
        if cost_matrix is not None:
            j = rng.randint(0, n - 1)
            if rng.random() < 0.3:
                best_j = 0
                best_cost = float('inf')
                for jj in range(n):
                    if jj == i or jj == i + 1:
                        continue
                    pred = current[jj-1] if jj > 0 else current[0]
                    c = cost_matrix[job][pred]
                    if c < best_cost:
                        best_cost = c
                        best_j = jj
                if rng.random() < 0.5:
                    j = best_j
        else:
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


def _order_crossover(parent1, parent2):
    """Order Crossover (OX) for permutations."""
    n = len(parent1)
    if n <= 1:
        return list(parent1)
    c1 = random.randint(0, n - 2)
    c2 = random.randint(c1 + 1, n - 1)
    
    child = [None] * n
    child[c1:c2+1] = parent1[c1:c2+1]
    
    fill_values = [v for v in parent2 if v not in child]
    idx = (c2 + 1) % n
    for v in fill_values:
        child[idx] = v
        idx = (idx + 1) % n
    return child


def _swap_mutation(perm, rate=0.1):
    """Swap mutation for permutations."""
    n = len(perm)
    if n <= 1:
        return list(perm)
    child = list(perm)
    num_swaps = max(1, int(n * rate))
    for _ in range(num_swaps):
        i, j = random.sample(range(n), 2)
        child[i], child[j] = child[j], child[i]
    return child


def _genetic_algorithm(pt_list, initial_pops, deadline, pop_size=30, generations=50, cost_matrix=None):
    """
    Genetic Algorithm with Order Crossover and swap mutation.
    """
    n = len(initial_pops[0])
    if n <= 1:
        return initial_pops[0], _makespan_fast(pt_list, initial_pops[0])
    
    # Initialize population
    population = []
    fitnesses = []
    for p in initial_pops:
        ms = _makespan_fast(pt_list, p)
        population.append(list(p))
        fitnesses.append(ms)
    
    # Add more random individuals if needed
    while len(population) < pop_size:
        rng = random.Random(len(population))
        p = list(range(n))
        rng.shuffle(p)
        ms = _makespan_fast(pt_list, p)
        population.append(p)
        fitnesses.append(ms)
    
    best_perm = population[0]
    best_ms = fitnesses[0]
    for i in range(1, pop_size):
        if fitnesses[i] < best_ms:
            best_ms = fitnesses[i]
            best_perm = population[i]
    
    for gen in range(generations):
        if time.time() > deadline:
            break
        
        # Sort by fitness
        sorted_indices = sorted(range(pop_size), key=lambda i: fitnesses[i])
        population = [population[i] for i in sorted_indices]
        fitnesses = [fitnesses[i] for i in sorted_indices]
        
        new_population = []
        # Elitism: keep top 2
        new_population.append(list(population[0]))
        new_population.append(list(population[1]))
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection
            t1 = random.randint(0, min(4, pop_size-1))
            t2 = random.randint(0, min(4, pop_size-1))
            parent1 = population[t1] if fitnesses[t1] < fitnesses[t2] else population[t2]
            
            t1 = random.randint(0, min(4, pop_size-1))
            t2 = random.randint(0, min(4, pop_size-1))
            parent2 = population[t1] if fitnesses[t1] < fitnesses[t2] else population[t2]
            
            # Crossover
            child = _order_crossover(parent1, parent2)
            
            # Mutation
            if random.random() < 0.3:
                child = _swap_mutation(child, rate=0.05)
            
            new_population.append(child)
        
        # Evaluate new population
        for i in range(pop_size):
            fitnesses[i] = _makespan_fast(pt_list, new_population[i])
            if fitnesses[i] < best_ms:
                best_ms = fitnesses[i]
                best_perm = list(new_population[i])
        
        population = new_population
    
    return best_perm, best_ms


def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = pt.shape
    start = time.time()
    deadline = start + 145.0

    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))

    pt_list = pt.tolist()

    # Pre-compute insertion cost matrix for guided search
    cost_matrix = _build_cost_matrix(pt_list, n_jobs)

    # Build diverse initial solutions
    candidates = [
        _neh_initial(pt),
        _palmer_initial(pt),
        _lpt_initial(pt),
        _spt_initial(pt),
        _bottleneck_initial(pt),
        _mean_initial(pt),
        _first_machine_initial(pt),
        _last_machine_initial(pt),
    ]
    # Add random starts
    for s in range(6):
        candidates.append(_random_initial(n_jobs, 100 + s))

    # Evaluate all, pick best
    best_ms = float('inf')
    best_perm = candidates[0]
    for cand in candidates:
        ms = _makespan_fast(pt_list, cand)
        if ms < best_ms:
            best_ms = ms
            best_perm = cand

    # Sort candidates by makespan so we polish the best ones first
    candidate_results = []
    for cand in candidates:
        ms = _makespan_fast(pt_list, cand)
        candidate_results.append((ms, cand))
    candidate_results.sort(key=lambda x: x[0])

    # Multi-start local search: polish top candidates
    for ms0, cand in candidate_results:
        if time.time() > deadline - 20.0:
            break
        # Skip if too far from best
        if ms0 > best_ms * 1.2:
            continue

        # Insertion local search
        perm, ms = _local_search_insertion(pt_list, cand, deadline - 20.0, cost_matrix=cost_matrix)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

        # 2-opt segment reversal if time permits
        if time.time() < deadline - 15.0:
            perm2, ms2 = _local_search_2opt(pt_list, perm, deadline - 15.0, cost_matrix=cost_matrix)
            if ms2 < best_ms:
                best_ms = ms2
                best_perm = perm2

    # Run Genetic Algorithm with the best solutions as seeds
    if time.time() < deadline - 10.0:
        ga_pops = [cand for _, cand in candidate_results[:10]]
        perm, ms = _genetic_algorithm(pt_list, ga_pops, deadline - 10.0, pop_size=30, generations=50, cost_matrix=cost_matrix)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Simulated annealing from best to escape local optima
    if time.time() < deadline - 8.0:
        perm, ms = _simulated_annealing(pt_list, best_perm, deadline - 8.0, max_iter=200000, seed=7, cost_matrix=cost_matrix)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Final polish with insertion local search
    if time.time() < deadline - 2.0:
        perm, ms = _local_search_insertion(pt_list, best_perm, deadline - 1.0, cost_matrix=cost_matrix)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Final 2-opt polish
    if time.time() < deadline - 0.5:
        perm, ms = _local_search_2opt(pt_list, best_perm, deadline - 0.5, cost_matrix=cost_matrix)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    return best_perm
# EVOLVE-BLOCK-END
