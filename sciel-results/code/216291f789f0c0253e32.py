"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan_fast(perm, pt):
    """Optimized makespan computation using 2D array for speed."""
    n_machines = pt.shape[1]
    n_jobs = len(perm)
    
    if n_machines == 0 or n_jobs == 0:
        return 0.0
    
    # Build completion matrix C[j][m] = completion time of job j on machine m
    # C[j][0] = C[j-1][0] + pt[perm[j], 0]
    # C[j][m] = max(C[j][m-1], C[j-1][m]) + pt[perm[j], m]
    C = np.zeros((n_jobs, n_machines), dtype=np.float64)
    pt_perm = pt[np.asarray(perm, dtype=np.int64)]
    
    # First machine: cumulative sum
    C[:, 0] = np.cumsum(pt_perm[:, 0])
    
    if n_machines > 1:
        for m in range(1, n_machines):
            # C[j][m] = max(C[j][m-1], C[j-1][m]) + pt_perm[j][m]
            # For j=0: C[0][m] = C[0][m-1] + pt_perm[0][m]
            # For j>0: C[j][m] = max(C[j][m-1], C[j-1][m]) + pt_perm[j][m]
            C[0, m] = C[0, m-1] + pt_perm[0, m]
            for j in range(1, n_jobs):
                c_prev_m = C[j-1, m]
                c_curr_m1 = C[j, m-1]
                C[j, m] = (c_curr_m1 if c_curr_m1 > c_prev_m else c_prev_m) + pt_perm[j, m]
    
    return C[n_jobs-1, n_machines-1]


def _neh_heuristic(pt, rng=None):
    """NEH (Nawaz, Enscore, Ham) heuristic for flow shop with optional randomization."""
    n_jobs = pt.shape[0]
    n_machines = pt.shape[1]
    
    # Sort jobs by total processing time descending
    totals = pt.sum(axis=1)
    if rng is not None:
        # Add small random perturbation to totals for diversity
        noise = rng.random(n_jobs) * 0.01 * np.max(totals)
        totals = totals + noise
    order = list(np.argsort(-totals))
    
    # Insert each job into best position
    perm = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_fast(candidate, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return perm


def _johnsons_rule(pt):
    """Johnson's rule for 2-machine equivalent using first/second half sums."""
    n_jobs, n_machines = pt.shape
    if n_machines < 2:
        return list(range(n_jobs))
    
    mid = n_machines // 2
    a = pt[:, :mid].sum(axis=1)
    b = pt[:, mid:].sum(axis=1)
    
    left = []
    right = []
    for j in range(n_jobs):
        if a[j] <= b[j]:
            left.append((a[j], j))
        else:
            right.append((b[j], j))
    left.sort(key=lambda x: x[0])
    right.sort(key=lambda x: -x[0])
    johnson_order = [x[1] for x in left] + [x[1] for x in right]
    return johnson_order


def _local_search_insertion(perm, pt, deadline=None):
    """Insertion-based local search with swap moves."""
    n_jobs = len(perm)
    best_perm = list(perm)
    best_ms = _makespan_fast(best_perm, pt)
    improved = True
    
    while improved:
        improved = False
        # Phase 1: insertion moves
        for i in range(n_jobs):
            if deadline is not None and time.time() > deadline:
                break
            job = best_perm[i]
            for j in range(n_jobs):
                if i == j:
                    continue
                candidate = best_perm[:i] + best_perm[i+1:]
                insert_pos = j if j < i else j - 1
                candidate = candidate[:insert_pos] + [job] + candidate[insert_pos:]
                ms = _makespan_fast(candidate, pt)
                if ms < best_ms - 1e-12:
                    best_ms = ms
                    best_perm = candidate
                    improved = True
                    break
            if deadline is not None and time.time() > deadline:
                break
        
        # Phase 2: swap moves
        if not improved and n_jobs > 1:
            for i in range(n_jobs):
                if deadline is not None and time.time() > deadline:
                    break
                for j in range(i + 1, n_jobs):
                    if deadline is not None and time.time() > deadline:
                        break
                    candidate = list(best_perm)
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    ms = _makespan_fast(candidate, pt)
                    if ms < best_ms - 1e-12:
                        best_ms = ms
                        best_perm = candidate
                        improved = True
                        break
            if deadline is not None and time.time() > deadline:
                break
    
    return best_perm, best_ms


def _order_crossover(p1, p2, rng):
    """Order Crossover (OX1) for permutations."""
    n = len(p1)
    if n < 3:
        return list(p1)
    
    i1, i2 = sorted(rng.integers(0, n, size=2))
    if i1 == i2:
        i2 = (i1 + 1) % n
    
    child = [None] * n
    # Copy segment from p1
    for i in range(i1, i2 + 1):
        child[i] = p1[i]
    
    # Fill remaining positions with elements from p2 in order
    remaining = [x for x in p2 if x not in child]
    pos = (i2 + 1) % n
    for x in remaining:
        child[pos] = x
        pos = (pos + 1) % n
    
    return child


def _mutation_swap(perm, rng, n_swaps=1):
    """Swap mutation: randomly swap n_swaps pairs of positions."""
    n = len(perm)
    if n < 2:
        return list(perm)
    mutated = list(perm)
    for _ in range(n_swaps):
        i, j = rng.integers(0, n, size=2)
        if i != j:
            mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


def _genetic_algorithm(pt, deadline, rng, pop_size=20, generations=50):
    """Genetic Algorithm with OX crossover and swap mutation."""
    n_jobs = pt.shape[0]
    if n_jobs < 3:
        return None, None
    
    # Initialize population with diverse heuristics
    population = []
    # NEH variants
    for _ in range(max(1, pop_size // 3)):
        population.append(_neh_heuristic(pt, rng))
    # Johnson's rule
    population.append(_johnsons_rule(pt))
    # SPT
    population.append(list(np.argsort(pt[:, 0])))
    # LPT on last machine
    population.append(list(np.argsort(-pt[:, -1])))
    # Random permutations
    while len(population) < pop_size:
        random_perm = list(rng.permutation(n_jobs))
        population.append(random_perm)
    
    # Evaluate initial population
    evals = [(perm, _makespan_fast(perm, pt)) for perm in population]
    evals.sort(key=lambda x: x[1])
    best_perm, best_ms = evals[0]
    
    # GA loop
    for gen in range(generations):
        if time.time() > deadline - 5.0:
            break
        
        # Selection: tournament selection
        def select_tournament():
            idx1 = rng.integers(0, len(evals))
            idx2 = rng.integers(0, len(evals))
            if evals[idx1][1] < evals[idx2][1]:
                return evals[idx1][0]
            else:
                return evals[idx2][0]
        
        # Create offspring
        offspring = []
        n_offspring = pop_size
        for _ in range(n_offspring):
            p1 = select_tournament()
            p2 = select_tournament()
            
            # Crossover
            child = _order_crossover(p1, p2, rng)
            
            # Mutation
            if rng.random() < 0.5:
                n_swaps = 1 if n_jobs < 10 else 2
                child = _mutation_swap(child, rng, n_swaps)
            
            offspring.append(child)
        
        # Evaluate offspring
        for child in offspring:
            if time.time() > deadline - 5.0:
                break
            ms = _makespan_fast(child, pt)
            if ms < best_ms - 1e-12:
                best_ms = ms
                best_perm = child
            # Add to population
            evals.append((child, ms))
        
        # Truncate population to keep best
        evals.sort(key=lambda x: x[1])
        evals = evals[:pop_size]
        
        # Elitism: ensure best is in population
        if evals[0][1] > best_ms + 1e-12:
            evals[0] = (best_perm, best_ms)
    
    return best_perm, best_ms


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    deadline = start_time + 147.0  # Safety margin before 150s limit
    
    pt = np.asarray(processing_times)
    n_jobs, n_machines = pt.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    rng = np.random.default_rng(42)
    
    # Start with NEH heuristic
    best_perm = _neh_heuristic(pt)
    best_ms = _makespan_fast(best_perm, pt)
    
    # Local search with insertion moves
    if time.time() < deadline - 10.0:
        best_perm, best_ms = _local_search_insertion(best_perm, pt, deadline)
    
    # Try alternative heuristics and local search
    if time.time() < deadline - 15.0 and n_jobs > 5:
        # Try SPT
        spt_order = list(np.argsort(pt[:, 0]))
        if spt_order != best_perm:
            alt_perm = list(spt_order)
            alt_ms = _makespan_fast(alt_perm, pt)
            if time.time() < deadline - 10.0:
                alt_perm, alt_ms = _local_search_insertion(alt_perm, pt, deadline)
            if alt_ms < best_ms - 1e-12:
                best_perm = alt_perm
                best_ms = alt_ms
        
        # Try LPT on last machine
        lpt_last = list(np.argsort(-pt[:, -1]))
        if lpt_last != best_perm:
            alt_perm = list(lpt_last)
            alt_ms = _makespan_fast(alt_perm, pt)
            if time.time() < deadline - 10.0:
                alt_perm, alt_ms = _local_search_insertion(alt_perm, pt, deadline)
            if alt_ms < best_ms - 1e-12:
                best_perm = alt_perm
                best_ms = alt_ms
        
        # Try Johnson's rule
        if n_machines >= 2 and time.time() < deadline - 15.0:
            johnson_order = _johnsons_rule(pt)
            if johnson_order != best_perm:
                alt_perm = list(johnson_order)
                alt_ms = _makespan_fast(alt_perm, pt)
                if time.time() < deadline - 10.0:
                    alt_perm, alt_ms = _local_search_insertion(alt_perm, pt, deadline)
                if alt_ms < best_ms - 1e-12:
                    best_perm = alt_perm
                    best_ms = alt_ms
    
    # Run Genetic Algorithm if time allows
    if time.time() < deadline - 20.0 and n_jobs > 5:
        ga_perm, ga_ms = _genetic_algorithm(pt, deadline, rng, pop_size=20, generations=30)
        if ga_perm is not None and ga_ms < best_ms - 1e-12:
            best_perm = ga_perm
            best_ms = ga_ms
    
    # Final polish: local search on best found
    if time.time() < deadline - 3.0:
        best_perm, best_ms = _local_search_insertion(best_perm, pt, deadline)
    
    return list(best_perm)
# EVOLVE-BLOCK-END
