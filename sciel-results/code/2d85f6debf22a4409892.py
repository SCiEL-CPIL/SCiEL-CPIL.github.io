import numpy as np
import time
from itertools import permutations


def _makespan_fast(perm, pt):
    """Faster makespan using numpy where possible."""
    n = len(perm)
    m = pt.shape[1]
    if n == 0:
        return 0.0
    if m == 1:
        return pt[perm, 0].sum()
    
    sub = pt[perm, :]
    comp = np.zeros((n, m))
    comp[0, 0] = sub[0, 0]
    for j in range(1, m):
        comp[0, j] = comp[0, j-1] + sub[0, j]
    for i in range(1, n):
        comp[i, 0] = comp[i-1, 0] + sub[i, 0]
        for j in range(1, m):
            comp[i, j] = max(comp[i-1, j], comp[i, j-1]) + sub[i, j]
    
    return comp[-1, -1]


def _flow_time(perm, pt):
    """Compute total flow time (sum of completion times of all jobs)."""
    n = len(perm)
    m = pt.shape[1]
    if n == 0:
        return 0.0
    
    sub = pt[perm, :]
    comp = np.zeros((n, m))
    comp[0, 0] = sub[0, 0]
    for j in range(1, m):
        comp[0, j] = comp[0, j-1] + sub[0, j]
    for i in range(1, n):
        comp[i, 0] = comp[i-1, 0] + sub[i, 0]
        for j in range(1, m):
            comp[i, j] = max(comp[i-1, j], comp[i, j-1]) + sub[i, j]
    
    # Flow time is sum of completion times on last machine
    return comp[:, -1].sum()


def _eval_objectives(perm, pt):
    """Evaluate both makespan and flow time."""
    n = len(perm)
    m = pt.shape[1]
    if n == 0:
        return 0.0, 0.0
    if m == 1:
        ms = pt[perm, 0].sum()
        ft = np.cumsum(pt[perm, 0]).sum()
        return ms, ft
    
    sub = pt[perm, :]
    comp = np.zeros((n, m))
    comp[0, 0] = sub[0, 0]
    for j in range(1, m):
        comp[0, j] = comp[0, j-1] + sub[0, j]
    for i in range(1, n):
        comp[i, 0] = comp[i-1, 0] + sub[i, 0]
        for j in range(1, m):
            comp[i, j] = max(comp[i-1, j], comp[i, j-1]) + sub[i, j]
    
    ms = comp[-1, -1]
    ft = comp[:, -1].sum()
    return ms, ft


def _neh_order(pt):
    """NEH heuristic."""
    n = pt.shape[0]
    m = pt.shape[1]
    totals = pt.sum(axis=1)
    order = np.argsort(-totals)
    
    perm = [order[0]]
    
    for idx in range(1, n):
        job = order[idx]
        best_pos = 0
        best_ms = float('inf')
        
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_fast(new_perm, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        perm.insert(best_pos, job)
    
    return perm


def _brute_force(pt):
    n = pt.shape[0]
    if n > 10:
        return None
    best_perm = None
    best_ms = float('inf')
    for perm in permutations(range(n)):
        ms = _makespan_fast(list(perm), pt)
        if ms < best_ms:
            best_ms = ms
            best_perm = list(perm)
    return best_perm


def _local_search_insertion(perm, pt, time_limit=10.0):
    """Local search using insertion moves."""
    n = len(perm)
    if n <= 1:
        return perm
    
    best_perm = list(perm)
    best_ms = _makespan_fast(best_perm, pt)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n):
            if (time.time() - start_time) >= time_limit:
                break
            job = best_perm[i]
            rest = best_perm[:i] + best_perm[i+1:]
            for j in range(n):
                if j == i or (j > 0 and j == i + 1):
                    continue
                new_perm = rest[:j] + [job] + rest[j:]
                ms = _makespan_fast(new_perm, pt)
                if ms < best_ms - 1e-9:
                    best_perm = new_perm
                    best_ms = ms
                    improved = True
                    i = -1
                    break
    return best_perm


def _non_dominated_sort(fronts, objectives):
    """Simple non-dominated sorting. Returns list of fronts (each a list of indices)."""
    n = len(objectives)
    domination_count = np.zeros(n, dtype=int)
    dominated_set = [set() for _ in range(n)]
    fronts = []
    
    for i in range(n):
        for j in range(i + 1, n):
            # Check if i dominates j
            i_dom_j = all(objectives[i][k] <= objectives[j][k] for k in range(len(objectives[i])))
            j_dom_i = all(objectives[j][k] <= objectives[i][k] for k in range(len(objectives[j])))
            i_strict = any(objectives[i][k] < objectives[j][k] for k in range(len(objectives[i])))
            j_strict = any(objectives[j][k] < objectives[i][k] for k in range(len(objectives[j])))
            
            if i_dom_j and i_strict:
                domination_count[j] += 1
                dominated_set[i].add(j)
            elif j_dom_i and j_strict:
                domination_count[i] += 1
                dominated_set[j].add(i)
    
    # First front
    front = [i for i in range(n) if domination_count[i] == 0]
    if front:
        fronts.append(front)
    
    current_front = front
    while current_front:
        next_front = set()
        for i in current_front:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.add(j)
        if next_front:
            current_front = list(next_front)
            fronts.append(current_front)
        else:
            break
    
    return fronts


def _crowding_distance(front_indices, objectives):
    """Compute crowding distance for individuals in a front."""
    if len(front_indices) <= 2:
        return {i: float('inf') for i in front_indices}
    
    distances = {i: 0.0 for i in front_indices}
    n_obj = len(objectives[front_indices[0]])
    
    for k in range(n_obj):
        sorted_idx = sorted(front_indices, key=lambda i: objectives[i][k])
        distances[sorted_idx[0]] = float('inf')
        distances[sorted_idx[-1]] = float('inf')
        
        obj_range = objectives[sorted_idx[-1]][k] - objectives[sorted_idx[0]][k]
        if obj_range == 0:
            continue
            
        for i in range(1, len(sorted_idx) - 1):
            prev_idx = sorted_idx[i-1]
            next_idx = sorted_idx[i+1]
            distances[sorted_idx[i]] += (objectives[next_idx][k] - objectives[prev_idx][k]) / obj_range
    
    return distances


def _moa_ea(pt, initial_perms, time_limit=30.0):
    """Multi-Objective Evolutionary Algorithm."""
    n = pt.shape[0]
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Initialize population
    population = []
    objectives = []
    
    for perm in initial_perms:
        ms, ft = _eval_objectives(perm, pt)
        population.append(list(perm))
        objectives.append([ms, ft])
    
    # Add some random individuals
    while len(population) < 20:
        perm = list(rng.permutation(n))
        ms, ft = _eval_objectives(perm, pt)
        population.append(perm)
        objectives.append([ms, ft])
    
    pop_size = len(population)
    
    while (time.time() - start_time) < time_limit:
        # Non-dominated sorting
        fronts = _non_dominated_sort(population, objectives)
        
        # Select parents from best fronts
        selected = []
        for front in fronts:
            if len(selected) + len(front) >= pop_size // 2:
                # Use crowding distance to select
                cd = _crowding_distance(front, objectives)
                sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                needed = pop_size // 2 - len(selected)
                selected.extend(sorted_front[:needed])
                break
            else:
                selected.extend(front)
        
        # Generate offspring
        offspring = []
        offspring_obj = []
        
        for _ in range(pop_size // 2):
            # Crossover: pick two parents, do uniform crossover
            if len(selected) < 2:
                parent1 = rng.integers(0, pop_size)
                parent2 = rng.integers(0, pop_size)
            else:
                parent1 = selected[rng.integers(0, len(selected))]
                parent2 = selected[rng.integers(0, len(selected))]
            
            p1 = population[parent1]
            p2 = population[parent2]
            
            # Order crossover
            if n < 2:
                child = list(p1)
            else:
                i, j = sorted(rng.choice(n, 2, replace=False))
                child = [None] * n
                child[i:j+1] = p1[i:j+1]
                remaining = [x for x in p2 if x not in child]
                pos = 0
                for k in range(n):
                    if child[k] is None:
                        child[k] = remaining[pos]
                        pos += 1
            
            # Mutation: insertion move
            if n > 1 and rng.random() < 0.7:
                a = rng.integers(0, n)
                b = rng.integers(0, n)
                if a != b:
                    job = child[a]
                    child.pop(a)
                    child.insert(b, job)
            
            ms, ft = _eval_objectives(child, pt)
            offspring.append(child)
            offspring_obj.append([ms, ft])
        
        # Combine and select
        combined_pop = population + offspring
        combined_obj = objectives + offspring_obj
        
        # Non-dominated sort
        fronts = _non_dominated_sort(combined_pop, combined_obj)
        
        new_pop = []
        new_obj = []
        for front in fronts:
            if len(new_pop) + len(front) >= pop_size:
                cd = _crowding_distance(front, combined_obj)
                sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                needed = pop_size - len(new_pop)
                for idx in sorted_front[:needed]:
                    new_pop.append(combined_pop[idx])
                    new_obj.append(combined_obj[idx])
                break
            else:
                for idx in front:
                    new_pop.append(combined_pop[idx])
                    new_obj.append(combined_obj[idx])
        
        population = new_pop
        objectives = new_obj
        
        if (time.time() - start_time) >= time_limit - 5.0:
            break
    
    # Return best makespan solution
    best_idx = 0
    best_ms = objectives[0][0]
    for i in range(len(objectives)):
        if objectives[i][0] < best_ms:
            best_ms = objectives[i][0]
            best_idx = i
    
    return population[best_idx]


def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    
    if n == 0:
        return []
    if n == 1:
        return [0]
    if m == 1:
        return list(np.argsort(pt[:, 0]))
    
    # Brute force for small instances
    if n <= 10:
        return _brute_force(pt)
    
    deadline = 140.0  # Safety margin for 150s limit
    start_time = time.time()
    
    candidates = []
    
    # NEH
    neh_perm = _neh_order(pt)
    candidates.append(neh_perm)
    
    # Palmer
    palmer_keys = np.zeros(n)
    for j in range(n):
        for k in range(m):
            palmer_keys[j] += (n - 2*k - 1) * pt[j, k]
    palmer_perm = list(np.argsort(palmer_keys))
    candidates.append(palmer_perm)
    
    # Random (multiple)
    rng = np.random.default_rng(123)
    for _ in range(10):
        candidates.append(list(rng.permutation(n)))
    
    # Pick best initial
    best_perm = candidates[0]
    best_ms = _makespan_fast(best_perm, pt)
    for perm in candidates[1:]:
        ms = _makespan_fast(perm, pt)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    # MOEA for diversity exploration
    elapsed = time.time() - start_time
    if elapsed < deadline - 20.0:
        moea_budget = min(40.0, deadline - 20.0 - elapsed)
        # Use best candidates as initial population for MOEA
        moea_initial = candidates[:min(10, len(candidates))]
        moea_result = _moa_ea(pt, moea_initial, time_limit=moea_budget)
        moea_ms = _makespan_fast(moea_result, pt)
        if moea_ms < best_ms:
            best_perm = moea_result
            best_ms = moea_ms
    
    # Local Search on best solution
    elapsed = time.time() - start_time
    if elapsed < deadline - 10.0:
        ls_budget = min(30.0, deadline - 10.0 - elapsed)
        best_perm = _local_search_insertion(best_perm, pt, time_limit=ls_budget)
        best_ms = _makespan_fast(best_perm, pt)
    
    # Final polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 2.0:
        polish_budget = min(5.0, deadline - 2.0 - elapsed)
        best_perm = _local_search_insertion(best_perm, pt, time_limit=polish_budget)
        best_ms = _makespan_fast(best_perm, pt)
    
    return list(best_perm)
