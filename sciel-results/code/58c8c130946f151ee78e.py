import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    D = 11
    
    # Generate all integer vectors with norm squared 4
    # Type 1: 2 * e_i (and -2 * e_i)
    # Type 2: four coordinates are +/-1, rest 0
    
    candidates = []
    
    # Type 1: One 2, rest 0
    for i in range(D):
        for s in [1, -1]:
            v = np.zeros(D, dtype=np.int32)
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four 1s, rest 0
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int32)
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Conflict matrix: 1 if dot > 2 (and i != j)
    Conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    # Helper: check if a set of indices forms an independent set
    def is_independent(indices):
        if len(indices) <= 1:
            return True
        idx_arr = np.array(indices, dtype=np.int64)
        sub = Conflict[np.ix_(idx_arr, idx_arr)]
        return np.sum(sub) == 0  # diagonal is 0, off-diagonal should be 0
    
    # Min-degree greedy with random tie-breaking
    def run_greedy(seed):
        if time.time() > deadline:
            return []
            
        rng = np.random.default_rng(seed)
        active = np.ones(N, dtype=bool)
        deg = Conflict.sum(axis=1).astype(np.int64)
        selected = []
        
        while active.any():
            if time.time() > deadline:
                break
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
                
            active_deg = deg[active_indices]
            min_deg = np.min(active_deg)
            candidates_for_pick = active_indices[active_deg == min_deg]
            idx = rng.choice(candidates_for_pick)
            
            selected.append(idx)
            
            neighbors_mask = Conflict[idx].astype(bool)
            removed_mask = np.zeros(N, dtype=bool)
            removed_mask[idx] = True
            removed_mask[neighbors_mask & active] = True
            
            removed_indices = np.where(removed_mask)[0]
            
            if len(removed_indices) > 0:
                removal_counts = Conflict[removed_indices, :].sum(axis=0).astype(np.int64)
                deg[active] -= removal_counts[active]
                
            active[removed_mask] = False
            
        return selected
    
    # Local search: try to add vectors to a set
    def local_search_add(selected_list, max_iters=100):
        if time.time() > deadline:
            return selected_list
            
        selected_set = set(selected_list)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Find all vectors compatible with the entire selected set
            sel_arr = np.array(selected_list, dtype=np.int64) if selected_list else np.array([], dtype=np.int64)
            if len(sel_arr) > 0:
                conflicts_with_selected = Conflict[sel_arr, :].max(axis=0).astype(bool)
            else:
                conflicts_with_selected = np.zeros(N, dtype=bool)
                
            # Candidates: not in selected, not conflicting
            candidates_mask = ~conflicts_with_selected
            candidates_mask[list(selected_set)] = False
            candidate_indices = np.where(candidates_mask)[0]
            
            if len(candidate_indices) == 0:
                break
                
            # Try adding one (random order)
            rng = np.random.default_rng()
            rng.shuffle(candidate_indices)
            
            for cidx in candidate_indices:
                # Check if adding this creates no new conflicts (should be fine by construction)
                selected_list.append(cidx)
                selected_set.add(cidx)
                improved = True
                break
                
        return selected_list
    
    # Genetic algorithm with population
    def genetic_algorithm(pop_size=10, generations=50):
        if time.time() > deadline:
            return []
            
        # Initialize population with greedy solutions
        population = []
        for i in range(min(pop_size, 20)):
            if time.time() > deadline:
                break
            sol = run_greedy(i * 100 + 7)
            if len(sol) > 0:
                # Apply local search
                sol = local_search_add(sol)
                population.append(set(sol))
                
        if not population:
            return []
        
        best_set = max(population, key=len)
        best_size = len(best_set)
        
        for gen in range(generations):
            if time.time() > deadline - 5:
                break
                
            new_population = []
            
            # Selection: keep best individuals
            population.sort(key=len, reverse=True)
            elite = population[:max(1, pop_size // 3)]
            new_population.extend(elite)
            
            # Crossover and mutation to fill rest
            while len(new_population) < pop_size:
                if time.time() > deadline - 5:
                    break
                    
                # Select parents (tournament)
                p1 = population[np.random.randint(len(population))]
                p2 = population[np.random.randint(len(population))]
                
                # Crossover: union of compatible elements
                child = set()
                all_candidates = list(p1) + list(p2)
                # Remove duplicates
                all_candidates = list(set(all_candidates))
                
                # Greedily add compatible elements
                for idx in all_candidates:
                    # Check if idx is compatible with all in child
                    if len(child) > 0:
                        child_arr = np.array(list(child), dtype=np.int64)
                        if Conflict[idx, child_arr].any():
                            continue
                    child.add(idx)
                    
                # Mutation: try to add random compatible vectors
                if len(child) > 0:
                    child_arr = np.array(list(child), dtype=np.int64)
                    conflicts = Conflict[child_arr, :].max(axis=0).astype(bool)
                else:
                    conflicts = np.zeros(N, dtype=bool)
                    
                conflicts[list(child)] = True
                candidates = np.where(~conflicts)[0]
                
                if len(candidates) > 0:
                    # Add 1-3 random candidates
                    n_add = min(3, len(candidates))
                    if n_add > 0:
                        chosen = np.random.choice(candidates, size=n_add, replace=False)
                        for c in chosen:
                            # Verify compatibility
                            if len(child) > 0:
                                if Conflict[c, np.array(list(child), dtype=np.int64)].any():
                                    continue
                            child.add(c)
                            
                new_population.append(child)
                
            population = new_population
            
            # Update best
            for sol in population:
                if len(sol) > best_size:
                    best_size = len(sol)
                    best_set = set(sol)
                    
        return list(best_set)
    
    # Phase 1: Get initial solutions via greedy
    initial_solutions = []
    for r in range(30):
        if time.time() > deadline - 30:
            break
        sol = run_greedy(r * 100 + 42)
        if len(sol) > 0:
            sol = local_search_add(sol)
            initial_solutions.append(set(sol))
            
    if not initial_solutions:
        # Fallback
        return candidates[:1]
    
    best_set = max(initial_solutions, key=len)
    best_size = len(best_set)
    
    # Phase 2: Genetic algorithm
    ga_result = genetic_algorithm(pop_size=12, generations=40)
    if len(ga_result) > best_size:
        best_set = set(ga_result)
        best_size = len(ga_result)
    
    # Phase 3: Final local search on best
    best_list = list(best_set)
    best_list = local_search_add(best_list)
    if len(best_list) > best_size:
        best_set = set(best_list)
        best_size = len(best_list)
    
    # Return as integer array
    if best_size > 0:
        result = candidates[list(best_set)].astype(np.int64)
    else:
        result = candidates[:1].astype(np.int64)
        
    return result
