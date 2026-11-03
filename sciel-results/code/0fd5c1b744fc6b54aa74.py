import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Generate Weight 2, 3, and 4 vectors
    # Max norm squared is 4 (from Weight 4)
    # Condition: dist_sq >= 4
    candidates = []
    norms_sq = []
    
    # Weight 2
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            v[positions[0]] = signs[0]
            v[positions[1]] = signs[1]
            candidates.append(v)
            norms_sq.append(2)
            
    # Weight 3
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(3)
            
    # Weight 4
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(4)
            
    candidates = np.array(candidates, dtype=np.int64)
    norms_sq = np.array(norms_sq, dtype=np.int64)
    n_candidates = len(candidates)
    
    # Compute dot products
    dots = np.dot(candidates, candidates.T)
    
    # Determine conflicts
    # dist_sq[i, j] = norms_sq[i] + norms_sq[j] - 2*dots[i, j]
    # Conflict if dist_sq[i, j] < 4
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflicts = dist_sq < 4
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbors as bitsets for fast operations
    # Use Python integers as bitsets
    neighbors_bitset = [0] * n_candidates
    for i in range(n_candidates):
        bit = 0
        for j in range(n_candidates):
            if conflicts[i, j]:
                bit |= (1 << j)
        neighbors_bitset[i] = bit
    
    # Also keep sets for some operations
    neighbors = [set() for _ in range(n_candidates)]
    for i in range(n_candidates):
        neigh_indices = np.where(conflicts[i])[0]
        neighbors[i] = set(neigh_indices.tolist())
        
    def greedy_min_degree():
        """
        Greedy selection picking the vertex with the minimum number of 
        conflicts with the *remaining* available vertices.
        """
        available = set(range(n_candidates))
        selected = []
        
        while available:
            if time.time() > deadline - 2.0:
                break
            
            # Find vertex with min degree in available subgraph
            # Sample if large to save time
            if len(available) > 300:
                sample = random.sample(list(available), 300)
            else:
                sample = list(available)
                
            best_v = None
            min_deg = float('inf')
            for v in sample:
                deg = len(neighbors[v] & available)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            
            if best_v is None:
                break
                
            selected.append(best_v)
            available.discard(best_v)
            # Remove neighbors of best_v from available
            for n in neighbors[best_v]:
                if n in available:
                    available.discard(n)
                    
        return selected

    def local_search(S):
        """
        Improve an independent set S using addition and 1-for-k swaps.
        """
        S_set = set(S)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # 1. Try to add any vertex that fits
            order = random.sample(range(n_candidates), n_candidates)
            
            for v in order:
                if time.time() > deadline - 5.0:
                    break
                if v not in S_set:
                    if not (neighbors[v] & S_set):
                        S_set.add(v)
                        improved = True
            
            # 2. Try 1-for-k swaps
            if not improved:
                S_list = list(S_set)
                random.shuffle(S_list)
                
                for v in S_list:
                    if time.time() > deadline - 5.0:
                        break
                    
                    test_set = S_set - {v}
                    
                    candidates_for_swap = []
                    for u in range(n_candidates):
                        if u not in test_set and u != v:
                            if not (neighbors[u] & test_set):
                                candidates_for_swap.append(u)
                    
                    if not candidates_for_swap:
                        continue
                        
                    available_swap = set(candidates_for_swap)
                    added_swap = []
                    
                    while available_swap:
                        if time.time() > deadline - 5.0:
                            break
                        if len(available_swap) > 50:
                            sample = random.sample(list(available_swap), 50)
                        else:
                            sample = list(available_swap)
                            
                        best_u = None
                        min_deg = float('inf')
                        for u in sample:
                            deg = len(neighbors[u] & available_swap)
                            if deg < min_deg:
                                min_deg = deg
                                best_u = u
                                
                        if best_u is None:
                            break
                            
                        added_swap.append(best_u)
                        available_swap.discard(best_u)
                        for n in neighbors[best_u]:
                            if n in available_swap:
                                available_swap.discard(n)
                        
                        if len(added_swap) >= 3: 
                             break

                    if len(added_swap) > 1:
                        S_set = test_set.union(added_swap)
                        improved = True

        return list(S_set)

    best_selected = []
    np.random.seed(42)
    random.seed(42)
    
    # Phase 1: Randomized Greedy Restarts
    greedy_deadline = start_time + (deadline - start_time) * 0.4
    
    num_trials = 0
    while time.time() < greedy_deadline:
        num_trials += 1
        random.seed(42 + num_trials)
        sel = greedy_min_degree()
        if len(sel) > len(best_selected):
            best_selected = sel
            
    # Phase 2: Genetic Algorithm
    if best_selected and time.time() < deadline - 10.0:
        # Initialize population from greedy results and random sets
        pop_size = 20
        population = []
        
        # Add best greedy result
        population.append(set(best_selected))
        
        # Add some random independent sets
        for _ in range(pop_size - 1):
            random_set = set()
            available = set(range(n_candidates))
            while available and time.time() < deadline - 10.0:
                v = random.choice(list(available))
                random_set.add(v)
                available.discard(v)
                for n in neighbors[v]:
                    if n in available:
                        available.discard(n)
            population.append(random_set)
        
        # Genetic algorithm loop
        ga_deadline = deadline - 5.0
        while time.time() < ga_deadline:
            # Selection: keep top half
            population.sort(key=len, reverse=True)
            population = population[:pop_size//2]
            
            # Crossover and mutation
            new_population = []
            while len(new_population) < pop_size and time.time() < ga_deadline:
                # Pick two parents
                p1, p2 = random.sample(population, 2)
                
                # Crossover: union with some probability
                if random.random() < 0.5:
                    child = p1.union(p2)
                else:
                    child = p1.intersection(p2)
                
                # Repair: remove conflicts
                child_list = list(child)
                random.shuffle(child_list)
                repaired = set()
                for v in child_list:
                    if not (neighbors[v] & repaired):
                        repaired.add(v)
                
                # Mutation: try to add random vertices
                for _ in range(10):
                    v = random.randint(0, n_candidates - 1)
                    if v not in repaired and not (neighbors[v] & repaired):
                        repaired.add(v)
                
                new_population.append(repaired)
            
            population = new_population
            
            # Track best
            current_best = max(population, key=len)
            if len(current_best) > len(best_selected):
                best_selected = list(current_best)
            
            # Local search on best
            if time.time() < ga_deadline - 5.0:
                refined = local_search(best_selected)
                if len(refined) > len(best_selected):
                    best_selected = refined
    
    # Phase 3: Final local search
    if best_selected and time.time() < deadline - 2.0:
        refined = local_search(best_selected)
        if len(refined) > len(best_selected):
            best_selected = refined

    # Construct final array
    if not best_selected:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    result = candidates[best_selected]
    return result
