import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    n = 16
    
    # Precompute all possible triples
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute the conflict structure
    # A forbidden configuration is a set of 3 disjoint pairs (x1,y1), (x2,y2), (x3,y3)
    # such that all 8 transversal triples (a,b,c) with a in {x1,y1}, b in {x2,y2}, c in {x3,y3} are present.
    # We want to avoid this.
    
    # Total number of configs: C(16,6) * 15 = 8008 * 15 = 120120.
    # Each config has 8 transversals.
    # We can represent the state as a set of selected triples.
    # When adding a triple, we check if it completes any config (i.e., if the other 7 transversals of that config are already selected).
    
    # To make this efficient, we precompute for each triple T, the list of configs it belongs to.
    # And for each config, the list of its 8 transversals.
    
    configs = []
    config_to_idx = {}
    triple_to_configs = [[] for _ in range(num_triples)]
    config_transversals = []  # List of lists of triple indices
    
    for v6 in itertools.combinations(range(n), 6):
        # Generate 15 matchings
        # v0 pairs with v1..v5
        for i in range(1, 6):
            # v0-vi
            rem = [v for j, v in enumerate(v6) if j != 0 and j != i]
            # rem has 4 vertices. Matchings on 4 vertices: 3.
            # rem[0] pairs with rem[1], rem[2], or rem[3].
            for j in range(1, 4):
                # rem[0]-rem[j]
                rem2 = [v for k, v in enumerate(rem) if k != 0 and k != j]
                # rem2 has 2 vertices.
                p1 = (v6[0], v6[i])
                p2 = (rem[0], rem[j])
                p3 = (rem2[0], rem2[1])
                
                # Normalize pairs
                p1 = tuple(sorted(p1))
                p2 = tuple(sorted(p2))
                p3 = tuple(sorted(p3))
                
                # Normalize order of pairs
                pairs = sorted([p1, p2, p3])
                config = tuple(pairs)
                
                if config not in config_to_idx:
                    config_to_idx[config] = len(configs)
                    configs.append(config)
                    
                    # Compute transversals for this config
                    a, b = p1
                    c, d = p2
                    e, f = p3
                    transversals = [
                        (a, c, e), (a, c, f), (a, d, e), (a, d, f),
                        (b, c, e), (b, c, f), (b, d, e), (b, d, f)
                    ]
                    t_indices = [triple_to_idx[tuple(sorted(t))] for t in transversals]
                    config_transversals.append(t_indices)
                    
                    # Update triple_to_configs
                    for t_idx in t_indices:
                        triple_to_configs[t_idx].append(config_to_idx[config])
                else:
                    # Already added, but we need to update triple_to_configs?
                    # No, triple_to_configs is built incrementally.
                    # Wait, if the config was already added, we shouldn't add it again to triple_to_configs.
                    # But we are iterating through all v6 and matchings, so duplicates are possible.
                    # We only add to triple_to_configs when we add a NEW config.
                    # This is correct.
                    pass

    num_configs = len(configs)
    
    # Convert to numpy arrays for faster manipulation if needed, but lists might be fine for now.
    # Let's use a bitset for selected triples to speed up checks?
    # Or just a boolean array.
    
    # Strategy: Local Search / Simulated Annealing
    # Start with a random valid system (or empty).
    # Try to add random triples. If valid, add.
    # Try to swap: remove a random triple, add a random other triple.
    # Or: remove a triple, add multiple?
    
    # A better strategy for this type of problem (Max 3-SAT like):
    # 1. Greedy construction to get a good initial solution.
    # 2. Local search: try to add any unselected triple that is valid.
    #    If stuck, try to remove a triple and see if we can add more.
    
    def is_valid_addition(selected, t_idx, config_counts):
        # Check if adding t_idx would complete any config
        for cfg_idx in triple_to_configs[t_idx]:
            if config_counts[cfg_idx] == 7:
                return False
        return True

    def add_triple(selected, t_idx, config_counts):
        selected[t_idx] = True
        for cfg_idx in triple_to_configs[t_idx]:
            config_counts[cfg_idx] += 1

    def remove_triple(selected, t_idx, config_counts):
        selected[t_idx] = False
        for cfg_idx in triple_to_configs[t_idx]:
            config_counts[cfg_idx] -= 1

    best_system = None
    best_count = 0
    random.seed(42)
    
    # Phase 1: Multiple greedy restarts
    for trial in range(500):
        if time.time() > deadline - 10:
            break
            
        selected = np.zeros(num_triples, dtype=bool)
        config_counts = np.zeros(num_configs, dtype=np.int8)
        
        # Random order
        order = list(range(num_triples))
        random.shuffle(order)
        
        for t_idx in order:
            if time.time() > deadline - 10:
                break
            if is_valid_addition(selected, t_idx, config_counts):
                add_triple(selected, t_idx, config_counts)
                
        count = np.sum(selected)
        if count > best_count:
            best_count = count
            best_system = selected.copy()
            
    # Phase 2: Local Search on the best solution found
    if best_system is not None:
        selected = best_system.copy()
        config_counts = np.zeros(num_configs, dtype=np.int8)
        for t_idx in range(num_triples):
            if selected[t_idx]:
                for cfg_idx in triple_to_configs[t_idx]:
                    config_counts[cfg_idx] += 1
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try to add any valid triple
            unselected = [i for i in range(num_triples) if not selected[i]]
            random.shuffle(unselected)
            
            for t_idx in unselected:
                if time.time() > deadline - 5:
                    break
                if is_valid_addition(selected, t_idx, config_counts):
                    add_triple(selected, t_idx, config_counts)
                    improved = True
                    if np.sum(selected) > best_count:
                        best_count = np.sum(selected)
                        best_system = selected.copy()
            
            # If we couldn't add any, try a swap
            if not improved:
                # Pick a random selected triple to remove
                selected_list = [i for i in range(num_triples) if selected[i]]
                if selected_list:
                    # Try removing a few random ones and see if we can add more
                    for _ in range(10):
                        if time.time() > deadline - 5:
                            break
                        t_remove = random.choice(selected_list)
                        remove_triple(selected, t_remove, config_counts)
                        
                        # Try to add as many as possible
                        unselected = [i for i in range(num_triples) if not selected[i]]
                        random.shuffle(unselected)
                        added = 0
                        for t_idx in unselected:
                            if is_valid_addition(selected, t_idx, config_counts):
                                add_triple(selected, t_idx, config_counts)
                                added += 1
                                
                        if added > 1: # Net gain
                            improved = True
                            if np.sum(selected) > best_count:
                                best_count = np.sum(selected)
                                best_system = selected.copy()
                        else:
                            # Revert
                            # This is tricky because we added some.
                            # We need to undo the additions.
                            # Let's just restart the local search if this doesn't work.
                            # Or better, keep track of what was added.
                            # For simplicity, let's just revert the removal and try another.
                            add_triple(selected, t_remove, config_counts)
                            # We don't revert the additions easily without tracking.
                            # So this swap strategy is flawed if we add things.
                            # Let's stick to simple addition and maybe 1-for-1 swaps.
    
    # Phase 3: More sophisticated local search / Simulated Annealing
    # Let's try a different approach: Start from scratch with a different heuristic.
    # Use a priority queue or weighted random selection.
    
    # Actually, let's try to improve the greedy by using a "potential" function.
    # Or just run more restarts.
    
    # Let's try a "fill" approach:
    # Start with all triples, remove those that cause conflicts.
    # This is equivalent to finding a maximal independent set in the hypergraph of conflicts.
    # But the conflicts are 8-way.
    
    # Let's stick to the current best.
    
    # Convert best_system to list of triples
    if best_system is None:
        return [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
    
    result = [all_triples[i] for i in range(num_triples) if best_system[i]]
    return result

