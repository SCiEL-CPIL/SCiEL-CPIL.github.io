"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def construct():
    """
    Construct a triple system on 16 points without the complete 3-partite K(2,2,2).
    Uses a local search with an efficient 'addable set' maintenance strategy.
    """
    n = 16
    num_triples = 560  # C(16,3)
    
    # Precompute all possible triples
    all_triples = []
    triple_index = {}
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triple = (i, j, k)
                triple_index[triple] = len(all_triples)
                all_triples.append(triple)
    
    # Precompute all K(2,2,2) configurations
    # A K(2,2,2) is defined by 3 disjoint pairs.
    # The 8 transversal triples are formed by picking one from each pair.
    # We need to enumerate all unique sets of 8 triples.
    
    k222_configs = []
    seen_configs = set()
    
    # To generate K(2,2,2), we pick 6 distinct points and partition them into 3 pairs.
    # There are 15 ways to partition 6 points into 3 pairs.
    # For each partition, the 8 triples are determined.
    
    def get_pairings(points):
        points = list(points)
        pairings = []
        def rec(remaining, current):
            if not remaining:
                pairings.append(current[:])
                return
            first = remaining[0]
            rest = remaining[1:]
            for i in range(len(rest)):
                second = rest[i]
                new_remaining = rest[:i] + rest[i+1:]
                rec(new_remaining, current + [(first, second)])
        rec(points, [])
        return pairings

    for six_points in combinations(range(n), 6):
        pairings = get_pairings(six_points)
        for pairing in pairings:
            p1, p2, p3 = pairing
            triples_in_config = []
            for a in p1:
                for b in p2:
                    for c in p3:
                        triple = tuple(sorted((a, b, c)))
                        idx = triple_index[triple]
                        triples_in_config.append(idx)
            config_key = tuple(sorted(triples_in_config))
            if config_key not in seen_configs:
                seen_configs.add(config_key)
                k222_configs.append(triples_in_config)
    
    num_configs = len(k222_configs)
    
    # Precompute for each triple, which configs it belongs to
    triple_to_configs = [[] for _ in range(num_triples)]
    for c_idx, config in enumerate(k222_configs):
        for t_idx in config:
            triple_to_configs[t_idx].append(c_idx)
            
    random.seed(42)
    start_time = time.time()
    time_limit = 115
    
    best_solution = None
    best_count = -1
    
    def greedy_init():
        sol = [False] * num_triples
        counts = [0] * num_configs
        # addable: set of indices that can be added without violating constraints
        # Initially, all are addable
        addable = set(range(num_triples))
        
        # Greedy: pick random from addable
        while addable:
            if time.time() - start_time > time_limit:
                break
            # Pick random
            idx = random.choice(list(addable))
            # Add idx
            sol[idx] = True
            addable.remove(idx)
            
            # Update counts and addable set
            for c_idx in triple_to_configs[idx]:
                counts[c_idx] += 1
                if counts[c_idx] == 7:
                    # Any other triple in this config that is currently 0 is no longer addable
                    for t_idx in k222_configs[c_idx]:
                        if t_idx != idx and not sol[t_idx] and t_idx in addable:
                            addable.remove(t_idx)
                            
        return sol, counts, addable, sum(sol)
    
    def add_triple(sol, counts, addable, idx):
        sol[idx] = True
        addable.remove(idx)
        for c_idx in triple_to_configs[idx]:
            counts[c_idx] += 1
            if counts[c_idx] == 7:
                for t_idx in k222_configs[c_idx]:
                    if t_idx != idx and not sol[t_idx] and t_idx in addable:
                        addable.remove(t_idx)
                        
    def remove_triple(sol, counts, addable, idx):
        sol[idx] = False
        for c_idx in triple_to_configs[idx]:
            counts[c_idx] -= 1
            if counts[c_idx] == 6:
                # Triples in this config that are 0 might become addable
                for t_idx in k222_configs[c_idx]:
                    if t_idx != idx and not sol[t_idx]:
                        # Check if t_idx is safe to add
                        # It is safe if for all C' containing t_idx, counts[C'] < 7
                        safe = True
                        for c_prime in triple_to_configs[t_idx]:
                            if counts[c_prime] == 7:
                                safe = False
                                break
                        if safe:
                            addable.add(t_idx)
        # Note: idx itself is not added to addable here, it's just removed from sol.
        # We might want to add it back to addable if it's safe?
        # Actually, if we remove it, it becomes 0. If it's safe to add, it should be in addable.
        # Let's check if idx is safe to add now.
        safe = True
        for c_idx in triple_to_configs[idx]:
            if counts[c_idx] == 7:
                safe = False
                break
        if safe:
            addable.add(idx)

    for restart in range(200):
        if time.time() - start_time > time_limit:
            break
            
        sol, counts, addable, current_count = greedy_init()
        
        if current_count > best_count:
            best_count = current_count
            best_solution = sol[:]
            
        stagnation_counter = 0
        
        while time.time() - start_time < time_limit:
            if addable:
                # Pick random from addable
                idx = random.choice(list(addable))
                add_triple(sol, counts, addable, idx)
                current_count += 1
                stagnation_counter = 0
                
                if current_count > best_count:
                    best_count = current_count
                    best_solution = sol[:]
            else:
                # Stuck: no addable triples.
                # We need to remove a triple to create space.
                # Heuristic: Pick a triple that is in many tight (count 7) configs.
                # Or pick a random triple from a tight config.
                
                # Find all tight configs
                tight_configs = [c for c in range(num_configs) if counts[c] == 7]
                
                if not tight_configs:
                    # This shouldn't happen if addable is empty and we have triples,
                    # unless all triples are 0? No, if addable is empty, it means
                    # every 0-triple is blocked by some count 7 config.
                    # So there must be tight configs.
                    break
                
                # Pick a random tight config
                c_idx = random.choice(tight_configs)
                # Pick a random active triple in this config
                active_in_config = [t for t in k222_configs[c_idx] if sol[t]]
                if not active_in_config:
                    # Should not happen
                    break
                kick_idx = random.choice(active_in_config)
                
                remove_triple(sol, counts, addable, kick_idx)
                current_count -= 1
                
                # After removal, check if we improved best? No, count decreased.
                # But we might be able to add multiple triples next.
                
                # If we are stuck in a loop, restart
                stagnation_counter += 1
                if stagnation_counter > 50:
                    break
    
    if best_solution is None:
        best_solution, _, _, best_count = greedy_init()

    result = [list(all_triples[i]) for i in range(num_triples) if best_solution[i]]
    return result
# EVOLVE-BLOCK-END
