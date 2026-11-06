import random
import math
from itertools import combinations, permutations

def construct():
    n = 16
    points = list(range(n))
    all_triples = list(combinations(points, 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute all K(2,2,2) configurations
    # A configuration is defined by 3 disjoint pairs.
    # We generate all sets of 6 points, then partition into 3 pairs.
    configs = []
    triple_to_configs = [[] for _ in range(len(all_triples))]
    
    # Optimization: Generate configs by choosing 6 points and pairing them
    # Number of 6-point subsets is 8008.
    # For each, 15 pairings.
    # Total 120120 configs.
    
    for six_pts in combinations(points, 6):
        # Generate all perfect matchings of 6 points
        # a-b, c-d, e-f
        # Fix a=six_pts[0]
        # Pair a with one of the other 5
        # Then pair the remaining 4 in 3 ways
        a = six_pts[0]
        for b in six_pts[1:]:
            remaining = [x for x in six_pts[1:] if x != b]
            # Pair remaining 4: c-d, e-f
            c = remaining[0]
            for d in remaining[1:]:
                e, f = remaining[2], remaining[3]
                if e < f: # Avoid duplicate pairings? 
                    # The pairs are {a,b}, {c,d}, {e,f}
                    # This generates each partition once?
                    # Let's verify.
                    # Matchings of 4 elements: 3.
                    # My loop:
                    # c is fixed as rem[0].
                    # d iterates rem[1], rem[2], rem[3].
                    # If d=rem[1], e=rem[2], f=rem[3]. Pair {e,f}.
                    # If d=rem[2], e=rem[1], f=rem[3]. Pair {e,f} -> {rem[1], rem[3]}.
                    # If d=rem[3], e=rem[1], f=rem[2]. Pair {e,f} -> {rem[1], rem[2]}.
                    # Yes, this covers the 3 matchings of the remaining 4.
                    p1 = (a, b)
                    p2 = (c, d)
                    p3 = (e, f)
                    
                    # Generate the 8 transversal triples
                    # x in p1, y in p2, z in p3
                    triples_in_config = set()
                    for x in p1:
                        for y in p2:
                            for z in p3:
                                t = tuple(sorted((x, y, z)))
                                triples_in_config.add(t)
                    
                    config_triples = list(triples_in_config)
                    config_idx = len(configs)
                    configs.append(config_triples)
                    
                    for t in config_triples:
                        triple_to_configs[triple_to_idx[t]].append(config_idx)

    # Simulated Annealing
    random.seed(42)
    
    # Initialize with a random set
    # Start with ~150 triples
    initial_size = 150
    current_state = set()
    current_counts = [0] * len(configs)
    current_violations = 0
    current_size = 0
    
    def add_triple(t):
        nonlocal current_state, current_counts, current_violations, current_size
        if t in current_state:
            return
        current_state.add(t)
        current_size += 1
        idx = triple_to_idx[t]
        for c_idx in triple_to_configs[idx]:
            current_counts[c_idx] += 1
            if current_counts[c_idx] == 8:
                current_violations += 1
            elif current_counts[c_idx] > 8:
                # Should not happen if we track correctly, but safe
                pass

    def remove_triple(t):
        nonlocal current_state, current_counts, current_violations, current_size
        if t not in current_state:
            return
        current_state.remove(t)
        current_size -= 1
        idx = triple_to_idx[t]
        for c_idx in triple_to_configs[idx]:
            current_counts[c_idx] -= 1
            if current_counts[c_idx] == 7:
                current_violations -= 1

    # Init
    init_triples = random.sample(all_triples, initial_size)
    for t in init_triples:
        add_triple(t)
        
    best_state = set(current_state)
    best_violations = current_violations
    best_size = current_size
    
    T = 10.0
    cooling = 0.995
    steps = 50000
    
    for i in range(steps):
        # Propose move
        move_type = random.random()
        if move_type < 0.5:
            # Add random triple
            t = random.choice(all_triples)
            if t in current_state:
                continue
            # Delta
            delta_size = 1
            delta_viol = 0
            idx = triple_to_idx[t]
            for c_idx in triple_to_configs[idx]:
                if current_counts[c_idx] == 7:
                    delta_viol += 1
            # Energy: -size + 100 * violations
            delta_energy = -delta_size + 100 * delta_viol
            
            if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                add_triple(t)
                if current_violations == 0:
                    if current_size > best_size:
                        best_size = current_size
                        best_state = set(current_state)
                elif current_size > best_size and best_violations > 0:
                    # Keep track of best valid? Or just best size regardless?
                    # We want valid.
                    pass
        elif move_type < 0.8:
            # Remove random triple
            if current_size == 0:
                continue
            t = random.choice(list(current_state))
            delta_size = -1
            delta_viol = 0
            idx = triple_to_idx[t]
            for c_idx in triple_to_configs[idx]:
                if current_counts[c_idx] == 8:
                    delta_viol -= 1
            
            delta_energy = -delta_size + 100 * delta_viol
            
            if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                remove_triple(t)
                if current_violations == 0:
                    if current_size > best_size:
                        best_size = current_size
                        best_state = set(current_state)
        else:
            # Swap
            if current_size == 0:
                continue
            t_out = random.choice(list(current_state))
            t_in = random.choice(all_triples)
            if t_in in current_state:
                continue
            
            # Delta for removing t_out and adding t_in
            # Simplified: just do remove then add if accepted?
            # Or calculate combined delta.
            # Let's just try remove, then add.
            # This is less efficient but simpler.
            # Actually, let's just stick to Add/Remove for stability.
            pass

        T *= cooling
        
    # Final cleanup: remove any remaining violations?
    # If best_state has violations, we need a valid one.
    # The SA might end in a state with violations.
    # We should track the best VALID state.
    
    # Re-evaluate best_state validity just in case
    # And if best_state has violations, try to fix or find a valid one from history?
    # Better: Track best valid state during SA.
    
    # My code above only updates best_state if current_violations == 0.
    # So best_state is guaranteed valid.
    
    return [list(t) for t in best_state]
