import time
import random
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n = 16
    # All possible triples
    all_triples = list(combinations(range(n), 3))
    triple_index = {t: i for i, t in enumerate(all_triples)}
    m = len(all_triples)  # 560
    
    # Precompute for each triple, which forbidden configurations it belongs to
    # A forbidden configuration is defined by 3 disjoint pairs
    # For a given triple (a,b,c), it's part of a forbidden config if there exist
    # a' != a, b' != b, c' != c such that {a,a'},{b,b'},{c,c'} are disjoint pairs
    # and all 8 transversals are present.
    
    # Alternative: for each triple, precompute the set of "complementary" vertices
    # that could form a forbidden config with it.
    
    # Let's use a different approach: maintain a set of active triples.
    # When considering adding triple T, check if it completes any forbidden config.
    
    # A forbidden config is 6 vertices {a1,b1,a2,b2,a3,b3} partitioned into 3 pairs,
    # with all 8 transversal triples present.
    
    # For efficiency, let's precompute all forbidden configurations.
    # Number of ways to choose 6 vertices from 16: C(16,6) = 8008
    # For each 6-set, number of ways to partition into 3 pairs: 15
    # Total forbidden configs: 8008 * 15 = 120120
    
    # For each forbidden config, store the 8 triples it contains.
    
    # This might be too many. Let's use a smarter approach.
    
    # Greedy approach with backtracking-like local search:
    
    # State: set of active triples
    # Score: number of active triples
    # Constraint: no forbidden config has all 8 triples active
    
    # Let's use a greedy construction first, then local search.
    
    # For greedy: randomly order triples, add if doesn't violate constraint.
    
    # To check if adding triple T violates: for each pair of other active triples
    # that together with T could form part of a forbidden config... this is complex.
    
    # Better: precompute for each triple T, the list of "witness" configurations
    # where T is one of the 8 transversals.
    
    # Let's precompute: for each triple T=(a,b,c), find all 6-sets containing {a,b,c}
    # where a,b,c can be assigned to 3 different pairs.
    
    # Actually, let me just use a direct approach with bitsets for speed.
    
    # Represent the set of active triples as a bitset (560 bits).
    # For each forbidden config, store a bitmask of its 8 triples.
    # A config is violated if (active & config_mask) == config_mask.
    
    # But checking all 120120 configs each time is expensive.
    
    # Alternative: for each triple, precompute which configs it belongs to.
    # When adding triple T, only check configs that contain T.
    
    # Let's precompute: config_list_for_triple[t] = list of config indices containing triple t
    
    # First, enumerate all forbidden configs:
    configs = []  # list of (6-tuple of vertices, list of 8 triple indices)
    triple_to_configs = [[] for _ in range(m)]
    
    # For each 6-subset, for each pairing of the 6 vertices into 3 pairs:
    for six in combinations(range(n), 6):
        # All perfect matchings of 6 vertices: 15 ways
        # Pairings: (0,1)(2,3)(4,5), (0,1)(2,4)(3,5), etc.
        for p1 in range(5):
            for p2 in range(4):
                for p3 in range(2):
                    # Build pairs from indices 0-5
                    pairs = [(six[0], six[1]), (six[2], six[3]), (six[4], six[5])]
                    # This only gives 1 pairing. Need all 15.
                    # Let me enumerate differently.
                    pass
    
    # Let me enumerate pairings properly:
    # For 6 vertices v0..v5, a pairing is a partition into 3 pairs.
    # Fix v0's partner: 5 choices. Then fix v1's partner (if not v0's): 3 choices. Then remaining 2.
    # Total: 5*3*1 = 15.
    
    def get_pairings(six):
        """Yield all 15 pairings of 6 vertices as tuples of 3 pairs."""
        v = list(six)
        for i1 in range(1, 6):
            pair1 = (v[0], v[i1])
            remaining = [v[j] for j in range(6) if j != 0 and j != i1]
            for i2 in range(1, 4):
                pair2 = (remaining[0], remaining[i2])
                pair3 = (remaining[1], remaining[3]) if i2 != 1 else (remaining[1], remaining[3])
                # Wait, this is getting messy. Let me just brute force.
                rem2 = [x for x in remaining if x != remaining[0] and x != remaining[i2]]
                pair3 = (rem2[0], rem2[1])
                yield (pair1, pair2, pair3)
    
    # Actually, let me just generate all 15 pairings explicitly:
    def all_pairings(six):
        v = list(six)
        # Pairing is defined by which element pairs with v[0], then which of remaining pairs with v[1] (if not already paired)
        results = []
        for a in range(1, 6):
            p1 = (v[0], v[a])
            rem = [v[i] for i in range(6) if i != 0 and i != a]
            for b in range(1, 4):
                p2 = (rem[0], rem[b])
                rem2 = [x for x in rem if x != rem[0] and x != rem[b]]
                p3 = (rem2[0], rem2[1])
                results.append((p1, p2, p3))
        return results
    
    # Now enumerate all forbidden configs:
    # For each 6-set and each pairing, compute the 8 transversal triples.
    for six in combinations(range(n), 6):
        for pairing in all_pairings(six):
            # For 3 pairs (a1,b1), (a2,b2), (a3,b3), the 8 transversals are:
            # (a1 or b1, a2 or b2, a3 or b3)
            p1, p2, p3 = pairing
            transversals = []
            for x in p1:
                for y in p2:
                    for z in p3:
                        t = tuple(sorted((x, y, z)))
                        transversals.append(t)
            # Convert to indices
            t_indices = []
            for t in transversals:
                if t in triple_index:
                    t_indices.append(triple_index[t])
                else:
                    # Should not happen
                    break
            if len(t_indices) == 8:
                configs.append(t_indices)
                for idx in t_indices:
                    triple_to_configs[idx].append(len(configs) - 1)
    
    # Now do greedy construction:
    random.seed(42)
    
    def greedy_construction():
        order = list(range(m))
        random.shuffle(order)
        active = [False] * m
        active_count = 0
        # config_violation_count[c] = number of triples in config c that are active
        config_counts = [0] * len(configs)
        
        for t_idx in order:
            # Check if adding t_idx violates any config
            can_add = True
            affected_configs = triple_to_configs[t_idx]
            for c_idx in affected_configs:
                if config_counts[c_idx] == 7:  # 7 already active, adding makes 8
                    can_add = False
                    break
            if can_add:
                active[t_idx] = True
                active_count += 1
                for c_idx in affected_configs:
                    config_counts[c_idx] += 1
        
        return active, active_count
    
    def local_search(active, active_count, max_time=100):
        config_counts = [0] * len(configs)
        for t_idx in range(m):
            if active[t_idx]:
                for c_idx in triple_to_configs[t_idx]:
                    config_counts[c_idx] += 1
        
        best_active = active[:]
        best_count = active_count
        best_config_counts = config_counts[:]
        
        while time.time() < max_time:
            improved = False
            # Try adding inactive triples
            inactive = [i for i in range(m) if not active[i]]
            random.shuffle(inactive)
            
            for t_idx in inactive:
                affected = triple_to_configs[t_idx]
                can_add = True
                for c_idx in affected:
                    if config_counts[c_idx] == 7:
                        can_add = False
                        break
                if can_add:
                    active[t_idx] = True
                    active_count += 1
                    for c_idx in affected:
                        config_counts[c_idx] += 1
                    improved = True
            
            if not improved:
                # Try swapping: remove one active, add up to two inactive
                # Or just try removing one that's "blocking" many additions
                # Simpler: random restart within local search
                # Try to find a triple to remove that allows adding more
                active_list = [i for i in range(m) if active[i]]
                if not active_list:
                    break
                random.shuffle(active_list)
                
                for rem_idx in active_list:
                    # Try removing rem_idx and see if we can add more
                    old_count = active_count
                    active[rem_idx] = False
                    active_count -= 1
                    for c_idx in triple_to_configs[rem_idx]:
                        config_counts[c_idx] -= 1
                    
                    # Now try to add as many as possible
                    inactive2 = [i for i in range(m) if not active[i]]
                    random.shuffle(inactive2)
                    for t_idx in inactive2:
                        affected = triple_to_configs[t_idx]
                        can_add = True
                        for c_idx in affected:
                            if config_counts[c_idx] == 7:
                                can_add = False
                                break
                        if can_add:
                            active[t_idx] = True
                            active_count += 1
                            for c_idx in affected:
                                config_counts[c_idx] += 1
                    
                    if active_count > best_count:
                        best_active = active[:]
                        best_count = active_count
                        best_config_counts = config_counts[:]
                        improved = True
                    else:
                        # Revert
                        active[rem_idx] = True
                        active_count = old_count
                        for c_idx in triple_to_configs[rem_idx]:
                            config_counts[c_idx] += 1
                        # Revert additions
                        for t_idx in inactive2:
                            if active[t_idx] and t_idx != rem_idx:
                                # This was added, need to check if it was in original
                                pass
                        # This revert is complex. Let me use a different approach.
                        break
                    
                    if improved:
                        break
            
            if not improved:
                # Try a random perturbation
                if active_count > 10:
                    # Remove a random triple
                    active_list = [i for i in range(m) if active[i]]
                    rem_idx = random.choice(active_list)
                    active[rem_idx] = False
                    active_count -= 1
                    for c_idx in triple_to_configs[rem_idx]:
                        config_counts[c_idx] -= 1
        
        return best_active, best_count
    
    # Run multiple greedy constructions and pick the best
    best_result = []
    best_m = 0
    
    while time.time() < deadline:
        active, count = greedy_construction()
        if count > best_m:
            best_m = count
            best_result = active[:]
        
        # Local search
        active, count = local_search(active, count, max_time=deadline - 5)
        if count > best_m:
            best_m = count
            best_result = active[:]
        
        if time.time() > deadline - 10:
            break
    
    # Convert to list of triples
    result = [list(all_triples[i]) for i in range(m) if best_result[i]]
    return result
