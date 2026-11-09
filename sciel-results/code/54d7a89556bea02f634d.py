import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    n = 16
    
    # Precompute all possible triples
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute the conflict graph
    # Two triples conflict if they are part of a common forbidden configuration K(2,2,2)
    # A forbidden configuration consists of 3 disjoint pairs (x1,y1), (x2,y2), (x3,y3)
    # and the 8 transversals.
    # If we add a triple T, it conflicts with any other triple T' that is a transversal
    # of the same 3 pairs.
    # However, the constraint is "no three disjoint pairs have all eight transversal triples present".
    # This means we cannot have a set of 8 triples that form a K(2,2,2).
    # Greedy approach: If we add a triple, we must ensure we don't complete any K(2,2,2).
    # This is not simply a pairwise conflict. It's a 8-way hyperedge constraint.
    
    # Let's re-evaluate. The constraint is: For any 3 disjoint pairs, not all 8 transversals are in the system.
    # This is a hypergraph constraint.
    # A simpler sufficient condition for validity (which might be suboptimal but fast):
    # If we ensure that for any 3 disjoint pairs, at most 7 transversals are present, we are valid.
    # Actually, the problem asks to avoid the configuration where ALL 8 are present.
    # So we can have 7.
    
    # Efficient Check:
    # When adding a new triple T, we need to check if there exists a set of 3 disjoint pairs
    # such that T is one of the 8 transversals, and the other 7 are already in the system.
    
    # Let's precompute for each triple T, the list of "potential forbidden configurations" it belongs to.
    # A configuration is defined by the 3 pairs.
    # For a given triple T=(a,b,c), the 3 pairs must be formed by pairing a, b, c with 3 other distinct elements.
    # Let the other elements be d, e, f.
    # The pairs are (a,d), (b,e), (c,f) in some permutation.
    # There are 3! = 6 ways to assign d,e,f to a,b,c.
    # And C(13,3) = 286 ways to choose d,e,f from the remaining 13 vertices.
    # Total potential configs per triple: 6 * 286 = 1716.
    
    # For each such config, we have the 8 transversals.
    # If we add T, we check if the other 7 are in the system.
    
    # To speed this up, we can maintain a count of how many transversals of each config are in the system.
    # Let's index configs.
    
    # A config is uniquely identified by the set of 6 vertices involved and the pairing structure?
    # No, the 3 disjoint pairs on 6 vertices define the config.
    # Number of ways to partition 6 vertices into 3 pairs: (6-1)!! = 5*3*1 = 15.
    # Total number of 6-vertex subsets: C(16,6) = 8008.
    # Total configs: 8008 * 15 = 120120.
    
    # Let's precompute all configs.
    # A config is a tuple of 3 pairs: ((x1,y1), (x2,y2), (x3,y3)).
    # To avoid duplicates, we enforce x1 < y1, x2 < y2, x3 < y3, and x1 < x2 < x3.
    
    configs = []
    config_to_idx = {}
    for v6 in itertools.combinations(range(n), 6):
        # Generate all perfect matchings on these 6 vertices
        # Vertices: v0, v1, v2, v3, v4, v5
        # Pairings:
        # (v0,v1), (v2,v3), (v4,v5)
        # (v0,v1), (v2,v4), (v3,v5)
        # ...
        # There are 15 matchings.
        # We can generate them by pairing v0 with v1..v5, then pairing the rest.
        for p1 in range(1, 6):
            if v6[p1] < v6[0]: continue # Enforce order x1 < x2 < x3? 
            # Actually, let's just generate all 15 and normalize.
            # Easier: iterate through all permutations of the 6 vertices, take first 2 as pair1, next 2 as pair2, last 2 as pair3.
            # This generates duplicates.
            
            # Let's use a standard way to generate matchings.
            # Fix v0. It pairs with v1, v2, v3, v4, or v5.
            # Case 1: v0-v1. Remaining v2,v3,v4,v5. Matchings on 4 vertices: 3.
            # Case 2: v0-v2. Remaining v1,v3,v4,v5. Matchings on 4 vertices: 3.
            # ...
            
            pass # This logic is getting complex. Let's just iterate all subsets of 3 pairs.
            
    # Alternative: Iterate all triples of pairs.
    # Total pairs: C(16,2) = 120.
    # Triples of disjoint pairs:
    # Choose 6 vertices: C(16,6).
    # Partition into 3 pairs: 15 ways.
    # Total configs = 8008 * 15 = 120120.
    
    # Let's generate them.
    all_configs = []
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
    
    num_configs = len(configs)
    
    # Map each triple to the configs it belongs to.
    # A triple belongs to a config if it is one of the 8 transversals.
    triple_to_configs = [[] for _ in range(num_triples)]
    
    for cfg_idx, config in enumerate(configs):
        p1, p2, p3 = config
        # 8 transversals
        # p1=(a,b), p2=(c,d), p3=(e,f)
        a, b = p1
        c, d = p2
        e, f = p3
        
        transversals = [
            (a, c, e), (a, c, f), (a, d, e), (a, d, f),
            (b, c, e), (b, c, f), (b, d, e), (b, d, f)
        ]
        
        for t in transversals:
            t_sorted = tuple(sorted(t))
            if t_sorted in triple_to_idx:
                triple_to_configs[triple_to_idx[t_sorted]].append(cfg_idx)
    
    # Now, greedy with restarts.
    # State: set of selected triples.
    # Count of transversals present for each config.
    
    best_system = None
    best_count = 0
    
    random.seed(42)
    
    # To make it faster, we can use a "blocked" list.
    # If a config has 7 transversals present, the 8th is blocked.
    # When we add a triple, we increment counts for its configs.
    # If any count reaches 8, we remove the triple and mark it as blocked?
    # No, we just can't add it if it would make a count 8.
    
    # Optimization: 
    # Maintain `config_counts` array.
    # Maintain `selected` boolean array.
    
    for trial in range(1000):
        if time.time() > deadline - 5:
            break
            
        selected = [False] * num_triples
        config_counts = [0] * num_configs
        selected_indices = []
        
        # Random order of triples
        order = list(range(num_triples))
        random.shuffle(order)
        
        for t_idx in order:
            if time.time() > deadline - 5:
                break
                
            # Check if adding this triple is valid
            # It is invalid if it belongs to any config where count is already 7.
            valid = True
            for cfg_idx in triple_to_configs[t_idx]:
                if config_counts[cfg_idx] == 7:
                    valid = False
                    break
            
            if valid:
                selected[t_idx] = True
                selected_indices.append(t_idx)
                for cfg_idx in triple_to_configs[t_idx]:
                    config_counts[cfg_idx] += 1
        
        if len(selected_indices) > best_count:
            best_count = len(selected_indices)
            best_system = [all_triples[i] for i in selected_indices]
            
        # Local Search / Hill Climbing
        # Try to swap out a triple for two others? Or just add more?
        # Since we are greedy, we might be stuck.
        # Let's try a simple improvement: 
        # Iterate through all unselected triples, see if they can be added.
        # This is just the greedy again, but maybe in a different order.
        
        # A more powerful move: 
        # If we remove a triple, we might be able to add multiple others.
        # This is complex. Let's stick to many greedy restarts with different shuffles.
        
        # To improve quality, we can bias the shuffle.
        # Triples with fewer conflicts (smaller len(triple_to_configs)) are "safer".
        # But we want to maximize count, so "safer" might not be better.
        
        # Let's try a different heuristic:
        # Start with a random subset, then greedily add.
        
        if trial % 10 == 0:
            # Occasionally do a "dense" start
            # Pick random 100 triples, then greedy fill
            initial_selected = set(random.sample(range(num_triples), min(100, num_triples)))
            selected = [False] * num_triples
            config_counts = [0] * num_configs
            for t_idx in initial_selected:
                valid = True
                for cfg_idx in triple_to_configs[t_idx]:
                    if config_counts[cfg_idx] == 7:
                        valid = False
                        break
                if valid:
                    selected[t_idx] = True
                    for cfg_idx in triple_to_configs[t_idx]:
                        config_counts[cfg_idx] += 1
            
            # Now greedy add
            order = list(range(num_triples))
            random.shuffle(order)
            for t_idx in order:
                if selected[t_idx]:
                    continue
                if time.time() > deadline - 5:
                    break
                valid = True
                for cfg_idx in triple_to_configs[t_idx]:
                    if config_counts[cfg_idx] == 7:
                        valid = False
                        break
                if valid:
                    selected[t_idx] = True
                    for cfg_idx in triple_to_configs[t_idx]:
                        config_counts[cfg_idx] += 1
            
            count = sum(selected)
            if count > best_count:
                best_count = count
                best_system = [all_triples[i] for i in range(num_triples) if selected[i]]

    if best_system is None:
        return [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
    
    return best_system
