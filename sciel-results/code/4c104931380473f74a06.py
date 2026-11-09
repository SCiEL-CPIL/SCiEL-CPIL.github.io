import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    n = 16
    
    # Precompute all possible triples
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Generate all configs (3 disjoint pairs)
    configs = []
    config_to_idx = {}
    for v6 in itertools.combinations(range(n), 6):
        # Generate 15 matchings on 6 vertices
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
    triple_to_configs = [[] for _ in range(num_triples)]
    
    for cfg_idx, config in enumerate(configs):
        p1, p2, p3 = config
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
    
    # Precompute for each config, the list of triple indices that are transversals
    config_to_triples = [[] for _ in range(num_configs)]
    for t_idx, cfgs in enumerate(triple_to_configs):
        for c_idx in cfgs:
            config_to_triples[c_idx].append(t_idx)
    
    # Strategy: Greedy Vertex Cover (Deletion)
    # 1. Start with ALL triples.
    # 2. While there exists a "full" config (count == 8), remove a triple from it.
    # 3. This is equivalent to finding a vertex cover in the hypergraph of full configs.
    # 4. To maximize the remaining set, we want to remove as few triples as possible.
    # 5. A greedy heuristic for vertex cover: pick the vertex (triple) that appears in the most full configs.
    # 6. This minimizes the number of removals needed to break all full configs.
    
    best_system = None
    best_count = 0
    
    random.seed(42)
    
    # We will run this deletion process multiple times with different tie-breaking or randomizations.
    # Actually, since we start with all triples, the set of "full configs" is fixed initially.
    # The greedy choice might be deterministic if we break ties consistently.
    # To diversify, we can randomize the order of processing or the choice among max-degree vertices.
    
    for trial in range(200):
        if time.time() > deadline - 5:
            break
        
        # State:
        # selected: boolean array, initially all True.
        # config_counts: array of counts, initially 8 for all configs.
        
        selected = np.ones(num_triples, dtype=bool)
        config_counts = np.full(num_configs, 8, dtype=np.int32)
        
        # Queue of full configs
        # We need to process configs that become full? No, we start with all full.
        # We remove a triple, which decrements counts of its configs.
        # If a count drops from 8 to 7, it's no longer full.
        # We only care about configs that are currently full (count == 8).
        
        # Optimization: 
        # Maintain a list of "active" full configs.
        # Initially, all are active.
        # When we remove a triple T, for each config C containing T:
        #   config_counts[C] -= 1
        #   if config_counts[C] == 7: C is no longer full (remove from active set? or just mark inactive)
        
        # We need to efficiently find the triple with the highest "degree" in the active full config set.
        # Let's maintain `triple_degree` = number of active full configs containing this triple.
        
        triple_degree = np.zeros(num_triples, dtype=np.int32)
        for c_idx in range(num_configs):
            for t_idx in config_to_triples[c_idx]:
                triple_degree[t_idx] += 1
        
        # Active full configs set
        # Using a set might be slow. Let's use a list and a "visited" or "active" flag.
        # Or just iterate through all configs to find the best triple each time?
        # Num configs ~ 120k. Num triples ~ 560.
        # Iterating 120k * 560 is 67M ops. Too slow per step if we do it many times.
        
        # Better: Maintain `triple_degree` dynamically.
        # When a config becomes inactive (count 7), decrement degree of its triples.
        # When a triple is removed, decrement degree of its triples? No, the triple is gone.
        # We need to pick the unselected triple with max degree.
        
        # Let's use a max-heap or just scan?
        # Scanning 560 triples is fast.
        # Updating degrees: when a config deactivates, we update degrees of its 8 triples.
        # Total deactivations: at most num_configs.
        # Total updates: 8 * num_configs.
        # This is efficient.
        
        active_configs = [True] * num_configs
        
        removed_count = 0
        
        while True:
            if time.time() > deadline - 5:
                break
            
            # Find unselected triple with max degree
            max_deg = -1
            best_t = -1
            for t_idx in range(num_triples):
                if selected[t_idx] and triple_degree[t_idx] > max_deg:
                    max_deg = triple_degree[t_idx]
                    best_t = t_idx
            
            if max_deg == 0:
                break # No more full configs
            
            # Remove best_t
            selected[best_t] = False
            removed_count += 1
            
            # Update degrees for configs containing best_t
            for c_idx in triple_to_configs[best_t]:
                if active_configs[c_idx]:
                    # This config loses a vertex, so its count drops.
                    # It was full (8), now becomes 7.
                    # It is no longer "active" (no longer forces a removal).
                    active_configs[c_idx] = False
                    # Decrement degree of all other triples in this config
                    for t_idx in config_to_triples[c_idx]:
                        if t_idx != best_t:
                            triple_degree[t_idx] -= 1
        
        current_count = num_triples - removed_count
        
        if current_count > best_count:
            best_count = current_count
            best_system = [all_triples[i] for i in range(num_triples) if selected[i]]
        
        # To improve, we can try a different heuristic:
        # Instead of max degree, use random selection among top-k degrees?
        # Or start with a random subset of triples?
        
        # Let's try a "repair" phase on the best solution found so far.
        # Actually, the deletion heuristic is quite strong.
        # Let's try randomizing the tie-breaking in the max-degree selection.
        
        if trial % 10 == 5:
            # Randomized version
            selected = np.ones(num_triples, dtype=bool)
            config_counts = np.full(num_configs, 8, dtype=np.int32)
            triple_degree = np.zeros(num_triples, dtype=np.int32)
            for c_idx in range(num_configs):
                for t_idx in config_to_triples[c_idx]:
                    triple_degree[t_idx] += 1
            active_configs = [True] * num_configs
            
            removed_count = 0
            
            while True:
                if time.time() > deadline - 5:
                    break
                
                # Find max degree
                max_deg = -1
                candidates = []
                for t_idx in range(num_triples):
                    if selected[t_idx]:
                        if triple_degree[t_idx] > max_deg:
                            max_deg = triple_degree[t_idx]
                            candidates = [t_idx]
                        elif triple_degree[t_idx] == max_deg:
                            candidates.append(t_idx)
                
                if max_deg == 0:
                    break
                
                # Pick randomly from candidates
                best_t = random.choice(candidates)
                
                selected[best_t] = False
                removed_count += 1
                
                for c_idx in triple_to_configs[best_t]:
                    if active_configs[c_idx]:
                        active_configs[c_idx] = False
                        for t_idx in config_to_triples[c_idx]:
                            if t_idx != best_t:
                                triple_degree[t_idx] -= 1
            
            current_count = num_triples - removed_count
            if current_count > best_count:
                best_count = current_count
                best_system = [all_triples[i] for i in range(num_triples) if selected[i]]
    
    if best_system is None:
        return [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
    
    return best_system
