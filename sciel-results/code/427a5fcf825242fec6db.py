import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)
    
    # Map triple tuple to index
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute all K(2,2,2) configurations
    # A configuration is defined by 3 disjoint pairs (6 points).
    # There are C(16,6) * 15 such configurations.
    # C(16,6) = 8008. 8008 * 15 = 120120.
    
    configs = []
    config_id_map = {}
    triples_in_config = []
    
    points = range(N)
    # Predefine the 15 pair partitions for 6 items
    # Items: 0,1,2,3,4,5
    partitions = [
        [(0, 1), (2, 3), (4, 5)],
        [(0, 1), (2, 4), (3, 5)],
        [(0, 1), (2, 5), (3, 4)],
        [(0, 2), (1, 3), (4, 5)],
        [(0, 2), (1, 4), (3, 5)],
        [(0, 2), (1, 5), (3, 4)],
        [(0, 3), (1, 2), (4, 5)],
        [(0, 3), (1, 4), (2, 5)],
        [(0, 3), (1, 5), (2, 4)],
        [(0, 4), (1, 2), (3, 5)],
        [(0, 4), (1, 3), (2, 5)],
        [(0, 4), (1, 5), (2, 3)],
        [(0, 5), (1, 2), (3, 4)],
        [(0, 5), (1, 3), (2, 4)],
        [(0, 5), (1, 4), (2, 3)]
    ]

    for pts in itertools.combinations(points, 6):
        p0, p1, p2, p3, p4, p5 = pts
        for part in partitions:
            # Map indices 0-5 to actual points
            pair_set = [
                (pts[part[0][0]], pts[part[0][1]]),
                (pts[part[1][0]], pts[part[1][1]]),
                (pts[part[2][0]], pts[part[2][1]])
            ]
            # Normalize for unique ID
            norm_pairs = tuple(sorted(tuple(sorted(p)) for p in pair_set))
            if norm_pairs not in config_id_map:
                cid = len(configs)
                config_id_map[norm_pairs] = cid
                configs.append(norm_pairs)
                
                (a,b), (c,d), (e,f) = norm_pairs
                tris = []
                for x in (a,b):
                    for y in (c,d):
                        for z in (e,f):
                            t = tuple(sorted((x,y,z)))
                            tris.append(triple_to_idx[t])
                triples_in_config.append(tris)
                
    num_configs = len(configs)
    
    # Reverse map: triple_index -> list of config_ids
    triple_to_configs = [[] for _ in range(num_triples)]
    for cid, tris in enumerate(triples_in_config):
        for t_idx in tris:
            triple_to_configs[t_idx].append(cid)
            
    # State
    present = [False] * num_triples
    config_counts = [0] * num_configs
    
    def add_triple(t_idx):
        present[t_idx] = True
        for cid in triple_to_configs[t_idx]:
            config_counts[cid] += 1
            if config_counts[cid] == 8:
                return False
        return True
        
    def remove_triple(t_idx):
        present[t_idx] = False
        for cid in triple_to_configs[t_idx]:
            config_counts[cid] -= 1
        return True

    def is_valid_addition(t_idx):
        if present[t_idx]:
            return False
        for cid in triple_to_configs[t_idx]:
            if config_counts[cid] == 7:
                return False
        return True

    def reset_state():
        for i in range(num_triples):
            present[i] = False
        for i in range(num_configs):
            config_counts[i] = 0

    def load_state(indices):
        reset_state()
        for idx in indices:
            add_triple(idx)

    def greedy_build(order):
        reset_state()
        current_set = []
        for t_idx in order:
            if is_valid_addition(t_idx):
                add_triple(t_idx)
                current_set.append(t_idx)
        return current_set

    def local_search(current_set, max_iterations=2000):
        # Load state
        load_state(current_set)
        count = len(current_set)
        current_set = set(current_set) # Use set for O(1) removal? 
        # Actually, list is better for iteration, but set for membership.
        # Let's stick to list and a set for fast removal if needed, 
        # but removal from list is O(N). N=560, so it's fine.
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iterations:
            if time.time() > deadline - 2:
                break
            iter_count += 1
            improved = False
            
            # 1. Try to add any valid triple
            missing = [i for i in range(num_triples) if not present[i]]
            random.shuffle(missing)
            
            added_any = False
            for t_idx in missing:
                if is_valid_addition(t_idx):
                    add_triple(t_idx)
                    current_set.append(t_idx)
                    count += 1
                    added_any = True
                    improved = True
                    # Don't break, keep adding if possible
            if added_any:
                continue
                
            # 2. Try swaps to unblock
            if not missing:
                break
                
            # Heuristic: Find a missing triple that is "close" to being valid
            # Or just pick a random missing triple and try to fix its blockers
            
            t_idx = random.choice(missing)
            conflicts = [cid for cid in triple_to_configs[t_idx] if config_counts[cid] == 7]
            
            if conflicts:
                # Try to resolve a random conflict
                cid = random.choice(conflicts)
                present_in_cid = [t for t in triples_in_config[cid] if present[t]]
                
                if present_in_cid:
                    # Try removing a random one and adding t_idx
                    # Shuffle to vary which one we remove
                    random.shuffle(present_in_cid)
                    success = False
                    for u_idx in present_in_cid:
                        # Remove u_idx
                        remove_triple(u_idx)
                        current_set.remove(u_idx)
                        count -= 1
                        
                        # Try to add t_idx
                        if is_valid_addition(t_idx):
                            add_triple(t_idx)
                            current_set.append(t_idx)
                            count += 1
                            improved = True
                            success = True
                            break
                        else:
                            # Revert
                            add_triple(u_idx)
                            current_set.append(u_idx)
                            count += 1
                            
                    if success:
                        continue # Restart search loop as state changed significantly
            
        return count, current_set

    best_set = []
    best_count = 0
    
    # Initial greedy runs
    num_restarts = 30
    for seed in range(num_restarts):
        if time.time() > deadline - 5: break
        random.seed(seed)
        order = list(range(num_triples))
        random.shuffle(order)
        s = greedy_build(order)
        if len(s) > best_count:
            best_count = len(s)
            best_set = s[:]
            
    # Local search on best
    if time.time() < deadline - 10 and best_set:
        for seed in range(10):
            if time.time() > deadline - 5: break
            random.seed(seed + 100)
            s_copy = best_set[:]
            count, s = local_search(s_copy)
            if count > best_count:
                best_count = count
                best_set = s[:]

    # More greedy runs with different seeds if time permits
    if time.time() < deadline - 10:
        for seed in range(100, 200):
            if time.time() > deadline - 5: break
            random.seed(seed)
            order = list(range(num_triples))
            random.shuffle(order)
            s = greedy_build(order)
            if len(s) > best_count:
                best_count = len(s)
                best_set = s[:]
            # Quick local search on good candidates
            if len(s) >= best_count - 5:
                s_copy = s[:]
                count, s2 = local_search(s_copy, max_iterations=200)
                if count > best_count:
                    best_count = count
                    best_set = s2[:]

    result = [all_triples[i] for i in best_set]
    return result
