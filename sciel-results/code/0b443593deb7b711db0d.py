import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)  # 560

    # Precompute index for O(1) lookup
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Precompute all K(2,2,2) configurations
    # A K(2,2,2) is defined by 3 disjoint pairs. The 8 transversal triples are the products.
    # If all 8 are present, it's a violation.
    
    configs = []
    config_id_map = {}
    triples_in_config = [] # List of lists of 8 triple indices

    points = range(N)
    for pts in itertools.combinations(points, 6):
        # Generate all partitions of these 6 points into 3 pairs
        p0, p1, p2, p3, p4, p5 = pts
        pairs = [
            [(p0, p1), (p2, p3), (p4, p5)],
            [(p0, p1), (p2, p4), (p3, p5)],
            [(p0, p1), (p2, p5), (p3, p4)],
            [(p0, p2), (p1, p3), (p4, p5)],
            [(p0, p2), (p1, p4), (p3, p5)],
            [(p0, p2), (p1, p5), (p3, p4)],
            [(p0, p3), (p1, p2), (p4, p5)],
            [(p0, p3), (p1, p4), (p2, p5)],
            [(p0, p3), (p1, p5), (p2, p4)],
            [(p0, p4), (p1, p2), (p3, p5)],
            [(p0, p4), (p1, p3), (p2, p5)],
            [(p0, p4), (p1, p5), (p2, p3)],
            [(p0, p5), (p1, p2), (p3, p4)],
            [(p0, p5), (p1, p3), (p2, p4)],
            [(p0, p5), (p1, p4), (p2, p3)]
        ]
        for pair_set in pairs:
            # Normalize pair order and pair internal order for unique ID
            norm_pairs = tuple(sorted(tuple(sorted(p)) for p in pair_set))
            if norm_pairs not in config_id_map:
                cid = len(configs)
                config_id_map[norm_pairs] = cid
                configs.append(norm_pairs)
                
                # Find the 8 transversal triples
                (a,b), (c,d), (e,f) = norm_pairs
                tris = []
                for x in (a,b):
                    for y in (c,d):
                        for z in (e,f):
                            t = tuple(sorted((x,y,z)))
                            tris.append(triple_to_idx[t])
                triples_in_config.append(tris)
                
    num_configs = len(configs)
    
    # For each triple, which configs does it belong to?
    triple_to_configs = [[] for _ in range(num_triples)]
    for cid, tris in enumerate(triples_in_config):
        for t_idx in tris:
            triple_to_configs[t_idx].append(cid)
            
    # State
    present = np.zeros(num_triples, dtype=bool)
    config_counts = np.zeros(num_configs, dtype=np.int8)
    
    # Helper to add/remove
    def add_triple(t_idx):
        present[t_idx] = True
        for cid in triple_to_configs[t_idx]:
            config_counts[cid] += 1
            if config_counts[cid] == 8:
                return False # Violation
        return True
        
    def remove_triple(t_idx):
        present[t_idx] = False
        for cid in triple_to_configs[t_idx]:
            config_counts[cid] -= 1
        return True

    def is_valid_addition(t_idx):
        if present[t_idx]:
            return False
        # Check if any config it belongs to already has 7 others
        for cid in triple_to_configs[t_idx]:
            if config_counts[cid] == 7:
                return False
        return True

    def run_local_search(seed_idx):
        random.seed(seed_idx)
        present[:] = False
        config_counts[:] = 0
        current_set = set()
        
        # Greedy Phase: Random order
        order = list(range(num_triples))
        random.shuffle(order)
        
        for t_idx in order:
            if is_valid_addition(t_idx):
                add_triple(t_idx)
                current_set.add(t_idx)
                
        count = len(current_set)
        
        # Local Search Phase
        improved = True
        no_improve_count = 0
        max_no_improve = 30
        
        while improved and time.time() < deadline - 10:
            improved = False
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Perform a "kick": remove a random small number of triples to shake up the state
                # This might allow a different structure to emerge
                if count > 10:
                    kick_size = random.randint(2, 5)
                    to_remove = random.sample(list(current_set), kick_size)
                    for t in to_remove:
                        remove_triple(t)
                        current_set.remove(t)
                        count -= 1
                    improved = True
                    no_improve_count = 0
                    continue

            # Try to add any missing triple
            missing = [i for i in range(num_triples) if not present[i]]
            if not missing:
                break
            random.shuffle(missing)
            
            added_any = False
            for t_idx in missing:
                if is_valid_addition(t_idx):
                    add_triple(t_idx)
                    current_set.add(t_idx)
                    count += 1
                    added_any = True
                    improved = True
                    no_improve_count = 0
                    if time.time() > deadline - 10:
                        break
            
            if not added_any and time.time() < deadline - 10:
                # Try to unblock by swapping
                # Pick a random missing triple
                t_idx = random.choice(missing)
                # Find a conflict
                conflicts = []
                for cid in triple_to_configs[t_idx]:
                    if config_counts[cid] == 7:
                        conflicts.append(cid)
                    
                if conflicts:
                    cid = random.choice(conflicts)
                    # Get the 7 present triples in this config
                    present_in_cid = []
                    for t in triples_in_config[cid]:
                        if present[t]:
                            present_in_cid.append(t)
                        
                    if present_in_cid:
                        u_idx = random.choice(present_in_cid)
                        # Remove U
                        remove_triple(u_idx)
                        current_set.remove(u_idx)
                        count -= 1
                        
                        # Try to add T
                        if is_valid_addition(t_idx):
                            add_triple(t_idx)
                            current_set.add(t_idx)
                            count += 1
                            improved = True
                            no_improve_count = 0
                        else:
                            # Failed to add T, put U back
                            add_triple(u_idx)
                            current_set.add(u_idx)
                            count += 1
                            
        return count, list(current_set)

    best_count = 0
    best_set = []
    
    # Run multiple restarts
    num_restarts = 0
    while time.time() < deadline - 10:
        seed = num_restarts
        count, s = run_local_search(seed)
        if count > best_count:
            best_count = count
            best_set = s
        num_restarts += 1
        
    # Convert indices back to triples
    result = [all_triples[i] for i in best_set]
    
    return result
