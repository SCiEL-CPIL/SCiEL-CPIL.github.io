import time
import random
import itertools
import numpy as np

def construct():
    # Deadline guard: 120s limit, aim for 110s max execution
    start_time = time.time()
    deadline = start_time + 110
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)
    
    # Precompute index for O(1) lookup
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute for each triple, which K(2,2,2) configurations it is part of.
    # A K(2,2,2) is defined by 3 disjoint pairs. There are 8 transversal triples.
    # If all 8 are present, it's a violation.
    # Instead of checking all K(2,2,2)s, we can check for each new triple T,
    # does it complete any K(2,2,2) that already has 7 other triples present?
    
    # Let's precompute for each triple T, the list of other 7 triples that form a K(2,2,2) with it?
    # No, that's too many.
    # Better: For each K(2,2,2) configuration (defined by 3 disjoint pairs), 
    # store the 8 triples.
    # Number of ways to pick 3 disjoint pairs from 16 points:
    # 16 * 15 * 14 * 13 * 12 * 11 / (2^3 * 3!) = 201870 * 1320 / 48 ?
    # Let's just generate them.
    
    # However, 201870 * 8 is a lot of storage but manageable.
    # Actually, we can be smarter.
    # Violation check: Add triple T. 
    # For each pair of disjoint pairs (p1, p2) such that T is transversal to them,
    # the third pair p3 is determined by the remaining points? No.
    
    # Let's stick to the efficient method used in similar problems:
    # Maintain a count for each K(2,2,2) configuration of how many of its 8 triples are present.
    # If count reaches 8, it's a violation.
    # If count is 7, adding the 8th is forbidden.
    
    # Number of K(2,2,2) configurations on 16 points:
    # Choose 6 points: C(16,6) = 8008
    # Partition 6 points into 3 pairs: 15 ways (6*4*2 / 6)
    # Total configs = 8008 * 15 = 120120
    # This is very small. We can maintain an array of size 120120.
    
    # Precompute all configurations
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
                
            else:
                # Should not happen if logic is correct, but just in case
                pass
                
    num_configs = len(configs)
    
    # For each triple, which configs does it belong to?
    # Each triple is part of how many K(2,2,2)s?
    # A triple (a,b,c). We need to pick 3 other points? No.
    # The K(2,2,2) uses 6 points. The triple uses 3.
    # The other 3 points must form the other 3 elements of the transversals?
    # Actually, for a fixed triple T=(a,b,c), and a fixed config C, T is in C if T is one of the 8 transversals.
    # Let's build the reverse map: triple_index -> list of config_ids
    
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

    # Initialize with a random valid set?
    # Or start empty and greedily add?
    # Greedy from empty is slow. 
    # Let's start with a random subset and repair/improve.
    # Or use the star system as a seed? The star system has 105 triples.
    # Let's try to start with a random permutation of all triples and greedily add.
    
    best_count = 0
    best_set = []
    
    # Strategy: Multiple restarts with random orderings, greedy add.
    # Then local search: try to swap out a triple for two? Or just add if possible.
    # Since we maximize count, we want to add as many as possible.
    # Local search: Randomly remove a triple, then try to add other triples.
    # Or: Randomly try to add a triple. If conflict, remove one of the conflicting triples and add the new one (if it improves? No, count stays same).
    # Actually, if we remove one and add one, count is same. We need to find a sequence that increases count.
    # Or simply: Greedy add is good. Local search: try to add. If blocked, try removing a random triple from the blocking config, then add.
    
    def local_search(current_set_indices):
        # current_set_indices is a list of indices
        # Reset state
        present[:] = False
        config_counts[:] = 0
        for idx in current_set_indices:
            add_triple(idx)
            
        count = len(current_set_indices)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try to add any missing triple
            missing = [i for i in range(num_triples) if not present[i]]
            random.shuffle(missing)
            
            for t_idx in missing:
                if is_valid_addition(t_idx):
                    add_triple(t_idx)
                    count += 1
                    improved = True
                    # Don't break, try to add more
                    continue
                    
            if not improved:
                # Try swaps: remove one, add one? No, that doesn't increase count.
                # We need to find a "free" triple.
                # If we can't add any, we are stuck.
                # Try to remove a triple that is "least useful"?
                # Or just restart with different seed.
                pass
                
        return count, current_set_indices

    # Better Local Search: 
    # 1. Greedy Add
    # 2. While possible: Pick a random triple T that is NOT present.
    #    If T can be added, add it.
    #    If T cannot be added, it conflicts with some config C (count 7).
    #    Pick a random triple U from C that IS present.
    #    Remove U.
    #    Try to add T.
    #    If T can be added, count stays same (remove 1, add 1). 
    #    BUT, removing U might unblock other triples!
    #    So after swap, we should try to add more.
    
    def run_local_search(seed_idx):
        random.seed(seed_idx)
        present[:] = False
        config_counts[:] = 0
        current_set = []
        
        # Greedy Phase: Random order
        order = list(range(num_triples))
        random.shuffle(order)
        
        for t_idx in order:
            if is_valid_addition(t_idx):
                add_triple(t_idx)
                current_set.append(t_idx)
                
        count = len(current_set)
        
        # Local Search Phase
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try to add any missing triple
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
                    # We can continue trying to add more in this pass
                    # But we need to be careful not to spend too much time
                    if time.time() > deadline - 2:
                        break
            
            if not added_any and time.time() < deadline - 2:
                # Try to unblock by swapping
                # Pick a random missing triple
                if missing:
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
                                current_set.append(t_idx)
                                count += 1
                                improved = True # We changed the state, might unblock others
                                # Note: count is same, but structure changed
                            else:
                                # Failed to add T, put U back? 
                                # Or keep U removed? 
                                # If we can't add T, we lost a triple. Bad.
                                # Put U back.
                                add_triple(u_idx)
                                current_set.append(u_idx)
                                count += 1
                                # Mark as no improvement for this step? 
                                # But we tried.
                                pass
                            
        return count, current_set

    best_count = 0
    best_set = []
    
    # Run multiple restarts
    num_restarts = 0
    while time.time() < deadline - 5:
        seed = num_restarts
        count, s = run_local_search(seed)
        if count > best_count:
            best_count = count
            best_set = s[:]
            # Print for debugging? No, keep quiet.
        num_restarts += 1
        
    # Convert indices back to triples
    result = [all_triples[i] for i in best_set]
    
    return result
