import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)  # 560

    # Precompute index for O(1) lookup
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Precompute all K(2,2,2) configurations
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
    
    def add_triple(t_idx):
        present[t_idx] = True
        for cid in triple_to_configs[t_idx]:
            config_counts[cid] += 1
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

    def get_violations():
        # Returns list of configs that are fully present (count == 8)
        # In SA, we allow violations but penalize them
        # Actually, for SA to work well, we usually want to stay in valid space or allow small violations.
        # Let's stick to valid space but use moves that might temporarily violate?
        # No, the constraint is hard. We must maintain validity.
        # So the "energy" is just -count. We want to maximize count.
        # The challenge is getting stuck.
        # Moves: 
        # 1. Add 1 triple (if valid)
        # 2. Remove 1 triple
        # 3. Remove k triples, Add k triples (simultaneous)
        
        # To make "Remove k, Add k" work, we need to ensure the final state is valid.
        # This is hard to check incrementally if we remove first.
        # Strategy: 
        #   - Pick k triples to remove.
        #   - Pick k triples to add.
        #   - Check if the set (Current - Remove + Add) is valid.
        #   - This requires checking all configs affected by these 2k triples.
        
        pass

    def run_simulated_annealing(seed_idx):
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
                
        current_count = len(current_set)
        best_count = current_count
        best_set = list(current_set)
        
        # SA Parameters
        T = 1.0
        T_min = 0.01
        alpha = 0.995 # Cooling rate
        
        # Precompute list of indices for faster random choice?
        all_indices = list(range(num_triples))
        
        while T > T_min and time.time() < deadline - 5:
            # Choose move type
            # 50% Add 1, 20% Remove 1, 30% Swap k (k=2,3)
            r = random.random()
            
            if r < 0.5:
                # Try to add 1 random missing triple
                missing = [i for i in all_indices if not present[i]]
                if not missing:
                    T *= alpha
                    continue
                t_idx = random.choice(missing)
                if is_valid_addition(t_idx):
                    add_triple(t_idx)
                    current_set.add(t_idx)
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_set = list(current_set)
                else:
                    # Penalty? In SA, we usually accept worse states. 
                    # But we can't add invalid triples. 
                    # So "worse" means we failed to improve. 
                    # Standard SA on MaxSAT usually flips bits. 
                    # Here, we are constrained to valid space. 
                    # So we can only accept moves that result in a valid state.
                    # If we can't add, we just don't add. This is effectively a random walk in the valid space.
                    # To escape local maxima, we need to REMOVE triples.
                    pass
            elif r < 0.7:
                # Try to remove 1 random present triple
                if current_count > 0:
                    t_idx = random.choice(list(current_set))
                    remove_triple(t_idx)
                    current_set.remove(t_idx)
                    current_count -= 1
                    # Acceptance: Since we are maximizing, removing is always "worse" in terms of count.
                    # We accept it with probability exp(-delta/T) where delta = 1 (loss of 1 triple).
                    # But wait, if we remove, we might be able to add 2 later?
                    # The move is "Remove 1". The state is valid. 
                    # We accept if random() < exp(-1/T).
                    if random.random() < np.exp(-1.0 / T):
                        # Keep the removal
                        pass
                    else:
                        # Revert
                        add_triple(t_idx)
                        current_set.add(t_idx)
                        current_count += 1
            else:
                # Try to Remove k, Add k
                k = random.choice([2, 3])
                if current_count < k:
                    T *= alpha
                    continue
                
                # Pick k triples to remove
                to_remove = random.sample(list(current_set), k)
                # Pick k triples to add (from missing)
                missing = [i for i in all_indices if not present[i]]
                if len(missing) < k:
                    T *= alpha
                    continue
                to_add = random.sample(missing, k)
                
                # Check if (Current - to_remove + to_add) is valid
                # We need to check all configs affected by to_remove or to_add
                affected_configs = set()
                for t in to_remove:
                    affected_configs.update(triple_to_configs[t])
                for t in to_add:
                    affected_configs.update(triple_to_configs[t])
                    
                valid = True
                # Simulate the change on config counts
                # We can just check the final counts for affected configs
                # But we need to be careful: if a config is affected by both, we need net change.
                
                # Let's just do a temporary removal/addition and check, then revert if invalid.
                # Since k is small, this is fast.
                
                # Save state of affected configs? No, just modify and check.
                # But if we modify and it's invalid, we must revert.
                
                # Optimization: Check validity before modifying?
                # For each affected config, calculate new count.
                # New count = Old Count - (# in to_remove) + (# in to_add)
                # Must be <= 7.
                
                # Also, we must ensure we don't add a triple that is already present (handled by sampling from missing)
                # And we don't remove a triple not present (handled by sampling from current)
                
                # Check constraint
                for cid in affected_configs:
                    old_count = config_counts[cid]
                    num_removed = 0
                    num_added = 0
                    # Count how many of to_remove are in this config
                    # And how many of to_add are in this config
                    # This is O(k) per config. k is small.
                    # But checking membership in a list of 8 is fast.
                    
                    # Actually, triple_to_configs gives us which configs a triple is in.
                    # So for a config cid, we can check if t is in triples_in_config[cid].
                    # Better: precompute a set for each config? 
                    # triples_in_config[cid] is a list of 8 indices.
                    # Let's use a set for faster lookup if we do this often.
                    # For now, linear scan in list of 8 is fine.
                    
                    for t in to_remove:
                        if t in triples_in_config[cid]:
                            num_removed += 1
                    for t in to_add:
                        if t in triples_in_config[cid]:
                            num_added += 1
                    
                    new_count = old_count - num_removed + num_added
                    if new_count > 7:
                        valid = False
                        break
                        
                if valid:
                    # Apply move
                    for t in to_remove:
                        remove_triple(t)
                        current_set.remove(t)
                    for t in to_add:
                        add_triple(t)
                        current_set.add(t)
                        
                    # Count doesn't change! We removed k and added k.
                    # So current_count is the same.
                    # But the structure changed.
                    # This allows us to jump to a different part of the valid space.
                    # We accept this move always? 
                    # In SA, if energy (count) doesn't change, we accept with prob 1 (or close to it).
                    # Actually, if count is same, it's a neutral move. We accept it to explore.
                    
                    # Check if we found a better state? No, count is same.
                    # But maybe from this new state, we can add more?
                    # The SA loop will handle adding/removing in subsequent steps.
                    
                else:
                    # Move rejected, do nothing
                    pass
            
            T *= alpha
            
        return best_count, best_set

    best_count = 0
    best_set = []
    
    # Run multiple restarts
    num_restarts = 0
    while time.time() < deadline - 10:
        seed = num_restarts
        count, s = run_simulated_annealing(seed)
        if count > best_count:
            best_count = count
            best_set = s
        num_restarts += 1
        
    # Convert indices back to triples
    result = [all_triples[i] for i in best_set]
    
    return result
