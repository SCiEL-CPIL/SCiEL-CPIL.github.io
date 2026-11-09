import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n = 16
    points = list(range(n))
    
    # Generate all triples
    all_triples = list(itertools.combinations(points, 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Generate all K(2,2,2) configurations
    # A config is defined by 3 disjoint pairs: (a1, a2), (b1, b2), (c1, c2)
    # The 8 triples are (ai, bj, ck)
    
    # We need to iterate over all ways to pick 6 distinct points and partition them into 3 pairs.
    # Number of ways to pick 6 points from 16: C(16,6) = 8008
    # For each set of 6 points, number of ways to partition into 3 pairs is 15.
    # Total configs: 8008 * 15 = 120120
    
    configs = []
    # Mapping from triple index to list of config indices
    triple_to_configs = [[] for _ in range(num_triples)]
    
    # Helper to get index of a triple
    def get_idx(t):
        return triple_to_idx[t]
        
    # Iterate over all combinations of 6 points
    for p6 in itertools.combinations(points, 6):
        # Generate all 15 perfect matchings of these 6 points
        # Points: 0,1,2,3,4,5
        # Matchings:
        # (0,1)(2,3)(4,5)
        # (0,1)(2,4)(3,5)
        # (0,1)(2,5)(3,4)
        # (0,2)(1,3)(4,5)
        # (0,2)(1,4)(3,5)
        # (0,2)(1,5)(3,4)
        # (0,3)(1,2)(4,5)
        # (0,3)(1,4)(2,5)
        # (0,3)(1,5)(2,4)
        # (0,4)(1,2)(3,5)
        # (0,4)(1,3)(2,5)
        # (0,4)(1,5)(2,3)
        # (0,5)(1,2)(3,4)
        # (0,5)(1,3)(2,4)
        # (0,5)(1,4)(2,3)
        
        # We can generate these programmatically
        # Fix 0 with 1,2,3,4,5.
        # If 0-1, remaining 2,3,4,5 have 3 matchings.
        # If 0-2, remaining 1,3,4,5 have 3 matchings.
        # etc.
        
        matchings = []
        # Pair 0 with 1
        rest = [p6[1], p6[2], p6[3], p6[4], p6[5]]
        # Actually easier to just hardcode the 15 partitions of indices 0..5
        # and map to p6
        
        # Partitions of {0,1,2,3,4,5} into 3 pairs
        # 1. (0,1)(2,3)(4,5)
        # 2. (0,1)(2,4)(3,5)
        # 3. (0,1)(2,5)(3,4)
        # 4. (0,2)(1,3)(4,5)
        # 5. (0,2)(1,4)(3,5)
        # 6. (0,2)(1,5)(3,4)
        # 7. (0,3)(1,2)(4,5)
        # 8. (0,3)(1,4)(2,5)
        # 9. (0,3)(1,5)(2,4)
        # 10. (0,4)(1,2)(3,5)
        # 11. (0,4)(1,3)(2,5)
        # 12. (0,4)(1,5)(2,3)
        # 13. (0,5)(1,2)(3,4)
        # 14. (0,5)(1,3)(2,4)
        # 15. (0,5)(1,4)(2,3)
        
        partitions = [
            ((0,1), (2,3), (4,5)),
            ((0,1), (2,4), (3,5)),
            ((0,1), (2,5), (3,4)),
            ((0,2), (1,3), (4,5)),
            ((0,2), (1,4), (3,5)),
            ((0,2), (1,5), (3,4)),
            ((0,3), (1,2), (4,5)),
            ((0,3), (1,4), (2,5)),
            ((0,3), (1,5), (2,4)),
            ((0,4), (1,2), (3,5)),
            ((0,4), (1,3), (2,5)),
            ((0,4), (1,5), (2,3)),
            ((0,5), (1,2), (3,4)),
            ((0,5), (1,3), (2,4)),
            ((0,5), (1,4), (2,3))
        ]
        
        for part in partitions:
            # part is 3 pairs of indices into p6
            # p6 is a tuple of 6 points
            # We need to form the 8 triples
            # The pairs are (a1,a2), (b1,b2), (c1,c2)
            # We can assign the 3 pairs to a,b,c in any order?
            # The problem says "no three disjoint pairs have all eight transversal triples present".
            # The set of 8 triples is invariant to the order of the pairs.
            # So we just need the set of 8 triples.
            
            p_a1, p_a2 = p6[part[0][0]], p6[part[0][1]]
            p_b1, p_b2 = p6[part[1][0]], p6[part[1][1]]
            p_c1, p_c2 = p6[part[2][0]], p6[part[2][1]]
            
            # Generate 8 triples
            t_indices = []
            for i in range(2):
                a = p_a1 if i==0 else p_a2
                for j in range(2):
                    b = p_b1 if j==0 else p_b2
                    for k in range(2):
                        c = p_c1 if k==0 else p_c2
                        t = tuple(sorted((a,b,c)))
                        t_indices.append(triple_to_idx[t])
                        
            config_id = len(configs)
            configs.append(t_indices)
            for t_idx in t_indices:
                triple_to_configs[t_idx].append(config_id)
                
    num_configs = len(configs)
    
    # State
    in_set = [False] * num_triples
    conf_counts = [0] * num_configs
    current_size = 0
    
    def add_triple(t_idx):
        nonlocal current_size
        # Check validity
        for c in triple_to_configs[t_idx]:
            if conf_counts[c] >= 7:
                return False
        # Add
        in_set[t_idx] = True
        current_size += 1
        for c in triple_to_configs[t_idx]:
            conf_counts[c] += 1
        return True
        
    def remove_triple(t_idx):
        nonlocal current_size
        in_set[t_idx] = False
        current_size -= 1
        for c in triple_to_configs[t_idx]:
            conf_counts[c] -= 1
            
    # Greedy Initialization
    # Random order
    order = list(range(num_triples))
    random.shuffle(order)
    for t_idx in order:
        if add_triple(t_idx):
            pass
            
    best_size = current_size
    best_set = in_set[:]
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    alpha = 0.999
    
    # Precompute list of indices for faster access?
    # We need to pick random t_idx not in set, and random t_idx in set.
    
    while time.time() < deadline:
        # Pick a move
        # 1. Add random
        # 2. Remove random
        # 3. Swap
        
        move_type = random.random()
        if move_type < 0.3:
            # Add
            # Pick random index
            t_idx = random.randint(0, num_triples - 1)
            if not in_set[t_idx]:
                # Try to add
                valid = True
                for c in triple_to_configs[t_idx]:
                    if conf_counts[c] >= 7:
                        valid = False
                        break
                if valid:
                    add_triple(t_idx)
                    if current_size > best_size:
                        best_size = current_size
                        best_set = in_set[:]
                else:
                    # Rejection
                    pass
        elif move_type < 0.6:
            # Remove
            # Pick random index in set
            # Efficient removal: keep a list of active indices?
            # Or just pick random and check.
            # Since set is dense, random pick is O(1) check.
            t_idx = random.randint(0, num_triples - 1)
            if in_set[t_idx]:
                remove_triple(t_idx)
                # Accept removal? 
                # We want to maximize size. Removal decreases size.
                # We only accept if it leads to a better state later? 
                # In SA, we accept with prob exp(-delta/T).
                # Delta = -1.
                # But we don't know if it leads to better state.
                # Standard SA on Max Independent Set:
                # Usually we do swaps or add/remove.
                # If we remove, we must add later to recover size.
                # Let's stick to Swap and Add. Removal alone is bad unless it unlocks many adds.
                # Let's just do Swap and Add.
                # Revert removal
                add_triple(t_idx)
        else:
            # Swap
            # Pick t_out from set, t_in from not set
            # If set is large, picking random in set is fast.
            # Picking random not in set: might need retries.
            
            # Pick t_out
            t_out = random.randint(0, num_triples - 1)
            if not in_set[t_out]:
                continue
                
            # Pick t_in
            # Try a few times to find one not in set
            t_in = -1
            for _ in range(10):
                cand = random.randint(0, num_triples - 1)
                if not in_set[cand]:
                    t_in = cand
                    break
            if t_in == -1:
                continue
                
            # Check if swap is valid
            # Remove t_out temporarily?
            # We need to check if t_in can be added AFTER t_out is removed.
            # Current state: t_out in, t_in out.
            # New state: t_out out, t_in in.
            
            # Check validity of t_in in state without t_out
            # For each config c in t_in:
            # count_new = conf_counts[c] - (1 if t_out in c else 0)
            # We need count_new < 7.
            
            valid = True
            # Precompute if t_out is in configs of t_in?
            # It's hard to know quickly without checking.
            # But we can just iterate configs of t_in.
            
            # We also need to ensure that removing t_out doesn't break anything?
            # Removing a triple never creates a K(2,2,2) if it wasn't there.
            # It only reduces counts. So removing is always "safe" in terms of validity of existing triples.
            # The only risk is adding t_in.
            
            for c in triple_to_configs[t_in]:
                # Count of t_in's configs in current set
                cnt = conf_counts[c]
                # Subtract t_out if it contributes
                # How to check if t_out is in config c?
                # We don't have a direct map.
                # But we can check: is t_out one of the 8 triples in config c?
                # We have `configs` list.
                # `configs[c]` is a list of 8 indices.
                # Checking membership in a list of 8 is O(8).
                # Total O(1700 * 8) = 13600 ops. A bit slow.
                
                # Alternative:
                # We can precompute a set of configs for t_out?
                # No, we need to know if they intersect.
                
                # Let's just do the check.
                # If t_out is in configs[c], then cnt_effective = cnt - 1
                # else cnt_effective = cnt
                # We need cnt_effective < 7.
                
                # To check if t_out is in configs[c]:
                # We can't do it fast.
                # However, note that t_out and t_in are distinct.
                # If they share a config, they are in the same K(2,2,2).
                
                # Let's optimize:
                # If conf_counts[c] < 7, it's safe regardless of t_out.
                # Only if conf_counts[c] == 7, we need to check if t_out is in c.
                # If t_out is in c, then removing it makes count 6, so safe.
                # If t_out is NOT in c, count stays 7, so unsafe.
                
                if conf_counts[c] >= 7:
                    # Check if t_out is in config c
                    # configs[c] is a list of 8 ints
                    if t_out not in configs[c]:
                        valid = False
                        break
                        
            if valid:
                # Perform swap
                remove_triple(t_out)
                add_triple(t_in)
                
                if current_size > best_size:
                    best_size = current_size
                    best_set = in_set[:]
            else:
                # Rejection
                pass
                
        # Update temperature
        T *= alpha
        
    # Construct result
    result = []
    for i in range(num_triples):
        if best_set[i]:
            result.append(list(all_triples[i]))
            
    return result

