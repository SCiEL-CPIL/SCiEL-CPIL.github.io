import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    # Enumerate all K(2,2,2) forbidden configurations
    forbidden_configs = []
    
    for six_points in itertools.combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = six_points
        pairings = [
            [(p0,p1),(p2,p3),(p4,p5)],
            [(p0,p1),(p2,p4),(p3,p5)],
            [(p0,p1),(p2,p5),(p3,p4)],
            [(p0,p2),(p1,p3),(p4,p5)],
            [(p0,p2),(p1,p4),(p3,p5)],
            [(p0,p2),(p1,p5),(p3,p4)],
            [(p0,p3),(p1,p2),(p4,p5)],
            [(p0,p3),(p1,p4),(p2,p5)],
            [(p0,p3),(p1,p5),(p2,p4)],
            [(p0,p4),(p1,p2),(p3,p5)],
            [(p0,p4),(p1,p3),(p2,p5)],
            [(p0,p4),(p1,p5),(p2,p3)],
            [(p0,p5),(p1,p2),(p3,p4)],
            [(p0,p5),(p1,p3),(p2,p4)],
            [(p0,p5),(p1,p4),(p2,p3)],
        ]
        
        for pairs in pairings:
            (a1,b1), (a2,b2), (a3,b3) = pairs
            transversals = []
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        t = tuple(sorted([(a1,b1)[i], (a2,b2)[j], (a3,b3)[k]]))
                        if t in triple_index:
                            transversals.append(triple_index[t])
            
            if len(transversals) == 8:
                forbidden_configs.append(frozenset(transversals))
    
    forbidden_configs = list(set(forbidden_configs))
    num_configs = len(forbidden_configs)
    
    triple_to_configs = [[] for _ in range(num_triples)]
    config_to_triples = [set(c) for c in forbidden_configs]
    for ci, config in enumerate(forbidden_configs):
        for t in config:
            triple_to_configs[t].append(ci)
    
    def greedy_delete():
        present = [True] * num_triples
        active = [True] * num_configs
        
        triple_active_count = [0] * num_triples
        for t in range(num_triples):
            for ci in triple_to_configs[t]:
                if active[ci]:
                    triple_active_count[t] += 1
        
        num_active = num_configs
        
        while num_active > 0:
            if time.time() > deadline - 10:
                break
            
            max_count = -1
            candidates = []
            for t in range(num_triples):
                if present[t]:
                    if triple_active_count[t] > max_count:
                        max_count = triple_active_count[t]
                        candidates = [t]
                    elif triple_active_count[t] == max_count:
                        candidates.append(t)
            
            if not candidates:
                break
                
            best_t = random.choice(candidates)
            
            present[best_t] = False
            
            for ci in triple_to_configs[best_t]:
                if active[ci]:
                    active[ci] = False
                    num_active -= 1
                    
                    for t_other in config_to_triples[ci]:
                        if t_other != best_t and present[t_other]:
                            triple_active_count[t_other] -= 1
            
        return present

    def local_search(present):
        """
        Unified local search: Add 1, Swap 1-1, Swap 1-2.
        Maintains config_count and unsafe_count.
        """
        config_count = [0] * num_configs
        for t in range(num_triples):
            if present[t]:
                for ci in triple_to_configs[t]:
                    config_count[ci] += 1
        
        # unsafe_count[t] = number of configs c containing t where config_count[c] >= 7
        unsafe_count = [0] * num_triples
        for t in range(num_triples):
            if not present[t]:
                for ci in triple_to_configs[t]:
                    if config_count[ci] >= 7:
                        unsafe_count[t] += 1
        
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # 1. Try Add 1
            safe_candidates = [t for t in range(num_triples) if not present[t] and unsafe_count[t] == 0]
            if safe_candidates:
                t_add = random.choice(safe_candidates)
                present[t_add] = True
                improved = True
                
                # Update config_count
                for ci in triple_to_configs[t_add]:
                    config_count[ci] += 1
                    
                    # Update unsafe_count for other absent triples in this config
                    if config_count[ci] == 7:
                        for t_other in config_to_triples[ci]:
                            if not present[t_other] and t_other != t_add:
                                unsafe_count[t_other] += 1
                continue

            # 2. Try Swap 1-1
            # Find t_in (absent) and t_out (present) such that:
            # - t_in is unsafe (unsafe_count[t_in] > 0)
            # - Let C_in_be = {c | t_in in c and config_count[c] >= 7}
            # - t_out must be in ALL c in C_in_be
            # - For all c containing t_in but NOT t_out, config_count[c] <= 6.
            
            # Optimization: Iterate over unsafe t_in
            # To speed up, we can limit the number of t_in we check per iteration
            checked = 0
            found_swap = False
            for t_in in range(num_triples):
                if checked > 200: # Limit work per iteration
                    break
                if present[t_in]:
                    continue
                if unsafe_count[t_in] == 0:
                    continue # Would have been caught by Add 1
                
                checked += 1
                
                # Get critical configs for t_in
                critical_configs = [ci for ci in triple_to_configs[t_in] if config_count[ci] >= 7]
                if not critical_configs:
                    continue
                
                # Find intersection of triples in critical_configs
                # We need t_out to be in all these configs.
                # Start with triples of first config
                intersection = set(config_to_triples[critical_configs[0]])
                for ci in critical_configs[1:]:
                    intersection &= config_to_triples[ci]
                    if not intersection:
                        break
                
                if not intersection:
                    continue
                
                # Pick a random t_out from intersection
                # t_out must be present
                t_out_candidates = [t for t in intersection if present[t]]
                if not t_out_candidates:
                    continue
                
                t_out = random.choice(t_out_candidates)
                
                # Verify validity of swap (t_out -> t_in)
                # Configs affected: those containing t_in or t_out
                valid = True
                for ci in triple_to_configs[t_in]:
                    if t_out in config_to_triples[ci]:
                        # Count unchanged
                        pass
                    else:
                        # Count increases by 1
                        if config_count[ci] >= 7:
                            valid = False
                            break
                
                if valid:
                    # Perform swap
                    present[t_out] = False
                    present[t_in] = True
                    improved = True
                    found_swap = True
                    
                    # Update config_count
                    for ci in triple_to_configs[t_out]:
                        config_count[ci] -= 1
                        # If count drops to 6, some unsafe_count might decrease
                        # Specifically, for t_other in config, if config_count[ci] was 7 and is now 6,
                        # and t_other is absent, unsafe_count[t_other] decreases by 1.
                        if config_count[ci] == 6:
                            for t_other in config_to_triples[ci]:
                                if not present[t_other]:
                                    unsafe_count[t_other] -= 1
                            
                    for ci in triple_to_configs[t_in]:
                        config_count[ci] += 1
                        # If count rises to 7, some unsafe_count might increase
                        if config_count[ci] == 7:
                            for t_other in config_to_triples[ci]:
                                if not present[t_other]:
                                    unsafe_count[t_other] += 1
                    break
            
            if found_swap:
                continue

            # 3. Try Swap 1-2
            # Find t_out (present) and t_in1, t_in2 (absent)
            # This is harder. We'll just try a few random t_outs and see if we can find 2 t_ins.
            # For a given t_out, which t_ins become safe?
            # t_in is safe if all its critical configs contain t_out.
            
            # Pick a random t_out
            present_indices = [t for t in range(num_triples) if present[t]]
            if not present_indices:
                break
                
            t_out = random.choice(present_indices)
            
            # Find t_ins that become safe if t_out is removed
            # We need t_in such that for all c in critical_configs(t_in), t_out in c.
            # And for all c containing t_in but not t_out, config_count[c] <= 6.
            
            newly_safe = []
            for t_in in range(num_triples):
                if present[t_in]:
                    continue
                if unsafe_count[t_in] == 0:
                    # Already safe, but we are in Swap 1-2 mode, so we assume we need 2.
                    # Actually if one is safe, we could do Swap 1-1 or Add 1.
                    # But we are here because Add 1 and Swap 1-1 failed.
                    # So unsafe_count[t_in] > 0 for all absent t_in? No, maybe we missed some.
                    # But if unsafe_count[t_in] == 0, it would have been added in step 1.
                    continue
                
                critical_configs = [ci for ci in triple_to_configs[t_in] if config_count[ci] >= 7]
                if not critical_configs:
                    continue
                
                all_contain_t_out = True
                for ci in critical_configs:
                    if t_out not in config_to_triples[ci]:
                        all_contain_t_out = False
                        break
                
                if not all_contain_t_out:
                    continue
                
                # Check if adding t_in (with t_out removed) is valid
                # i.e., for all c containing t_in but not t_out, config_count[c] <= 6
                valid = True
                for ci in triple_to_configs[t_in]:
                    if t_out not in config_to_triples[ci]:
                        if config_count[ci] >= 7:
                            valid = False
                            break
                
                if valid:
                    newly_safe.append(t_in)
            
            if len(newly_safe) >= 2:
                # Pick two
                t_in1, t_in2 = random.sample(newly_safe, 2)
                
                # Verify swap (t_out -> t_in1, t_in2)
                valid = True
                for ci in triple_to_configs[t_in1]:
                    if t_out in config_to_triples[ci]:
                        # Count change: -1 (t_out) + 1 (t_in1) = 0?
                        # Wait, t_in2 might also be in ci.
                        pass
                # We need to check the net change for each config.
                # Net change = (1 if t_in1 in c) + (1 if t_in2 in c) - (1 if t_out in c)
                # New count = config_count[c] + net_change
                # Must be <= 7.
                
                affected_configs = set()
                for ci in triple_to_configs[t_out]:
                    affected_configs.add(ci)
                for ci in triple_to_configs[t_in1]:
                    affected_configs.add(ci)
                for ci in triple_to_configs[t_in2]:
                    affected_configs.add(ci)
                    
                for ci in affected_configs:
                    change = 0
                    if t_in1 in config_to_triples[ci]:
                        change += 1
                    if t_in2 in config_to_triples[ci]:
                        change += 1
                    if t_out in config_to_triples[ci]:
                        change -= 1
                    new_count = config_count[ci] + change
                    if new_count > 7:
                        valid = False
                        break
                
                if valid:
                    # Perform swap
                    present[t_out] = False
                    present[t_in1] = True
                    present[t_in2] = True
                    improved = True
                    
                    # Update config_count
                    for ci in triple_to_configs[t_out]:
                        config_count[ci] -= 1
                        if config_count[ci] == 6:
                            for t_other in config_to_triples[ci]:
                                if not present[t_other]:
                                    unsafe_count[t_other] -= 1
                    for ci in triple_to_configs[t_in1]:
                        config_count[ci] += 1
                        if config_count[ci] == 7:
                            for t_other in config_to_triples[ci]:
                                if not present[t_other]:
                                    unsafe_count[t_other] += 1
                    for ci in triple_to_configs[t_in2]:
                        config_count[ci] += 1
                        if config_count[ci] == 7:
                            for t_other in config_to_triples[ci]:
                                if not present[t_other]:
                                    unsafe_count[t_other] += 1
                    break

        return present

    best_present = None
    best_count = 0
    
    num_restarts = 0
    while time.time() < deadline - 15:
        present = greedy_delete()
        present = local_search(present)
        
        count = sum(present)
        if count > best_count:
            best_count = count
            best_present = present
        
        num_restarts += 1
        if num_restarts > 200:
            break
    
    if best_present is None:
        return []

    return [list(all_triples[i]) for i in range(num_triples) if best_present[i]]
