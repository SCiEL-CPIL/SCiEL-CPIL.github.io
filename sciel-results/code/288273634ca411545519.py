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

    def local_search_add(present):
        config_count = [0] * num_configs
        for t in range(num_triples):
            if present[t]:
                for ci in triple_to_configs[t]:
                    config_count[ci] += 1
        
        unsafe_count = [0] * num_triples
        safe_candidates = []
        for t in range(num_triples):
            if not present[t]:
                is_safe = True
                for ci in triple_to_configs[t]:
                    if config_count[ci] >= 7:
                        unsafe_count[t] += 1
                        is_safe = False
                if is_safe:
                    safe_candidates.append(t)
        
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            
            improved = False
            if safe_candidates:
                t = random.choice(safe_candidates)
                safe_candidates.remove(t)
                present[t] = True
                improved = True
                
                for ci in triple_to_configs[t]:
                    old_count = config_count[ci]
                    config_count[ci] = old_count + 1
                    
                    if old_count == 6:
                        for t_other in config_to_triples[ci]:
                            if t_other != t and not present[t_other]:
                                if unsafe_count[t_other] == 0:
                                    unsafe_count[t_other] = 1
                                    if t_other in safe_candidates:
                                        safe_candidates.remove(t_other)
                                else:
                                    unsafe_count[t_other] += 1
        
        return present

    def swap_search(present, max_steps=50):
        """Try to swap: remove one present triple, add two previously unsafe ones."""
        if time.time() > deadline - 5:
            return present
        
        config_count = [0] * num_configs
        for t in range(num_triples):
            if present[t]:
                for ci in triple_to_configs[t]:
                    config_count[ci] += 1
        
        unsafe_count = [0] * num_triples
        for t in range(num_triples):
            if not present[t]:
                for ci in triple_to_configs[t]:
                    if config_count[ci] >= 7:
                        unsafe_count[t] += 1
        
        # Try a few random swaps
        for _ in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            # Pick a random present triple to potentially remove
            present_indices = [t for t in range(num_triples) if present[t]]
            if not present_indices:
                break
            remove_t = random.choice(present_indices)
            
            # If we remove remove_t, which unsafe triples become safe?
            # A triple t is safe after removing remove_t if all configs containing t
            # have count <= 6 after the removal.
            # Configs containing remove_t will have their count reduced by 1.
            
            newly_safe = []
            for t in range(num_triples):
                if not present[t] and unsafe_count[t] > 0:
                    # Check if all configs with count >= 7 for t are among those containing remove_t
                    all_become_safe = True
                    for ci in triple_to_configs[t]:
                        if config_count[ci] >= 7:
                            if remove_t not in config_to_triples[ci]:
                                all_become_safe = False
                                break
                    if all_become_safe:
                        newly_safe.append(t)
            
            if len(newly_safe) < 2:
                continue
            
            # Pick two to add
            add_t1, add_t2 = random.sample(newly_safe, 2)
            
            # Verify: after removing remove_t and adding add_t1, add_t2, no config exceeds 7
            # Only check affected configs
            affected_configs = set()
            for ci in triple_to_configs[remove_t]:
                affected_configs.add(ci)
            for ci in triple_to_configs[add_t1]:
                affected_configs.add(ci)
            for ci in triple_to_configs[add_t2]:
                affected_configs.add(ci)
                
            valid = True
            for ci in affected_configs:
                new_count = config_count[ci]
                if remove_t in config_to_triples[ci]:
                    new_count -= 1
                if add_t1 in config_to_triples[ci]:
                    new_count += 1
                if add_t2 in config_to_triples[ci]:
                    new_count += 1
                if new_count > 7:
                    valid = False
                    break
            
            if valid:
                present[remove_t] = False
                present[add_t1] = True
                present[add_t2] = True
                # Update config_count
                for ci in triple_to_configs[remove_t]:
                    config_count[ci] -= 1
                for ci in triple_to_configs[add_t1]:
                    config_count[ci] += 1
                for ci in triple_to_configs[add_t2]:
                    config_count[ci] += 1
                # Recompute unsafe_count (expensive but rare)
                unsafe_count = [0] * num_triples
                for t in range(num_triples):
                    if not present[t]:
                        for ci in triple_to_configs[t]:
                            if config_count[ci] >= 7:
                                unsafe_count[t] += 1
                break
        
        return present

    best_present = None
    best_count = 0
    
    num_restarts = 0
    while time.time() < deadline - 15:
        present = greedy_delete()
        present = local_search_add(present)
        present = swap_search(present)
        
        count = sum(present)
        if count > best_count:
            best_count = count
            best_present = present
        
        num_restarts += 1
        if num_restarts > 100:
            break
    
    if best_present is None:
        return []

    return [list(all_triples[i]) for i in range(num_triples) if best_present[i]]
