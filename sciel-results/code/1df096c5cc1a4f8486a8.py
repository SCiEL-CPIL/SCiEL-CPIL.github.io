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
    config_to_triples = [list(c) for c in forbidden_configs]
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
            candidates = []
            for t in range(num_triples):
                if not present[t] and unsafe_count[t] == 0:
                    candidates.append(t)
            
            if candidates:
                t = random.choice(candidates)
                present[t] = True
                improved = True
                
                for ci in triple_to_configs[t]:
                    old_count = config_count[ci]
                    config_count[ci] = old_count + 1
                    
                    if old_count == 6:
                        for t_other in config_to_triples[ci]:
                            if t_other != t and not present[t_other]:
                                unsafe_count[t_other] += 1
        
        return present
    
    def swap_search(present):
        """Try 1-swap moves: remove one present triple, add one absent triple."""
        config_count = [0] * num_configs
        for t in range(num_triples):
            if present[t]:
                for ci in triple_to_configs[t]:
                    config_count[ci] += 1
        
        # unsafe_count[t] for absent t: number of configs where adding t would complete
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
            
            # Try to find a beneficial swap: remove t1 (present), add t2 (absent)
            # such that the net effect is safe and we can then add more
            # Simple approach: try removing a "bad" present triple and adding a "good" absent one
            
            # Find present triples with high config participation
            present_triples = [t for t in range(num_triples) if present[t]]
            absent_triples = [t for t in range(num_triples) if not present[t] and unsafe_count[t] == 0]
            
            if not present_triples or not absent_triples:
                break
            
            # Randomly try a few swaps
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                
                t_remove = random.choice(present_triples)
                t_add = random.choice(absent_triples)
                
                # Simulate: remove t_remove, check if t_add becomes safe
                # After removing t_remove, config counts decrease for configs containing t_remove
                new_config_count = config_count[:]
                for ci in triple_to_configs[t_remove]:
                    new_config_count[ci] -= 1
                
                # Check if t_add is safe with new counts
                safe = True
                for ci in triple_to_configs[t_add]:
                    if new_config_count[ci] >= 7:
                        safe = False
                        break
                
                if safe:
                    # Perform the swap
                    present[t_remove] = False
                    present[t_add] = True
                    
                    # Update config_count
                    for ci in triple_to_configs[t_remove]:
                        config_count[ci] -= 1
                    for ci in triple_to_configs[t_add]:
                        config_count[ci] += 1
                    
                    # Update unsafe_count
                    # Reset and recompute for efficiency in this small case
                    unsafe_count = [0] * num_triples
                    for t in range(num_triples):
                        if not present[t]:
                            for ci in triple_to_configs[t]:
                                if config_count[ci] >= 7:
                                    unsafe_count[t] += 1
                    
                    improved = True
                    break
        
        # After swaps, try to add more
        present = local_search_add(present)
        
        return present
    
    best_present = None
    best_count = 0
    
    # Run multiple instances of Greedy Delete + Local Search + Swap Search
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
