import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (150s limit)
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)  # 560
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    # Enumerate all K(2,2,2) forbidden configurations
    # A K(2,2,2) is three disjoint pairs {a1,b1}, {a2,b2}, {a3,b3}
    # The 8 transversal triples are all choices (ai, aj, ak) for i,j,k in {0,1}
    forbidden_configs = set()
    
    for six_points in itertools.combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = six_points
        # All 15 ways to partition 6 points into 3 pairs
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
                forbidden_configs.add(frozenset(transversals))
    
    forbidden_configs = list(forbidden_configs)
    num_configs = len(forbidden_configs)
    
    # Build incidence: which configs each triple belongs to
    triple_to_configs = [[] for _ in range(num_triples)]
    config_to_triples = [list(c) for c in forbidden_configs]
    for ci, config in enumerate(forbidden_configs):
        for t in config:
            triple_to_configs[t].append(ci)
    
    # Greedy deletion: start with all triples, remove to eliminate all forbidden configs
    def greedy_delete():
        present = [True] * num_triples
        config_count = [8] * num_configs  # each config has 8 triples initially
        active_configs = [True] * num_configs
        num_active = num_configs
        
        # Count how many active configs each triple is in
        triple_active_count = [0] * num_triples
        for t in range(num_triples):
            for ci in triple_to_configs[t]:
                triple_active_count[t] += 1
        
        while num_active > 0:
            if time.time() > deadline - 15:
                break
            
            # Find the triple that is in the most active configs (to delete)
            # Among triples that are in at least one active config
            best_t = -1
            best_count = 0
            for t in range(num_triples):
                if not present[t]:
                    continue
                if triple_active_count[t] > best_count:
                    best_count = triple_active_count[t]
                    best_t = t
            
            if best_t == -1:
                break
            
            # Delete this triple
            present[best_t] = False
            for ci in triple_to_configs[best_t]:
                config_count[ci] -= 1
                if config_count[ci] == 0 and active_configs[ci]:
                    active_configs[ci] = False
                    num_active -= 1
                    # Update triple_active_count for other triples in this config
                    for t in config_to_triples[ci]:
                        if present[t]:
                            triple_active_count[t] -= 1
        
        return present, config_count
    
    # Local search: try to add back deleted triples if safe
    def local_search_add(present, config_count, max_iterations=5000):
        present = present[:]
        config_count = config_count[:]
        count = sum(present)
        
        # Precompute which triples are not present
        not_present = [i for i in range(num_triples) if not present[i]]
        
        for _ in range(max_iterations):
            if time.time() > deadline - 10:
                break
            if not not_present:
                break
            
            # Pick a random not-present triple
            idx = random.randint(0, len(not_present) - 1)
            t = not_present[idx]
            
            # Check if adding it is safe
            safe = True
            for ci in triple_to_configs[t]:
                if config_count[ci] == 7:
                    safe = False
                    break
            
            if safe:
                present[t] = True
                for ci in triple_to_configs[t]:
                    config_count[ci] += 1
                count += 1
                not_present.pop(idx)
        
        return present, config_count, count
    
    # Simulated annealing style: try swaps (remove one, add another)
    def swap_search(present, config_count, max_iterations=3000):
        present = present[:]
        config_count = config_count[:]
        count = sum(present)
        
        for _ in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            # Try: remove a random present triple, then add a random not-present triple
            present_indices = [i for i in range(num_triples) if present[i]]
            if not present_indices:
                break
            
            remove_t = random.choice(present_indices)
            
            # Find a safe triple to add
            added = False
            for attempt in range(20):
                add_t = random.randint(0, num_triples - 1)
                if present[add_t]:
                    continue
                
                # Check if we can swap: remove remove_t, add add_t
                # After removing remove_t, check if add_t is safe
                safe = True
                for ci in triple_to_configs[add_t]:
                    # config_count[ci] will be config_count[ci] - (1 if remove_t in config ci else 0)
                    current = config_count[ci]
                    if remove_t in config_to_triples[ci]:
                        current -= 1
                    if current == 7:
                        safe = False
                        break
                
                if safe:
                    # Perform the swap
                    present[remove_t] = False
                    for ci in triple_to_configs[remove_t]:
                        config_count[ci] -= 1
                    
                    present[add_t] = True
                    for ci in triple_to_configs[add_t]:
                        config_count[ci] += 1
                    added = True
                    break
            
            # If we didn't add, we haven't changed anything
        
        return present, config_count, count
    
    best_system = None
    best_count = 0
    
    # Phase 1: Greedy deletion
    present, config_count = greedy_delete()
    count = sum(present)
    if count > best_count:
        best_count = count
        best_system = present[:]
    
    # Phase 2: Local search to add back triples
    present, config_count, count = local_search_add(present, config_count)
    if count > best_count:
        best_count = count
        best_system = present[:]
    
    # Phase 3: More rounds of greedy delete + local search with different random seeds
    while time.time() < deadline - 20:
        random_order = list(range(num_triples))
        random.shuffle(random_order)
        
        # Greedy deletion with a random tie-breaking
        present = [True] * num_triples
        config_count = [8] * num_configs
        active_configs = [True] * num_configs
        num_active = num_configs
        
        triple_active_count = [0] * num_triples
        for t in range(num_triples):
            for ci in triple_to_configs[t]:
                triple_active_count[t] += 1
        
        while num_active > 0:
            if time.time() > deadline - 20:
                break
            
            # Find candidates with max active config count
            max_count = 0
            for t in range(num_triples):
                if present[t] and triple_active_count[t] > max_count:
                    max_count = triple_active_count[t]
            
            if max_count == 0:
                break
            
            candidates = [t for t in range(num_triples) 
                         if present[t] and triple_active_count[t] == max_count]
            best_t = random.choice(candidates)
            
            present[best_t] = False
            for ci in triple_to_configs[best_t]:
                config_count[ci] -= 1
                if config_count[ci] == 0 and active_configs[ci]:
                    active_configs[ci] = False
                    num_active -= 1
                    for t in config_to_triples[ci]:
                        if present[t]:
                            triple_active_count[t] -= 1
        
        count = sum(present)
        
        # Local search
        present, config_count, count = local_search_add(present, config_count, max_iterations=3000)
        
        if count > best_count:
            best_count = count
            best_system = present[:]
        
        # Try swap search from the best state
        if time.time() < deadline - 15:
            ps, cs, sc = swap_search(best_system, config_count, max_iterations=2000)
            if sc > best_count:
                best_count = sc
                best_system = ps[:]
    
    if best_system is None:
        # Fallback
        return [list(all_triples[i]) for i in range(num_triples)]
    
    return [list(all_triples[i]) for i in range(num_triples) if best_system[i]]
