import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145

    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)

    triple_index = {t: i for i, t in enumerate(all_triples)}

    # Enumerate all K(2,2,2) forbidden configurations
    forbidden_configs = []
    
    # For each set of 6 points, there are 15 ways to partition into 3 pairs
    # For each partition, the 8 transversals form a forbidden config
    for six_points in itertools.combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = six_points
        # All 15 pairings of 6 elements into 3 pairs
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
    for ci, config in enumerate(forbidden_configs):
        for t in config:
            triple_to_configs[t].append(ci)

    def construct_greedy(random_order):
        system = [False] * num_triples
        config_count = [0] * num_configs
        system_count = 0
        
        for t_idx in random_order:
            safe = True
            for ci in triple_to_configs[t_idx]:
                if config_count[ci] == 7:
                    safe = False
                    break
            
            if safe:
                system[t_idx] = True
                system_count += 1
                for ci in triple_to_configs[t_idx]:
                    config_count[ci] += 1
        
        return system, system_count

    def local_search(system, config_count, max_iters=10000):
        """Try to improve by swapping triples."""
        system = system[:]
        config_count = config_count[:]
        count = sum(1 for x in system if x)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Try removing a random selected triple and adding a random unselected one
            selected = [i for i in range(num_triples) if system[i]]
            unselected = [i for i in range(num_triples) if not system[i]]
            
            if not selected or not unselected:
                break
            
            # Try to find a beneficial swap
            improved = False
            random.shuffle(unselected)
            random.shuffle(selected)
            
            for new_t in unselected[:50]:
                if time.time() > deadline - 5:
                    break
                # Check if new_t is safe
                safe = True
                for ci in triple_to_configs[new_t]:
                    if config_count[ci] == 7:
                        safe = False
                        break
                if not safe:
                    continue
                
                # Try removing a random selected triple
                for old_t in selected[:50]:
                    # Remove old_t
                    system[old_t] = False
                    for ci in triple_to_configs[old_t]:
                        config_count[ci] -= 1
                    
                    # Check if new_t is now safe
                    safe2 = True
                    for ci in triple_to_configs[new_t]:
                        if config_count[ci] == 7:
                            safe2 = False
                            break
                    
                    if safe2:
                        system[new_t] = True
                        for ci in triple_to_configs[new_t]:
                            config_count[ci] += 1
                        count += 1
                        improved = True
                        break
                    
                    # Revert
                    for ci in triple_to_configs[old_t]:
                        config_count[ci] += 1
                    system[old_t] = True
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return system, config_count, count

    best_system = None
    best_count = 0
    
    # Phase 1: Multiple greedy passes
    while time.time() < deadline - 30:
        random_order = list(range(num_triples))
        random.shuffle(random_order)
        
        system, count = construct_greedy(random_order)
        
        if count > best_count:
            best_count = count
            best_system = system[:]
    
    # Phase 2: Local search on best system
    if best_system is not None and time.time() < deadline - 10:
        config_count = [0] * num_configs
        for t_idx in range(num_triples):
            if best_system[t_idx]:
                for ci in triple_to_configs[t_idx]:
                    config_count[ci] += 1
        
        system, config_count, count = local_search(best_system, config_count)
        if count > best_count:
            best_count = count
            best_system = system[:]
        
        # Try multiple local search starts
        while time.time() < deadline - 10:
            # Perturb the current best and try again
            system = best_system[:]
            config_count = [0] * num_configs
            for t_idx in range(num_triples):
                if system[t_idx]:
                    for ci in triple_to_configs[t_idx]:
                        config_count[ci] += 1
            
            # Remove a few random triples to perturb
            selected = [i for i in range(num_triples) if system[i]]
            random.shuffle(selected)
            for t_idx in selected[:3]:
                system[t_idx] = False
                for ci in triple_to_configs[t_idx]:
                    config_count[ci] -= 1
            
            system, config_count, count = local_search(system, config_count)
            if count > best_count:
                best_count = count
                best_system = system[:]
    
    if best_system is None:
        return [[0, a, b] for a in range(1, n) for b in range(a + 1, n)]
    
    return [list(all_triples[i]) for i in range(num_triples) if best_system[i]]
