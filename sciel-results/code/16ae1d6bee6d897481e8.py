import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)
    
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    # Enumerate all K(2,2,2) forbidden configurations
    forbidden_configs = []
    
    for six_points in itertools.combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = six_points
        pairs_list = [
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
        
        for pairs in pairs_list:
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
    
    best_system = None
    best_count = 0
    
    # Phase 1: Greedy with random orderings
    while time.time() < deadline - 25:
        random_order = list(range(num_triples))
        random.shuffle(random_order)
        
        system, count = construct_greedy(random_order)
        
        if count > best_count:
            best_count = count
            best_system = system[:]
    
    # Phase 2: Local search - try to add more triples
    if best_system is not None and time.time() < deadline - 5:
        system = best_system[:]
        config_count = [0] * num_configs
        for t_idx in range(num_triples):
            if system[t_idx]:
                for ci in triple_to_configs[t_idx]:
                    config_count[ci] += 1
        
        candidate_triples = [t_idx for t_idx in range(num_triples) if not system[t_idx]]
        random.shuffle(candidate_triples)
        
        while time.time() < deadline - 5:
            added = False
            for i, t_idx in enumerate(candidate_triples):
                safe = True
                for ci in triple_to_configs[t_idx]:
                    if config_count[ci] == 7:
                        safe = False
                        break
                
                if safe:
                    system[t_idx] = True
                    for ci in triple_to_configs[t_idx]:
                        config_count[ci] += 1
                    best_count = sum(1 for x in system if x)
                    best_system = system[:]
                    candidate_triples.pop(i)
                    added = True
                    break
            
            if not added:
                break
    
    if best_system is None:
        return [[0, a, b] for a in range(1, n) for b in range(a + 1, n)]
    
    return [list(all_triples[i]) for i in range(num_triples) if best_system[i]]
