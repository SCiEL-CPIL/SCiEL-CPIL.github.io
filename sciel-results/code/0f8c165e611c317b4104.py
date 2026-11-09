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
        
        return system, system_count, config_count
    
    def local_search(system, config_count, max_steps=200):
        """Simple local search: try to add safe triples."""
        system = system[:]
        config_count = config_count[:]
        count = sum(1 for x in system if x)
        
        for _ in range(max_steps):
            if time.time() > deadline - 3:
                break
            
            # Try a random unselected triple
            t_idx = random.randint(0, num_triples - 1)
            if system[t_idx]:
                continue
            
            safe = True
            for ci in triple_to_configs[t_idx]:
                if config_count[ci] == 7:
                    safe = False
                    break
            
            if safe:
                system[t_idx] = True
                for ci in triple_to_configs[t_idx]:
                    config_count[ci] += 1
                count += 1
        
        return system, config_count, count
    
    best_system = None
    best_count = 0
    
    # Run multiple greedy passes with local search
    while time.time() < deadline - 5:
        random_order = list(range(num_triples))
        random.shuffle(random_order)
        
        system, count, config_count = construct_greedy(random_order)
        
        # Apply local search
        system, config_count, count = local_search(system, config_count)
        
        if count > best_count:
            best_count = count
            best_system = system
    
    if best_system is None:
        # Fallback: simple construction
        result = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    result.append([i, j, k])
        return result
    
    return [list(all_triples[i]) for i in range(num_triples) if best_system[i]]
