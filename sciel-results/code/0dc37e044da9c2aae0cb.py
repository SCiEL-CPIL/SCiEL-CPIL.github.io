import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

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
        
        return system, system_count, config_count

    def local_search_fast(system, config_count, max_iters=500):
        """Fast local search: random swaps."""
        system = system[:]
        config_count = config_count[:]
        count = sum(1 for x in system if x)
        
        # Pre-compute list of selected and unselected indices for faster access?
        # No, just sample randomly from range.
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Pick a random unselected triple
            new_t = random.randint(0, num_triples - 1)
            if system[new_t]:
                continue
                
            # Check if new_t is safe
            blocking_configs = []
            for ci in triple_to_configs[new_t]:
                if config_count[ci] == 7:
                    blocking_configs.append(ci)
            
            if not blocking_configs:
                # Safe to add
                system[new_t] = True
                for ci in triple_to_configs[new_t]:
                    config_count[ci] += 1
                count += 1
                continue
            
            # Try to remove one of the selected triples that is in a blocking config
            # Pick a random blocking config
            ci = random.choice(blocking_configs)
            # Get all selected triples in this config
            # We don't have config_to_selected_triples easily, but we can scan?
            # Better: Pre-compute config_to_triples (static)
            # For now, let's just try a random selected triple.
            # This is inefficient. Let's stick to simple greedy restarts if local search is too complex/slow.
            # Given the time limit, Greedy with many restarts is safer.
            pass
            
        return system, config_count, count

    best_system = None
    best_count = 0
    best_config_count = None
    
    # Phase 1: Multiple greedy passes
    # Estimate time per greedy pass: ~5600 triples * ~10 configs check = 56000 ops. Very fast.
    # We can do many restarts.
    
    while time.time() < deadline - 10:
        random_order = list(range(num_triples))
        random.shuffle(random_order)
        
        system, count, config_count = construct_greedy(random_order)
        
        if count > best_count:
            best_count = count
            best_system = system
            best_config_count = config_count
        
        # If we have a good solution, try a few local search steps on it
        if count >= best_count and best_count > 0:
            # Simple local search: try to add any safe triple
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                # Try a random sample of unselected triples
                for _ in range(100):
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
                        improved = True
                        break # Restart sample loop
                    
            if count > best_count:
                best_count = count
                best_system = system
                best_config_count = config_count

    if best_system is None:
        return [[0, a, b] for a in range(1, n) for b in range(a + 1, n)]
    
    return [list(all_triples[i]) for i in range(num_triples) if best_system[i]]
