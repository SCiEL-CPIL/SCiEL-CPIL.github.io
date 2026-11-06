import time
import random
from itertools import combinations

def _precompute():
    """Precompute all triples and K(2,2,2) configurations."""
    n = 16
    triples = []
    triple_index = {}
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                triples.append((a, b, c))
                triple_index[(a, b, c)] = len(triples) - 1
    m = len(triples)  # 560

    # Find all K(2,2,2) configurations
    configs = []
    config_to_triples = []
    
    all_points = list(range(n))
    for six in combinations(all_points, 6):
        s = list(six)
        # Generate all perfect matchings of the 6 points
        matchings = []
        def gen_matchings(pts, used, current):
            if not pts:
                matchings.append(current[:])
                return
            first = None
            for i in range(len(pts)):
                if i not in used:
                    first = i
                    break
            if first is None:
                return
            used.add(first)
            for j in range(first + 1, len(pts)):
                if j not in used:
                    used.add(j)
                    current.append((pts[first], pts[j]))
                    gen_matchings(pts, used, current)
                    current.pop()
                    used.remove(j)
            used.remove(first)

        gen_matchings(s, set(), [])
        for matching in matchings:
            (a, a2), (b, b2), (c, c2) = matching
            trans = [
                (a, b, c), (a, b, c2), (a2, b, c), (a2, b, c2),
                (a, b2, c), (a, b2, c2), (a2, b2, c), (a2, b2, c2)
            ]
            indices = []
            valid = True
            for t in trans:
                if t in triple_index:
                    indices.append(triple_index[t])
                else:
                    valid = False
                    break
            if valid:
                configs.append(tuple(indices))
                config_to_triples.append(indices)

    K = len(configs)
    
    # Map: Triple index -> List of Config indices
    triple_to_configs = [[] for _ in range(m)]
    for ci in range(K):
        for ti in config_to_triples[ci]:
            triple_to_configs[ti].append(ci)

    return triples, configs, config_to_triples, triple_to_configs, m, K

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    triples, configs, config_to_triples, triple_to_configs, m, K = _precompute()
    
    def check_deadline():
        return time.time() > deadline

    # State
    present = [False] * m
    config_weights = [0] * K
    violated_configs = set()
    
    def add_triple(t_idx):
        present[t_idx] = True
        for ci in triple_to_configs[t_idx]:
            config_weights[ci] += 1
            if config_weights[ci] == 8:
                violated_configs.add(ci)
                
    def remove_triple(t_idx):
        present[t_idx] = False
        for ci in triple_to_configs[t_idx]:
            config_weights[ci] -= 1
            if config_weights[ci] == 7:
                if ci in violated_configs:
                    violated_configs.discard(ci)

    # Initialization: Greedy random
    random.seed(42)
    order = list(range(m))
    random.shuffle(order)
    
    for i in order:
        if check_deadline():
            break
        # Check if adding i is safe
        safe = True
        for ci in triple_to_configs[i]:
            if config_weights[ci] == 7:
                safe = False
                break
        if safe:
            add_triple(i)

    # Local Search
    absent = [i for i in range(m) if not present[i]]
    
    while not check_deadline():
        if not absent:
            break
        
        # Pick a random absent triple
        t = random.choice(absent)
        
        # Calculate violations caused by adding t
        new_violations = []
        for ci in triple_to_configs[t]:
            if config_weights[ci] == 7:
                new_violations.append(ci)
                
        if not new_violations:
            add_triple(t)
            absent.remove(t)
        else:
            if len(new_violations) == 1:
                ci = new_violations[0]
                candidates = [ti for ti in config_to_triples[ci] if present[ti] and ti != t]
                if candidates:
                    u = random.choice(candidates)
                    remove_triple(u)
                    add_triple(t)
                    absent.remove(t)
                    absent.append(u)
            # If len > 1, skip

    # Final Repair Loop
    # Bounded repair
    repair_deadline = time.time() + 4.0
    
    while violated_configs and time.time() < repair_deadline:
        ci = random.choice(list(violated_configs))
        candidates = [ti for ti in config_to_triples[ci] if present[ti]]
        if not candidates:
            violated_configs.discard(ci)
            continue
        ti = random.choice(candidates)
        remove_triple(ti)
        
    # Panic Mode: If violations remain, remove all triples in violated configs
    # This is fast and guarantees validity
    if violated_configs:
        to_remove = set()
        for ci in violated_configs:
            for ti in config_to_triples[ci]:
                if present[ti]:
                    to_remove.add(ti)
        
        for ti in to_remove:
            remove_triple(ti)
            
    # Build result
    result = [list(triples[i]) for i in range(m) if present[i]]
    return result
