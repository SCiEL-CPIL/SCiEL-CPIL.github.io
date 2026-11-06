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
    deadline = start_time + 145 # 5s safety margin
    
    triples, configs, config_to_triples, triple_to_configs, m, K = _precompute()
    
    def elapsed():
        return time.time() - start_time
        
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

    # Initialize: Random 40% density
    random.seed(42)
    for i in range(m):
        if random.random() < 0.4:
            add_triple(i)
            
    # Phase 1: Repair to Validity
    while violated_configs:
        if check_deadline():
            break
        # Pick a random violated config
        ci = random.choice(list(violated_configs))
        # Pick a random present triple in this config
        # Find present triples in config_to_triples[ci]
        candidates = [ti for ti in config_to_triples[ci] if present[ti]]
        if not candidates:
            # Should not happen if logic is correct, but discard if stuck
            violated_configs.discard(ci)
            continue
        ti = random.choice(candidates)
        remove_triple(ti)
        
    # Phase 2: Local Search Optimization
    # Goal: Maximize sum(present)
    # Moves:
    # 1. Add absent triple if it doesn't create violations.
    # 2. Swap: Add absent triple t, remove present triple t' if t creates exactly 1 violation and t' is in that violation.
    
    absent_triples = [i for i in range(m) if not present[i]]
    
    while not check_deadline():
        # Shuffle absent list occasionally to avoid bias
        if random.random() < 0.01:
            random.shuffle(absent_triples)
            
        if not absent_triples:
            break
            
        t = random.choice(absent_triples)
        
        # Check if adding t is safe
        # Calculate which configs would become violated
        new_violations = []
        for ci in triple_to_configs[t]:
            if config_weights[ci] == 7:
                new_violations.append(ci)
                
        if not new_violations:
            # Safe add
            add_triple(t)
            absent_triples.remove(t)
        elif len(new_violations) == 1:
            # Try Swap
            ci = new_violations[0]
            # Find a present triple t' in ci, t' != t
            # We want to remove t' to fix ci.
            # Prefer t' that is in few other configs? Or just random?
            # Let's pick one that minimizes "loss" (i.e., is in few other configs)
            best_t_prime = -1
            min_degree = float('inf')
            for ti in config_to_triples[ci]:
                if ti == t:
                    continue
                if not present[ti]:
                    continue
                deg = len(triple_to_configs[ti])
                if deg < min_degree:
                    min_degree = deg
                    best_t_prime = ti
                    
            if best_t_prime != -1:
                # Perform Swap
                remove_triple(best_t_prime)
                add_triple(t)
                # Update absent list
                if t in absent_triples:
                    absent_triples.remove(t)
                absent_triples.append(best_t_prime)
            # If no best_t_prime, do nothing (reject)
        else:
            # Too many violations, reject
            pass
            
    # Final Safety Check and Repair
    # This ensures validity even if the local search logic had a bug or time ran out during a complex update
    while violated_configs:
        if time.time() > deadline - 2:
            # Force repair to finish quickly
            pass
        ci = random.choice(list(violated_configs))
        candidates = [ti for ti in config_to_triples[ci] if present[ti]]
        if not candidates:
            violated_configs.discard(ci)
            continue
        ti = random.choice(candidates)
        remove_triple(ti)
        
    # Build result
    result = [list(triples[i]) for i in range(m) if present[i]]
    return result
