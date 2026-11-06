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
        # There are 15 matchings for 6 points
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
    # We maintain a list of absent triples
    absent = [i for i in range(m) if not present[i]]
    
    # To speed up, we can prioritize triples that are "close" to being added
    # But simple random restarts of local search might be better.
    
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
            # Try to find a swap
            # We need to remove a triple u such that u is in ALL new_violations
            # and removing u doesn't cause issues (it never does)
            # But we also want u to be "cheap" to remove (low degree?)
            
            # Find intersection of all new_violations configs
            # If intersection is empty, we can't fix it with a single swap.
            # We could try a 2-swap, but that's complex.
            # Let's stick to 1-swap for robustness.
            
            if len(new_violations) == 1:
                ci = new_violations[0]
                # Find a present triple in ci
                candidates = [ti for ti in config_to_triples[ci] if present[ti] and ti != t]
                if candidates:
                    # Pick one randomly
                    u = random.choice(candidates)
                    # Perform swap
                    remove_triple(u)
                    add_triple(t)
                    # Update absent list
                    absent.remove(t)
                    absent.append(u)
            # If len > 1, we skip for now. This might limit performance but ensures validity.
            
    # Final Repair Loop - CRITICAL
    # Ensure no violated configs remain
    # This loop is simple and guaranteed to terminate if we don't hit deadline
    # If we hit deadline, we return the best valid set found so far.
    # But we must return a VALID set.
    # So we must ensure the repair finishes.
    # To ensure it finishes, we limit the number of steps or use a more efficient repair.
    # Since removing a triple is O(deg), and deg is small, this is fast.
    
    # We use a strict deadline for the repair phase to ensure we return something.
    repair_deadline = time.time() + 5.0
    
    while violated_configs and time.time() < repair_deadline:
        ci = random.choice(list(violated_configs))
        # Remove a random present triple from this config
        candidates = [ti for ti in config_to_triples[ci] if present[ti]]
        if not candidates:
            # Should not happen if state is consistent, but safeguard
            violated_configs.discard(ci)
            continue
        ti = random.choice(candidates)
        remove_triple(ti)
        
    # If we still have violated configs (due to time), we must remove them all.
    # This might reduce the score, but ensures validity.
    while violated_configs:
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
