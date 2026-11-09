import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    triple_index = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute: for each pair of points, which triples contain both
    pair_to_triples = {}
    for t in all_triples:
        a, b, c = t
        pair_to_triples[(a, b)] = pair_to_triples.get((a, b), []) + [t]
        pair_to_triples[(b, a)] = pair_to_triples.get((b, a), []) + [t]
    
    # For each set of 6 points, precompute the 10 ways to partition into 3 pairs
    # and the 8 transversal triples for each such partition
    six_points_configs = {}
    for six in itertools.combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = six
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
        configs = []
        for pairs in pairings:
            # For 3 disjoint pairs (a,b),(c,d),(e,f), the 8 transversal triples are:
            # (a,c,e), (a,c,f), (a,d,e), (a,d,f), (b,c,e), (b,c,f), (b,d,e), (b,d,f)
            (a,b),(c,d),(e,f) = pairs
            transversal = [
                tuple(sorted((a,c,e))), tuple(sorted((a,c,f))),
                tuple(sorted((a,d,e))), tuple(sorted((a,d,f))),
                tuple(sorted((b,c,e))), tuple(sorted((b,c,f))),
                tuple(sorted((b,d,e))), tuple(sorted((b,d,f))),
            ]
            configs.append(transversal)
        six_points_configs[six] = configs
    
    def is_valid(triples_set):
        """Check if no K(2,2,2) is present."""
        for six, configs in six_points_configs.items():
            for transversal in configs:
                count = sum(1 for t in transversal if t in triples_set)
                if count == 8:
                    return False
        return True
    
    def count_violations(triples_set):
        """Count number of K(2,2,2) configurations present."""
        count = 0
        for six, configs in six_points_configs.items():
            for transversal in configs:
                c = sum(1 for t in transversal if t in triples_set)
                if c == 8:
                    count += 1
        return count
    
    # Greedy construction: add triples in random order, skip if it creates a violation
    def greedy_random():
        random.shuffle(all_triples)
        triples_set = set()
        for t in all_triples:
            new_set = triples_set | {t}
            if is_valid(new_set):
                triples_set.add(t)
        return triples_set
    
    # Local search: try to add a random triple, or swap one out for another
    def local_search(triples_set, max_steps=5000):
        current = set(triples_set)
        current_size = len(current)
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Try adding a random triple
            t = random.choice(all_triples)
            if t not in current:
                new_set = current | {t}
                if is_valid(new_set):
                    current = new_set
                    current_size += 1
                    continue
            
            # Try swapping: remove one, add another
            if current:
                remove_t = random.choice(list(current))
                current.discard(remove_t)
                
                # Try to add a different triple
                for _ in range(20):
                    add_t = random.choice(all_triples)
                    if add_t != remove_t:
                        new_set = current | {add_t}
                        if is_valid(new_set):
                            current = new_set
                            break
                else:
                    # Couldn't find a good swap, put it back
                    current.add(remove_t)
        
        return current
    
    # Run greedy with multiple random seeds
    best_triples = set()
    best_size = 0
    
    # First, a deterministic greedy to get a baseline
    deterministic_order = sorted(all_triples, key=lambda t: (t[0], t[1], t[2]))
    for t in deterministic_order:
        new_set = best_triples | {t}
        if is_valid(new_set):
            best_triples.add(t)
    best_size = len(best_triples)
    
    # Now try randomized greedy + local search
    random.seed(42)
    
    while time.time() < deadline:
        # Greedy random
        g_result = greedy_random()
        
        # Local search
        ls_result = local_search(g_result, max_steps=3000)
        
        if len(ls_result) > best_size:
            best_triples = ls_result
            best_size = len(ls_result)
        
        # Check time
        if time.time() > deadline:
            break
    
    # Final local search on best
    final = local_search(best_triples, max_steps=5000)
    if len(final) > best_size:
        best_triples = final
        best_size = len(final)
    
    # Convert to list of lists
    return [list(t) for t in sorted(best_triples)]
# EVOLVE-BLOCK-END
