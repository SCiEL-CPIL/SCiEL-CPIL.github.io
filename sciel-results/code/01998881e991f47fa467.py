import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    triple_index = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute all K(2,2,2) configurations (sets of 8 transversal triples)
    # For each 6-point set and each of the 15 pairings, get the 8 transversal triples
    k222_configs = []
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
        for pairs in pairings:
            (a,b),(c,d),(e,f) = pairs
            transversal = [
                tuple(sorted((a,c,e))), tuple(sorted((a,c,f))),
                tuple(sorted((a,d,e))), tuple(sorted((a,d,f))),
                tuple(sorted((b,c,e))), tuple(sorted((b,c,f))),
                tuple(sorted((b,d,e))), tuple(sorted((b,d,f))),
            ]
            k222_configs.append(frozenset(transversal))
    
    num_configs = len(k222_configs)
    
    # For each triple, precompute which K(2,2,2) configs it participates in
    # And for each config, which triples are in it
    config_triples = [set(cfg) for cfg in k222_configs]
    triple_configs = [set() for _ in range(num_triples)]
    for ci, cfg in enumerate(k222_configs):
        for t in cfg:
            ti = triple_index[t]
            triple_configs[ti].add(ci)
    
    def can_add(triple_set, t):
        """Check if adding triple t to triple_set would create a K(2,2,2)."""
        ti = triple_index[t]
        for ci in triple_configs[ti]:
            cfg = config_triples[ci]
            # Check if all other 7 triples in this config are already in triple_set
            all_present = True
            for other in cfg:
                if other == t:
                    continue
                if other not in triple_set:
                    all_present = False
                    break
            if all_present:
                return False
        return True
    
    def is_valid_full(triple_set):
        """Full validity check - O(num_configs * 8)."""
        for cfg in k222_configs:
            if cfg.issubset(triple_set):
                return False
        return True
    
    # Greedy construction with fast checking
    def greedy_random():
        order = all_triples[:]
        random.shuffle(order)
        triples_set = set()
        for t in order:
            if can_add(triples_set, t):
                triples_set.add(t)
        return triples_set
    
    # Local search with fast checking
    def local_search(triples_set, max_steps=10000):
        current = set(triples_set)
        current_list = list(current)
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Try adding a random triple not in current
            t = random.choice(all_triples)
            if t not in current:
                if can_add(current, t):
                    current.add(t)
                    continue
            
            # Try swapping: remove one, add another
            if current:
                idx = random.randrange(len(current_list))
                remove_t = current_list[idx]
                current.discard(remove_t)
                
                found = False
                for _ in range(30):
                    add_t = random.choice(all_triples)
                    if add_t != remove_t and add_t not in current:
                        if can_add(current, add_t):
                            current.add(add_t)
                            found = True
                            break
                if not found:
                    current.add(remove_t)
        
        return current
    
    # Run multiple greedy + local search rounds
    best_triples = set()
    best_size = 0
    best_greedy = set() # Guaranteed valid fallback
    
    random.seed(42)
    
    def is_valid_set(s):
        return is_valid_full(s)

    # Run many greedy + local search iterations
    while time.time() < deadline:
        g_result = greedy_random()
        
        # g_result is guaranteed valid by construction
        if len(g_result) > len(best_greedy):
            best_greedy = g_result
            
        if len(g_result) > best_size:
            best_triples = g_result
            best_size = len(g_result)
        
        # Local search
        ls_result = local_search(g_result, max_steps=2000)
        
        # Verify local search result
        if is_valid_set(ls_result):
            if len(ls_result) > best_size:
                best_triples = ls_result
                best_size = len(ls_result)
        else:
            # If local search produced invalid result, fall back to greedy for this iteration
            # (This shouldn't happen if can_add is correct, but acts as a safety net)
            pass
        
        if time.time() > deadline:
            break
    
    # Final local search on best
    if time.time() < deadline - 5:
        final = local_search(best_triples, max_steps=5000)
        if is_valid_set(final):
            if len(final) > best_size:
                best_triples = final
                best_size = len(final)
    
    # If best_triples is empty or invalid (shouldn't happen), use best_greedy
    if not best_triples or not is_valid_set(best_triples):
        best_triples = best_greedy
    
    # Final check
    if not is_valid_set(best_triples):
        # Last resort: return empty or a very small valid set
        # An empty set is valid.
        # Or we can try to build a small one
        return []

    # Convert to list of lists
    return [list(t) for t in sorted(best_triples)]
# EVOLVE-BLOCK-END
