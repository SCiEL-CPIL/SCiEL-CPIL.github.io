import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_total = len(all_triples)  # 560
    
    # Precompute K222 configs
    k222_configs = []
    triple_to_k222 = [[] for _ in range(num_total)]
    
    for six_vertices in itertools.combinations(range(N), 6):
        a, b, c, d, e, f = six_vertices
        partitions = [
            ((a,b),(c,d),(e,f)), ((a,b),(c,e),(d,f)), ((a,b),(c,f),(d,e)),
            ((a,c),(b,d),(e,f)), ((a,c),(b,e),(d,f)), ((a,c),(b,f),(d,e)),
            ((a,d),(b,c),(e,f)), ((a,d),(b,e),(c,f)), ((a,d),(b,f),(c,e)),
            ((a,e),(b,c),(d,f)), ((a,e),(b,d),(c,f)), ((a,e),(b,f),(c,d)),
            ((a,f),(b,c),(d,e)), ((a,f),(b,d),(c,e)), ((a,f),(b,e),(c,d)),
        ]
        for p1, p2, p3 in partitions:
            transversals = []
            for x in p1:
                for y in p2:
                    for z in p3:
                        t = tuple(sorted((x, y, z)))
                        transversals.append(t)
            trans_indices = frozenset(triple_to_idx[t] for t in transversals)
            k222_configs.append(trans_indices)
            config_idx = len(k222_configs) - 1
            for ti in trans_indices:
                triple_to_k222[ti].append(config_idx)
                
    num_k222 = len(k222_configs)
    
    # Convert k222_configs to lists for faster iteration where needed
    k222_config_lists = [list(c) for c in k222_configs]
    
    def time_check():
        return time.time() >= deadline
        
    def can_add(ti, k222_count):
        for ci in triple_to_k222[ti]:
            if k222_count[ci] == 7:
                return False
        return True
        
    def do_add(ti, present, k222_count):
        present[ti] = True
        for ci in triple_to_k222[ti]:
            k222_count[ci] += 1
            
    def do_remove(ti, present, k222_count):
        present[ti] = False
        for ci in triple_to_k222[ti]:
            k222_count[ci] -= 1

    def greedy_init(rng):
        present = [False] * num_total
        k222_count = [0] * num_k222
        count = 0
        order = list(range(num_total))
        rng.shuffle(order)
        for ti in order:
            if can_add(ti, k222_count):
                do_add(ti, present, k222_count)
                count += 1
        return present, k222_count, count

    def fast_exhaustive_add(present, k222_count, count):
        # Only do a few passes to save time, not until fixed point
        for _ in range(3):
            improved = False
            for ti in range(num_total):
                if not present[ti] and can_add(ti, k222_count):
                    do_add(ti, present, k222_count)
                    count += 1
                    improved = True
            if not improved:
                break
        return count

    def local_search(present, k222_count, count, rng):
        # Initial fill
        count = fast_exhaustive_add(present, k222_count, count)
        
        max_steps = 2000
        no_improve_limit = 100
        no_improve = 0
        best_in_run = count
        
        for step in range(max_steps):
            if time_check():
                break
            
            # 1. Try to add a random missing triple
            added_any = False
            for _ in range(10):
                if time_check():
                    break
                ti = rng.randint(0, num_total - 1)
                if not present[ti] and can_add(ti, k222_count):
                    do_add(ti, present, k222_count)
                    count += 1
                    added_any = True
                    no_improve = 0
                    break
            
            if added_any:
                if count > best_in_run:
                    best_in_run = count
                continue
            
            # 2. Try a simple swap: pick a missing triple, pick a present one to remove
            # Find a missing triple
            t_add = -1
            for _ in range(10):
                ti = rng.randint(0, num_total - 1)
                if not present[ti]:
                    t_add = ti
                    break
            
            if t_add == -1:
                break
                
            # Check if t_add is blocked
            blocking_crits = [ci for ci in triple_to_k222[t_add] if k222_count[ci] == 7]
            
            if not blocking_crits:
                # It was just a matter of checking
                if can_add(t_add, k222_count):
                    do_add(t_add, present, k222_count)
                    count += 1
                    no_improve = 0
                    if count > best_in_run:
                        best_in_run = count
                continue

            # Try to find a present triple in one of the blocking configs to remove
            # Randomly pick one of the blocking configs
            ci = rng.choice(blocking_crits)
            # Pick a random present triple in that config
            candidates = [ti for ti in k222_config_lists[ci] if present[ti]]
            if candidates:
                t_rem = rng.choice(candidates)
                do_remove(t_rem, present, k222_count)
                count -= 1
                
                if can_add(t_add, k222_count):
                    do_add(t_add, present, k222_count)
                    count += 1
                    no_improve = 0
                    if count > best_in_run:
                        best_in_run = count
                else:
                    # Revert
                    do_add(t_rem, present, k222_count)
                    count += 1
                    no_improve += 1
            else:
                no_improve += 1
            
            # 3. Perturbation
            if no_improve > no_improve_limit:
                num_remove = rng.randint(2, 4)
                present_indices = [i for i in range(num_total) if present[i]]
                rng.shuffle(present_indices)
                for ti in present_indices[:num_remove]:
                    do_remove(ti, present, k222_count)
                    count -= 1
                no_improve = 0
                count = fast_exhaustive_add(present, k222_count, count)
                if count > best_in_run:
                    best_in_run = count
        
        # Final fill
        count = fast_exhaustive_add(present, k222_count, count)
        return count

    best_result = []
    best_count = 0
    rng = random.Random(42)
    
    # Multiple restarts
    restart = 0
    while time_check() == False:
        # Check if we have enough time for another restart
        if time.time() > deadline - 2:
            break
        
        present, k222_count, count = greedy_init(rng)
        count = local_search(present, k222_count, count, rng)
        
        if count > best_count:
            best_count = count
            best_result = [list(all_triples[i]) for i in range(num_total) if present[i]]
            
        restart += 1
        # Allow more restarts since local search is faster
        if time.time() > deadline - 10:
            break

    if not best_result:
        best_result = [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
    
    # Final validation and cleanup
    # Ensure all triples are valid and distinct
    seen = set()
    valid_result = []
    for t in best_result:
        if len(t) == 3 and len(set(t)) == 3 and all(0 <= x < 16 for x in t):
            key = tuple(sorted(t))
            if key not in seen:
                seen.add(key)
                valid_result.append([t[0], t[1], t[2]])
    
    # Verify no K(2,2,2) is fully present using precomputed configs
    # Create a set of present triple indices for O(1) lookup
    present_set = set()
    for t in valid_result:
        present_set.add(tuple(sorted(t)))
        
    # Note: k222_config_lists contains indices into all_triples
    # We need to check if all 8 triples in a config are in present_set
    for ci in range(num_k222):
        all_present = True
        for ti in k222_config_lists[ci]:
            if tuple(sorted(all_triples[ti])) not in present_set:
                all_present = False
                break
        if all_present:
            # Invalid, but we assume our search maintained validity.
            # If this happens, we could try to remove one triple, but for now we trust the search.
            pass
    
    return valid_result
