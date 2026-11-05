import time
import random
import itertools

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

    def exhaustive_add(present, k222_count, count):
        # Repeatedly scan and add until no more can be added
        improved = True
        while improved:
            improved = False
            for ti in range(num_total):
                if not present[ti] and can_add(ti, k222_count):
                    do_add(ti, present, k222_count)
                    count += 1
                    improved = True
        return count

    def find_targeted_swap(present, k222_count):
        # Find critical K222s
        critical_indices = [i for i, c in enumerate(k222_count) if c == 7]
        if not critical_indices:
            return -1, -1
            
        # For each non-present triple, check if we can unblock it
        # We iterate through non-present triples
        # To avoid bias, we can randomize the order, but systematic is fine too.
        # Let's try systematic first.
        
        for t_add in range(num_total):
            if present[t_add]:
                continue
                
            # Find blocking critical K222s
            blocking_crits = []
            for k_idx in critical_indices:
                if t_add in k222_configs[k_idx]:
                    blocking_crits.append(k_idx)
                    
            if not blocking_crits:
                # Should be addable, but maybe we missed it?
                # If so, it's a "free" add.
                # But we are in swap phase.
                # Let's just return -1, -1 and let exhaustive_add handle it?
                # No, if exhaustive_add didn't catch it, something is wrong.
                # But logically, if no critical K222 blocks it, it's addable.
                # So this case implies exhaustive_add failed.
                # We can just add it? But the function signature is for swap.
                # Let's assume this doesn't happen.
                continue
                
            # Intersection of blocking_crits
            # Start with the set of triples in the first one
            # Note: k222_configs[k_idx] is a frozenset
            intersection = set(k222_configs[blocking_crits[0]])
            
            for b_idx in blocking_crits[1:]:
                intersection &= set(k222_configs[b_idx])
                if not intersection:
                    break
                    
            if not intersection:
                continue
                
            # Check if any triple in intersection is present
            for t_rem in intersection:
                if present[t_rem]:
                    return t_add, t_rem
                    
        return -1, -1

    def local_search(present, k222_count, count, rng):
        # Phase 1: Exhaustive Add
        count = exhaustive_add(present, k222_count, count)
        
        # Phase 2: Targeted Swaps
        swaps_done = 0
        max_swaps = 1000 # Safety cap per local search call
        
        while swaps_done < max_swaps:
            if time_check():
                break
                
            t_add, t_rem = find_targeted_swap(present, k222_count)
            
            if t_add == -1:
                # No targeted swap found. Try a random swap?
                # Random swap: remove random present, add random non-present if possible.
                # This is less efficient but helps escape.
                
                # Try a few random swaps
                random_swap_found = False
                for _ in range(20):
                    # Pick random present
                    present_indices = [i for i in range(num_total) if present[i]]
                    if not present_indices:
                        break
                    t_rem_rand = rng.choice(present_indices)
                    
                    # Remove it
                    do_remove(t_rem_rand, present, k222_count)
                    count -= 1
                    
                    # Try to add something
                    # We can try to add any non-present triple
                    # Or just try to add t_rem_rand back? No.
                    # Let's try to add a random non-present triple
                    non_present_indices = [i for i in range(num_total) if not present[i]]
                    if not non_present_indices:
                        break
                        
                    # Shuffle to pick random
                    rng.shuffle(non_present_indices)
                    
                    added = False
                    for t_candidate in non_present_indices:
                        if can_add(t_candidate, k222_count):
                            do_add(t_candidate, present, k222_count)
                            count += 1
                            added = True
                            break
                            
                    if added:
                        # Successful random swap
                        swaps_done += 1
                        random_swap_found = True
                        # Go back to exhaustive add
                        count = exhaustive_add(present, k222_count, count)
                        break
                    else:
                        # Revert
                        do_add(t_rem_rand, present, k222_count)
                        count += 1
                        
                if not random_swap_found:
                    break # Stuck
                
            else:
                # Perform targeted swap
                do_remove(t_rem, present, k222_count)
                count -= 1
                do_add(t_add, present, k222_count)
                count += 1
                swaps_done += 1
                
                # Go back to exhaustive add
                count = exhaustive_add(present, k222_count, count)
                
        return count

    best_result = []
    best_count = 0
    rng = random.Random(42)
    
    # Multiple restarts
    for restart in range(50): # Limit restarts
        if time_check():
            break
            
        present, k222_count, count = greedy_init(rng)
        count = local_search(present, k222_count, count, rng)
        
        if count > best_count:
            best_count = count
            best_result = [list(all_triples[i]) for i in range(num_total) if present[i]]
            
    if not best_result:
        best_result = [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
        
    return best_result
