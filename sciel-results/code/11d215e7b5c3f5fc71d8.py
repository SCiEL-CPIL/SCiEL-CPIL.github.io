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

    def exhaustive_add(present, k222_count, count):
        improved = True
        while improved:
            improved = False
            for ti in range(num_total):
                if not present[ti] and can_add(ti, k222_count):
                    do_add(ti, present, k222_count)
                    count += 1
                    improved = True
        return count

    def local_search(present, k222_count, count, rng):
        # Phase 1: Exhaustive Add
        count = exhaustive_add(present, k222_count, count)
        
        # Phase 2: WalkSAT-style stochastic local search
        # Maintain a list of present indices for fast random sampling
        present_indices = [i for i in range(num_total) if present[i]]
        
        max_steps = 5000
        no_improve_limit = 100
        
        for step in range(max_steps):
            if time_check():
                break
            
            # 1. Try to add any missing triple (climb phase)
            # Instead of scanning all 560, we can pick a random subset of missing ones
            # Or just scan a few random missing ones
            # Let's try a few random missing ones first
            added_any = False
            for _ in range(10):
                if time_check():
                    break
                # Pick a random index
                ti = rng.randint(0, num_total - 1)
                if not present[ti] and can_add(ti, k222_count):
                    do_add(ti, present, k222_count)
                    count += 1
                    present_indices.append(ti)
                    added_any = True
                    break
            
            if added_any:
                continue
            
            # 2. If no add, try a swap
            # Pick a random missing triple
            # To ensure we pick a missing one, we might need a list of missing ones
            # But maintaining a list of missing is expensive.
            # Let's just pick random indices until we find a missing one
            t_add = -1
            for _ in range(20):
                ti = rng.randint(0, num_total - 1)
                if not present[ti]:
                    t_add = ti
                    break
            
            if t_add == -1:
                # All present? Unlikely. Or we are stuck.
                break
                
            # Find blocking critical K222s for t_add
            blocking_crits = []
            for ci in triple_to_k222[t_add]:
                if k222_count[ci] == 7:
                    blocking_crits.append(ci)
                    
            if not blocking_crits:
                # Should have been added. Maybe we missed it?
                # If can_add was false, there must be a blocker.
                # If can_add was true, we should have added it.
                # So this case implies can_add was false but no count==7?
                # That's impossible if can_add checks count==7.
                # Unless there's a bug.
                # Let's just try to add it again?
                if can_add(t_add, k222_count):
                    do_add(t_add, present, k222_count)
                    count += 1
                    present_indices.append(t_add)
                    continue
                else:
                    # Still blocked? Weird.
                    continue

            # Pick a random blocker
            blocker_ci = rng.choice(blocking_crits)
            
            # Pick a random present triple in this blocker to remove
            # We need to find present triples in k222_config_lists[blocker_ci]
            # This is O(8)
            candidates = []
            for ti in k222_config_lists[blocker_ci]:
                if present[ti]:
                    candidates.append(ti)
                    
            if not candidates:
                continue
                
            t_rem = rng.choice(candidates)
            
            # Perform swap
            do_remove(t_rem, present, k222_count)
            count -= 1
            present_indices.remove(t_rem) # O(N), but N is smallish
            
            # Try to add t_add
            if can_add(t_add, k222_count):
                do_add(t_add, present, k222_count)
                count += 1
                present_indices.append(t_add)
            else:
                # Failed swap, revert
                do_add(t_rem, present, k222_count)
                count += 1
                present_indices.append(t_rem)
                
        # Final exhaustive add
        count = exhaustive_add(present, k222_count, count)
        
        return count

    best_result = []
    best_count = 0
    rng = random.Random(42)
    
    # Multiple restarts
    # Estimate time per restart.
    # Greedy: 560 * (check) ~ fast
    # Local search: 5000 steps * (random pick + check) ~ fast
    # Let's try to run as many as possible.
    
    restart = 0
    while time_check() == False:
        # Allow 1 second for a restart roughly?
        # If we run out of time, break.
        
        present, k222_count, count = greedy_init(rng)
        count = local_search(present, k222_count, count, rng)
        
        if count > best_count:
            best_count = count
            best_result = [list(all_triples[i]) for i in range(num_total) if present[i]]
            
        restart += 1
        # Safety break if we've done too many and are close to deadline
        if time.time() > deadline - 2:
            break

    if not best_result:
        best_result = [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
        
    return best_result
