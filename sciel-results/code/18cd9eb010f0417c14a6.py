import time
import math
import random

N = 64

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Precompute coordinates
    xs = [i // N for i in range(N * N)]
    ys = [i % N for i in range(N * N)]
    
    all_points = list(range(N * N))
    
    def dist2(i, j):
        dx = xs[i] - xs[j]
        dy = ys[i] - ys[j]
        return dx * dx + dy * dy

    def run_local_search(initial_selected, initial_dist_sets, rng, time_limit):
        selected = set(initial_selected)
        dist_sets = {p: set(d) for p, d in initial_dist_sets.items()}
        unselected = [p for p in all_points if p not in selected]
        
        def can_add(p):
            if p in selected:
                return False
            seen_from_p = set()
            for q in selected:
                d = dist2(p, q)
                if d in seen_from_p:
                    return False
                if d in dist_sets[q]:
                    return False
                seen_from_p.add(d)
            return True

        def add(p):
            selected.add(p)
            dist_sets[p] = set()
            for q in selected:
                if q == p:
                    continue
                d = dist2(p, q)
                dist_sets[p].add(d)
                dist_sets[q].add(d)

        def remove(p):
            selected.discard(p)
            if p not in dist_sets:
                return
            for q in selected:
                if q == p:
                    continue
                d = dist2(p, q)
                if d in dist_sets[q]:
                    dist_sets[q].discard(d)
            del dist_sets[p]
            
        def try_swap(p_out, p_in):
            if p_in in selected or p_out not in selected:
                return False
            seen_from_pin = set()
            for r in selected:
                if r == p_out:
                    continue
                d = dist2(p_in, r)
                if d in seen_from_pin:
                    return False
                seen_from_pin.add(d)
                # Check if this distance already exists for r, 
                # unless it was contributed by p_out
                if d in dist_sets[r]:
                    # Check if p_out is the source of this distance in dist_sets[r]
                    # We don't store sources, so we must check if removing p_out removes this distance.
                    # dist_sets[r] contains distances from r to all q in selected.
                    # If dist(r, p_out) == d, then p_out is the source.
                    # If dist(r, p_out) != d, then some other q is the source, and conflict remains.
                    if dist2(r, p_out) != d:
                        return False
                    # If dist(r, p_out) == d, then after removing p_out, this distance is gone from dist_sets[r].
                    # So it's safe to add p_in with this distance.
            
            remove(p_out)
            add(p_in)
            return True

        def try_2_swap(p_out1, p_out2, p_in1, p_in2):
            if p_in1 in selected or p_in2 in selected:
                return False
            if p_out1 not in selected or p_out2 not in selected:
                return False
            if p_out1 == p_out2 or p_in1 == p_in2:
                return False

            # Simulate removal of p_out1, p_out2
            # We need to check validity of adding p_in1, p_in2 to S \ {p_out1, p_out2}
            
            # 1. Check p_in1 against S \ {p_out1, p_out2} and p_in2
            seen_from_pin1 = set()
            for r in selected:
                if r == p_out1 or r == p_out2:
                    continue
                d = dist2(p_in1, r)
                if d in seen_from_pin1:
                    return False
                seen_from_pin1.add(d)
                
                # Check conflict with dist_sets[r]
                if d in dist_sets[r]:
                    # Check if the conflict is solely due to p_out1 or p_out2
                    # If dist(r, p_out1) == d or dist(r, p_out2) == d, then removing them resolves it.
                    # If neither, conflict remains.
                    if dist2(r, p_out1) != d and dist2(r, p_out2) != d:
                        return False
                    # If one of them is the source, it's resolved.
                    # Note: It's possible both are sources? No, distances are unique in valid set.
                    # So at most one source.
                    
            # Check p_in1 against p_in2
            d_12 = dist2(p_in1, p_in2)
            if d_12 in seen_from_pin1:
                return False
            seen_from_pin1.add(d_12)
            
            # 2. Check p_in2 against S \ {p_out1, p_out2} and p_in1
            seen_from_pin2 = set()
            for r in selected:
                if r == p_out1 or r == p_out2:
                    continue
                d = dist2(p_in2, r)
                if d in seen_from_pin2:
                    return False
                seen_from_pin2.add(d)
                
                if d in dist_sets[r]:
                    if dist2(r, p_out1) != d and dist2(r, p_out2) != d:
                        return False
            
            # Check p_in2 against p_in1 (symmetric, already checked d_12 for pin1, need to check for pin2's uniqueness)
            # d_12 is in seen_from_pin2? No, we just added it to seen_from_pin1.
            # We need to ensure d_12 is not in seen_from_pin2 (distances from pin2 to others).
            # We haven't added d_12 to seen_from_pin2 yet.
            if d_12 in seen_from_pin2:
                return False
            seen_from_pin2.add(d_12)

            # Perform swap
            remove(p_out1)
            remove(p_out2)
            add(p_in1)
            add(p_in2)
            return True

        best_size = len(selected)
        
        while time.time() < time_limit:
            # Try to add
            added = False
            if unselected:
                # Shuffle unselected to randomize order
                random_idx = rng.randrange(len(unselected))
                p = unselected[random_idx]
                if can_add(p):
                    add(p)
                    unselected.pop(random_idx)
                    added = True
                    best_size = max(best_size, len(selected))
            
            if not added:
                # Try swaps
                if len(selected) >= 2 and len(unselected) >= 2:
                    # 2-swap
                    p_out1 = rng.choice(list(selected))
                    p_out2 = rng.choice([p for p in selected if p != p_out1])
                    p_in1 = rng.choice(unselected)
                    p_in2 = rng.choice([p for p in unselected if p != p_in1])
                    
                    if try_2_swap(p_out1, p_out2, p_in1, p_in2):
                        unselected.remove(p_in1)
                        unselected.remove(p_in2)
                        unselected.append(p_out1)
                        unselected.append(p_out2)
                elif len(selected) >= 1 and len(unselected) >= 1:
                    # 1-swap
                    p_out = rng.choice(list(selected))
                    p_in = rng.choice(unselected)
                    if try_swap(p_out, p_in):
                        unselected.remove(p_in)
                        unselected.append(p_out)

        return selected

    def greedy_construct(rng):
        selected = set()
        dist_sets = {}
        
        def can_add(p):
            if p in selected:
                return False
            seen_from_p = set()
            for q in selected:
                d = dist2(p, q)
                if d in seen_from_p:
                    return False
                if d in dist_sets[q]:
                    return False
                seen_from_p.add(d)
            return True

        def add(p):
            selected.add(p)
            dist_sets[p] = set()
            for q in selected:
                if q == p:
                    continue
                d = dist2(p, q)
                dist_sets[p].add(d)
                dist_sets[q].add(d)

        shuffled_points = all_points.copy()
        rng.shuffle(shuffled_points)
        
        for p in shuffled_points:
            if time.time() > deadline - 10:
                break
            if can_add(p):
                add(p)
        
        return selected, dist_sets

    best_selected = set()
    
    # Multi-start
    num_starts = 10
    time_per_start = (deadline - time.time()) / num_starts
    
    for i in range(num_starts):
        if time.time() > deadline - 5:
            break
            
        rng = random.Random(42 + i)
        selected, dist_sets = greedy_construct(rng)
        
        if len(selected) > len(best_selected):
            best_selected = set(selected)
            
        # Run local search
        time_limit = time.time() + time_per_start
        # Pass copies to local search
        local_selected = run_local_search(selected, dist_sets, rng, time_limit)
        
        if len(local_selected) > len(best_selected):
            best_selected = local_selected

    result = [(i // N, i % N) for i in best_selected]
    return result

# EVOLVE-BLOCK-END
