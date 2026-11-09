import time
import random

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 90s limit
    
    def recalc_reachable(S):
        """Compute reachable sums for set S, return None if 200 is reachable."""
        reachable = {0}
        for x in S:
            new_sums = {s + x for s in reachable if s + x <= 200}
            reachable.update(new_sums)
            if 200 in reachable:
                return None
        return reachable
    
    def greedy_init(order):
        """Greedy construction given an ordering of candidates."""
        selected = set()
        reachable = {0}
        for x in order:
            if (200 - x) not in reachable:
                selected.add(x)
                new_sums = {s + x for s in reachable if s + x <= 200}
                reachable.update(new_sums)
        return selected, reachable
    
    # Try multiple greedy orderings and keep the best
    best_S = set()
    best_R = None
    best_size = 0
    
    orderings = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
        list(range(50, 0, -1)) + list(range(100, 50, -1)),
        list(range(100, 49, -1)) + list(range(49, 0, -1)),
        [x for x in range(1, 101) if x % 2 == 0] + [x for x in range(1, 101) if x % 2 == 1],
        [x for x in range(1, 101) if x % 2 == 1] + [x for x in range(1, 101) if x % 2 == 0],
        # Additional diverse orderings
        [x for x in range(1, 101) if x <= 10] + list(range(100, 10, -1)),
        list(range(100, 90, -1)) + [x for x in range(90, 0, -1) if x % 3 != 0],
        sorted(range(1, 101), key=lambda x: abs(x - 50)),
        [x for x in range(1, 101) if x % 5 in [0, 1]] + [x for x in range(1, 101) if x % 5 not in [0, 1]],
    ]
    
    for order in orderings:
        S, R = greedy_init(order)
        if len(S) > best_size:
            best_S = S
            best_R = R
            best_size = len(S)
    
    S = best_S
    R = best_R
    missing = set(range(1, 101)) - S
    
    # Local search with multiple strategies
    no_progress_count = 0
    max_no_progress = 100
    
    while time.time() < deadline:
        # Phase 1: Try to add elements (sorted by how "safe" they are)
        added_any = False
        for x in sorted(missing, key=lambda v: (200 - v) in R, reverse=False):
            if time.time() > deadline:
                break
            if (200 - x) not in R:
                S.add(x)
                missing.discard(x)
                new_sums = {s + x for s in R if s + x <= 200}
                R.update(new_sums)
                added_any = True
                break
        
        if added_any:
            no_progress_count = 0
            continue
        
        # Phase 2: Try single removals to free up space for additions
        removed_any = False
        for y in list(S):
            if time.time() > deadline:
                break
            S_test = S - {y}
            R_test = recalc_reachable(S_test)
            if R_test is not None:
                # Check if removing y allows adding something
                can_add = False
                best_add = None
                for x in missing:
                    if (200 - x) not in R_test:
                        can_add = True
                        if best_add is None or x > best_add:
                            best_add = x
                if can_add and best_add is not None:
                    # Remove y, add best_add
                    S = S_test
                    R = R_test
                    S.add(best_add)
                    missing.discard(best_add)
                    new_sums = {s + best_add for s in R if s + best_add <= 200}
                    R.update(new_sums)
                    removed_any = True
                    no_progress_count = 0
                    break
        
        if removed_any:
            continue
        
        # Phase 3: Try swaps (remove y, add x) where net size stays same or we can then add more
        swap_done = False
        candidates = list(missing)
        if candidates and S:
            random.shuffle(candidates)
            for x in candidates[:min(len(candidates), 20)]:
                if time.time() > deadline:
                    break
                for y in list(S):
                    if time.time() > deadline:
                        break
                    S_test = S - {y}
                    S_test.add(x)
                    R_test = recalc_reachable(S_test)
                    if R_test is not None:
                        # Check if this swap enables further additions
                        extra_adds = 0
                        temp_R = R_test
                        temp_missing = missing - {x}
                        temp_missing.add(y)
                        for z in sorted(temp_missing, reverse=True):
                            if (200 - z) not in temp_R:
                                extra_adds += 1
                                new_sums = {s + z for s in temp_R if s + z <= 200}
                                temp_R.update(new_sums)
                        
                        if extra_adds > 0:
                            # Accept the swap and additions
                            S = S_test
                            R = R_test
                            missing = set(range(1, 101)) - S
                            # Add the extra elements
                            for z in sorted(temp_missing, reverse=True):
                                if z in missing and (200 - z) not in R:
                                    S.add(z)
                                    missing.discard(z)
                                    new_sums = {s + z for s in R if s + z <= 200}
                                    R.update(new_sums)
                            swap_done = True
                            no_progress_count = 0
                            break
                if swap_done:
                    break
        
        if swap_done:
            continue
        
        # Phase 4: Random restarts if stuck
        no_progress_count += 1
        if no_progress_count >= max_no_progress:
            # Random perturbation: remove k random elements, then re-greedy
            k = random.randint(2, min(5, len(S)))
            if k > 0 and S:
                to_remove = random.sample(list(S), k)
                S = S - set(to_remove)
                R = recalc_reachable(S)
                if R is None:
                    # Shouldn't happen, but recover
                    R = {0}
                    for x in S:
                        new_sums = {s + x for s in R if s + x <= 200}
                        R.update(new_sums)
                missing = set(range(1, 101)) - S
                no_progress_count = 0
                # Re-greedy fill
                for x in sorted(missing, reverse=True):
                    if (200 - x) not in R:
                        S.add(x)
                        missing.discard(x)
                        new_sums = {s + x for s in R if s + x <= 200}
                        R.update(new_sums)
    
    # Final polish pass: try all single additions one more time
    for x in sorted(missing, reverse=True):
        if (200 - x) not in R:
            S.add(x)
            missing.discard(x)
            new_sums = {s + x for s in R if s + x <= 200}
            R.update(new_sums)
    
    return list(S)
