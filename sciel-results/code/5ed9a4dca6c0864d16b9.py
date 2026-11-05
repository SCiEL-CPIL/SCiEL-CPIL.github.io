"""B2[2] set construction: greedy add + targeted swap search with deadline guard."""
import time
import random

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    N = 1000
    
    # Phase 1: Greedy ascending add
    chosen = []
    counts = {}
    
    for x in range(1, N + 1):
        ok = True
        for y in chosen:
            if y == x:
                continue
            s = x + y
            if counts.get(s, 0) > 1:
                ok = False
                break
        if ok:
            s = 2 * x
            if counts.get(s, 0) > 1:
                ok = False
        if ok:
            for y in chosen:
                if y != x:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
            s = 2 * x
            counts[s] = counts.get(s, 0) + 1
            chosen.append(x)
    
    best_set = set(chosen)
    best_cnt = counts
    
    def try_add(x, S, cnt):
        """Check if x can be added to set S with count dict cnt."""
        for y in S:
            if y == x:
                continue
            s = x + y
            if cnt.get(s, 0) > 1:
                return False
        s = 2 * x
        if cnt.get(s, 0) > 1:
            return False
        return True
    
    def remove_from_counts(x, S, cnt):
        """Remove x from counts, assuming x in S."""
        for y in S:
            if y == x:
                continue
            s = x + y
            cnt[s] -= 1
            if cnt[s] == 0:
                del cnt[s]
        s = 2 * x
        cnt[s] -= 1
        if cnt[s] == 0:
            del cnt[s]
    
    def add_to_counts(x, S, cnt):
        """Add x to counts, assuming x not in S."""
        for y in S:
            if y == x:
                continue
            s = x + y
            cnt[s] = cnt.get(s, 0) + 1
        s = 2 * x
        cnt[s] = cnt.get(s, 0) + 1
    
    # Phase 2: Iterative swap + add passes
    no_improve_passes = 0
    max_no_improve = 5
    max_swaps_per_pass = 2000
    max_candidates = 100
    
    while time.time() < deadline - 10.0 and no_improve_passes < max_no_improve:
        improved = False
        swaps_done = 0
        chosen_list = sorted(best_set)
        
        # Swap phase
        for y in chosen_list:
            if time.time() > deadline - 5.0 or swaps_done >= max_swaps_per_pass:
                break
            
            remove_from_counts(y, best_set, best_cnt)
            best_set.discard(y)
            
            found = False
            candidates = [x for x in range(1, N + 1) if x not in best_set]
            random.shuffle(candidates)
            for x in candidates[:max_candidates]:
                if try_add(x, best_set, best_cnt):
                    add_to_counts(x, best_set, best_cnt)
                    best_set.add(x)
                    improved = True
                    swaps_done += 1
                    found = True
                    break
            if not found:
                add_to_counts(y, best_set, best_cnt)
                best_set.add(y)
        
        # Add phase
        if time.time() < deadline - 5.0:
            for x in range(1, N + 1):
                if time.time() > deadline - 3.0:
                    break
                if x not in best_set and try_add(x, best_set, best_cnt):
                    add_to_counts(x, best_set, best_cnt)
                    best_set.add(x)
                    improved = True
        
        if not improved:
            no_improve_passes += 1
        else:
            no_improve_passes = 0
    
    return sorted(best_set)
