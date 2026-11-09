"""Optimized 3-AP-free reciprocal weight maximizer."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 75.0
    N = 256
    random.seed(42)
    
    # Precompute conflict graph as bitmasks
    conflict_mask = [0] * (N + 1)
    for d in range(1, N + 1):
        for a in range(1, N + 1 - 2 * d):
            b = a + d
            c = a + 2 * d
            conflict_mask[a] |= (1 << b) | (1 << c)
            conflict_mask[b] |= (1 << a) | (1 << c)
            conflict_mask[c] |= (1 << a) | (1 << b)
    
    def can_add(x, sel_mask):
        return (conflict_mask[x] & sel_mask) == 0
    
    def greedy():
        sel_mask = 0
        sel_list = []
        for x in range(1, N + 1):
            if can_add(x, sel_mask):
                sel_mask |= (1 << x)
                sel_list.append(x)
        return sel_list, sel_mask
    
    def score_of_list(lst):
        return sum(1.0 / x for x in lst)
    
    best_list, best_mask = greedy()
    best_score = score_of_list(best_list)
    
    # Local search with bitmask
    def local_search(sel_list, sel_mask, max_time=10.0):
        sel_list = list(sel_list)
        sel_mask = sel_mask
        cur_score = score_of_list(sel_list)
        improved = True
        t0 = time.time()
        while improved and (time.time() - t0) < max_time and (time.time() - start_time) < DEADLINE - 5:
            improved = False
            # Try single additions
            for x in range(1, N + 1):
                if not (sel_mask >> x) & 1 and can_add(x, sel_mask):
                    sel_mask |= (1 << x)
                    sel_list.append(x)
                    cur_score += 1.0 / x
                    improved = True
                    break
            # Try single swaps: remove x, add y
            if not improved:
                # Sort by weight ascending (remove least valuable first)
                candidates = sorted(sel_list, key=lambda v: 1.0/v)
                for x in candidates:
                    new_mask = sel_mask & ~(1 << x)
                    # Find best y to add
                    best_y = -1
                    best_gain = 0
                    for y in range(1, N + 1):
                        if not (new_mask >> y) & 1 and can_add(y, new_mask):
                            gain = 1.0/y - 1.0/x
                            if gain > best_gain:
                                best_gain = gain
                                best_y = y
                    if best_y > 0:
                        sel_mask = new_mask | (1 << best_y)
                        sel_list.remove(x)
                        sel_list.append(best_y)
                        cur_score += best_gain
                        improved = True
                        break
        return sel_list, sel_mask, cur_score
    
    # Multi-start
    for restart in range(50):
        if (time.time() - start_time) > DEADLINE - 10:
            break
        
        if restart == 0:
            s_list, s_mask = greedy()
        elif restart < 10:
            # Perturb best
            s_list = list(best_list)
            s_mask = best_mask
            k = random.randint(2, 5)
            to_remove = random.sample(s_list, min(k, len(s_list)))
            for x in to_remove:
                s_mask &= ~(1 << x)
                s_list.remove(x)
            # Re-greedy
            for x in range(1, N + 1):
                if not (s_mask >> x) & 1 and can_add(x, s_mask):
                    s_mask |= (1 << x)
                    s_list.append(x)
        else:
            # Random restart
            s_list = []
            s_mask = 0
            # Random subset of small numbers as seed
            seed_size = random.randint(0, 5)
            seeds = random.sample(range(1, 50), seed_size)
            for x in sorted(seeds):
                if can_add(x, s_mask):
                    s_mask |= (1 << x)
                    s_list.append(x)
            for x in range(1, N + 1):
                if not (s_mask >> x) & 1 and can_add(x, s_mask):
                    s_mask |= (1 << x)
                    s_list.append(x)
        
        s_list, s_mask, s_score = local_search(s_list, s_mask, max_time=8.0)
        if s_score > best_score:
            best_score = s_score
            best_list = s_list
            best_mask = s_mask
    
    if not best_list:
        best_list = [1]
    
    return best_list
# EVOLVE-BLOCK-END
