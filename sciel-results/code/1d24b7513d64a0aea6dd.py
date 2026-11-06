import time
import random

# EVOLVE-BLOCK-START

def _build_triples(n_max=180):
    """Build all triples (a, b, c) with a < b < c, a in [1,n_max], and 1/a = 1/b + 1/c."""
    triples = []
    for a in range(1, n_max + 1):
        for b in range(a + 1, n_max + 1):
            diff = b - a
            if diff == 0:
                continue
            if (a * b) % diff != 0:
                continue
            c = (a * b) // diff
            if c > n_max or c <= b:
                continue
            triples.append((a, b, c))
    return triples

def _init_data(n_max=180):
    triples = _build_triples(n_max)
    # Map element to list of triple indices
    elem_to_triples = {i: [] for i in range(1, n_max + 1)}
    for idx, (a, b, c) in enumerate(triples):
        elem_to_triples[a].append(idx)
        elem_to_triples[b].append(idx)
        elem_to_triples[c].append(idx)
    
    # Precompute for each triple, the other two elements for each position
    # triple_data[idx] = [(a, b, c), (a, b), (a, c), (b, c)] is not quite right.
    # We need: for element x in triple t, what are the other two?
    # Let's just store the triples and use them directly.
    return triples, elem_to_triples

def _check_validity(S, triples, elem_to_triples):
    """Check if S is a valid set. Returns True if no triple is fully contained in S."""
    for idx, (a, b, c) in enumerate(triples):
        if a in S and b in S and c in S:
            return False
    return True

def _greedy_construct(triples, elem_to_triples, n_max, random_seed=None, deadline=None):
    """
    Greedy construction: Start with empty set, add elements one by one.
    Or start with all, remove.
    Let's try starting with all and removing, but with randomization.
    """
    if random_seed is not None:
        random.seed(random_seed)
        
    S = set(range(1, n_max + 1))
    
    # Count conflicts for each element
    # conflict_count[x] = number of triples fully in S that contain x
    # Initially, all triples are "active" (all 3 in S)
    triple_active = [True] * len(triples)
    conflict_count = {i: 0 for i in range(1, n_max + 1)}
    
    for idx, (a, b, c) in enumerate(triples):
        conflict_count[a] += 1
        conflict_count[b] += 1
        conflict_count[c] += 1
    
    active_triples = len(triples)
    
    while active_triples > 0:
        if deadline and time.time() > deadline:
            break
            
        # Find elements with max conflict
        # To add randomness, pick from top K or shuffle
        # Let's pick the element with max conflict, breaking ties randomly
        max_conf = max(conflict_count[x] for x in S)
        if max_conf == 0:
            break
            
        candidates = [x for x in S if conflict_count[x] == max_conf]
        if not candidates:
            break
        to_remove = random.choice(candidates)
        
        S.remove(to_remove)
        
        # Update conflict counts for triples containing to_remove
        for tidx in elem_to_triples[to_remove]:
            if triple_active[tidx]:
                active_triples -= 1
                triple_active[tidx] = False
                a, b, c = triples[tidx]
                for x in (a, b, c):
                    if x in S:
                        conflict_count[x] -= 1
            else:
                # Triple already inactive, but to_remove was part of it?
                # If triple was inactive, it means one of the elements was already removed.
                # Removing to_remove doesn't change the fact that it's inactive.
                # But we must be careful: conflict_count[x] is the number of ACTIVE triples containing x.
                # If triple_active[tidx] is False, it was already decremented when the first element was removed.
                # So we don't decrement again.
                pass
                
    return sorted(S)

def _local_search(S, triples, elem_to_triples, n_max, deadline, max_iter=5000):
    """
    Local search:
    1. Try to add any element from Removed to S if valid (0-1 move).
    2. Try to swap: remove y from S, add x from Removed (1-1 move) if valid.
    """
    S = set(S)
    removed = set(range(1, n_max + 1)) - S
    
    # Precompute for fast validity check
    # For each element x, list of triples containing x.
    
    def can_add(x):
        """Check if x can be added to S without creating a triple."""
        for tidx in elem_to_triples[x]:
            a, b, c = triples[tidx]
            # Check if the other two are in S
            others = [y for y in (a, b, c) if y != x]
            if all(y in S for y in others):
                return False
        return True

    improved = True
    iters = 0
    while improved and iters < max_iter:
        if deadline and time.time() > deadline:
            break
        improved = False
        iters += 1
        
        # 1. Try 0-1 moves (Add)
        added_any = False
        # Shuffle removed to get different orders
        rem_list = list(removed)
        random.shuffle(rem_list)
        for x in rem_list:
            if can_add(x):
                S.add(x)
                removed.discard(x)
                improved = True
                added_any = True
        if added_any:
            continue # Restart search as S changed significantly
            
        # 2. Try 1-1 moves (Swap)
        # Try to remove y from S and add x from removed
        # This is expensive: O(|S| * |Removed| * check_cost)
        # check_cost is small (degree of x).
        # We can limit the number of swaps tried per iteration.
        
        # Heuristic: Only swap if it might lead to an add?
        # Or just accept valid swaps to escape local minima.
        
        # To avoid infinite loops, we can use a tabu list or limit swaps.
        # Let's try a few random swaps.
        s_list = list(S)
        random.shuffle(s_list)
        swaps_tried = 0
        max_swaps_per_iter = 10
        
        for y in s_list:
            if swaps_tried >= max_swaps_per_iter:
                break
            for x in rem_list:
                if swaps_tried >= max_swaps_per_iter:
                    break
                # Check if S - {y} + {x} is valid
                # This is equivalent to:
                # 1. x can be added to S - {y}
                # 2. No triple is created by x that involves elements in S - {y}
                
                # We can simulate:
                # Temporarily remove y, check if x can be added, then restore y.
                # But can_add(x) checks against S.
                # We need a modified check: can_add_x_given_removed_y(x, y)
                
                # A triple containing x is invalid if the other two are in S\{y}.
                # So, for each triple t containing x:
                #   others = [a,b,c] - {x}
                #   If y in others:
                #       Check if the other one (other2) is in S.
                #       If other2 in S, then triple is NOT formed (because y is removed).
                #       Wait, if y is removed, the triple {x, other1, other2} is only formed if other1, other2 are in S\{y}.
                #       If y is one of the others, say others = [y, z].
                #       Then the triple is {x, y, z}.
                #       In S\{y}, y is gone. So the triple is NOT formed.
                #       So we only care about triples where y is NOT one of the other two.
                #   If y not in others:
                #       Check if both others are in S.
                #       If both in S, then triple IS formed.
                
                valid_swap = True
                for tidx in elem_to_triples[x]:
                    a, b, c = triples[tidx]
                    others = [v for v in (a, b, c) if v != x]
                    # others has 2 elements
                    o1, o2 = others
                    if y == o1 or y == o2:
                        # y is part of this triple.
                        # Since y is removed, this triple cannot be formed.
                        # So this triple is OK.
                        pass
                    else:
                        # y is not part of this triple.
                        # Check if o1 and o2 are in S.
                        if o1 in S and o2 in S:
                            valid_swap = False
                            break
                            
                if valid_swap:
                    # Perform swap
                    S.remove(y)
                    S.add(x)
                    removed.discard(x)
                    removed.add(y)
                    improved = True
                    swaps_tried += 1
                    break # Restart swap search for this y? Or continue?
                    # Let's break out of x loop and continue to next y or restart
            if improved:
                break # Restart entire local search loop
        
    return sorted(S)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n_max = 180
    triples, elem_to_triples = _init_data(n_max)
    
    best_set = []
    best_size = 0
    
    # Multi-start with different random seeds
    # We can run several iterations of Greedy + Local Search
    # Time budget per iteration: ~5-10s?
    # Total time 85s.
    
    num_restarts = 10
    time_per_restart = (deadline - start_time) / num_restarts
    
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # 1. Greedy Construction
        # Use different strategies for different restarts
        # Strategy 1: Standard Greedy (max conflict)
        # Strategy 2: Randomized Greedy
        # Strategy 3: Start with specific subsets?
        
        # Let's just use Randomized Greedy
        seed = i
        greedy_set = _greedy_construct(triples, elem_to_triples, n_max, random_seed=seed, deadline=deadline - 5.0)
        
        # 2. Local Search
        ls_set = _local_search(greedy_set, triples, elem_to_triples, n_max, deadline - 2.0, max_iter=2000)
        
        if len(ls_set) > best_size:
            best_size = len(ls_set)
            best_set = ls_set
            
    # Final polish on the best set found
    if time.time() < deadline - 1.0 and best_set:
        final_set = _local_search(best_set, triples, elem_to_triples, n_max, deadline - 0.5, max_iter=5000)
        if len(final_set) > best_size:
            best_set = final_set

    if not best_set:
        # Fallback: Upper half
        best_set = list(range(91, 181))

    return best_set

# EVOLVE-BLOCK-END
