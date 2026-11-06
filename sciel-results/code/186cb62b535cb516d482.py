import time
import random

N = 512

def _can_add(S, x):
    """
    Check if x can be added to the set S (represented as an integer bitset)
    without creating a 4-term arithmetic progression.
    S has bit i set if i is in the set.
    Uses fail-fast: check cases with fewer iterations first.
    """
    # Precompute max_d for each case
    md1 = (N - x) // 3
    md2 = min(x - 1, (N - x) // 2)
    md3 = min((x - 1) // 2, N - x)
    md4 = (x - 1) // 3
    
    # Order cases by loop length (shortest first) for fail-fast
    cases = []
    if md1 > 0:
        cases.append((md1, 1, x))
    if md2 > 0:
        cases.append((md2, 2, x))
    if md3 > 0:
        cases.append((md3, 3, x))
    if md4 > 0:
        cases.append((md4, 4, x))
    
    cases.sort(key=lambda t: t[0])
    
    for md, case_type, _ in cases:
        if case_type == 1:
            # x is 1st term: x, x+d, x+2d, x+3d
            for d in range(1, md + 1):
                if (S >> (x + d)) & 1 and (S >> (x + 2 * d)) & 1 and (S >> (x + 3 * d)) & 1:
                    return False
        elif case_type == 2:
            # x is 2nd term: x-d, x, x+d, x+2d
            for d in range(1, md + 1):
                if (S >> (x - d)) & 1 and (S >> (x + d)) & 1 and (S >> (x + 2 * d)) & 1:
                    return False
        elif case_type == 3:
            # x is 3rd term: x-2d, x-d, x, x+d
            for d in range(1, md + 1):
                if (S >> (x - 2 * d)) & 1 and (S >> (x - d)) & 1 and (S >> (x + d)) & 1:
                    return False
        else:  # case_type == 4
            # x is 4th term: x-3d, x-2d, x-d, x
            for d in range(1, md + 1):
                if (S >> (x - 3 * d)) & 1 and (S >> (x - 2 * d)) & 1 and (S >> (x - d)) & 1:
                    return False
    
    return True

def _greedy_construct(order):
    """
    Greedily construct a set by adding elements in the given order.
    """
    S = 0
    for x in order:
        if _can_add(S, x):
            S |= (1 << x)
    return S

def _local_search_add(S):
    """
    Try to add any excluded element to S. Repeat until no more can be added.
    """
    changed = True
    while changed:
        changed = False
        for x in range(1, N + 1):
            if not (S >> x) & 1:
                if _can_add(S, x):
                    S |= (1 << x)
                    changed = True
    return S

def _popcount(S):
    """
    Count the number of bits set in S.
    """
    return bin(S).count('1')

def _constraint_score(x):
    """
    Calculate the number of potential 4-APs involving x.
    Lower score means less constrained.
    """
    score = 0
    score += (N - x) // 3
    score += min(x - 1, (N - x) // 2)
    score += min((x - 1) // 2, N - x)
    score += (x - 1) // 3
    return score

def _conflict_count(S, x):
    """
    Count how many existing 4-APs would be created by adding x to S.
    Used to prioritize which elements to remove in swap moves.
    """
    count = 0
    md1 = (N - x) // 3
    for d in range(1, md1 + 1):
        if (S >> (x + d)) & 1 and (S >> (x + 2 * d)) & 1 and (S >> (x + 3 * d)) & 1:
            count += 1
    md2 = min(x - 1, (N - x) // 2)
    for d in range(1, md2 + 1):
        if (S >> (x - d)) & 1 and (S >> (x + d)) & 1 and (S >> (x + 2 * d)) & 1:
            count += 1
    md3 = min((x - 1) // 2, N - x)
    for d in range(1, md3 + 1):
        if (S >> (x - 2 * d)) & 1 and (S >> (x - d)) & 1 and (S >> (x + d)) & 1:
            count += 1
    md4 = (x - 1) // 3
    for d in range(1, md4 + 1):
        if (S >> (x - 3 * d)) & 1 and (S >> (x - 2 * d)) & 1 and (S >> (x - d)) & 1:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_S = 0
    best_size = 0
    
    # Pre-compute constraint scores
    scores = {x: _constraint_score(x) for x in range(1, N + 1)}
    
    # 1. Multiple diverse greedy constructions
    # Strategy A: Sort by constraint score (ascending)
    smart_order = sorted(range(1, N + 1), key=lambda x: scores[x])
    S_smart = _greedy_construct(smart_order)
    S_smart = _local_search_add(S_smart)
    best_S = S_smart
    best_size = _popcount(S_smart)
    
    # Strategy B: Sort by constraint score (descending) - try most constrained first
    desc_order = sorted(range(1, N + 1), key=lambda x: -scores[x])
    S_desc = _greedy_construct(desc_order)
    S_desc = _local_search_add(S_desc)
    size = _popcount(S_desc)
    if size > best_size:
        best_size = size
        best_S = S_desc
    
    # Strategy C: Natural order
    S_nat = _greedy_construct(list(range(1, N + 1)))
    S_nat = _local_search_add(S_nat)
    size = _popcount(S_nat)
    if size > best_size:
        best_size = size
        best_S = S_nat
    
    # Strategy D: Reverse natural order
    S_rev = _greedy_construct(list(range(N, 0, -1)))
    S_rev = _local_search_add(S_rev)
    size = _popcount(S_rev)
    if size > best_size:
        best_size = size
        best_S = S_rev
    
    # 2. Random Greedy: A few restarts
    for _ in range(15):
        if time.time() > deadline - 5.0: break
        order = list(range(1, N + 1))
        random.shuffle(order)
        S_rand = _greedy_construct(order)
        S_rand = _local_search_add(S_rand)
        size = _popcount(S_rand)
        if size > best_size:
            best_size = size
            best_S = S_rand

    # 3. Iterated Local Search (ILS) with Swap Moves
    # We introduce a swap move: remove one element, then try to add two others.
    # This allows escaping local optima where no single addition is possible,
    # but a swap increases the size.
    
    # Pre-compute conflict counts for current best set to prioritize removals
    while time.time() < deadline:
        # Strategy A: Perturb and Re-add (Standard ILS)
        # Strategy B: Swap Move (Remove 1, Add 2)
        
        # 60% chance to try a swap move, 40% chance to try perturb/re-add
        if random.random() < 0.6:
            # --- Swap Move ---
            S = best_S
            elements = [i for i in range(1, N + 1) if (S >> i) & 1]
            
            if len(elements) > 0:
                # Prioritize removing elements with high conflict count
                # (removing them frees up more space)
                conflict_scores = []
                for elem in elements:
                    # Compute how many 4-APs involve this element
                    md1 = (N - elem) // 3
                    c = 0
                    for d in range(1, md1 + 1):
                        if (S >> (elem + d)) & 1 and (S >> (elem + 2 * d)) & 1 and (S >> (elem + 3 * d)) & 1:
                            c += 1
                    md2 = min(elem - 1, (N - elem) // 2)
                    for d in range(1, md2 + 1):
                        if (S >> (elem - d)) & 1 and (S >> (elem + d)) & 1 and (S >> (elem + 2 * d)) & 1:
                            c += 1
                    md3 = min((elem - 1) // 2, N - elem)
                    for d in range(1, md3 + 1):
                        if (S >> (elem - 2 * d)) & 1 and (S >> (elem - d)) & 1 and (S >> (elem + d)) & 1:
                            c += 1
                    md4 = (elem - 1) // 3
                    for d in range(1, md4 + 1):
                        if (S >> (elem - 3 * d)) & 1 and (S >> (elem - 2 * d)) & 1 and (S >> (elem - d)) & 1:
                            c += 1
                    conflict_scores.append((c, elem))
                
                # Sort by conflict count descending, take top candidates
                conflict_scores.sort(key=lambda t: -t[0])
                top_removals = [elem for _, elem in conflict_scores[:min(20, len(elements))]]
                
                # Pick a random element from top removals or random
                if random.random() < 0.7:
                    rem = random.choice(top_removals)
                else:
                    rem = random.choice(elements)
                
                S &= ~(1 << rem)
                
                # Now try to add two elements that are not in S
                found_swap = False
                
                # Get list of excluded elements
                excluded = [i for i in range(1, N + 1) if not (S >> i) & 1]
                
                # Shuffle excluded to randomize
                random.shuffle(excluded)
                
                # Try a limited number of add1 candidates
                for add1 in excluded[:30]:
                    if _can_add(S, add1):
                        S_temp = S | (1 << add1)
                        # Now try to find add2
                        for add2 in excluded[30:]:
                            if add2 == add1:
                                continue
                            if not (S_temp >> add2) & 1:
                                if _can_add(S_temp, add2):
                                    # Valid swap found
                                    S = S_temp | (1 << add2)
                                    found_swap = True
                                    break
                    if found_swap:
                        break
                
                if found_swap:
                    # We made a swap. Now run add-only local search to fill in any more
                    S = _local_search_add(S)
                    size = _popcount(S)
                    if size > best_size:
                        best_size = size
                        best_S = S
                else:
                    # No swap found, revert to best_S
                    pass

        else:
            # --- Perturb and Re-add ---
            k = random.randint(1, 10)
            S = best_S
            elements = [i for i in range(1, N + 1) if (S >> i) & 1]
            
            if len(elements) > k:
                to_remove = random.sample(elements, k)
                for x in to_remove:
                    S &= ~(1 << x)
            
            # Run local search on the perturbed set
            S = _local_search_add(S)
            size = _popcount(S)
            
            if size > best_size:
                best_size = size
                best_S = S
            else:
                # If no improvement, occasionally try a fresh random restart
                if random.random() < 0.03:
                    order = list(range(1, N + 1))
                    random.shuffle(order)
                    S_rand = _greedy_construct(order)
                    S_rand = _local_search_add(S_rand)
                    size = _popcount(S_rand)
                    if size > best_size:
                        best_size = size
                        best_S = S_rand

    # Convert best_S to sorted list
    result = [i for i in range(1, N + 1) if (best_S >> i) & 1]
    return result
