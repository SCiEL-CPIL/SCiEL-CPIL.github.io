import time
import random

N = 128
MAX_TIME = 85.0  # safety margin for 90s wall-clock limit

def _precompute():
    """Precompute sum_pairs and partners for fast validity checks."""
    sum_pairs = [[] for _ in range(129)]
    for a in range(1, 129):
        for b in range(a + 1, 129):
            s = a + b
            if s <= 128:
                sum_pairs[s].append((a, b))
    
    partners = [[] for _ in range(129)]
    for x in range(1, 129):
        for y in range(1, 129):
            if y != x and x + y <= 128:
                partners[x].append(y)
    
    return sum_pairs, partners

SUM_PAIRS, PARTNERS = _precompute()

def _can_add(S_mask, x):
    """
    Check if S_mask U {x} is a valid set (no forbidden triple).
    Uses bit operations for speed.
    """
    # Case 1: x is a member of the triple (a = x).
    # Need distinct b, c in S such that x+b, x+c, b+c are all in S U {x}.
    # Since x+b, x+c > x, they must be in S. b+c in S or b+c == x.
    # V = { y in S | x+y in S } = S & (S >> x)
    V_mask = S_mask & (S_mask >> x)
    if V_mask.bit_count() >= 2:
        v_list = []
        temp = V_mask
        while temp:
            lsb = temp & -temp
            v_list.append(lsb.bit_length() - 1)
            temp -= lsb
        n_v = len(v_list)
        for i in range(n_v):
            b = v_list[i]
            for j in range(i + 1, n_v):
                c = v_list[j]
                s = b + c
                if s == x or ((S_mask >> s) & 1):
                    return False

    # Case 2: x is a sum of two members (a+b = x).
    # Need distinct a, b, c in S with a+b=x, a+c in S, b+c in S.
    for a, b in SUM_PAIRS[x]:
        if ((S_mask >> a) & 1) == 0:
            continue
        if ((S_mask >> b) & 1) == 0:
            continue
        # c in S \ {a,b} such that a+c in S and b+c in S
        U_mask = (S_mask & (S_mask >> a)) & (S_mask & (S_mask >> b))
        U_mask &= ~(1 << a)
        U_mask &= ~(1 << b)
        if U_mask != 0:
            return False

    return True

def _mask_to_set(S_mask):
    """Convert bitmask to set of integers."""
    s = set()
    temp = S_mask
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        s.add(idx)
        temp -= lsb
    return s

def _set_to_mask(s_set):
    """Convert set to bitmask."""
    mask = 0
    for x in s_set:
        mask |= (1 << x)
    return mask

def _greedy_construct(order, S_mask=0):
    """Greedy construction given a fixed order of candidates."""
    for x in order:
        if _can_add(S_mask, x):
            S_mask |= (1 << x)
    return S_mask

def _count_conflicts_bitmask(S_mask):
    """
    Returns a dictionary mapping each element in S_mask to the number of 
    forbidden triples (a,b,c) in S_mask that contain that element.
    """
    S_set = _mask_to_set(S_mask)
    s_list = sorted(S_set)
    n = len(s_list)
    counts = {x: 0 for x in s_list}
    
    for i in range(n):
        a = s_list[i]
        if i + 1 >= n:
            break
        if a + s_list[i+1] > 128:
            break
            
        for j in range(i + 1, n):
            b = s_list[j]
            ab = a + b
            if ab > 128:
                break
            if not ((S_mask >> ab) & 1):
                continue
            for k in range(j + 1, n):
                c = s_list[k]
                bc = b + c
                if bc > 128:
                    break
                ac = a + c
                if ((S_mask >> ac) & 1) and ((S_mask >> bc) & 1):
                    counts[a] += 1
                    counts[b] += 1
                    counts[c] += 1
    return counts

def _greedy_remove_until_valid(S_mask, start_time, deadline_margin=5.0):
    """
    Start with S_mask and remove elements with highest conflict count until valid.
    """
    while True:
        if time.time() - start_time > MAX_TIME - deadline_margin:
            break
        S_set = _mask_to_set(S_mask)
        counts = _count_conflicts_bitmask(S_mask)
        if not counts or all(v == 0 for v in counts.values()):
            break  # Valid
        
        # Find element with max count
        max_count = max(counts.values())
        candidates = [x for x, v in counts.items() if v == max_count]
        # Tie-break: remove smallest to preserve top-half density
        remove_el = min(candidates)
        S_mask &= ~(1 << remove_el)
        
    return S_mask

def _try_greedy_add(S_mask, start_time, deadline_margin=5.0):
    """Try to greedily add elements to improve the set."""
    S_set = _mask_to_set(S_mask)
    candidates = [x for x in range(1, N+1) if x not in S_set]
    candidates.sort()
    
    improved = True
    while improved and time.time() - start_time < MAX_TIME - deadline_margin:
        improved = False
        for cand in candidates[:]:
            if time.time() - start_time > MAX_TIME - deadline_margin:
                break
            if _can_add(S_mask, cand):
                S_mask |= (1 << cand)
                candidates = [x for x in candidates if x != cand]
                improved = True
    
    return S_mask

def _try_remove_add(S_mask, start_time, deadline_margin=5.0):
    """Try removing one element and adding one or more others."""
    S_set = _mask_to_set(S_mask)
    best_mask = S_mask
    best_len = S_mask.bit_count()
    
    for remove_el in sorted(S_set):
        if time.time() - start_time > MAX_TIME - deadline_margin:
            break
        remaining_mask = S_mask & ~(1 << remove_el)
        candidates = [x for x in range(1, N+1) if not ((remaining_mask >> x) & 1)]
        candidates.sort()
        
        # Try to add as many as possible from remaining_set
        current_mask = remaining_mask
        for cand in candidates:
            if time.time() - start_time > MAX_TIME - deadline_margin:
                break
            if _can_add(current_mask, cand):
                current_mask |= (1 << cand)
        
        if current_mask.bit_count() > best_len:
            best_mask = current_mask
            best_len = current_mask.bit_count()
    
    return best_mask

def _local_search(S_mask, start_time, max_passes=30, swap_limit=15, deadline_margin=5.0):
    """Improve S via additions and systematic swaps."""
    S_set = _mask_to_set(S_mask)
    not_in_S = [x for x in range(1, 129) if x not in S_set]

    for _ in range(max_passes):
        if time.time() - start_time > MAX_TIME - deadline_margin:
            break

        improved = False
        random.shuffle(not_in_S)

        for x in not_in_S[:]:
            if time.time() - start_time > MAX_TIME - deadline_margin:
                break

            # Try to add x directly
            if _can_add(S_mask, x):
                S_mask |= (1 << x)
                S_set.add(x)
                not_in_S.remove(x)
                improved = True
                continue

            # Try swaps: remove y from S, add x
            S_list = list(S_set)
            random.shuffle(S_list)

            found_swap = False
            for y in S_list[:swap_limit]:
                if time.time() - start_time > MAX_TIME - deadline_margin:
                    break
                S_temp_mask = S_mask ^ (1 << y)
                if _can_add(S_temp_mask, x):
                    S_mask = S_temp_mask | (1 << x)
                    S_set.remove(y)
                    S_set.add(x)
                    not_in_S.append(y)
                    if x in not_in_S:
                        not_in_S.remove(x)
                    found_swap = True
                    improved = True
                    break

            if found_swap:
                break

        if not improved:
            break

    return S_mask

def construct():
    start_time = time.time()
    random.seed(42)
    
    # Precompute degree for ordering
    degree = [0] * 129
    for x in range(1, 129):
        degree[x] = len(PARTNERS[x])
    
    degree_order = sorted(range(1, 129), key=lambda x: degree[x])
    
    best_mask = 0
    
    # Multiple greedy constructions with diverse orderings
    S_mask = _greedy_construct(list(range(1, 129)), 0)
    if S_mask.bit_count() > best_mask.bit_count():
        best_mask = S_mask
    
    S_mask = _greedy_construct(list(range(128, 0, -1)), 0)
    if S_mask.bit_count() > best_mask.bit_count():
        best_mask = S_mask
    
    S_mask = _greedy_construct(list(degree_order), 0)
    if S_mask.bit_count() > best_mask.bit_count():
        best_mask = S_mask
    
    S_mask = _greedy_construct(list(reversed(degree_order)), 0)
    if S_mask.bit_count() > best_mask.bit_count():
        best_mask = S_mask
    
    for i in range(50):
        if time.time() > MAX_TIME - 10.0:
            break
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        S_mask = _greedy_construct(candidates, 0)
        if S_mask.bit_count() > best_mask.bit_count():
            best_mask = S_mask
    
    # Local search from best set
    if time.time() < MAX_TIME - 5.0:
        best_mask = _local_search(best_mask, start_time, max_passes=30)
    
    # More random restarts with local search
    for i in range(50):
        if time.time() > MAX_TIME - 5.0:
            break
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        S_mask = _greedy_construct(candidates, 0)
        if time.time() < MAX_TIME - 5.0:
            S_mask = _local_search(S_mask, start_time, max_passes=20)
        if S_mask.bit_count() > best_mask.bit_count():
            best_mask = S_mask
    
    # Final: try to add any remaining elements
    if time.time() < MAX_TIME - 2.0:
        for x in range(1, 129):
            if (best_mask >> x) & 1:
                continue
            if _can_add(best_mask, x):
                best_mask |= (1 << x)
    
    # One more local search pass
    if time.time() < MAX_TIME - 1.0 and best_mask.bit_count() < 128:
        best_mask = _local_search(best_mask, start_time, max_passes=15)
    
    # Convert mask to list of ints
    result = []
    temp = best_mask
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        result.append(int(idx))
        temp -= lsb
    
    return result
