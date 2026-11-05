import random
import time
from itertools import combinations

def can_add(s, a):
    """
    Check if adding element a to set s maintains validity.
    Conditions:
    1. a doesn't divide sum of any 3 distinct elements in s
    2. No element in s divides sum of a + any 2 distinct other elements in s
    """
    s_list = sorted(s)
    n = len(s_list)
    s_set = set(s_list)
    
    # Condition 1: a | (x+y+z) for 3 distinct x,y,z in s
    if n >= 3:
        max_sum3 = s_list[-1] + s_list[-2] + s_list[-3]
        if a <= max_sum3:
            min_sum3 = s_list[0] + s_list[1] + s_list[2]
            start_k = (min_sum3 + a - 1) // a
            end_k = max_sum3 // a
            if start_k <= end_k:
                for k in range(start_k, end_k + 1):
                    target = k * a
                    found = False
                    for i in range(n - 2):
                        xi = s_list[i]
                        if xi + s_list[i+1] + s_list[i+2] > target:
                            break
                        for j in range(i + 1, n - 1):
                            xj = s_list[j]
                            if xi + xj + s_list[j+1] > target:
                                break
                            rem = target - xi - xj
                            if rem in s_set and rem > xj:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        return False
    
    # Condition 2: For each b in s, b | (a + x + y) for 2 distinct x,y in s\{b}
    for idx_b, b in enumerate(s_list):
        others = s_list[:idx_b] + s_list[idx_b+1:]
        m = len(others)
        if m < 2:
            continue
        
        max_sum = a + others[-1] + others[-2]
        if b > max_sum:
            continue
        
        min_sum = a + others[0] + others[1]
        start_k = (min_sum + b - 1) // b
        end_k = max_sum // b
        if start_k > end_k:
            continue
        
        others_set = set(others)
        for k in range(start_k, end_k + 1):
            target = k * b
            need = target - a
            if need < 0:
                continue
            # Check if any pair in others sums to need
            found = False
            for i in range(m - 1):
                xi = others[i]
                if xi + others[i+1] > need:
                    break
                rem = need - xi
                if rem in others_set and rem > xi:
                    found = True
                    break
            if found:
                return False
    
    return True

def is_valid_fast(s):
    """Full validity check using can_add logic for each element."""
    s_list = sorted(s)
    n = len(s_list)
    if n < 4:
        return True
    
    for idx_a, a in enumerate(s_list):
        others = s_list[:idx_a] + s_list[idx_a+1:]
        m = len(others)
        if m < 3:
            continue
        
        max_sum3 = others[-1] + others[-2] + others[-3]
        if a > max_sum3:
            continue
        
        min_sum3 = others[0] + others[1] + others[2]
        start_k = (min_sum3 + a - 1) // a
        end_k = max_sum3 // a
        if start_k > end_k:
            continue
        
        others_set = set(others)
        for k in range(start_k, end_k + 1):
            target = k * a
            found = False
            for i in range(m - 2):
                xi = others[i]
                if xi + others[i+1] + others[i+2] > target:
                    break
                for j in range(i + 1, m - 1):
                    xj = others[j]
                    if xi + xj + others[j+1] > target:
                        break
                    rem = target - xi - xj
                    if rem in others_set and rem > xj:
                        found = True
                        break
                if found:
                    break
            if found:
                return False
    return True

def greedy_descending():
    """Greedy: add elements from 192 down to 1."""
    selected = set()
    for x in range(192, 0, -1):
        if can_add(selected, x):
            selected.add(x)
    return selected

def greedy_with_gaps():
    """Greedy with strategic gaps to reduce divisibility conflicts."""
    selected = set()
    # Start with large numbers, skip some to create gaps
    for x in range(192, 0, -1):
        if can_add(selected, x):
            selected.add(x)
    return selected

def local_search(s, rng, deadline, max_iters=500):
    """Hill-climbing local search with 1-swap and 2-swap moves."""
    s = set(s)
    cur_size = len(s)
    stall = 0
    all_nums = set(range(1, 193))
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try adding an element
        candidates = sorted(all_nums - s, reverse=True)
        rng.shuffle(candidates)
        for x in candidates[:20]:
            if can_add(s, x):
                s.add(x)
                cur_size += 1
                improved = True
                stall = 0
                break
        
        if improved:
            continue
        
        # Try 1-swap: remove small, add large
        elems = sorted(s)
        for rem in elems[:10]:
            candidates = sorted(all_nums - s, reverse=True)
            rng.shuffle(candidates)
            for add in candidates[:15]:
                if add == rem:
                    continue
                new_s = (s - {rem}) | {add}
                if is_valid_fast(new_s):
                    s = new_s
                    improved = True
                    stall = 0
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try 2-swap
        if rng.random() < 0.6:
            elems = sorted(s)
            rem_pairs = list(combinations(elems[:12], 2))
            rng.shuffle(rem_pairs)
            candidates = sorted(all_nums - s, reverse=True)
            rng.shuffle(candidates)
            add_pairs = list(combinations(candidates[:12], 2))
            
            for rem1, rem2 in rem_pairs[:8]:
                for add1, add2 in add_pairs[:8]:
                    new_s = (s - {rem1, rem2}) | {add1, add2}
                    if is_valid_fast(new_s):
                        s = new_s
                        improved = True
                        stall = 0
                        break
                if improved:
                    break
        
        if not improved:
            stall += 1
            if stall > 50:
                break
    
    return s

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    rng = random.Random(12345)
    
    # Start with greedy descending
    best = greedy_descending()
    best_size = len(best)
    
    # Local search from greedy
    result = local_search(best, rng, deadline, max_iters=400)
    if len(result) > best_size:
        best_size = len(result)
        best = result
    
    # Multiple random restarts with diverse strategies
    rounds = 0
    while time.time() < deadline:
        rounds += 1
        
        if rounds <= 5:
            # Greedy with random perturbation
            start_s = set()
            for x in range(192, 0, -1):
                if can_add(start_s, x):
                    start_s.add(x)
        elif rounds <= 10:
            # Random start biased toward large numbers
            k = rng.randint(35, 45)
            weights = [x for x in range(1, 193)]
            start_s = set(rng.choices(range(1, 193), weights=weights, k=k))
            valid_s = set()
            for x in sorted(start_s, reverse=True):
                if can_add(valid_s, x):
                    valid_s.add(x)
            start_s = valid_s
        else:
            # Mix of greedy and random
            start_s = set()
            for x in range(192, 0, -1):
                if rng.random() < 0.85 and can_add(start_s, x):
                    start_s.add(x)
                elif can_add(start_s, x):
                    start_s.add(x)
        
        result = local_search(start_s, rng, deadline, max_iters=300)
        
        if len(result) > best_size:
            best_size = len(result)
            best = result
        
        if best_size >= 48:
            break
    
    return list(best)
