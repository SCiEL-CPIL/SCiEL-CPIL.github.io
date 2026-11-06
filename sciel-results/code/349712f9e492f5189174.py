import time
import random
import numpy as np
from itertools import combinations

def _is_valid(selected):
    """Check if selected is a valid set using vectorized operations."""
    n = len(selected)
    if n < 4:
        return True
    sel = np.array(selected, dtype=np.int64)
    for idx_a in range(n):
        a = sel[idx_a]
        # Get all other elements
        mask = np.ones(n, dtype=bool)
        mask[idx_a] = False
        others = sel[mask]
        if len(others) < 3:
            continue
        # Generate all triples of others
        o_idx = np.arange(len(others))
        i, j, k = np.meshgrid(o_idx, o_idx, o_idx, indexing='ij')
        # Keep only i < j < k
        valid = (i < j) & (j < k)
        i, j, k = i[valid], j[valid], k[valid]
        sums = others[i] + others[j] + others[k]
        if np.any(sums % a == 0):
            return False
    return True

def _can_add_vectorized(x, S):
    """
    Check if x can be added to set S without violating the condition.
    S is a list of valid integers. Uses vectorized operations for speed.
    """
    n = len(S)
    if n == 0:
        return True
    
    S_arr = np.array(S, dtype=np.int64)
    
    # 1. x does not divide sum of any 3 distinct elements in S
    if n >= 3:
        i, j, k = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
        valid = (i < j) & (j < k)
        i, j, k = i[valid], j[valid], k[valid]
        sums = S_arr[i] + S_arr[j] + S_arr[k]
        if np.any(sums % x == 0):
            return False
    
    # 2. No a in S divides x + b + c for distinct b, c in S\{a}
    for idx_a, a in enumerate(S_arr):
        # Get indices of elements != a position
        mask = np.ones(n, dtype=bool)
        mask[idx_a] = False
        other_indices = np.where(mask)[0]
        if len(other_indices) < 2:
            continue
        # Generate all pairs from other_indices
        oi, oj = np.meshgrid(other_indices, other_indices, indexing='ij')
        valid = oi < oj
        oi, oj = oi[valid], oj[valid]
        sums = x + S_arr[oi] + S_arr[oj]
        if np.any(sums % a == 0):
            return False
    
    return True

def _greedy(order, deadline):
    """Greedy construction following given order."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add_vectorized(x, selected):
            selected.append(x)
    return selected

def _local_search(selected, pool, deadline, max_iters=100):
    """
    Improve selected by adding from pool or swapping.
    """
    selected = list(selected)
    selected_set = set(selected)
    best = selected
    best_len = len(selected)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        
        # Try to add - check all pool elements
        for x in pool:
            if x in selected_set:
                continue
            if _can_add_vectorized(x, selected):
                selected.append(x)
                selected_set.add(x)
                if len(selected) > best_len:
                    best = list(selected)
                    best_len = len(selected)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swap: remove y from selected, add x from pool
        if len(selected) > 0 and len(pool) > 0:
            # Try multiple random swaps
            found_swap = False
            for _ in range(10):
                y = random.choice(selected)
                y_idx = selected.index(y)
                x = random.choice(pool)
                if x in selected_set:
                    continue
                
                candidate = selected[:y_idx] + selected[y_idx+1:]
                if _can_add_vectorized(x, candidate):
                    candidate.append(x)
                    selected = candidate
                    selected_set = set(selected)
                    found_swap = True
                    break
            
            if found_swap:
                improved = True
        
        if not improved:
            break
    
    return best

def _is_prime(x):
    if x < 2:
        return False
    if x < 4:
        return True
    if x % 2 == 0 or x % 3 == 0:
        return False
    i = 5
    while i * i <= x:
        if x % i == 0 or x % (i + 2) == 0:
            return False
        i += 6
    return True

def _num_divisors(x):
    cnt = 0
    for d in range(1, int(x**0.5) + 1):
        if x % d == 0:
            cnt += 1
            if d != x // d:
                cnt += 1
    return cnt

def _build_conflict_graph(all_nums):
    """
    Build a graph where edges indicate conflict potential.
    For each pair (a, b), count how many c exist such that a | (b + c + d) 
    for some other d, or similar. This is complex, so we use a simpler heuristic:
    count the number of pairs (c, d) in all_nums where a divides (b + c + d).
    Higher count means a and b are more likely to conflict when combined with others.
    """
    n = len(all_nums)
    conflict_count = np.zeros((n, n), dtype=np.int32)
    
    for i in range(n):
        a = all_nums[i]
        for j in range(i+1, n):
            b = all_nums[j]
            # Count how many pairs (c, d) with c != a, c != b, d != a, d != b, c != d
            # such that a divides (b + c + d) or b divides (a + c + d)
            count = 0
            for k in range(n):
                c = all_nums[k]
                if c == a or c == b:
                    continue
                for l in range(k+1, n):
                    d = all_nums[l]
                    if d == a or d == b:
                        continue
                    if (b + c + d) % a == 0:
                        count += 1
                    if (a + c + d) % b == 0:
                        count += 1
            conflict_count[i, j] = count
            conflict_count[j, i] = count
    
    return conflict_count

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best = []
    best_len = 0
    
    def _update_best(s):
        nonlocal best, best_len
        if len(s) > best_len and _is_valid(s):
            best = list(s)
            best_len = len(s)
    
    all_nums = list(range(1, 193))
    
    # Phase 1: Greedy with different orderings
    # Order 1: Descending
    if time.time() < deadline:
        order = sorted(all_nums, reverse=True)
        res = _greedy(order, deadline)
        _update_best(res)
        
    # Order 2: Primes first (descending), then composites
    if time.time() < deadline:
        primes = [x for x in all_nums if _is_prime(x)]
        composites = [x for x in all_nums if not _is_prime(x)]
        order = sorted(primes, reverse=True) + sorted(composites, reverse=True)
        res = _greedy(order, deadline)
        _update_best(res)
        
    # Order 3: Few divisors first
    if time.time() < deadline:
        order = sorted(all_nums, key=lambda x: (_num_divisors(x), -x))
        res = _greedy(order, deadline)
        _update_best(res)
    
    # Phase 2: Graph-based ordering
    # Build conflict graph and use it to prioritize elements
    if time.time() < deadline:
        try:
            conflict = _build_conflict_graph(all_nums)
            # Degree of each node
            degree = conflict.sum(axis=1)
            # Sort by degree (ascending - less conflicts first) then by value (descending)
            indices = np.argsort(degree)
            order = [all_nums[i] for i in indices]
            # Within same degree, prefer larger values
            order = sorted(order, key=lambda x: (degree[all_nums.index(x)], -x))
            res = _greedy(order, deadline)
            _update_best(res)
        except:
            pass
    
    # Phase 3: Local Search on best found so far
    if time.time() < deadline and best:
        pool = [x for x in all_nums if x not in set(best)]
        res = _local_search(best, pool, deadline, max_iters=200)
        _update_best(res)
        
    # Phase 4: Random Restart Greedy + Local Search
    rng = random.Random(42)
    while time.time() < deadline:
        # Random order, but biased towards larger numbers
        order = all_nums[:]
        rng.shuffle(order)
        
        if rng.random() < 0.5:
            order = sorted(order, key=lambda x: rng.random() + x/192.0, reverse=True)
        else:
            order = sorted(order, reverse=True)
            
        res = _greedy(order, deadline)
        if len(res) > best_len:
            _update_best(res)
            # Immediate local search
            if time.time() < deadline:
                pool = [x for x in all_nums if x not in set(res)]
                res_ls = _local_search(res, pool, deadline, max_iters=50)
                _update_best(res_ls)
                
    return best
