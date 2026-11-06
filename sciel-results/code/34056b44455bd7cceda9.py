"""Rewritten program: faster conflict checks + stronger local search."""
# EVOLVE-BLOCK-START
import time
from random import Random

def _instance(size, seed, limit=10000):
    rng = Random(seed)
    values = set()
    while len(values) < size:
        x = rng.randint(-limit, limit)
        if x != 0:
            values.add(x)
    return sorted(values)

def _is_sum_free(S):
    """O(n^2) validation."""
    s = set(S)
    for a in S:
        for b in S:
            if (a + b) in s:
                return False
    return True

def _conflicts(S_set, x):
    """Return set of elements in S_set that conflict with x (including x itself if 2x in S_set).
    O(|S_set|) time.
    """
    conf = set()
    if 2 * x in S_set:
        conf.add(x)
    for c in S_set:
        if x + c in S_set:
            conf.add(x)
            conf.add(c)
    # c1 + c2 == x
    for c1 in S_set:
        target = x - c1
        if target in S_set:
            conf.add(c1)
            conf.add(target)
            conf.add(x)
    return conf

def _greedy_order(S, order):
    """Greedy: add elements in given order if no conflict."""
    S_set = set()
    result = []
    for x in order:
        if not _conflicts(S_set, x):
            S_set.add(x)
            result.append(x)
    return result

def _local_search(S, rng, deadline):
    """Local search with add, swap, and 2-for-1 moves. Fast conflict checks."""
    all_elements = set(S)
    S_set = set(S)
    excluded = list(all_elements - S_set)
    
    best = len(S_set)
    best_set = set(S_set)
    
    if not _is_sum_free(S_set):
        # Rebuild greedily
        S_set = set(_greedy_order(list(S), sorted(S, key=lambda v: -abs(v))))
        excluded = list(all_elements - S_set)
        best = len(S_set)
        best_set = set(S_set)
    
    def try_add(x):
        nonlocal best, best_set
        if not _conflicts(S_set, x):
            S_set.add(x)
            excluded.remove(x)
            if len(S_set) > best:
                best = len(S_set)
                best_set = set(S_set)
            return True
        return False
    
    def try_swap(x, c):
        nonlocal best, best_set
        # Remove c, add x
        new_set = (S_set - {c}) | {x}
        if _is_sum_free(new_set):
            S_set = new_set
            excluded.remove(x)
            if len(S_set) > best:
                best = len(S_set)
                best_set = set(S_set)
            return True
        return False
    
    def try_2for1(x1, x2, c):
        nonlocal best, best_set
        new_set = (S_set - {c}) | {x1, x2}
        if _is_sum_free(new_set):
            S_set = new_set
            excluded.remove(x1)
            excluded.remove(x2)
            if len(S_set) > best:
                best = len(S_set)
                best_set = set(S_set)
            return True
        return False
    
    while time.time() < deadline:
        improved = False
        
        # Pass 1: add moves
        for x in list(excluded):
            if time.time() >= deadline:
                break
            if try_add(x):
                improved = True
        
        # Pass 2: swap moves (add x, remove one conflict)
        for x in list(excluded):
            if time.time() >= deadline:
                break
            conf = _conflicts(S_set, x)
            if not conf:
                # Should have been added in pass 1
                continue
            for c in list(conf):
                if c == x:
                    continue
                if c in S_set:
                    if try_swap(x, c):
                        improved = True
                        break
        
        # Pass 3: 2-for-1 moves (add two excluded, remove one member)
        if len(excluded) >= 2 and time.time() < deadline:
            excl = list(excluded)
            # Limit the search space for speed
            max_pairs = 200
            pairs_checked = 0
            for i in range(len(excl)):
                if time.time() >= deadline or pairs_checked >= max_pairs:
                    break
                x1 = excl[i]
                for j in range(i + 1, len(excl)):
                    if time.time() >= deadline or pairs_checked >= max_pairs:
                        break
                    x2 = excl[j]
                    pairs_checked += 1
                    # Try removing each member (limit to first few for speed)
                    for c in list(S_set):
                        if try_2for1(x1, x2, c):
                            improved = True
                            break
        
        if not improved:
            # Stuck: shuffle and try once more
            rng.shuffle(excluded)
            any_move = False
            for x in list(excluded):
                if time.time() >= deadline:
                    break
                if try_add(x):
                    any_move = True
            if not any_move:
                break
    
    return list(best_set)

def construct():
    deadline = time.time() + 140.0
    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    for idx, (size, seed) in enumerate(instances):
        S = _instance(size, seed)
        rng = Random(seed + 12345)
        
        # Multiple initial candidates
        cand_odd = [x for x in S if x % 2 != 0]
        cand_abs = _greedy_order(S, sorted(S, key=lambda v: -abs(v)))
        cand_desc = _greedy_order(S, sorted(S, reverse=True))
        cand_asc = _greedy_order(S, sorted(S))
        
        candidates = [cand_odd, cand_abs, cand_desc, cand_asc]
        best_init = max(candidates, key=len)
        
        # Allocate time: give more to larger instances
        remaining = deadline - time.time()
        # Proportional allocation by size
        total_size = sum(s for s, _ in instances)
        frac = size / total_size
        inst_deadline = time.time() + remaining * frac * 0.9
        
        result = _local_search(best_init, rng, inst_deadline)
        
        # Verify
        if not _is_sum_free(result):
            result = best_init
            if not _is_sum_free(result):
                result = cand_odd
        
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
