"""Combined program: precomputed divisibility + structured search with pruning."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

_DEADLINE = 75
_START = time.time()

def _precompute_smooths(limit):
    smooths = []
    for x in range(1, limit + 1):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooths.append(x)
    return smooths

def _solve_target(n, smooths, divs, index_map, pool_indices_cache):
    """Find minimal antichain representation for target n."""
    # Filter smooths <= n
    pool = [x for x in smooths if x <= n]
    if not pool:
        return None
    
    pool_set = set(pool)
    m = len(pool)
    
    # k=1
    if n in pool_set:
        return [n]
        
    # k=2
    for a in pool:
        if a * 2 >= n:
            break
        b = n - a
        if b in pool_set:
            if b % a != 0:
                return [a, b]
                
    # k >= 3
    # Get indices of pool elements in the full smooths array
    pool_indices = [index_map[x] for x in pool]
    
    max_k = min(30, m)
    
    for k in range(3, max_k + 1):
        # Pruning: check if n is achievable with k distinct elements from pool
        # Min sum of k smallest
        min_sum = sum(pool[:k])
        if min_sum > n:
            break
        # Max sum of k largest
        max_sum = sum(pool[-k:])
        if max_sum < n:
            continue
        
        for combo in combinations(range(m), k):
            # Check sum with early termination
            s = 0
            for idx in combo:
                s += pool[idx]
                if s > n:
                    break
            if s != n:
                continue
            
            # Check antichain using precomputed divisibility
            valid = True
            for i in range(k):
                gi = pool_indices[combo[i]]
                for j in range(i + 1, k):
                    gj = pool_indices[combo[j]]
                    if divs[gi][gj] or divs[gj][gi]:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                return [pool[i] for i in combo]
                
    return None

def construct():
    global _START
    _START = time.time()
    
    # Precompute
    smooths = _precompute_smooths(128)
    m = len(smooths)
    index_map = {x: i for i, x in enumerate(smooths)}
    
    # Precompute divisibility matrix
    # divs[i][j] = True if smooths[i] divides smooths[j]
    divs = [[False] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if smooths[j] % smooths[i] == 0:
                divs[i][j] = True
                
    answers = [None] * 128
    
    for n in range(1, 129):
        if time.time() - _START > _DEADLINE:
            break
        ans = _solve_target(n, smooths, divs, index_map, None)
        if ans is None:
            # Fallback: should not happen for 1..128
            if n in index_map:
                ans = [n]
            else:
                # Last resort: try randomized search
                import random
                rng = random.Random(n)
                pool = [x for x in smooths if x <= n]
                found = False
                for k in range(3, min(30, len(pool)) + 1):
                    if time.time() - _START > _DEADLINE - 2:
                        break
                    for _ in range(500):
                        terms = rng.sample(pool, k)
                        if sum(terms) == n:
                            # Check antichain
                            valid = True
                            for i in range(k):
                                for j in range(i + 1, k):
                                    a, b = terms[i], terms[j]
                                    if a > b:
                                        a, b = b, a
                                    if b % a == 0:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if valid:
                                ans = sorted(terms)
                                found = True
                                break
                    if found:
                        break
                if not found:
                    ans = []
        answers[n - 1] = ans
        
    return answers
# EVOLVE-BLOCK-END
