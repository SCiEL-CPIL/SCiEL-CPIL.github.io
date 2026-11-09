import time
import sys

def _smooth_set(limit, primes):
    """
    Generate a set of all numbers <= limit that are 'smooth' 
    (i.e., their prime factors are a subset of 'primes').
    """
    s = set()
    # We can generate these more efficiently, but for limit ~20000, 
    # checking each number is acceptable if primes list is small.
    for n in range(2, limit + 1):
        m = n
        for p in primes:
            while m % p == 0:
                m //= p
        if m == 1:
            s.add(n)
    return s

def _find_clique(cands, smooth, size, deadline):
    """
    Find a clique of given 'size' in the graph defined by 'cands' 
    where an edge exists if (cands[i] + cands[j]) is in 'smooth'.
    """
    n = len(cands)
    # Precompute adjacency list
    # adj[i] is a set of indices j > i such that edge (i,j) exists
    adj = [set() for _ in range(n)]
    
    # Optimization: Use a set for smooth numbers for O(1) lookup
    smooth_set = smooth
    
    for i in range(n):
        ai = cands[i]
        for j in range(i + 1, n):
            if (ai + cands[j]) in smooth_set:
                adj[i].add(j)
                adj[j].add(i)
                
    # Order candidates by degree (descending) to find cliques faster
    order = sorted(range(n), key=lambda i: len(adj[i]), reverse=True)
    
    best = []
    
    def bt(current, pool):
        nonlocal best
        if time.time() > deadline:
            return
        if len(current) == size:
            best = list(current)
            return
        # Pruning: if remaining pool is too small
        if len(pool) < size - len(current):
            return
            
        for k, idx in enumerate(pool):
            if time.time() > deadline:
                return
            # Check if idx connects to all in current
            if not all(idx in adj[c] for c in current):
                continue
            
            # New pool: candidates after idx in the current pool that connect to idx
            new_pool = [x for x in pool[k + 1:] if x in adj[idx]]
            
            current.append(idx)
            bt(current, new_pool)
            current.pop()
            
            if best:
                return

    bt([], order)
    return best

def _count_distinct_primes(nums):
    all_primes = set()
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            s = nums[i] + nums[j]
            d = 2
            while d * d <= s:
                while s % d == 0:
                    all_primes.add(d)
                    s //= d
                d += 1
            if s > 1:
                all_primes.add(s)
    return len(all_primes)

def construct():
    t0 = time.time()
    # Keep a safety margin. The limit is 60s (or 90s in some contexts, 
    # but the prompt says 60s for evaluation, 90s for evaluator kill).
    # We will use 50s as our internal deadline to be safe.
    deadline = t0 + 50.0
    
    # We want to minimize the number of distinct primes.
    # The smallest possible set is {2}, then {2,3}, etc.
    prime_sets = [
        [2],
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11]
    ]
    
    # We search for a clique in the graph of integers 1..N.
    # If we find a clique of size 12 for a prime set P, 
    # the number of distinct primes in the sums is <= |P|.
    # We iterate P from smallest to largest.
    
    # Start with a reasonable N. 10000 is the max allowed value.
    # However, smaller N might be faster to search if the graph is dense.
    # But we need distinct integers <= 10000.
    # Let's try N=10000 directly.
    
    N = 10000
    cands = list(range(1, N + 1))
    
    best_result = None
    best_score = float('inf')
    
    for ps in prime_sets:
        if time.time() > deadline - 5.0:
            break
            
        # Generate smooth numbers up to 2*N (max sum is 10000+9999=19999)
        smooth = _smooth_set(2 * N, ps)
        
        # If the number of smooth numbers is very low, a clique of 12 might not exist.
        # But let's just run the search.
        
        res = _find_clique(cands, smooth, 12, deadline - 5.0)
        
        if res:
            out = [cands[i] for i in res]
            if len(out) == 12 and len(set(out)) == 12:
                # Verify
                score = _count_distinct_primes(out)
                if score < best_score:
                    best_score = score
                    best_result = out
                # If we found a solution with score == len(ps), it's optimal for this ps.
                # Since we iterate ps in increasing size, the first valid ps gives the global minimum 
                # (assuming we check ps in order of size).
                # Actually, a solution for ps={2} has score 1 (or 2 if sum is 0? no, positive ints).
                # If we find a solution for ps={2}, score is 1. That's the best possible.
                if score <= len(ps):
                    return out

    if best_result:
        return best_result

    # Fallback: If no clique found (unlikely for small prime sets), 
    # return a set that likely has few primes.
    # Powers of 2 have sums that are not always powers of 2, but are smooth with {2}?
    # 2^a + 2^b = 2^b(2^(a-b)+1). If a-b=1, sum is 3*2^b (prime 3 involved).
    # So powers of 2 involve {2, 3, ...}.
    
    # A safe fallback: 1, 2, ..., 12
    return list(range(1, 13))
