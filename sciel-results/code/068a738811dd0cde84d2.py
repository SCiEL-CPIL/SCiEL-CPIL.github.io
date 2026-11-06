import time

_START = time.time()

def _spf_sieve(limit):
    spf = list(range(limit + 1))
    if limit >= 0:
        spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    for i in range(2, int(limit ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

# We need to factor sums up to 20000
_SPF = _spf_sieve(20000)

# Include primes up to 113 to be safe for mask construction
PRIMES_LIST = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
P2IDX = {p: i for i, p in enumerate(PRIMES_LIST)}

def _build_masks(limit=20000):
    """
    Precompute prime factor masks for all numbers up to limit.
    mask[x] is a bitmask where bit i is set if PRIMES_LIST[i] divides x.
    If x has a prime factor not in PRIMES_LIST, mask[x] is -1 (invalid).
    """
    masks = [-1] * (limit + 1)
    masks[0] = 0
    masks[1] = 0
    
    for x in range(2, limit + 1):
        t = x
        m = 0
        ok = True
        while t > 1:
            p = _SPF[t]
            idx = P2IDX.get(p)
            if idx is None:
                ok = False
                break
            m |= 1 << idx
            while t % p == 0:
                t //= p
        if ok:
            masks[x] = m
    return masks

_MASKS = _build_masks(20000)

def _is_smooth(x, p_mask):
    m = _MASKS[x]
    if m < 0:
        return False
    return (m & ~p_mask) == 0

def _find_clique(A, p_mask, k, node_budget, deadline):
    """
    Finds a clique of size k in the graph defined by A and smoothness.
    Uses degree-based ordering for better pruning.
    """
    n = len(A)
    if n < k:
        return None
    
    # Build adjacency list as bitmasks
    adj = [0] * n
    for i in range(n):
        ai = A[i]
        for j in range(i + 1, n):
            if _is_smooth(ai + A[j], p_mask):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Precompute higher mask: higher[v] has bits set for all indices > v
    higher = [0] * n
    for v in range(n - 2, -1, -1):
        higher[v] = higher[v + 1] | (1 << (v + 1))
        
    # Compute global degrees for initial sorting heuristic if needed, 
    # but we will compute local degrees in DFS for better branching.
    
    nodes_visited = [0]
    aborted = [False]
    
    def dfs(cand_mask, need, path):
        if need == 0:
            return list(path)
        
        if time.time() > deadline:
            aborted[0] = True
            return None
            
        if cand_mask.bit_count() < need:
            return None
            
        nodes_visited[0] += 1
        if nodes_visited[0] > node_budget:
            aborted[0] = True
            return None
            
        # Get candidates and their local degrees (neighbors in cand_mask that are higher)
        # We want to try vertices with high "potential" first.
        # Potential = number of neighbors in (cand_mask & higher[v])
        
        m = cand_mask
        cands = []
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            m &= m - 1
            
            # Neighbors of v that are in cand_mask and higher than v
            nc = cand_mask & adj[v] & higher[v]
            deg = nc.bit_count()
            
            # Pruning: if deg + 1 < need, this vertex cannot lead to a solution
            if deg + 1 < need:
                continue
                
            cands.append((deg, v, nc))
            
        # Sort by degree descending to find solutions faster
        cands.sort(key=lambda x: -x[0])
        
        for deg, v, nc in cands:
            res = dfs(nc, need - 1, path + [v])
            if res is not None:
                return res
                
        return None

    initial_mask = (1 << n) - 1
    res = dfs(initial_mask, k, [])
    
    if res is not None:
        # Final verification to be absolutely sure
        vals = [A[v] for v in res]
        for i in range(k):
            for j in range(i + 1, k):
                if not _is_smooth(vals[i] + vals[j], p_mask):
                    return None
        return vals
        
    return None

def construct():
    deadline = time.time() + 80.0  # Safety margin for 90s limit
    
    # Strategy: Try small prime sets first.
    # If we find a clique with k primes, the score is seed/k.
    # We want to minimize k.
    
    # We will try k=2, 3, 4, ...
    # For each k, we generate the list of k-smooth numbers <= 10000.
    # Then we try to find a clique of size 12.
    
    # To speed up, we limit the size of the candidate pool A.
    # Smaller numbers are more likely to have smooth sums? 
    # Actually, sums of small numbers are small, and small numbers are more likely to be smooth.
    # So sorting by value (ascending) is a good heuristic.
    
    # Let's define a helper to get the mask for the first k primes
    def get_mask(k):
        m = 0
        for i in range(k):
            m |= 1 << P2IDX[PRIMES_LIST[i]]
        return m

    # Try k from 2 up to 15
    for k in range(2, 15):
        if time.time() > deadline - 10:
            break
            
        pm = get_mask(k)
        
        # Generate smooth numbers
        # Optimization: We can stop generating if we have enough candidates?
        # But we need to check sums, so we need the actual values.
        # The list of k-smooth numbers up to 10000 grows slowly.
        # k=2: ~100, k=3: ~500, k=4: ~2000, etc.
        
        smooth_nums = [x for x in range(1, 10001) if _is_smooth(x, pm)]
        
        if len(smooth_nums) < 12:
            continue
            
        # We might have too many candidates for clique search if k is large.
        # For k=2, 3, the list is small.
        # For k=4, 5, it might be larger.
        # We can limit the candidate pool to the first T numbers.
        # Which T? 
        # If we pick too few, we might miss the clique.
        # If we pick too many, the clique search is slow.
        
        # Heuristic: Start with a small T, increase if no clique found.
        # Or, just use the whole list if it's small enough.
        
        # Let's estimate the cost.
        # Clique search is exponential in k (12).
        # But with good pruning, it's fast.
        
        # Let's try with the full list first for small k.
        # For larger k, limit the list size.
        
        max_candidates = len(smooth_nums)
        if k >= 5:
            max_candidates = min(len(smooth_nums), 2000) # Limit for larger k
        elif k >= 4:
            max_candidates = min(len(smooth_nums), 1000)
            
        A = smooth_nums[:max_candidates]
        
        # Try to find clique
        # Use a sub-deadline for each attempt
        sub_deadline = min(time.time() + 5.0, deadline - 10.0)
        node_budget = 100000 # Limit DFS nodes to avoid getting stuck
        
        res = _find_clique(A, pm, 12, node_budget, sub_deadline)
        
        if res is not None:
            return res
            
        # If we didn't find it in the first max_candidates, maybe try a different subset?
        # Or just move to next k?
        # Moving to next k is safer for time.
        # But what if the clique is in the tail of the smooth numbers?
        # Unlikely, as small numbers are "denser" in terms of smooth sums.
        
    # Fallback: Return a known valid set.
    # The set [1, 2, 3, 5, 6, 9, 10, 15, 18, 25, 27, 30] uses primes 2,3,5,7,11,13,17,19,23,29,31,37,43.
    # That's 13 primes.
    # Score would be seed/13.
    # If seed is 1, score is ~0.07.
    # But we hope to find a better one.
    
    # Let's try a bit harder with a specific known good configuration if time permits.
    # Actually, the fallback should be something that definitely works.
    # The previous program's fallback was [1, 2, 3, 5, 6, 9, 10, 15, 18, 25, 27, 30].
    
    return [1, 2, 3, 5, 6, 9, 10, 15, 18, 25, 27, 30]
