"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import lru_cache

N = 12
LCM = 27720  # lcm(1..12)

def _lcm(a, b):
    return a * b // gcd(a, b)

def _build_next_tables():
    """For each divisor i in 1..12, build a table next_i[r] = smallest positive integer
    in (r, r+LCM] that is divisible by i.
    Actually we need: for a given m, the candidates for b_i are multiples of i in (m, m+L].
    The k-th multiple of i after m is: m + ((i - m % i) % i or i) ... let's think differently.
    
    For a given m and divisor i, the multiples of i in (m, m+L] are:
    m + delta where delta = ((i - 1 - m) % i) + 1, then +i, +2i, ...
    So the first multiple of i in (m, m+L] is m + d_i where d_i = ((i - 1 - m) % i) + 1 if (m % i) != i-1... 
    Let me just compute: the smallest multiple of i strictly greater than m is:
    if m % i == 0: m + i
    else: m + (i - m % i)
    Which is m + ((i - m % i) % i) + (0 if m%i==0 else 0)... 
    Simpler: next_mult = m + i - (m % i) if m % i != 0 else m + i
    = m + i - (m % i) if m % i else m + i
    Actually: m + (i - m % i) % i + (i if m % i == 0 else 0)
    Let's just use: if m % i == 0: next = m + i else: next = m + (i - m % i)
    This equals: m + i - (m % i) when m%i != 0, and m+i when m%i==0.
    Unified: m + i - (m % i) works when m%i != 0. When m%i==0, m%i=0, so m+i-0=m+i. 
    Wait: i - (m%i) when m%i=0 gives i, so m+i. When m%i=k>0, gives i-k. So m + i - (m%i) always works!
    """
    pass

def _compute_f_for_residue(r):
    """Compute f(12, m) where m ≡ r (mod LCM). Since the structure repeats, 
    we can compute for m = r (or any representative).
    
    f is the least L >= 1 such that there exist distinct b_1,...,b_12 with 
    i | b_i and m < b_i <= m + L for all i.
    
    For each i, the candidates in (m, m+L] are multiples of i: 
    m + i - (m%i), m + 2*i - (m%i), ... up to m+L.
    The number of candidates for i in (m, m+L] is floor(L / i) possibly +1 depending on offset.
    
    We need to check if there's a perfect matching. We'll binary search on L.
    """
    m = r  # representative
    
    def has_matching(L):
        # Build bipartite graph: left = divisors 1..12, right = candidate integers
        # For each i, candidates are multiples of i in (m, m+L]
        # We need to assign distinct b_i to each i.
        
        # Collect all candidates
        # For each i, the multiples in (m, m+L]:
        # first = m + i - (m % i)  [this is > m]
        # then first + i, first + 2i, ... <= m + L
        # 
        # We'll use a matching approach. Since N=12 is small, we can use 
        # a simple DFS-based bipartite matching.
        
        # Map each candidate value to an index
        # But candidates can be large. Instead, for each i, list its candidates.
        
        # For matching, we need: can we assign to each i a distinct multiple of i in (m, m+L]?
        
        # Let's collect all candidate values and build adjacency.
        # Actually, let's use the standard approach: for each divisor i, 
        # the set of candidate positions is {m + i - (m%i) + k*i : k >= 0, m + i - (m%i) + k*i <= m + L}
        
        # Let's just enumerate candidates for each i and do bipartite matching.
        # Number of candidates per i is at most L/i + 1, so for L up to ~200, 
        # total candidates is manageable.
        
        # Build candidate list for each i
        candidates = {}
        for i in range(1, N + 1):
            first = m + i - (m % i)
            if first <= m:  # shouldn't happen since i - (m%i) >= 1
                first += i
            cands = []
            v = first
            while v <= m + L:
                cands.append(v)
                v += i
            candidates[i] = cands
        
        # Now do bipartite matching: left nodes = 1..12, right nodes = all distinct candidate values
        # Actually, we can do it more directly with a DFS matching on the divisor side.
        
        # Let's use a simple approach: try to assign greedily with backtracking, or use 
        # the standard bipartite matching (Hopcroft-Karp or simple DFS).
        
        # Since N=12 is tiny, let's use a simple DFS-based augmenting path matching.
        
        # Right side: all candidate values. But there could be many. 
        # Instead, let's think of it as: for each divisor i, it can match to any of its candidates.
        # We want a system of distinct representatives.
        
        # Use Hall's theorem check? Or just do matching.
        
        # Let's collect all unique candidate values and build the graph.
        all_cands = set()
        for i in range(1, N + 1):
            all_cands.update(candidates[i])
        
        # Map candidate values to indices
        cand_list = sorted(all_cands)
        cand_to_idx = {v: idx for idx, v in enumerate(cand_list)}
        
        # Adjacency: for each divisor i, which candidate indices it can use
        adj = {}
        for i in range(1, N + 1):
            adj[i] = [cand_to_idx[v] for v in candidates[i]]
        
        # Bipartite matching: left = 1..12, right = 0..len(cand_list)-1
        match_right = [-1] * len(cand_list)
        
        def dfs(u, visited):
            for v in adj[u]:
                if visited[v]:
                    continue
                visited[v] = True
                if match_right[v] == -1 or dfs(match_right[v], visited):
                    match_right[v] = u
                    return True
            return False
        
        count = 0
        for i in range(1, N + 1):
            visited = [False] * len(cand_list)
            if dfs(i, visited):
                count += 1
        
        return count == N
    
    # Binary search on L
    # Lower bound: at least max(i) = 12? No, L must be at least enough to have candidates.
    # Actually L >= 1. Let's binary search.
    lo, hi = 1, 400  # f is at most 144 per the task, but let's be safe
    
    # Check if hi is enough
    if not has_matching(hi):
        # Try larger
        hi = 1000
        if not has_matching(hi):
            return hi  # shouldn't happen
    
    while lo < hi:
        mid = (lo + hi) // 2
        if has_matching(mid):
            hi = mid
        else:
            lo = mid + 1
    
    return lo

def _precompute_all_residues(deadline):
    """Precompute f(12, r) for all r in 0..LCM-1, with time guard."""
    results = {}
    for r in range(LCM):
        if time.time() > deadline:
            break
        results[r] = _compute_f_for_residue(r)
    return results

def construct():
    deadline = time.time() + 75  # safety margin
    
    # Precompute f for all residues
    results = _precompute_all_residues(deadline)
    
    if not results:
        return 12
    
    # Find the best residue
    best_r = max(results, key=lambda r: results[r])
    best_f = results[best_r]
    
    # Now find m in 0..1000000 with m % LCM == best_r
    # m = best_r, best_r + LCM, best_r + 2*LCM, ...
    # All have the same f value.
    # We just need any valid m. Use m = best_r (which is in 0..27719, well within range).
    
    m = best_r
    if m > 1000000:
        m = best_r % 1000000
    
    return int(m)

# EVOLVE-BLOCK-END
