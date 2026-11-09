"""Triangle-free graph with alpha < 10, maximizing n.
Uses Cayley graph constructions with sum-free sets on Z_n,
combined with exact verification and local search refinement.
"""
import time
import random

def _is_triangle_free(n, adj):
    """Check triangle-freeness using bitmasks."""
    for i in range(n):
        a = adj[i]
        while a:
            lsb = a & (-a)
            a ^= lsb
            j = lsb.bit_length() - 1
            common = adj[j] & adj[i] & ~(1 << i)
            if common:
                return False
    return True

def _has_independent_set_size(n, adj, target=10):
    """Branch and bound: returns True if independent set of size >= target exists."""
    # Upper bound: greedy packing
    def ub(cand):
        count = 0
        mask = cand
        while mask:
            best_v = -1
            best_deg = n + 1
            m = mask
            while m:
                lsb = m & (-m)
                m ^= lsb
                v = lsb.bit_length() - 1
                deg = (adj[v] & mask).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            if best_v < 0:
                break
            count += 1
            mask &= ~(1 << best_v)
            mask &= ~adj[best_v]
        return count

    def bnb(cand, sz):
        if sz >= target:
            return True
        if cand == 0:
            return False
        if sz + ub(cand) < target:
            return False
        # Pick vertex with min degree in cand
        best_v = -1
        best_deg = n + 1
        m = cand
        while m:
            lsb = m & (-m)
            m ^= lsb
            v = lsb.bit_length() - 1
            deg = (adj[v] & cand).bit_count()
            if deg < best_deg:
                best_deg = deg
                best_v = v
        # Include
        if bnb(cand & ~(1 << best_v) & ~adj[best_v], sz + 1):
            return True
        # Exclude
        if bnb(cand & ~(1 << best_v), sz):
            return True
        return False

    return bnb((1 << n) - 1, 0)

def _edges_from_adj(n, adj):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def _build_cayley(n, S):
    """Build Cayley graph on Z_n with connection set S (must be symmetric)."""
    adj = [0] * n
    S_set = set(S)
    for i in range(n):
        for s in S_set:
            j = (i + s) % n
            if j != i:
                adj[i] |= (1 << j)
    return adj

def _check_sum_free_symmetric(n, S):
    """Check S is symmetric (-S = S mod n) and sum-free."""
    S_set = set(S)
    # Symmetric
    for s in S_set:
        if (-s) % n not in S_set and s != 0:
            return False
    # Sum-free
    for s1 in S_set:
        for s2 in S_set:
            if (s1 + s2) % n in S_set:
                return False
    return True

def _try_cayley_constructions(deadline):
    """Try Cayley graph constructions for n from 40 down to 30."""
    best = None
    
    for n in range(40, 29, -1):
        if time.time() > deadline:
            break
        
        candidates = []
        
        # Modulo-based sum-free sets: S = {s : s mod m in R}
        for m in range(3, n):
            for r_size in range(1, m // 2 + 1):
                for start in range(m):
                    R = set((start + i) % m for i in range(r_size))
                    # Check if S = {s in 1..n-1 : s mod m in R} is symmetric
                    S = set()
                    for s in range(1, n):
                        if s % m in R:
                            S.add(s)
                    # Check symmetry
                    sym = True
                    for s in S:
                        if (-s) % n not in S and s != 0:
                            sym = False
                            break
                    if not sym:
                        continue
                    # Check sum-free
                    sf = True
                    S_list = list(S)
                    for s1 in S_list:
                        for s2 in S_list:
                            if (s1 + s2) % n in S:
                                sf = False
                                break
                        if not sf:
                            break
                    if sf and len(S) >= 2:
                        candidates.append(S)
        
        # Also try specific known patterns
        # For n divisible by 7: S = {s : s mod 7 in {1,2,3}}
        if n % 7 == 0:
            S = set(s for s in range(1, n) if s % 7 in (1, 2, 3))
            if _check_sum_free_symmetric(n, S):
                candidates.append(S)
        
        # For n divisible by 5: S = {s : s mod 5 in {1,2}}
        if n % 5 == 0:
            S = set(s for s in range(1, n) if s % 5 in (1, 2))
            if _check_sum_free_symmetric(n, S):
                candidates.append(S)
        
        # Sort by size descending (more edges = smaller alpha)
        candidates.sort(key=len, reverse=True)
        
        for S in candidates[:30]:
            if time.time() > deadline:
                break
            adj = _build_cayley(n, S)
            if not _is_triangle_free(n, adj):
                continue
            if not _has_independent_set_size(n, adj, 10):
                edges = _edges_from_adj(n, adj)
                if best is None or n > best['n']:
                    best = {'n': n, 'edges': edges}
                break
    
    return best

def _local_search(n, deadline):
    """Local search: start from random triangle-free graph, optimize edge density."""
    best = None
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        random.seed(attempt * 7919 + 13)
        
        # Start with random graph, then remove edges to fix triangles
        adj = [0] * n
        p = 0.15  # Edge probability
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Iteratively remove edges to eliminate triangles
        for _ in range(5000):
            found = False
            for i in range(n):
                a = adj[i]
                while a:
                    lsb = a & (-a)
                    a ^= lsb
                    j = lsb.bit_length() - 1
                    common = adj[j] & adj[i] & ~(1 << i)
                    if common:
                        lsb2 = common & (-common)
                        k = lsb2.bit_length() - 1
                        adj[j] &= ~(1 << k)
                        adj[k] &= ~(1 << j)
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        
        if not _is_triangle_free(n, adj):
            continue
        
        if not _has_independent_set_size(n, adj, 10):
            edges = _edges_from_adj(n, adj)
            if best is None or len(edges) > len(best['edges']):
                best = {'n': n, 'edges': edges}
            break  # Found valid graph for this n
    
    return best

def construct():
    """Construct a triangle-free graph with no independent set of size 10, maximizing n."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Try Cayley constructions first (deterministic, fast)
    result = _try_cayley_constructions(deadline)
    if result and result['n'] >= 39:
        return result
    
    # Try local search for high n
    for n in range(40, 30, -1):
        if time.time() > deadline:
            break
        if result and result['n'] >= n:
            continue
        res = _local_search(n, deadline)
        if res:
            if result is None or res['n'] > result['n']:
                result = res
            if res['n'] >= 39:
                return result
    
    # Fallback: Andrásfai graph And(9) on 26 vertices
    if result is None or result['n'] < 26:
        n = 26
        edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
        result = {'n': n, 'edges': edges}
    
    return result
