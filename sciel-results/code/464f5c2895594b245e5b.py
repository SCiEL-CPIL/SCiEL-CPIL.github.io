"""Direct (5,5)-free construction: Paley family over prime powers q ≡ 1 (mod 4),
with in-program verification (no K5, no independent 5-set) via bitset clique search.
Extended with greedy vertex-extension to try to exceed 37 vertices."""
import time
import random

# EVOLVE-BLOCK-START

def _popcount(x):
    try:
        return x.bit_count()
    except AttributeError:
        return bin(x).count('1')

def _primes_upto(n):
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]

def _paley_edges(q):
    residues = {(x * x) % q for x in range(1, q)}
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.append((i, j))
    return edges

def _adj_bitsets(n, edges):
    adj = [0] * n
    for i, j in edges:
        adj[i] |= (1 << j)
        adj[j] |= (1 << i)
    return adj

def _has_clique(adj, n, k, node_limit):
    """Return True if a k-clique exists, False if proven absent, None if budget hit."""
    full = (1 << n) - 1
    nodes = [0]

    def dfs(cand, size):
        if size == k:
            return True
        if nodes[0] > node_limit:
            return None
        need = k - size
        if _popcount(cand) < need:
            return False
        while cand:
            lsb = cand & (-cand)
            v = lsb.bit_length() - 1
            cand ^= lsb
            nodes[0] += 1
            nxt = cand & adj[v]
            r = dfs(nxt, size + 1)
            if r is True:
                return True
            if r is None:
                return None
        return False

    return dfs(full, 0)

def _is_55_free(n, edges, node_limit=400000):
    """True: verified no K5 and no independent 5-set; False: violated; None: unknown."""
    adj = _adj_bitsets(n, edges)
    full = (1 << n) - 1
    comp = [full ^ (1 << i) ^ adj[i] for i in range(n)]
    r1 = _has_clique(adj, n, 5, node_limit)
    if r1 is None:
        return None
    if r1:
        return False
    r2 = _has_clique(comp, n, 5, node_limit)
    if r2 is None:
        return None
    if r2:
        return False
    return True

def _find_all_k4(adj, n):
    """Find all 4-cliques in the graph. Returns list of frozensets."""
    k4s = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                for k in range(j + 1, n):
                    if adj[i] & (1 << k) and adj[j] & (1 << k):
                        for l in range(k + 1, n):
                            if adj[i] & (1 << l) and adj[j] & (1 << l) and adj[k] & (1 << l):
                                k4s.append(frozenset([i, j, k, l]))
    return k4s

def _find_all_i4(adj, n):
    """Find all independent 4-sets. Returns list of frozensets."""
    full = (1 << n) - 1
    comp = [full ^ (1 << i) ^ adj[i] for i in range(n)]
    i4s = []
    for i in range(n):
        for j in range(i + 1, n):
            if comp[i] & (1 << j):
                for k in range(j + 1, n):
                    if comp[i] & (1 << k) and comp[j] & (1 << k):
                        for l in range(k + 1, n):
                            if comp[i] & (1 << l) and comp[j] & (1 << l) and comp[k] & (1 << l):
                                i4s.append(frozenset([i, j, k, l]))
    return i4s

def _try_extend(n, edges, k4s, i4s, deadline):
    """Try to extend the graph by one vertex. Returns (new_n, new_edges, new_k4s, new_i4s) or None."""
    if time.time() > deadline:
        return None
    
    adj = _adj_bitsets(n, edges)
    full = (1 << n) - 1
    
    # For each possible new vertex, we need to choose a subset S of [0, n)
    # such that:
    # 1. No K4 in S (i.e., no K4 is fully contained in S)
    # 2. No I4 in V\S (i.e., no I4 is fully contained in V\S)
    
    # Strategy: try random subsets S
    for attempt in range(200):
        if time.time() > deadline:
            return None
        
        # Random subset: each vertex included with prob 0.5
        S = set()
        for v in range(n):
            if random.random() < 0.5:
                S.add(v)
        
        # Check condition 1: no K4 fully in S
        ok1 = True
        for k4 in k4s:
            if k4.issubset(S):
                ok1 = False
                break
        if not ok1:
            continue
        
        # Check condition 2: no I4 fully in V\S
        V_S = set(range(n)) - S
        ok2 = True
        for i4 in i4s:
            if i4.issubset(V_S):
                ok2 = False
                break
        if not ok2:
            continue
        
        # Valid extension found!
        new_n = n + 1
        new_edges = list(edges)
        for v in S:
            new_edges.append((v, new_n))
        
        # Update k4s and i4s
        new_k4s = []
        for k4 in k4s:
            new_k4s.append(k4)
        # New K4s involving the new vertex: any 3-clique in S
        # But we only need to track K4s for future extensions
        # For now, just add the new vertex's connections
        # Actually, we need to recompute or update carefully
        # Let's just mark that we need to recompute
        # For simplicity, return the new graph and let the caller recompute if needed
        # But that's expensive. Let's just return and let the main loop recompute.
        
        return (new_n, new_edges, None, None)  # None means recompute
    
    return None

def construct():
    start = time.time()
    deadline = start + 280.0  # safety margin before 300s kill

    primes = _primes_upto(200)
    q_set = set()
    for p in primes:
        if p % 4 == 1:
            q_set.add(p)
        q = p * p
        if q <= 200 and q % 4 == 1:
            q_set.add(q)
    # Always include the verified seed size as a guaranteed fallback.
    q_set.add(37)
    # Largest first: the first verified (5,5)-free one is the best.
    q_candidates = sorted(q_set, reverse=True)

    best = None
    for q in q_candidates:
        if time.time() > deadline:
            break
        edges = _paley_edges(q)
        res = _is_55_free(q, edges)
        if res is True:
            best = (q, edges)
            break
        # res False (violates) or None (budget) -> skip, continue to smaller q.

    if best is None:
        # Guaranteed valid fallback: Paley graph on 37 vertices.
        q = 37
        best = (q, _paley_edges(q))

    # Try to extend the best graph by adding vertices
    n, edges = best
    if time.time() < deadline - 10.0:  # need at least 10s for verification
        # Compute K4s and I4s for the current graph
        adj = _adj_bitsets(n, edges)
        k4s = _find_all_k4(adj, n)
        i4s = _find_all_i4(adj, n)
        
        # Try to extend one vertex at a time
        for _ in range(10):  # try up to 10 extensions
            if time.time() > deadline - 10.0:
                break
            result = _try_extend(n, edges, k4s, i4s, deadline)
            if result is None:
                break
            new_n, new_edges, _, _ = result
            # Verify the new graph
            res = _is_55_free(new_n, new_edges)
            if res is True:
                n, edges = new_n, new_edges
                # Recompute K4s and I4s for the new graph
                adj = _adj_bitsets(n, edges)
                k4s = _find_all_k4(adj, n)
                i4s = _find_all_i4(adj, n)
            else:
                break
    
    return (n, edges)

# EVOLVE-BLOCK-END
