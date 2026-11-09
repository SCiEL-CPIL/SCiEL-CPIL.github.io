"""Triangle-free graph with alpha < 10, maximizing n.
Combines Cayley graph constructions with structural local search:
initializes with a maximal triangle-free graph, then iteratively
removes edges from smallest independent sets to shrink alpha."""
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
        if bnb(cand & ~(1 << best_v) & ~adj[best_v], sz + 1):
            return True
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
    for s in S_set:
        if (-s) % n not in S_set and s != 0:
            return False
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
        
        # Modulo-based sum-free sets
        for m in range(3, n):
            for r_size in range(1, m // 2 + 1):
                for start in range(m):
                    R = set((start + i) % m for i in range(r_size))
                    S = set()
                    for s in range(1, n):
                        if s % m in R:
                            S.add(s)
                    sym = True
                    for s in S:
                        if (-s) % n not in S and s != 0:
                            sym = False
                            break
                    if not sym:
                        continue
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
        
        if n % 7 == 0:
            S = set(s for s in range(1, n) if s % 7 in (1, 2, 3))
            if _check_sum_free_symmetric(n, S):
                candidates.append(S)
        
        if n % 5 == 0:
            S = set(s for s in range(1, n) if s % 5 in (1, 2))
            if _check_sum_free_symmetric(n, S):
                candidates.append(S)
        
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

def _make_maximal_triangle_free(n, seed_val):
    """Create a random maximal triangle-free graph."""
    rng = random.Random(seed_val)
    adj = [0] * n
    
    # Start with a sparse random graph
    p = 0.1
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Remove triangles
    for _ in range(10000):
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
    
    # Make maximal: try adding edges that don't create triangles
    for _ in range(500):
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    # Check if adding edge creates triangle
                    common = adj[i] & adj[j]
                    if common == 0:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        improved = True
        if not improved:
            break
    
    return adj

def _structural_local_search(n, deadline):
    """
    Structural local search: start from maximal triangle-free graph,
    iteratively find small independent sets and remove edges within them
    to shrink the independence number.
    """
    best = None
    
    for attempt in range(300):
        if time.time() > deadline:
            break
        
        seed_val = attempt * 31337 + 7
        adj = _make_maximal_triangle_free(n, seed_val)
        
        if not _is_triangle_free(n, adj):
            continue
        
        # Iteratively shrink independence number
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            if not _has_independent_set_size(n, adj, 10):
                edges = _edges_from_adj(n, adj)
                if best is None or len(edges) > len(best['edges']):
                    best = {'n': n, 'edges': edges}
                break
            
            # Find a greedy independent set
            iset = []
            cand = (1 << n) - 1
            while cand:
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
                iset.append(best_v)
                cand &= ~(1 << best_v)
                cand &= ~adj[best_v]
            
            if len(iset) < 10:
                # Already no independent set of size 10
                edges = _edges_from_adj(n, adj)
                if best is None or len(edges) > len(best['edges']):
                    best = {'n': n, 'edges': edges}
                break
            
            # Remove edges within the independent set to break it
            # Strategy: for each pair in iset, remove edge if it exists
            # But independent set means no edges exist between them!
            # So we need to ADD edges? No, we can't add edges (triangle-free).
            # Instead, we should remove edges that connect iset vertices to others,
            # which forces the independent set to be smaller.
            
            # Better strategy: pick two vertices in iset and remove an edge
            # that allows them to be connected, thereby breaking the independent set.
            # Actually, we need to add edges between non-adjacent pairs in iset.
            # But adding edges may create triangles.
            
            # Alternative: remove edges incident to iset vertices to reduce
            # their connectivity, making it harder to form large independent sets.
            
            # Let's try: for a random pair (u, v) in iset, try to add edge u-v
            # if it doesn't create a triangle.
            u, v = iset[0], iset[1]
            if not (adj[u] & (1 << v)):
                common = adj[u] & adj[v]
                if common == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    # Verify still triangle-free
                    if not _is_triangle_free(n, adj):
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
        
        # After iterations, check if we have a valid graph
        if not _has_independent_set_size(n, adj, 10):
            if _is_triangle_free(n, adj):
                edges = _edges_from_adj(n, adj)
                if best is None or len(edges) > len(best['edges']):
                    best = {'n': n, 'edges': edges}
    
    return best

def _local_search(n, deadline):
    """Local search: start from random triangle-free graph, optimize edge density."""
    best = None
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        random.seed(attempt * 7919 + 13)
        
        adj = [0] * n
        p = 0.15
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
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
            break
    
    return best

def construct():
    """Construct a triangle-free graph with no independent set of size 10, maximizing n."""
    start_time = time.time()
    deadline = start_time + 115
    
    # Try Cayley constructions first (deterministic, fast)
    result = _try_cayley_constructions(deadline)
    if result and result['n'] >= 39:
        return result
    
    # Try structural local search for high n
    for n in range(40, 30, -1):
        if time.time() > deadline:
            break
        if result and result['n'] >= n:
            continue
        res = _structural_local_search(n, deadline)
        if res:
            if result is None or res['n'] > result['n']:
                result = res
            if res['n'] >= 39:
                return result
    
    # Try standard local search
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
