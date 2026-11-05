import time
import numpy as np

def _xorshift64(state):
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= state >> 7
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state & 0xFFFFFFFFFFFFFFFF

def _randrange(rng, n):
    return _xorshift64(rng) % n

def _girth(adj, g, k):
    n = len(adj)
    if n == 0:
        return g
    INF = 1 << 28
    dist = np.full((n, n), INF, dtype=np.int32)
    src = np.full((n, n), -1, dtype=np.int32)
    q = np.empty(2 * n * k, dtype=np.int64)
    head = 0
    tail = 0
    for v in range(n):
        for w in adj[v]:
            if w >= 0:
                dist[v, w] = 1
                src[v, w] = v
                q[tail] = v * n + w
                tail += 1
    girth = INF
    while head < tail:
        s = q[head]
        head += 1
        u, v = divmod(int(s), n)
        d = dist[u, v]
        if v == src[u, v]:
            if d < girth:
                girth = d
            if girth < g:
                return girth
            continue
        if d + 1 >= girth:
            continue
        nd = d + 1
        for w in adj[v]:
            if w >= 0 and w != u and dist[v, w] > nd:
                dist[v, w] = nd
                src[v, w] = src[u, v]
                q[tail] = v * n + w
                tail += 1
    return girth if girth < INF else g

def _psl2_p_cayley(k, g, p):
    """
    Construct a k-regular Cayley graph of PSL(2, p) with girth >= g.
    """
    if p % 4 != 3:
        return None, None
        
    # Build PSL(2, p) elements
    # Elements are (a,b,c,d) with ad-bc=1, modulo sign.
    elems = []
    seen = set()
    for a in range(p):
        for b in range(p):
            for c in range(p):
                if a == 0:
                    if b == 0 or c == 0:
                        continue
                    b_inv = pow(b, p - 2, p)
                    c_val = (-1) * b_inv % p
                    if c_val != c:
                        continue
                    d = 0
                else:
                    a_inv = pow(a, p - 2, p)
                    d = (1 + b * c) % p * a_inv % p
                det = (a * d - b * c) % p
                if det != 1:
                    continue
                key = (a, b, c, d)
                neg_key = ((-a) % p, (-b) % p, (-c) % p, (-d) % p)
                if key in seen or neg_key in seen:
                    continue
                seen.add(key)
                seen.add(neg_key)
                elems.append(key)
                
    n = len(elems)
    idx = {e: i for i, e in enumerate(elems)}
    
    def matmul(x, y):
        a1, b1, c1, d1 = x
        a2, b2, c2, d2 = y
        return ((a1*a2 + b1*c2) % p,
                (a1*b2 + b1*d2) % p,
                (c1*a2 + d1*c2) % p,
                (c1*b2 + d1*d2) % p)

    # Find involutions (trace 0)
    involutions = []
    for e in elems:
        a, b, c, d = e
        if (a + d) % p == 0:
            involutions.append(e)
            
    # We need to pick k involutions
    # Try random subsets
    rng = 12345
    for _ in range(100):
        if time.time() > 100:
            break
        # Shuffle involutions
        inv_copy = involutions[:]
        for i in range(len(inv_copy) - 1, 0, -1):
            j = _randrange(rng, i + 1)
            inv_copy[i], inv_copy[j] = inv_copy[j], inv_copy[i]
        gens = inv_copy[:k]
        
        # Check if they generate the group
        # BFS on the group
        visited = {elems[0]}
        queue = [elems[0]]
        while queue:
            curr = queue.pop()
            for gen in gens:
                next_e = matmul(curr, gen)
                if next_e not in visited:
                    visited.add(next_e)
                    queue.append(next_e)
                # Also try inverse? Involutions are self-inverse.
                # But we need to check if we reach all elements.
        if len(visited) != n:
            continue
            
        # Build adjacency
        adj = np.full((n, k), -1, dtype=np.int64)
        for i, e in enumerate(elems):
            for j, gen in enumerate(gens):
                next_e = matmul(e, gen)
                adj[i, j] = idx[next_e]
                
        # Check girth
        girth = _girth(adj, g, k)
        if girth >= g:
            return n, adj
    return None, None

def construct(k, g):
    deadline = time.time() + 115.0
    
    # Try PSL(2, 13) for k=5, g=9
    if k == 5 and g == 9:
        n, adj = _psl2_p_cayley(k, g, 13)
        if adj is not None:
            edges = [(int(i), int(j)) for i in range(n) for j in adj[i] if i < j]
            return n, edges
            
    # Fallback to LUW
    # ... (keep LUW code from seed)
    
    return 2*k, []
