import random
import sys
from collections import deque

def _girth_nonbacktracking(n, adj):
    """Exact girth via non-backtracking BFS from every vertex."""
    best = 10**9
    for s in range(n):
        dist = {}
        dq = deque()
        for nb in adj[s]:
            dist[(s, nb)] = 1
            dq.append((s, nb, 1))
        while dq:
            v, prev, d = dq.popleft()
            if d >= best:
                continue
            for nb in adj[v]:
                if nb == prev:
                    continue
                if nb == s:
                    cyc = d + 1
                    if cyc < best:
                        best = cyc
                    continue
                st = (v, nb)
                if st in dist:
                    cyc = d + dist[st]
                    if cyc < best:
                        best = cyc
                else:
                    dist[st] = d + 1
                    dq.append((v, nb, d + 1))
    return best

def _is_k_regular(adj, n, k):
    return all(len(adj[i]) == k for i in range(n))

def _build_cayley_graph(q):
    """Construct a 3-regular Cayley graph on SL_2(F_q)."""
    # Elements of SL_2(F_q) are 2x2 matrices with determinant 1.
    # We represent them as tuples (a, b, c, d) for [[a, b], [c, d]].
    # a*d - b*c == 1 mod q.
    
    elements = []
    index = {}
    for a in range(q):
        for b in range(q):
            for c in range(q):
                # d is determined by a*d - b*c = 1 => a*d = 1 + b*c
                # If a != 0, d = (1 + b*c) * a^{-1}
                # If a == 0, then -b*c = 1 => b*c = -1.
                if a != 0:
                    # Find inverse of a mod q
                    # q is prime in our simple implementation
                    inv_a = pow(a, q-2, q)
                    d = (1 + b*c) * inv_a % q
                    elements.append((a, b, c, d))
                    index[(a, b, c, d)] = len(elements) - 1
                else:
                    # a=0 => -b*c = 1 => b*c = q-1
                    if b != 0:
                        inv_b = pow(b, q-2, q)
                        c_val = (q-1) * inv_b % q
                        d = 0 # d can be anything? No, det=1.
                        # Wait, if a=0, det = -bc = 1 => bc = -1.
                        # d is free? No, the matrix is [[0, b], [c, d]].
                        # det = 0*d - b*c = -bc = 1.
                        # So d is not constrained by det=1?
                        # SL_2 requires det=1.
                        # So for a=0, we must have bc = -1. d is arbitrary?
                        # No, the matrix is determined by a,b,c,d.
                        # If a=0, we have 3 free variables b,c,d with constraint bc=-1.
                        # So we iterate b, c, d.
                        pass # Handled in the a!=0 case? No.
                        # Let's just iterate all a,b,c and solve for d if possible.
                        # If a=0, we cannot solve for d.
                        # We must handle a=0 separately.
    
    # Correct enumeration:
    elements = []
    index = {}
    for a in range(q):
        for b in range(q):
            for c in range(q):
                for d in range(q):
                    if (a*d - b*c) % q == 1:
                        elements.append((a, b, c, d))
                        index[(a, b, c, d)] = len(elements) - 1
                        
    n = len(elements)
    adj = [[] for _ in range(n)]
    
    # Generators for 3-regular graph
    # A = [[0, -1], [1, 0]]
    # B = [[1, 1], [0, 1]]
    # C = [[1, 0], [1, 1]]
    # Note: -1 mod q is q-1.
    gen_A = (0, q-1, 1, 0)
    gen_B = (1, 1, 0, 1)
    gen_C = (1, 0, 1, 1)
    
    def mat_mul(X, Y):
        a1, b1, c1, d1 = X
        a2, b2, c2, d2 = Y
        return ((a1*a2 + b1*c2) % q, (a1*b2 + b1*d2) % q, 
                (c1*a2 + d1*c2) % q, (c1*b2 + d1*d2) % q)
        
    for i, elem in enumerate(elements):
        for gen in [gen_A, gen_B, gen_C]:
            neighbor = mat_mul(elem, gen)
            j = index[neighbor]
            adj[i].append(j)
            
    # Check regularity
    if not _is_k_regular(adj, n, 3):
        return None, None
    return n, adj

def construct(k, g):
    if k != 3:
        # Fallback to random for other k
        return _random_construct(k, g)
        
    best = None
    # Try increasing q
    # Girth of these graphs grows with q.
    # For g=16, we need a sufficiently large q.
    # n(3,16) is 936.
    # |SL_2(7)| = 336. Girth is likely small.
    # |SL_2(11)| = 1320.
    # |SL_2(13)| = 2184.
    
    for q in [7, 11, 13, 17, 19]:
        if q % 2 == 0: continue # q should be odd prime for this simple impl
        
        n, adj = _build_cayley_graph(q)
        if adj is None:
            continue
            
        gr = _girth_nonbacktracking(n, adj)
        if gr >= g:
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            if best is None or n < best[0]:
                best = (n, edges)
            # If we found a valid graph, we can break if n is close to bound
            if n <= 600:
                break
                
    if best is None:
        # Fallback: Random search
        best = _random_construct(k, g)
        
    return best

def _random_construct(k, g):
    # Simple random fallback
    best = None
    for n in range(512, 1000, 8):
        if n * k % 2 != 0: continue
        for _ in range(10):
            adj = _gen_random_k_regular(n, k)
            if adj is None: continue
            gr = _girth_nonbacktracking(n, adj)
            if gr >= g:
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                if best is None or n < best[0]:
                    best = (n, edges)
                break
        if best is not None and best[0] <= 600:
            break
    return best

def _gen_random_k_regular(n, k):
    if n * k % 2 != 0: return None
    stubs = []
    for i in range(n):
        stubs.extend([i]*k)
    random.shuffle(stubs)
    adj = [[] for _ in range(n)]
    for i in range(0, len(stubs), 2):
        a, b = stubs[i], stubs[i+1]
        if a == b: continue
        if b in adj[a]: continue
        adj[a].append(b)
        adj[b].append(a)
    if any(len(adj[i]) != k for i in range(n)):
        return None
    return adj
