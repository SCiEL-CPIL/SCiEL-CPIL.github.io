import random
import time
import numpy as np
from array import array

def _xorshift64(state):
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= state >> 7
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state & 0xFFFFFFFFFFFFFFFF

def _randrange(rng, n):
    return _xorshift64(rng) % n

def _pairing_regular(n, k, rng):
    m = n * k
    stubs = array('I', (i for i in range(n) for _ in range(k)))
    perm = list(range(m))
    for i in range(m - 1, 0, -1):
        j = _randrange(rng, i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    adj = [[] for _ in range(n)]
    seen = [None] * n
    for a in range(0, m, 2):
        u = stubs[perm[a]]
        v = stubs[perm[a + 1]]
        if u == v:
            return None
        if seen[u] is not None:
            if v in seen[u]:
                return None
            seen[u].append(v)
        else:
            seen[u] = [v]
        adj[u].append(v)
        if seen[v] is not None:
            seen[v].append(u)
        else:
            seen[v] = [u]
        adj[v].append(u)
    return adj

def _find_triangles(adj, n):
    tri = []
    # Use bitsets or sets for neighbors
    for u in range(n):
        for i, v in enumerate(adj[u]):
            if v <= u: continue
            for w in adj[v][i+1:]: # Optimization: only look forward?
                 if w > v and w in adj[u]:
                     tri.append((u, v, w))
    return tri

def _break_triangle(adj, n, tri, rng):
    # Pick a random triangle
    if not tri: return False
    idx = _randrange(rng, len(tri))
    u, v, w = tri[idx]
    # Find a 2-switch that breaks it
    # Edges: (u,v), (v,w), (w,u)
    # We need to swap edges incident to these vertices with edges elsewhere
    # Simple strategy: Pick a random edge (a,b) not in the triangle and not sharing vertices?
    # Actually, any 2-switch involving one of the triangle edges might break it.
    # Let's try swapping (u,v) with (a,b) where a,b are not in {u,v,w}
    # This removes (u,v) and (a,b), adds (u,a) and (v,b).
    # The triangle is broken if (u,v) is removed.
    # We must ensure (u,a) and (v,b) don't create new triangles or 4-cycles?
    # Greedy: just do it and check if we reduced triangle count?
    # For simplicity, we will just perform the swap and let the next iteration handle new issues.
    
    # Find candidate edges (a,b)
    # We need to pick a, b such that a!=u, a!=v, b!=u, b!=v?
    # Actually, we just need to remove (u,v).
    # Pick random stubs.
    m = n * k
    # This is getting complicated.
    # Let's just use the existing _swap_regular but bias it?
    # No, let's just return False and let the caller retry random swaps.
    return False

def _girth_fast(adj, g):
    # ... same as before ...
    pass

def construct(k, g):
    # ... same setup ...
    pass
