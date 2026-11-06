"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import sys

def _has_c5(n, adj):
    """Check if graph has a 5-cycle using adjacency matrix (numpy not needed, use sets)."""
    # For each vertex, look for two non-adjacent neighbors that share a common neighbor
    # Actually, a 5-cycle v0-v1-v2-v3-v4-v0 exists iff there are vertices
    # a, b, c, d, e with edges ab, bc, cd, de, ea and all distinct.
    # Efficient check: for each pair of non-adjacent vertices (u, v),
    # check if there exist x, y such that u-x-y-v and x-y are edges, 
    # and u-x, x-y, y-v edges, with u,v,x,y distinct and no edge u-v.
    # Actually simpler: a 5-cycle exists iff there is a path of length 4 between
    # two non-adjacent vertices where the internal vertices are distinct.
    # 
    # Let's use: for each vertex v, look at its neighbors. For each pair of
    # non-adjacent neighbors (a, b) of v, check if there's a path of length 2
    # from a to b not going through v, i.e., a-w-b where w != v.
    # Then v-a-w-b-v would be a 4-cycle, not 5.
    #
    # Better: 5-cycle = v0-v1-v2-v3-v4-v0. 
    # Fix v0. Then v1, v4 are neighbors of v0. v2 is neighbor of v1 (not v0).
    # v3 is neighbor of v2 (not v0, v1). v4 must be neighbor of v3 and v0.
    #
    # Let me just do a direct search for small n.
    for v0 in range(n):
        nbs0 = list(adj[v0])
        for v1 in nbs0:
            for v4 in nbs0:
                if v4 <= v1:
                    continue
                # v0-v1-...-v4-v0
                # Need v2 neighbor of v1, not v0, not v4
                # Need v3 neighbor of v2 and v4, not v0, v1
                for v2 in adj[v1]:
                    if v2 == v0 or v2 == v4:
                        continue
                    for v3 in adj[v2]:
                        if v3 == v0 or v3 == v1 or v3 == v4:
                            continue
                        if v3 in adj[v4]:
                            return True
    return False

def _alpha_ge(n, adj, k=9):
    """Check if there's an independent set of size k using Bron-Kerbosch on complement."""
    # Independent set in G = clique in complement
    # Complement adjacency: non-adjacent in G (and not self)
    comp = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u+1, n):
            if v not in adj[u]:
                comp[u].add(v)
                comp[v].add(u)
    
    # Bron-Kerbosch to find if clique of size k exists
    found = [False]
    
    def bk(R, P, X):
        if found[0]:
            return
        if len(R) + len(P) < k:
            return
        if len(R) >= k:
            found[0] = True
            return
        if not P:
            return
        # Pick pivot from P ∪ X with max degree in P
        pivot_set = P | X
        pivot = max(pivot_set, key=lambda x: len(P & comp[x]))
        candidates = P - comp[pivot]
        for v in list(candidates):
            new_R = R | {v}
            new_P = P & comp[v]
            new_X = X & comp[v]
            P.remove(v)
            X.add(v)
            bk(new_R, new_P, new_X)
            P.add(v)
            X.remove(v)
            if found[0]:
                return
    
    bk(set(), set(range(n)), set())
    return found[0]

def _is_valid(n, adj):
    """Check no C5 and no independent set of size 9."""
    if _has_c5(n, adj):
        return False
    if _alpha_ge(n, adj, 9):
        return False
    return True

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _try_random_search(n, time_limit, rng):
    """Try random graphs with various edge probabilities to find valid graph on n vertices."""
    best = None
    start = time.time()
    
    # Try a range of edge densities
    densities = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    
    for p in densities:
        while time.time() - start < time_limit:
            edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if rng.random() < p:
                        edges.append([u, v])
            adj = _build_adj(n, edges)
            if _is_valid(n, adj):
                return edges
    return None

def _try_local_search(n, time_limit, rng):
    """Local search: start from random graph, flip edges to satisfy constraints."""
    start = time.time()
    
    # Start with moderate density
    p0 = 0.5
    edges_set = set()
    for u in range(n):
        for v in range(u+1, n):
            if rng.random() < p0:
                edges_set.add((u, v))
    
    adj = _build_adj(n, [(u, v) for u, v in edges_set])
    
    def has_c5_now():
        return _has_c5(n, adj)
    
    def alpha_now():
        return _alpha_ge(n, adj, 9)
    
    # Try to fix violations by flipping edges
    for iteration in range(5000):
        if time.time() - start > time_limit:
            break
        
        c5_viol = has_c5_now()
        alpha_viol = alpha_now()
        
        if not c5_viol and not alpha_viol:
            return [(u, v) for u, v in edges_set]
        
        # Pick a random edge to flip
        all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
        u, v = rng.choice(all_pairs)
        
        if (u, v) in edges_set:
            # Remove edge
            edges_set.remove((u, v))
            adj[u].discard(v)
            adj[v].discard(u)
        else:
            # Add edge
            edges_set.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
    
    return None

def _try_structural(n, time_limit, rng):
    """Try structured constructions."""
    start = time.time()
    
    # Try blow-ups of smaller C5-free graphs
    # K4 is C5-free with alpha 1
    # Disjoint union of K4s: n=4k, alpha=k
    # For n=33, we can't do exact multiples
    
    # Try: some number of K4s plus extra vertices
    # 8 K4s = 32 vertices, alpha 8. Need to add 1 more vertex.
    # Connect the 33rd vertex to at least 1 vertex in each K4 to reduce alpha.
    # But we need alpha < 9, so with 8 K4s (alpha 8) plus 1 vertex,
    # if the new vertex is isolated from all, alpha = 9. Bad.
    # If connected to at least 1 in each K4, alpha stays 8.
    
    # Let's try 8 K4s + 1 vertex connected to one in each K4
    if n == 33:
        edges = []
        # 8 K4s: vertices 0-31
        for b in range(8):
            base = 4 * b
            for i in range(4):
                for j in range(i+1, 4):
                    edges.append([base+i, base+j])
        # Vertex 32: connect to one vertex in each K4
        for b in range(8):
            edges.append([32, 4*b])
        
        adj = _build_adj(n, edges)
        if _is_valid(n, adj):
            return edges
    
    # Try 7 K4s + 5 extra vertices
    if n == 33:
        edges = []
        # 7 K4s: vertices 0-27
        for b in range(7):
            base = 4 * b
            for i in range(4):
                for j in range(i+1, 4):
                    edges.append([base+i, base+j])
        # 5 extra vertices: 28-32
        # Connect them to form a K5 among themselves? No, K5 has no C5 but...
        # Actually we need the whole graph C5-free.
        # If we make the 5 extra vertices a K5, that's fine (no C5 in K5).
        for i in range(5):
            for j in range(i+1, 5):
                edges.append([28+i, 28+j])
        # Now connect extra vertices to K4s to control alpha
        # Alpha of 7 K4s is 7. With 5 extra in a K5 (alpha 1 among them),
        # total alpha could be up to 7+1=8 if we don't connect.
        # But we need to ensure no C5 is formed.
        # Connecting extra vertices to K4s might create C5s.
        # Let's just connect each extra vertex to one vertex in each K4.
        for i in range(5):
            for b in range(7):
                edges.append([28+i, 4*b])
        
        adj = _build_adj(n, edges)
        if _is_valid(n, adj):
            return edges
    
    return None

def construct():
    deadline = time.time() + 140  # Leave 10s safety margin
    
    rng = random.Random(42)
    
    # Try structural constructions first (fast)
    for n in [33, 34, 35]:
        if time.time() > deadline:
            break
        result = _try_structural(n, 5, rng)
        if result is not None:
            return {'n': n, 'edges': result}
    
    # Try random search
    for n in [33, 34, 35]:
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 2:
            break
        result = _try_random_search(n, remaining - 5, rng)
        if result is not None:
            return {'n': n, 'edges': result}
    
    # Try local search
    for n in [33, 34]:
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 2:
            break
        result = _try_local_search(n, remaining - 5, rng)
        if result is not None:
            return {'n': n, 'edges': result}
    
    # Fallback: 8 K4s = 32 vertices
    edges = [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    return {'n': 32, 'edges': edges}
# EVOLVE-BLOCK-END
