"""Program for finding colourings of K_n without monochromatic K2,2,2."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _find_violations(adj):
    """Find all monochromatic K2,2,2 subgraphs.
    Returns a list of tuples (color, vertices_tuple).
    vertices_tuple is (a, b, c, d, e, f) representing pairs (a,b), (c,d), (e,f).
    """
    n = adj.shape[0]
    violations = []
    
    # Precompute neighbor bitmasks for color 0 and 1
    # adj[i,j] is 0 or 1
    neigh0 = [0] * n
    neigh1 = [0] * n
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if adj[i,j] == 0:
                neigh0[i] |= (1 << j)
            else:
                neigh1[i] |= (1 << j)
                
    def _get_verts(mask):
        verts = []
        m = mask
        while m:
            v = (m & -m).bit_length() - 1
            verts.append(v)
            m &= m - 1
        return verts

    # Check color 1
    for a in range(n):
        for b in range(a+1, n):
            if adj[a,b] != 1: continue
            S1 = neigh1[a] & neigh1[b]
            if bin(S1).count('1') < 4: continue
            verts1 = _get_verts(S1)
            L1 = len(verts1)
            for i in range(L1):
                c = verts1[i]
                for j in range(i+1, L1):
                    d = verts1[j]
                    if adj[c,d] != 1: continue
                    T1 = neigh1[c] & neigh1[d] & S1
                    if bin(T1).count('1') < 2: continue
                    vertsT1 = _get_verts(T1)
                    # Need two disjoint vertices e, f in T1
                    # T1 is disjoint from {a,b,c,d} automatically?
                    # a,b are not in N(a) or N(b).
                    # c,d are in S1 = N(a) & N(b).
                    # Is c in N(c)? No. So c not in T1.
                    # So T1 is disjoint from {a,b,c,d}.
                    # We just need any 2 from T1.
                    if len(vertsT1) >= 2:
                        e = vertsT1[0]
                        f = vertsT1[1]
                        violations.append((1, (a,b,c,d,e,f)))
                        
    # Check color 0
    for a in range(n):
        for b in range(a+1, n):
            if adj[a,b] != 0: continue
            S0 = neigh0[a] & neigh0[b]
            if bin(S0).count('1') < 4: continue
            verts0 = _get_verts(S0)
            L0 = len(verts0)
            for i in range(L0):
                c = verts0[i]
                for j in range(i+1, L0):
                    d = verts0[j]
                    if adj[c,d] != 0: continue
                    T0 = neigh0[c] & neigh0[d] & S0
                    if bin(T0).count('1') < 2: continue
                    vertsT0 = _get_verts(T0)
                    if len(vertsT0) >= 2:
                        e = vertsT0[0]
                        f = vertsT0[1]
                        violations.append((0, (a,b,c,d,e,f)))
                        
    return violations

def _paley_coloring(p):
    """Generate Paley coloring for prime p ≡ 1 (mod 4)."""
    squares = {x * x % p for x in range(1, p)}
    adj = np.zeros((p, p), dtype=np.int8)
    colors = []
    for i in range(p):
        for j in range(i+1, p):
            c = 1 if (j - i) % p in squares else 0
            colors.append(c)
            if c == 1:
                adj[i][j] = 1
                adj[j][i] = 1
    return adj, colors

def _extend_and_fix(adj_base, target_n, max_steps=50000, deadline=None):
    """Try to extend a valid coloring of size n_base to target_n by adding vertices and fixing violations."""
    n_base = adj_base.shape[0]
    if n_base >= target_n:
        return adj_base, True
    
    adj = np.zeros((target_n, target_n), dtype=np.int8)
    adj[:n_base, :n_base] = adj_base
    
    # Initialize new edges randomly
    for i in range(n_base, target_n):
        for j in range(i+1, target_n):
            c = random.randint(0, 1)
            adj[i,j] = c
            adj[j,i] = c
        for j in range(n_base):
            c = random.randint(0, 1)
            adj[i,j] = c
            adj[j,i] = c
            
    steps = 0
    while steps < max_steps:
        if deadline and time.time() > deadline:
            break
        violations = _find_violations(adj)
        if not violations:
            return adj, True
            
        # Pick a random violation
        v = random.choice(violations)
        color, verts = v
        a,b,c,d,e,f = verts
        
        # Candidate edges to flip: all 15 edges in the K2,2,2
        # Edges: (a,b), (c,d), (e,f) are NOT in the K2,2,2 cross edges?
        # Wait, K2,2,2 is complete tripartite.
        # Vertices are partitioned into 3 sets of size 2: {a,b}, {c,d}, {e,f}.
        # Edges in K2,2,2 are ALL edges BETWEEN these sets.
        # Total edges = 2*2 + 2*2 + 2*2 = 12.
        # The edges (a,b), (c,d), (e,f) are NOT in the K2,2,2 subgraph.
        # They are "internal" to the parts.
        # So we can only flip the 12 cross edges.
        
        cross_edges = [
            (a,c), (a,d), (b,c), (b,d),
            (a,e), (a,f), (b,e), (b,f),
            (c,e), (c,f), (d,e), (d,f)
        ]
        
        # To break the monochromatic K2,2,2 of `color`, we must flip an edge of `color` to `1-color`.
        # All 12 edges are `color`.
        # Pick one randomly.
        u, v_idx = random.choice(cross_edges)
        
        # Flip it
        adj[u, v_idx] = 1 - adj[u, v_idx]
        adj[v_idx, u] = adj[u, v_idx]
        
        steps += 1
    return adj, False

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110.0
    
    best_result = {'n': 1, 'colors': [0]}
    
    # 1. Get base Paley 29
    adj29, colors29 = _paley_coloring(29)
    # Verify base
    if not _find_violations(adj29):
        best_result = {'n': 29, 'colors': colors29}
    else:
        # Should not happen, but fallback
        best_result = {'n': 1, 'colors': [0]}
        return best_result

    # 2. Try to extend to 30
    if time.time() < DEADLINE:
        adj30, success30 = _extend_and_fix(adj29, 30, max_steps=20000, deadline=DEADLINE)
        if success30:
            # Convert adj30 to colors list
            colors30 = []
            for i in range(30):
                for j in range(i+1, 30):
                    colors30.append(int(adj30[i,j]))
            return {'n': 30, 'colors': colors30}
            
    # 3. Try to extend to 31 (if time permits, though 30 failed, 31 is harder)
    # Maybe restart from scratch for 31? Or extend 29 to 31 directly?
    # Extending 29 to 31 adds 2 vertices.
    if time.time() < DEADLINE - 5:
        adj31, success31 = _extend_and_fix(adj29, 31, max_steps=20000, deadline=DEADLINE)
        if success31:
            colors31 = []
            for i in range(31):
                for j in range(i+1, 31):
                    colors31.append(int(adj31[i,j]))
            return {'n': 31, 'colors': colors31}

    return best_result

# EVOLVE-BLOCK-END
