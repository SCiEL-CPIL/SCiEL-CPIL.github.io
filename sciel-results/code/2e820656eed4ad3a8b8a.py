"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _build_paley(n):
    q = n
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        mask = 0
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                mask |= (1 << j)
                adj[j] |= (1 << i)
        adj[i] = mask
    return adj

def _find_triangles(mask, adj, n):
    """Find all triangles in the subgraph induced by 'mask'."""
    triangles = set()
    verts = [i for i in range(n) if (mask >> i) & 1]
    for u in verts:
        neigh_u = adj[u] & mask
        for v_idx in range(len(verts)):
            v = verts[v_idx]
            if v <= u: continue
            if not ((adj[u] >> v) & 1): continue
            common = adj[u] & adj[v] & mask
            while common:
                w_bit = common & -common
                w = w_bit.bit_length() - 1
                tri_mask = (1 << u) | (1 << v) | (1 << w)
                triangles.add(tri_mask)
                common ^= w_bit
    return list(triangles)

def _find_indep_triangles(mask, adj, n):
    """Find all independent triangles in the subgraph induced by 'mask'."""
    full_mask = (1 << n) - 1
    comp_adj = [0] * n
    for i in range(n):
        comp_adj[i] = (~adj[i]) & full_mask
    return _find_triangles(mask, comp_adj, n)

def _compute_4_sets(adj, n):
    """Compute all 4-cliques and 4-independent sets."""
    c4 = set()
    i4 = set()
    full_mask = (1 << n) - 1
    
    all_tris = _find_triangles(full_mask, adj, n)
    for tri in all_tris:
        verts = []
        t = tri
        while t:
            b = t & -t
            v = b.bit_length() - 1
            verts.append(v)
            t ^= b
        if len(verts) < 3: continue
        u, v, w = verts
        common_neigh = adj[u] & adj[v] & adj[w] & full_mask
        common_neigh &= ~tri
        while common_neigh:
            b = common_neigh & -common_neigh
            x = b.bit_length() - 1
            c4.add(tri | (1 << x))
            common_neigh ^= b
            
    comp_adj = [0] * n
    for i in range(n):
        comp_adj[i] = (~adj[i]) & full_mask
    all_ind_tris = _find_triangles(full_mask, comp_adj, n)
    for tri in all_ind_tris:
        verts = []
        t = tri
        while t:
            b = t & -t
            v = b.bit_length() - 1
            verts.append(v)
            t ^= b
        if len(verts) < 3: continue
        u, v, w = verts
        common_neigh = comp_adj[u] & comp_adj[v] & comp_adj[w] & full_mask
        common_neigh &= ~tri
        while common_neigh:
            b = common_neigh & -common_neigh
            x = b.bit_length() - 1
            i4.add(tri | (1 << x))
            common_neigh ^= b
            
    return list(c4), list(i4)

def _try_add_vertex(adj, n, c4_list, i4_list):
    """Try to add a new vertex using backtracking on its neighbor mask."""
    def backtrack(idx, S, Z):
        if idx == n:
            return S
        
        for C in c4_list:
            if (C & S) == C:
                return None
        
        for I in i4_list:
            if (I & Z) == I:
                return None
        
        new_S = S | (1 << idx)
        res = backtrack(idx + 1, new_S, Z)
        if res is not None:
            return res
            
        new_Z = Z | (1 << idx)
        res = backtrack(idx + 1, S, new_Z)
        if res is not None:
            return res
            
        return None

    return backtrack(0, 0, 0)

def construct():
    start_time = time.time()
    
    n = 37
    adj = _build_paley(n)
    c4_list, i4_list = _compute_4_sets(adj, n)
    
    for _ in range(5):
        if time.time() - start_time > 250:
            break
            
        S = _try_add_vertex(adj, n, c4_list, i4_list)
        if S is None:
            break
            
        adj.append(0)
        for i in range(n):
            if (S >> i) & 1:
                adj[i] |= (1 << n)
                adj[n] |= (1 << i)
        
        n += 1
        
        new_c4 = []
        tris_in_S = _find_triangles(S, adj, n-1)
        for tri in tris_in_S:
            new_c4.append(tri | (1 << (n-1)))
            
        full_old_mask = (1 << (n-1)) - 1
        Z = full_old_mask & ~S
        new_i4 = []
        ind_tris_in_Z = _find_indep_triangles(Z, adj, n-1)
        for tri in ind_tris_in_Z:
            new_i4.append(tri | (1 << (n-1)))
            
        c4_list.extend(new_c4)
        i4_list.extend(new_i4)
        
    edges = []
    for i in range(n):
        mask = adj[i]
        for j in range(i + 1, n):
            if (mask >> j) & 1:
                edges.append((i, j))
                
    return n, edges

# EVOLVE-BLOCK-END
