import time
import numpy as np

def _has_k222_fast(adj_bits, n):
    """Fast K2,2,2 check using bitmasks."""
    for a in range(n):
        for b in range(a + 1, n):
            common_ab = adj_bits[a] & adj_bits[b]
            common_ab &= ~(1 << a)
            common_ab &= ~(1 << b)
            
            if common_ab.bit_count() < 4:
                continue
            
            cn_list = []
            mask = common_ab
            while mask:
                lsb = mask & (-mask)
                idx = lsb.bit_length() - 1
                cn_list.append(idx)
                mask ^= lsb
            
            m = len(cn_list)
            for ci in range(m):
                c = cn_list[ci]
                for di in range(ci + 1, m):
                    d = cn_list[di]
                    common_cd = adj_bits[c] & adj_bits[d]
                    intersection = common_ab & common_cd
                    intersection &= ~(1 << c)
                    intersection &= ~(1 << d)
                    
                    if intersection.bit_count() >= 2:
                        return True
    return False

def _check_validity_fast(colors, n):
    """Check validity using bitmask representation."""
    adj0 = [0] * n
    adj1 = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c == 0:
                adj0[i] |= (1 << j)
                adj0[j] |= (1 << i)
            else:
                adj1[i] |= (1 << j)
                adj1[j] |= (1 << i)
            idx += 1
    return not _has_k222_fast(adj0, n) and not _has_k222_fast(adj1, n)

def _build_paley(n):
    """Build Paley coloring for prime n ≡ 1 (mod 4)."""
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            colors.append(0 if diff in squares else 1)
    return colors

def _build_paley_subgraph(prime, remove_vertex):
    """Build Paley coloring for K_prime with one vertex removed."""
    squares = set()
    for x in range(1, prime):
        squares.add((x * x) % prime)
    
    vertices = [v for v in range(prime) if v != remove_vertex]
    n = len(vertices)
    
    colors = []
    for i_idx in range(n):
        for j_idx in range(i_idx + 1, n):
            old_i = vertices[i_idx]
            old_j = vertices[j_idx]
            diff = (old_j - old_i) % prime
            colors.append(0 if diff in squares else 1)
    return colors

def _try_extend_29_to_30(colors29, n29=29, deadline=None):
    """Try to extend a 29-vertex coloring to 30 vertices using constraint propagation."""
    if deadline is None:
        deadline = time.time() + 10
    
    # Build adjacency for the 29-vertex graph
    adj0 = [0] * n29
    adj1 = [0] * n29
    idx = 0
    for i in range(n29):
        for j in range(i + 1, n29):
            c = colors29[idx]
            if c == 0:
                adj0[i] |= (1 << j)
                adj0[j] |= (1 << i)
            else:
                adj1[i] |= (1 << j)
                adj1[j] |= (1 << i)
            idx += 1
    
    # Precompute for each pair (u,v) the set of common neighbors in each color
    # For extending with new vertex v=29, we need to check if adding edges
    # from 29 to existing vertices creates a K2,2,2
    
    # Key insight: A K2,2,2 involving new vertex 29 must have 29 in one of the three pairs.
    # If 29 is paired with vertex i in color c, then we need two more pairs (j,k) and (l,m)
    # from {0..28}\{i} such that all edges between {29,i}, {j,k}, {l,m} are color c.
    # This means: edges (29,j),(29,k),(29,l),(29,m) must be color c,
    # edges (i,j),(i,k),(i,l),(i,m) must be color c,
    # and edges (j,k),(j,l),(j,m),(k,l),(k,m),(l,m) must be color c.
    
    # So for each color c and each vertex i, if edge(29,i)=c, then we cannot have
    # two disjoint pairs (j,k) and (l,m) from N_c(i) \ {i} where all mutual edges are color c.
    # I.e., the subgraph induced by N_c(i) \ {i} in color c must be K2,2-free.
    
    # Let's precompute for each vertex i and color c, the set of vertices j such that edge(i,j)=c
    # Then for each i, check if the color-c subgraph on N_c(i) contains K2,2
    
    # For a given assignment of new vertex edges, check validity
    def _check_extension(new_edges):
        """Check if adding vertex 29 with given edge colors is valid."""
        full_adj0 = [0] * 30
        full_adj1 = [0] * 30
        
        for i in range(n29):
            for j in range(i + 1, n29):
                idx2 = i * (n29 - 1) - i * (i + 1) // 2 + (j - i - 1)
                c = colors29[idx2]
                if c == 0:
                    full_adj0[i] |= (1 << j)
                    full_adj0[j] |= (1 << i)
                else:
                    full_adj1[i] |= (1 << j)
                    full_adj1[j] |= (1 << i)
        
        for i in range(n29):
            c = new_edges[i]
            if c == 0:
                full_adj0[29] |= (1 << i)
                full_adj0[i] |= (1 << 29)
            else:
                full_adj1[29] |= (1 << i)
                full_adj1[i] |= (1 << 29)
        
        return not _has_k222_fast(full_adj0, 30) and not _has_k222_fast(full_adj1, 30)
    
    # Systematic search: try all 2^29 is too many. Instead, use the constraint that
    # for each i, the color-c neighborhood of i (including 29 if edge(29,i)=c) 
    # must not contain K2,2 in color c.
    
    # Greedy + backtracking with pruning
    import random
    random.seed(12345)
    
    # Try a few structured approaches
    # Approach 1: For each possible color of edge(29,0), try to complete greedily
    for seed_color in [0, 1]:
        if time.time() > deadline:
            break
        new_edges = [None] * n29
        new_edges[0] = seed_color
        
        # Greedily assign remaining edges
        success = True
        for i in range(1, n29):
            if time.time() > deadline:
                success = False
                break
            
            # Try both colors
            found = False
            for c in [0, 1]:
                new_edges[i] = c
                # Quick check: does this create an immediate K2,2,2?
                # We can do a partial check by seeing if any K2,2,2 is fully determined
                # For now, just track and do full check at the end
                found = True
                break
            
            if not found:
                success = False
                break
        
        if success and new_edges.count(None) == 0:
            if _check_extension(new_edges):
                # Build full color list
                full_colors = []
                for i in range(30):
                    for j in range(i + 1, 30):
                        if j == 29:
                            full_colors.append(new_edges[i])
                        else:
                            idx2 = i * (n29 - 1) - i * (i + 1) // 2 + (j - i - 1)
                            full_colors.append(colors29[idx2])
                return full_colors
    
    # Approach 2: Random restarts with constraint checking
    for attempt in range(5000):
        if time.time() > deadline:
            break
        
        new_edges = [random.randint(0, 1) for _ in range(n29)]
        
        if _check_extension(new_edges):
            full_colors = []
            for i in range(30):
                for j in range(i + 1, 30):
                    if j == 29:
                        full_colors.append(new_edges[i])
                    else:
                        idx2 = i * (n29 - 1) - i * (i + 1) // 2 + (j - i - 1)
                        full_colors.append(colors29[idx2])
            return full_colors
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    # Start with Paley graph on 29 vertices
    p29 = 29
    colors29 = _build_paley(p29)
    
    # Verify
    if not _check_validity_fast(colors29, p29):
        p17 = 17
        colors17 = _build_paley(p17)
        return {'n': p17, 'colors': colors17}
    
    best_n = p29
    best_colors = colors29[:]
    
    # Try Paley on 31 with vertex removed - this is the most promising path to n=30
    p31 = 31
    for remove in range(p31):
        if time.time() > deadline - 5:
            break
        
        colors30 = _build_paley_subgraph(p31, remove)
        
        if _check_validity_fast(colors30, 30):
            return {'n': 30, 'colors': colors30}
    
    # Try extending 29 to 30 with constraint-based search
    if time.time() < deadline - 10:
        result = _try_extend_29_to_30(colors29, deadline=deadline - 5)
        if result is not None:
            return {'n': 30, 'colors': result}
    
    return {'n': best_n, 'colors': best_colors}
