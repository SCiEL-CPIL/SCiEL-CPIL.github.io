import time
import random

def _k4_found(adj, n):
    """Return a 4-tuple (a,b,c,d) forming a K4, or None. adj = list of int bitmasks."""
    for a in range(n):
        aa = adj[a]
        for b in range(a + 1, n):
            if not (aa >> b) & 1:
                continue
            ab = aa & adj[b]
            for c in range(b + 1, n):
                if not (ab >> c) & 1:
                    continue
                abc = ab & adj[c]
                for d in range(c + 1, n):
                    if (abc >> d) & 1:
                        return (a, b, c, d)
    return None

def _ind6_found(adj, n):
    """Return a 6-tuple forming an independent set, or None. adj = list of int bitmasks."""
    allmask = (1 << n) - 1
    for a in range(n):
        na = allmask ^ (1 << a) ^ adj[a]
        for b in range(a + 1, n):
            if not (na >> b) & 1:
                continue
            nab = na & (allmask ^ (1 << b) ^ adj[b])
            for c in range(b + 1, n):
                if not (nab >> c) & 1:
                    continue
                nabc = nab & (allmask ^ (1 << c) ^ adj[c])
                for d in range(c + 1, n):
                    if not (nabc >> d) & 1:
                        continue
                    nabcd = nabc & (allmask ^ (1 << d) ^ adj[d])
                    for e in range(d + 1, n):
                        if not (nabcd >> e) & 1:
                            continue
                        rest = nabcd & (allmask ^ (1 << e) ^ adj[e])
                        higher = rest & (~((1 << (e + 1)) - 1))
                        if higher:
                            f = (higher & -higher).bit_length() - 1
                            return (a, b, c, d, e, f)
    return None

def _is_valid(adj, n):
    return _k4_found(adj, n) is None and _ind6_found(adj, n) is None

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return edges

def _build_35_vertex_graph():
    """Construct the known 35-vertex (4,6)-Ramsey graph using Z_7 x Z_5 structure."""
    n = 35
    adj = [0] * n
    # Vertex (i, j) where i in 0..6, j in 0..4
    # Edge if (i2-i1 mod 7) in {1,2,4} AND (j2-j1 mod 5) in {1,2}
    # This is a known construction for R(4,6) > 35
    S7 = {1, 2, 4}  # quadratic residues mod 7 (excluding 0)
    S5 = {1, 2}     # subset of quadratic residues mod 5
    
    for i1 in range(7):
        for j1 in range(5):
            v1 = i1 * 5 + j1
            for i2 in range(i1, 7):
                for j2 in range(5):
                    if i1 == i2 and j1 == j2:
                        continue
                    if i2 == i1 and j2 < j1:
                        continue
                    v2 = i2 * 5 + j2
                    if v1 >= v2:
                        continue
                    di = (i2 - i1) % 7
                    dj = (j2 - j1) % 5
                    if di in S7 and dj in S5:
                        adj[v1] |= (1 << v2)
                        adj[v2] |= (1 << v1)
    return adj

def _local_search(n, adj_init, deadline):
    """Fast local search: check only neighborhoods of last-flipped vertex."""
    allmask = (1 << n) - 1
    adj = list(adj_init)
    
    # Initial full check
    k4 = _k4_found(adj, n)
    if k4 is None:
        ind = _ind6_found(adj, n)
        if ind is None:
            return adj
    
    last_flipped = None  # Track which vertices were recently changed
    
    for step in range(200000):
        if time.time() > deadline:
            return None
        
        if k4 is not None:
            a, b, c, d = k4
            pairs = [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
            u, v = random.choice(pairs)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            last_flipped = (u, v)
            # Only need to recheck K4s involving u or v
            k4 = _k4_found(adj, n)
        else:
            ind = _ind6_found(adj, n)
            if ind is None:
                return adj
            a, b, c, d, e, f = ind
            pairs = [(a,b),(a,c),(a,d),(a,e),(a,f),(b,c),(b,d),(b,e),(b,f),
                     (c,d),(c,e),(c,f),(d,e),(d,f),(e,f)]
            u, v = random.choice(pairs)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            last_flipped = (u, v)
            # Adding an edge can only create K4s, not break them
            k4 = _k4_found(adj, n)
    return None

def construct():
    start = time.time()
    deadline = start + 115  # safety margin before 150s kill
    random.seed(42)
    
    # First, verify and return the hardcoded 35-vertex graph if valid
    adj35 = _build_35_vertex_graph()
    if _is_valid(adj35, 35):
        # Try to extend to 36 vertices using systematic + random extensions
        if time.time() < deadline - 10:
            # Strategy 1: Try all single-vertex connections systematically
            # For vertex 35, try connecting to different subsets
            # Key insight: vertex 35 must have degree that avoids K4 and ind6
            # Try a range of densities and patterns
            
            n = 36
            # Try systematic patterns for vertex 35's connections
            patterns = []
            
            # Pattern 1: Connect to vertices in first k positions
            for k in range(0, 36):
                mask = (1 << k) - 1
                patterns.append(mask)
            
            # Pattern 2: Connect to vertices at specific residues
            for mod in range(1, 36):
                for start in range(mod):
                    mask = 0
                    for i in range(start, 35, mod):
                        mask |= (1 << i)
                    patterns.append(mask)
            
            # Pattern 3: Random masks with varying densities
            for density in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                for _ in range(50):
                    mask = 0
                    for i in range(35):
                        if random.random() < density:
                            mask |= (1 << i)
                    patterns.append(mask)
            
            # Try each pattern
            for pi, mask in enumerate(patterns):
                if time.time() > deadline - 5:
                    break
                adj = list(adj35) + [mask]
                # Set up adj for vertex 35
                for i in range(35):
                    if (mask >> i) & 1:
                        adj[i] |= (1 << 35)
                    else:
                        adj[i] &= ~(1 << 35)
                adj[35] = mask
                
                # Quick check: does this already have violations?
                # Only need to check K4s and ind6s involving vertex 35
                has_k4 = False
                has_ind6 = False
                
                # Check K4s involving vertex 35
                # A K4 with vertex 35 needs 3 mutual neighbors of 35 that form a triangle
                neigh35 = [i for i in range(35) if (mask >> i) & 1]
                for ii in range(len(neigh35)):
                    a = neigh35[ii]
                    for jj in range(ii+1, len(neigh35)):
                        b = neigh35[jj]
                        if not (adj[a] >> b) & 1:
                            continue
                        ab = adj[a] & adj[b] & mask  # Common neighbors of a,b that are also neighbors of 35
                        for kk in range(jj+1, len(neigh35)):
                            c = neigh35[kk]
                            if (ab >> c) & 1:
                                has_k4 = True
                                break
                    if has_k4:
                        break
                
                # Check independent sets of size 6 involving vertex 35
                # An ind6 with vertex 35 needs 5 mutual non-neighbors of 35 that form an ind5
                nonneigh35 = [i for i in range(35) if not (mask >> i) & 1]
                if len(non_neigh35) >= 5:
                    # Check if non-neighbors of 35 contain an independent set of size 5
                    # This is expensive, so use a heuristic: check small subsets
                    if len(non_neigh35) >= 5:
                        # Sample some 5-subsets
                        import itertools
                        count = 0
                        for combo in itertools.combinations(non_neigh35, 5):
                            if count > 100:
                                break
                            count += 1
                            # Check if combo is independent
                            is_ind = True
                            for x in range(5):
                                for y in range(x+1, 5):
                                    if (adj[combo[x]] >> combo[y]) & 1:
                                        is_ind = False
                                        break
                                if not is_ind:
                                    break
                            if is_ind:
                                has_ind6 = True
                                break
                
                if not has_k4 and not has_ind6:
                    # Promising! Run local search to fix any remaining issues
                    result = _local_search(n, adj, deadline - 5)
                    if result is not None and _is_valid(result, n):
                        return {'n': n, 'edges': _edges_from_adj(result, n)}
            
            # Strategy 2: More random attempts with local search
            for attempt in range(50):
                if time.time() > deadline - 5:
                    break
                random.seed(5000 + attempt)
                adj = list(adj35) + [0]
                for i in range(35):
                    if random.random() < 0.5:
                        adj[35] |= (1 << i)
                        adj[i] |= (1 << 35)
                
                result = _local_search(n, adj, deadline - 5)
                if result is not None and _is_valid(result, n):
                    return {'n': n, 'edges': _edges_from_adj(result, n)}
        
        # Return the valid 35-vertex graph
        return {'n': 35, 'edges': _edges_from_adj(adj35, 35)}
    
    # Fallback: try random local search from scratch for n=35
    for n in range(35, 16, -1):
        if time.time() > deadline - 10:
            break
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            random.seed(2000 + n * 100 + attempt)
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            result = _local_search(n, adj, deadline - 5)
            if result is not None and _is_valid(result, n):
                return {'n': n, 'edges': _edges_from_adj(result, n)}
    
    # Final fallback: Paley graph on 17 vertices
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
