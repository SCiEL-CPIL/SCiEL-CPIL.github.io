import time
import random
import math

def _creates_c4_fast(adj_int, n, u, v):
    """Check if adding edge (u,v) creates a C4. Returns True if C4 would form."""
    nu = adj_int[u]
    nv = adj_int[v]
    
    # C4 exists if there's x in N(u), y in N(v) with x~y
    # i.e., (union of N(x) for x in N(u)) intersects N(v)
    # Or equivalently: (union of N(y) for y in N(v)) intersects N(u)
    
    deg_u = bin(nu).count('1')
    deg_v = bin(nv).count('1')
    
    if deg_u <= deg_v:
        union = 0
        temp = nu
        while temp:
            lsb = temp & -temp
            x = lsb.bit_length() - 1
            union |= adj_int[x]
            temp ^= lsb
        return bool(union & nv)
    else:
        union = 0
        temp = nv
        while temp:
            lsb = temp & -temp
            y = lsb.bit_length() - 1
            union |= adj_int[y]
            temp ^= lsb
        return bool(union & nu)

def _common_neighbors_count(adj_int, u, v):
    """Count common neighbors of u and v."""
    return bin(adj_int[u] & adj_int[v]).count('1')

def _build_polarity_graph(n, seed):
    """
    Build a polarity graph on n vertices.
    A polarity graph on a finite field or using quadratic forms is C4-free.
    We use the standard construction: vertices are 0..n-1, 
    edge (i,j) if (i-j)^2 is a quadratic non-residue mod p (for prime p).
    For non-prime n, we use a simpler approach with a random symmetric matrix
    that avoids C4 by construction using a known C4-free structure.
    
    Actually, let's use a simpler deterministic C4-free seed:
    Use a bipartite-like structure or a known C4-free graph.
    
    Better approach: Use the polarity graph over GF(q) where q is a prime power.
    For n vertices, find the smallest prime p >= n, build polarity graph on p vertices,
    then take the first n vertices.
    
    For simplicity and robustness, let's use a different approach:
    Start with a random graph and remove C4s, or use a known C4-free construction.
    
    Let me use the following: For n up to 47, I'll use a greedy construction
    but with a smarter initial seeding using a bipartite graph structure.
    """
    adj_int = [0] * n
    degrees = [0] * n
    
    # Strategy: Use a "blow-up" of a small C4-free graph, or
    # use a random graph and iteratively remove C4-creating edges.
    
    # Better: Use the fact that polarity graphs are C4-free.
    # For a prime p, the polarity graph has vertices 0..p-1,
    # and (i,j) is an edge if (i-j)^2 is a quadratic non-residue mod p,
    # or i=j and -1 is a non-residue (self-loop, which we ignore for simple graphs).
    
    # Find smallest prime >= n
    def is_prime(m):
        if m < 2:
            return False
        if m == 2:
            return True
        if m % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(m)) + 1, 2):
            if m % i == 0:
                return False
        return True
    
    p = n
    while not is_prime(p):
        p += 1
    
    # Build quadratic residues mod p
    qr = set()
    for x in range(p):
        qr.add((x * x) % p)
    
    # Build polarity graph on p vertices
    # Edge (i,j) for i < j if (i-j)^2 mod p is NOT in qr (i.e., quadratic non-residue)
    # Note: (i-j)^2 = (j-i)^2, so this is symmetric.
    # Also, for i=j, (i-i)^2 = 0, which is in qr, so no self-loop.
    
    polarity_adj = [0] * p
    for i in range(p):
        for j in range(i+1, p):
            diff = (i - j) % p
            sq = (diff * diff) % p
            if sq not in qr:
                polarity_adj[i] |= (1 << j)
                polarity_adj[j] |= (1 << i)
    
    # Take first n vertices
    for i in range(n):
        adj_int[i] = polarity_adj[i] & ((1 << n) - 1)
        degrees[i] = bin(adj_int[i]).count('1')
    
    return adj_int, degrees

def _local_search(adj_int, n, min_deg_required, time_limit, seed):
    """
    Local search to maximize minimum degree while maintaining C4-free property.
    Uses edge additions with smarter heuristics and edge swaps.
    """
    start_time = time.time()
    degrees = [bin(adj_int[i]).count('1') for i in range(n)]
    
    best_min_deg = min(degrees)
    best_adj = adj_int[:]
    best_degrees = degrees[:]
    
    # Precompute non-neighbor lists for efficiency
    def get_non_neighbors(u):
        result = []
        for v in range(n):
            if v != u and not (adj_int[u] & (1 << v)):
                result.append(v)
        return result
    
    for iteration in range(500):
        if time.time() - start_time > time_limit:
            break
        
        # Check if we've reached the target
        if min(degrees) >= min_deg_required:
            break
        
        # Try to add edges to improve minimum degree
        # Focus on vertices with lowest degree
        sorted_verts = sorted(range(n), key=lambda i: degrees[i])
        
        improved = False
        for u in sorted_verts[:15]:  # Try to improve the 15 lowest-degree vertices
            if time.time() - start_time > time_limit:
                break
            if degrees[u] >= min_deg_required:
                continue
            
            # Find non-neighbors of u, sorted by potential benefit
            non_neighbors = get_non_neighbors(u)
            
            if not non_neighbors:
                continue
            
            # Score each non-neighbor by how much it helps
            # Prefer vertices that are also low-degree (to boost min degree)
            # and have fewer common neighbors with u (less likely to create C4)
            scored = []
            for v in non_neighbors:
                if degrees[v] >= min_deg_required:
                    score = 0  # Don't prioritize high-degree vertices
                else:
                    score = (min_deg_required - degrees[v]) * 10  # Boost low-degree vertices
                
                # Penalize vertices with many common neighbors (more likely to cause C4)
                cn = _common_neighbors_count(adj_int, u, v)
                score -= cn
                
                scored.append((score, v))
            
            scored.sort(key=lambda x: -x[0])
            
            # Try top candidates
            for score, v in scored[:20]:
                if time.time() - start_time > time_limit:
                    break
                if _creates_c4_fast(adj_int, n, u, v):
                    continue
                # Add edge (u, v)
                adj_int[u] |= (1 << v)
                adj_int[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
                improved = True
                break
        
        if not improved:
            # Try edge swaps: remove edge (a,b) and add (c,d)
            # This can help when direct additions are blocked
            low_vert = min(range(n), key=lambda i: degrees[i])
            
            # Get neighbors of low_vert
            neighbors = []
            temp = adj_int[low_vert]
            while temp:
                lsb = temp & -temp
                nb = lsb.bit_length() - 1
                neighbors.append(nb)
                temp ^= lsb
            
            random.shuffle(neighbors)
            
            # Get non-neighbors of low_vert
            non_neighbors = get_non_neighbors(low_vert)
            
            if non_neighbors:
                # Try 2-swap: remove (low_vert, x) and add (low_vert, y)
                # where x is a neighbor and y is a non-neighbor
                for x in neighbors[:10]:
                    if time.time() - start_time > time_limit:
                        break
                    
                    for y in non_neighbors[:10]:
                        if time.time() - start_time > time_limit:
                            break
                        
                        # Check if removing (low_vert, x) and adding (low_vert, y) is C4-free
                        # First, check that adding (low_vert, y) doesn't create C4 without x
                        # This is complex, so let's just try it and verify
                        
                        # Remove edge (low_vert, x)
                        adj_int[low_vert] ^= (1 << x)
                        adj_int[x] ^= (1 << low_vert)
                        degrees[low_vert] -= 1
                        degrees[x] -= 1
                        
                        # Check if we can add (low_vert, y)
                        if not (adj_int[low_vert] & (1 << y)):
                            if not _creates_c4_fast(adj_int, n, low_vert, y):
                                adj_int[low_vert] |= (1 << y)
                                adj_int[y] |= (1 << low_vert)
                                degrees[low_vert] += 1
                                degrees[y] += 1
                                improved = True
                                break
                        
                        if not improved:
                            # Restore the edge
                            adj_int[low_vert] |= (1 << x)
                            adj_int[x] |= (1 << low_vert)
                            degrees[low_vert] += 1
                            degrees[x] += 1
                        
                        if improved:
                            break
            
            # If still not improved, try a more aggressive swap
            if not improved and time.time() - start_time < time_limit - 0.1:
                # Try to find a vertex w such that removing some edge incident to w
                # allows adding an edge to low_vert
                high_verts = sorted(range(n), key=lambda i: -degrees[i])
                
                for w in high_verts[:5]:
                    if time.time() - start_time > time_limit:
                        break
                    if w == low_vert:
                        continue
                    
                    # Get neighbors of w
                    w_neighbors = []
                    temp = adj_int[w]
                    while temp:
                        lsb = temp & -temp
                        nb = lsb.bit_length() - 1
                        w_neighbors.append(nb)
                        temp ^= lsb
                    
                    for z in w_neighbors[:5]:
                        if time.time() - start_time > time_limit:
                            break
                        if z == low_vert:
                            continue
                        
                        # Try removing (w, z) and adding (low_vert, z) if z is non-neighbor of low_vert
                        if not (adj_int[low_vert] & (1 << z)):
                            # Remove (w, z)
                            adj_int[w] ^= (1 << z)
                            adj_int[z] ^= (1 << w)
                            degrees[w] -= 1
                            degrees[z] -= 1
                            
                            # Check if we can add (low_vert, z)
                            if not _creates_c4_fast(adj_int, n, low_vert, z):
                                adj_int[low_vert] |= (1 << z)
                                adj_int[z] |= (1 << low_vert)
                                degrees[low_vert] += 1
                                degrees[z] += 1
                                improved = True
                                break
                            else:
                                # Restore
                                adj_int[w] |= (1 << z)
                                adj_int[z] |= (1 << w)
                                degrees[w] += 1
                                degrees[z] += 1
                        
                        if improved:
                            break
                    
                    if improved:
                        break
        
        # Track best
        if min(degrees) > best_min_deg:
            best_min_deg = min(degrees)
            best_adj = adj_int[:]
            best_degrees = degrees[:]
    
    return best_adj, best_degrees

def _construct_c4_free(n, min_deg_required, time_limit, seed):
    """Construct a C4-free graph on n vertices with min degree >= min_deg_required."""
    random.seed(seed)
    start_time = time.time()
    
    # Try multiple polarity graph seeds
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
        
        adj_int, degrees = _build_polarity_graph(n, seed + attempt)
        
        # Check if already good enough
        if min(degrees) >= min_deg_required:
            edges = []
            for u in range(n):
                temp = adj_int[u]
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    if u < v:
                        edges.append([u, v])
                    temp ^= lsb
            return {'n': n, 'edges': edges}
        
        # Apply local search
        remaining = time_limit - (time.time() - start_time)
        if remaining < 0.5:
            continue
        
        # Give more time to local search for harder instances
        t_ls = min(remaining * 0.8, 8.0)
        best_adj, best_degrees = _local_search(adj_int, n, min_deg_required, t_ls, seed + attempt)
        
        if min(best_degrees) >= min_deg_required:
            edges = []
            for u in range(n):
                temp = best_adj[u]
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    if u < v:
                        edges.append([u, v])
                    temp ^= lsb
            return {'n': n, 'edges': edges}
    
    return None

def construct():
    start_time = time.time()
    deadline = 118.0
    
    # Try n=47, 46, 45, ...
    # n=47 -> min_deg 8
    # n=46 -> min_deg 7
    # n=45 -> min_deg 6
    
    for n in range(47, 38, -1):
        if time.time() - start_time > deadline:
            break
        remaining = deadline - (time.time() - start_time)
        if remaining < 2:
            break
            
        min_deg_req = max(1, n - 39)
        
        # Allocate more time for larger n
        if n >= 46:
            t_limit = min(40.0, remaining - 2)
        else:
            t_limit = min(25.0, remaining - 2)
        
        result = _construct_c4_free(n, min_deg_req, t_limit, 42)
        if result:
            return result
            
    return {'n': 39, 'edges': []}
