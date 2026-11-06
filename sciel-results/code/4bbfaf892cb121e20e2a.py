import time
import random
from itertools import combinations

def _has_k4_bitset(adj, n):
    """Check for K4 using bitsets. adj is list of ints."""
    # adj[i] has bit j set if edge (i,j) exists
    for a in range(n):
        # Get neighbors of a
        na = adj[a]
        # Iterate over neighbors b > a
        b_mask = na & ((1 << (n - a - 1)) - 1) << (a + 1)
        while b_mask:
            lsb = b_mask & -b_mask
            b = lsb.bit_length() - 1
            b_mask ^= lsb
            # Get common neighbors of a and b, greater than b
            nb = adj[b]
            common = na & nb
            # Mask to bits > b
            common &= ((1 << (n - b - 1)) - 1) << (b + 1)
            # Check if common contains a triangle
            # Iterate over c in common
            c_mask = common
            while c_mask:
                lsb_c = c_mask & -c_mask
                c = lsb_c.bit_length() - 1
                c_mask ^= lsb_c
                # Check if c and d are connected for d > c in common
                nc = adj[c]
                # Common neighbors of a, b, c greater than c
                d_mask = common & nc
                d_mask &= ((1 << (n - c - 1)) - 1) << (c + 1)
                if d_mask:
                    return True
    return False

def _find_k4_bitset(adj, n):
    """Find a K4, return (a,b,c,d) or None."""
    for a in range(n):
        na = adj[a]
        b_mask = na & ((1 << (n - a - 1)) - 1) << (a + 1)
        while b_mask:
            lsb = b_mask & -b_mask
            b = lsb.bit_length() - 1
            b_mask ^= lsb
            nb = adj[b]
            common = na & nb
            common &= ((1 << (n - b - 1)) - 1) << (b + 1)
            c_mask = common
            while c_mask:
                lsb_c = c_mask & -c_mask
                c = lsb_c.bit_length() - 1
                c_mask ^= lsb_c
                nc = adj[c]
                d_mask = common & nc
                d_mask &= ((1 << (n - c - 1)) - 1) << (c + 1)
                if d_mask:
                    lsb_d = d_mask & -d_mask
                    d = lsb_d.bit_length() - 1
                    return (a, b, c, d)
    return None

def _has_ind6_bitset(adj, n):
    """Check for Independent Set of size 6 in G, which is K6 in G_comp."""
    # Build complement adjacency
    # comp_adj[i] has bit j set if NOT edge (i,j) in G, and i!=j
    comp_adj = [0] * n
    full_mask = (1 << n) - 1
    for i in range(n):
        # Invert bits, clear i-th bit, mask to n bits
        comp_adj[i] = (full_mask ^ adj[i]) & ~(1 << i)
    
    # Check for K6 in comp_adj
    # Use a simple backtracking or heuristic
    # For speed, we can just check random 6-sets? No, must be exact for validity.
    # But for search, we can use a fast check.
    # Let's use a recursive clique finder with pruning.
    
    def find_clique_size_k(vertices, k):
        if k == 0:
            return True
        if not vertices:
            return False
        # Order vertices by degree in subgraph?
        # Simple check: if len(vertices) < k, return False
        if len(vertices) < k:
            return False
        
        # Pick a vertex, say the first one
        v = vertices[0]
        rest = vertices[1:]
        
        # Recurse with v included: need clique of size k-1 in N(v) intersect rest
        # N(v) in comp_adj
        nv = comp_adj[v]
        neighbors_in_rest = [u for u in rest if (nv >> u) & 1]
        if find_clique_size_k(neighbors_in_rest, k - 1):
            return True
        
        # Recurse with v excluded
        if find_clique_size_k(rest, k):
            return True
            
        return False

    # This recursive function might be slow.
    # Alternative: Just check if alpha(G) >= 6 using a greedy upper bound?
    # For validity check, we need exact.
    # For search, we can use a heuristic:
    # If we find an Ind6, return True.
    # If we don't, assume False (risk of false negative).
    
    # Let's try a randomized check for speed in the loop, and exact check at the end.
    # For now, implement a fast exact check using bitsets and recursion.
    
    # Optimization: Use bitsets in recursion
    def has_clique_k(bits, k):
        if k == 0: return True
        if bits == 0: return False
        if bin(bits).count('1') < k: return False
        
        # Pick lowest set bit
        lsb = bits & -bits
        v = lsb.bit_length() - 1
        bits_rest = bits ^ lsb
        
        # Neighbors of v in the candidate set
        nv = comp_adj[v] & bits_rest
        if has_clique_k(nv, k - 1):
            return True
        if has_clique_k(bits_rest, k):
            return True
        return False

    return has_clique_k((1 << n) - 1, 6)

def _find_ind6_bitset(adj, n):
    """Find an Ind6, return tuple or None."""
    # Similar to above but returns the set
    comp_adj = [0] * n
    full_mask = (1 << n) - 1
    for i in range(n):
        comp_adj[i] = (full_mask ^ adj[i]) & ~(1 << i)
        
    def find_clique(bits, k, path):
        if k == 0:
            return tuple(path)
        if bits == 0 or bin(bits).count('1') < k:
            return None
        
        lsb = bits & -bits
        v = lsb.bit_length() - 1
        bits_rest = bits ^ lsb
        
        nv = comp_adj[v] & bits_rest
        res = find_clique(nv, k - 1, path + [v])
        if res:
            return res
        res = find_clique(bits_rest, k, path)
        if res:
            return res
        return None

    return find_clique((1 << n) - 1, 6, [])

def _is_valid_bitset(adj, n):
    return not _has_k4_bitset(adj, n) and not _has_ind6_bitset(adj, n)

def _sa_build(n, seed=42, time_limit=10.0):
    random.seed(seed)
    adj = [0] * n
    # Init with random p=0.5
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    start_time = time.time()
    temp = 10.0
    min_temp = 0.1
    
    for iter in range(10000):
        if time.time() - start_time > time_limit:
            break
            
        # Check violations
        k4 = _find_k4_bitset(adj, n)
        ind6 = _find_ind6_bitset(adj, n)
        
        if not k4 and not ind6:
            return adj
            
        # Pick a violation
        if k4 and (not ind6 or random.random() < 0.5):
            # Remove edge from K4
            a, b, c, d = k4
            edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
            u, v = random.choice(edges)
            # Flip edge
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
        else:
            # Add edge to Ind6
            quad = ind6
            pairs = list(combinations(quad, 2))
            u, v = random.choice(pairs)
            # Flip edge
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
        temp *= 0.995
        if temp < min_temp:
            temp = min_temp

    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_adj = None
    best_n = 1
    
    # Try n=35 first, as it's the target
    for n in [35, 34, 33, 32, 31, 30]:
        if time.time() > deadline - 20:
            break
            
        found = False
        for seed in range(20):
            if time.time() > deadline - 10:
                break
            adj = _sa_build(n, seed, time_limit=5.0)
            if _is_valid_bitset(adj, n):
                best_adj = adj
                best_n = n
                found = True
                break
        if found:
            break
            
    if best_adj is None:
        # Fallback: Random graph on smaller n
        n = 20
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        best_adj = adj
        best_n = n

    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return {'n': best_n, 'edges': edges}
