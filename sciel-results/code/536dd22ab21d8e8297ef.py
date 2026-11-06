import time
import random
import math

_DEADLINE = None

def _count_k4_in_subset(adj, n, subset):
    """Count K4s in the subgraph induced by subset (list of vertices)."""
    # subset is a list of vertex indices
    k4_count = 0
    m = len(subset)
    # Map subset indices to 0..m-1 for easier handling? No, use original indices.
    # For small m, just check all 4-combinations.
    for i in range(m):
        u = subset[i]
        for j in range(i+1, m):
            v = subset[j]
            if not (adj[u] >> v) & 1:
                continue
            # Common neighbors in subset
            common = adj[u] & adj[v]
            # Filter to only those in subset
            # This is inefficient if subset is large.
            # But subset size is |N(u) cap N(v)| which is <= 34.
            # Actually, for incremental update, we just need the count.
            # Let's stick to the full check for simplicity in the first pass, 
            # but optimize if needed.
            pass
    return k4_count

def _full_energy(adj, n):
    """Calculate exact number of K4s and Ind6s."""
    k4_count = 0
    ind6_count = 0
    
    # Count K4s
    # Iterate over all 4-combinations
    # For n=36, C(36,4) = 58905
    for a in range(n):
        aa = adj[a]
        for b in range(a+1, n):
            if not (aa >> b) & 1:
                continue
            ab = aa & adj[b]
            for c in range(b+1, n):
                if not (ab >> c) & 1:
                    continue
                abc = ab & adj[c]
                for d in range(c+1, n):
                    if (abc >> d) & 1:
                        k4_count += 1
                        
    # Count Ind6s
    # Iterate over all 6-combinations
    # For n=36, C(36,6) = 1,947,792
    allmask = (1 << n) - 1
    for a in range(n):
        na = allmask ^ (1 << a) ^ adj[a]
        for b in range(a+1, n):
            if not (na >> b) & 1:
                continue
            nab = na & (allmask ^ (1 << b) ^ adj[b])
            for c in range(b+1, n):
                if not (nab >> c) & 1:
                    continue
                nabc = nab & (allmask ^ (1 << c) ^ adj[c])
                for d in range(c+1, n):
                    if not (nabc >> d) & 1:
                        continue
                    nabcd = nabc & (allmask ^ (1 << d) ^ adj[d])
                    for e in range(d+1, n):
                        if not (nabcd >> e) & 1:
                            continue
                        rest = nabcd & (allmask ^ (1 << e) ^ adj[e])
                        # Check if there is any f > e in rest
                        higher = rest & (~((1 << (e + 1)) - 1))
                        if higher:
                            ind6_count += 1
                            
    return k4_count + ind6_count

def _delta_k4(adj, n, u, v, adding):
    """Calculate change in K4 count when flipping edge (u,v)."""
    # Common neighbors
    common = adj[u] & adj[v]
    # Count edges in the subgraph induced by common
    # This is the number of triangles in the neighborhood intersection?
    # No, it's the number of edges between pairs in common.
    # A K4 containing (u,v) corresponds to an edge (x,y) in common.
    k4_delta = 0
    # Iterate over pairs in common
    # To be fast, we can iterate over bits in common
    # common is a bitmask
    c = common
    while c:
        x_bit = c & -c
        x = x_bit.bit_length() - 1
        c ^= x_bit
        # Count neighbors of x in common
        # (common >> x) & 1 is 0? No, x is in common.
        # We need neighbors of x in common.
        # neighbors_x_in_common = adj[x] & common
        # Count bits in neighbors_x_in_common
        # But we only want to count each edge once.
        # So we can just sum popcount(adj[x] & common) for all x in common / 2?
        # No, that counts edges.
        # Let's just do it simply for now, n is small.
        pass
    
    # Optimized: 
    # The number of K4s containing (u,v) is the number of edges in G[common].
    # Let's compute it.
    common_list = []
    c = common
    while c:
        x_bit = c & -c
        x = x_bit.bit_length() - 1
        common_list.append(x)
        c ^= x_bit
        
    # Count edges in common_list
    edge_count = 0
    for i in range(len(common_list)):
        x = common_list[i]
        for j in range(i+1, len(common_list)):
            y = common_list[j]
            if (adj[x] >> y) & 1:
                edge_count += 1
                
    if adding:
        return edge_count
    else:
        return -edge_count

def _delta_ind6(adj, n, u, v, adding):
    """Calculate change in Ind6 count when flipping edge (u,v)."""
    # Non-common neighbors (non-neighbors of u AND non-neighbors of v)
    allmask = (1 << n) - 1
    non_common = allmask ^ adj[u] ^ adj[v]
    # Remove u and v themselves?
    # adj[u] does not contain u. adj[v] does not contain v.
    # So non_common contains u and v?
    # adj[u] has 0 at u. So ~adj[u] has 1 at u.
    # So u is in non_common.
    # We should remove u and v from consideration as the "other 4" vertices.
    non_common ^= (1 << u)
    non_common ^= (1 << v)
    
    # We need to count Ind4s in the subgraph induced by non_common.
    # An Ind6 containing (u,v) corresponds to an Ind4 in G[non_common].
    # If we ADD edge (u,v), we REMOVE these Ind6s. So delta is -count.
    # If we REMOVE edge (u,v), we ADD these Ind6s. So delta is +count.
    
    # Count Ind4s in non_common
    ind4_count = 0
    # List of vertices in non_common
    nc_list = []
    c = non_common
    while c:
        x_bit = c & -c
        x = x_bit.bit_length() - 1
        nc_list.append(x)
        c ^= x_bit
        
    m = len(nc_list)
    # Check all 4-combinations in nc_list
    # If m is large, this is slow.
    # For n=36, m can be up to 34. C(34,4) ~ 46k.
    # This is fast.
    for i in range(m):
        a = nc_list[i]
        na = non_common ^ (1 << a) ^ adj[a]
        for j in range(i+1, m):
            b = nc_list[j]
            if not (na >> b) & 1:
                continue
            nab = na & (non_common ^ (1 << b) ^ adj[b])
            for k in range(j+1, m):
                c_vert = nc_list[k]
                if not (nab >> c_vert) & 1:
                    continue
                nabc = nab & (non_common ^ (1 << c_vert) ^ adj[c_vert])
                for l in range(k+1, m):
                    d = nc_list[l]
                    if (nabc >> d) & 1:
                        ind4_count += 1
                        
    if adding:
        return -ind4_count
    else:
        return ind4_count

def _simulated_annealing(n, initial_adj, deadline):
    adj = initial_adj[:]
    current_energy = _full_energy(adj, n)
    best_adj = adj[:]
    best_energy = current_energy
    
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        T *= cooling
        if T < 0.01:
            break
            
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
            
        is_edge = (adj[u] >> v) & 1
        adding = not is_edge
        
        d_k4 = _delta_k4(adj, n, u, v, adding)
        d_ind6 = _delta_ind6(adj, n, u, v, adding)
        d_energy = d_k4 + d_ind6
        
        if d_energy < 0 or random.random() < math.exp(-d_energy / T):
            # Flip edge
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            current_energy += d_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_adj = adj[:]
                if best_energy == 0:
                    return best_adj
        else:
            pass
            
    return best_adj

def construct():
    global _DEADLINE
    start = time.time()
    _DEADLINE = start + 110
    
    n_target = 36
    
    # Try to find a 36-vertex graph using Simulated Annealing
    # Multiple restarts with different initializations
    
    best_adj = None
    best_energy = float('inf')
    
    # Initialization 1: Random
    for restart in range(20):
        if time.time() > _DEADLINE:
            break
        random.seed(42 + restart)
        adj = [0] * n_target
        for i in range(n_target):
            for j in range(i+1, n_target):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        res_adj = _simulated_annealing(n_target, adj, _DEADLINE)
        e = _full_energy(res_adj, n_target)
        if e < best_energy:
            best_energy = e
            best_adj = res_adj
            if e == 0:
                return {'n': 36, 'edges': _edges_from_adj(best_adj, 36)}
                
    # Initialization 2: Based on 35-vertex graph
    # Build 35-vertex graph
    adj35 = [0] * 35
    S7 = {1, 2, 4}
    S5 = {1, 2, 3, 4}
    for i1 in range(7):
        for j1 in range(5):
            v1 = i1 * 5 + j1
            for i2 in range(i1 + 1, 7):
                for j2 in range(5):
                    v2 = i2 * 5 + j2
                    di = (i2 - i1) % 7
                    dj = (j2 - j1) % 5
                    if di in S7 and dj in S5:
                        adj35[v1] |= (1 << v2)
                        adj35[v2] |= (1 << v1)
                        
    # Extend to 36
    for restart in range(20):
        if time.time() > _DEADLINE:
            break
        random.seed(100 + restart)
        adj = adj35[:] + [0]
        # Connect vertex 35 to random subset of 0..34
        for i in range(35):
            if random.random() < 0.5:
                adj[35] |= (1 << i)
                adj[i] |= (1 << 35)
                
        res_adj = _simulated_annealing(36, adj, _DEADLINE)
        e = _full_energy(res_adj, 36)
        if e < best_energy:
            best_energy = e
            best_adj = res_adj
            if e == 0:
                return {'n': 36, 'edges': _edges_from_adj(best_adj, 36)}
                
    if best_energy == 0:
        return {'n': 36, 'edges': _edges_from_adj(best_adj, 36)}
        
    # Fallback to 35
    return {'n': 35, 'edges': _edges_from_adj(adj35, 35)}

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return edges
