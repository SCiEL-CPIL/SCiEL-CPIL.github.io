"""Construct a C4-free graph with high minimum degree using finite geometry / combinatorial designs."""
import time
import random
from itertools import combinations

def _is_c4_free(n, edges):
    """Check if graph is C4-free using bitsets for speed."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            # Count bits in common - if >= 2, C4 exists
            c = common
            if c & (c - 1):  # More than one bit set
                return False
    return True

def _min_degree_from_adj(n, adj):
    """Compute minimum degree from bitset adjacency."""
    min_d = n
    for i in range(n):
        d = bin(adj[i]).count('1')
        if d < min_d:
            min_d = d
    return min_d

def _creates_c4(adj, u, v, n):
    """Check if adding edge (u,v) creates a C4."""
    # C4: u-v-w-x-u where w in N(v), x in N(u), w~x
    nv = adj[v]
    nu = adj[u]
    w_mask = nv & ~(1 << u)
    while w_mask:
        w_bit = w_mask & (-w_mask)
        w = w_bit.bit_length() - 1
        w_mask ^= w_bit
        # x must be in N(u), not v, not w, and adjacent to w
        x_mask = nu & ~(1 << v) & ~w_bit
        if x_mask & adj[w]:
            return True
    return False

def _try_add_edge(adj, deg, u, v, n):
    """Try to add edge (u,v) if it doesn't create a C4."""
    if adj[u] & (1 << v):
        return False  # already present
    if _creates_c4(adj, u, v, n):
        return False
    adj[u] |= 1 << v
    adj[v] |= 1 << u
    deg[u] += 1
    deg[v] += 1
    return True

def _try_remove_edge(adj, deg, u, v):
    """Remove edge (u,v)."""
    if not (adj[u] & (1 << v)):
        return False
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)
    deg[u] -= 1
    deg[v] -= 1
    return True

def _local_search(adj, deg, n, min_deg_needed, time_limit, deadline):
    """Local search: try to increase minimum degree via edge swaps."""
    for step in range(50000):
        if time.time() > deadline:
            break
        if time.time() > time_limit:
            break
        
        # Find vertices with minimum degree
        min_d = min(deg)
        if min_d >= min_deg_needed:
            return True
        
        # Find a low-degree vertex
        low_v = deg.index(min_d)
        
        # Try to find an edge to add for low_v
        improved = False
        candidates = [v for v in range(n) if v != low_v and not (adj[low_v] & (1 << v))]
        random.shuffle(candidates)
        
        for v in candidates:
            if time.time() > deadline:
                return False
            if _try_add_edge(adj, deg, low_v, v, n):
                improved = True
                break
        
        if not improved:
            # Try edge swap: remove an edge incident to a high-degree vertex, add edge to low_v
            # Find a vertex with high degree
            high_v = deg.index(max(deg))
            if high_v == low_v:
                continue
                
            # Try removing an edge from high_v and adding to low_v
            neighbors_high = []
            hv_mask = adj[high_v]
            while hv_mask:
                h_bit = hv_mask & (-hv_mask)
                h = h_bit.bit_length() - 1
                hv_mask ^= h_bit
                neighbors_high.append(h)
            random.shuffle(neighbors_high)
            
            for h in neighbors_high:
                if time.time() > deadline:
                    return False
                # Remove (high_v, h), try add (low_v, h) or (low_v, high_v)
                # Option 1: move edge from high_v-h to low_v-h
                if not (adj[low_v] & (1 << h)):
                    if not _creates_c4(adj, low_v, h, n):
                        _try_remove_edge(adj, deg, high_v, h)
                        _try_add_edge(adj, deg, low_v, h, n)
                        improved = True
                        break
                # Option 2: swap - remove (high_v, h), add (low_v, high_v)
                if not improved and not (adj[low_v] & (1 << high_v)):
                    if not _creates_c4(adj, low_v, high_v, n):
                        _try_remove_edge(adj, deg, high_v, h)
                        _try_add_edge(adj, deg, low_v, high_v, n)
                        improved = True
                        break
            
            if not improved:
                # Try 2-swap: remove two edges, add two edges
                break
    
    return _min_degree_from_adj(n, adj) >= min_deg_needed

def _construct_from_hypergraph(n, k, time_budget):
    """
    Construct C4-free graph from a linear k-uniform hypergraph.
    Vertices of graph = vertices of hypergraph.
    Two vertices are adjacent if they share a hyperedge.
    Linearity (any pair in at most one hyperedge) ensures C4-free.
    """
    start = time.time()
    deadline = start + time_budget
    
    # For n vertices, we want each vertex to have degree >= n-39
    # In the hypergraph model, deg(v) = sum over edges e containing v of (|e|-1)
    # If we use k-uniform edges, each edge contributes (k-1) to each vertex in it
    # We need: for each v, sum_{e ni v} (k-1) >= n-39
    
    # Try different hypergraph constructions
    best_result = None
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        random.seed(attempt * 7919 + 13)
        
        # Strategy: randomly assign vertices to k-uniform hyperedges
        # Ensure linearity by tracking which pairs are already covered
        k = 3  # 3-uniform: each edge connects 3 vertices, contributing degree 2 each
        
        # Number of hyperedges needed: each vertex needs degree >= n-39
        # With k=3, each hyperedge containing v gives v degree 2
        # So v needs at least ceil((n-39)/2) hyperedges
        # Total hyperedges ~ n * ceil((n-39)/2) / 3
        
        target_deg = n - 39
        edges_per_vertex = (target_deg + 1) // 2  # ceil((n-39)/2)
        
        # Track which pairs are in a hyperedge (for linearity)
        pair_used = [[False] * n for _ in range(n)]
        
        adj = [0] * n
        deg = [0] * n
        hyperedges = []
        
        # Build hyperedges
        for v in range(n):
            if time.time() > deadline:
                break
            
            needed = edges_per_vertex
            for _ in range(needed):
                # Find two other vertices to pair with v
                # They must not form a pair already used with each other or with v
                candidates = [u for u in range(n) if u != v and not (adj[v] & (1 << u))]
                if len(candidates) < 2:
                    # Try with existing neighbors too (but check linearity)
                    candidates = [u for u in range(n) if u != v]
                
                random.shuffle(candidates)
                
                found = False
                for i in range(min(len(candidates), 50)):
                    u = candidates[i]
                    for j in range(i+1, min(len(candidates), i+51)):
                        w = candidates[j]
                        # Check linearity: pairs (v,u), (v,w), (u,w) not already in a hyperedge
                        if not pair_used[v][u] and not pair_used[v][w] and not pair_used[u][w]:
                            # Add hyperedge {v, u, w}
                            pair_used[v][u] = True
                            pair_used[v][w] = True
                            pair_used[u][w] = True
                            hyperedges.append((v, u, w))
                            
                            # Update graph adjacency
                            adj[v] |= (1 << u) | (1 << w)
                            adj[u] |= (1 << v) | (1 << w)
                            adj[w] |= (1 << v) | (1 << u)
                            deg[v] += 2
                            deg[u] += 2
                            deg[w] += 2
                            found = True
                            break
                    if found:
                        break
                
                if not found:
                    break
        
        # Check if we achieved minimum degree
        if min(deg) >= target_deg:
            # Verify C4-free
            edge_list = []
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u] & (1 << v):
                        edge_list.append([u, v])
            
            if _is_c4_free(n, edge_list):
                if best_result is None or len(edge_list) > len(best_result[1]):
                    best_result = (n, edge_list)
        
        # Try local search to improve
        if time.time() > deadline:
            break
        
        if min(deg) < target_deg:
            # Try local search
            if _local_search(adj, deg, n, target_deg, time.time() + 5, deadline):
                edge_list = []
                for u in range(n):
                    for v in range(u+1, n):
                        if adj[u] & (1 << v):
                            edge_list.append([u, v])
                
                if _is_c4_free(n, edge_list):
                    if best_result is None or len(edge_list) > len(best_result[1]):
                        best_result = (n, edge_list)
        
        if best_result is not None and best_result[0] == n:
            break
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    best_n = 39
    best_edges = []
    
    # Try to build graphs for n from 47 down to 40
    for target_n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        min_deg_needed = target_n - 39
        
        # Method 1: Hypergraph-based construction
        time_left = deadline - time.time()
        if time_left > 10:
            result = _construct_from_hypergraph(target_n, 3, time_left - 5)
            if result is not None and result[0] >= target_n:
                if target_n > best_n:
                    best_n = target_n
                    best_edges = result[1]
                break
        
        # Method 2: Random greedy with local search
        time_left = deadline - time.time()
        if time_left > 5:
            for seed in range(200):
                if time.time() > deadline:
                    break
                
                random.seed(seed * 31 + 7)
                edges = []
                adj = [0] * target_n
                deg = [0] * target_n
                
                # Generate all possible edges
                all_edges = list(combinations(range(target_n), 2))
                random.shuffle(all_edges)
                
                for u, v in all_edges:
                    if time.time() > deadline:
                        break
                    if _try_add_edge(adj, deg, u, v, target_n):
                        edges.append([u, v])
                
                # Check minimum degree
                if min(deg) >= min_deg_needed:
                    if _is_c4_free(target_n, edges):
                        if target_n > best_n:
                            best_n = target_n
                            best_edges = edges
                        break
                
                # Try local search
                if time.time() > deadline:
                    break
                if _local_search(adj, deg, target_n, min_deg_needed, time.time() + 3, deadline):
                    edges = []
                    for u in range(target_n):
                        for v in range(u+1, target_n):
                            if adj[u] & (1 << v):
                                edges.append([u, v])
                    if _is_c4_free(target_n, edges):
                        if target_n > best_n:
                            best_n = target_n
                            best_edges = edges
                        break
        
        # If we found a valid graph for this target_n, we can stop trying smaller n
        if best_n >= target_n:
            break
    
    if best_n >= 40:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}
