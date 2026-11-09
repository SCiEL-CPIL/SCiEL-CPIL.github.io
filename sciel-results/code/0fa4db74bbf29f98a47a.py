"""Rewritten program for K2,3-free dense graphs on 20 vertices."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _is_k23_free(adj):
    """Check if adjacency matrix (numpy bool array) is K2,3-free."""
    n = adj.shape[0]
    # For every pair (u, v), count common neighbours.
    # common = adj @ adj  (but we need to exclude the edge u-v itself from common neighbours? 
    # Actually, common neighbours of u and v are vertices w such that adj[u,w] and adj[v,w].
    # If u and v are adjacent, w cannot be u or v because adj[u,u]=0.
    # So common_count = (adj @ adj)[u, v] for u != v.
    # We need common_count <= 2 for all u != v.
    common = adj @ adj
    # Mask out diagonal and check off-diagonal
    mask = ~np.eye(n, dtype=bool)
    return np.all(common[mask] <= 2)

def _count_edges(adj):
    return int(np.sum(adj) // 2)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n = 20
    best_edges = None
    best_count = 0
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m_total = len(all_edges)
    
    # Index mapping for edges
    edge_index = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
    
    # For each edge, precompute the list of pairs (u, v) that this edge affects
    # When we add edge (a, b), it increases common neighbours for all pairs (a, x) and (b, x)
    # where x is a common neighbour of a and b... no, that's not right.
    
    # Let's use a different approach: vectorized greedy with randomization and restarts.
    
    def greedy_construction(seed):
        rng = random.Random(seed)
        adj = np.zeros((n, n), dtype=bool)
        
        # Shuffle edge order
        edges_shuffled = all_edges[:]
        rng.shuffle(edges_shuffled)
        
        # Track common neighbour counts for all pairs
        # common[u, v] = number of common neighbours of u and v
        common = np.zeros((n, n), dtype=np.int32)
        
        for (u, v) in edges_shuffled:
            # Check if adding (u, v) would violate K2,3 constraint
            # Adding (u, v) increases common[u, x] by 1 for all x adjacent to v (x != u)
            # and common[v, x] by 1 for all x adjacent to u (x != v)
            
            # More precisely: for any pair (a, b), the new common count is:
            # old_common[a, b] + [a==u and b in N(v)] + [a==v and b in N(u)] + [b==u and a in N(v)] + [b==v and a in N(u)]
            # But we only care about pairs that are affected.
            
            # Affected pairs are:
            # 1. (u, x) for x in N(v) - common[u, x] increases by 1
            # 2. (v, x) for x in N(u) - common[v, x] increases by 1
            
            # Check if any of these would exceed 2
            # For (u, x) where x in N(v): common[u, x] + 1 <= 2, i.e., common[u, x] <= 1
            # For (v, x) where x in N(u): common[v, x] + 1 <= 2, i.e., common[v, x] <= 1
            
            ok = True
            # Check pairs (u, x) for x in N(v), x != u
            for x in range(n):
                if x != u and adj[v, x]:
                    if common[u, x] >= 2:
                        ok = False
                        break
            if not ok:
                continue
                
            # Check pairs (v, x) for x in N(u), x != v
            for x in range(n):
                if x != v and adj[u, x]:
                    if common[v, x] >= 2:
                        ok = False
                        break
            if not ok:
                continue
                
            # Also need to check: does adding (u, v) create a K2,3 involving u and v themselves?
            # common[u, v] increases by 0 (since u and v are not neighbours of each other in the common count)
            # Wait, common[u, v] is the number of w such that adj[u,w] and adj[v,w].
            # Adding edge (u, v) doesn't change common[u, v] because w cannot be u or v.
            # So we're fine.
            
            # Add the edge
            adj[u, v] = True
            adj[v, u] = True
            
            # Update common counts
            # For x in N(v), x != u: common[u, x] += 1
            for x in range(n):
                if x != u and adj[v, x]:
                    common[u, x] += 1
                    common[x, u] += 1
            # For x in N(u), x != v: common[v, x] += 1
            for x in range(n):
                if x != v and adj[u, x]:
                    common[v, x] += 1
                    common[x, v] += 1
        
        return adj
    
    # Run multiple greedy constructions with different seeds
    num_restarts = 500
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        adj = greedy_construction(i)
        count = _count_edges(adj)
        if count > best_count:
            best_count = count
            best_edges = adj.copy()
    
    # Local search: try to add more edges by removing and re-adding
    if best_edges is not None and time.time() < deadline:
        adj = best_edges.copy()
        common = (adj @ adj).astype(np.int32)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add each missing edge
            for (u, v) in all_edges:
                if time.time() > deadline:
                    break
                if adj[u, v]:
                    continue
                    
                # Check if we can add (u, v)
                ok = True
                for x in range(n):
                    if x != u and adj[v, x]:
                        if common[u, x] >= 2:
                            ok = False
                            break
                if ok:
                    for x in range(n):
                        if x != v and adj[u, x]:
                            if common[v, x] >= 2:
                                ok = False
                                break
                
                if ok:
                    adj[u, v] = True
                    adj[v, u] = True
                    for x in range(n):
                        if x != u and adj[v, x]:
                            common[u, x] += 1
                            common[x, u] += 1
                    for x in range(n):
                        if x != v and adj[u, x]:
                            common[v, x] += 1
                            common[x, v] += 1
                    improved = True
                    best_count = _count_edges(adj)
                    best_edges = adj.copy()
    
    # Convert best adjacency matrix to edge list
    if best_edges is None:
        # Fallback: five disjoint K4s
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_edges[i, j]:
                edges.append([i, j])
    
    return edges

# EVOLVE-BLOCK-END
