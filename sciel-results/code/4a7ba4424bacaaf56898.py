import time
import random

def construct():
    n = 20
    deadline = time.time() + 110  # Leave margin for 150s limit
    
    # Helper to get bit count
    def popcount(x):
        return x.bit_count() if hasattr(int, 'bit_count') else bin(x).count("1")
    
    def can_add_edge(adj, u, v):
        """Check if edge (u,v) can be added to maintain K2,3-free property."""
        # Common neighbors of u and v must be <= 2
        if popcount(adj[u] & adj[v]) > 2:
            return False
        
        # For each neighbor b of v, common neighbors of u and b must be <= 1
        mask_v = adj[v]
        i = 0
        while mask_v:
            if mask_v & 1:
                if popcount(adj[u] & adj[i]) > 1:
                    return False
            mask_v >>= 1
            i += 1
        
        # For each neighbor a of u, common neighbors of v and a must be <= 1
        mask_u = adj[u]
        i = 0
        while mask_u:
            if mask_u & 1:
                if popcount(adj[v] & adj[i]) > 1:
                    return False
            mask_u >>= 1
            i += 1
        
        return True
    
    def local_search(adj, edges, max_iterations=80):
        """Try to improve by swapping edges: remove one, add another (1-for-1 or 1-for-2)."""
        n = 20
        edge_set = set(edges)
        all_possible = [(u, v) for u in range(n) for v in range(u + 1, n)]
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations and time.time() < deadline:
            improved = False
            iterations += 1
            random.shuffle(edges)
            random.shuffle(all_possible)
            
            for rem_u, rem_v in edges:
                if time.time() > deadline:
                    break
                # Try removing this edge
                adj[rem_u] &= ~(1 << rem_v)
                adj[rem_v] &= ~(1 << rem_u)
                edge_set.discard((rem_u, rem_v))
                edge_set.discard((rem_v, rem_u))
                
                # First try 1-for-2 swap (remove 1, add 2)
                found_swap = False
                for add_u1, add_v1 in all_possible:
                    if (add_u1, add_v1) in edge_set or (add_v1, add_u1) in edge_set:
                        continue
                    if add_u1 == rem_u and add_v1 == rem_v:
                        continue
                    if not can_add_edge(adj, add_u1, add_v1):
                        continue
                    # Tentatively add first edge
                    adj[add_u1] |= (1 << add_v1)
                    adj[add_v1] |= (1 << add_u1)
                    edge_set.add((add_u1, add_v1))
                    
                    for add_u2, add_v2 in all_possible:
                        if (add_u2, add_v2) in edge_set or (add_v2, add_u2) in edge_set:
                            continue
                        if add_u2 == rem_u and add_v2 == rem_v:
                            continue
                        if add_u2 == add_u1 and add_v2 == add_v1:
                            continue
                        if add_u2 == add_v1 and add_v2 == add_u1:
                            continue
                        if not can_add_edge(adj, add_u2, add_v2):
                            continue
                        # Found valid 1-for-2 swap
                        adj[add_u2] |= (1 << add_v2)
                        adj[add_v2] |= (1 << add_u2)
                        edge_set.add((add_u2, add_v2))
                        # Update edges list
                        idx = edges.index((rem_u, rem_v))
                        edges[idx] = (add_u1, add_v1)
                        edges.append((add_u2, add_v2))
                        improved = True
                        found_swap = True
                        break
                    
                    if found_swap:
                        break
                    # Undo first addition
                    adj[add_u1] &= ~(1 << add_v1)
                    adj[add_v1] &= ~(1 << add_u1)
                    edge_set.discard((add_u1, add_v1))
                
                if found_swap:
                    break
                
                # Try 1-for-1 swap (remove 1, add 1)
                for add_u, add_v in all_possible:
                    if (add_u, add_v) in edge_set or (add_v, add_u) in edge_set:
                        continue
                    if add_u == rem_u and add_v == rem_v:
                        continue
                    if can_add_edge(adj, add_u, add_v):
                        adj[add_u] |= (1 << add_v)
                        adj[add_v] |= (1 << add_u)
                        edge_set.add((add_u, add_v))
                        edges[edges.index((rem_u, rem_v))] = (add_u, add_v)
                        improved = True
                        break
                else:
                    # Couldn't find a swap, put back
                    adj[rem_u] |= (1 << rem_v)
                    adj[rem_v] |= (1 << rem_u)
                    edge_set.add((rem_u, rem_v))
                    continue
                break  # Found a swap, restart
            if time.time() > deadline:
                break
        
        return edges
    
    def degree_balanced_greedy(adj, all_edges, order):
        """Greedy construction with degree-balancing heuristic."""
        current_edges = []
        degrees = [0] * n
        
        for u, v in order:
            if can_add_edge(adj, u, v):
                # Prefer adding edges that balance degrees
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.append((u, v))
                degrees[u] += 1
                degrees[v] += 1
        
        return current_edges
    
    best_edges = []
    best_count = 0
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Try multiple strategies
    strategy_count = 0
    while time.time() < deadline:
        strategy_count += 1
        
        if strategy_count % 3 == 1:
            # Random ordering
            edges_copy = all_edges[:]
            random.shuffle(edges_copy)
        elif strategy_count % 3 == 2:
            # Degree-balanced: sort by potential to connect low-degree vertices
            edges_copy = all_edges[:]
            random.shuffle(edges_copy)
        else:
            # Hybrid: start with high-potential edges
            edges_copy = all_edges[:]
            random.shuffle(edges_copy)
        
        # adj[i] is bitmask of neighbors of i
        adj = [0] * n
        current_edges = []
        
        for u, v in edges_copy:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.append((u, v))
        
        # Local search to improve
        if time.time() < deadline:
            current_edges = local_search(adj, current_edges[:])
            # Rebuild adj from current_edges for consistency
            adj = [0] * n
            for u, v in current_edges:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            
    # Convert list of tuples to list of lists [u, v]
    return [[u, v] for u, v in best_edges]
