import time
import random
import numpy as np

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for the 120s limit

    def is_k23_free(adj_matrix):
        """Check if the graph is K2,3-free."""
        for u in range(n):
            for v in range(u + 1, n):
                # Count common neighbors of u and v
                common_count = np.sum(adj_matrix[u] & adj_matrix[v])
                if common_count >= 3:
                    return False
        return True

    def count_edges(adj_matrix):
        return int(np.sum(adj_matrix) // 2)

    def greedy_add(adj_matrix, common_matrix, max_time):
        """Try to greedily add edges while maintaining K2,3-freeness."""
        improved = True
        while improved and time.time() < max_time:
            improved = False
            # Try all possible edges in a random order
            pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
            random.shuffle(pairs)
            for u, v in pairs:
                if adj_matrix[u, v]:
                    continue
                # Check if adding (u,v) would create K2,3
                if can_add_edge(adj_matrix, common_matrix, u, v):
                    add_edge(adj_matrix, common_matrix, u, v)
                    improved = True
        return adj_matrix, common_matrix

    def can_add_edge(adj_matrix, common_matrix, u, v):
        """Check if edge (u,v) can be added without creating K2,3."""
        if adj_matrix[u, v]:
            return False
        # Condition 1: common[u,v] <= 2
        if common_matrix[u, v] > 2:
            return False
        # Condition 2: For all w in N(v), common[u,w] <= 1
        nv = np.nonzero(adj_matrix[v])[0]
        nv = nv[nv != u]  # Exclude u
        if len(nv) > 0:
            if np.max(common_matrix[u, nv]) > 1:
                return False
        # Condition 3: For all w in N(u), common[v,w] <= 1
        nu = np.nonzero(adj_matrix[u])[0]
        nu = nu[nu != v]  # Exclude v
        if len(nu) > 0:
            if np.max(common_matrix[v, nu]) > 1:
                return False
        return True

    def add_edge(adj_matrix, common_matrix, u, v):
        """Add edge (u,v) and update common neighbor counts."""
        nv = np.nonzero(adj_matrix[v])[0]
        nu = np.nonzero(adj_matrix[u])[0]
        adj_matrix[u, v] = 1
        adj_matrix[v, u] = 1
        if len(nv) > 0:
            common_matrix[u, nv] += 1
            common_matrix[nv, u] += 1
        if len(nu) > 0:
            common_matrix[v, nu] += 1
            common_matrix[nu, v] += 1

    def remove_edge(adj_matrix, common_matrix, u, v):
        """Remove edge (u,v) and update common neighbor counts."""
        nv = np.nonzero(adj_matrix[v])[0]
        nu = np.nonzero(adj_matrix[u])[0]
        adj_matrix[u, v] = 0
        adj_matrix[v, u] = 0
        if len(nv) > 0:
            common_matrix[u, nv] -= 1
            common_matrix[nv, u] -= 1
        if len(nu) > 0:
            common_matrix[v, nu] -= 1
            common_matrix[nu, v] -= 1

    def repair(adj_matrix, common_matrix):
        """Remove edges to make the graph K2,3-free."""
        # This is a simple repair: remove edges that cause K2,3 violations
        # We'll iterate and remove problematic edges
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if adj_matrix[u, v]:
                        # Check if this edge is part of a K2,3
                        # For each pair of common neighbors (w1, w2) of u and v
                        # and another common neighbor w3, we have a K2,3
                        common = np.nonzero(adj_matrix[u] & adj_matrix[v])[0]
                        if len(common) >= 3:
                            # Remove this edge
                            remove_edge(adj_matrix, common_matrix, u, v)
                            changed = True
        return adj_matrix, common_matrix

    def local_search(adj_matrix, common_matrix, max_time):
        """Perform local search with edge swaps."""
        current_count = count_edges(adj_matrix)
        best_count = current_count
        best_adj = adj_matrix.copy()
        best_common = common_matrix.copy()
        
        while time.time() < max_time:
            # Try random edge swaps: remove one edge, try to add another
            edges = [(u, v) for u in range(n) for v in range(u + 1, n) if adj_matrix[u, v]]
            if not edges:
                break
            
            rem_u, rem_v = random.choice(edges)
            remove_edge(adj_matrix, common_matrix, rem_u, rem_v)
            
            # Try to add a random non-edge
            non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj_matrix[u, v]]
            random.shuffle(non_edges)
            
            added = False
            for add_u, add_v in non_edges:
                if time.time() > max_time:
                    break
                if can_add_edge(adj_matrix, common_matrix, add_u, add_v):
                    add_edge(adj_matrix, common_matrix, add_u, add_v)
                    added = True
                    new_count = count_edges(adj_matrix)
                    if new_count > best_count:
                        best_count = new_count
                        best_adj = adj_matrix.copy()
                        best_common = common_matrix.copy()
                    break
            
            if not added:
                # Restore the removed edge
                add_edge(adj_matrix, common_matrix, rem_u, rem_v)
            
            # Also try pure additions
            pairs = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj_matrix[u, v]]
            random.shuffle(pairs)
            for u, v in pairs:
                if time.time() > max_time:
                    break
                if can_add_edge(adj_matrix, common_matrix, u, v):
                    add_edge(adj_matrix, common_matrix, u, v)
                    new_count = count_edges(adj_matrix)
                    if new_count > best_count:
                        best_count = new_count
                        best_adj = adj_matrix.copy()
                        best_common = common_matrix.copy()
        
        return best_adj, best_common, best_count

    # Main algorithm: multiple restarts with different initializations
    best_edges = []
    best_score = 0

    for restart in range(30):
        if time.time() > deadline:
            break
        
        # Initialize with a random graph or structured graph
        if restart % 3 == 0:
            # Start with 5 disjoint K4s
            adj = np.zeros((n, n), dtype=np.int8)
            for k in range(5):
                for i in range(4):
                    for j in range(i + 1, 4):
                        u = 4 * k + i
                        v = 4 * k + j
                        adj[u, v] = 1
                        adj[v, u] = 1
        elif restart % 3 == 1:
            # Start with a random sparse graph
            adj = np.zeros((n, n), dtype=np.int8)
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.3:
                        adj[u, v] = 1
                        adj[v, u] = 1
        else:
            # Start with a random graph with higher density
            adj = np.zeros((n, n), dtype=np.int8)
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        adj[u, v] = 1
                        adj[v, u] = 1
        
        # Repair if needed
        common = (adj @ adj).astype(np.int8)
        adj, common = repair(adj, common)
        
        # Greedy add
        adj, common = greedy_add(adj, common, deadline)
        
        # Local search
        adj, common, count = local_search(adj, common, deadline)
        
        if count > best_score:
            best_score = count
            best_adj = adj.copy()
    
    # Convert to edge list
    best_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                best_edges.append([u, v])
    
    return best_edges
