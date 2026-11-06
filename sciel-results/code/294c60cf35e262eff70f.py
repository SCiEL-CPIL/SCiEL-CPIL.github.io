import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n = 31
    target_edges = 91
    q = 5
    
    # 1. Construct Polarity Graph of PG(2, 5)
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    assert len(points) == 31
    
    adj = np.zeros((n, n), dtype=np.int8)
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            p = points[i]
            q_pt = points[j]
            val = (p[0]*q_pt[0] + p[1]*q_pt[1] + p[2]*q_pt[2]) % q
            if val == 0:
                edges.append((i, j))
                adj[i, j] = 1
                adj[j, i] = 1
                
    def count_c4_fast(adj_mat):
        # #C4s = (1/2) * sum_{u < v} C(cn(u,v), 2) where cn(u,v) = (A^2)[u,v] for u != v
        A2 = np.dot(adj_mat, adj_mat)
        c2 = np.where(A2 >= 2, A2 * (A2 - 1) // 2, 0)
        c2[np.diag_indices(n)] = 0
        total = np.sum(c2) // 2
        return int(total)

    def c4_cost(adj_mat, u, v):
        """Compute the number of C4s that edge (u,v) participates in.
        This is the number of common neighbors of u and v minus 1 (since u-v is already an edge).
        Actually, each common neighbor w forms a path u-w-v, and for each pair of common neighbors w1, w2,
        the cycle u-w1-v-w2-u is a C4 if w1 != w2 and there's no chord.
        More precisely: #C4s containing edge (u,v) = C(common_neighbors(u,v), 2)
        where common_neighbors are vertices adjacent to both u and v (excluding u and v themselves).
        """
        # Common neighbors of u and v
        common = np.sum(adj_mat[u] & adj_mat[v]) - 2  # subtract self-loops if any, but we don't have them
        # Wait, adj[u][u] = 0 and adj[v][v] = 0, so common = sum of adj[u][w] * adj[v][w] for all w
        # This includes w=u (adj[v][u]=1) and w=v (adj[u][v]=1), so subtract 2
        if common < 0:
            common = 0
        return common * (common - 1) // 2

    current_edges = len(edges)
    
    # Adjust edge count to target_edges
    if current_edges > target_edges:
        to_remove = current_edges - target_edges
        for _ in range(to_remove):
            best_edge = None
            best_c4 = float('inf')
            for e in edges:
                u, v = e
                adj[u, v] = 0
                adj[v, u] = 0
                new_c4 = count_c4_fast(adj)
                adj[u, v] = 1
                adj[v, u] = 1
                if new_c4 < best_c4:
                    best_c4 = new_c4
                    best_edge = e
            if best_edge:
                u, v = best_edge
                edges.remove(best_edge)
                adj[u, v] = 0
                adj[v, u] = 0
                
    elif current_edges < target_edges:
        to_add = target_edges - target_edges  # already at target, this shouldn't happen
        # Add edges with minimum C4 increase
        for _ in range(to_add):
            best_edge = None
            best_c4 = float('inf')
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v] == 0:
                        adj[u, v] = 1
                        adj[v, u] = 1
                        new_c4 = count_c4_fast(adj)
                        adj[u, v] = 0
                        adj[v, u] = 0
                        if new_c4 < best_c4:
                            best_c4 = new_c4
                            best_edge = (u, v)
            if best_edge:
                u, v = best_edge
                edges.append(best_edge)
                adj[u, v] = 1
                adj[v, u] = 1

    # Now we have exactly 91 edges.
    # Perform Local Search with incremental C4 counting
    best_c4_overall = float('inf')
    best_adj_overall = None
    
    def incremental_c4_delta_remove(adj_mat, u, v):
        """Compute change in C4 count when removing edge (u,v).
        Negative means C4 count decreases.
        When removing edge (u,v), we lose all C4s that use this edge.
        #C4s using edge (u,v) = C(common_neighbors(u,v), 2)
        """
        common = int(np.sum(adj_mat[u] & adj_mat[v])) - 2
        if common < 0:
            common = 0
        return -common * (common - 1) // 2

    def incremental_c4_delta_add(adj_mat, u, v):
        """Compute change in C4 count when adding edge (u,v).
        Positive means C4 count increases.
        When adding edge (u,v), we gain all C4s that use this edge.
        #C4s using edge (u,v) = C(common_neighbors(u,v), 2)
        """
        common = int(np.sum(adj_mat[u] & adj_mat[v]))
        # Note: when adding, adj[u][v] and adj[v][u] are 0, so common doesn't include u or v
        return common * (common - 1) // 2

    num_restarts = 8
    time_per_restart = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Reset to base graph
        adj_r = np.zeros((n, n), dtype=np.int8)
        edges_r = []
        for i in range(n):
            for j in range(i + 1, n):
                p = points[i]
                q_pt = points[j]
                val = (p[0]*q_pt[0] + p[1]*q_pt[1] + p[2]*q_pt[2]) % q
                if val == 0:
                    edges_r.append((i, j))
                    adj_r[i, j] = 1
                    adj_r[j, i] = 1
        
        # Find best edge to add with minimum C4 increase
        best_edge = None
        best_c4_add = float('inf')
        for u in range(n):
            for v in range(u + 1, n):
                if adj_r[u, v] == 0:
                    delta = incremental_c4_delta_add(adj_r, u, v)
                    if delta < best_c4_add:
                        best_c4_add = delta
                        best_edge = (u, v)
        
        if best_edge:
            u, v = best_edge
            edges_r.append(best_edge)
            adj_r[u, v] = 1
            adj_r[v, u] = 1
        
        current_c4 = count_c4_fast(adj_r)
        best_c4 = current_c4
        best_adj_r = adj_r.copy()
        
        # Local search: try swapping edges to reduce C4 count
        # Use simulated annealing with incremental updates
        temp = 2.0
        cooling = 0.9999
        restart_deadline = start_time + time_per_restart * (restart + 1)
        
        # Precompute edge list for faster access
        edge_set = set(edges_r)
        
        while time.time() < restart_deadline:
            # Pick a random edge to swap
            e_idx = random.randint(0, len(edges_r) - 1)
            u, v = edges_r[e_idx]
            
            # Try to find a non-neighbor (a,b) to swap with
            found = False
            attempts = 0
            while not found and attempts < 100:
                a = random.randint(0, n - 1)
                b = random.randint(0, n - 1)
                if a != b and a < b and adj_r[a, b] == 0:
                    if not (a == u and b == v) and not (a == v and b == u):
                        found = True
                attempts += 1
                
            if not found:
                continue
            
            # Compute delta for removing (u,v) and adding (a,b)
            delta_remove = incremental_c4_delta_remove(adj_r, u, v)
            delta_add = incremental_c4_delta_add(adj_r, a, b)
            delta_total = delta_remove + delta_add
            
            if delta_total < 0:
                # Accept: C4 count decreases
                adj_r[u, v] = 0
                adj_r[v, u] = 0
                adj_r[a, b] = 1
                adj_r[b, a] = 1
                edges_r[e_idx] = (a, b)
                current_c4 += delta_total
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_adj_r = adj_r.copy()
            elif delta_total == 0:
                # Accept with some probability to explore
                if random.random() < 0.5:
                    adj_r[u, v] = 0
                    adj_r[v, u] = 0
                    adj_r[a, b] = 1
                    adj_r[b, a] = 1
                    edges_r[e_idx] = (a, b)
            else:
                # Reject with probability based on temperature
                if random.random() < np.exp(-delta_total / temp):
                    adj_r[u, v] = 0
                    adj_r[v, u] = 0
                    adj_r[a, b] = 1
                    adj_r[b, a] = 1
                    edges_r[e_idx] = (a, b)
                    current_c4 += delta_total
            
            temp *= cooling
            if temp < 0.01:
                temp = 0.01
        
        if best_c4 < best_c4_overall:
            best_c4_overall = best_c4
            best_adj_overall = best_adj_r.copy()
    
    if best_adj_overall is None:
        best_adj_overall = adj.copy()
    
    final_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj_overall[i, j] == 1:
                final_edges.append([i, j])
                
    return final_edges
