import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    n = 40
    best_edges = None
    best_count = 0
    
    def time_left():
        return deadline - time.time()
    
    def is_c6_free(adj, u, v):
        """Check if adding edge (u,v) to graph with adjacency matrix adj creates a C6.
        We need to check if there's a simple path of length 5 from u to v in the current graph."""
        # Fast check: if no walk of length 5, definitely no simple path
        # A^5[u][v] counts walks of length 5
        # For speed, do a bounded DFS
        # Path: u -> x1 -> x2 -> x3 -> x4 -> v, all distinct
        adj_local = adj.copy()
        adj_local[u, v] = 0  # don't use the edge we're adding
        adj_local[v, u] = 0
        
        # DFS from u, find path of length 5 to v
        # Depth 5 means 5 edges, 6 vertices
        # Use iterative DFS with path tracking
        stack = [(u, [u], 0)]
        while stack:
            node, path, depth = stack.pop()
            if depth == 5:
                if node == v:
                    return False  # found a C6
                continue
            if depth == 4:
                # One more edge needed, check if v is adjacent
                if adj_local[node, v] == 1 and v not in path:
                    return False  # found a C6
                continue
            for nb in range(n):
                if adj_local[node, nb] == 1 and nb not in path:
                    stack.append((nb, path + [nb], depth + 1))
        return True  # no C6 found
    
    def count_edges(adj):
        return int(np.sum(adj) // 2)
    
    def edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    edges.append([i, j])
        return edges
    
    def random_regular_adj(target_deg):
        """Generate a random graph with approximately target_deg degree per vertex."""
        adj = np.zeros((n, n), dtype=np.int8)
        # Start with random edges
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < target_deg / n:
                    adj[i, j] = 1
                    adj[j, i] = 1
        return adj
    
    def local_search(adj):
        """Improve graph by adding edges and doing swaps."""
        improved = True
        while improved and time_left() > 0.1:
            improved = False
            # Try adding edges
            for i in range(n):
                if time_left() < 0.1:
                    break
                for j in range(i+1, n):
                    if adj[i, j] == 0:
                        if is_c6_free(adj, i, j):
                            adj[i, j] = 1
                            adj[j, i] = 1
                            improved = True
                            break
                if improved:
                    break
            if improved:
                continue
            # Try 1-edge removal + 2-edge addition (swap)
            current_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i, j]:
                        current_edges.append((i, j))
            random.shuffle(current_edges)
            for (ri, rj) in current_edges:
                if time_left() < 0.1:
                    break
                adj[ri, rj] = 0
                adj[rj, ri] = 0
                added = 0
                for i in range(n):
                    if added >= 2:
                        break
                    for j in range(i+1, n):
                        if adj[i, j] == 0:
                            if is_c6_free(adj, i, j):
                                adj[i, j] = 1
                                adj[j, i] = 1
                                added += 1
                                if added >= 2:
                                    break
                if added == 2:
                    improved = True
                    break
                else:
                    # Revert
                    adj[ri, rj] = 1
                    adj[rj, ri] = 1
                    for i in range(n):
                        for j in range(i+1, n):
                            if adj[i, j] and (i,j) != (ri,rj):
                                # check if this was just added
                                pass
                    # Simpler: just revert the additions
                    for i in range(n):
                        if time_left() < 0.05:
                            break
                        for j in range(i+1, n):
                            if adj[i, j] and not (i == ri and j == rj) and not (i == rj and j == ri):
                                # We need to track which edges were added. This is messy.
                                pass
                    # Let me redo this more carefully
                    break
        return adj
    
    # Better local search with proper tracking
    def local_search_v2(adj):
        improved = True
        while improved and time_left() > 0.2:
            improved = False
            # Try adding edges
            for i in range(n):
                if time_left() < 0.2:
                    break
                for j in range(i+1, n):
                    if adj[i, j] == 0:
                        if is_c6_free(adj, i, j):
                            adj[i, j] = 1
                            adj[j, i] = 1
                            improved = True
                            break
                if improved:
                    break
            if improved:
                continue
            # Try swaps: remove one edge, add two
            current_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i, j]:
                        current_edges.append((i, j))
            random.shuffle(current_edges)
            for (ri, rj) in current_edges:
                if time_left() < 0.2:
                    break
                # Remove edge (ri, rj)
                adj[ri, rj] = 0
                adj[rj, ri] = 0
                added_edges = []
                for i in range(n):
                    if len(added_edges) >= 2:
                        break
                    for j in range(i+1, n):
                        if adj[i, j] == 0:
                            if is_c6_free(adj, i, j):
                                adj[i, j] = 1
                                adj[j, i] = 1
                                added_edges.append((i, j))
                                if len(added_edges) >= 2:
                                    break
                if len(added_edges) == 2:
                    improved = True
                    break
                else:
                    # Revert
                    for (ai, aj) in added_edges:
                        adj[ai, aj] = 0
                        adj[aj, ai] = 0
                    adj[ri, rj] = 1
                    adj[rj, ri] = 1
        return adj
    
    # Kick: random perturbation
    def kick(adj, num_kicks=5):
        for _ in range(num_kicks):
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i != j:
                adj[i, j] = random.randint(0, 1)
                adj[j, i] = adj[i, j]
        return adj
    
    # Initialize with random regular graph
    for attempt in range(200):
        if time_left() < 1.0:
            break
        target_deg = 5.0 + random.uniform(-1, 1)
        adj = random_regular_adj(target_deg)
        # Ensure it's C6-free (it should be for low degree, but verify)
        # Run local search
        adj = local_search_v2(adj.copy())
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_edges = edges_from_adj(adj)
            # Try kicks and more local search
            for kick_iter in range(5):
                if time_left() < 1.0:
                    break
                adj_kicked = kick(adj.copy(), num_kicks=random.randint(3, 8))
                adj_kicked = local_search_v2(adj_kicked)
                cnt_k = count_edges(adj_kicked)
                if cnt_k > best_count:
                    best_count = cnt_k
                    best_edges = edges_from_adj(adj_kicked)
                    adj = adj_kicked.copy()
    
    # Final: try to add more edges to best
    if best_edges is not None and time_left() > 1.0:
        adj_best = np.zeros((n, n), dtype=np.int8)
        for (u, v) in best_edges:
            adj_best[u, v] = 1
            adj_best[v, u] = 1
        adj_best = local_search_v2(adj_best)
        cnt_final = count_edges(adj_best)
        if cnt_final > best_count:
            best_count = cnt_final
            best_edges = edges_from_adj(adj_best)
    
    if best_edges is None:
        # Fallback: 8 disjoint K5s
        best_edges = []
        for k in range(8):
            best_edges += [[5*k + a, 5*k + b] for a in range(5) for b in range(a+1, 5)]
    
    return best_edges
