import numpy as np
import time
import random

N = 31
E = 91
DEADLINE_SECONDS = 115.0

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE_SECONDS
    
    # Generate points of PG(2,5)
    points = []
    for x in range(5):
        for y in range(5):
            points.append((x, y, 1))
    for x in range(5):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    # Construct Polarity Graph
    adj = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        for j in range(i + 1, N):
            p1 = points[i]
            p2 = points[j]
            dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % 5
            if dot == 0:
                adj[i, j] = 1
                adj[j, i] = 1
    
    current_edges = int(np.sum(adj) // 2)
    
    # Fast C4 count using matrix multiplication
    def count_c4(adj):
        A2 = adj.astype(np.int64) @ adj.astype(np.int64)
        i_idx, j_idx = np.triu_indices(N, k=1)
        common = A2[i_idx, j_idx]
        terms = common * (common - 1) // 2
        total = int(np.sum(terms))
        return total // 2
    
    def get_edges(adj):
        edges = []
        for i in range(N):
            for j in range(i+1, N):
                if adj[i,j] == 1:
                    edges.append((i,j))
        return edges
    
    def add_edge(adj, u, v):
        adj[u, v] = 1
        adj[v, u] = 1
    
    def remove_edge(adj, u, v):
        adj[u, v] = 0
        adj[v, u] = 0
    
    # Incremental C4 delta: compute change in C4 count when adding/removing edge (u,v)
    def c4_delta_add(adj, u, v):
        """Return increase in C4 count when adding edge (u,v) to graph where it's absent."""
        # New C4s must contain edge (u,v)
        # A C4 containing (u,v) is: u-a-b-v where a~u, b~v, a~b, a!=v, b!=u, a!=b
        # Count such paths
        delta = 0
        for a in range(N):
            if a == u or a == v:
                continue
            if adj[u, a] == 1:
                for b in range(N):
                    if b == u or b == v or b == a:
                        continue
                    if adj[v, b] == 1 and adj[a, b] == 1:
                        delta += 1
        return delta
    
    def c4_delta_remove(adj, u, v):
        """Return decrease in C4 count when removing edge (u,v) from graph where it's present."""
        # C4s that contain edge (u,v): u-a-b-v where a~u, b~v, a~b, a!=v, b!=u, a!=b
        # After removal, these C4s are gone
        delta = 0
        for a in range(N):
            if a == u or a == v:
                continue
            if adj[u, a] == 1:
                for b in range(N):
                    if b == u or b == v or b == a:
                        continue
                    if adj[v, b] == 1 and adj[a, b] == 1:
                        delta += 1
        return delta
    
    # Adjust edge count to E=91
    while current_edges > E:
        best_edge = None
        max_contrib = -1
        for i in range(N):
            for j in range(i+1, N):
                if adj[i,j] == 1:
                    contrib = c4_delta_remove(adj, i, j)
                    if contrib > max_contrib:
                        max_contrib = contrib
                        best_edge = (i, j)
        if best_edge:
            remove_edge(adj, best_edge[0], best_edge[1])
            current_edges -= 1
        else:
            break
    
    while current_edges < E:
        best_edge = None
        min_increase = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                if adj[i,j] == 0:
                    increase = c4_delta_add(adj, i, j)
                    if increase < min_increase:
                        min_increase = increase
                        best_edge = (i, j)
                        if min_increase == 0:
                            break
            if min_increase == 0:
                break
        if best_edge:
            add_edge(adj, best_edge[0], best_edge[1])
            current_edges += 1
        else:
            break
    
    current_c4 = count_c4(adj)
    best_c4 = current_c4
    best_adj = adj.copy()
    
    # Precompute list of non-edges for faster access
    def get_non_edges(adj):
        non_edges = []
        for i in range(N):
            for j in range(i+1, N):
                if adj[i,j] == 0:
                    non_edges.append((i,j))
        return non_edges
    
    # Local search: try edge swaps with incremental C4 delta
    no_improvement = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        improved = False
        edges = get_edges(adj)
        random.shuffle(edges)
        non_edges = get_non_edges(adj)
        random.shuffle(non_edges)
        
        for u, v in edges:
            if time.time() > deadline - 2:
                break
            
            # Try swapping (u,v) with each non-edge
            # Delta = -c4_delta_remove(u,v) + c4_delta_add(a,b) after removing (u,v)
            # But we need to be careful: after removing (u,v), the delta for adding (a,b) changes
            
            best_add = None
            best_delta = float('inf')
            
            # Limit candidates for speed
            candidates = non_edges[:40]
            
            for a, b in candidates:
                # Check if (a,b) is still a non-edge (it should be, we haven't modified adj)
                if adj[a, b] == 1:
                    continue
                
                # Compute delta: remove (u,v), add (a,b)
                # First, compute C4s lost by removing (u,v)
                delta_remove = c4_delta_remove(adj, u, v)
                
                # Then, compute C4s gained by adding (a,b) to graph without (u,v)
                # Temporarily remove (u,v)
                adj[u, v] = 0
                adj[v, u] = 0
                delta_add = c4_delta_add(adj, a, b)
                # Restore
                adj[u, v] = 1
                adj[v, u] = 1
                
                net_delta = delta_add - delta_remove
                
                if net_delta < best_delta:
                    best_delta = net_delta
                    best_add = (a, b)
            
            if best_add and best_delta < 0:
                # Commit the swap
                adj[u, v] = 0
                adj[v, u] = 0
                adj[best_add[0], best_add[1]] = 1
                adj[best_add[1], best_add[0]] = 1
                current_c4 += best_delta
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_adj = adj.copy()
                improved = True
                no_improvement = 0
                # Update non_edges list
                non_edges = get_non_edges(adj)
                random.shuffle(non_edges)
        
        if not improved:
            no_improvement += 1
            if no_improvement >= max_no_improve:
                # Try a random restart: perturb the current best
                if time.time() < deadline - 15:
                    adj = best_adj.copy()
                    current_c4 = count_c4(adj)
                    # Randomly swap a few edges
                    for _ in range(3):
                        if time.time() > deadline - 5:
                            break
                        edges_now = get_edges(adj)
                        random.shuffle(edges_now)
                        u, v = random.choice(edges_now)
                        non_edges_now = get_non_edges(adj)
                        if non_edges_now:
                            a, b = random.choice(non_edges_now)
                            adj[u, v] = 0
                            adj[v, u] = 0
                            adj[a, b] = 1
                            adj[b, a] = 1
                    no_improvement = 0
    
    # Return the best solution found
    if count_c4(best_adj) < count_c4(adj):
        adj = best_adj
    
    return get_edges(adj)

if __name__ == "__main__":
    edges = construct()
    print(f"Edges: {len(edges)}")
    adj = np.zeros((N, N), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    def count_c4_final(adj):
        A2 = adj.astype(np.int64) @ adj.astype(np.int64)
        i_idx, j_idx = np.triu_indices(N, k=1)
        common = A2[i_idx, j_idx]
        terms = common * (common - 1) // 2
        total = int(np.sum(terms))
        return total // 2
    print(f"C4s: {count_c4_final(adj)}")
