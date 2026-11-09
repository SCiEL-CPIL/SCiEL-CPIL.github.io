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
        # Only non-adjacent pairs form C4s with 2 common neighbors
        # But we need to be careful: the standard formula counts all pairs
        # C4 count = sum over all pairs (i,j) of C(common(i,j), 2) / 2
        # But this overcounts if i,j are adjacent (forms triangles not C4s)
        # Actually for subgraph C4s, the formula is:
        # sum over all unordered pairs {i,j} of C(common_neighbors(i,j), 2)
        # Each C4 is counted twice (once for each pair of opposite vertices)
        # So divide by 2
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
    
    # Precompute neighbor sets for faster common neighbor calculation
    def get_neighbors(adj, v):
        return set(np.where(adj[v] == 1)[0])
    
    def common_neighbors(adj, u, v):
        return np.sum(adj[u] & adj[v])
    
    # Adjust edge count to E=91
    # If too many edges, remove edges that are in the most C4s
    # If too few, add edges that create the fewest C4s
    
    def c4_contribution_of_edge(adj, u, v):
        """Count C4s that contain edge (u,v)."""
        # C4s containing (u,v): paths of length 3 from u to v not using (u,v)
        # i.e., u-a-b-v where a~u, b~v, a~b, a!=v, b!=u
        count = 0
        for a in range(N):
            if a == u or a == v:
                continue
            if adj[u, a] == 1:
                for b in range(N):
                    if b == u or b == v or b == a:
                        continue
                    if adj[v, b] == 1 and adj[a, b] == 1:
                        count += 1
        # Each C4 is counted once for this edge
        return count
    
    def c4_increase_from_adding_edge(adj, u, v):
        """Count new C4s created by adding edge (u,v)."""
        # New C4s must use edge (u,v)
        # Same as c4_contribution_of_edge but for non-edge
        count = 0
        for a in range(N):
            if a == u or a == v:
                continue
            if adj[u, a] == 1:
                for b in range(N):
                    if b == u or b == v or b == a:
                        continue
                    if adj[v, b] == 1 and adj[a, b] == 1:
                        count += 1
        return count
    
    # Greedy adjustment
    while current_edges > E:
        best_edge = None
        max_contrib = -1
        for i in range(N):
            for j in range(i+1, N):
                if adj[i,j] == 1:
                    contrib = c4_contribution_of_edge(adj, i, j)
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
                    increase = c4_increase_from_adding_edge(adj, i, j)
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
    
    # Local search: try edge swaps (remove one edge, add another)
    # Use exact C4 delta calculation
    
    def try_swap(adj, remove_u, remove_v, add_u, add_v):
        """Try swapping edge (remove_u, remove_v) with edge (add_u, add_v)."""
        # Remove edge
        adj[remove_u, remove_v] = 0
        adj[remove_v, remove_u] = 0
        # Add edge
        adj[add_u, add_v] = 1
        adj[add_v, add_u] = 1
        c4 = count_c4(adj)
        # Restore
        adj[add_u, add_v] = 0
        adj[add_v, add_u] = 0
        adj[remove_u, remove_v] = 1
        adj[remove_v, remove_u] = 1
        return c4
    
    # Multiple rounds of local search with random restarts
    no_improvement = 0
    max_no_improve = 30
    
    while time.time() < deadline:
        improved = False
        edges = get_edges(adj)
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                break
            
            # Try removing (u,v) and adding a random non-edge
            non_edges = []
            for i in range(N):
                for j in range(i+1, N):
                    if adj[i,j] == 0:
                        non_edges.append((i,j))
            
            if not non_edges:
                break
            
            random.shuffle(non_edges)
            # Try a limited number of candidates for speed
            candidates = non_edges[:20]
            
            best_add = None
            best_c4_after = float('inf')
            
            for a, b in candidates:
                c4_after = try_swap(adj, u, v, a, b)
                if c4_after < best_c4_after:
                    best_c4_after = c4_after
                    best_add = (a, b)
            
            if best_add and best_c4_after < best_c4:
                # Commit the swap
                adj[u, v] = 0
                adj[v, u] = 0
                adj[best_add[0], best_add[1]] = 1
                adj[best_add[1], best_add[0]] = 1
                best_c4 = best_c4_after
                improved = True
                no_improvement = 0
                if best_c4 < best_c4:
                    best_c4 = best_c4
                    best_adj = adj.copy()
        if not improved:
            no_improvement += 1
            if no_improvement >= max_no_improve:
                # Try a random restart: perturb the current best
                if time.time() < deadline - 10:
                    adj = best_adj.copy()
                    best_c4 = count_c4(adj)
                    # Randomly swap a few edges
                    for _ in range(5):
                        if time.time() > deadline:
                            break
                        edges_now = get_edges(adj)
                        random.shuffle(edges_now)
                        u, v = random.choice(edges_now)
                        non_edges_now = []
                        for i in range(N):
                            for j in range(i+1, N):
                                if adj[i,j] == 0:
                                    non_edges_now.append((i,j))
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
