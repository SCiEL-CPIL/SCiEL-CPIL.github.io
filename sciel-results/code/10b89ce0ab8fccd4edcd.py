import random
import time
import sys
import numpy as np

def construct():
    n = 20
    start = time.time()
    deadline = start + 115.0  # safety margin under 120s limit
    rng = random.Random(42)

    def verify(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            if u == v or not (0 <= u < n and 0 <= v < n):
                return None
            if v in adj[u]:
                return None  # duplicate
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            for v in range(u + 1, n):
                if len(adj[u] & adj[v]) > 2:
                    return None
        return edges

    # Fallback: 5 disjoint K4s (10 edges)
    def fallback():
        e = []
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    e.append([base + a, base + b])
        return e

    best = None
    best_e = 0

    # ---- Vectorized NumPy approach ----
    # Use adjacency matrix and common-neighbor count matrix for fast checks
    
    def compute_common(adj_mat):
        """Compute common neighbor counts from adjacency matrix."""
        # common[u][v] = number of common neighbors of u and v
        # common = A @ A - D (diagonal), but we need off-diagonal
        common = np.dot(adj_mat, adj_mat)
        # Subtract diagonal (self-loops in the product)
        np.fill_diagonal(common, 0)
        return common
    
    def can_add_vec(adj_mat, common, u, v):
        """Check if edge (u,v) can be added without creating K2,3."""
        if adj_mat[u, v] == 1:
            return False
        # Adding u-v: for any w in adj[u], common[w][v] must be <= 1
        # because u becomes a common neighbor of w and v
        # Similarly for w in adj[v], common[u][w] must be <= 1
        adj_u = np.where(adj_mat[u] == 1)[0]
        adj_v = np.where(adj_mat[v] == 1)[0]
        
        # Check neighbors of u
        if len(adj_u) > 0:
            if np.any(common[adj_u, v] > 1):
                return False
        # Check neighbors of v
        if len(adj_v) > 0:
            if np.any(common[u, adj_v] > 1):
                return False
        return True
    
    def greedy_build_np(seed_rng):
        """Build K2,3-free graph using vectorized operations."""
        adj_mat = np.zeros((n, n), dtype=np.int8)
        common = np.zeros((n, n), dtype=np.int32)
        
        edges_to_try = [(u, v) for u in range(n) for v in range(u+1, n)]
        seed_rng.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            if adj_mat[u, v] == 1:
                continue
            
            # Check if we can add this edge
            adj_u = np.where(adj_mat[u] == 1)[0]
            adj_v = np.where(adj_mat[v] == 1)[0]
            
            valid = True
            if len(adj_u) > 0:
                if np.any(common[adj_u, v] > 1):
                    valid = False
            if valid and len(adj_v) > 0:
                if np.any(common[u, adj_v] > 1):
                    valid = False
            
            if valid:
                # Add edge
                adj_mat[u, v] = 1
                adj_mat[v, u] = 1
                
                # Update common neighbor counts
                # For each w in adj[u] (excluding v): common[w][v] += 1
                for w in adj_u:
                    common[w, v] += 1
                    common[v, w] += 1
                # For each w in adj[v] (excluding u): common[u][w] += 1
                for w in adj_v:
                    common[u, w] += 1
                    common[w, u] += 1
        
        # Extract edges
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj_mat[u, v] == 1:
                    edges.append([u, v])
        return edges
    
    def local_search_np(edges, max_iters=200):
        """Local search using NumPy for efficiency."""
        adj_mat = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            adj_mat[u, v] = 1
            adj_mat[v, u] = 1
        
        common = compute_common(adj_mat)
        
        current_edges = len(edges)
        improved = True
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline - 10.0:
            improved = False
            iters += 1
            
            # Find all candidate edges that can be added
            candidates = []
            for u in range(n):
                for v in range(u+1, n):
                    if adj_mat[u, v] == 0:
                        adj_u = np.where(adj_mat[u] == 1)[0]
                        adj_v = np.where(adj_mat[v] == 1)[0]
                        
                        valid = True
                        if len(adj_u) > 0:
                            if np.any(common[adj_u, v] > 1):
                                valid = False
                        if valid and len(adj_v) > 0:
                            if np.any(common[u, adj_v] > 1):
                                valid = False
                        
                        if valid:
                            candidates.append((u, v))
            
            if candidates:
                # Add a random candidate
                u, v = rng.choice(candidates)
                adj_mat[u, v] = 1
                adj_mat[v, u] = 1
                
                # Update common
                adj_u = np.where(adj_mat[u] == 1)[0]
                adj_v = np.where(adj_mat[v] == 1)[0]
                for w in adj_u:
                    if w != v:
                        common[w, v] += 1
                        common[v, w] += 1
                for w in adj_v:
                    if w != u:
                        common[u, w] += 1
                        common[w, u] += 1
                improved = True
            else:
                # No edge can be added; try removing a random edge
                edge_list = [(u, v) for u in range(n) for v in range(u+1, n) if adj_mat[u, v] == 1]
                if edge_list and len(edge_list) > 15:
                    u, v = rng.choice(edge_list)
                    adj_mat[u, v] = 0
                    adj_mat[v, u] = 0
                    
                    # Update common
                    adj_u = np.where(adj_mat[u] == 1)[0]
                    adj_v = np.where(adj_mat[v] == 1)[0]
                    for w in adj_u:
                        common[w, v] -= 1
                        common[v, w] -= 1
                    for w in adj_v:
                        common[u, w] -= 1
                        common[w, u] -= 1
                    improved = True
        
        # Extract edges
        result = []
        for u in range(n):
            for v in range(u+1, n):
                if adj_mat[u, v] == 1:
                    result.append([u, v])
        return result
    
    # Run multiple greedy builds with different seeds
    for trial in range(30):
        if time.time() > deadline - 15.0:
            break
        
        trial_rng = random.Random(trial * 12345 + 6789)
        edges = greedy_build_np(trial_rng)
        edges = verify(edges)
        
        if edges is not None:
            # Try local search to improve
            improved = local_search_np(edges, max_iters=150)
            improved = verify(improved)
            
            if improved is not None and len(improved) > best_e:
                best_e = len(improved)
                best = improved
            elif edges is not None and len(edges) > best_e:
                best_e = len(edges)
                best = edges
    
    # Final verification
    if best is not None:
        best = verify(best)
    
    if best is None:
        return fallback()
    return best

if __name__ == "__main__":
    result = construct()
    print(f"Edges: {len(result)}")
