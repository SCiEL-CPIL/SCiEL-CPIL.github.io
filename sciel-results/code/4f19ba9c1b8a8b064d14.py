"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill
    
    n = 16
    target_edges = 65
    
    # Build adjacency matrix
    # Start with K_{8,8} + one internal edge
    adj = np.zeros((n, n), dtype=np.uint8)
    # K_{8,8}: vertices 0-7 vs 8-15
    for u in range(8):
        for v in range(8, 16):
            adj[u, v] = 1
            adj[v, u] = 1
    # Add one internal edge
    adj[0, 1] = 1
    adj[1, 0] = 1
    
    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    edges.append([u, v])
        return edges
    
    def compute_score(adj):
        """Compute max triangle degree sum. Vectorized."""
        # Find all triangles: for each edge (u,v), common neighbors w
        deg = adj.sum(axis=1).astype(np.int32)
        # Triangles: (u,v,w) where u<v<w and all three edges exist
        # Use matrix multiplication: (A^2)[u,v] = number of common neighbors
        A2 = adj @ adj
        max_score = 0
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    # Common neighbors of u and v
                    common = np.where((adj[u] & adj[v]) > 0)[0]
                    for w in common:
                        if w > v:  # u < v < w to avoid duplicates
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_score:
                                max_score = s
        return max_score
    
    def try_swap(adj, u1, v1, u2, v2):
        """Try removing edge (u1,v1) and adding edge (u2,v2).
        Returns new adj if valid, else None."""
        if adj[u1, v1] == 0:
            return None
        if adj[u2, v2] == 1:
            return None
        if u1 == v1 or u2 == v2:
            return None
        # Check we don't create self-loops or duplicate
        new_adj = adj.copy()
        new_adj[u1, v1] = 0
        new_adj[v1, u1] = 0
        new_adj[u2, v2] = 1
        new_adj[v2, u2] = 1
        # Verify edge count
        if new_adj.sum() != adj.sum():
            return None
        return new_adj
    
    # Initial score
    current_score = compute_score(adj)
    best_adj = adj.copy()
    best_score = current_score
    
    # Simulated Annealing
    T = 5.0
    T_min = 0.01
    cooling = 0.9995
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    step = 0
    while time.time() < deadline - 5:
        step += 1
        
        # Pick a random edge to remove
        existing_edges = [(u, v) for u in range(n) for v in range(u+1, n) if adj[u, v]]
        if not existing_edges:
            break
        u1, v1 = random.choice(existing_edges)
        
        # Pick a random edge to add
        u2, v2 = random.choice(all_edges)
        
        new_adj = try_swap(adj, u1, v1, u2, v2)
        if new_adj is None:
            continue
        
        new_score = compute_score(new_adj)
        
        if new_score < current_score:
            adj = new_adj
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_adj = adj.copy()
        else:
            # Accept with probability exp(-(delta)/T)
            delta = new_score - current_score
            if delta > 0:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    adj = new_adj
                    current_score = new_score
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Final hill-climbing polish
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for u1 in range(n):
            for v1 in range(u1+1, n):
                if not adj[u1, v1]:
                    continue
                for u2 in range(n):
                    for v2 in range(u2+1, n):
                        if adj[u2, v2]:
                            continue
                        new_adj = try_swap(adj, u1, v1, u2, v2)
                        if new_adj is None:
                            continue
                        new_score = compute_score(new_adj)
                        if new_score < current_score:
                            adj = new_adj
                            current_score = new_score
                            if current_score < best_score:
                                best_score = current_score
                                best_adj = adj.copy()
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    # Return best found
    final_adj = best_adj
    edges = get_edges(final_adj)
    
    # Verify
    assert len(edges) == target_edges, f"Expected {target_edges} edges, got {len(edges)}"
    assert final_adj.sum() == 2 * target_edges
    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
