"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n = 16
    target_edges = 65
    
    def build_adj(edges):
        """Build adjacency bitmasks and degrees from edge list."""
        adj = [0] * n
        deg = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg
    
    def compute_score_from_adj(adj, deg):
        """Compute max triangle degree sum using bitmasks."""
        max_sum = 0
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    continue
                # Common neighbors of i and j, with index > j
                common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
                while common:
                    k = common & (-common)
                    k_idx = k.bit_length() - 1
                    s = deg[i] + deg[j] + deg[k_idx]
                    if s > max_sum:
                        max_sum = s
                    common &= common - 1
        return max_sum
    
    def compute_score(edges):
        """Compute the maximum triangle degree sum."""
        adj, deg = build_adj(edges)
        return compute_score_from_adj(adj, deg)
    
    def is_valid(edges):
        """Check if edges form a simple graph with exactly target_edges edges."""
        if len(edges) != target_edges:
            return False
        seen = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in seen:
                return False
            seen.add(key)
        return True
    
    def local_search(edges, max_iters=20000):
        """Try to improve score via edge swaps using bitmask adjacency."""
        current = edges[:]
        adj, deg = build_adj(current)
        current_score = compute_score_from_adj(adj, deg)
        
        # Precompute non-edges for faster random sampling
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 5.0:
                break
            
            # Try removing a random edge and adding a random non-edge
            edge_idx = random.randint(0, len(current) - 1)
            u1, v1 = current[edge_idx]
            
            # Try several random non-edges
            for _ in range(15):
                u2 = random.randint(0, n - 1)
                v2 = random.randint(0, n - 1)
                if u2 == v2:
                    continue
                u2, v2 = min(u2, v2), max(u2, v2)
                if (u2, v2) == (u1, v1):
                    continue
                if (adj[u2] >> v2) & 1:
                    continue
                
                # Perform swap: remove (u1,v1), add (u2,v2)
                new_adj = adj[:]
                new_deg = deg[:]
                
                # Remove edge (u1, v1)
                new_adj[u1] &= ~(1 << v1)
                new_adj[v1] &= ~(1 << u1)
                new_deg[u1] -= 1
                new_deg[v1] -= 1
                
                # Add edge (u2, v2)
                new_adj[u2] |= (1 << v2)
                new_adj[v2] |= (1 << u2)
                new_deg[u2] += 1
                new_deg[v2] += 1
                
                new_score = compute_score_from_adj(new_adj, new_deg)
                
                if new_score < current_score:
                    # Accept the move
                    adj = new_adj
                    deg = new_deg
                    current_score = new_score
                    current[edge_idx] = [u2, v2]
                    improved = True
                    break
        
        return current, current_score
    
    # Base construction: K_{8,8} + one edge inside part A
    # Vertices 0-7: part A, vertices 8-15: part B
    base_edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    
    # Verify base
    if not is_valid(base_edges):
        raise ValueError("Base construction is invalid")
    
    base_score = compute_score(base_edges)
    best_edges = base_edges[:]
    best_score = base_score
    
    # Run local search from the base
    if time.time() < deadline - 15.0:
        ls_edges, ls_score = local_search(base_edges, max_iters=30000)
        if ls_score < best_score:
            best_score = ls_score
            best_edges = ls_edges[:]
    
    # Multiple restarts with diverse perturbations
    restart_count = 0
    while time.time() < deadline - 10.0 and restart_count < 40:
        restart_count += 1
        
        # Create perturbation: swap 2-5 random edges
        perturbed = best_edges[:]
        num_swaps = random.randint(2, 5)
        
        adj_p, deg_p = build_adj(perturbed)
        
        for _ in range(num_swaps):
            # Remove a random edge
            rem_idx = random.randint(0, len(perturbed) - 1)
            u1, v1 = perturbed[rem_idx]
            
            # Find a random non-edge to add
            added = False
            for _ in range(50):
                u2 = random.randint(0, n - 1)
                v2 = random.randint(0, n - 1)
                if u2 == v2:
                    continue
                u2, v2 = min(u2, v2), max(u2, v2)
                if (u2, v2) == (u1, v1):
                    continue
                if (adj_p[u2] >> v2) & 1:
                    continue
                
                # Perform swap
                adj_p[u1] &= ~(1 << v1)
                adj_p[v1] &= ~(1 << u1)
                deg_p[u1] -= 1
                deg_p[v1] -= 1
                adj_p[u2] |= (1 << v2)
                adj_p[v2] |= (1 << u2)
                deg_p[u2] += 1
                deg_p[v2] += 1
                perturbed[rem_idx] = [u2, v2]
                added = True
                break
            
            if not added:
                # Revert
                adj_p[u1] |= (1 << v1)
                adj_p[v1] |= (1 << u1)
                deg_p[u1] += 1
                deg_p[v1] += 1
                perturbed[rem_idx] = [u1, v1]
        
        if not is_valid(perturbed):
            continue
        
        pert_score = compute_score_from_adj(adj_p, deg_p)
        
        if pert_score < best_score:
            best_score = pert_score
            best_edges = perturbed[:]
        
        # Local search from perturbation
        if time.time() < deadline - 10.0:
            ls_edges2, ls_score2 = local_search(perturbed, max_iters=8000)
            if ls_score2 < best_score:
                best_score = ls_score2
                best_edges = ls_edges2[:]
    
    # Final verification
    if not is_valid(best_edges):
        best_edges = base_edges[:]
        best_score = base_score
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
