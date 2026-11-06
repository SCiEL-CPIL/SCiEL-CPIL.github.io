"""Optimized K2,3-free graph construction on 20 vertices."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 20
    
    def popcount(x):
        """Count set bits in integer x."""
        return bin(x).count('1')
    
    def common_neighbors_count(adj, u, v):
        """Count common neighbors of u and v using bitsets."""
        return popcount(adj[u] & adj[v])
    
    def is_k23_free(adj):
        """Check if the graph has no K2,3 subgraph."""
        for u in range(n):
            for v in range(u + 1, n):
                if common_neighbors_count(adj, u, v) >= 3:
                    return False
        return True
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps the graph K2,3-free."""
        if u == v:
            return False
        if (adj[u] >> v) & 1:
            return False  # Edge already exists
        
        # Check pair (u, v):
        if popcount(adj[u] & adj[v]) >= 3:
            return False
        
        # Check pairs (v, y) for y in neighbors of u
        for y in range(n):
            if y == u or y == v:
                continue
            if (adj[u] >> y) & 1:  # y is a neighbor of u
                if popcount(adj[v] & adj[y]) >= 2:
                    return False
        
        # Check pairs (u, x) for x in neighbors of v
        for x in range(n):
            if x == u or x == v:
                continue
            if (adj[v] >> x) & 1:  # x is a neighbor of v
                if popcount(adj[u] & adj[x]) >= 2:
                    return False
        
        return True
    
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def edge_count(adj):
        total = 0
        for i in range(n):
            total += popcount(adj[i])
        return total // 2
    
    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges
    
    # Greedy construction with randomization
    def greedy_build(seed):
        rng = random.Random(seed)
        adj = [0] * n
        
        # Try to add edges in random order
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_edges)
        
        for u, v in all_edges:
            if can_add_edge(adj, u, v):
                add_edge(adj, u, v)
        
        # Saturate: try to add any remaining valid edges
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1) and can_add_edge(adj, u, v):
                        add_edge(adj, u, v)
                        changed = True
                        if time.time() > deadline - 2:
                            return adj
        return adj
    
    # Improved local search with proper state management
    def local_search(adj, rng, max_steps=5000):
        adj = adj[:]  # Copy
        best_count = edge_count(adj)
        best_adj = adj[:]
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # Try to add any single missing edge
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1) and can_add_edge(adj, u, v):
                        add_edge(adj, u, v)
                        best_count += 1
                        best_adj = adj[:]
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # Try 1-for-2 swap: remove one edge, try to add two others
            edges_list = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        edges_list.append((u, v))
            
            if not edges_list:
                break
            
            # Randomly select an edge to remove
            ru, rv = rng.choice(edges_list)
            
            # Save state for potential revert
            saved_adj = adj[:]
            
            remove_edge(adj, ru, rv)
            
            # Try to add two new edges
            added_edges = []
            candidates = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1) and (u, v) != (ru, rv):
                        candidates.append((u, v))
            
            rng.shuffle(candidates)
            
            for u, v in candidates:
                if len(added_edges) >= 2:
                    break
                if can_add_edge(adj, u, v):
                    add_edge(adj, u, v)
                    added_edges.append((u, v))
            
            new_count = edge_count(adj)
            
            if new_count > best_count:
                best_count = new_count
                best_adj = adj[:]
            else:
                # Revert if not improved
                adj = saved_adj[:]
            
        return best_adj
    
    # MILP-based approach
    def milp_construct():
        """Try to find dense K2,3-free graph using MILP."""
        if time.time() > deadline - 30:
            return None
        
        # Variables: x_{u,v} for each edge (u,v), u < v
        edge_index = {}
        edge_list = []
        for u in range(n):
            for v in range(u + 1, n):
                idx = len(edge_list)
                edge_index[(u, v)] = idx
                edge_list.append((u, v))
        
        m = len(edge_list)  # Number of variables
        
        # Objective: maximize sum of x_{u,v}
        c = -np.ones(m)  # Minimize negative of sum
        
        # Constraints: For each pair (u,v), sum of x_{w,z} where w,z are common neighbors <= 2
        # For pair (u,v), common neighbors are vertices w such that (u,w) and (v,w) are both edges
        # This means: for each pair (u,v), sum over w of x_{u,w} * x_{v,w} <= 2
        # This is a quadratic constraint, not linear. We need to linearize.
        
        # Alternative: For each triple (u,v,w) where w is a potential common neighbor,
        # we need: x_{u,w} + x_{v,w} - y_{u,v,w} <= 1 where y_{u,v,w} indicates
        # that both edges exist. But this gets complex.
        
        # Simpler approach: For each pair (u,v) and each vertex w (w != u, v),
        # we can't have all three edges (u,w), (v,w), and (u,v) form a K2,3 with another vertex.
        # Actually, K2,3 means two vertices u,v with three common neighbors.
        # So for each pair (u,v), the number of vertices w such that both (u,w) and (v,w) are edges must be <= 2.
        
        # This is: for each pair (u,v), sum_w x_{u,w} * x_{v,w} <= 2
        # To linearize: introduce z_{u,v,w} = x_{u,w} * x_{v,w}
        # Constraints: z_{u,v,w} <= x_{u,w}, z_{u,v,w} <= x_{v,w}, z_{u,v,w} >= x_{u,w} + x_{v,w} - 1
        
        # Number of z variables: for each pair (u,v) and each w != u,v: C(20,2) * 18 = 190 * 18 = 3420
        # Total variables: 190 + 3420 = 3610
        
        # This might be too large for quick solving. Let's try a simpler heuristic-based approach instead.
        
        # For now, return None to skip MILP if it's too complex
        return None
    
    # Run multiple greedy builds and pick the best
    best_adj = None
    best_count = 0
    
    # Try a few different seeds
    num_trials = 30
    for trial in range(num_trials):
        if time.time() > deadline - 15:
            break
        
        adj = greedy_build(trial * 12345 + 678)
        count = edge_count(adj)
        
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    # Local search on the best result
    if best_adj is not None and time.time() < deadline - 20:
        rng = random.Random(42)
        best_adj = local_search(best_adj, rng, max_steps=5000)
    
    # Try MILP if we have time
    if time.time() < deadline - 30:
        milp_result = milp_construct()
        if milp_result is not None:
            milp_count = edge_count(milp_result)
            if milp_count > best_count:
                best_adj = milp_result
                best_count = milp_count
    
    # Verify the result is valid
    if best_adj is None:
        return []
    
    if not is_k23_free(best_adj):
        return []
    
    return adj_to_edges(best_adj)

# EVOLVE-BLOCK-END
