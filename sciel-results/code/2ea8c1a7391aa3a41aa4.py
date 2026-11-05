"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _has_triangle(n, edges_set, u, v):
    """Check if adding edge (u, v) creates a triangle."""
    # Find common neighbors of u and v
    # This is slow if done naively. We maintain adjacency sets.
    pass # Not used in the new approach directly, we check during construction

def _construct_maximal_triangle_free(n, seed):
    """Construct a maximal triangle-free graph on n vertices using random edge ordering."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    
    # Generate all possible edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    
    rng.shuffle(edges)
    
    for u, v in edges:
        # Check if adding (u, v) creates a triangle
        # A triangle is formed if there exists w such that (u, w) and (v, w) are in adj
        # Common neighbors of u and v
        if adj[u].isdisjoint(adj[v]):
            # No common neighbor, so no triangle
            adj[u].add(v)
            adj[v].add(u)
        # If they share a neighbor, adding (u, v) creates a triangle, so skip
    
    # Convert to edge list
    edge_list = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edge_list.append([i, j])
    return edge_list

def _check_alpha_less_than(n, adj, k=10):
    """
    Check if the graph has an independent set of size k.
    Returns True if NO independent set of size k exists (i.e., alpha < k).
    Uses branch and bound with bitmasks.
    """
    # Build adjacency masks
    adj_mask = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        adj_mask[i] = mask
    
    full_mask = (1 << n) - 1
    
    # We want to find an independent set of size k.
    # If we can't find one, alpha < k.
    
    best_found = 0 # Not really needed, we just need to know if size k is reachable
    
    def dfs(candidates, current_size, depth):
        # Pruning: if we can't reach size k, return False
        if current_size + bin(candidates).count('1') < k:
            return False
        
        if current_size >= k:
            return True
            
        if candidates == 0:
            return False
            
        # Heuristic: Pick a vertex with max degree in the candidate subgraph?
        # Or just pick the first one?
        # Picking a vertex with high degree in the *original* graph might be better for pruning the "include" branch?
        # Actually, for independent set, we want to remove vertices.
        # Let's pick the vertex with the maximum degree in the induced subgraph on candidates.
        
        # To keep it fast, let's just pick the least significant bit (vertex 0, then 1, etc.) 
        # But ordering matters. 
        # A simple improvement: sort candidates by degree in the candidate set?
        # That's expensive to compute each step.
        
        # Let's try picking the vertex with the highest degree in the *original* graph among candidates.
        # Precompute degrees?
        
        v = (candidates & -candidates).bit_length() - 1
        
        # Branch 1: Include v
        # New candidates must be non-neighbors of v
        new_candidates = candidates & ~adj_mask[v] & ~(1 << v)
        if dfs(new_candidates, current_size + 1, depth + 1):
            return True
        
        # Branch 2: Exclude v
        new_candidates = candidates & ~(1 << v)
        if dfs(new_candidates, current_size, depth + 1):
            return True
            
        return False

    return not dfs(full_mask, 0, 0)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    best_n = 0
    best_edges = []
    
    # We want to maximize n.
    # Try n from 40 down to 1.
    for n in range(40, 0, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple random seeds
        for seed in range(50):
            if time.time() > deadline:
                break
                
            edges = _construct_maximal_triangle_free(n, seed)
            
            # Build adjacency list for checking
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
                
            # Check if alpha < 10
            # This is the expensive part.
            # If n is large, this might be slow.
            # But if the graph is dense (maximal triangle-free), alpha is likely small.
            # If alpha >= 10, we discard this graph.
            
            if _check_alpha_less_than(n, adj, 10):
                best_n = n
                best_edges = edges
                found = True
                break
        
        if found:
            # Since we iterate n downwards, the first found is the largest n
            # We can break here if we want to be aggressive, but maybe we can find a better one for the same n?
            # The task says "Maximize n". So any valid graph for the largest n is optimal.
            break
            
    # Fallback if no graph found (should not happen for small n)
    if best_n == 0:
        # Small triangle-free graph, e.g., C5 (n=5) has alpha=2.
        # C5 is triangle-free.
        best_n = 5
        best_edges = [[0,1], [1,2], [2,3], [3,4], [4,0]]
        
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
