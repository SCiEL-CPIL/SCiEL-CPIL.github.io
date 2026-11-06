"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_triangle_free(n, edges):
    """Check if graph is triangle-free using adjacency sets."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w > v and w in adj[u]:
                    return False
    return True

def _max_independent_set_size(n, edges):
    """Exact branch and bound for maximum independent set size."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Use bitmasks for efficiency
    # Build adjacency bitmask
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    best = 0
    
    def branch(candidates, size):
        nonlocal best
        if size + len(candidates) <= best:
            return
        if not candidates:
            best = max(best, size)
            return
        
        # Pick a vertex with maximum degree in the subgraph induced by candidates
        # For independent set, we want vertices with few neighbors in candidates
        # Actually for independent set, we pick vertex and either include or exclude
        # Let's use a simple branch and bound
        
        # Greedy upper bound: size + number of candidates
        # Better: use a greedy coloring bound on the complement
        # For now, simple bound
        
        # Pick a vertex
        # Choose vertex with minimum degree in candidate subgraph for better branching
        min_deg = float('inf')
        chosen = -1
        for v in candidates:
            deg = 0
            for w in candidates:
                if w != v and (adj_mask[v] >> w) & 1:
                    deg += 1
            if deg < min_deg:
                min_deg = deg
                chosen = v
        
        # Branch 1: include chosen
        new_cand = [v for v in candidates if v != chosen and not (adj_mask[chosen] >> v) & 1]
        branch(new_cand, size + 1)
        
        # Branch 2: exclude chosen
        new_cand = [v for v in candidates if v != chosen]
        branch(new_cand, size)
    
    branch(list(range(n)), 0)
    return best

def _validate(n, edges):
    """Check both conditions."""
    if not _is_triangle_free(n, edges):
        return False
    max_ind = _max_independent_set_size(n, edges)
    return max_ind < 10

def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s kill
    
    # Try known constructions for larger n
    # The seed gives n=26. We know R(3,10) <= 41, so max valid n is 40.
    # Known results: R(3,10) is between 39 and 41.
    # So n=39 is known to be valid (largest known order).
    # Let's try to construct or verify graphs for n=39, 40.
    
    # For now, let's start with the known Andrasfai-type constructions or random search
    # Actually, let me try a random local search approach to find triangle-free graphs
    # with no independent set of size 10.
    
    best_n = 26
    best_edges = [[i, j] for i in range(26) for j in range(i + 1, 26) if (j - i) % 26 % 3 == 1]
    
    # Verify the seed
    if _validate(best_n, best_edges):
        pass  # It's valid
    else:
        # Fallback to empty graph if seed fails (shouldn't happen)
        best_n = 1
        best_edges = []
    
    # Try to improve by adding vertices or using random search
    # For larger n, we need more sophisticated approaches.
    # Let's try random graphs with triangle-free constraint and check independent set.
    
    # Strategy: For each target n from 39 down to 27, try random triangle-free graphs
    # and check if alpha < 10.
    
    for target_n in [39, 40, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27]:
        if time.time() > deadline - 30:
            break
        
        found = False
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts and not found:
            attempts += 1
            if time.time() > deadline - 30:
                break
            
            # Generate random triangle-free graph
            # Start with empty graph, add edges that don't create triangles
            n = target_n
            edges = []
            adj = [set() for _ in range(n)]
            
            # Randomly order vertices and add edges
            # Use a random permutation
            perm = list(range(n))
            random.shuffle(perm)
            
            # Try to add edges randomly
            possible_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    possible_edges.append((i, j))
            random.shuffle(possible_edges)
            
            for u, v in possible_edges:
                # Check if adding (u,v) creates a triangle
                creates_triangle = False
                for w in adj[u]:
                    if w in adj[v]:
                        creates_triangle = True
                        break
                if not creates_triangle:
                    # Add edge with some probability to get varied densities
                    if random.random() < 0.3:  # Keep density moderate
                        adj[u].add(v)
                        adj[v].add(u)
                        edges.append([u, v])
            
            if len(edges) == 0:
                continue
            
            # Check if this graph has no independent set of size 10
            # This is the expensive check
            if time.time() > deadline - 30:
                break
            
            max_ind = _max_independent_set_size(n, edges)
            if max_ind < 10:
                # Found a valid graph!
                if n > best_n:
                    best_n = n
                    best_edges = edges
                found = True
                break
        
        if found and best_n >= target_n:
            break
    
    # Final check: ensure we return a valid graph
    if not _validate(best_n, best_edges):
        # Fallback to seed
        best_n = 26
        best_edges = [[i, j] for i in range(26) for j in range(i + 1, 26) if (j - i) % 26 % 3 == 1]
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
