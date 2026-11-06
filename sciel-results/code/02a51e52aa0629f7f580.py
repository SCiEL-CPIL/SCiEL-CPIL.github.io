import time
import random
from itertools import combinations

def _build_adj_bitmasks(edges, n):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _creates_k4(adj, u, v, n):
    """Check if adding edge (u,v) creates a K4."""
    common = adj[u] & adj[v]
    common &= ~(1 << u)
    common &= ~(1 << v)
    
    # Check if any edge exists within common
    c = common
    while c:
        lsb = c & -c
        k_idx = lsb.bit_length() - 1
        c ^= lsb
        others = common & ~(1 << k_idx)
        if adj[k_idx] & others:
            return True
    return False

def _get_triangles(adj, n):
    """Get list of triangle masks."""
    triangles = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                continue
            common = adj[i] & adj[j]
            common &= ~(1 << i)
            common &= ~(1 << j)
            c = common
            while c:
                lsb = c & -c
                k_idx = lsb.bit_length() - 1
                c ^= lsb
                if k_idx > j: # Ensure i < j < k to avoid duplicates
                    triangles.append((1 << i) | (1 << j) | (1 << k_idx))
                elif k_idx < i:
                    triangles.append((1 << k_idx) | (1 << i) | (1 << j))
                # else k_idx is between i and j, already covered? 
                # No, the loop over i,j,k with i<j<k is safer.
    # Let's redo with i<j<k for clarity and correctness
    triangles = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                continue
            common = adj[i] & adj[j]
            common &= ~(1 << i)
            common &= ~(1 << j)
            # k must be > j to ensure i<j<k? No, k can be anywhere.
            # But if we iterate i, j, and then k from common, we might get duplicates if we don't enforce order.
            # Let's just collect all and use a set? Or enforce i < j < k.
            c = common
            while c:
                lsb = c & -c
                k_idx = lsb.bit_length() - 1
                c ^= lsb
                if k_idx > j:
                    triangles.append((1 << i) | (1 << j) | (1 << k_idx))
                elif k_idx < i:
                    # This triangle is (k, i, j) with k < i < j
                    # It will be generated when i'=k, j'=i? 
                    # No, the outer loops are i, j.
                    # If we enforce i < j < k, we only add when k > j.
                    # What about triangles where k < i?
                    # They will be found when the outer loop hits i=k, j=i? No, j must be > i.
                    # So triangle (k, i, j) with k < i < j:
                    # Outer loop i=k, j=i? No, j is the second index.
                    # If outer loop is i, j, then for triangle (a,b,c) with a<b<c:
                    # It is found when i=a, j=b, and k=c is in common(a,b).
                    # So yes, just checking k > j is sufficient if we iterate all i, j.
                    pass
    return triangles

def _has_triangle_free_set_of_size(adj, n, k, triangles):
    """Check if there is a triangle-free induced subgraph of size k."""
    # If k > n, return False
    if k > n:
        return False
        
    # Iterate through all combinations of k vertices
    # To speed up, we can use bitmasks and check subsets?
    # For n=18, k=9, combinations are 48620.
    # For each combination, check if it contains any triangle.
    
    # Precompute triangle masks
    t_masks = triangles
    
    for comb in combinations(range(n), k):
        mask = 0
        for v in comb:
            mask |= (1 << v)
        
        # Check if mask contains any triangle
        contains_triangle = False
        for t in t_masks:
            if (mask & t) == t:
                contains_triangle = True
                break
        
        if not contains_triangle:
            return True
            
    return False

def _find_max_triangle_free_size(adj, n, triangles):
    """Find the exact size of the largest triangle-free induced subgraph."""
    for k in range(n, -1, -1):
        if _has_triangle_free_set_of_size(adj, n, k, triangles):
            return k
    return 0

def _random_k4_free_graph(n, seed=None):
    if seed is not None:
        random.seed(seed)
        
    adj = [0] * n
    edges = []
    possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(possible_edges)
    
    for u, v in possible_edges:
        if not _creates_k4(adj, u, v, n):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])
            
    return edges, adj

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n = 18
    
    best_edges = None
    best_score = float('inf') # We minimize the size
    
    # Precompute combinations for speed?
    # We'll just use itertools in the loop.
    
    for attempt in range(100):
        if time.time() > deadline:
            break
            
        if attempt == 0:
            # Try a known good structure? 
            # Or just random
            edges, adj = _random_k4_free_graph(n, seed=attempt)
        else:
            # Local search step? 
            # For simplicity, restart with different seed
            # Or perturb best_edges?
            edges, adj = _random_k4_free_graph(n, seed=attempt + 100)
            
        triangles = _get_triangles(adj, n)
        current_size = _find_max_triangle_free_size(adj, n, triangles)
        
        if current_size < best_score:
            best_score = current_size
            best_edges = edges[:]
            
        if best_score <= 8:
            # We found a graph with max TF set size <= 8
            # This is likely optimal or very good
            # We can break or continue to see if 7 is possible
            # f(18) is likely 8 or 9.
            pass
        
        # Local improvement
        improved = True
        while improved and time.time() < deadline:
            improved = False
            triangles = _get_triangles(adj, n)
            current_size = _find_max_triangle_free_size(adj, n, triangles)
            
            if current_size < best_score:
                best_score = current_size
                best_edges = edges[:]
                
            if current_size <= 8:
                break # Can't do much better quickly, maybe check 7?
                
            # Find a triangle-free set of size current_size
            # We need to modify it.
            # Get one such set
            tf_set = None
            for comb in combinations(range(n), current_size):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                contains_triangle = False
                for t in triangles:
                    if (mask & t) == t:
                        contains_triangle = True
                        break
                if not contains_triangle:
                    tf_set = set(comb)
                    break
            
            if tf_set is None:
                break # Should not happen
                
            # Try to add an edge inside tf_set
            added = False
            vertices_in_set = list(tf_set)
            random.shuffle(vertices_in_set)
            for i in range(len(vertices_in_set)):
                for j in range(i + 1, len(vertices_in_set)):
                    u = vertices_in_set[i]
                    v = vertices_in_set[j]
                    if not (adj[u] & (1 << v)):
                        if not _creates_k4(adj, u, v, n):
                            # Add edge
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            edges.append([u, v])
                            added = True
                            break
                if added:
                    break
                    
            if not added:
                # Cannot add any edge inside the set
                # Try removing a random edge to allow new additions?
                # Or just break and restart
                if random.random() < 0.5:
                    if edges:
                        e_idx = random.randint(0, len(edges) - 1)
                        u, v = edges.pop(e_idx)
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        improved = True # Force loop to continue
                else:
                    break
                    
    if best_edges is None:
        # Fallback
        edges, _ = _random_k4_free_graph(n, seed=42)
        best_edges = edges
        
    return best_edges
