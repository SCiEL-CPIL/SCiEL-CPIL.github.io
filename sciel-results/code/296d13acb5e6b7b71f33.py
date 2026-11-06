"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def _has_c4(n, adj):
    """Check if graph has a C4 using bitmasks."""
    for u in range(n):
        for v in range(u + 1, n):
            # Count common neighbors
            common = 0
            for w in range(n):
                if (adj[u] >> w) & 1 and (adj[v] >> w) & 1:
                    common += 1
                    if common >= 2:
                        return True
    return False

def _independent_set_size(n, adj, target=11):
    """Check if there exists an independent set of size >= target using branching."""
    # Use a simple branch and bound
    from functools import lru_cache
    
    def max_independent_set(vertices_tuple, remaining):
        if remaining <= 0:
            return 0
        if not vertices_tuple:
            return 0
        if len(vertices_tuple) < remaining:
            return len(vertices_tuple)
        
        v = vertices_tuple[0]
        rest = vertices_tuple[1:]
        
        # Option 1: include v
        # Remove v and all its neighbors
        new_rest = []
        for u in rest:
            if not ((adj[v] >> u) & 1):
                new_rest.append(u)
        include_val = 1 + max_independent_set(tuple(new_rest), remaining - 1)
        
        # Option 2: exclude v
        exclude_val = max_independent_set(tuple(rest), remaining)
        
        return max(include_val, exclude_val)
    
    return max_independent_set(tuple(range(n)), target)

def _verify_graph(n, edges):
    """Verify the graph is C4-free and has no independent set of size 11."""
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Check C4
    if _has_c4(n, adj):
        return False
    
    # Check independent set of size 11
    if _independent_set_size(n, adj, 11) >= 11:
        return False
    
    return True

def _try_paley_like_construction():
    """Try to build a larger C4-free graph with alpha < 11."""
    # Use known constructions
    # The best known is 39 vertices
    
    # Try a construction based on finite fields
    # For q = 7, we can get certain structures
    
    # Let's try a systematic search with greedy + local improvement
    best_n = 0
    best_edges = []
    
    start_time = time.time()
    deadline = start_time + 100  # Leave margin
    
    # Try different starting structures
    # Start with disjoint cliques and add edges
    
    # Known good construction: use a (v,k,lambda) design based approach
    # Or use the known 39-vertex construction
    
    # Let's try to build from scratch with random restarts
    import random
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        random.seed(attempt)
        
        # Start with n vertices, try to add edges while maintaining C4-free
        # and keeping alpha < 11
        
        for n_try in [39, 38, 37, 36, 35]:
            if time.time() > deadline:
                break
            
            # Random initial graph
            edges = []
            adj = [0] * n_try
            
            # Add random edges while checking C4
            all_pairs = list(itertools.combinations(range(n_try), 2))
            random.shuffle(all_pairs)
            
            for u, v in all_pairs:
                if time.time() > deadline:
                    break
                
                # Check if adding edge (u,v) creates C4
                # C4 exists if there are two common neighbors of u and v
                # Actually, adding edge creates C4 if there exist w,x such that
                # w is neighbor of both u and v, and x is neighbor of both u and v
                # More precisely: adding (u,v) creates C4 if there exist distinct
                # w1, w2 both adjacent to both u and v (but not counting the new edge)
                
                common = []
                for w in range(n_try):
                    if w != u and w != v:
                        if (adj[u] >> w) & 1 and (adj[v] >> w) & 1:
                            common.append(w)
                
                if len(common) >= 2:
                    continue  # Would create C4
                
                # Also need to check that this doesn't create C4 in other ways
                # A C4 is a cycle of length 4. Adding edge (u,v) creates C4 if
                # there exist w, x such that u-w-v-x-u is a cycle, i.e.,
                # w adjacent to u and v, x adjacent to u and v, w != x
                # This is exactly the common neighbor check above
                
                # Add the edge
                edges.append([u, v])
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            # Now check if alpha < 11
            if _independent_set_size(n_try, adj, 11) < 11:
                if n_try > best_n:
                    best_n = n_try
                    best_edges = edges.copy()
                    if n_try >= 39:
                        break
            
            if n_try >= 39:
                break
    
    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback to the original
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

def construct():
    """Construct a C4-free graph with no independent set of size 11, maximizing n."""
    start_time = time.time()
    
    # Try to find a better construction
    result = _try_paley_like_construction()
    
    # Verify the result
    if _verify_graph(result['n'], result['edges']):
        return result
    
    # Fallback
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

# EVOLVE-BLOCK-END
