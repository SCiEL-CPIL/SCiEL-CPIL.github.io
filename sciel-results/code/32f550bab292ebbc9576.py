import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def _has_c4_bitmask(n, adj):
    """Check if graph has a C4 using bitmasks. Returns True if C4 found."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if common & (common - 1):  # at least two bits set
                return True
    return False

def _max_independent_set_size_bitmask(n, adj, target=11):
    """
    Check if there exists an independent set of size >= target.
    Returns the size of the maximum independent set found (capped at target).
    Uses branch and bound.
    """
    
    def max_is(vertices_mask, remaining):
        if remaining <= 0:
            return 0
        if not vertices_mask:
            return 0
        
        # Upper bound: if number of vertices left < remaining, return count
        count = 0
        temp = vertices_mask
        while temp:
            temp &= temp - 1
            count += 1
        if count < remaining:
            return count
        
        # Pick a vertex to branch on - use first set bit
        lsb = vertices_mask & -vertices_mask
        v = lsb.bit_length() - 1
        
        # Option 1: include v
        new_mask = vertices_mask & ~(1 << v) & ~adj[v]
        include_val = 1 + max_is(new_mask, remaining - 1)
        
        # Option 2: exclude v
        exclude_val = max_is(vertices_mask ^ (1 << v), remaining)
        
        return max(include_val, exclude_val)
    
    all_vertices = (1 << n) - 1
    return max_is(all_vertices, target)

def _verify_graph(n, edges):
    """Verify the graph is C4-free and has no independent set of size 11."""
    if n < 1:
        return False
    adj = [0] * n
    for u, v in edges:
        if u == v:
            return False
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        if adj[u] & (1 << v):
            return False # Duplicate edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    if _has_c4_bitmask(n, adj):
        return False
    
    if _max_independent_set_size_bitmask(n, adj, 11) >= 11:
        return False
    
    return True

def _build_polarity_graph_q5():
    """
    Construct the polarity graph of PG(2,5).
    n = 31.
    """
    q = 5
    # Points of PG(2, q)
    points = []
    # Affine points: (1, x, y)
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    # Points at infinity: (0, 1, x)
    for x in range(q):
        points.append((0, 1, x))
    # Point at infinity: (0, 0, 1)
    points.append((0, 0, 1))
    
    n = len(points)
    adj = [0] * n
    edges = []
    
    # Polarity: u . v = 0 mod q
    # Using standard dot product
    for i in range(n):
        u = points[i]
        for j in range(i + 1, n):
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if dot == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append([i, j])
                
    return n, adj, edges

def _get_valid_neighbor_sets(adj, n, new_v):
    """
    Generate all valid neighbor sets for a new vertex new_v.
    A neighbor set S is valid if adding new_v with neighbors S doesn't create a C4.
    This means: for any s1, s2 in S, N(s1) & N(s2) = 0 in the existing graph.
    Returns list of bitmasks representing valid neighbor sets.
    """
    # We need S such that no two vertices in S share a neighbor.
    # This is equivalent to S being an independent set in the graph H
    # where H has vertices 0..n-1 and edges between u,v if N(u)&N(v) != 0.
    
    # Build conflict graph: u conflicts with v if they share a neighbor
    # For efficiency, we'll use backtracking to find all maximal valid sets
    # But for now, let's just generate candidate sets greedily
    
    valid_sets = []
    
    # First, add the empty set (new vertex is isolated)
    valid_sets.append(0)
    
    # Try adding each single vertex
    for s in range(n):
        valid_sets.append(1 << s)
    
    # Try pairs: s1, s2 are compatible if N(s1) & N(s2) == 0
    for s1 in range(n):
        for s2 in range(s1 + 1, n):
            if adj[s1] & adj[s2] == 0:
                valid_sets.append((1 << s1) | (1 << s2))
    
    return valid_sets

def _construct_via_backtracking(target_n, deadline):
    """
    Use backtracking to construct a C4-free graph with target_n vertices
    and no independent set of size 11.
    """
    # Start with a base graph (PG(2,5) polarity)
    n_base, adj_base, edges_base = _build_polarity_graph_q5()
    
    if n_base >= target_n:
        # We already have enough vertices, check if valid
        if not _has_c4_bitmask(n_base, adj_base):
            if _max_independent_set_size_bitmask(n_base, adj_base, 11) < 11:
                return {'n': n_base, 'edges': edges_base}
        return None
    
    # We need to add (target_n - n_base) vertices
    # Use a different approach: try random edge selections for new vertices
    
    best = None
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        random.seed(attempt * 7919 + target_n)
        
        adj = adj_base.copy()
        edges = edges_base.copy()
        n = n_base
        
        success = True
        for _ in range(target_n - n_base):
            new_v = n
            n += 1
            adj.append(0)
            
            # Try to find a good neighbor set for new_v
            # Constraint: no two neighbors of new_v can share a neighbor in existing graph
            # i.e., if s1, s2 are both neighbors of new_v, then N(s1) & N(s2) must be empty
            
            # Greedy: pick neighbors one at a time, ensuring constraint
            chosen = []
            available = set(range(new_v))
            
            # Shuffle for randomness
            avail_list = list(available)
            random.shuffle(avail_list)
            
            for s in avail_list:
                if time.time() > deadline:
                    success = False
                    break
                if not available:
                    break
                
                # Check if s is compatible with all already chosen
                compatible = True
                for s2 in chosen:
                    if adj[s] & adj[s2] != 0:
                        compatible = False
                        break
                
                if compatible:
                    chosen.append(s)
                    available.discard(s)
                    # Remove all vertices that conflict with s
                    # A vertex t conflicts with s if N(t) & N(s) != 0
                    # i.e., if adj[t] & adj[s] != 0
                    to_remove = []
                    for t in available:
                        if adj[t] & adj[s] != 0:
                            to_remove.append(t)
                    for t in to_remove:
                        available.discard(t)
                
                # Limit degree to avoid too sparse connections
                if len(chosen) >= 6:
                    break
            
            if not success:
                break
            
            # Add edges
            for s in chosen:
                adj[new_v] |= (1 << s)
                adj[s] |= (1 << new_v)
                edges.append([new_v, s])
        
        if not success:
            continue
        
        # Verify C4-free
        if _has_c4_bitmask(n, adj):
            continue
        
        # Check independence number
        if _max_independent_set_size_bitmask(n, adj, 11) < 11:
            best = {'n': n, 'edges': edges.copy()}
            break
    
    return best

def _construct_large_graph():
    """
    Try to construct a graph with n > 31 using multiple strategies.
    """
    deadline = time.time() + 110
    
    # Strategy 1: Base polarity graph
    n_base, adj_base, edges_base = _build_polarity_graph_q5()
    
    if not _verify_graph(n_base, edges_base):
        # Fallback
        edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    best_result = {'n': n_base, 'edges': edges_base}
    
    # Strategy 2: Try to extend to larger n
    for target_n in range(43, n_base, -1):
        if time.time() > deadline:
            break
        
        result = _construct_via_backtracking(target_n, deadline)
        if result is not None:
            best_result = result
            break
    
    return best_result

def construct():
    """Construct a C4-free graph with no independent set of size 11, maximizing n."""
    start_time = time.time()
    
    # Try to construct a large graph
    result = _construct_large_graph()
    
    # Verify the result
    if _verify_graph(result['n'], result['edges']):
        return result
    
    # Fallback: polarity graph of PG(2,5) (31 vertices, C4-free)
    n_base, adj_base, edges_base = _build_polarity_graph_q5()
    return {'n': n_base, 'edges': edges_base}

# EVOLVE-BLOCK-END
