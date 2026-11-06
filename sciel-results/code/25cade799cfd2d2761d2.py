import time
import random

def _has_c4_bitmask(n, adj):
    """Check if graph has a C4 using bitmasks. Returns True if C4 found."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if common & (common - 1):  # at least two bits set
                return True
    return False

def _has_independent_set(adj, n, target):
    """
    Check if there exists an independent set of size >= target.
    Uses branch and bound with bitmasks and max-degree heuristic.
    Returns True if an independent set of size 'target' exists.
    """
    if n < target:
        return False
    
    # adj is a list of bitmasks
    
    def exists_is(vertices_mask, k):
        if k == 0:
            return True
        if not vertices_mask:
            return False
        
        # Upper bound: if number of vertices left < k, impossible
        if vertices_mask.bit_count() < k:
            return False
        
        # Heuristic: Pick the vertex with the highest degree in the induced subgraph.
        # This tends to prune the "include" branch more effectively.
        # Calculate degree of each vertex in vertices_mask
        max_deg = -1
        best_v = -1
        v_mask = vertices_mask
        while v_mask:
            lsb = v_mask & -v_mask
            v = lsb.bit_length() - 1
            # Degree of v in induced subgraph is popcount(adj[v] & vertices_mask)
            deg = (adj[v] & vertices_mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            v_mask ^= lsb
        
        if best_v == -1:
            return False
            
        v = best_v
        v_bit = 1 << v
        
        # Option 1: Include v
        # Remove v and its neighbors
        new_mask_include = vertices_mask & ~v_bit & ~adj[v]
        if exists_is(new_mask_include, k - 1):
            return True
            
        # Option 2: Exclude v
        new_mask_exclude = vertices_mask ^ v_bit
        if exists_is(new_mask_exclude, k):
            return True
            
        return False

    all_vertices = (1 << n) - 1
    return exists_is(all_vertices, target)

def _build_polarity_graph_q5():
    """
    Construct the polarity graph of PG(2,5).
    n = 31.
    """
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for x in range(q):
        points.append((0, 1, x))
    points.append((0, 0, 1))
    
    n = len(points)
    adj = [0] * n
    edges = []
    
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

def _construct_graph():
    deadline = time.time() + 110
    
    # Start with base graph
    n_base, adj_base, edges_base = _build_polarity_graph_q5()
    
    # Verify base
    if _has_c4_bitmask(n_base, adj_base) or _has_independent_set(adj_base, n_base, 11):
        # Fallback if base is invalid (should not happen)
        return {'n': 31, 'edges': edges_base}
        
    best_n = n_base
    best_edges = edges_base.copy()
    
    # Try to extend to 39, 38, ...
    for target_n in range(39, n_base, -1):
        if time.time() > deadline:
            break
            
        found = False
        
        # Use backtracking to extend the graph
        # We will try to add vertices n_base to target_n-1
        
        def backtrack(current_n, current_adj, current_edges):
            nonlocal found, best_n, best_edges, deadline
            
            if time.time() > deadline:
                return False
                
            if current_n == target_n:
                # Final check
                if not _has_c4_bitmask(current_n, current_adj) and not _has_independent_set(current_adj, current_n, 11):
                    found = True
                    best_n = current_n
                    best_edges = current_edges.copy()
                    return True
                return False
            
            new_v = current_n
            # Generate candidate neighbor sets for new_v
            # A neighbor set S is valid if S is an independent set in the conflict graph H.
            # H has vertices 0..current_n-1. u~w in H if adj[u] & adj[w] != 0.
            
            # We will generate a few random candidate sets to keep it fast, 
            # but use a smarter selection than pure random.
            # Or, we can try to find large independent sets in H.
            
            # For simplicity and speed, we'll try a mix of random and greedy sets.
            # However, to improve on the previous version, we'll try more candidates 
            # and use the improved IS checker to prune faster.
            
            # Let's try to construct S by picking vertices from 0..current_n-1
            # such that they are pairwise non-adjacent in the conflict graph.
            
            # Heuristic: Pick vertices with low degree in the original graph?
            # Or just random?
            
            # To ensure we find a solution, we should explore the space more systematically.
            # But given time limits, we'll use a "Random Restart with Backtracking" hybrid.
            
            # Generate 50 random candidate sets
            candidates = []
            for _ in range(50):
                # Greedy construction of an independent set in H
                S = 0
                # Shuffle vertices
                verts = list(range(current_n))
                random.shuffle(verts)
                for v in verts:
                    # Check if v is compatible with current S
                    # v is compatible if for all u in S, adj[u] & adj[v] == 0
                    # This is equivalent to: (adj[v] & S) == 0 ? No.
                    # adj[u] & adj[v] == 0 means u and v have no common neighbors.
                    # This is not the same as u and v being non-adjacent.
                    # The condition for C4-free is: N(u) and N(v) intersect in at most 1 vertex.
                    # Wait, the condition "No two vertices may have two common neighbours" is:
                    # |N(u) \cap N(v)| <= 1.
                    # My previous logic for H was: u~w in H if adj[u] & adj[w] != 0.
                    # This checks if u and w are adjacent? No, adj[u] is the neighbor mask of u.
                    # adj[u] & adj[w] is the mask of common neighbors of u and w.
                    # So if (adj[u] & adj[w]) has >= 2 bits, it's a C4.
                    # So for the new vertex new_v, if we connect it to u and w,
                    # we create a C4 if u and w have >= 2 common neighbors.
                    # So S must be a set of vertices such that for all u, w in S, 
                    # |N(u) \cap N(w)| <= 1.
                    
                    # This is a "C4-free" constraint on S itself? No.
                    # It's a constraint on the pairs in S.
                    
                    # So we need to check pairwise compatibility.
                    compatible = True
                    for u in S:
                        if (adj[u] & adj[v]).bit_count() > 1:
                            compatible = False
                            break
                    if compatible:
                        S |= (1 << v)
                        if S.bit_count() >= 8: # Limit degree
                            break
                if S:
                    candidates.append(S)
            
            # Try each candidate
            for S in candidates:
                if time.time() > deadline:
                    return False
                    
                # Check if S is valid (pairwise compatible)
                # Already done in generation, but double check
                valid = True
                s_list = []
                temp_S = S
                while temp_S:
                    lsb = temp_S & -temp_S
                    u = lsb.bit_length() - 1
                    s_list.append(u)
                    temp_S ^= lsb
                for i in range(len(s_list)):
                    for j in range(i+1, len(s_list)):
                        if (adj[s_list[i]] & adj[s_list[j]]).bit_count() > 1:
                            valid = False
                            break
                    if not valid: break
                if not valid:
                    continue
                
                # Apply S
                new_adj = current_adj.copy()
                new_adj.append(0)
                new_edges = current_edges.copy()
                
                for u in s_list:
                    new_adj[new_v] |= (1 << u)
                    new_adj[u] |= (1 << new_v)
                    new_edges.append([new_v, u])
                
                # Check C4
                if _has_c4_bitmask(new_v + 1, new_adj):
                    continue
                    
                # Check IS
                if _has_independent_set(new_adj, new_v + 1, 11):
                    continue
                
                # Recurse
                if backtrack(new_v + 1, new_adj, new_edges):
                    return True
            
            return False
        
        # Run backtracking with a few random seeds/restarts
        for attempt in range(20):
            if time.time() > deadline:
                break
            random.seed(attempt * 12345 + target_n)
            if backtrack(n_base, adj_base.copy(), edges_base.copy()):
                found = True
                break
                
        if found:
            break
            
    return {'n': best_n, 'edges': best_edges}

def construct():
    start_time = time.time()
    result = _construct_graph()
    return result
