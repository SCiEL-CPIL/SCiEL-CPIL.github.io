"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    def is_triangle_free(adj):
        """Check if graph is triangle-free using adjacency bitmasks."""
        for i in range(16):
            for j in range(i + 1, 16):
                if adj[i] & (1 << j):
                    # Check common neighbors
                    if adj[i] & adj[j]:
                        return False
        return True
    
    _colorable_cache = {}
    
    def is_3_colorable(adj):
        """Check if graph is 3-colorable using backtracking with bitmasks.
        Returns True if 3-colorable, False if not (i.e., 4-chromatic or higher).
        """
        # Use tuple of adj as cache key
        key = tuple(adj)
        if key in _colorable_cache:
            return _colorable_cache[key]
        
        # Precompute neighbor masks
        neighbors = [adj[i] for i in range(16)]
        
        # Order vertices by degree (descending) for better pruning
        degrees = [bin(adj[i]).count('1') for i in range(16)]
        order = sorted(range(16), key=lambda v: -degrees[v])
        
        # color_masks[c] = bitmask of vertices with color c
        color_masks = [0, 0, 0]
        
        # Track which vertices are colored
        colored_mask = 0
        
        def backtrack(pos):
            nonlocal colored_mask
            if pos == 16:
                return True
            
            v = order[pos]
            nmask = neighbors[v]
            
            # Determine available colors using bitmask operations
            # A color c is forbidden if any neighbor of v has color c
            forbidden = 0
            if nmask & color_masks[0]:
                forbidden |= 1
            if nmask & color_masks[1]:
                forbidden |= 2
            if nmask & color_masks[2]:
                forbidden |= 4
            
            # If all 3 colors forbidden, dead end
            if forbidden == 7:
                return False
            
            # Try each available color
            if not (forbidden & 1):
                color_masks[0] |= (1 << v)
                colored_mask |= (1 << v)
                if backtrack(pos + 1):
                    return True
                color_masks[0] &= ~(1 << v)
                colored_mask &= ~(1 << v)
            
            if not (forbidden & 2):
                color_masks[1] |= (1 << v)
                colored_mask |= (1 << v)
                if backtrack(pos + 1):
                    return True
                color_masks[1] &= ~(1 << v)
                colored_mask &= ~(1 << v)
            
            if not (forbidden & 4):
                color_masks[2] |= (1 << v)
                colored_mask |= (1 << v)
                if backtrack(pos + 1):
                    return True
                color_masks[2] &= ~(1 << v)
                colored_mask &= ~(1 << v)
            
            return False
        
        result = backtrack(0)
        _colorable_cache[key] = result
        return result
    
    def count_edges(adj):
        count = 0
        for i in range(16):
            count += bin(adj[i]).count('1')
        return count // 2
    
    def adj_to_edges(adj):
        edges = []
        for i in range(16):
            for j in range(i + 1, 16):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges
    
    def edges_to_adj(edges):
        adj = [0] * 16
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    # Start with a known 4-chromatic triangle-free graph
    # Mycielski of C5 (Grötzsch graph) on 11 vertices, plus 5 isolated
    # Let's use a better starting point - try to construct a denser one
    
    # Start with empty graph and build up
    # Actually, let's start with a random triangle-free graph and add edges
    
    def random_triangle_free_graph():
        """Generate a random triangle-free graph."""
        adj = [0] * 16
        # Try to add random edges while maintaining triangle-free
        all_edges = [(i, j) for i in range(16) for j in range(i + 1, 16)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            # Check if adding (u,v) creates a triangle
            common = adj[u] & adj[v]
            if common == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj
    
    def greedy_maximize(adj):
        """Greedily add edges to maximize count while maintaining triangle-free and non-3-colorable."""
        # Try edges in a fixed order (by sum of degrees, then index) for determinism
        all_possible = [(i, j) for i in range(16) for j in range(i + 1, 16) 
                       if not (adj[i] & (1 << j))]
        all_possible.sort(key=lambda e: (-(bin(adj[e[0]]).count('1') + bin(adj[e[1]]).count('1')), e[0], e[1]))
        for u, v in all_possible:
            if time.time() > deadline:
                break
            common = adj[u] & adj[v]
            if common:
                continue
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if is_3_colorable(adj):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        return adj
    
    def try_add_edge(adj, u, v):
        """Try to add edge (u,v) if it doesn't create a triangle."""
        if adj[u] & (1 << v):
            return False  # already exists
        common = adj[u] & adj[v]
        if common:
            return False  # would create triangle
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True
    
    def try_remove_edge(adj, u, v):
        """Remove edge (u,v)."""
        if adj[u] & (1 << v):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            return True
        return False
    
    best_adj = None
    best_count = 0
    
    # Phase 1: Generate initial solutions
    for _ in range(80):
        adj = random_triangle_free_graph()
        adj = greedy_maximize(adj)
        cnt = count_edges(adj)
        if cnt > best_count and not is_3_colorable(adj):
            best_adj = adj[:]
            best_count = cnt
    
    # If we didn't find a non-3-colorable graph, force one
    if best_adj is None:
        # Use the Grötzsch graph as starting point
        adj = [0] * 16
        # C5 on vertices 0-4
        for i in range(5):
            adj[i] |= (1 << ((i + 1) % 5))
            adj[(i + 1) % 5] |= (1 << i)
        # Mycielski construction: vertex 5 is center connected to 0-4
        for i in range(5):
            adj[5] |= (1 << i)
            adj[i] |= (1 << 5)
        # Vertices 6-10: copies of 0-4, connected to neighbors' copies and to 11
        for i in range(5):
            for j in range(5):
                if adj[i] & (1 << j):
                    adj[6 + i] |= (1 << (6 + j))
                    adj[6 + j] |= (1 << (6 + i))
            adj[6 + i] |= (1 << 11)
            adj[11] |= (1 << (6 + i))
        # Connect 11 to 5
        adj[11] |= (1 << 5)
        adj[5] |= (1 << 11)
        # Wait, standard Mycielski: M(G) has V(G) U V'(G) U {u}
        # where V'(G) = {v' : v in V(G)}, u connected to all V'(G)
        # v' connected to N(v) in V(G)
        
        # Let me redo this properly
        adj = [0] * 16
        # Original C5: vertices 0-4
        for i in range(5):
            adj[i] |= (1 << ((i + 1) % 5))
            adj[(i + 1) % 5] |= (1 << i)
        
        # Mycielski: add vertices 5-9 as copies of 0-4, and vertex 10 as universal to copies
        for i in range(5):
            # Copy i is vertex 5+i
            # Connect 5+i to neighbors of i in original
            for j in range(5):
                if adj[i] & (1 << j):
                    adj[5 + i] |= (1 << j)
                    adj[j] |= (1 << (5 + i))
            # Connect 5+i to vertex 10
            adj[5 + i] |= (1 << 10)
            adj[10] |= (1 << (5 + i))
        
        # This is the Grötzsch graph on 11 vertices
        # Verify it's triangle-free and 4-chromatic
        if is_triangle_free(adj) and not is_3_colorable(adj):
            best_adj = adj[:]
            best_count = count_edges(adj)
        else:
            # Fallback: just use it anyway
            best_adj = adj[:]
            best_count = count_edges(adj)
    
    # Phase 2: Local search to maximize edges
    if best_adj is not None:
        current_adj = best_adj[:]
        current_count = best_count
        
        no_improve_count = 0
        max_no_improve = 5000
        
        # Precompute all possible edge pairs
        all_edge_pairs = [(i, j) for i in range(16) for j in range(i + 1, 16)]
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            # Try to add edges systematically
            improved = False
            
            for u, v in all_edge_pairs:
                if time.time() > deadline:
                    break
                # Skip if edge already exists
                if current_adj[u] & (1 << v):
                    continue
                # Check if adding (u,v) keeps triangle-free
                common = current_adj[u] & current_adj[v]
                if common:
                    continue
                
                # Add the edge
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                
                # Check if still not 3-colorable
                if not is_3_colorable(current_adj):
                    new_count = current_count + 1
                    current_count = new_count
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = current_adj[:]
                    improved = True
                    # Don't break - try to add more edges in this pass
                else:
                    # Roll back
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
            
            if not improved:
                no_improve_count += 1
                # Try swap moves: remove one edge, add another
                existing_edges = [(i, j) for i in range(16) for j in range(i + 1, 16) 
                                if current_adj[i] & (1 << j)]
                random.shuffle(existing_edges)
                found_swap = False
                for u, v in existing_edges:
                    if time.time() > deadline:
                        break
                    # Remove (u,v)
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                    
                    # Try to add a different edge
                    for a, b in all_edge_pairs:
                        if time.time() > deadline:
                            break
                        if a == u and b == v:
                            continue
                        if current_adj[a] & (1 << b):
                            continue
                        common = current_adj[a] & current_adj[b]
                        if common:
                            continue
                        current_adj[a] |= (1 << b)
                        current_adj[b] |= (1 << a)
                        if not is_3_colorable(current_adj):
                            # Net edge count unchanged, but structure changed
                            # Try to then add another edge to gain
                            for c, d in all_edge_pairs:
                                if time.time() > deadline:
                                    break
                                if current_adj[c] & (1 << d):
                                    continue
                                if (c, d) == (a, b):
                                    continue
                                common2 = current_adj[c] & current_adj[d]
                                if common2:
                                    continue
                                current_adj[c] |= (1 << d)
                                current_adj[d] |= (1 << c)
                                if not is_3_colorable(current_adj):
                                    current_count += 1
                                    if current_count > best_count:
                                        best_count = current_count
                                        best_adj = current_adj[:]
                                    found_swap = True
                                    break
                                else:
                                    current_adj[c] &= ~(1 << d)
                                    current_adj[d] &= ~(1 << c)
                            if found_swap:
                                no_improve_count = 0
                                break
                            else:
                                # Roll back the added edge
                                current_adj[a] &= ~(1 << b)
                                current_adj[b] &= ~(1 << a)
                        else:
                            current_adj[a] &= ~(1 << b)
                            current_adj[b] &= ~(1 << a)
                    
                    if found_swap:
                        break
                    else:
                        # Restore removed edge
                        current_adj[u] |= (1 << v)
                        current_adj[v] |= (1 << u)
                
                if not found_swap:
                    no_improve_count += 1
            else:
                no_improve_count = 0
        
        # Phase 3: More aggressive search with restarts
        if time.time() < deadline:
            for restart in range(200):
                if time.time() > deadline:
                    break
                
                # Start from a perturbed version of best_adj
                adj = best_adj[:]
                # Remove some random edges
                existing = [(i, j) for i in range(16) for j in range(i + 1, 16) 
                           if adj[i] & (1 << j)]
                random.shuffle(existing)
                num_remove = random.randint(1, 6)
                for u, v in existing[:num_remove]:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                
                # Try to add edges greedily in degree-ordered
                all_possible = [(i, j) for i in range(16) for j in range(i + 1, 16) 
                               if not (adj[i] & (1 << j))]
                all_possible.sort(key=lambda e: (-(bin(adj[e[0]]).count('1') + bin(adj[e[1]]).count('1')), e[0], e[1]))
                
                for u, v in all_possible:
                    if time.time() > deadline:
                        break
                    common = adj[u] & adj[v]
                    if common:
                        continue
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    if not is_3_colorable(adj):
                        cnt = count_edges(adj)
                        if cnt > best_count:
                            best_count = cnt
                            best_adj = adj[:]
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
    
    if best_adj is None:
        # Absolute fallback: return the Grötzsch graph on 11 vertices + 5 isolated
        adj = [0] * 16
        for i in range(5):
            adj[i] |= (1 << ((i + 1) % 5))
            adj[(i + 1) % 5] |= (1 << i)
        for i in range(5):
            for j in range(5):
                if adj[i] & (1 << j):
                    adj[5 + i] |= (1 << j)
                    adj[j] |= (1 << (5 + i))
            adj[5 + i] |= (1 << 10)
            adj[10] |= (1 << (5 + i))
        return adj_to_edges(adj)
    
    return adj_to_edges(best_adj)
# EVOLVE-BLOCK-END
