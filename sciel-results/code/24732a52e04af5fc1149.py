import time
import random

# ---------- Exact 3-colorability (triangle-free graph, 16 vertices) ----------
def _is_3_colorable(adj):
    """adj: list of 16 sets/lists. Returns True if graph is 3-colorable."""
    n = len(adj)
    if n == 0:
        return True
    
    # Precompute neighbor bitmasks
    adj_mask = [0] * n
    for u in range(n):
        mask = 0
        for v in adj[u]:
            mask |= (1 << v)
        adj_mask[u] = mask
    
    # Static ordering by descending degree is a good start, but MRV is better.
    # We will implement MRV dynamically in the backtrack.
    
    color = [-1] * n
    # color_mask[c] is a bitmask of vertices colored c
    color_mask = [0, 0, 0]
    
    # Precompute degrees for initial ordering
    degrees = [len(adj[i]) for i in range(n)]
    
    def backtrack():
        # Find uncolored vertex with minimum remaining values (MRV)
        # If all colored, return True
        best_v = -1
        min_options = 4
        best_options_mask = 0
        
        for v in range(n):
            if color[v] != -1:
                continue
            
            # Calculate available colors
            avail = 0
            for c in range(3):
                if (adj_mask[v] & color_mask[c]) == 0:
                    avail |= (1 << c)
            
            num_avail = bin(avail).count('1')
            if num_avail == 0:
                return False # Dead end
            
            if num_avail < min_options:
                min_options = num_avail
                best_v = v
                best_options_mask = avail
                if num_avail == 1:
                    break # Can't get better than 1
        
        if best_v == -1:
            return True # All vertices colored
        
        v = best_v
        
        # Try each available color
        # Order colors by least constraining value (LCV)? 
        # For 3 colors, just try them.
        for c in range(3):
            if (best_options_mask & (1 << c)) == 0:
                continue
            
            # Assign color c to v
            color[v] = c
            color_mask[c] |= (1 << v)
            
            if backtrack():
                return True
            
            # Unassign
            color[v] = -1
            color_mask[c] &= ~(1 << v)
        
        return False

    return backtrack()

def _build_adj(edges, n=16):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_triangle_free(adj):
    n = len(adj)
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[u]:
                    if w != v and w in adj[v]:
                        return False
    return True

def _can_add_edge(u, v, adj):
    if u == v or v in adj[u]:
        return False
    au = adj[u]
    for w in au:
        if w in adj[v]:
            return False
    return True

def _groetzsch_edges():
    # Grötzsch graph: 11 vertices, 20 edges
    # Vertices 0-4 form a 5-cycle
    # Vertices 5-9 form a 5-cycle
    # Vertex 10 connected to 5-9
    # Edges:
    # Cycle 0-1-2-3-4-0
    # Cycle 5-6-7-8-9-5
    # 10-5, 10-6, 10-7, 10-8, 10-9
    # Cross edges:
    # 0-5, 0-6
    # 1-6, 1-7
    # 2-7, 2-8
    # 3-8, 3-9
    # 4-9, 4-5
    edges = []
    # 5-cycles
    edges += [[0,1], [1,2], [2,3], [3,4], [4,0]]
    edges += [[5,6], [6,7], [7,8], [8,9], [9,5]]
    # Spokes to 10
    edges += [[10,5], [10,6], [10,7], [10,8], [10,9]]
    # Cross
    edges += [[0,5], [0,6]]
    edges += [[1,6], [1,7]]
    edges += [[2,7], [2,8]]
    edges += [[3,8], [3,9]]
    edges += [[4,9], [4,5]]
    return edges

def construct():
    start = time.time()
    deadline = start + 115.0 # Safety margin
    n = 16
    
    best_edges = []
    best_count = -1
    
    def elapsed_ok():
        return time.time() < deadline

    def run_local_search(initial_edges):
        adj = _build_adj(initial_edges, n)
        edges = [list(e) for e in initial_edges]
        
        # Verify initial state
        if not _is_triangle_free(adj) or _is_3_colorable(adj):
            return adj, edges, len(edges)

        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        # Phase 1: Greedy addition
        changed = True
        while changed and elapsed_ok():
            changed = False
            # Shuffle pairs to avoid bias
            random.shuffle(all_pairs)
            for u, v in all_pairs:
                if not elapsed_ok():
                    break
                if v in adj[u]:
                    continue
                if not _can_add_edge(u, v, adj):
                    continue
                
                new_adj = [s.copy() for s in adj]
                new_adj[u].add(v)
                new_adj[v].add(u)
                
                if _is_3_colorable(new_adj):
                    continue
                
                # Accept
                adj = new_adj
                edges.append([u, v])
                changed = True
                break # Restart scan with new state

        # Phase 2: Local Search (1-1 Swaps)
        # Try to escape local optima by swapping edges
        # We want to maintain 4-chromaticity and triangle-free property.
        # If we can swap an edge to allow adding more edges, we do it.
        
        improved = True
        swap_limit = 1000 # Limit swaps per restart to avoid infinite loops
        swaps_done = 0
        
        while improved and elapsed_ok() and swaps_done < swap_limit:
            improved = False
            swaps_done += 1
            
            # Try to add any edge first
            random.shuffle(all_pairs)
            added = False
            for u, v in all_pairs:
                if not elapsed_ok():
                    break
                if v in adj[u]:
                    continue
                if not _can_add_edge(u, v, adj):
                    continue
                
                new_adj = [s.copy() for s in adj]
                new_adj[u].add(v)
                new_adj[v].add(u)
                
                if _is_3_colorable(new_adj):
                    continue
                
                # Accept
                adj = new_adj
                edges.append([u, v])
                improved = True
                added = True
                break
            
            if added:
                continue
            
            # No edge could be added. Try a 1-1 swap.
            # Pick a random edge to remove
            if not edges:
                break
                
            idx = random.randint(0, len(edges) - 1)
            u_rem, v_rem = edges[idx]
            
            # Remove it
            cand_adj = [s.copy() for s in adj]
            cand_adj[u_rem].discard(v_rem)
            cand_adj[v_rem].discard(u_rem)
            cand_edges = edges[:idx] + edges[idx+1:]
            
            # Try to add a different edge
            # Filter pairs that are not present and are triangle-free w.r.t cand_adj
            candidate_pairs = []
            for u, v in all_pairs:
                if v in cand_adj[u]:
                    continue
                if _can_add_edge(u, v, cand_adj):
                    candidate_pairs.append((u, v))
            
            if not candidate_pairs:
                continue
            
            random.shuffle(candidate_pairs)
            
            # Try a few candidates
            found_swap = False
            for u_add, v_add in candidate_pairs[:20]:
                if not elapsed_ok():
                    break
                
                tmp_adj = [s.copy() for s in cand_adj]
                tmp_adj[u_add].add(v_add)
                tmp_adj[v_add].add(u_add)
                
                # Check if still 4-chromatic
                if _is_3_colorable(tmp_adj):
                    continue
                
                # Accept swap
                adj = tmp_adj
                edges = cand_edges + [[u_add, v_add]]
                found_swap = True
                improved = True
                break
            
            if not found_swap:
                # If no swap worked, we might be stuck. 
                # In a full implementation, we might try 2-opt or restart.
                # For now, break to avoid wasting time.
                break

        return adj, edges, len(edges)

    # Run multiple restarts
    # Start with Grötzsch + some random edges? Or just Grötzsch padded?
    # The Grötzsch graph has 11 vertices. We have 16.
    # We can add 5 isolated vertices and then let the search connect them.
    # Or we can start with a random 4-chromatic graph.
    
    # Better: Start with Grötzsch on vertices 0-10, and leave 11-15 isolated.
    # The search will try to connect them.
    
    base_edges = _groetzsch_edges()
    
    # Run several restarts with different random seeds
    for i in range(20):
        if not elapsed_ok():
            break
        
        random.seed(i * 12345 + 67890)
        
        # We can also try starting with a slightly different base
        # e.g., random subset of edges from a denser graph?
        # But Grötzsch is a solid 4-chromatic core.
        
        adj, edges, count = run_local_search(base_edges)
        
        if count > best_count:
            best_count = count
            best_edges = [list(e) for e in edges]
            
    # Final validation
    if best_edges:
        final_adj = _build_adj(best_edges, n)
        if _is_triangle_free(final_adj) and not _is_3_colorable(final_adj):
            return best_edges
    
    # Fallback
    return base_edges

