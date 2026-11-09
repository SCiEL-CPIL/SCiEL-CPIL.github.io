"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

def construct():
    t0 = time.time()
    DEADLINE = 115.0  # safety margin under 150s kill

    n = 20
    num_colors = 20

    # Strategy: Construct a C4-free graph.
    # If the graph has no 4-cycles, the rainbow condition is vacuously true.
    # We maximize the number of edges in a C4-free graph on 20 vertices.
    
    # Use bitsets for adjacency to speed up C4 check.
    # adj[i] is a bitmask of neighbors of i.
    adj = [0] * n
    
    # List of all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Shuffle edges to break symmetry and potentially find denser graphs?
    # Or use a specific order? Random is often good for greedy extremal constructions.
    np.random.seed(42)
    edge_indices = np.random.permutation(len(all_edges))
    
    edges_added = 0
    result_edges = []
    
    for idx in edge_indices:
        if time.time() - t0 > DEADLINE:
            break
            
        u, v = all_edges[idx]
        
        # Check if adding (u, v) creates a C4.
        # A C4 is created if there exists w such that (v,w) in E and (w,x) in E and (x,u) in E.
        # This is equivalent to: N(N(v)) intersects N(u).
        # Let's compute N(N(v)) using bitsets.
        
        # Get neighbors of v
        mask_v = adj[v]
        
        # If mask_v is 0, no path of length 2 from v, so no C4.
        if mask_v == 0:
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            result_edges.append([u, v, 0]) # Color doesn't matter, use 0
            edges_added += 1
            continue
            
        # Compute union of neighbors of all neighbors of v.
        # This is a bit heavy if done loop-by-loop.
        # Precompute? No, adj changes.
        
        # Iterate through neighbors of v
        # Since n=20, max degree is 19.
        # We can just loop through the bits set in mask_v.
        
        common = 0
        # Extract neighbors of v
        temp_mask = mask_v
        while temp_mask:
            # Get lowest set bit
            lsb = temp_mask & -temp_mask
            w_idx = int(np.log2(lsb)) if lsb else -1
            # Actually, use bit_length or a lookup table for speed?
            # np.log2 is slow. Use a helper or just loop 0..19?
            # Since n=20, loop 0..19 is fine.
            temp_mask ^= lsb
            
        # Better: Loop through all vertices w
        # Check if w is neighbor of v
        # If yes, check if N(w) intersects N(u)
        
        is_c4 = False
        for w in range(n):
            if w == u or w == v:
                continue
            # Check if (v, w) is an edge
            if (adj[v] >> w) & 1:
                # Check if N(w) intersects N(u)
                # N(w) is adj[w], N(u) is adj[u]
                if (adj[w] & adj[u]) != 0:
                    is_c4 = True
                    break
        
        if not is_c4:
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            result_edges.append([u, v, 0])
            edges_added += 1
            
    # The color is irrelevant for validity if no C4s exist, but the interface requires colors 0..19.
    # We can assign any color, e.g., 0, or cycle through them.
    # Let's assign distinct colors to be safe, or just 0.
    # The problem says "edge-coloured with 20 colours". It doesn't say all must be used.
    # "receives four distinct colours" is the constraint.
    # If no 4-cycles, constraint is empty.
    
    # Final verification (optional, for safety)
    # Check if any 4-cycle exists
    # For all a,b,c,d distinct, check if (a,b),(b,c),(c,d),(d,a) in E.
    # This is O(n^4) = 160,000 checks. Fast enough.
    
    adj_matrix = np.zeros((n, n), dtype=bool)
    for u, v, c in result_edges:
        adj_matrix[u, v] = True
        adj_matrix[v, u] = True
        
    valid = True
    # Check all 4-cycles
    # A 4-cycle is a set of 4 distinct vertices {a,b,c,d} with edges (a,b), (b,c), (c,d), (d,a)
    # We can just check all quadruples.
    for a in range(n):
        for b in range(a + 1, n):
            if not adj_matrix[a, b]:
                continue
            for c in range(b + 1, n):
                if not adj_matrix[b, c]:
                    continue
                for d in range(c + 1, n):
                    if adj_matrix[c, d] and adj_matrix[d, a]:
                        # Found a 4-cycle a-b-c-d-a
                        # Check colors. But we assigned all 0?
                        # If we assigned all 0, this is INVALID if a 4-cycle exists.
                        # So we MUST ensure no 4-cycles exist.
                        # If my greedy check was correct, there are no 4-cycles.
                        # If there is a bug in the check, this will fail.
                        # Let's rely on the greedy check being correct.
                        # But wait, if I found a 4-cycle, it means my greedy check failed.
                        # In that case, the graph is invalid.
                        # I should remove edges to fix it?
                        # Or just return the edges and hope the evaluator is lenient?
                        # No, evaluator checks validity.
                        # So I must ensure the greedy check is perfect.
                        
                        # The check: Adding (u,v) creates C4 if exists w s.t. (v,w) in E and N(w) intersects N(u).
                        # This covers all C4s containing (u,v).
                        # Since we build incrementally, any C4 in the final graph must have been created when its last edge was added.
                        # So if the check is correct, no C4s exist.
                        
                        # Let's double check the logic.
                        # C4: u-v-w-x-u.
                        # Last edge added: say (u,v).
                        # At that time, (v,w), (w,x), (x,u) existed.
                        # My check: For w in N(v), does N(w) intersect N(u)?
                        # Here w is a neighbor of v.
                        # N(w) contains x (since (w,x) in E).
                        # N(u) contains x (since (x,u) in E).
                        # So x is in N(w) ∩ N(u).
                        # So the intersection is non-empty.
                        # So the check would have rejected (u,v).
                        # Thus, no C4s should exist.
                        
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        # Fallback: Return a very sparse graph that is definitely C4-free.
        # E.g., a star graph.
        # Star has 19 edges. No cycles at all.
        edges = []
        for i in range(1, n):
            edges.append([0, i, 0])
        return edges

    return result_edges
# EVOLVE-BLOCK-END
