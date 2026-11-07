"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    best_signs = None
    best_radius = float('inf')
    
    # Pre-compute edge indices by type
    # Edges are: top cycle (m), bottom cycle (m), matching (m)
    top_edges = list(range(m))
    bottom_edges = list(range(m, 2*m))
    match_edges = list(range(2*m, 3*m))
    
    def evaluate(signs):
        nonlocal best_signs, best_radius
        try:
            radius, details = check_lift(graph, signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
            return radius
        except Exception:
            return None
    
    # Key insight: The spectral radius of the signed adjacency matrix is invariant under gauge transformations.
    # A gauge transformation corresponds to flipping signs on all edges incident to a vertex.
    # 
    # For a prism graph, we can fix the gauge by setting all edges of a spanning tree to +1.
    # A spanning tree of the prism graph has 2m-1 edges. The remaining edges are the "chords".
    # The number of chords is: 3m - (2m-1) = m+1.
    # 
    # Specifically, we can choose a spanning tree that includes:
    # - All m matching edges (i, i+m) for i = 0..m-1
    # - m-1 edges from the top cycle (e.g., top_edges[0..m-2])
    # - m-1 edges from the bottom cycle (e.g., bottom_edges[0..m-2])
    # Total: m + (m-1) + (m-1) = 3m-2 edges... wait, that's too many.
    # 
    # Actually, a spanning tree of a graph with n vertices has n-1 edges.
    # n = 2m, so the spanning tree has 2m-1 edges.
    # Total edges = 3m. So chords = 3m - (2m-1) = m+1.
    # 
    # Let's choose a specific spanning tree:
    # - All m matching edges
    # - Top cycle edges: top_edges[0], top_edges[1], ..., top_edges[m-2] (m-1 edges)
    # - Bottom cycle edges: bottom_edges[0], bottom_edges[1], ..., bottom_edges[m-2] (m-1 edges)
    # Total: m + (m-1) + (m-1) = 3m-2. But we need 2m-1.
    # 
    # Wait, that's not right. Let me think again.
    # 
    # The prism graph has 2m vertices. A spanning tree has 2m-1 edges.
    # Total edges in prism = 3m.
    # Number of chords (non-tree edges) = 3m - (2m-1) = m+1.
    # 
    # Let's construct a specific spanning tree:
    # Start with vertex 0. Add matching edge (0, m) -> connects to vertex m.
    # From vertex 0, add top_edges[0] = (0,1) -> connects to vertex 1.
    # From vertex m, add bottom_edges[0] = (m, m+1) -> connects to vertex m+1.
    # Continue: from vertex i, add top_edges[i] = (i, i+1) for i=1..m-2.
    # From vertex m+i, add bottom_edges[i] = (m+i, m+i+1) for i=1..m-2.
    # 
    # This gives us:
    # - Matching edges: all m of them
    # - Top cycle: top_edges[0..m-2] (m-1 edges)
    # - Bottom cycle: bottom_edges[0..m-2] (m-1 edges)
    # Total: m + (m-1) + (m-1) = 3m-2. Still too many!
    # 
    # I made an error. Let me count the vertices connected:
    # Start: {0}
    # Add (0,m): {0, m}
    # Add (0,1): {0, 1, m}
    # Add (m, m+1): {0, 1, m, m+1}
    # ...
    # After adding all m matching edges and (m-1) top and (m-1) bottom edges:
    # We have 2m vertices.
    # Number of edges: m + (m-1) + (m-1) = 3m-2.
    # But a spanning tree should have 2m-1 edges.
    # 
    # Ah, I see the issue. The matching edges alone form a matching, not a tree.
    # Let me reconsider.
    # 
    # Actually, the prism graph consists of two m-cycles connected by a perfect matching.
    # A spanning tree can be constructed as follows:
    # - Take all m matching edges.
    # - From the top cycle, take m-1 edges (e.g., all except top_edges[m-1]).
    # - From the bottom cycle, take 0 edges (since the matching already connects the two cycles).
    # Wait, that gives m + (m-1) = 2m-1 edges. 
    # 
    # Let's verify: if we take all matching edges and m-1 top cycle edges, do we have a spanning tree?
    # The matching edges connect each top vertex i to bottom vertex m+i.
    # The m-1 top cycle edges connect top vertices in a path (if we omit one edge from the cycle).
    # So the top vertices are connected in a path, and each top vertex is connected to its corresponding bottom vertex.
    # The bottom vertices are NOT connected to each other except through the top vertices.
    # So the graph is connected. Number of edges = m + (m-1) = 2m-1. Number of vertices = 2m.
    # Yes, this is a spanning tree!
    # 
    # So the chords are:
    # - 1 edge from the top cycle: top_edges[m-1]
    # - m edges from the bottom cycle: bottom_edges[0..m-1]
    # Total chords: m+1.
    # 
    # The gauge-fixed representation has all tree edges set to +1, and the m+1 chord edges are free variables.
    # We can enumerate all 2^(m+1) possible sign assignments for the chords.
    # For m=4: 2^5 = 32
    # For m=5: 2^6 = 64
    # For m=7: 2^8 = 256
    # For m=9: 2^10 = 1024
    # This is trivially fast!
    
    # Identify the tree edges and chord edges
    # Tree edges: all matching edges + top_edges[0..m-2]
    # Chord edges: top_edges[m-1] + bottom_edges[0..m-1]
    
    tree_edge_indices = match_edges + top_edges[:m-1]
    chord_edge_indices = [top_edges[m-1]] + bottom_edges
    
    def gauge_fix_to_tree(signs):
        """Apply gauge transformation so that all tree edges are +1.
        This is done by fixing the signs of a spanning tree to +1.
        The remaining (chord) edges determine the gauge class.
        """
        # We want to find a gauge transformation that makes all tree edges +1.
        # A gauge transformation is determined by the signs assigned to vertices.
        # Let s_v be the sign at vertex v (+1 or -1).
        # The sign of edge (u,v) in the new gauge is: s_u * s_v * original_sign.
        # We want s_u * s_v * original_sign = +1 for all tree edges (u,v).
        # This means s_v = s_u * original_sign for each tree edge (u,v).
        # 
        # We can fix s_0 = +1 and propagate through the tree.
        
        signs = signs.copy()
        
        # Build adjacency for the tree
        # Tree edges: matching (i, i+m) for i=0..m-1, and top_edges[0..m-2] = (i, i+1) for i=0..m-2
        
        # Fix s_0 = +1
        vertex_sign = np.ones(n, dtype=int)
        
        # Propagate through matching edges: (i, i+m)
        # For matching edge i: vertex_sign[i+m] = vertex_sign[i] * signs[match_edges[i]]
        for i in range(m):
            vertex_sign[i+m] = vertex_sign[i] * signs[match_edges[i]]
        
        # Propagate through top cycle edges: (i, i+1) for i=0..m-2
        # For top edge i: vertex_sign[i+1] = vertex_sign[i] * signs[top_edges[i]]
        for i in range(m-1):
            vertex_sign[i+1] = vertex_sign[i] * signs[top_edges[i]]
        
        # Now apply the gauge transformation to all edges
        # New sign of edge (u,v) = vertex_sign[u] * vertex_sign[v] * original_sign
        for i, (u, v) in enumerate(edges):
            signs[i] = vertex_sign[u] * vertex_sign[v] * signs[i]
        
        return signs
    
    def exhaustive_search_chords():
        """Exhaustively enumerate all 2^(m+1) sign assignments for chord edges."""
        nonlocal best_signs, best_radius
        
        num_chords = len(chord_edge_indices)
        
        for mask in range(2 ** num_chords):
            if time.time() > deadline - 3:
                return
            
            # Create a base sign vector with all +1
            base_signs = np.ones(num_edges, dtype=int)
            
            # Set chord edge signs based on mask
            for j, edge_idx in enumerate(chord_edge_indices):
                if (mask >> j) & 1:
                    base_signs[edge_idx] = -1
            
            # Apply gauge fixing to make tree edges +1
            signs = gauge_fix_to_tree(base_signs)
            
            # Verify that tree edges are indeed +1 (sanity check)
            # for idx in tree_edge_indices:
            #     if signs[idx] != 1:
            #         continue  # Should not happen
            
            radius = evaluate(signs)
    
    # Run exhaustive search over chord assignments
    exhaustive_search_chords()
    
    # If exhaustive search found a solution, we're done (it's the global optimum)
    if best_signs is not None:
        return best_signs
    
    # Fallback: if exhaustive search failed (shouldn't happen), use random sampling
    rng = np.random.default_rng(42)
    for _ in range(1000):
        if time.time() > deadline - 5:
            break
        signs = rng.choice([-1, 1], size=num_edges)
        evaluate(signs)
    
    if best_signs is None:
        return np.ones(num_edges, dtype=int)
    
    return best_signs
# EVOLVE-BLOCK-END
