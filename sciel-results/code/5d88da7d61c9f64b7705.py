"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safe margin before 90s limit

    # Fallback: 7-edge Fano Plane (Score 1.4286)
    # The Fano plane is a known 3-chromatic 3-uniform hypergraph.
    fano_edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    fano_colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Helper: Check if a hypergraph is 2-colorable.
    # A hypergraph is 2-colorable if we can assign colors {0, 1} to vertices
    # such that no edge is monochromatic.
    # We use a bitset-based backtracking solver for speed.
    def is_2_colorable(edges):
        n = 9
        # Map each edge to a bitmask of its vertices
        edge_masks = []
        for e in edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
        
        # Precompute for each vertex, the list of edge masks it participates in
        # And for each edge, the list of vertices
        vertex_to_edges = [[] for _ in range(n)]
        for i, mask in enumerate(edge_masks):
            for v in range(n):
                if mask & (1 << v):
                    vertex_to_edges[v].append(i)
        
        def backtrack(idx, color_mask):
            # color_mask: bit i is 1 if vertex i is colored 1, 0 if colored 0.
            # We only assign colors to vertices 0..idx-1.
            # If idx == n, check if all edges are non-monochromatic.
            # An edge is monochromatic if all its vertices have the same color.
            # i.e., if (edge_mask & color_mask) == 0 or (edge_mask & color_mask) == edge_mask
            
            if idx == n:
                for mask in edge_masks:
                    # Check if mask is subset of color_mask (all 1s)
                    if (mask & color_mask) == mask:
                        return False
                    # Check if mask is disjoint from color_mask (all 0s)
                    if (mask & color_mask) == 0:
                        return False
                return True
            
            # Optimization: If we've assigned colors to some vertices,
            # check if any edge is already forced to be monochromatic.
            # An edge is forced mono if all its unassigned vertices are constrained
            # to match the color of the assigned vertices.
            # This is complex to check on the fly without unit propagation.
            # For small n and few edges, simple backtracking is okay, but we can add a simple prune.
            
            # Try color 0 for vertex idx
            if backtrack(idx + 1, color_mask):
                return True
            
            # Try color 1 for vertex idx
            if backtrack(idx + 1, color_mask | (1 << idx)):
                return True
            
            return False
        
        return backtrack(0, 0)

    def find_3_coloring(edges):
        """
        Finds a valid 3-coloring for the hypergraph.
        Returns list of colors (0, 1, 2) or None.
        """
        n = 9
        # Simple backtracking for 3-coloring
        colors = [-1] * n
        
        def bt(idx):
            if idx == n:
                # Verify
                for e in edges:
                    if colors[e[0]] == colors[e[1]] == colors[e[2]]:
                        return False
                return True
            
            v = idx
            for c in range(3):
                colors[v] = c
                # Pruning: Check if any edge containing v is now monochromatic
                # given the current assignments of its other vertices.
                valid = True
                for e in edges:
                    if v in e:
                        others = [u for u in e if u != v]
                        # If all others are assigned and equal to c, then this edge is mono.
                        if all(colors[u] == c for u in others if colors[u] != -1):
                            if all(colors[u] != -1 for u in others):
                                valid = False
                                break
                if valid and bt(idx + 1):
                    return True
            
            colors[v] = -1
            return False
        
        if bt(0):
            return colors
        return None

    # Target: Find a 6-edge 3-chromatic hypergraph.
    # If found, score is 1.5 (seed 9 / 6).
    # If not, return Fano plane (7 edges), score 1.4286.
    
    import random
    random.seed(42)
    
    all_triples = list(itertools.combinations(range(9), 3))
    
    found_6 = False
    best_edges_6 = None
    best_colors_6 = None
    
    # We will perform a random search.
    # Probability of a random 6-edge hypergraph being 3-chromatic is low,
    # but we have ~85 seconds.
    # is_2_colorable is fast (2^9 = 512 leaves max, pruned).
    # Let's try as many as possible.
    
    max_iters = 50000
    for i in range(max_iters):
        if time.time() > deadline - 2.0:
            break
            
        edges = random.sample(all_triples, 6)
        edges_list = [list(e) for e in edges]
        
        if not is_2_colorable(edges_list):
            colors = find_3_coloring(edges_list)
            if colors is not None:
                # Verify
                valid = True
                for e in edges_list:
                    if colors[e[0]] == colors[e[1]] == colors[e[2]]:
                        valid = False
                        break
                if valid:
                    best_edges_6 = edges_list
                    best_colors_6 = colors
                    found_6 = True
                    break
    
    if found_6:
        # Ensure types are Python int and list
        final_edges = [list(map(int, e)) for e in best_edges_6]
        final_colors = list(map(int, best_colors_6))
        return {'edges': final_edges, 'colors': final_colors}
    else:
        # Fallback to 7-edge Fano Plane
        final_edges = [list(map(int, e)) for e in fano_edges]
        final_colors = list(map(int, fano_colors))
        return {'edges': final_edges, 'colors': final_colors}

# EVOLVE-BLOCK-END
