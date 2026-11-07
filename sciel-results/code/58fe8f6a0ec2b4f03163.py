import itertools
import time

def is_2_colorable(edges, num_vertices):
    """
    Checks if a hypergraph is 2-colorable.
    Returns True if 2-colorable, False otherwise.
    Uses brute force for small num_vertices.
    """
    # If num_vertices is small, we can just try all 2^N colorings
    if num_vertices <= 20:
        for mask in range(1 << num_vertices):
            # Check if this coloring works
            valid = True
            for e in edges:
                # Count colors in edge
                # If all bits are same (all 0 or all 1), it's monochromatic
                c0 = (mask >> e[0]) & 1
                c1 = (mask >> e[1]) & 1
                c2 = (mask >> e[2]) & 1
                if c0 == c1 == c2:
                    valid = False
                    break
            if valid:
                return True
    return False

def find_min_chromatic_3_hypergraph(max_vertices=9, max_edges=7):
    """
    Searches for the minimum number of edges in a 3-chromatic 3-uniform hypergraph.
    """
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # We want to find the minimum edge count.
    # Try edge counts from 5 up to max_edges.
    for num_edges in range(5, max_edges + 1):
        # Try vertex counts from 6 up to max_vertices
        # A 3-chromatic 3-uniform hypergraph must have at least 6 vertices?
        # Actually, with 5 vertices, max edges is 10.
        # Let's just iterate over vertex counts.
        for num_vertices in range(6, max_vertices + 1):
            # Generate all possible triples
            all_triples = list(itertools.combinations(range(num_vertices), 3))
            
            # We want to check if there exists a subset of 'num_edges' triples
            # that is NOT 2-colorable.
            # To speed up, we can use a heuristic or random sampling if the space is large.
            # For small num_vertices, exhaustive is fine.
            
            # Number of combinations
            # If num_vertices=6, num_edges=5: C(20,5) = 15504
            # If num_vertices=7, num_edges=5: C(35,5) = 324632
            # If num_vertices=8, num_edges=5: C(56,5) = 3.8M (might be slow)
            # If num_vertices=9, num_edges=5: C(84,5) = 41M (too slow)
            
            # So we should restrict the search.
            # It is known that the minimum is likely on 6 or 7 vertices.
            
            if time.time() > deadline:
                break
                
            # For larger vertex counts, we might need to be smarter.
            # But let's try 6 and 7 first.
            
            if num_vertices <= 7:
                # Exhaustive search
                found = False
                for combo in itertools.combinations(all_triples, num_edges):
                    if time.time() > deadline:
                        break
                    if not is_2_colorable(combo, num_vertices):
                        # Found a 3-chromatic hypergraph
                        # Map to 0..8
                        # The combo uses vertices 0..num_vertices-1
                        # We need to output 9 colors.
                        # Vertices not in the hypergraph can be any color.
                        # We need to provide a 3-coloring.
                        # Since it's 3-chromatic, it requires 3 colors.
                        # We can find a 3-coloring by backtracking or just use a known one if we recognize it.
                        # But the task requires returning a specific coloring.
                        # We can find a 3-coloring by brute force 3^N.
                        
                        # Find a 3-coloring
                        coloring = None
                        for mask in range(3 ** num_vertices):
                            valid = True
                            colors = []
                            temp = mask
                            for i in range(num_vertices):
                                colors.append(temp % 3)
                                temp //= 3
                            
                            for e in combo:
                                if colors[e[0]] == colors[e[1]] == colors[e[2]]:
                                    valid = False
                                    break
                            if valid:
                                coloring = colors
                                break
                        
                        if coloring:
                            # Extend coloring to 9 vertices
                            full_colors = coloring + [0] * (9 - num_vertices)
                            # Ensure full_colors are ints
                            full_colors = [int(c) for c in full_colors]
                            
                            # Ensure edges are lists of ints
                            edges_out = [list(e) for e in combo]
                            
                            return {
                                'edges': edges_out,
                                'colors': full_colors
                            }
                if time.time() > deadline:
                    break
            else:
                # For larger vertices, we might skip exhaustive or use random
                # But 5 edges on 8+ vertices is likely 2-colorable if it's sparse?
                # Actually, we can just skip if it's too slow.
                pass

    # Fallback: Fano Plane
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    return {
        'edges': edges,
        'colors': colors
    }

def construct():
    return find_min_chromatic_3_hypergraph()
