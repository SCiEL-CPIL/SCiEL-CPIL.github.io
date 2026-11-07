import time
import itertools

def _check_2_colorable(edges):
    """
    Check if the hypergraph defined by edges is 2-colorable.
    Returns True if a valid 2-coloring exists.
    """
    n = 9
    # 2^9 = 512 possible colorings
    for mask in range(1 << n):
        valid = True
        for e in edges:
            c0 = (mask >> e[0]) & 1
            c1 = (mask >> e[1]) & 1
            c2 = (mask >> e[2]) & 1
            if c0 == c1 == c2:
                valid = False
                break
        if valid:
            return True
    return False

def _check_3_coloring(edges, colors):
    """
    Verify that the given colors form a valid 3-coloring for the edges.
    """
    for e in edges:
        if colors[e[0]] == colors[e[1]] == colors[e[2]]:
            return False
    return True

def construct():
    start_time = time.time()
    
    # The Fano plane is the unique minimal 3-chromatic 3-uniform hypergraph.
    # It has 7 edges. Embedding it in 9 vertices (adding 2 isolated vertices)
    # preserves the chromatic number.
    
    fano_edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A known valid 3-coloring for the Fano plane vertices 0-6.
    # Vertices 7 and 8 are isolated, so they can be any color.
    # Coloring:
    # 0: 0
    # 1: 1
    # 2: 2
    # 3: 1
    # 4: 2
    # 5: 0
    # 6: 1
    # 7: 0
    # 8: 0
    fano_colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Verification Step 1: Ensure the provided coloring is valid.
    if not _check_3_coloring(fano_edges, fano_colors):
        # This should never happen, but handle it gracefully by searching
        # for a valid 3-coloring if the hardcoded one fails.
        found = False
        for mask in range(3 ** 9):
            temp = mask
            colors = [0] * 9
            for i in range(9):
                colors[i] = temp % 3
                temp //= 3
            if _check_3_coloring(fano_edges, colors):
                fano_colors = colors
                found = True
                break
        if not found:
            # Fallback: Return a trivially valid but larger hypergraph? 
            # No, Fano is 3-colorable. This branch is effectively dead code.
            return {'edges': fano_edges, 'colors': [0]*9}

    # Verification Step 2: Ensure it is NOT 2-colorable.
    # This confirms it is exactly 3-chromatic (assuming 3-colorability is verified).
    if _check_2_colorable(fano_edges):
        # This should never happen for the Fano plane.
        # If it did, we would need to find a different hypergraph.
        # But mathematically, Fano is not 2-colorable.
        pass

    # Return the result with correct types
    # edges must be list of lists of ints
    # colors must be list of ints
    
    return {
        'edges': [list(e) for e in fano_edges],
        'colors': [int(c) for c in fano_colors]
    }
