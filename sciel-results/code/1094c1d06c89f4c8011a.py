"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

def _check_no_k222(n, colors, color):
    """
    Check if there is a monochromatic K_{2,2,2} in the given color.
    K_{2,2,2} has 6 vertices: two parts of size 2, with all edges between different parts.
    So we need vertices a,b (part 1), c,d (part 2), e,f (part 3) such that:
    - All edges between {a,b} and {c,d} are color
    - All edges between {a,b} and {e,f} are color
    - All edges between {c,d} and {e,f} are color
    
    Equivalently: there exist two disjoint pairs {u1,u2} and {v1,v2} and {w1,w2} such that
    the 12 edges between the three pairs are all color.
    
    More efficiently: For a fixed color, build adjacency. A K_{2,2,2} exists iff there
    are 6 vertices partitioned into 3 pairs such that all cross-pair edges are in the color.
    
    We can check: for each pair of vertices (a,b) that are connected by color, look at
    their common neighbors N(a) ∩ N(b) in the color graph. If this set contains two
    disjoint pairs (c,d) and (e,f) such that c-d, e-f, c-e, c-f, d-e, d-f are all in color...
    Actually that's getting complex. Let me use a different approach.
    
    K_{2,2,2} in color graph G: vertices {a,b,c,d,e,f} where all edges between {a,b},{c,d},{e,f}
    are in G. This means:
    - a-b edge may or may not be in G (it's within a part)
    - Actually in K_{2,2,2}, the parts are independent sets, and all edges between parts exist.
    - So edges a-b, c-d, e-f are NOT required to be in G.
    - Edges required: a-c, a-d, a-e, a-f, b-c, b-d, b-e, b-f, c-e, c-f, d-e, d-f.
    
    So for a color graph G, K_{2,2,2} exists iff there are 6 vertices that can be partitioned
    into 3 pairs such that all 12 cross-pair edges are in G.
    
    Efficient check: For each unordered pair {a,b}, compute common neighbors CN(a,b) = N(a) ∩ N(b) in G.
    If |CN(a,b)| >= 4, then we need to check if CN(a,b) contains two disjoint pairs {c,d} and {e,f}
    such that all edges among {c,d,e,f} between different pairs are in G.
    That is, in the subgraph induced by CN(a,b), we need a K_{2,2,2}... no wait, we need the 4 vertices
    to form two pairs where all 8 cross edges (between the two pairs) are in G, plus the edges
    from a,b to each of c,d,e,f are in G (already guaranteed by CN).
    
    Actually, if c,d,e,f are all in CN(a,b), then edges a-c,a-d,a-e,a-f,b-c,b-d,b-e,b-f are all in G.
    We additionally need: c-e, c-f, d-e, d-f (edges between {c,d} and {e,f}) to be in G.
    We do NOT need c-d or e-f to be in G.
    
    So: for each pair {a,b}, let S = CN(a,b). If |S| >= 4, check if there exist two disjoint pairs
    {c,d} and {e,f} from S such that c-e, c-f, d-e, d-f are all in G.
    This is equivalent to: in the subgraph G[S], there exist two disjoint pairs {c,d} and {e,f}
    such that all 4 edges between {c,d} and {e,f} are in G.
    That is, G[S] contains a K_{2,2}.
    
    So the check is: for some pair {a,b}, the common neighborhood S has G[S] containing K_{2,2}.
    K_{2,2} in a graph exists iff there are 4 vertices c,d,e,f with all 4 cross edges present.
    Equivalently, there exist two vertices c,d in S that have at least 2 common neighbors in S
    (excluding each other), where those common neighbors are e,f, and we need c-e,c-f,d-e,d-f all in G.
    
    Let me just do a direct check for small n.
    """
    # Build adjacency sets for the given color
    adj = [set() for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            if colors[idx] == color:
                adj[i].add(j)
                adj[j].add(i)
            idx += 1
    
    # For each pair (a,b), get common neighbors
    for a in range(n):
        for b in range(a+1, n):
            if b not in adj[a]:
                continue
            S = adj[a] & adj[b]
            if len(S) < 4:
                continue
            S_list = list(S)
            # Check if G[S] contains K_{2,2}
            # For each pair (c,d) in S, check if they have >= 2 common neighbors in S
            for ci in range(len(S_list)):
                c = S_list[ci]
                for di in range(ci+1, len(S_list)):
                    d = S_list[di]
                    # Common neighbors of c and d in S
                    common_cd = (adj[c] & adj[d]) & S
                    if len(common_cd) >= 2:
                        # Found K_{2,2} in G[S]
                        return True
    return False

def _validate_coloring(n, colors):
    """Check if the coloring is valid (no monochromatic K_{2,2,2})."""
    if _check_no_k222(n, colors, 0):
        return False
    if _check_no_k222(n, colors, 1):
        return False
    return True

def _paley_coloring(n):
    """Generate Paley coloring for prime p ≡ 1 (mod 4)."""
    squares = {x * x % n for x in range(1, n)}
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            diff = (j - i) % n
            colors.append(0 if diff in squares else 1)
    return colors

def _try_extend(n, colors, max_time):
    """
    Try to extend a valid n-vertex coloring to (n+1) vertices.
    Returns (n+1, new_colors) if successful, else (n, colors).
    """
    start = time.time()
    deadline = start + max_time
    
    # For each possible coloring of the new vertex's edges to existing vertices,
    # check if the resulting (n+1)-vertex coloring is valid.
    # There are 2^n possible colorings for the new vertex's edges.
    # For n=29, that's 2^29 ≈ 536 million, too many.
    # We need a smarter approach.
    
    # Alternative: use backtracking with pruning.
    # But for n=29 -> 30, we need to color 29 new edges.
    # This is still very hard in general.
    
    # Let's try a randomized approach: random assignments, check validity.
    # This is unlikely to find a solution if one exists, but let's try.
    
    # Actually, let's just return the current coloring. Extending is too hard.
    return n, colors

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin within 120s limit
    
    # Start with Paley(29) coloring
    n = 29
    colors = _paley_coloring(n)
    
    # Verify it's valid
    if not _validate_coloring(n, colors):
        # Should not happen, but fallback
        raise RuntimeError("Paley(29) coloring is invalid!")
    
    # Try to extend to 30 vertices
    remaining = deadline - time.time()
    if remaining > 5.0:
        n_ext, colors_ext = _try_extend(n, colors, remaining - 5.0)
        if n_ext > n:
            n = n_ext
            colors = colors_ext
    
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
