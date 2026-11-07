"""Rewritten program: find a 3-chromatic 3-uniform hypergraph on 9 vertices with minimum edges."""
# EVOLVE-BLOCK-START
import itertools
import random
from functools import lru_cache

def _is_2_colorable(edges, n):
    """Check if the hypergraph can be 2-colored (bipartite in hypergraph sense).
    A 2-coloring assigns each vertex color 0 or 1 such that no edge is monochromatic.
    We use backtracking with pruning.
    """
    # Convert edges to frozensets for faster lookup
    edge_set = [set(e) for e in edges]
    
    # Order edges by size (all size 3 here) but also by overlap to prune early
    # Sort edges so that edges sharing many vertices come first
    def edge_key(e):
        return -len(e)  # all same size, but for generality
    
    # Use a simple backtracking approach
    # Assign colors to vertices one by one
    # State: color assignment for vertices 0..i-1, check constraints for edges fully assigned
    
    # Better approach: try all 2^n colorings? n=9 means 512, very fast.
    # Let's just brute force all 2^9 = 512 colorings.
    for mask in range(1 << n):
        valid = True
        for e in edges:
            colors_in_edge = [(mask >> v) & 1 for v in e]
            if all(c == colors_in_edge[0] for c in colors_in_edge):
                valid = False
                break
        if valid:
            return True  # Found a 2-coloring, so it IS 2-colorable
    return False  # No 2-coloring exists, so it's NOT 2-colorable (i.e., 3-chromatic or higher)

def _is_3_chromatic(edges, n):
    """Check if the hypergraph is exactly 3-chromatic:
    1. Not 2-colorable (requires 3 colors)
    2. Is 3-colorable (can be colored with 3 colors)
    """
    # Check not 2-colorable
    if _is_2_colorable(edges, n):
        return False
    
    # Check 3-colorable: try all 3^n colorings? 3^9 = 19683, still fast.
    # But we can be smarter: use backtracking with constraint propagation.
    # For n=9, brute force 3^9 = 19683 is fine.
    for mask in range(3 ** n):
        valid = True
        for e in edges:
            colors_in_edge = []
            for v in e:
                colors_in_edge.append((mask // (3 ** v)) % 3)
            if len(set(colors_in_edge)) == 1:
                valid = False
                break
        if valid:
            return True
    return False

def construct():
    n = 9
    # We want to find a 3-uniform hypergraph on vertices 0..8 with minimum edges
    # that is exactly 3-chromatic.
    
    # The minimum number of edges for a 3-chromatic 3-uniform hypergraph:
    # Known results: The smallest 3-chromatic 3-uniform hypergraphs have 6 edges?
    # Let's think: with 5 edges, can we force 3 colors?
    # Actually, let's search from small edge counts upward.
    
    # For each number of edges m from 3 to some bound, enumerate all m-edge subsets
    # of the C(9,3)=84 possible edges and check if any is 3-chromatic.
    
    # But C(84, m) grows fast. Let's be smart.
    
    # Start with m=3: C(84,3) = 95284, each check is ~512 + 19683 operations... too slow for all.
    # Let's use a randomized approach with backtracking for smaller m, and smarter pruning.
    
    # Actually, let's use a different strategy:
    # 1. Start with m=4, 5, 6, ... 
    # 2. For each m, use randomized sampling with verification.
    # 3. Once we find a valid one, return it.
    
    # But we need to minimize edge count. So we must check m=3 first, then m=4, etc.
    
    # For m=3: Is there a 3-chromatic 3-uniform hypergraph with 3 edges on 9 vertices?
    # With 3 edges, at most 9 vertices are involved. The hypergraph is very sparse.
    # It's unlikely to be 3-chromatic. Let's verify: a 3-uniform hypergraph with 3 edges
    # has at most 9 vertices with degree 1 each (if disjoint) or fewer.
    # Such a sparse hypergraph is always 2-colorable. So m>=4.
    
    # Let's try m=4, 5, 6, ...
    
    all_edges = list(itertools.combinations(range(n), 3))
    
    random.seed(42)
    
    for m in range(3, 15):  # Try up to 14 edges
        # For small m, we can be more systematic
        if m <= 5:
            # Try random samples
            for attempt in range(50000):
                edges = random.sample(all_edges, m)
                edges = [list(e) for e in edges]
                if _is_3_chromatic(edges, n):
                    # Found a valid 3-chromatic hypergraph with m edges
                    # Now find a 3-coloring
                    coloring = _find_3_coloring(edges, n)
                    if coloring is not None:
                        return {'edges': edges, 'colors': coloring}
        else:
            # For larger m, use a constructive approach
            # Build up by adding edges that force more colors
            for attempt in range(20000):
                edges = random.sample(all_edges, m)
                edges = [list(e) for e in edges]
                if _is_3_chromatic(edges, n):
                    coloring = _find_3_coloring(edges, n)
                    if coloring is not None:
                        return {'edges': edges, 'colors': coloring}
    
    # Fallback: return a known valid instance
    # The complete 3-uniform hypergraph on 5 vertices (K_5^3) has 10 edges and is 3-chromatic
    # But let's use a smaller known example.
    # Actually, let's just construct one manually.
    # Known: the Fano plane related structures...
    # Let's use a simpler construction:
    # Vertices 0-8. Edges: (0,1,2), (0,3,4), (0,5,6), (1,3,5), (2,4,6), (1,4,6)
    # This is 6 edges. Let's verify it's 3-chromatic.
    edges = [[0,1,2], [0,3,4], [0,5,6], [1,3,5], [2,4,6], [1,4,6]]
    if _is_3_chromatic(edges, n):
        coloring = _find_3_coloring(edges, n)
        if coloring is not None:
            return {'edges': edges, 'colors': coloring}
    
    # Absolute fallback
    edges = [list(e) for e in itertools.combinations(range(5), 3)]
    return {'edges': edges, 'colors': [0,0,1,1,2,0,0,0,0]}

def _find_3_coloring(edges, n):
    """Find a 3-coloring of the hypergraph, or return None if not 3-colorable."""
    for mask in range(3 ** n):
        valid = True
        for e in edges:
            colors_in_edge = []
            for v in e:
                colors_in_edge.append((mask // (3 ** v)) % 3)
            if len(set(colors_in_edge)) == 1:
                valid = False
                break
        if valid:
            # Convert mask to color list
            coloring = []
            for v in range(n):
                coloring.append((mask // (3 ** v)) % 3)
            return coloring
    return None
# EVOLVE-BLOCK-END
