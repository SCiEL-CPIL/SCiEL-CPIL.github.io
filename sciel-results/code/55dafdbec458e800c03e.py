"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from functools import lru_cache

# Deadline guard: kill after 90s wall-clock, safety margin 5s
DEADLINE = 90.0
SAFETY_MARGIN = 5.0

def _is_2_colorable(edges, n=9):
    """Check if the hypergraph admits a 2-coloring (every edge non-monochromatic).
    Uses backtracking with memoization on vertex subsets."""
    if not edges:
        return True
    # Sort edges by size (all 3) and by vertex set for determinism
    edges_sorted = sorted([tuple(sorted(e)) for e in edges])
    
    # Quick check: if any vertex appears in all edges with a constraint... 
    # Actually just do backtracking
    
    @lru_cache(maxsize=None)
    def solve(remaining_edges, colored):
        """remaining_edges: tuple of edge tuples, colored: tuple of (vertex, color) pairs or None for uncolored"""
        if not remaining_edges:
            return True
        edge = remaining_edges[0]
        rest = remaining_edges[1:]
        # Try all 2-colorings of this edge that are non-monochromatic
        # Edge has 3 vertices; valid 2-colorings: 001, 010, 011, 100, 101, 110 (up to complement)
        # We can fix the first vertex to 0 WLOG for the edge (symmetry)
        # Actually we need to be careful: the coloring is global
        # Let's just try both assignments for the first uncolored vertex in the edge
        
        # Find first uncolored vertex in this edge
        colored_dict = dict(colored) if colored else {}
        uncolored_in_edge = [v for v in edge if v not in colored_dict]
        
        if not uncolored_in_edge:
            # All vertices in edge are colored; check if non-monochromatic
            colors_in_edge = [colored_dict[v] for v in edge]
            if len(set(colors_in_edge)) == 1:
                return False
            return solve(rest, colored)
        
        v = uncolored_in_edge[0]
        for c in [0, 1]:
            new_colored = colored + ((v, c),) if colored else ((v, c),)
            # Check: is there a conflict? An edge where all other vertices are colored to c
            # and this is the last uncolored - then it would be monochromatic
            # But we'll catch it when we process that edge
            if solve(rest, new_colored):
                return True
        return False
    
    # Start with no colored vertices
    result = solve(tuple(edges_sorted), ())
    solve.cache_clear()
    return result

def _has_valid_3_coloring(edges, n=9):
    """Check if there exists a 3-coloring where every edge is non-monochromatic.
    This is almost always true for sparse hypergraphs, so we can construct one."""
    # Greedy: assign colors to avoid monochromatic edges
    # For the Fano plane, a known 3-coloring works
    # General approach: try to find a 3-coloring via backtracking
    @lru_cache(maxsize=None)
    def solve(remaining, colors_assigned):
        if not remaining:
            return True
        edge = remaining[0]
        rest = remaining[1:]
        # Try all 3^3 = 27 colorings of this edge that are non-monochromatic
        # But we can use the already-assigned colors for vertices in the edge
        colors_dict = dict(colors_assigned) if colors_assigned else {}
        
        # Determine which vertices in this edge are already colored
        fixed = {v: colors_dict[v] for v in edge if v in colors_dict}
        uncolored = [v for v in edge if v not in colors_dict]
        
        if not uncolored:
            # Check if current coloring makes this edge monochromatic
            if len(set(fixed.values())) == 1:
                return False
            return solve(rest, colors_assigned)
        
        # Try all valid colorings for the uncolored vertices
        for combo in itertools.product([0, 1, 2], repeat=len(uncolored)):
            new_fixed = dict(fixed)
            for v, c in zip(uncolored, combo):
                new_fixed[v] = c
            # Check non-monochromatic
            if len(set(new_fixed.values())) > 1:
                new_assigned = colors_assigned + tuple((v, c) for v, c in zip(uncolored, combo)) if colors_assigned else tuple((v, c) for v, c in zip(uncolored, combo))
                if solve(rest, new_assigned):
                    return True
        return False
    
    edges_sorted = tuple(sorted([tuple(sorted(e)) for e in edges]))
    result = solve(edges_sorted, ())
    solve.cache_clear()
    return result

def _find_3_coloring(edges, n=9):
    """Find an explicit 3-coloring where every edge is non-monochromatic."""
    @lru_cache(maxsize=None)
    def solve(remaining, colors_assigned):
        if not remaining:
            return colors_assigned
        edge = remaining[0]
        rest = remaining[1:]
        colors_dict = dict(colors_assigned) if colors_assigned else {}
        fixed = {v: colors_dict[v] for v in edge if v in colors_dict}
        uncolored = [v for v in edge if v not in colors_dict]
        
        if not uncolored:
            if len(set(fixed.values())) == 1:
                return None
            return solve(rest, colors_assigned)
        
        for combo in itertools.product([0, 1, 2], repeat=len(uncolored)):
            new_fixed = dict(fixed)
            for v, c in zip(uncolored, combo):
                new_fixed[v] = c
            if len(set(new_fixed.values())) > 1:
                new_assigned = colors_assigned + tuple((v, c) for v, c in zip(uncolored, combo)) if colors_assigned else tuple((v, c) for v, c in zip(uncolored, combo))
                result = solve(rest, new_assigned)
                if result is not None:
                    return result
        return None
    
    edges_sorted = tuple(sorted([tuple(sorted(e)) for e in edges]))
    result = solve(edges_sorted, ())
    solve.cache_clear()
    
    if result is None:
        # Should not happen for valid hypergraphs, but fallback
        return [0, 0, 1, 1, 2, 0, 0, 0, 0]
    
    color_map = dict(result)
    return [color_map.get(i, 0) for i in range(n)]

def construct():
    start_time = time.time()
    
    # The Fano plane is the known minimum 3-chromatic 3-uniform hypergraph with 7 edges
    # Vertices: 0-6 (Fano), 7-8 (isolated)
    fano_edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # Verify: no 2-coloring exists
    if _is_2_colorable(fano_edges, 9):
        # This should never happen for Fano plane
        # Fallback to a known 3-chromatic hypergraph
        edges = [
            (0, 1, 2), (0, 1, 3), (0, 2, 4), (0, 3, 5), (0, 4, 6),
            (0, 5, 7), (0, 6, 8), (1, 2, 5), (1, 3, 6), (1, 4, 7),
            (2, 3, 7), (2, 4, 8), (3, 4, 8), (5, 6, 7), (5, 7, 8), (6, 7, 8)
        ]
    else:
        edges = fano_edges
    
    # Find a valid 3-coloring
    colors = _find_3_coloring(edges, 9)
    
    # Verify the coloring is valid (every edge non-monochromatic)
    for e in edges:
        if colors[e[0]] == colors[e[1]] == colors[e[2]]:
            # Fallback coloring
            colors = [0, 0, 1, 1, 2, 0, 0, 0, 0]
            break
    
    # Ensure types are correct
    edges_list = [list(e) for e in edges]
    colors_list = [int(c) for c in colors]
    
    return {'edges': edges_list, 'colors': colors_list}
# EVOLVE-BLOCK-END
