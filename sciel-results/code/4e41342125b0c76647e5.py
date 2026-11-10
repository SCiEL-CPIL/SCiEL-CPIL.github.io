"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Hard deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 140s wall-clock safety margin (evaluator kills at 150s)

def _time_remaining():
    return _DEADLINE - time.time()

def _is_2_colorable(n, edges, deadline):
    """
    Check if the hypergraph (n vertices, edges as list of 5-sets) is 2-colorable
    using backtracking with unit propagation.
    Returns True if 2-colorable, False if not.
    """
    m = len(edges)
    # For each vertex, precompute the list of edges it belongs to
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    # State: color of each vertex: -1 = uncolored, 0 = color A, 1 = color B
    # An edge is "satisfied" (violated for 2-coloring) if all 5 vertices are the same color.
    # We want to find a coloring where NO edge is monochromatic.
    # Equivalently, we want to avoid any edge being all-0 or all-1.
    
    # Unit propagation idea: if 4 vertices of an edge are colored the same, the 5th must be the other color.
    # If all 5 are uncolored or mixed, no immediate constraint.
    # Contradiction: if an edge has 5 vertices all colored 0, or all colored 1 -> fail.
    # Also if 4 are 0 and 1 is 1 -> ok. 4 are 0 and 1 is uncolored -> force it to 1.
    # 4 are 0 and 1 is already 1 -> ok. 4 are 0 and 1 is 0 -> contradiction.
    
    # We'll implement a recursive backtracking with propagation.
    # For efficiency, we track for each edge the count of 0s and 1s.
    
    # Let's use a simpler but effective approach:
    # Assign vertices one by one (in some order), with propagation.
    
    # Precompute edge membership as bitmasks or just use lists.
    # n <= 26, so we can use integer bitmasks for color classes.
    # color0_mask: vertices colored 0
    # color1_mask: vertices colored 1
    # uncolored_mask: vertices uncolored
    
    # Actually, let's use a dict or list for colors, and propagate.
    
    def propagate(colors, edge_color_counts):
        """
        colors: list of -1, 0, 1 for each vertex
        edge_color_counts: list of (count0, count1) for each edge
        Returns False if contradiction, else updates colors and edge_color_counts.
        """
        changed = True
        while changed:
            changed = False
            for i, e in enumerate(edges):
                c0, c1 = edge_color_counts[i]
                # Check for contradiction: all 5 same color
                if c0 == 5 or c1 == 5:
                    return False
                # If 4 of one color, force the 5th
                if c0 == 4:
                    # Find the uncolored vertex in this edge
                    for v in e:
                        if colors[v] == -1:
                            colors[v] = 1
                            edge_color_counts[i] = (4, 1)
                            changed = True
                            break
                    # Also need to update other edges containing v
                    # We'll handle this in a separate pass or just recompute
                elif c1 == 4:
                    for v in e:
                        if colors[v] == -1:
                            colors[v] = 0
                            edge_color_counts[i] = (1, 4)
                            changed = True
                            break
            # After forcing a color, we need to update edge counts for other edges
            # This is getting complex. Let's use a simpler propagation:
            # Just check all edges after each assignment.
        return True
    
    # Simpler approach: recursive backtracking with full propagation after each step.
    # Given n <= 26 and m <= 200, this should be fast enough for small m.
    
    # Let's use a cleaner implementation:
    
    def _backtrack(vertex_order, idx, colors):
        if idx == len(vertex_order):
            # All vertices colored. Check all edges are not monochromatic.
            for e in edges:
                c = colors[e[0]]
                if all(colors[v] == c for v in e[1:]):
                    return False
            return True
        
        v = vertex_order[idx]
        
        # Try color 0
        colors[v] = 0
        if _propagate_and_check(colors):
            if _backtrack(vertex_order, idx + 1, colors):
                return True
        colors[v] = -1
        
        # Try color 1
        colors[v] = 1
        if _propagate_and_check(colors):
            if _backtrack(vertex_order, idx + 1, colors):
                return True
        colors[v] = -1
        
        return False
    
    def _propagate_and_check(colors):
        """
        Given current colors (with some -1), propagate constraints and check for contradictions.
        Returns False if contradiction, True otherwise.
        """
        # Propagation loop
        while True:
            propagated = False
            for e in edges:
                vals = [colors[v] for v in e]
                c0 = vals.count(0)
                c1 = vals.count(1)
                c_un = vals.count(-1)
                
                if c0 == 5 or c1 == 5:
                    return False
                
                if c0 == 4 and c_un == 1:
                    # Find the uncolored vertex
                    for v in e:
                        if colors[v] == -1:
                            colors[v] = 1
                            propagated = True
                            break
                elif c1 == 4 and c_un == 1:
                    for v in e:
                        if colors[v] == -1:
                            colors[v] = 0
                            propagated = True
                            break
            
            if not propagated:
                break
        
        # Check for any edge that's already monochromatic (shouldn't happen if propagation works)
        for e in edges:
            vals = [colors[v] for v in e]
            if -1 not in vals:
                if all(v == 0 for v in vals) or all(v == 1 for v in vals):
                    return False
        
        return True
    
    # Vertex ordering: use a heuristic. Try vertices with highest degree first.
    degree = [len(vertex_edges[v]) for v in range(n)]
    vertex_order = sorted(range(n), key=lambda v: -degree[v])
    
    colors = [-1] * n
    return _backtrack(vertex_order, 0, colors)


def _try_random_search(n, target_edges, deadline, rng):
    """
    Try to find a non-2-colorable hypergraph with exactly target_edges edges on n vertices.
    Uses random edge selection and checks.
    """
    all_edges = list(combinations(range(n), 5))
    rng.shuffle(all_edges)
    
    edges = []
    edge_set = set()
    
    for e in all_edges:
        if len(edges) >= target_edges:
            break
        if e not in edge_set:
            edges.append(e)
            edge_set.add(e)
        
        # Periodically check if we've found a non-2-colorable one
        if len(edges) % 5 == 0:
            if _time_remaining() < 5:
                return None
            if not _is_2_colorable(n, edges, deadline):
                return edges
    
    # Final check
    if len(edges) == target_edges and not _is_2_colorable(n, edges, deadline):
        return edges
    
    return None


def _try_greedy_add(n, initial_edges, deadline, rng):
    """
    Start from initial_edges and greedily add edges to make it non-2-colorable.
    """
    edges = list(initial_edges)
    edge_set = set(tuple(e) for e in edges)
    
    all_edges = list(combinations(range(n), 5))
    rng.shuffle(all_edges)
    
    current_count = len(edges)
    
    for e in all_edges:
        if _time_remaining() < 10:
            break
        if tuple(e) in edge_set:
            continue
        if current_count >= 200:
            break
        
        edges.append(e)
        edge_set.add(tuple(e))
        current_count += 1
        
        # Check if non-2-colorable
        if not _is_2_colorable(n, edges, deadline):
            return edges
    
    return None


def construct():
    """
    Find a 5-uniform hypergraph with no proper 2-coloring, minimizing the number of edges.
    Returns {'n': n, 'edges': [[v1, ..., v5], ...]}
    """
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 140.0
    
    best_result = None
    best_edge_count = 201  # Start with the trivial complete graph on 9 vertices (126 edges)
    
    # First, get a valid baseline: complete 5-graph on 9 vertices
    baseline_edges = [list(e) for e in combinations(range(9), 5)]
    best_result = {'n': 9, 'edges': baseline_edges}
    best_edge_count = len(baseline_edges)
    
    # Try to find smaller constructions
    # Known: best is 51 edges. Let's search for smaller ones.
    
    # Strategy: For various n, try to find non-2-colorable hypergraphs with few edges.
    # Use random search with different n values.
    
    rng = random.Random(42)
    
    # Try n from 10 to 26, and for each, try to find small edge counts
    # We'll use a random approach: pick random edges and check.
    
    for n in range(10, 27):
        if _time_remaining() < 15:
            break
        
        # Try different target edge counts, starting from low
        # The theoretical minimum is around 29 (since 28 is always 2-colorable)
        for target in range(29, 60):
            if _time_remaining() < 10:
                break
            
            if target >= best_edge_count:
                break
            
            # Try a few random attempts
            for attempt in range(3):
                if _time_remaining() < 5:
                    break
                
                result = _try_random_search(n, target, _DEADLINE, rng)
                if result is not None and len(result) < best_edge_count:
                    best_edge_count = len(result)
                    best_result = {'n': n, 'edges': [list(e) for e in result]}
                    break  # Found one with this target, move to next target
    
    # If we found something, return it
    if best_result is not None:
        return best_result
    
    # Fallback: return the baseline
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
# EVOLVE-BLOCK-END
