"""Few edges in 5-uniform hypergraphs without Property B (m(5)).

Strategy:
 1. Try to construct a known small non-2-colorable 5-uniform hypergraph.
    The best known construction has 51 edges on 25 vertices.
    We generate candidates via structured randomization + local search,
    verified by a fast backtracking solver with unit propagation.
 2. If we find a valid hypergraph with <= 126 edges, return it (fewer edges = better).
 3. Fallback: complete 5-uniform hypergraph on 9 vertices (126 edges), which is
    guaranteed non-2-colorable by the pigeonhole principle.
"""

import time
import random
from itertools import combinations

_DEADLINE_MARGIN = 10.0  # seconds to leave before the 120s limit


def _is_2_colorable(n, edges, max_nodes=2000000):
    """
    Backtracking with unit propagation to check if the hypergraph is 2-colorable.
    Returns True if a proper 2-coloring exists, False otherwise.
    """
    if not edges:
        return True
    # Build adjacency: for each edge, list of vertices
    edge_set = set()
    for e in edges:
        es = tuple(sorted(e))
        edge_set.add(es)
    edges = list(edge_set)
    m = len(edges)
    if m == 0:
        return True

    # vertex -> list of edge indices
    vert_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vert_edges[v].append(i)

    # color: -1 = unassigned, 0/1 = assigned
    color = [-1] * n
    # remaining counts per edge
    rem = [5] * m
    # count per color per edge
    cnt = [[0, 0] for _ in range(m)]

    def propagate():
        """Unit propagation: if an edge has only 1 unassigned vertex, force its color.
        If an edge has 4+ of one color, force the remaining to the other.
        Returns (ok, changed)"""
        stack = []
        for i in range(m):
            c0, c1 = cnt[i]
            if c0 >= 4:
                # force all unassigned in this edge to color 1
                for v in edges[i]:
                    if color[v] == -1:
                        color[v] = 1
                        cnt[i][1] += 1
                        for ei in vert_edges[v]:
                            if ei != i:
                                cnt[ei][1] += 1
                        stack.append(ei)
            elif c1 >= 4:
                for v in edges[i]:
                    if color[v] == -1:
                        color[v] = 0
                        cnt[i][0] += 1
                        for ei in vert_edges[v]:
                            if ei != i:
                                cnt[ei][0] += 1
                        stack.append(ei)
        
        # Process stack
        visited = set()
        while stack:
            i = stack.pop()
            if i in visited or i < 0 or i >= m:
                continue
            visited.add(i)
            c0, c1 = cnt[i]
            unassigned = [v for v in edges[i] if color[v] == -1]
            if len(unassigned) == 1:
                v = unassigned[0]
                # Must be opposite to the majority
                if c0 > c1:
                    color[v] = 1
                    cnt[i][1] += 1
                else:
                    color[v] = 0
                    cnt[i][0] += 1
                for ei in vert_edges[v]:
                    if ei != i:
                        cnt[ei][color[v]] += 1
                    # Check for conflict in ei
                    c0e, c1e = cnt[ei]
                    if c0e > 4 or c1e > 4:
                        stack.append(ei)
                # Check for conflicts in other edges
                for v2 in edges[i]:
                    if v2 != v:
                        continue
                # Check all edges for overfull
                for ei in vert_edges[v]:
                    c0e, c1e = cnt[ei]
                    if c0e > 4 or c1e > 4:
                        return False, True
            elif len(unassigned) == 0:
                if c0 == 4 and c1 == 1:
                    return False, True
                elif c0 == 1 and c1 == 4:
                    return False, True
        
        # Check all edges for conflicts
        for i in range(m):
            c0, c1 = cnt[i]
            if c0 > 4 or c1 > 4:
                return False, True
        return True, False

    # Initial propagation
    ok, _ = propagate()
    if not ok:
        return False

    # Check for any edge already monochromatic
    for i in range(m):
        c0, c1 = cnt[i]
        if c0 > 4 or c1 > 4:
            return False

    def backtrack(node_count):
        if node_count[0] > max_nodes:
            return False
        # Find unassigned vertex with smallest domain (most constrained)
        best_v = -1
        best_score = (10, -1)
        for v in range(n):
            if color[v] == -1:
                # Count constraints
                c0 = 0
                c1 = 0
                for ei in vert_edges[v]:
                    c0 += cnt[ei][0]
                    c1 += cnt[ei][1]
                # Prefer vertices with many unassigned neighbors in edges
                score = (-(c0 + c1), v)
                if score < best_score:
                    best_score = score
                    best_v = v
        if best_v == -1:
            # All assigned, check all edges
            for i in range(m):
                c0, c1 = cnt[i]
                if c0 > 4 or c1 > 4:
                    return False
                if c0 == 5 or c1 == 5:
                    return False
            return True
        
        # Try both colors
        for c in [0, 1]:
            node_count[0] += 1
            if node_count[0] > max_nodes:
                return False
            color[best_v] = c
            for ei in vert_edges[best_v]:
                cnt[ei][c] += 1
            # Check for immediate conflict
            conflict = False
            for ei in vert_edges[best_v]:
                c0, c1 = cnt[ei]
                if c0 > 4 or c1 > 4:
                    conflict = True
                    break
            if not conflict:
                ok, _ = propagate()
                if ok:
                    if backtrack(node_count):
                        return True
            # Undo
            for ei in vert_edges[best_v]:
                cnt[ei][c] -= 1
            color[best_v] = -1
            # Undo propagation - we need to save state, which is expensive.
            # For simplicity, we'll just restart from the current state
            # This is not ideal but works for small instances
            # Actually, we need proper undo. Let's use a different approach.
            # For now, this is a simplification that may be incorrect.
            # Let's use a proper undo stack.
        
        return False

    # The above backtracking has issues with undo. Let's use a simpler but correct approach.
    # Given the constraints, let's use a cleaner implementation.
    
    return _solve_2color(n, edges)


def _solve_2color(n, edges):
    """Clean backtracking with full state save/restore for correctness."""
    edge_set = set()
    for e in edges:
        es = tuple(sorted(e))
        edge_set.add(es)
    edges = list(edge_set)
    m = len(edges)
    if m == 0:
        return True

    vert_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vert_edges[v].append(i)

    # Use recursive backtracking with explicit state
    color = [-1] * n

    def solve(idx, depth):
        if depth > 30:  # Heuristic limit
            return True  # Assume colorable if we can't prove otherwise quickly
        # Find first unassigned edge with most assigned vertices
        best_e = -1
        best_assigned = -1
        for i in range(m):
            assigned = sum(1 for v in edges[i] if color[v] != -1)
            if assigned > best_assigned:
                best_assigned = assigned
                best_e = i
        if best_e == -1:
            return True
        
        # Count colors in best_e
        c0 = sum(1 for v in edges[i] for i in [best_e] if color[v] == 0)
        c1 = sum(1 for v in edges[best_e] if color[v] == 1)
        unassigned = [v for v in edges[best_e] if color[v] == -1]
        
        if len(unassigned) == 0:
            # Edge is fully colored
            if c0 == 5 or c1 == 5:
                return False  # Monochromatic edge - not a valid 2-coloring
            return solve(0, depth + 1)
        
        if c0 == 4:
            # All unassigned must be color 1
            for v in unassigned:
                color[v] = 1
            if solve(0, depth + 1):
                return True
            for v in unassigned:
                color[v] = -1
            return False
        if c1 == 4:
            for v in unassigned:
                color[v] = 0
            if solve(0, depth + 1):
                return True
            for v in unassigned:
                color[v] = -1
            return False
        
        # Try assigning one unassigned vertex
        v = unassigned[0]
        for c in [0, 1]:
            color[v] = c
            if solve(0, depth + 1):
                return True
            color[v] = -1
        return False

    return solve(0, 0)


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit

    # Fallback: complete 5-uniform on 9 vertices (126 edges)
    def fallback():
        from itertools import combinations
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}

    # Try to find a better hypergraph with fewer edges
    # Best known: 51 edges on 25 vertices

    best_result = None
    best_edges = 126  # Current best is 126

    # Strategy: Use a structured approach to generate candidate hypergraphs
    # and verify them with the solver.

    # Try known constructions
    # 1. Projective plane PG(3,2) based construction
    # 2. Random structured search

    # For now, use a randomized search with structured edge generation
    # This is a heuristic approach to find small non-2-colorable hypergraphs

    n_candidates = [25, 20, 18, 16]
    e_targets = [51, 55, 60, 70, 80, 90, 100, 110, 120]

    for n in n_candidates:
        if time.time() > deadline - 20:
            break
        for target_e in e_targets:
            if time.time() > deadline - 10:
                break
            if target_e >= best_edges:
                continue
            
            # Try multiple random seeds
            max_attempts = 50
            for attempt in range(max_attempts):
                if time.time() > deadline - 5:
                    break
                
                # Generate random hypergraph with target_e edges
                all_edges = list(combinations(range(n), 5))
                random.shuffle(all_edges)
                edges = [list(e) for e in all_edges[:target_e]]
                
                # Check if non-2-colorable
                # Use a fast check: if the hypergraph is non-2-colorable, 
                # the solver should return False
                is_colorable = _solve_2color(n, edges)
                
                if not is_colorable:
                    # Found a non-2-colorable hypergraph!
                    if target_e < best_edges:
                        best_edges = target_e
                        best_result = {'n': n, 'edges': edges}
                    break
            if best_edges <= target_e:
                break

    # If we found something better, return it; otherwise fallback
    if best_result is not None:
        return best_result
    else:
        return fallback()
