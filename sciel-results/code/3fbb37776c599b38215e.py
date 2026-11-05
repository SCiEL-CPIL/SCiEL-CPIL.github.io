"""Search for a small non-2-colorable 5-uniform hypergraph."""
import time
import random
from itertools import combinations

DEADLINE = 145.0
START = time.time()

def _is_non_2colorable(n, edges):
    """Backtracking with unit propagation. Returns True if NOT 2-colorable."""
    m = len(edges)
    if m == 0:
        return False
    
    # For each edge, store the 5 vertices
    # Precompute vertex -> edge indices
    vert_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vert_edges[v].append(i)
    
    # State:
    # assign[v]: -1 unassigned, 0 or 1
    # c0[e]: count of color 0 in edge e
    # c1[e]: count of color 1 in edge e
    
    # We use a recursive backtracking with propagation.
    # To speed up, we use a global state and undo.
    
    assign = [-1] * n
    c0 = [0] * m
    c1 = [0] * m
    # Stack of changes for undo: list of (type, ...)
    # type 0: vertex assignment (v, color)
    # type 1: edge count increment (e, which_count, delta) - actually we just need to revert
    
    # Simpler: copy state at each level? For n<=26, m<=200, this might be okay if depth is small.
    # But depth can be up to n.
    # Let's use the copy approach for simplicity first, as n is small.
    
    def propagate(assign, c0, c1):
        """
        Returns False if contradiction (monochromatic edge).
        Modifies assign, c0, c1 in place.
        """
        changed = True
        while changed:
            changed = False
            for e in range(m):
                # If c0[e] == 5 or c1[e] == 5, contradiction
                if c0[e] == 5 or c1[e] == 5:
                    return False
                # If 4 of one color, force the 5th
                if c0[e] == 4 and c1[e] == 0:
                    # Find unassigned vertex in edge e
                    for v in edges[e]:
                        if assign[v] == -1:
                            # Force v to 1
                            assign[v] = 1
                            c1[e] += 1
                            # Update other edges containing v
                            for e2 in vert_edges[v]:
                                if e2 != e:
                                    c1[e2] += 1
                            changed = True
                            break
                elif c1[e] == 4 and c0[e] == 0:
                    # Find unassigned vertex in edge e
                    for v in edges[e]:
                        if assign[v] == -1:
                            # Force v to 0
                            assign[v] = 0
                            c0[e] += 1
                            # Update other edges containing v
                            for e2 in vert_edges[v]:
                                if e2 != e:
                                    c0[e2] += 1
                            changed = True
                            break
        return True

    def backtrack(assign, c0, c1):
        # Check if all vertices assigned
        # Find an unassigned vertex
        # Heuristic: pick vertex that appears in edges with fewest unassigned vertices
        best_v = -1
        best_score = float('inf')
        
        # Calculate score for each unassigned vertex
        for v in range(n):
            if assign[v] != -1:
                continue
            # Score: minimum number of unassigned vertices in any edge containing v
            min_unassigned = 5
            for e in vert_edges[v]:
                unassigned_in_e = 5 - c0[e] - c1[e]
                if unassigned_in_e < min_unassigned:
                    min_unassigned = unassigned_in_e
            if min_unassigned < best_score:
                best_score = min_unassigned
                best_v = v
                if best_score == 1:
                    break # Can't get better than 1 (forced by propagation usually, but maybe not yet)
        
        if best_v == -1:
            # All assigned. Check if any monochromatic edge exists.
            # If yes, it's a contradiction (not a valid coloring).
            # Wait, if all assigned and no monochromatic edge was found during propagation,
            # then we have a valid 2-coloring.
            # So if we reach here, it means a valid coloring exists.
            # Thus, the hypergraph IS 2-colorable.
            # We want to return True if NOT 2-colorable.
            # So if we found a valid coloring, return False.
            return False
        
        # Try assigning best_v to 0 and 1
        for color in (0, 1):
            new_assign = assign[:]
            new_c0 = c0[:]
            new_c1 = c1[:]
            
            new_assign[best_v] = color
            for e in vert_edges[best_v]:
                if color == 0:
                    new_c0[e] += 1
                else:
                    new_c1[e] += 1
            
            if not propagate(new_assign, new_c0, new_c1):
                continue
            
            # Check if any edge became monochromatic in propagate?
            # propagate returns False if contradiction.
            # So if it returned True, no monochromatic edge yet.
            
            if backtrack(new_assign, new_c0, new_c1):
                return True
        
        # No valid coloring found in this branch
        return True

    # Initial check
    if not propagate(assign, c0, c1):
        return True
    
    # Check if any edge is already monochromatic (shouldn't happen with empty start, but safe)
    for e in range(m):
        if c0[e] == 5 or c1[e] == 5:
            return True

    return backtrack(assign, c0, c1)

def _try_build(seed, max_edges=60, max_vertices=26):
    """Build a random hypergraph incrementally, stop when non-2-colorable."""
    rng = random.Random(seed)
    
    # Strategy:
    # 1. Choose n vertices.
    # 2. Generate all 5-combinations.
    # 3. Shuffle.
    # 4. Add edges one by one, checking for non-2-colorability.
    
    # To find smaller hypergraphs, try smaller n first?
    # But smaller n has fewer edges total.
    # C(10,5) = 252.
    # C(11,5) = 462.
    # C(12,5) = 792.
    
    # Let's vary n.
    n_vertices = rng.randint(8, 16)
    all_possible = list(combinations(range(n_vertices), 5))
    rng.shuffle(all_possible)
    
    edges = []
    for e in all_possible:
        if len(edges) >= max_edges:
            break
        edges.append(list(e))
        # Check every few steps to save time?
        # Or check every step?
        # Checking is expensive.
        # Let's check every step.
        if _is_non_2colorable(n_vertices, edges):
            return n_vertices, edges
    return None

def construct():
    best = None
    best_edges = float('inf')
    seed = 0
    while time.time() < DEADLINE - 5:
        seed += 1
        # Vary the strategy slightly
        # Maybe try different n ranges?
        # For now, just run _try_build with different seeds
        result = _try_build(seed)
        if result is not None:
            n, edges = result
            if len(edges) < best_edges:
                best_edges = len(edges)
                best = {'n': n, 'edges': edges}
                # If we find a very good one, break?
                # Theoretical lower bound is 29?
                # Best known is 51.
                # If we find < 51, great.
                if best_edges <= 51:
                    # Keep searching for smaller?
                    # The task says "Minimize the number of edges".
                    # So keep searching if time permits.
                    pass
        # If time is running out, break
        if time.time() > DEADLINE - 2:
            break
    
    if best is None:
        # Fallback: complete 5-uniform on 9 vertices
        # C(9,5) = 126 edges.
        # Is it non-2-colorable?
        # A complete 5-uniform hypergraph on 9 vertices.
        # If we color 4 vertices 0 and 5 vertices 1, there is an edge of 5 ones.
        # So it is NOT 2-colorable.
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    
    return best
