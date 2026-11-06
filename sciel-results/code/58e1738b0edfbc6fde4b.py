"""Few edges in 5-uniform hypergraphs without Property B (m(5)).
Strategy: random seed hypergraph -> iteratively remove non-critical edges
while verifying non-2-colourability with a fast exact backtracker (unit propagation).
"""
import time
import random
from itertools import combinations

START = time.time()
DEADLINE = 150.0
SAFETY = 8.0


def _time_left():
    return DEADLINE - SAFETY - (time.time() - START)


def _is_not_2_colorable(n, edges):
    """Exact check: does the 5-uniform hypergraph have NO proper 2-colouring?
    Uses backtracking with unit propagation.
    Returns True if no valid 2-colouring exists (i.e. hypergraph is not 2-colourable).
    """
    m = len(edges)
    if m == 0:
        return False  # trivially 2-colourable

    # Build incidence: for each edge, list of vertex indices
    # We'll use a recursive backtracker with unit propagation.
    # State: assignment array (-1 unknown, 0/1 colour)
    # An edge is "satisfied" if it has both colours.
    # An edge is "critical" if all assigned vertices have the same colour -> must get the other colour from unassigned.
    # Unit propagation: if an edge has all its assigned vertices as colour c, then all unassigned vertices in that edge must be 1-c.
    # If an edge has all 5 vertices assigned same colour -> conflict (edge is monochromatic -> valid colouring for that edge).
    # Wait: "proper 2-colouring" means every edge has BOTH colours. So a monochromatic edge is INVALID.
    # We want to find an assignment where NO edge is monochromatic.
    # So: if an edge has 4 vertices assigned colour c, the 5th must be 1-c.
    # If an edge has all 5 assigned same colour -> conflict (this assignment path is dead).

    # For speed, precompute edge membership
    # Use bitmasks for vertices
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)

    # Count bits
    def popcount(x):
        return bin(x).count('1')

    assignment = [-1] * n  # -1, 0, 1

    # We'll use iterative deepening / simple backtracking with ordering
    # To speed up, order vertices by degree
    vertex_degree = [0] * n
    for e in edges:
        for v in e:
            vertex_degree[v] += 1

    # Order: most constrained first (highest degree)
    vertex_order = sorted(range(n), key=lambda v: -vertex_degree[v])

    def propagate():
        """Apply unit propagation. Returns False on conflict, True otherwise.
        Modifies assignment in-place."""
        changed = True
        while changed:
            changed = False
            for i, mask in enumerate(edge_masks):
                # Count assigned vertices of each colour in this edge
                c0 = 0
                c1 = 0
                unassigned_count = 0
                for v in range(n):
                    if mask & (1 << v):
                        a = assignment[v]
                        if a == 0:
                            c0 += 1
                        elif a == 1:
                            c1 += 1
                        else:
                            unassigned_count += 1

                if c0 > 0 and c1 > 0:
                    continue  # already both colours
                if c0 == 0 and c1 == 0:
                    continue  # all unassigned
                # One colour dominates
                if c1 == 0:
                    # All assigned are 0, need at least one 1
                    if unassigned_count == 0:
                        return False  # conflict: all 5 are 0
                    # All unassigned in this edge must be 1
                    for v in range(n):
                        if mask & (1 << v) and assignment[v] == -1:
                            assignment[v] = 1
                            changed = True
                elif c0 == 0:
                    # All assigned are 1, need at least one 0
                    if unassigned_count == 0:
                        return False
                    for v in range(n):
                        if mask & (1 << v) and assignment[v] == -1:
                            assignment[v] = 0
                            changed = True
        return True

    def backtrack(idx):
        """Try to find a valid 2-colouring. Returns True if found."""
        if idx == n:
            # All assigned, check all edges have both colours
            for mask in edge_masks:
                c0 = 0
                c1 = 0
                for v in range(n):
                    if mask & (1 << v):
                        if assignment[v] == 0:
                            c0 += 1
                        elif assignment[v] == 1:
                            c1 += 1
                if c0 == 0 or c1 == 0:
                    return False
            return True

        v = vertex_order[idx]
        # Try 0 and 1
        for color in (0, 1):
            assignment[v] = color
            if propagate():
                if backtrack(idx + 1):
                    return True
            assignment[v] = -1
        return False

    return not backtrack(0)


def _is_not_2_colorable_fast(n, edges):
    """Faster version using bitmasks and optimized propagation."""
    m = len(edges)
    if m == 0:
        return False

    # Precompute edges as lists of vertex indices for fast access
    edge_verts = [list(e) for e in edges]

    assignment = [-1] * n

    # Vertex ordering by degree
    deg = [0] * n
    for e in edge_verts:
        for v in e:
            deg[v] += 1
    vorder = sorted(range(n), key=lambda v: -deg[v])

    def propagate():
        changed = True
        while changed:
            changed = False
            for e in edge_verts:
                c0 = c1 = un = 0
                for v in e:
                    a = assignment[v]
                    if a == 0:
                        c0 += 1
                    elif a == 1:
                        c1 += 1
                    else:
                        un += 1
                if c0 > 0 and c1 > 0:
                    continue
                if c0 == 0 and c1 == 0:
                    continue
                if c1 == 0:
                    if un == 0:
                        return False
                    for v in e:
                        if assignment[v] == -1:
                            assignment[v] = 1
                            changed = True
                else:  # c0 == 0
                    if un == 0:
                        return False
                    for v in e:
                        if assignment[v] == -1:
                            assignment[v] = 0
                            changed = True
        return True

    def bt(idx):
        if idx == n:
            for e in edge_verts:
                has0 = has1 = False
                for v in e:
                    if assignment[v] == 0:
                        has0 = True
                    elif assignment[v] == 1:
                        has1 = True
                if not has0 or not has1:
                    return False
            return True
        v = vorder[idx]
        for col in (0, 1):
            assignment[v] = col
            if propagate():
                if bt(idx + 1):
                    return True
            assignment[v] = -1
        return False

    return not bt(0)


def _try_remove_edge(n, edges, idx):
    """Try removing edge at idx. Returns True if hypergraph remains non-2-colourable."""
    new_edges = edges[:idx] + edges[idx+1:]
    if _is_not_2_colorable_fast(n, new_edges):
        edges.pop(idx)
        return True
    return False


def construct():
    """Construct a small 5-uniform hypergraph with no proper 2-colouring."""
    start = time.time()
    deadline = start + 142.0  # safety margin

    rng = random.Random(42)

    # Start with a random hypergraph
    # Use n=12 vertices, start with ~60 random edges
    best_n = 12
    best_edges = None

    # Try several random seeds and sizes
    for n in [10, 11, 12, 13, 14]:
        if time.time() > deadline:
            break
        for trial in range(3):
            if time.time() > deadline:
                break
            all_possible = list(combinations(range(n), 5))
            rng.shuffle(all_possible)

            # Start with a subset
            num_edges = min(len(all_possible), 50 + rng.randint(0, 20))
            edges = [list(e) for e in all_possible[:num_edges]]

            # Check if non-2-colourable
            if not _is_not_2_colorable_fast(n, edges):
                continue

            # Try to reduce: remove edges one by one
            changed = True
            while changed and time.time() < deadline:
                changed = False
                # Try removing edges in random order
                order = list(range(len(edges)))
                rng.shuffle(order)
                for idx in order:
                    if time.time() > deadline:
                        break
                    if _try_remove_edge(n, edges, idx):
                        changed = True
                        break  # restart after a removal

            if best_edges is None or len(edges) < len(best_edges):
                best_n = n
                best_edges = edges[:]

            if len(edges) <= 28:
                break
        if best_edges is not None and len(best_edges) <= 28:
            break

    if best_edges is None:
        # Fallback: complete graph on 9 vertices (known non-2-colourable)
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}

    return {'n': best_n, 'edges': best_edges}
