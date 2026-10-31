"""Combinatorial branch-and-bound over a triangular lattice for hexagon packing."""
import math
import time

N = 11

# 6 outward face normals of a regular hexagon (flat-top, faces at 0,60,...,300)
_NORMALS = [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in (0, 60, 120, 180, 240, 300)]

# Lattice: triangular lattice with spacing just under the non-overlap threshold 2.0
# Use spacing 2.0 exactly (safe: equal distance => touching, allowed for disjoint interiors).
# Use a slightly smaller spacing to allow tiny rotations? Keep 2.0 for validity safety.
_SPACING = 2.0
_BX = _SPACING
_BY = _SPACING * math.sqrt(3.0) / 2.0

# Build a bounded lattice of candidate cells around origin.
# The optimal packing of 11 unit hexagons: a hexagon of side 2 has 7 cells (center + ring of 6),
# but we need 11. A compact cluster of 11 cells. The minimal enclosing hexagon side for a
# hexagonal cluster of 11 cells is around 3-ish. We'll generate a generous lattice window.
_LATTICE_R = 4  # number of lattice steps in each direction

def _gen_lattice():
    pts = []
    for i in range(-_LATTICE_R, _LATTICE_R + 1):
        for j in range(-_LATTICE_R, _LATTICE_R + 1):
            x = i * _BX + (j % 2) * _BX / 2.0
            y = j * _BY
            pts.append((x, y))
    return pts

_LATTICE = _gen_lattice()
_NCELLS = len(_LATTICE)

# Precompute pairwise distances to enforce non-overlap (>= 2.0 - eps)
_EPS = 1e-9
_MIN_DIST = 2.0 - _EPS
_DIST_OK = [[True] * _NCELLS for _ in range(_NCELLS)]
for a in range(_NCELLS):
    ax, ay = _LATTICE[a]
    for b in range(a + 1, _NCELLS):
        bx, by = _LATTICE[b]
        d2 = (ax - bx) ** 2 + (ay - by) ** 2
        ok = d2 >= _MIN_DIST * _MIN_DIST - 1e-12
        _DIST_OK[a][b] = ok
        _DIST_OK[b][a] = ok

# Support function value for a set of centers relative to outer center = origin.
# We'll fix outer center at origin and minimize max_i (n_i . c_i + sqrt(3)) / sqrt(3)
# but we can also shift outer center; better to minimize over outer center too.
# For a given set of centers, the minimal s = (max over all 6 directions of (max_i n_i.c_i) + sqrt(3)) / sqrt(3)
# but that's with outer center at origin. To allow shifting outer center, we minimize
# max_i (max_j n_i.c_j - n_i.o) over o. The optimal o equalizes extremes; the minimal
# half-width in direction n_i is (max_j n_i.c_j - min_j n_i.c_j)/2, and s_i = that + sqrt(3)/... 
# Actually outer apothem must be >= inner apothem + max_i (n_i.(c_j - o)).
# min over o of max_i [ max_j n_i.c_j - n_i.o ] = min_o max_i [A_i - n_i.o]
# where A_i = max_j n_i.c_j. By symmetry of hexagon, optimal o makes all 6 equal:
# A_i - n_i.o = t for all i. This is solvable. But simpler: the minimal enclosing hexagon
# (same orientation) has side s where the 6 support planes are at distance s*sqrt(3)/2 from o.
# The minimal s = (max_i [ (max_j n_i.c_j + max_j (-n_i.c_j) ) / 2 ] + sqrt(3)/2 ) ... 
# Let me just compute: for direction i, the width of the cluster in direction n_i is W_i = max_j n_i.c_j - min_j n_i.c_j.
# The outer hexagon needs half-width >= W_i/2 + sqrt(3)/2 in each of the 6 face directions.
# So s = max_i (W_i/2 + sqrt(3)/2) / (sqrt(3)/2) = max_i (W_i + sqrt(3)) / sqrt(3).
# This is independent of o (we center the outer hexagon at the cluster's center). 

_SQRT3 = math.sqrt(3.0)

def _support_widths(centers):
    # centers: list of (x,y)
    # compute W_i for each of 6 directions
    if not centers:
        return 0.0
    # gather projections
    proj = [[0.0, 0.0] for _ in range(6)]  # [max, min]
    for (cx, cy) in centers:
        for i in range(6):
            v = _NORMALS[i][0] * cx + _NORMALS[i][1] * cy
            if v > proj[i][0]:
                proj[i][0] = v
            if v < proj[i][1]:
                proj[i][1] = v
    best = 0.0
    for i in range(6):
        w = proj[i][0] - proj[i][1]
        s = (w + _SQRT3) / _SQRT3
        if s > best:
            best = s
    return best

# Branch and bound: choose N cells from lattice, non-overlapping, minimize support width.
# Use a greedy/branch approach with pruning.

import sys
sys.setrecursionlimit(100000)

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    # We'll do a depth-first branch and bound.
    # State: current selected list, current best objective (global).
    # Prune: if current partial support width (max over selected so far) already >= best, prune.
    # Also, the final objective is monotonic in the set, so partial width is a lower bound? 
    # Not exactly: adding more cells can only increase W_i, so partial width is a lower bound. Good.

    best_obj = [float('inf')]
    best_set = [None]

    # Precompute: for quick lower bound, we track current max projection per direction.
    # We'll maintain arrays cur_max[6], cur_min[6] for the selected set.

    # Order cells: sort by some heuristic (e.g., distance from origin) to find good solutions early.
    cell_order = sorted(range(_NCELLS), key=lambda idx: _LATTICE[idx][0] ** 2 + _LATTICE[idx][1] ** 2)

    # We need exactly N cells. Use branch and bound with a fixed N.
    # To make it tractable, we'll limit the search: only consider cells within a radius.
    # A cluster of 11 cells has diameter ~ 4-5, so radius ~ 3 from centroid. We'll restrict
    # to cells within a bounding box of the lattice (already limited by _LATTICE_R=4).

    # Additionally, to reduce symmetry, we can fix that the "lowest" cell (by some ordering) 
    # is included, or fix the centroid near origin. Simpler: just search.

    # For speed, we'll use a smarter approach: iterative deepening / best-first.
    # But let's try DFS with strong pruning and a good initial solution.

    # Initial solution: a hexagonal cluster of 11 cells.
    # The 7-cell hexagon (side 2) plus 4 more. Let's build a known compact arrangement.
    # Lattice indices for a hexagon of radius 2 (7 cells): center + 6 neighbors.
    # Let's find the center cell and its neighbors in the lattice.
    # Actually, let's just do a greedy construction for an initial bound.

    def _greedy_initial():
        # Pick cells greedily to form a compact cluster.
        # Start from the origin cell, add nearest non-overlapping cells.
        # Find the lattice index closest to (0,0).
        origin_idx = min(range(_NCELLS), key=lambda i: _LATTICE[i][0] ** 2 + _LATTICE[i][1] ** 2)
        selected = [origin_idx]
        while len(selected) < N:
            sel_set = set(selected)
            # Find the candidate cell that minimizes the increase in support width,
            # among non-overlapping cells.
            best_cell = None
            best_obj_val = float('inf')
            for c in cell_order:
                if c in sel_set:
                    continue
                # check overlap with all selected
                ok = True
                for s in selected:
                    if not _DIST_OK[s][c]:
                        ok = False
                        break
                if not ok:
                    continue
                # compute new support width
                new_centers = [_LATTICE[s] for s in selected] + [_LATTICE[c]]
                obj = _support_widths(new_centers)
                if obj < best_obj_val:
                    best_obj_val = obj
                    best_cell = c
            if best_cell is None:
                break
            selected.append(best_cell)
        if len(selected) == N:
            centers = [_LATTICE[s] for s in selected]
            obj = _support_widths(centers)
            return selected, obj
        return None, float('inf')

    init_set, init_obj = _greedy_initial()
    if init_set is not None:
        best_obj[0] = init_obj
        best_set[0] = list(init_set)

    # Now DFS branch and bound.
    # We'll use a stack-based DFS to avoid recursion depth issues and allow deadline checks.
    # State: (selected_indices_tuple, cur_max[6], cur_min[6], next_start_position_in_cell_order)
    # To keep it manageable, we'll limit the branching: only consider cells from a restricted
    # candidate set. We'll precompute for each partial selection which cells are still feasible.

    # Actually, a cleaner approach: since N is small (11) and the lattice is bounded,
    # we can use a best-first search (A*) with the support width as cost, expanding
    # the most promising partial sets. But that may use too much memory.

    # Let's do DFS with the following pruning:
    # 1. If len(selected) == N: evaluate, update best.
    # 2. If current support width >= best_obj: prune.
    # 3. If len(selected) + (remaining feasible cells) < N: prune.
    # 4. Deadline check.

    # To make the search faster, we'll restrict candidates to those within a certain
    # distance of the current cluster's bounding box, and use a good ordering.

    # Let's implement a recursive DFS with these prunings.
    # We'll maintain cur_max and cur_min as lists of 6 floats.

    def _dfs(selected, cur_max, cur_min, start_pos):
        # selected: list of lattice indices
        # cur_max, cur_min: lists of 6 floats (max/min projection per direction)
        # start_pos: index in cell_order to start from (to avoid permutations)
        if time.time() > DEADLINE:
            return
        if len(selected) == N:
            obj = best_obj[0]  # already computed as we go? Let's compute.
            # Actually we track cur_max/cur_min, so:
            wmax = 0.0
            for i in range(6):
                w = cur_max[i] - cur_min[i]
                s = (w + _SQRT3) / _SQRT3
                if s > wmax:
                    wmax = s
            if wmax < best_obj[0] - 1e-12:
                best_obj[0] = wmax
                best_set[0] = list(selected)
            return

        # Prune: current width already >= best
        cur_wmax = 0.0
        for i in range(6):
            w = cur_max[i] - cur_min[i]
            s = (w + _SQRT3) / _SQRT3
            if s > cur_wmax:
                cur_wmax = s
        if cur_wmax >= best_obj[0] - 1e-12:
            return

        # Remaining cells needed
        remaining_needed = N - len(selected)

        sel_set = set(selected)

        # Iterate over candidate cells from start_pos onward
        for pos in range(start_pos, len(cell_order)):
            if time.time() > DEADLINE:
                return
            c = cell_order[pos]
            if c in sel_set:
                continue
            # Check overlap with all selected
            ok = True
            for s in selected:
                if not _DIST_OK[s][c]:
                    ok = False
                    break
            if not ok:
                continue

            # Compute new cur_max/cur_min
            cx, cy = _LATTICE[c]
            new_max = list(cur_max)
            new_min = list(cur_min)
            for i in range(6):
                v = _NORMALS[i][0] * cx + _NORMALS[i][1] * cy
                if v > new_max[i]:
                    new_max[i] = v
                if v < new_min[i]:
                    new_min[i] = v

            # Prune: new width >= best
            new_wmax = 0.0
            for i in range(6):
                w = new_max[i] - new_min[i]
                s = (w + _SQRT3) / _SQRT3
                if s > new_wmax:
                    new_wmax = s
            if new_wmax >= best_obj[0] - 1e-12:
                continue

            # Recurse
            selected.append(c)
            _dfs(selected, new_max, new_min, pos + 1)
            selected.pop()

    # Initialize cur_max/cur_min from the first cell? We'll start with empty set.
    # But empty set has cur_max = -inf, cur_min = +inf. Let's handle by starting with one cell.
    # Actually, let's just start the DFS with an empty selection and handle the first cell specially.
    # Simpler: start with each possible first cell.

    # To reduce symmetry, we can fix that the selected set's "first" cell (in cell_order) 
    # is the minimum-index cell in the set. This is automatic with our start_pos logic.

    # Start DFS with empty set:
    # cur_max = [-inf]*6, cur_min = [inf]*6
    # But then the first iteration will add a cell and compute properly.
    # Let's use large negative/positive values.
    NEG_INF = -1e18
    POS_INF = 1e18
    init_max = [NEG_INF] * 6
    init_min = [POS_INF] * 6

    _dfs([], init_max, init_min, 0)

    # Build the result
    if best_set[0] is not None:
        centers = [_LATTICE[i] for i in best_set[0]]
        # Center the outer hexagon at the centroid of the cluster (for symmetry, but the
        # objective is shift-invariant, so we can set outer center to origin and centers as-is,
        # or shift so that the cluster is centered. The evaluator likely checks containment
        # with the given outer center. Let's set outer center to the average of centers for
        # a tight fit, and shift inner centers accordingly. Actually, the support width
        # formula assumed outer center at cluster center. Let's compute the centroid and
        # shift.
        cx_avg = sum(p[0] for p in centers) / N
        cy_avg = sum(p[1] for p in centers) / N
        # Shift all centers so that the cluster is centered at origin, and set outer center to origin.
        # This makes the fit tightest.
        shifted = [(p[0] - cx_avg, p[1] - cy_avg) for p in centers]
        # All rotations are 0 (lattice cells are axis-aligned hexagons).
        inner = [[x, y, 0.0] for (x, y) in shifted]
        outer_center = (0.0, 0.0)
        side = best_obj[0]
        angle = 0.0
        return inner, outer_center, side, angle
    else:
        # Fallback: return the seed-like solution
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

