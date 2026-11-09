# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from scipy.optimize import linprog

_TOL = 1e-7
_DEADLINE = 630.0


def _build_lp_for_pattern(pattern, n=10):
    """
    pattern: dict (i,j) -> 0 for x-separation, 1 for y-separation (i<j).
    Variables: [s_0..s_{n-1}, cx_0..cx_{n-1}, cy_0..cy_{n-1}]
    Objective: maximize sum(s) -> minimize -sum(s).
    """
    n = 10
    # index helpers
    def s_idx(i): return i
    def cx_idx(i): return n + i
    def cy_idx(i): return 2 * n + i

    c = np.zeros(3 * n)
    for i in range(n):
        c[s_idx(i)] = -1.0  # minimize -sum(s)

    A_ub = []
    b_ub = []

    # Containment: cx_i + s_i/2 <= 1  ->  cx_i + 0.5*s_i <= 1
    #               cx_i - s_i/2 >= 0  -> -cx_i + 0.5*s_i <= 0
    #               cy_i + s_i/2 <= 1  ->  cy_i + 0.5*s_i <= 1
    #               cy_i - s_i/2 >= 0  -> -cy_i + 0.5*s_i <= 0
    for i in range(n):
        row = np.zeros(3 * n)
        row[cx_idx(i)] = 1.0
        row[s_idx(i)] = 0.5
        A_ub.append(row)
        b_ub.append(1.0)

        row = np.zeros(3 * n)
        row[cx_idx(i)] = -1.0
        row[s_idx(i)] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)

        row = np.zeros(3 * n)
        row[cy_idx(i)] = 1.0
        row[s_idx(i)] = 0.5
        A_ub.append(row)
        b_ub.append(1.0)

        row = np.zeros(3 * n)
        row[cy_idx(i)] = -1.0
        row[s_idx(i)] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)

    # Pairwise separation (touching allowed):
    # if x-sep (0): either cx_j - cx_i >= (s_i+s_j)/2  OR  cx_i - cx_j >= (s_i+s_j)/2
    # We need to choose an orientation. To keep LP, we fix orientation by
    # ordering: assume cx_j >= cx_i + (s_i+s_j)/2  (i left of j).
    # This is a valid packing if the LP is feasible; if not, the pattern is infeasible.
    # Similarly for y.
    for (i, j), axis in pattern.items():
        if axis == 0:  # x-separation: j to the right of i
            row = np.zeros(3 * n)
            row[cx_idx(j)] = 1.0
            row[cx_idx(i)] = -1.0
            row[s_idx(i)] = 0.5
            row[s_idx(j)] = 0.5
            A_ub.append(-row)  # -cx_j + cx_i - 0.5s_i - 0.5s_j <= 0
            b_ub.append(0.0)
        else:  # y-separation: j above i
            row = np.zeros(3 * n)
            row[cy_idx(j)] = 1.0
            row[cy_idx(i)] = -1.0
            row[s_idx(i)] = 0.5
            row[s_idx(j)] = 0.5
            A_ub.append(-row)
            b_ub.append(0.0)

    A_ub = np.array(A_ub, dtype=np.float64)
    b_ub = np.array(b_ub, dtype=np.float64)

    # Bounds: s_i > 0 (use small positive), cx, cy in [0,1]
    bounds = []
    for i in range(n):
        bounds.append((1e-9, 1.0))  # s_i
    for i in range(n):
        bounds.append((0.0, 1.0))  # cx_i
    for i in range(n):
        bounds.append((0.0, 1.0))  # cy_i

    return c, A_ub, b_ub, bounds


def _solve_pattern(pattern):
    """Solve LP for a given pattern. Returns (value, centers, sides) or None if infeasible."""
    n = 10
    c, A_ub, b_ub, bounds = _build_lp_for_pattern(pattern, n)
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        return None
    if res.status != 0:
        return None
    x = res.x
    sides = x[:n]
    cx = x[n:2*n]
    cy = x[2*n:3*n]
    # Verify feasibility with tolerance
    if not np.all(np.isfinite(sides)) or not np.all(sides > 0):
        return None
    # Check containment
    for i in range(n):
        if cx[i] - sides[i]/2 < -_TOL or cx[i] + sides[i]/2 > 1 + _TOL:
            return None
        if cy[i] - sides[i]/2 < -_TOL or cy[i] + sides[i]/2 > 1 + _TOL:
            return None
    # Check pairwise non-overlap
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - _TOL
            if abs(cx[i]-cx[j]) < t and abs(cy[i]-cy[j]) < t:
                return None
    value = float(np.sum(sides))
    centers = np.column_stack([cx, cy]).astype(np.float64)
    sides = sides.astype(np.float64)
    return value, centers, sides


def squares_in_square10() -> tuple:
    n = 10
    t0 = time.time()
    deadline = t0 + _DEADLINE - 30.0  # safety margin

    pairs = list(combinations(range(n), 2))
    num_pairs = len(pairs)

    # Start with a known feasible pattern: grid-like
    # Pattern 0: all x-separations (forces a horizontal line, likely infeasible for large squares)
    # Better: use a 3x3 grid pattern. But let's just enumerate smartly.

    # We'll do a beam search: start with a random pattern, then try flipping one pair at a time.
    # But first, let's try a deterministic set of patterns.

    best_value = 0.0
    best_centers = None
    best_sides = None

    # Initial patterns to try
    patterns_to_try = []

    # Pattern A: "grid" - first 3 pairs x-sep, rest y-sep (arbitrary but let's be smarter)
    # Actually, let's use a heuristic: for a k x k grid, assign x-sep for same-row, y-sep for same-col.
    # But we don't have rows/cols yet. Let's just try all-x, all-y, and a few mixed.

    # All x
    pat_all_x = {p: 0 for p in pairs}
    patterns_to_try.append(pat_all_x)

    # All y
    pat_all_y = {p: 1 for p in pairs}
    patterns_to_try.append(pat_all_y)

    # A "cross" pattern: pairs (0,1),(0,2),(1,2) x-sep; rest y-sep
    # This is getting complicated. Let's just do a local search.

    # Start with a random pattern
    import random
    random.seed(42)

    # Initial pattern: random
    pat = {p: random.randint(0, 1) for p in pairs}

    # Try to solve; if infeasible, try flipping pairs
    # We'll do a simple local search: try flipping each pair, keep if feasible and improves.

    # But first, let's just try a few random patterns and the two extremes.

    for pat in patterns_to_try:
        if time.time() > deadline:
            break
        res = _solve_pattern(pat)
        if res is not None:
            val, c, s = res
            if val > best_value:
                best_value = val
                best_centers = c
                best_sides = s

    # Now do local search from a random feasible pattern
    # First, find a feasible pattern by random restarts
    for restart in range(200):
        if time.time() > deadline:
            break
        pat = {p: random.randint(0, 1) for p in pairs}
        res = _solve_pattern(pat)
        if res is not None:
            val, c, s = res
            if val > best_value:
                best_value = val
                best_centers = c
                best_sides = s
                # Local search: try flipping each pair
                for k in range(num_pairs):
                    if time.time() > deadline:
                        break
                    p = pairs[k]
                    new_pat = dict(pat)
                    new_pat[p] = 1 - new_pat[p]
                    res2 = _solve_pattern(new_pat)
                    if res2 is not None:
                        val2, c2, s2 = res2
                        if val2 > best_value:
                            best_value = val2
                            best_centers = c2
                            best_sides = s2
                            pat = new_pat
                            break  # restart local search from new best

    # If we never found a feasible pattern, return the baseline grid
    if best_centers is None:
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(n, side, dtype=np.float64)

    return best_centers, best_sides


# EVOLVE-BLOCK-END
