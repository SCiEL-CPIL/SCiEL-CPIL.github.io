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
    def s_idx(i): return i
    def cx_idx(i): return n + i
    def cy_idx(i): return 2 * n + i

    c = np.zeros(3 * n)
    for i in range(n):
        c[s_idx(i)] = -1.0

    A_ub = []
    b_ub = []

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

    for (i, j), axis in pattern.items():
        if axis == 0:
            row = np.zeros(3 * n)
            row[cx_idx(j)] = 1.0
            row[cx_idx(i)] = -1.0
            row[s_idx(i)] = 0.5
            row[s_idx(j)] = 0.5
            A_ub.append(-row)
            b_ub.append(0.0)
        else:
            row = np.zeros(3 * n)
            row[cy_idx(j)] = 1.0
            row[cy_idx(i)] = -1.0
            row[s_idx(i)] = 0.5
            row[s_idx(j)] = 0.5
            A_ub.append(-row)
            b_ub.append(0.0)

    A_ub = np.array(A_ub, dtype=np.float64)
    b_ub = np.array(b_ub, dtype=np.float64)

    bounds = []
    for i in range(n):
        bounds.append((1e-9, 1.0))
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, 1.0))

    return c, A_ub, b_ub, bounds


def _solve_pattern(pattern):
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
    if not np.all(np.isfinite(sides)) or not np.all(sides > 0):
        return None
    for i in range(n):
        if cx[i] - sides[i]/2 < -_TOL or cx[i] + sides[i]/2 > 1 + _TOL:
            return None
        if cy[i] - sides[i]/2 < -_TOL or cy[i] + sides[i]/2 > 1 + _TOL:
            return None
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - _TOL
            if abs(cx[i]-cx[j]) < t and abs(cy[i]-cy[j]) < t:
                return None
    value = float(np.sum(sides))
    centers = np.column_stack([cx, cy]).astype(np.float64)
    sides = sides.astype(np.float64)
    return value, centers, sides


def _generate_structured_patterns(n=10):
    """Generate structured patterns based on grid layouts and row/column assignments."""
    patterns = []
    pairs = list(combinations(range(n), 2))
    
    # Grid patterns: try different grid dimensions
    for rows in range(1, 5):
        for cols in range(1, 5):
            if rows * cols >= n:
                # Assign each square to a grid cell
                grid_assign = {}
                for i in range(n):
                    r = i // cols
                    c = i % cols
                    grid_assign[i] = (r, c)
                
                pattern = {}
                for (i, j) in pairs:
                    ri, ci = grid_assign[i]
                    rj, cj = grid_assign[j]
                    if ri == rj:
                        pattern[(i, j)] = 0  # same row, x-separation
                    elif ci == cj:
                        pattern[(i, j)] = 1  # same column, y-separation
                    else:
                        # Different row and column: choose based on relative position
                        # If j is to the right and below, use x-sep if horizontal distance larger
                        dx = abs(ci - cj)
                        dy = abs(ri - rj)
                        pattern[(i, j)] = 0 if dx >= dy else 1
                
                patterns.append(pattern)
    
    # Random structured patterns with bias
    import random
    random.seed(123)
    for _ in range(50):
        pattern = {}
        for (i, j) in pairs:
            # Bias towards certain patterns
            if random.random() < 0.3:
                pattern[(i, j)] = 0
            elif random.random() < 0.6:
                pattern[(i, j)] = 1
            else:
                pattern[(i, j)] = random.randint(0, 1)
        patterns.append(pattern)
    
    return patterns


def squares_in_square10() -> tuple:
    n = 10
    t0 = time.time()
    deadline = t0 + _DEADLINE - 30.0

    pairs = list(combinations(range(n), 2))
    num_pairs = len(pairs)

    best_value = 0.0
    best_centers = None
    best_sides = None
    best_pattern = None

    # Try structured patterns first
    structured_patterns = _generate_structured_patterns(n)
    for pat in structured_patterns:
        if time.time() > deadline:
            break
        res = _solve_pattern(pat)
        if res is not None:
            val, c, s = res
            if val > best_value:
                best_value = val
                best_centers = c
                best_sides = s
                best_pattern = pat

    # Simulated annealing-style pattern search
    import random
    random.seed(42)
    
    # Start from best found or a random pattern
    if best_pattern is not None:
        current_pattern = best_pattern.copy()
        current_value = best_value
    else:
        current_pattern = {p: random.randint(0, 1) for p in pairs}
        res = _solve_pattern(current_pattern)
        if res is not None:
            current_value, best_centers, best_sides = res
            best_value = current_value
            best_pattern = current_pattern
        else:
            # Fallback to grid
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

    # Simulated annealing
    temperature = 0.5
    cooling_rate = 0.995
    min_temperature = 0.01
    max_iterations = 5000

    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        if temperature < min_temperature:
            break

        # Randomly flip 1-3 pairs
        num_flips = random.randint(1, 3)
        new_pattern = current_pattern.copy()
        for _ in range(num_flips):
            p = random.choice(pairs)
            new_pattern[p] = 1 - new_pattern[p]

        res = _solve_pattern(new_pattern)
        if res is not None:
            new_value, _, _ = res
            delta = new_value - current_value
            if delta > 0 or random.random() < np.exp(delta / temperature):
                current_pattern = new_pattern
                current_value = new_value
                if new_value > best_value:
                    best_value = new_value
                    best_centers = res[1]
                    best_sides = res[2]
                    best_pattern = new_pattern
        temperature *= cooling_rate

    # Final polish: try single-flip improvements from best pattern
    if best_pattern is not None:
        polish_pattern = best_pattern.copy()
        polish_value = best_value
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for p in pairs:
                if time.time() > deadline:
                    break
                new_pat = polish_pattern.copy()
                new_pat[p] = 1 - new_pat[p]
                res = _solve_pattern(new_pat)
                if res is not None:
                    val, c, s = res
                    if val > polish_value:
                        polish_value = val
                        polish_pattern = new_pat
                        improved = True
                        if val > best_value:
                            best_value = val
                            best_centers = c
                            best_sides = s
                            best_pattern = new_pat

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
