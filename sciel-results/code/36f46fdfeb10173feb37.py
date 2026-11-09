# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

TOL = 1e-7
DEADLINE_MARGIN = 10.0  # seconds before 630s limit

def _is_valid(centers, sides, n=10):
    """Check if a packing is valid."""
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Check inside [-tol, 1+tol]^2
    for i in range(n):
        x0 = centers[i, 0] - sides[i] / 2
        x1 = centers[i, 0] + sides[i] / 2
        y0 = centers[i, 1] - sides[i] / 2
        y1 = centers[i, 1] + sides[i] / 2
        if x0 < -TOL - 1e-10 or x1 > 1 + TOL + 1e-10:
            return False
        if y0 < -TOL - 1e-10 or y1 > 1 + TOL + 1e-10:
            return False
    # Check non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - TOL
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True

def _solve_lp_for_separations(n, sep_x, sep_y, time_limit=5.0):
    """
    Solve LP to maximize sum of sides given fixed separation pattern.
    sep_x[i,j] = 1 if square i is to the left of j (x_i + s_i/2 <= x_j - s_j/2),
                = -1 if right, = 0 if not separated in x.
    sep_y similarly for y.
    """
    # Variables: cx_i, cy_i, s_i for i in 0..n-1
    # Total vars: 3n
    # Maximize sum(s_i) -> minimize -sum(s_i)
    
    c = np.zeros(3 * n)
    for i in range(n):
        c[2 * n + i] = -1.0  # maximize s_i
    
    A_ub = []
    b_ub = []
    
    # Inside unit square constraints:
    # cx_i - s_i/2 >= 0  =>  -cx_i + s_i/2 <= 0
    # cx_i + s_i/2 <= 1  =>   cx_i - s_i/2 <= 1
    # cy_i - s_i/2 >= 0  =>  -cy_i + s_i/2 <= 0
    # cy_i + s_i/2 <= 1  =>   cy_i - s_i/2 <= 1
    for i in range(n):
        # -cx_i + s_i/2 <= 0
        row = np.zeros(3 * n)
        row[i] = -1.0
        row[2 * n + i] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)
        
        # cx_i - s_i/2 <= 1
        row = np.zeros(3 * n)
        row[i] = 1.0
        row[2 * n + i] = -0.5
        A_ub.append(row)
        b_ub.append(1.0)
        
        # -cy_i + s_i/2 <= 0
        row = np.zeros(3 * n)
        row[n + i] = -1.0
        row[2 * n + i] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)
        
        # cy_i - s_i/2 <= 1
        row = np.zeros(3 * n)
        row[n + i] = 1.0
        row[2 * n + i] = -0.5
        A_ub.append(row)
        b_ub.append(1.0)
    
    # Separation constraints:
    # If sep_x[i,j] = 1: cx_i + s_i/2 <= cx_j - s_j/2  =>  cx_i - cx_j + s_i/2 + s_j/2 <= 0
    # If sep_x[i,j] = -1: cx_j + s_j/2 <= cx_i - s_i/2  =>  -cx_i + cx_j + s_i/2 + s_j/2 <= 0
    # Similarly for y.
    
    for i in range(n):
        for j in range(i + 1, n):
            if sep_x[i, j] == 1:
                row = np.zeros(3 * n)
                row[i] = 1.0
                row[j] = -1.0
                row[2 * n + i] = 0.5
                row[2 * n + j] = 0.5
                A_ub.append(row)
                b_ub.append(0.0)
            elif sep_x[i, j] == -1:
                row = np.zeros(3 * n)
                row[i] = -1.0
                row[j] = 1.0
                row[2 * n + i] = 0.5
                row[2 * n + j] = 0.5
                A_ub.append(row)
                b_ub.append(0.0)
            
            if sep_y[i, j] == 1:
                row = np.zeros(3 * n)
                row[n + i] = 1.0
                row[n + j] = -1.0
                row[2 * n + i] = 0.5
                row[2 * n + j] = 0.5
                A_ub.append(row)
                b_ub.append(0.0)
            elif sep_y[i, j] == -1:
                row = np.zeros(3 * n)
                row[n + i] = -1.0
                row[n + j] = 1.0
                row[2 * n + i] = 0.5
                row[2 * n + j] = 0.5
                A_ub.append(row)
                b_ub.append(0.0)
    
    # Lower bounds: s_i >= 0
    bounds = [(0, None)] * (3 * n)
    # cx, cy can be any real, but effectively bounded by constraints
    for i in range(n):
        bounds[i] = (None, None)
        bounds[n + i] = (None, None)
        bounds[2 * n + i] = (1e-10, None)  # s_i > 0
    
    if A_ub:
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        cx = result.x[:n]
        cy = result.x[n:2*n]
        s = result.x[2*n:]
        centers = np.stack([cx, cy], axis=1)
        return centers, s, -result.fun
    return None

def _generate_separation_patterns(n, max_patterns=2000):
    """
    Generate random separation patterns and solve LPs.
    Returns list of (value, centers, sides).
    """
    results = []
    np.random.seed(42)
    
    # Start with some structured patterns
    # Pattern 1: grid-like
    for attempt in range(max_patterns):
        sep_x = np.zeros((n, n), dtype=int)
        sep_y = np.zeros((n, n), dtype=int)
        
        # Random assignment: for each pair, choose separation direction
        # with some probability of no separation in x (meaning must separate in y)
        for i in range(n):
            for j in range(i + 1, n):
                # Decide if separated in x or y
                if np.random.random() < 0.5:
                    # Separate in x
                    if np.random.random() < 0.5:
                        sep_x[i, j] = 1
                        sep_x[j, i] = -1
                    else:
                        sep_x[i, j] = -1
                        sep_x[j, i] = 1
                else:
                    # Separate in y
                    if np.random.random() < 0.5:
                        sep_y[i, j] = 1
                        sep_y[j, i] = -1
                    else:
                        sep_y[i, j] = -1
                        sep_y[j, i] = 1
        
        result = _solve_lp_for_separations(n, sep_x, sep_y, time_limit=3.0)
        if result is not None:
            centers, s, val = result
            if _is_valid(centers, s, n):
                results.append((val, centers, s))
    
    return results

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 630 - DEADLINE_MARGIN
    
    # Baseline grid packing
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
    best_sides = np.full(n, side, dtype=np.float64)
    best_val = np.sum(best_sides)
    
    # Try LP-based optimization with various separation patterns
    results = _generate_separation_patterns(n, max_patterns=500)
    
    if results:
        results.sort(key=lambda x: -x[0])
        for val, centers, sides in results:
            if val > best_val:
                best_val = val
                best_centers = centers
                best_sides = sides
    
    # Check deadline
    if time.time() > deadline:
        return best_centers, best_sides
    
    # Try more patterns if time allows
    remaining = deadline - time.time()
    if remaining > 30:
        additional = _generate_separation_patterns(n, max_patterns=int(remaining / 0.5))
        if additional:
            additional.sort(key=lambda x: -x[0])
            for val, centers, sides in additional:
                if val > best_val:
                    best_val = val
                    best_centers = centers
                    best_sides = sides
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
