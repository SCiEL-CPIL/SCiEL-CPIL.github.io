"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26
import time
_start_time = time.time()
_DEADLINE = 150  # seconds, must finish before 120s eval limit with margin

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint

def _valid(arr):
    """Check if arrangement is valid: all circles inside unit square, disjoint interiors."""
    if arr is None or len(arr) != N:
        return False
    for i in range(N):
        x, y, r = arr[i]
        if r < 0 or x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
    for i in range(N):
        for j in range(i+1, N):
            xi, yi, ri = arr[i]
            xj, yj, rj = arr[j]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist < ri + rj - 1e-7:
                return False
    return True

def _score(arr):
    """Return sum of radii if valid, else -inf."""
    if not _valid(arr):
        return -np.inf
    return np.sum(arr[:, 2])

def _objective(x):
    """Negative sum of radii for minimization."""
    return -np.sum(x[2*N:])

def _constraints(x):
    """Return list of constraint dicts for SLSQP."""
    pos = x[:2*N]
    rad = x[2*N:]
    cons = []
    # Circle i center (xi, yi) radius ri
    for i in range(N):
        xi = pos[2*i]
        yi = pos[2*i+1]
        ri = rad[i]
        # Inside unit square: xi - ri >= 0, xi + ri <= 1, yi - ri >= 0, yi + ri <= 1
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2*i] - x[2*N+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2*i+1] - x[2*N+i]})
        # Radii positive
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*N+i]})
    # Disjoint interiors: dist(i,j) >= ri + rj
    for i in range(N):
        for j in range(i+1, N):
            xi, yi, ri = 2*i, 2*i+1, 2*N+i
            xj, yj, rj = 2*j, 2*j+1, 2*N+j
            def disjoint(x, i=i, j=j, xi=xi, yi=yi, ri=ri, xj=xj, yj=yj, rj=rj):
                dx = x[xi] - x[xj]
                dy = x[yi] - x[yj]
                return np.sqrt(dx*dx + dy*dy) - x[ri] - x[rj]
            cons.append({'type': 'ineq', 'fun': disjoint})
    return cons

def _bounds():
    """Bounds for SLSQP: positions in [0,1], radii in [0, 0.5]."""
    b = []
    for i in range(N):
        b.append((0, 1))  # x
        b.append((0, 1))  # y
    for i in range(N):
        b.append((0, 0.5))  # radius
    return b

def _initial_grid():
    """Create initial grid arrangement."""
    side = int(np.ceil(np.sqrt(N)))
    r0 = 0.5 / side
    arr = np.zeros((N, 3))
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N:
                break
            arr[count] = [(i + 0.5) / side, (j + 0.5) / side, r0 * 0.9]
            count += 1
    return arr

def _initial_hexagonal():
    """Create initial hexagonal packing arrangement."""
    # Hexagonal packing: rows offset
    # For 26 circles, try to fit in a hexagonal pattern
    # Row 0: circles at y=0.5/r, etc.
    # Let's use a simpler approach: pack rows with hexagonal offset
    n_rows = 5
    # Distribute 26 circles over 5 rows: 6,5,6,5,4 or similar
    row_counts = [6, 5, 6, 5, 4]
    arr = np.zeros((N, 3))
    count = 0
    # Estimate radius for hexagonal packing
    # In hexagonal packing, vertical spacing = r * sqrt(3), horizontal = 2r
    # For 6 circles in a row: width = 5*2r + 2r = 12r <= 1 => r <= 1/12
    # For 5 rows with hex offset: height = 4*r*sqrt(3) + 2r = r*(4*sqrt(3)+2)
    # Total height <= 1 => r <= 1/(4*sqrt(3)+2)
    r_est = 1.0 / (4 * np.sqrt(3) + 2)
    r_est = min(r_est, 1.0/12) * 0.8  # conservative initial
    for row_idx, n_circ in enumerate(row_counts):
        y = (row_idx + 0.5) / n_rows
        # Horizontal positions with hex offset
        if row_idx % 2 == 0:
            xs = np.linspace(r_est, 1 - r_est, n_circ)
        else:
            # Offset by half spacing
            spacing = 2 * r_est
            xs = np.linspace(r_est + spacing/2, 1 - r_est - spacing/2, n_circ)
            # Adjust to fit
            if n_circ > 1:
                xs = np.linspace(r_est, 1 - r_est, n_circ)
                xs += (spacing / 2) * (1 - 2 * (xs - r_est) / (1 - 2*r_est))
                xs = np.clip(xs, r_est, 1 - r_est)
        for k in range(n_circ):
            if count >= N:
                break
            arr[count] = [xs[k], y, r_est]
            count += 1
    return arr

def _optimize(arr, max_iter=2000, tol=1e-6):
    """Run SLSQP optimization on the arrangement."""
    x0 = np.concatenate([arr[:, 0], arr[:, 1], arr[:, 2]])
    cons = _constraints(x0)
    bounds = _bounds()
    try:
        res = minimize(_objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': max_iter, 'ftol': tol})
        if res.success or res.fun < -np.sum(arr[:, 2]):
            x_opt = res.x
            new_arr = np.column_stack([x_opt[:2*N:2], x_opt[1:2*N:2], x_opt[2*N:]])
            if _valid(new_arr):
                return new_arr
    except Exception:
        pass
    return arr

def _radius_growth(arr, max_iter=500, tol=1e-8):
    """Fix positions, maximize radii only."""
    x0 = np.concatenate([arr[:, 0], arr[:, 1], arr[:, 2]])
    # Only optimize radii, fix positions
    # Use SLSQP with positions fixed
    def obj_rad(x):
        return -np.sum(x[2*N:])
    
    # Constraints with fixed positions
    pos_fixed = x0[:2*N]
    cons = []
    for i in range(N):
        xi = pos_fixed[2*i]
        yi = pos_fixed[2*i+1]
        # Inside unit square
        cons.append({'type': 'ineq', 'fun': lambda x, i=i, xi=xi: xi - x[2*N+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i, xi=xi: 1 - xi - x[2*N+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i, yi=yi: yi - x[2*N+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i, yi=yi: 1 - yi - x[2*N+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*N+i]})
    for i in range(N):
        for j in range(i+1, N):
            xi, yi = pos_fixed[2*i], pos_fixed[2*i+1]
            xj, yj = pos_fixed[2*j], pos_fixed[2*j+1]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            def disjoint_rad(x, i=i, j=j, dist=dist):
                return dist - x[2*N+i] - x[2*N+j]
            cons.append({'type': 'ineq', 'fun': disjoint_rad})
    
    bounds = [(0, 1)] * (2*N) + [(0, 0.5)] * N
    # Fix positions by setting bounds to exact values
    bounds_fixed = []
    for i in range(N):
        bounds_fixed.append((x0[2*i], x0[2*i]))
        bounds_fixed.append((x0[2*i+1], x0[2*i+1]))
    for i in range(N):
        bounds_fixed.append((0, 0.5))
    
    try:
        res = minimize(obj_rad, x0, method='SLSQP', bounds=bounds_fixed, constraints=cons,
                       options={'maxiter': max_iter, 'ftol': tol})
        if res.success:
            x_opt = res.x
            new_arr = np.column_stack([x_opt[:2*N:2], x_opt[1:2*N:2], x_opt[2*N:]])
            if _valid(new_arr) and np.sum(new_arr[:, 2]) >= np.sum(arr[:, 2]) - 1e-6:
                return new_arr
    except Exception:
        pass
    return arr

def _polish(arr, max_iter=3000, tol=1e-8):
    """Final high-precision polish."""
    x0 = np.concatenate([arr[:, 0], arr[:, 1], arr[:, 2]])
    cons = _constraints(x0)
    bounds = _bounds()
    try:
        res = minimize(_objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': max_iter, 'ftol': tol})
        if res.success:
            x_opt = res.x
            new_arr = np.column_stack([x_opt[:2*N:2], x_opt[1:2*N:2], x_opt[2*N:]])
            if _valid(new_arr):
                return new_arr
    except Exception:
        pass
    return arr

def construct():
    global _start_time, _DEADLINE
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5:
        # Return trivial grid
        side = int(np.ceil(np.sqrt(N)))
        r0 = 0.5 / side
        cells = [((i + 0.5) / side, (j + 0.5) / side, r0 * 0.9) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Start with hexagonal initial guess
    arr = _initial_hexagonal()
    score = _score(arr)
    
    # Main optimization
    time_left = _DEADLINE - (time.time() - _start_time) - 10
    if time_left > 5:
        arr = _optimize(arr, max_iter=1500, tol=1e-7)
        score = _score(arr)
    
    # Radius growth pass
    time_left = _DEADLINE - (time.time() - _start_time) - 8
    if time_left > 5:
        arr = _radius_growth(arr, max_iter=400, tol=1e-8)
        score = _score(arr)
    
    # Another optimization round
    time_left = _DEADLINE - (time.time() - _start_time) - 6
    if time_left > 5:
        arr = _optimize(arr, max_iter=1000, tol=1e-7)
        score = _score(arr)
    
    # Final radius growth
    time_left = _DEADLINE - (time.time() - _start_time) - 5
    if time_left > 3:
        arr = _radius_growth(arr, max_iter=300, tol=1e-9)
        score = _score(arr)
    
    # Final polish
    time_left = _DEADLINE - (time.time() - _start_time) - 3
    if time_left > 3:
        arr = _polish(arr, max_iter=2000, tol=1e-9)
        score = _score(arr)
    
    if not _valid(arr):
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        r0 = 0.5 / side
        cells = [((i + 0.5) / side, (j + 0.5) / side, r0 * 0.9) for i in range(side) for j in range(side)]
        arr = np.array(cells[:N])
    
    return arr
# EVOLVE-BLOCK-END
