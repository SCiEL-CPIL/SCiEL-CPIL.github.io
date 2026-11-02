import numpy as np
from scipy.optimize import minimize, linprog
import time

N = 32
TIME_LIMIT = 115.0 # Safety margin

def _elapsed(start):
    return time.time() - start

def _solve_lp(x, y):
    """
    Solve LP to maximize sum of radii for fixed centers.
    Variables: r[0..N-1]
    Objective: min -sum(r)
    Constraints:
    1. r[i] + r[j] <= dist(i,j)
    2. r[i] <= x[i], 1-x[i], y[i], 1-y[i]
    3. r[i] >= 0
    """
    # Objective
    c = -np.ones(N)
    
    # Inequality constraints A_ub * r <= b_ub
    A_rows = []
    b_rows = []
    
    # Pairwise constraints
    # dist(i, j) - r[i] - r[j] >= 0  =>  r[i] + r[j] <= dist(i, j)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            A_rows.append(row)
            b_rows.append(d)
            
    # Boundary constraints
    # r[i] <= x[i]
    # r[i] <= 1 - x[i]
    # r[i] <= y[i]
    # r[i] <= 1 - y[i]
    for i in range(N):
        for bound in [x[i], 1.0 - x[i], y[i], 1.0 - y[i]]:
            row = np.zeros(N)
            row[i] = 1.0
            A_rows.append(row)
            b_rows.append(bound)
            
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds
    bounds = [(0, None)] * N
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return -res.fun, res.x
        else:
            # Fallback: small radii
            r = np.full(N, 0.01)
            return np.sum(r), r
    except Exception:
        r = np.full(N, 0.01)
        return np.sum(r), r

def _fitness_pos(params):
    """
    Objective function for position optimization.
    params: [x_0..x_N-1, y_0..y_N-1]
    Returns: -sum(radii)
    """
    x = params[:N]
    y = params[N:2*N]
    
    # Basic validity check to avoid LP failures or trivial solutions
    # Ensure centers are within [0, 1]
    if np.any(x < -0.1) or np.any(x > 1.1) or np.any(y < -0.1) or np.any(y > 1.1):
        return 1e10
        
    sum_r, r = _solve_lp(x, y)
    return -sum_r

def _optimize_positions(x, y, start, max_time=10.0):
    """
    Optimize positions using Nelder-Mead to maximize LP radius sum.
    """
    p0 = np.concatenate([x, y])
    
    # Check time
    if _elapsed(start) > TIME_LIMIT - 2:
        return x, y, _solve_lp(x, y)[1]
    
    try:
        # Nelder-Mead is robust for non-smooth objectives
        res = minimize(
            _fitness_pos, 
            p0, 
            method='Nelder-Mead',
            options={
                'maxiter': 2000,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True,
                'disp': False
            }
        )
        
        if _elapsed(start) > TIME_LIMIT - 1:
             # Interrupted, use best found so far? minimize doesn't support interrupt easily.
             # We rely on the time check in construct loop or just let it run if it's fast enough.
             # Nelder-Mead with 64 dims might be slow. 
             # Let's add a time check inside the callback if possible, or just limit maxiter.
             pass

        x_new = res.x[:N]
        y_new = res.x[N:2*N]
        
        # Get final radii
        sum_r, r_new = _solve_lp(x_new, y_new)
        
        return x_new, y_new, r_new

    except Exception as e:
        # Fallback
        sum_r, r = _solve_lp(x, y)
        return x, y, r

def _hex_init():
    pts = []
    rowh = np.sqrt(3.0)/2.0
    r = 0
    while len(pts) < N:
        for c in range(0, 12):
            x = c + (0.5 if (r % 2) else 0.0)
            y = r * rowh
            pts.append([x, y])
            if len(pts) >= N:
                break
        r += 1
    pts = np.array(pts[:N])
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span = max(xmax - xmin, ymax - ymin)
    if span < 1e-9:
        span = 1.0
    scale = 0.85 / span
    x = (pts[:,0] - xmin) * scale + 0.075
    y = (pts[:,1] - ymin) * scale + 0.075
    return x, y

def _rect_init(cols, rows):
    cells = []
    for j in range(rows):
        for i in range(cols):
            if len(cells) < N:
                x = (i + 0.5) / cols
                y = (j + 0.5) / rows
                cells.append([x, y])
    x = np.array([c[0] for c in cells])
    y = np.array([c[1] for c in cells])
    return x, y

def _random_init(seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    return x, y

def construct():
    start = time.time()
    
    inits = [
        _hex_init(),
        _rect_init(6, 6),
        _rect_init(7, 5),
        _rect_init(5, 7),
        _rect_init(8, 4),
        _rect_init(4, 8),
        _random_init(1),
        _random_init(42)
    ]
    
    best_score = -1e10
    best_x, best_y, best_r = None, None, None
    
    for x, y in inits:
        if _elapsed(start) > TIME_LIMIT - 10:
            break
            
        # Optimize positions
        x_opt, y_opt, r_opt = _optimize_positions(x, y, start)
        
        # Evaluate
        score, r_final = _solve_lp(x_opt, y_opt)
        
        if score > best_score:
            best_score = score
            best_x, best_y, best_r = x_opt, y_opt, r_final
            
    # Final Polish: Run another optimization on the best result found
    if best_x is not None and _elapsed(start) < TIME_LIMIT - 5:
        x_pol, y_pol, r_pol = _optimize_positions(best_x, best_y, start)
        score_pol, r_pol = _solve_lp(x_pol, y_pol)
        if score_pol > best_score:
            best_score = score_pol
            best_x, best_y, best_r = x_pol, y_pol, r_pol

    # Ensure validity (LP ensures this, but just in case of numerical issues)
    # The LP solution is guaranteed to satisfy constraints within solver tolerance.
    
    result = np.column_stack((best_x, best_y, best_r))
    return result
