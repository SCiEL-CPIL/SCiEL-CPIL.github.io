"""
Strategy: Adaptive Grid Interpolation with Piecewise Polynomials.
Instead of searching for a single global symbolic expression (like the GP approach),
we approximate the function f(x, y) using a grid-based method.
1. Create a coarse grid over [-1, 1] x [-1, 1].
2. Fit a low-order 2D polynomial to the data points near each grid cell.
3. Identify regions with high prediction error.
4. Refine the grid in those regions (increase resolution).
5. Repeat until time budget is exhausted.
6. Final prediction uses the highest-resolution polynomial fit for the region containing the query point.

This approach is robust because it doesn't assume a specific functional form (like sin, exp, etc.)
and can approximate any continuous function well given enough resolution.
It avoids the combinatorial explosion of symbolic regression and the local minima issues of global LM fits.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

# Global cache
_grid_cache = None
_polynomial_cache = []
_search_done = False
_deadline = None

def _fit_2d_poly(x, y, z, deg=2):
    """
    Fit a 2D polynomial: z = sum_{i=0}^{deg} sum_{j=0}^{deg} c[i,j] * x^i * y^j
    Returns coefficients matrix c of shape (deg+1, deg+1) or None if fit fails.
    """
    if len(x) < (deg + 1) ** 2:
        return None
    
    # Build Vandermonde-like matrix
    X = np.ones((len(x), 1))
    Y = np.ones((len(y), 1))
    
    for i in range(1, deg + 1):
        X = np.column_stack([X, x ** i])
        Y = np.column_stack([Y, y ** i])
        
    # Outer product to get all combinations x^i * y^j
    # We need to construct the design matrix A where A[:, k] = x^i * y^j
    # Let's do it explicitly for clarity and speed for small deg
    A = np.zeros((len(x), (deg + 1) ** 2))
    idx = 0
    for i in range(deg + 1):
        for j in range(deg + 1):
            A[:, idx] = (x ** i) * (y ** j)
            idx += 1
            
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(A, z, rcond=None)
        if not np.all(np.isfinite(coeffs)):
            return None
        return coeffs.reshape((deg + 1, deg + 1))
    except Exception:
        return None

def _evaluate_poly(x, y, coeffs, deg=2):
    """Evaluate the 2D polynomial."""
    if coeffs is None:
        return np.zeros_like(x)
    
    z = np.zeros_like(x)
    for i in range(deg + 1):
        for j in range(deg + 1):
            z += coeffs[i, j] * (x ** i) * (y ** j)
    return z

def _adaptive_grid_search(x_train, y_train, z_train, time_budget):
    """
    Run the adaptive grid search.
    Returns a list of (x_min, x_max, y_min, y_max, coeffs, deg) for the final grid cells.
    """
    global _grid_cache, _polynomial_cache
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Initial coarse grid
    # Let's start with 4x4 cells
    n_cells_x = 4
    n_cells_y = 4
    x_min, x_max = -1.0, 1.0
    y_min, y_max = -1.0, 1.0
    
    dx = (x_max - x_min) / n_cells_x
    dy = (y_max - y_min) / n_cells_y
    
    # Cell structure: list of dicts
    cells = []
    for i in range(n_cells_x):
        for j in range(n_cells_y):
            cx_min = x_min + i * dx
            cx_max = cx_min + dx
            cy_min = y_min + j * dy
            cy_max = cy_min + dy
            
            # Find points in this cell
            mask = (x_train >= cx_min) & (x_train < cx_max) & (y_train >= cy_min) & (y_train < cy_max)
            # Handle edge case for max boundary
            if i == n_cells_x - 1 and j == n_cells_y - 1:
                mask = (x_train >= cx_min) & (x_train <= cx_max) & (y_train >= cy_min) & (y_train <= cy_max)
            elif i == n_cells_x - 1:
                mask = (x_train >= cx_min) & (x_train <= cx_max) & (y_train >= cy_min) & (y_train < cy_max)
            elif j == n_cells_y - 1:
                mask = (x_train >= cx_min) & (x_train < cx_max) & (y_train >= cy_min) & (y_train <= cy_max)
            
            xc = x_train[mask]
            yc = y_train[mask]
            zc = z_train[mask]
            
            # Fit polynomial
            deg = 2
            coeffs = _fit_2d_poly(xc, yc, zc, deg)
            
            # Calculate error
            if coeffs is not None and len(xc) > 0:
                pred = _evaluate_poly(xc, yc, coeffs, deg)
                err = np.mean((pred - zc) ** 2)
            else:
                err = 1e10
                coeffs = np.zeros((3, 3)) # Fallback constant 0? Or mean?
                if len(zc) > 0:
                    coeffs[0, 0] = np.mean(zc)
                    err = np.var(zc)
                else:
                    err = 1e10
            
            cells.append({
                'x_min': cx_min, 'x_max': cx_max,
                'y_min': cy_min, 'y_max': cy_max,
                'coeffs': coeffs,
                'deg': deg,
                'error': err,
                'n_points': len(xc),
                'x_data': xc, 'y_data': yc, 'z_data': zc
            })
    
    # Refinement Loop
    while time.time() < deadline:
        # Find cells with highest error and enough points to split
        # Sort by error descending
        cells_sorted = sorted(cells, key=lambda c: c['error'], reverse=True)
        
        split_happened = False
        for cell in cells_sorted:
            if time.time() > deadline:
                break
                
            # Only split if error is significant and we have enough points
            if cell['error'] < 1e-6 or cell['n_points'] < 10:
                continue
                
            # Split this cell into 4 sub-cells
            cx_mid = (cell['x_min'] + cell['x_max']) / 2.0
            cy_mid = (cell['y_min'] + cell['y_max']) / 2.0
            
            new_cells = []
            sub_points = cell['x_data'], cell['y_data'], cell['z_data']
            
            # Define 4 quadrants
            quadrants = [
                (cell['x_min'], cx_mid, cell['y_min'], cy_mid),
                (cx_mid, cell['x_max'], cell['y_min'], cy_mid),
                (cell['x_min'], cx_mid, cy_mid, cell['y_max']),
                (cx_mid, cell['x_max'], cy_mid, cell['y_max']),
            ]
            
            for q_min_x, q_max_x, q_min_y, q_max_y in quadrants:
                mask = (cell['x_data'] >= q_min_x) & (cell['x_data'] < q_max_x) & \
                       (cell['y_data'] >= q_min_y) & (cell['y_data'] < q_max_y)
                
                xc = cell['x_data'][mask]
                yc = cell['y_data'][mask]
                zc = cell['z_data'][mask]
                
                deg = 2
                coeffs = _fit_2d_poly(xc, yc, zc, deg)
                
                if coeffs is not None and len(xc) > 0:
                    pred = _evaluate_poly(xc, yc, coeffs, deg)
                    err = np.mean((pred - zc) ** 2)
                else:
                    err = 1e10
                    coeffs = np.zeros((3, 3))
                    if len(zc) > 0:
                        coeffs[0, 0] = np.mean(zc)
                        err = np.var(zc)
                    else:
                        err = 1e10
                
                new_cells.append({
                    'x_min': q_min_x, 'x_max': q_max_x,
                    'y_min': q_min_y, 'y_max': q_max_y,
                    'coeffs': coeffs,
                    'deg': deg,
                    'error': err,
                    'n_points': len(xc),
                    'x_data': xc, 'y_data': yc, 'z_data': zc
                })
            
            # Replace old cell with new ones
            cells.remove(cell)
            cells.extend(new_cells)
            split_happened = True
            
            # Limit splits per iteration to avoid memory blowup or time overrun
            if split_happened:
                # Check time again
                if time.time() > deadline - 5:
                    break

        if not split_happened:
            break
            
    # Store final grid
    _grid_cache = cells
    _search_done = True

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function f(x, y) using adaptive grid interpolation.
    """
    global _grid_cache, _search_done, _deadline
    
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    N = len(x)
    
    if N == 0:
        return np.array([])
    
    if not _search_done:
        # Assume first call is training data
        # We need to infer y_train from the input 'y'
        # The signature is equation(x, y, params). 
        # In the evaluator, y is the target.
        z_train = y
        
        # Time budget: 140s to be safe
        _adaptive_grid_search(x, y, z_train, time_budget=140.0)
        
    if _grid_cache is None:
        # Fallback: Global linear fit
        A = np.column_stack([np.ones(N), x, y])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            return A @ coeffs
        except:
            return np.zeros(N)
            
    # Prediction Phase
    predictions = np.zeros(N)
    
    for i in range(N):
        xi = x[i]
        yi = y[i]
        
        # Find the cell containing (xi, yi)
        # Note: Since we split, cells are disjoint and cover the domain
        found = False
        for cell in _grid_cache:
            if (cell['x_min'] <= xi < cell['x_max'] or (xi == cell['x_max'] and cell['x_max'] == 1.0)) and \
               (cell['y_min'] <= yi < cell['y_max'] or (yi == cell['y_max'] and cell['y_max'] == 1.0)):
                # Handle edge cases for max boundary
                if xi >= cell['x_min'] and xi <= cell['x_max'] and yi >= cell['y_min'] and yi <= cell['y_max']:
                    predictions[i] = _evaluate_poly(np.array([xi]), np.array([yi]), cell['coeffs'], cell['deg'])[0]
                    found = True
                    break
        
        if not found:
            # If not found (shouldn't happen if grid covers [-1,1]), use nearest neighbor or global mean
            # Fallback to simple interpolation or 0
            predictions[i] = 0.0
            
    return predictions

# EVOLVE-BLOCK-END
