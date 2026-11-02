"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import linprog, minimize

N = 21
DEADLINE = 120.0  # seconds
SAFETY_MARGIN = 5.0  # seconds before deadline to stop


def _validate(circles, tol=1e-9):
    """Check that circles are valid: disjoint interiors, inside rectangle with perimeter <= 4."""
    if len(circles) != N:
        return False
    if np.any(circles[:, 2] <= 0):
        return False
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    xmin = np.min(x - r)
    xmax = np.max(x + r)
    ymin = np.min(y - r)
    ymax = np.max(y + r)
    
    width = xmax - xmin
    height = ymax - ymin
    perimeter = 2 * (width + height)
    
    if perimeter > 4 + tol:
        return False
    
    # Vectorized disjoint check
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if dist < r[i] + r[j] - tol:
                return False
    
    return True


def _compute_distances(positions):
    """Compute pairwise distance matrix."""
    n = len(positions)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d[i,j] = d[j,i] = np.sqrt((positions[i,0]-positions[j,0])**2 + (positions[i,1]-positions[j,1])**2)
    return d


def _solve_lp_fast(positions, w):
    """Given fixed positions and fixed width w, solve LP for max sum of radii.
    Returns (radii, sum_radii) or (None, -1) if infeasible."""
    n = len(positions)
    x = positions[:, 0]
    y = positions[:, 1]
    h = 2.0 - w
    
    # Shift so min x, min y are at 0
    x_shifted = x - x.min()
    y_shifted = y - y.min()
    
    if x_shifted.max() > w or y_shifted.max() > h:
        return None, -1
    
    # Precompute pairwise distances
    d_mat = _compute_distances(positions)
    
    # Build constraint matrix
    n_constraints = 4 * n + n * (n - 1) // 2
    A_ub = np.zeros((n_constraints, n))
    b_ub = np.zeros(n_constraints)
    
    row = 0
    # Boundary constraints
    for i in range(n):
        A_ub[row, i] = 1.0
        b_ub[row] = x_shifted[i]
        row += 1
        A_ub[row, i] = 1.0
        b_ub[row] = w - x_shifted[i]
        row += 1
        A_ub[row, i] = 1.0
        b_ub[row] = y_shifted[i]
        row += 1
        A_ub[row, i] = 1.0
        b_ub[row] = h - y_shifted[i]
        row += 1
    
    # Pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            d = d_mat[i, j]
            if d <= 1e-15:
                continue
            A_ub[row, i] = 1.0
            A_ub[row, j] = 1.0
            b_ub[row] = d
            row += 1
    
    A_ub = A_ub[:row]
    b_ub = b_ub[:row]
    
    c = -np.ones(n)
    bounds = [(0, None)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, np.sum(res.x)
    except Exception:
        pass
    
    return None, -1


def _solve_lp(positions):
    """Given fixed positions, solve LP for maximum sum of radii trying multiple aspect ratios."""
    n = len(positions)
    x = positions[:, 0]
    y = positions[:, 1]
    
    best_r = None
    best_sum = -1
    
    # Try a range of aspect ratios
    for w in np.linspace(0.3, 1.7, 20):
        r, total = _solve_lp_fast(positions, w)
        if r is not None and total > best_sum:
            best_sum = total
            best_r = r.copy()
    
    return best_r, best_sum


def _hexagonal_init(n, w, h, seed=42):
    """Initialize circles in hexagonal packing pattern."""
    rng = np.random.RandomState(seed)
    positions = np.zeros((n, 2))
    
    # Hexagonal packing: rows offset by half spacing
    dx = 1.0
    dy = dx * np.sqrt(3) / 2
    
    idx = 0
    row = 0
    while idx < n and row < 20:
        offset = (row % 2) * 0.5 * dx
        col = 0
        while idx < n and col < 20:
            x = offset + col * dx
            y = row * dy
            if x < w and y < h:
                positions[idx, 0] = x
                positions[idx, 1] = y
                idx += 1
            col += 1
        row += 1
    
    # Normalize to fit in [0,w] x [0,h]
    if idx > 0:
        positions[:, 0] = positions[:, 0] / max(positions[:, 0].max(), 1e-10) * w
        positions[:, 1] = positions[:, 1] / max(positions[:, 1].max(), 1e-10) * h
    
    # Add small random perturbation
    positions += rng.uniform(-0.01, 0.01, size=positions.shape)
    positions[:, 0] = np.clip(positions[:, 0], 0, w)
    positions[:, 1] = np.clip(positions[:, 1], 0, h)
    
    return positions


def _grid_init(n, w, h, seed=42):
    """Initialize circles in grid pattern."""
    rng = np.random.RandomState(seed)
    cols = int(np.ceil(np.sqrt(n * w / h)))
    rows = int(np.ceil(n / cols))
    
    positions = np.zeros((n, 2))
    for i in range(n):
        col = i % cols
        row = i // cols
        positions[i, 0] = (col + 0.5) * w / cols
        positions[i, 1] = (row + 0.5) * h / rows
    
    positions += rng.uniform(-0.01, 0.01, size=positions.shape)
    positions[:, 0] = np.clip(positions[:, 0], 0, w)
    positions[:, 1] = np.clip(positions[:, 1], 0, h)
    
    return positions


def _slsqp_polish(positions, w, max_iter=100):
    """Use SLSQP to polish positions for a fixed width w."""
    h = 2.0 - w
    n = len(positions)
    
    def objective(params):
        x = params[:n]
        y = params[n:2*n]
        r = params[2*n:3*n]
        
        # Check boundary violations
        for i in range(n):
            if x[i] - r[i] < -1e-8 or x[i] + r[i] > w + 1e-8:
                return 1e6
            if y[i] - r[i] < -1e-8 or y[i] + r[i] > h + 1e-8:
                return 1e6
        
        # Check overlap
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d < r[i] + r[j] - 1e-8:
                    return 1e6
        
        return -np.sum(r)
    
    # Initial guess
    pos0 = positions.copy()
    r0, _ = _solve_lp_fast(pos0, w)
    if r0 is None:
        r0 = np.full(n, 0.01)
    
    x0 = np.concatenate([pos0[:, 0], pos0[:, 1], r0])
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0, w))  # x
    for i in range(n):
        bounds.append((0, h))  # y
    for i in range(n):
        bounds.append((0, 1.0))  # r
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                      options={'maxiter': max_iter, 'ftol': 1e-10})
        if res.success:
            x_new = res.x[:n]
            y_new = res.x[n:2*n]
            r_new = res.x[2*n:3*n]
            if np.sum(r_new) > 0:
                return np.column_stack([x_new, y_new]), r_new, np.sum(r_new)
    except Exception:
        pass
    
    return None, None, -1


def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    # Initialize with a valid grid fallback to ensure we always have a valid result
    cols = 5
    rows = 5
    r_init = 0.09
    init_positions = []
    for i in range(cols):
        for j in range(rows):
            if len(init_positions) < N:
                x = (i + 0.5) / cols
                y = (j + 0.5) / rows
                init_positions.append([x, y, r_init])
    best_circles = np.array(init_positions[:N])
    best_sum = np.sum(best_circles[:, 2])
    best_w = 1.0
    
    # Try multiple aspect ratios with multiple initializations
    aspect_ratios = np.linspace(0.4, 1.6, 11)
    
    for w in aspect_ratios:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        h = 2.0 - w
        
        # Try both hexagonal and grid initializations
        for init_func in [_hexagonal_init, _grid_init]:
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            
            positions = init_func(N, w, h, seed=int(rng.randint(1000)))
            
            # Get initial radii with fixed w
            current_r, current_sum = _solve_lp_fast(positions, w)
            if current_r is None:
                continue
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = np.column_stack([positions[:, 0], positions[:, 1], current_r])
                best_w = w
            
            # Simulated annealing with fixed w (much faster)
            temp = 0.03
            cooling = 0.997
            current_pos = positions.copy()
            current_r = current_r.copy()
            current_sum = current_sum
            no_improve_count = 0
            max_no_improve = 200
            
            while time.time() - start_time < DEADLINE - SAFETY_MARGIN:
                # Random perturbation of one or two circles
                n_move = rng.choice([1, 1, 2], p=[0.6, 0.2, 0.2])
                idx = rng.choice(N, size=n_move, replace=False)
                dx = rng.uniform(-temp, temp, size=n_move)
                dy = rng.uniform(-temp, temp, size=n_move)
                
                new_pos = current_pos.copy()
                for k, i in enumerate(idx):
                    new_pos[i, 0] = np.clip(new_pos[i, 0] + dx[k], 0, w)
                    new_pos[i, 1] = np.clip(new_pos[i, 1] + dy[k], 0, h)
                
                r_new, sum_new = _solve_lp_fast(new_pos, w)
                
                if r_new is not None:
                    if sum_new > current_sum:
                        current_pos = new_pos
                        current_r = r_new
                        current_sum = sum_new
                        no_improve_count = 0
                        
                        if current_sum > best_sum:
                            best_sum = current_sum
                            best_circles = np.column_stack([current_pos[:, 0], current_pos[:, 1], current_r])
                            best_w = w
                    else:
                        # Accept with probability exp((sum_new - current_sum) / temp)
                        if temp > 1e-10:
                            prob = np.exp((sum_new - current_sum) / temp)
                            if rng.rand() < prob:
                                current_pos = new_pos
                                current_r = r_new
                                current_sum = sum_new
                        no_improve_count += 1
                
                temp *= cooling
                
                if no_improve_count > max_no_improve and temp < 0.001:
                    break
            
            # SLSQP polish if time permits
            if time.time() - start_time < DEADLINE - SAFETY_MARGIN - 5:
                polished_pos, polished_r, polished_sum = _slsqp_polish(current_pos, w, max_iter=50)
                if polished_pos is not None and polished_sum > best_sum:
                    best_sum = polished_sum
                    best_circles = np.column_stack([polished_pos[:, 0], polished_pos[:, 1], polished_r])
                    best_w = w
    
    # If no better solution found, or best_circles is None, use a valid grid fallback
    if best_circles is None:
        # Valid grid fallback: 5x5 grid in 1x1 square, radius 0.09
        # Perimeter = 2*(1+1) = 4.
        # Distance between centers = 0.2. 2*r = 0.18 < 0.2. Valid.
        cols = 5
        rows = 5
        r = 0.09
        positions = []
        for i in range(cols):
            for j in range(rows):
                if len(positions) < N:
                    x = (i + 0.5) / cols
                    y = (j + 0.5) / rows
                    positions.append([x, y, r])
        best_circles = np.array(positions[:N])
        best_sum = np.sum(best_circles[:, 2])
    
    # Final validation and adjustment
    if _validate(best_circles):
        return best_circles
    
    # If validation fails, shrink radii slightly to ensure validity
    circles = best_circles.copy()
    # Calculate max overlap violation to determine shrink factor
    # Simple approach: shrink by 5%
    circles[:, 2] *= 0.95
    if _validate(circles):
        return circles
    
    # Last resort: return the grid fallback directly (it is valid)
    cols = 5
    rows = 5
    r = 0.09
    positions = []
    for i in range(cols):
        for j in range(rows):
            if len(positions) < N:
                x = (i + 0.5) / cols
                y = (j + 0.5) / rows
                positions.append([x, y, r])
    return np.array(positions[:N])


if __name__ == "__main__":
    result = construct()
    print(f"Sum of radii: {np.sum(result[:, 2]):.6f}")
    print(f"Valid: {_validate(result)}")
