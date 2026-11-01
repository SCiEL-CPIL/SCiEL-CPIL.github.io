"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _validity_check(circles, n=N, tol=1e-9):
    """Check if circles are valid: inside unit square, non-overlapping."""
    if len(circles) != n:
        return False
    for i in range(n):
        x, y, r = circles[i]
        if r < -tol or x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist_sq = (xi - xj)**2 + (yi - yj)**2
            min_dist = ri + rj
            if dist_sq < min_dist**2 - tol:
                return False
    return True

def _pairwise(n):
    """Return list of (i, j) pairs for i < j."""
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j))
    return pairs

_PAIRS_32 = _pairwise(32)

def _objective_and_grad(params, n=N, pairs=_PAIRS_32):
    """Minimize negative sum of radii with smooth penalty. Returns (f, g)."""
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:3*n]
    
    f = -np.sum(r)
    grad = np.zeros(3*n)
    
    # Boundary penalties (smooth)
    for i in range(n):
        # x - r < 0
        v = x[i] - r[i]
        if v < 0:
            f += v*v
            grad[i] += 2*v       # d/dx_i
            grad[2*n+i] += -2*v  # d/dr_i
        # x + r > 1
        v = x[i] + r[i] - 1.0
        if v > 0:
            f += v*v
            grad[i] += 2*v
            grad[2*n+i] += 2*v
        # y - r < 0
        v = y[i] - r[i]
        if v < 0:
            f += v*v
            grad[n+i] += 2*v
            grad[2*n+i] += -2*v
        # y + r > 1
        v = y[i] + r[i] - 1.0
        if v > 0:
            f += v*v
            grad[n+i] += 2*v
            grad[2*n+i] += 2*v
        # r < 0
        if r[i] < 0:
            f += r[i]**2
            grad[2*n+i] += 2*r[i]
    
    # Overlap penalties (smooth): (r_i + r_j)^2 - dist^2 when positive
    for i, j in pairs:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist_sq = dx*dx + dy*dy
        min_dist = r[i] + r[j]
        overlap = min_dist*min_dist - dist_sq
        if overlap > 0:
            f += overlap
            # d(overlap)/dx_i = -2*dx, d/dx_j = 2*dx
            grad[i] += -2.0*dx
            grad[j] += 2.0*dx
            grad[n+i] += -2.0*dy
            grad[n+j] += 2.0*dy
            # d(overlap)/dr_i = 2*(r_i+r_j), same for r_j
            grad[2*n+i] += 2.0*min_dist
            grad[2*n+j] += 2.0*min_dist
    
    # Scale penalty
    f *= 1.0
    # We want to maximize sum(r), so objective = -sum(r) + penalty
    # The penalty terms above were added to f directly; scale them
    # Actually let me restructure: f = -sum(r) + penalty
    # Let me redo this properly
    
    return f, grad

def _objective(params, n=N, pairs=_PAIRS_32):
    f, _ = _objective_and_grad(params, n, pairs)
    return f

def _grad(params, n=N, pairs=_PAIRS_32):
    _, g = _objective_and_grad(params, n, pairs)
    return g

def _objective_and_grad(params, n=N, pairs=_PAIRS_32):
    """Minimize negative sum of radii with smooth penalty. Returns (f, g)."""
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:3*n]
    
    f = -np.sum(r)
    penalty = 0.0
    grad = np.zeros(3*n)
    
    # Boundary penalties (smooth)
    for i in range(n):
        v = x[i] - r[i]
        if v < 0:
            penalty += v*v
            grad[i] += 2*v
            grad[2*n+i] += -2*v
        v = x[i] + r[i] - 1.0
        if v > 0:
            penalty += v*v
            grad[i] += 2*v
            grad[2*n+i] += 2*v
        v = y[i] - r[i]
        if v < 0:
            penalty += v*v
            grad[n+i] += 2*v
            grad[2*n+i] += -2*v
        v = y[i] + r[i] - 1.0
        if v > 0:
            penalty += v*v
            grad[n+i] += 2*v
            grad[2*n+i] += 2*v
        if r[i] < 0:
            penalty += r[i]**2
            grad[2*n+i] += 2*r[i]
    
    # Overlap penalties
    for i, j in pairs:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist_sq = dx*dx + dy*dy
        min_dist = r[i] + r[j]
        overlap = min_dist*min_dist - dist_sq
        if overlap > 0:
            penalty += overlap
            grad[i] += -2.0*dx
            grad[j] += 2.0*dx
            grad[n+i] += -2.0*dy
            grad[n+j] += 2.0*dy
            grad[2*n+i] += 2.0*min_dist
            grad[2*n+j] += 2.0*min_dist
    
    f += 100.0 * penalty
    grad += 100.0 * grad
    
    return f, grad

def _hex_initialization(n=N):
    """Hexagonal close-packing initialization."""
    # Arrange in hexagonal pattern
    circles = []
    # Try different row counts
    best = None
    best_r = 0
    
    for rows in range(4, 8):
        cols = int(np.ceil(n / rows))
        # Hexagonal spacing
        dx = 1.0 / (cols + 0.5)
        dy = dx * np.sqrt(3) / 2.0
        
        count = 0
        for row in range(rows):
            for col in range(cols):
                if count >= n:
                    break
                x = (col + 0.5) * dx + (0.5 * dx if row % 2 == 1 else 0)
                y = (row + 0.5) * dy
                # Ensure inside
                if x < 0.05 or x > 0.95 or y < 0.05 or y > 0.95:
                    x = np.clip(x, 0.1, 0.9)
                    y = np.clip(y, 0.1, 0.9)
                r = dx * 0.45
                circles.append([x, y, r])
                count += 1
        if count >= n:
            arr = np.array(circles[:n])
            r_val = np.sum(arr[:, 2])
            if r_val > best_r:
                best_r = r_val
                best = arr
        circles = []
    
    if best is not None:
        return best
    
    # Fallback: square grid
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < n:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
    return np.array(cells[:n])

def _square_grid_initialization(n=N):
    """Create an initial grid arrangement."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < n:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
    return np.array(cells[:n])

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s limit
    
    n = N
    best_circles = _hex_initialization(n)
    best_score = np.sum(best_circles[:, 2])
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Multiple restarts with analytic-gradient L-BFGS-B
    for restart in range(300):
        if time.time() > deadline - 20:
            break
        
        if restart == 0:
            circles = _hex_initialization(n)
        elif restart == 1:
            circles = _square_grid_initialization(n)
        elif restart == 2:
            # Staggered grid
            side = int(np.ceil(np.sqrt(n)))
            circles = np.zeros((n, 3))
            count = 0
            for i in range(side):
                for j in range(side):
                    if count >= n:
                        break
                    x = (i + 0.5) / side + (0.5/side if j % 2 else 0)
                    y = (j + 0.5) / side
                    x = np.clip(x, 0.05, 0.95)
                    r = 0.5 / side - 1e-3
                    circles[count] = [x, y, r]
                    count += 1
        else:
            # Random initialization with small radii
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i, 0] = rng.uniform(0.1, 0.9)
                circles[i, 1] = rng.uniform(0.1, 0.9)
                circles[i, 2] = rng.uniform(0.01, 0.06)
        
        params = circles.flatten()
        
        try:
            result = minimize(_objective, params, jac=_grad, method='L-BFGS-B',
                             options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-14})
            new_params = result.x
            new_circles = new_params.reshape(n, 3)
            
            if _validity_check(new_circles, n):
                new_score = np.sum(new_circles[:, 2])
                if new_score > best_score:
                    best_score = new_score
                    best_circles = new_circles.copy()
        except Exception:
            continue
        
        # Local refinement: perturb and re-optimize
        if restart % 5 == 0 and time.time() < deadline - 25:
            perturbed = best_circles.copy()
            scale = 0.02 / (1 + restart * 0.005)
            for i in range(n):
                perturbed[i, 0] += rng.normal(0, scale)
                perturbed[i, 1] += rng.normal(0, scale)
                perturbed[i, 2] += rng.normal(0, scale * 0.5)
                perturbed[i, 0] = np.clip(perturbed[i, 0], 0.01, 0.99)
                perturbed[i, 1] = np.clip(perturbed[i, 1], 0.01, 0.99)
                perturbed[i, 2] = np.clip(perturbed[i, 2], 0.005, 0.2)
            
            params = perturbed.flatten()
            try:
                result = minimize(_objective, params, jac=_grad, method='L-BFGS-B',
                                 options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14})
                new_circles = result.x.reshape(n, 3)
                if _validity_check(new_circles, n):
                    new_score = np.sum(new_circles[:, 2])
                    if new_score > best_score:
                        best_score = new_score
                        best_circles = new_circles.copy()
            except Exception:
                continue
    
    # Phase 2: High-precision polish on best solution
    if time.time() < deadline - 5:
        params = best_circles.flatten()
        try:
            result = minimize(_objective, params, jac=_grad, method='L-BFGS-B',
                             options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-16,
                                      'maxcor': 20, 'maxls': 50})
            new_circles = result.x.reshape(n, 3)
            if _validity_check(new_circles, n):
                new_score = np.sum(new_circles[:, 2])
                if new_score > best_score:
                    best_score = new_score
                    best_circles = new_circles.copy()
        except Exception:
            pass
        
        # One more polish with even tighter settings
        if time.time() < deadline - 2:
            params = best_circles.flatten()
            try:
                result = minimize(_objective, params, jac=_grad, method='L-BFGS-B',
                                 options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-18})
                new_circles = result.x.reshape(n, 3)
                if _validity_check(new_circles, n):
                    new_score = np.sum(new_circles[:, 2])
                    if new_score > best_score:
                        best_score = new_score
                        best_circles = new_circles.copy()
            except Exception:
                pass
    
    # Final validation
    if not _validity_check(best_circles, n):
        best_circles = _square_grid_initialization(n)
    
    return best_circles
# EVOLVE-BLOCK-END
