"""Physics-based circle packing in perimeter-4 rectangle."""
import numpy as np
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N circles in a rectangle of perimeter 4 (i.e., width + height = 2).
    Maximize sum of radii.
    
    Approach: Use gradient-based optimization with penalty for overlaps
    and boundary violations. Start from a grid initialization.
    """
    # Rectangle: width w, height h, with w + h = 2
    # We'll parameterize by aspect ratio: w = 2*r/(1+r), h = 2/(1+r) where r = w/h
    # Or simpler: fix w and h with w+h=2, optimize positions and radii
    
    # Initial grid placement
    side = int(np.ceil(np.sqrt(N)))
    
    # Start with a square-ish rectangle
    # Let's use w = 1.2, h = 0.8 (w+h=2) as initial guess, or optimize
    # Actually, let's try multiple aspect ratios
    
    best_result = None
    best_score = -np.inf
    
    # Try several aspect ratios
    for aspect in [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        
        # Initialize positions in a grid within [0,w] x [0,h]
        positions = np.zeros((N, 2))
        for i in range(N):
            row = i // side
            col = i % side
            positions[i, 0] = (col + 0.5) * w / side
            positions[i, 1] = (row + 0.5) * h / side
        
        # Initial radii: small
        radii = np.full(N, 0.05)
        
        # Optimize
        result = _optimize_packing(positions, radii, w, h, N)
        if result is not None:
            score = np.sum(result[:, 2])
            if score > best_score:
                best_score = score
                best_result = result
    
    # Also try a more systematic approach: optimize w, h, positions, radii together
    result2 = _optimize_full(N)
    if result2 is not None:
        score2 = np.sum(result2[:, 2])
        if score2 > best_score:
            best_score = score2
            best_result = result2
    
    if best_result is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        w, h = 1.0, 1.0
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result


def _optimize_packing(positions, radii, w, h, N, max_iter=500):
    """
    Optimize positions and radii for fixed rectangle w x h (w+h=2).
    Maximize sum(radii) subject to:
    - Circles don't overlap: dist(i,j) >= r_i + r_j
    - Circles inside rectangle: r_i <= x_i <= w - r_i, r_i <= y_i <= h - r_i
    """
    # Combine into single vector: [x_1, y_1, r_1, x_2, y_2, r_2, ...]
    x0 = np.zeros(3 * N)
    for i in range(N):
        x0[3*i] = positions[i, 0]
        x0[3*i+1] = positions[i, 1]
        x0[3*i+2] = radii[i]
    
    def objective(vars_):
        pos = vars_[0::3]
        rad = vars_[1::3]
        # Maximize sum(rad) => minimize -sum(rad)
        return -np.sum(rad)
    
    def penalty(vars_):
        pos = vars_[0::3]
        rad = vars_[1::3]
        pen = 0.0
        
        # Overlap penalties
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i] - pos[j]
                dy = pos[i+1] - pos[j+1] if False else pos[3*i+1] - pos[3*j+1]
                dist = np.sqrt((pos[3*i] - pos[3*j])**2 + (pos[3*i+1] - pos[3*j+1])**2)
                min_dist = rad[i] + rad[j]
                if dist < min_dist:
                    pen += (min_dist - dist) ** 2
        
        # Boundary penalties
        for i in range(N):
            x, y, r = pos[3*i], pos[3*i+1], rad[i]
            if x < r: pen += (r - x) ** 2
            if x > w - r: pen += (x - (w - r)) ** 2
            if y < r: pen += (r - y) ** 2
            if y > h - r: pen += (y - (h - r)) ** 2
        
        return pen
    
    def total(vars_):
        return objective(vars_) + 100.0 * penalty(vars_)
    
    # Use Nelder-Mead or L-BFGS-B
    try:
        res = minimize(total, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8})
        if res.success or res.fun < total(x0):
            final_vars = res.x
            pos = final_vars[0::3]
            rad = final_vars[1::3]
            
            # Verify validity
            valid = True
            for i in range(N):
                x, y, r = pos[3*i], pos[3*i+1], rad[i]
                if r <= 0 or x < r - 1e-6 or x > w - r + 1e-6 or y < r - 1e-6 or y > h - r + 1e-6:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dx = pos[3*i] - pos[3*j]
                        dy = pos[3*i+1] - pos[3*j+1]
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < rad[i] + rad[j] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                result = np.zeros((N, 3))
                for i in range(N):
                    result[i] = [pos[3*i], pos[3*i+1], rad[i]]
                return result
    except Exception:
        pass
    
    return None


def _optimize_full(N, max_iter=300):
    """
    Full optimization: optimize w, h (with w+h=2), positions, radii.
    """
    # Parameterize: aspect = w/h, so w = 2*aspect/(1+aspect), h = 2/(1+aspect)
    # Variables: [log_aspect, x_1, y_1, r_1, ..., x_N, y_N, r_N]
    
    def unpack(vars_):
        log_aspect = vars_[0]
        aspect = np.exp(log_aspect)
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        pos_rad = vars_[1:]
        pos = pos_rad[0::3]
        rad = pos_rad[1::3]
        return w, h, pos, rad
    
    def objective(vars_):
        w, h, pos, rad = unpack(vars_)
        return -np.sum(rad)
    
    def penalty(vars_):
        w, h, pos, rad = unpack(vars_)
        pen = 0.0
        
        # Overlap
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[3*i] - pos[3*j]
                dy = pos[3*i+1] - pos[3*j+1]
                dist = np.sqrt(dx**2 + dy**2)
                min_dist = rad[i] + rad[j]
                if dist < min_dist:
                    pen += (min_dist - dist) ** 2
        
        # Boundary
        for i in range(N):
            x, y, r = pos[3*i], pos[3*i+1], rad[i]
            if x < r: pen += (r - x) ** 2
            if x > w - r: pen += (x - (w - r)) ** 2
            if y < r: pen += (r - y) ** 2
            if y > h - r: pen += (y - (h - r)) ** 2
        
        # Keep aspect ratio reasonable
        aspect = np.exp(vars_[0])
        if aspect < 0.3 or aspect > 3.0:
            pen += 10.0
        
        return pen
    
    def total(vars_):
        return objective(vars_) + 50.0 * penalty(vars_)
    
    # Initial guess: square, grid placement
    side = int(np.ceil(np.sqrt(N)))
    w0, h0 = 1.0, 1.0
    
    x0 = np.zeros(1 + 3*N)
    x0[0] = np.log(1.0)  # aspect = 1
    for i in range(N):
        row = i // side
        col = i % side
        x0[1 + 3*i] = (col + 0.5) * w0 / side
        x0[2 + 3*i] = (row + 0.5) * h0 / side
        x0[3 + 3*i] = 0.05
    
    try:
        res = minimize(total, x0, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-8})
        final_vars = res.x
        w, h, pos, rad = unpack(final_vars)
        
        # Verify
        valid = True
        for i in range(N):
            x, y, r = pos[3*i], pos[3*i+1], rad[i]
            if r <= 0 or x < r - 1e-6 or x > w - r + 1e-6 or y < r - 1e-6 or y > h - r + 1e-6:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos[3*i] - pos[3*j]
                    dy = pos[3*i+1] - pos[3*j+1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < rad[i] + rad[j] - 1e-6:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            result = np.zeros((N, 3))
            for i in range(N):
                result[i] = [pos[3*i], pos[3*i+1], rad[i]]
            return result
    except Exception:
        pass
    
    return None

# EVOLVE-BLOCK-END
