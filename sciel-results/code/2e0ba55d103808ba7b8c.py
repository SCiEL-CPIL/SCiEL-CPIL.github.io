"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import sys

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import sys

N = 26

def _constraints(params):
    """
    Define constraints for SLSQP.
    Returns a list of constraint dictionaries.
    Each constraint: {'type': 'ineq', 'fun': lambda x: ...}
    """
    n = N
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:]
    
    cons = []
    
    # Boundary constraints:
    # x_i - r_i >= 0
    # 1 - x_i - r_i >= 0
    # y_i - r_i >= 0
    # 1 - y_i - r_i >= 0
    # r_i >= 0.01 (small min radius)
    
    # To speed up SLSQP, we can vectorize the constraint functions if possible, 
    # but SLSQP expects a list of dicts or a single dict with 'fun' returning array.
    # Let's use a single 'ineq' constraint that returns an array of violations.
    
    def boundary_violations(p):
        xx = p[:n]
        yy = p[n:2*n]
        rr = p[2*n:]
        v = np.concatenate([
            xx - rr,
            1 - xx - rr,
            yy - rr,
            1 - yy - rr,
            rr - 0.01
        ])
        return v
    
    cons.append({'type': 'ineq', 'fun': boundary_violations})
    
    # Pairwise constraints:
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    # This is O(N^2) constraints.
    # We can return a single array of all pairwise violations.
    
    def pairwise_violations(p):
        xx = p[:n]
        yy = p[n:2*n]
        rr = p[2*n:]
        violations = []
        for i in range(n):
            for j in range(i+1, n):
                dx = xx[i] - xx[j]
                dy = yy[i] - yy[j]
                d_sq = dx*dx + dy*dy
                min_d_sq = (rr[i] + rr[j])**2
                violations.append(d_sq - min_d_sq)
        return np.array(violations)
    
    cons.append({'type': 'ineq', 'fun': pairwise_violations})
    
    return cons

def _objective(params):
    """
    Objective function: minimize -sum(radii)
    """
    n = N
    r = params[2*n:]
    return -np.sum(r)

def _init_grid(cols, rows):
    """Initialize with a grid pattern."""
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    count = 0
    for row in range(rows):
        for col in range(cols):
            if count >= N:
                break
            # Center of cell
            cx = (col + 0.5) / cols
            cy = (row + 0.5) / rows
            # Max possible radius in this cell is min(0.5/cols, 0.5/rows)
            # Start slightly smaller to avoid immediate constraint violation
            max_r = min(0.5/cols, 0.5/rows)
            x[count] = cx
            y[count] = cy
            r[count] = max_r * 0.9
            count += 1
            
    # If we have fewer than N, fill the rest with small circles in a spiral or random
    if count < N:
        for i in range(count, N):
            x[i] = 0.5 + 0.3 * np.cos(2*np.pi*i/N)
            y[i] = 0.5 + 0.3 * np.sin(2*np.pi*i/N)
            r[i] = 0.05
            
    return np.concatenate([x, y, r])

def _init_random():
    """Initialize with random positions and small radii."""
    x = np.random.uniform(0.1, 0.9, N)
    y = np.random.uniform(0.1, 0.9, N)
    r = np.random.uniform(0.05, 0.1, N)
    return np.concatenate([x, y, r])

def _init_hex():
    """Initialize with a hexagonal packing pattern."""
    # Hexagonal packing is denser than square.
    # Let's try to fit as many hexagonal circles as possible.
    # Radius r, horizontal spacing 2r, vertical spacing sqrt(3)r.
    # Try a few scales.
    best_params = None
    best_score = -np.inf
    
    for r_scale in [0.1, 0.12, 0.15]:
        r0 = r_scale
        # Estimate how many fit
        # This is heuristic. Just place them.
        x = []
        y = []
        rr = []
        row = 0
        while len(x) < N:
            col = 0
            while len(x) < N:
                # Hexagon offset
                offset = 0 if row % 2 == 0 else r0
                cx = r0 + offset + col * 2 * r0
                cy = r0 + row * np.sqrt(3) * r0
                if cx < 1 - r0 and cy < 1 - r0:
                    x.append(cx)
                    y.append(cy)
                    rr.append(r0)
                col += 1
            row += 1
            if row > 10: break # Safety
            
        if len(x) >= N:
            x = np.array(x[:N])
            y = np.array(y[:N])
            rr = np.array(rr[:N])
            params = np.concatenate([x, y, rr])
            # Check validity roughly
            valid = True
            for i in range(N):
                if x[i] < rr[i] or x[i] > 1-rr[i] or y[i] < rr[i] or y[i] > 1-rr[i]:
                    valid = False
            if valid:
                # Simple score: sum of radii
                score = np.sum(rr)
                if score > best_score:
                    best_score = score
                    best_params = params
    if best_params is not None:
        return best_params
    return _init_random()

def _refine_slsqp(params, deadline, max_iter=200):
    """Refine using SLSQP with hard constraints."""
    bounds = [(0, 1)] * (2 * N) + [(0.01, 0.5)] * N
    
    try:
        # SLSQP can be sensitive to initial guess.
        # We pass the constraints.
        cons = _constraints(params)
        
        result = minimize(
            _objective,
            params,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': max_iter, 'ftol': 1e-8}
        )
        return result.x
    except Exception:
        return params

def _refine_l_bfgs(params, deadline, max_iter=100):
    """Polish using L-BFGS-B with penalty method for speed."""
    PENALTY_SCALE = 1000.0
    
    def obj_pen(p):
        n = N
        x = p[:n]
        y = p[n:2*n]
        r = p[2*n:]
        pen = 0.0
        # Boundaries
        for i in range(n):
            if x[i] < r[i]: pen += PENALTY_SCALE * (r[i]-x[i])**2
            if x[i] > 1-r[i]: pen += PENALTY_SCALE * (r[i]-(1-x[i]))**2
            if y[i] < r[i]: pen += PENALTY_SCALE * (r[i]-y[i])**2
            if y[i] > 1-r[i]: pen += PENALTY_SCALE * (r[i]-(1-y[i]))**2
        # Pairs
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i]-x[j]
                dy = y[i]-y[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = r[i]+r[j]
                if d < min_d:
                    pen += PENALTY_SCALE * (min_d-d)**2
        return pen - np.sum(r)
    
    try:
        result = minimize(
            obj_pen,
            params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-10}
        )
        return result.x
    except Exception:
        return params

def _score(params):
    """Calculate the actual sum of radii if valid, else -inf."""
    n = N
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:]
    
    # Check validity
    for i in range(n):
        if x[i] < r[i] - 1e-8 or x[i] > 1 - r[i] + 1e-8:
            return -np.inf
        if y[i] < r[i] - 1e-8 or y[i] > 1 - r[i] + 1e-8:
            return -np.inf
        if r[i] < 0:
            return -np.inf
            
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i]-x[j]
            dy = y[i]-y[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < r[i] + r[j] - 1e-8:
                return -np.inf
                
    return np.sum(r)

def construct():
    """Construct an arrangement of 26 circles in the unit square maximizing sum of radii."""
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_params = None
    best_score = -np.inf
    
    # Initialization strategies
    inits = []
    inits.append(_init_grid(5, 6))
    inits.append(_init_grid(6, 5))
    inits.append(_init_grid(4, 7))
    inits.append(_init_hex())
    for _ in range(5):
        inits.append(_init_random())
        
    # Process each initialization
    for i, init_params in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        # 1. SLSQP Refinement
        refined_slsqp = _refine_slsqp(init_params, deadline, max_iter=100)
        score = _score(refined_slsqp)
        
        if score > best_score:
            best_score = score
            best_params = refined_slsqp.copy()
            
        # 2. L-BFGS-B Polish on the SLSQP result
        if time.time() < deadline - 10:
            polished = _refine_l_bfgs(refined_slsqp, deadline, max_iter=50)
            score2 = _score(polished)
            if score2 > best_score:
                best_score = score2
                best_params = polished.copy()
                
    # If we have a good candidate, do a final multi-start perturbation + polish
    if best_params is not None and best_score > 0:
        if time.time() < deadline - 15:
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                noise = np.random.normal(0, 0.01, 3*N)
                perturbed = best_params + noise
                # Clip to bounds
                perturbed[:2*N] = np.clip(perturbed[:2*N], 0, 1)
                perturbed[2*N:] = np.clip(perturbed[2*N:], 0.01, 0.5)
                
                # Quick polish
                polished = _refine_l_bfgs(perturbed, deadline, max_iter=50)
                score = _score(polished)
                if score > best_score:
                    best_score = score
                    best_params = polished.copy()
                    
    # Final check and return
    if best_params is not None and best_score > 0:
        # Ensure strict validity by clipping
        n = N
        x = best_params[:n].copy()
        y = best_params[n:2*n].copy()
        r = best_params[2*n:].copy()
        
        # Clip radii
        r = np.clip(r, 1e-6, 0.5)
        # Clip positions
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        
        # Re-check validity after clipping (clipping might cause overlaps if we weren't careful, 
        # but usually clipping to [r, 1-r] is safe for boundaries. 
        # Overlaps might persist if the optimizer didn't fully resolve them, 
        # but SLSQP should have handled that. 
        # If invalid, we might need to shrink radii slightly to be safe.)
        
        # Safety shrink: if any overlap, reduce all radii by 1%
        valid = True
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i]-x[j]
                dy = y[i]-y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < r[i] + r[j] - 1e-9:
                    valid = False
                    break
            if not valid: break
        
        if not valid:
            # Shrink radii to ensure validity
            scale = 0.95
            for _ in range(10):
                r_new = r * scale
                x_new = np.clip(x, r_new, 1-r_new)
                y_new = np.clip(y, r_new, 1-r_new)
                valid = True
                for i in range(n):
                    for j in range(i+1, n):
                        dx = x_new[i]-x_new[j]
                        dy = y_new[i]-y_new[j]
                        d = np.sqrt(dx*dx + dy*dy)
                        if d < r_new[i] + r_new[j] - 1e-9:
                            valid = False
                            break
                    if not valid: break
                if valid:
                    r = r_new
                    x = x_new
                    y = y_new
                    break
                scale *= 0.95
                
        return np.column_stack([x, y, r])
    
    # Fallback: simple grid
    side = 5
    cells = []
    count = 0
    for row in range(6):
        for col in range(5):
            if count >= N: break
            x_val = (col + 0.5) / 5
            y_val = (row + 0.5) / 6
            r_val = 0.5 / max(5, 6) - 0.01
            cells.append([x_val, y_val, r_val])
            count += 1
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
