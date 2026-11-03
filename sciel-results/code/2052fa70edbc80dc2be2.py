"""Rewritten program: gradient-based circle packing in perimeter-4 rectangle."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
DEADLINE = 115.0  # safety margin before 120s kill


def _objective_and_grad(x):
    """
    x layout: [cx (N), cy (N), r (N), W, H]
    Objective: maximize sum(r) subject to:
      - perimeter W + H <= 2  (since 2(W+H)=4 => W+H=2)
      - circles inside rectangle: cx_i >= r_i, cx_i <= W - r_i, cy_i >= r_i, cy_i <= H - r_i
      - non-overlap: dist(i,j) >= r_i + r_j
    We minimize -sum(r) with penalty for constraint violations.
    """
    cx = x[:N]
    cy = x[N:2*N]
    r = x[2*N:3*N]
    W = x[3*N]
    H = x[3*N+1]

    # Perimeter constraint: W + H = 2 (we enforce equality by parameterizing, but use penalty)
    perim_viol = max(0.0, (W + H) - 2.0)

    # Boundary violations
    bound_viol = 0.0
    for i in range(N):
        bound_viol += max(0.0, r[i] - cx[i])
        bound_viol += max(0.0, cx[i] - (W - r[i]))
        bound_viol += max(0.0, r[i] - cy[i])
        bound_viol += max(0.0, cy[i] - (H - r[i]))

    # Overlap violations
    overlap_viol = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.sqrt(dx*dx + dy*dy)
            overlap_viol += max(0.0, r[i] + r[j] - d)

    # Soft penalty: use smooth max via log-sum-exp style
    # Actually, for gradient-based, use quadratic penalty on violations
    penalty = 100.0 * (perim_viol**2 + bound_viol**2 + overlap_viol**2)

    obj = -np.sum(r) + penalty

    # Gradient
    grad = np.zeros_like(x)
    # d(-sum r)/dr = -1
    grad[2*N:3*N] = -1.0

    # Penalty gradients
    # Perimeter
    if perim_viol > 0:
        gp = 200.0 * perim_viol
        grad[3*N] += gp
        grad[3*N+1] += gp

    # Boundary
    for i in range(N):
        # cx >= r => violation if r - cx > 0
        v1 = max(0.0, r[i] - cx[i])
        if v1 > 0:
            g = 200.0 * v1
            grad[i] += g       # d/d cx: -1 * 2*v * 100 ... actually d/dcx of (r-cx)^2 = -2(r-cx)
            # Let me redo: penalty = 100*(v1^2), v1 = r[i]-cx[i]
            # d/dcx = 100 * 2*v1 * (-1) = -200*v1
            # d/dr  = 100 * 2*v1 * (1)  = 200*v1
            grad[i] += -200.0 * v1
            grad[2*N+i] += 200.0 * v1

        v2 = max(0.0, cx[i] - (W - r[i]))
        if v2 > 0:
            grad[i] += 200.0 * v2
            grad[2*N+i] += 200.0 * v2
            grad[3*N] += -200.0 * v2

        v3 = max(0.0, r[i] - cy[i])
        if v3 > 0:
            grad[N+i] += -200.0 * v3
            grad[2*N+i] += 200.0 * v3

        v4 = max(0.0, cy[i] - (H - r[i]))
        if v4 > 0:
            grad[N+i] += 200.0 * v4
            grad[2*N+i] += 200.0 * v4
            grad[3*N+1] += -200.0 * v4

    # Overlap
    for i in range(N):
        for j in range(i+1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.sqrt(dx*dx + dy*dy)
            v = r[i] + r[j] - d
            if v > 0:
                if d < 1e-12:
                    d = 1e-12
                    dx = 1e-12
                    dy = 0.0
                g = 200.0 * v
                # d/dcx_i of (r_i+r_j-d)^2 = 2*v * (-dx/d)
                grad[i] += g * (-dx/d)
                grad[N+i] += g * (-dy/d)
                grad[j] += g * (dx/d)
                grad[N+j] += g * (dy/d)
                grad[2*N+i] += g
                grad[2*N+j] += g

    return obj, grad


def _hex_init():
    """Initialize with a hexagonal-like packing pattern."""
    # Try to arrange 21 circles in a roughly hexagonal pattern
    # 6-6-5-4 or similar rows
    rows = [6, 6, 5, 4]  # 21 circles
    # Estimate radius: for a hex packing in a rectangle
    # Let's start with equal radii and optimize
    cx = np.zeros(N)
    cy = np.zeros(N)
    r = np.full(N, 0.1)
    
    idx = 0
    row_y = 0
    for ri, count in enumerate(rows):
        y = 0.2 + ri * 0.25
        for ci in range(count):
            x = 0.15 + ci * 0.22
            if idx < N:
                cx[idx] = x
                cy[idx] = y
                idx += 1
    
    W = 1.5
    H = 0.5
    x = np.concatenate([cx, cy, r, [W, H]])
    return x


def construct():
    t0 = time.time()
    best_x = None
    best_obj = np.inf

    # Multiple random restarts
    rng = np.random.default_rng(42)
    
    # Initial hex-like start
    x0 = _hex_init()
    
    # Also try a grid-based start
    side = int(np.ceil(np.sqrt(N)))
    cx_grid = np.zeros(N)
    cy_grid = np.zeros(N)
    r_grid = np.full(N, 0.5/side - 0.01)
    for i in range(N):
        cx_grid[i] = (i % side + 0.5) / side
        cy_grid[i] = (i // side + 0.5) / side
    W_grid = 1.5
    H_grid = 0.5
    x0_grid = np.concatenate([cx_grid, cy_grid, r_grid, [W_grid, H_grid]])
    
    starts = [x0, x0_grid]
    
    # Add a few random perturbations of the hex start
    for _ in range(4):
        x0_pert = x0.copy()
        x0_pert[:2*N] += rng.normal(0, 0.05, 2*N)
        x0_pert[2*N:3*N] = np.clip(x0_pert[2*N:3*N], 0.05, 0.2)
        starts.append(x0_pert)
    
    best_result = None
    
    for start in starts:
        if time.time() - t0 > DEADLINE - 10:
            break
        
        try:
            res = minimize(_objective_and_grad, start, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
            
            # Check if feasible and record
            x = res.x
            cx = x[:N]
            cy = x[N:2*N]
            r = x[2*N:3*N]
            W = x[3*N]
            H = x[3*N+1]
            
            # Check feasibility
            feasible = True
            if W + H > 2.0 + 1e-6:
                feasible = False
            for i in range(N):
                if cx[i] < r[i] - 1e-6 or cx[i] > W - r[i] + 1e-6:
                    feasible = False
                if cy[i] < r[i] - 1e-6 or cy[i] > H - r[i] + 1e-6:
                    feasible = False
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((cx[i]-cx[j])**2 + (cy[i]-cy[j])**2)
                    if d < r[i] + r[j] - 1e-6:
                        feasible = False
            
            if feasible:
                # Scale to use full perimeter
                # Current perimeter is 2*(W+H), we want it to be 4, so W+H=2
                # If W+H < 2, we can scale up
                scale = 2.0 / (W + H) if (W + H) > 1e-6 else 1.0
                cx_s = cx * scale
                cy_s = cy * scale
                r_s = r * scale
                W_s = W * scale
                H_s = H * scale
                
                # Verify after scaling
                ok = True
                if W_s + H_s > 2.0 + 1e-5:
                    ok = False
                for i in range(N):
                    if cx_s[i] < r_s[i] - 1e-5 or cx_s[i] > W_s - r_s[i] + 1e-5:
                        ok = False
                    if cy_s[i] < r_s[i] - 1e-5 or cy_s[i] > H_s - r_s[i] + 1e-5:
                        ok = False
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.sqrt((cx_s[i]-cx_s[j])**2 + (cy_s[i]-cy_s[j])**2)
                        if d < r_s[i] + r_s[j] - 1e-5:
                            ok = False
                
                if ok:
                    result = np.column_stack([cx_s, cy_s, r_s])
                    if best_result is None or np.sum(r_s) > np.sum(best_result[:, 2]):
                        best_result = result
        except Exception:
            pass
    
    if best_result is not None:
        return best_result
    
    # Fallback: return the grid solution
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])


if __name__ == "__main__":
    result = construct()
    print("Sum of radii:", np.sum(result[:, 2]))
    print("Number of circles:", len(result))
