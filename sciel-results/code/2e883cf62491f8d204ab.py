"""
Circle packing in perimeter-4 rectangle using SDP-inspired approach.

We pack 21 circles in a rectangle of perimeter 4 (so 2*(W+H)=4, i.e., W+H=2).
We maximize the sum of radii subject to non-overlap and containment constraints.

Approach:
- Use a grid-based initialization with optimization via scipy.optimize.
- The SDP relaxation idea suggests that we can formulate this as a convex problem
  in terms of squared distances, but for practical implementation within time limits,
  we use a direct nonlinear optimization with good initialization.
- We optimize over circle centers (x_i, y_i) and radii r_i, with the rectangle
  dimensions W, H also as variables (with W+H=2).

Constraints:
- For each circle i: r_i <= x_i, r_i <= W - x_i, r_i <= y_i, r_i <= H - y_i (containment)
- For each pair (i,j): (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 (non-overlap)
- W + H = 2, W > 0, H > 0

Objective: maximize sum(r_i)
"""

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

N = 21


def construct():
    # Variables: [W, H, x_1, y_1, r_1, x_2, y_2, r_2, ..., x_N, y_N, r_N]
    # Total dimension: 2 + 3*N = 65
    
    def unpack(v):
        W = v[0]
        H = v[1]
        rest = v[2:]
        centers = rest.reshape(N, 2)[:, :2]  # This won't work directly
        # Let's restructure: v[2:2+3*N] -> reshape to (N, 3) where each row is (x, y, r)
        return W, H, rest.reshape(N, 3)
    
    def objective(v):
        W, H, data = unpack(v)
        # Maximize sum of radii, so minimize negative
        return -np.sum(data[:, 2])
    
    def unpack2(v):
        W = v[0]
        H = v[1]
        data = v[2:].reshape(N, 3)
        return W, H, data
    
    # Objective: minimize -sum(r_i)
    def obj(v):
        W, H, data = unpack2(v)
        return -np.sum(data[:, 2])
    
    # Constraints:
    # 1. W + H = 2 (equality)
    # 2. W >= 0.1, H >= 0.1 (to avoid degenerate)
    # 3. For each circle i: 
    #    r_i <= x_i - 0  => x_i - r_i >= 0
    #    r_i <= W - x_i  => W - x_i - r_i >= 0
    #    r_i <= y_i      => y_i - r_i >= 0
    #    r_i <= H - y_i  => H - y_i - r_i >= 0
    # 4. For each pair (i,j), i<j:
    #    (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    
    # Let's use SLSQP with all constraints as functions
    
    def constraints_func(v):
        W, H, data = unpack2(v)
        x = data[:, 0]
        y = data[:, 1]
        r = data[:, 2]
        
        cons = []
        # Containment: x_i >= r_i
        cons.append(x - r)
        # W - x_i >= r_i
        cons.append(W - x - r)
        # y_i >= r_i
        cons.append(y - r)
        # H - y_i >= r_i
        cons.append(H - y - r)
        
        # Non-overlap for all pairs
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                rr = r[i] + r[j]
                cons.append(dx*dx + dy*dy - rr*rr)
        
        return np.concatenate(cons)
    
    # Bounds
    # W in [0.1, 1.9], H in [0.1, 1.9]
    # x_i in [0, W] but W is variable, so use loose bounds [0, 2]
    # y_i in [0, H] -> [0, 2]
    # r_i in [0, 1]
    
    bounds = [(0.1, 1.9), (0.1, 1.9)]
    for i in range(N):
        bounds.append((0, 2))  # x_i
        bounds.append((0, 2))  # y_i
        bounds.append((0, 1))  # r_i
    
    # Initial guess: grid layout
    # For N=21, use a 5x5 grid (25 cells), pick 21
    side = 5
    W0 = 1.0
    H0 = 1.0
    v0 = [W0, H0]
    for i in range(N):
        col = i % side
        row = i // side
        cx = (col + 0.5) / side * W0
        cy = (row + 0.5) / side * H0
        r = 0.5 / side * W0 * 0.9  # slightly smaller to avoid touching
        v0.extend([cx, cy, r])
    
    v0 = np.array(v0)
    
    # Equality constraint: W + H = 2
    def eq_constraint(v):
        W, H, _ = unpack2(v)
        return [W + H - 2.0]
    
    # Inequality constraints as a single function returning array >= 0
    def ineq_constraints(v):
        W, H, data = unpack2(v)
        x = data[:, 0]
        y = data[:, 1]
        r = data[:, 2]
        
        cons = []
        # Containment
        cons.extend((x - r).tolist())
        cons.extend((W - x - r).tolist())
        cons.extend((y - r).tolist())
        cons.extend((H - y - r).tolist())
        
        # Non-overlap
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                rr = r[i] + r[j]
                cons.append(dx*dx + dy*dy - rr*rr)
        
        return np.array(cons)
    
    # Use SLSQP
    from scipy.optimize import NonlinearConstraint, LinearConstraint
    
    # Equality: W + H = 2
    A_eq = np.zeros((1, 2 + 3*N))
    A_eq[0, 0] = 1.0
    A_eq[0, 1] = 1.0
    b_eq = [2.0]
    
    # Inequality constraints
    n_ineq = 4*N + N*(N-1)//2
    
    def ineq_func(v):
        W, H, data = unpack2(v)
        x = data[:, 0]
        y = data[:, 1]
        r = data[:, 2]
        
        cons = []
        cons.extend((x - r).tolist())
        cons.extend((W - x - r).tolist())
        cons.extend((y - r).tolist())
        cons.extend((H - y - r).tolist())
        
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                rr = r[i] + r[j]
                cons.append(dx*dx + dy*dy - rr*rr)
        
        return np.array(cons)
    
    ineq = NonlinearConstraint(ineq_func, 0, np.inf, nsls=100)
    eq = LinearConstraint(A_eq, b_eq, b_eq)
    
    result = minimize(
        obj, v0, method='SLSQP',
        bounds=bounds,
        constraints=[eq, ineq],
        options={'maxiter': 1000, 'ftol': 1e-12}
    )
    
    if result.success or result.fun < -0.1:
        W, H, data = unpack2(result.x)
        # Clamp to ensure feasibility
        x = np.clip(data[:, 0], 0, W)
        y = np.clip(data[:, 1], 0, H)
        r = np.clip(data[:, 2], 0, np.minimum(np.minimum(x, W-x), np.minimum(y, H-y)))
        
        # Ensure non-overlap by shrinking radii if needed
        # Simple iterative shrinking
        for iteration in range(50):
            violated = False
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    req = r[i] + r[j]
                    if dist < req:
                        # Shrink both radii proportionally
                        scale = dist / (req + 1e-10)
                        r[i] *= scale
                        r[j] *= scale
                        violated = True
            if not violated:
                break
        
        # Final containment check
        r = np.minimum(r, np.minimum(np.minimum(x, W-x), np.minimum(y, H-y)))
        r = np.maximum(r, 0)
        
        result_array = np.column_stack([x, y, r])
    else:
        # Fallback to grid
        side = 5
        W0 = 1.0
        H0 = 1.0
        result_array = []
        for i in range(N):
            col = i % side
            row = i // side
            cx = (col + 0.5) / side * W0
            cy = (row + 0.5) / side * H0
            r = 0.5 / side * W0 * 0.9
            result_array.append([cx, cy, r])
        result_array = np.array(result_array)
    
    return result_array
