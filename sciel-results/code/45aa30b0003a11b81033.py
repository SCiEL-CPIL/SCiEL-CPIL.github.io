"""Improved circle packing using trust-constr NLP with explicit constraints."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint, NonlinearConstraint
import itertools

def _initial_guess():
    """Generate an initial guess with circles in corners and center."""
    # Place circles in a pattern that uses corners effectively
    # Corners: 4, edges: 8, interior: 14
    positions = []
    radii = []
    
    # 4 corner circles
    corner_r = 0.18
    corners = [(corner_r, corner_r), (1-corner_r, corner_r), 
               (corner_r, 1-corner_r), (1-corner_r, 1-corner_r)]
    for x, y in corners:
        positions.append((x, y))
        radii.append(corner_r)
    
    # 8 edge circles (2 per edge, avoiding corners)
    edge_r = 0.12
    # Top edge
    positions.extend([(0.3, 1-edge_r), (0.7, 1-edge_r)])
    radii.extend([edge_r, edge_r])
    # Bottom edge
    positions.extend([(0.3, edge_r), (0.7, edge_r)])
    radii.extend([edge_r, edge_r])
    # Left edge
    positions.extend([(edge_r, 0.3), (edge_r, 0.7)])
    radii.extend([edge_r, edge_r])
    # Right edge
    positions.extend([(1-edge_r, 0.3), (1-edge_r, 0.7)])
    radii.extend([edge_r, edge_r])
    
    # 14 interior circles in a grid-like pattern
    interior_r = 0.08
    # 3x3 grid in the center region, plus 1 extra
    cx, cy = 0.5, 0.5
    spacing = 0.22
    for i in range(-1, 2):
        for j in range(-1, 2):
            if len(positions) < N:
                positions.append((cx + i*spacing, cy + j*spacing))
                radii.append(interior_r)
    
    # Add remaining circles if needed
    while len(positions) < N:
        positions.append((0.5, 0.5))
        radii.append(0.05)
    
    # Flatten to array: [x0, y0, r0, x1, y1, r1, ...]
    x = np.array([p[0] for p in positions[:N]])
    y = np.array([p[1] for p in positions[:N]])
    r = np.array(radii[:N])
    
    return np.concatenate([x, y, r])


def _objective(x):
    """Negative sum of radii (we minimize negative to maximize)."""
    r = x[2*N:]
    return -np.sum(r)


def _pairwise_constraints(x, eps=1e-8):
    """
    Constraints: for each pair (i,j), distance >= ri + rj
    Boundary: ri <= xi <= 1-ri, ri <= yi <= 1-ri
    All as g(x) >= 0 for trust-constr
    """
    x_c = x[:N]
    y_c = x[N:2*N]
    r = x[2*N:]
    
    constraints = []
    
    # Boundary constraints
    for i in range(N):
        # ri <= xi
        constraints.append(x_c[i] - r[i])
        # xi <= 1 - ri
        constraints.append(1 - x_c[i] - r[i])
        # ri <= yi
        constraints.append(y_c[i] - r[i])
        # yi <= 1 - ri
        constraints.append(1 - y_c[i] - r[i])
        # ri >= 0 (small positive)
        constraints.append(r[i])
    
    # Pairwise distance constraints
    for i in range(N):
        for j in range(i+1, N):
            dx = x_c[i] - x_c[j]
            dy = y_c[i] - y_c[j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            # dist >= min_dist  =>  dist^2 >= min_dist^2
            constraints.append(dist_sq - min_dist*min_dist)
    
    return np.array(constraints)


def _gradient_objective(x):
    """Gradient of objective: [0, 0, -1, -1, -1, ...]"""
    grad = np.zeros(3*N)
    grad[2*N:] = -1.0
    return grad


def _jacobian_constraints(x):
    """Jacobian of all constraints."""
    x_c = x[:N]
    y_c = x[N:2*N]
    r = x[2*N:]
    
    n_constraints = 5*N + N*(N-1)//2
    jac = np.zeros((n_constraints, 3*N))
    
    idx = 0
    # Boundary constraints
    for i in range(N):
        # x_i - r_i >= 0
        jac[idx, i] = 1.0
        jac[idx, 2*N+i] = -1.0
        idx += 1
        # 1 - x_i - r_i >= 0
        jac[idx, i] = -1.0
        jac[idx, 2*N+i] = -1.0
        idx += 1
        # y_i - r_i >= 0
        jac[idx, N+i] = 1.0
        jac[idx, 2*N+i] = -1.0
        idx += 1
        # 1 - y_i - r_i >= 0
        jac[idx, N+i] = -1.0
        jac[idx, 2*N+i] = -1.0
        idx += 1
        # r_i >= 0
        jac[idx, 2*N+i] = 1.0
        idx += 1
    
    # Pairwise constraints
    for i in range(N):
        for j in range(i+1, N):
            dx = x_c[i] - x_c[j]
            dy = y_c[i] - y_c[j]
            min_dist = r[i] + r[j]
            
            # d/dx_i: 2*dx
            jac[idx, i] = 2*dx
            # d/dx_j: -2*dx
            jac[idx, j] = -2*dx
            # d/dy_i: 2*dy
            jac[idx, N+i] = 2*dy
            # d/dy_j: -2*dy
            jac[idx, N+j] = -2*dy
            # d/dr_i: -2*min_dist
            jac[idx, 2*N+i] = -2*min_dist
            # d/dr_j: -2*min_dist
            jac[idx, 2*N+j] = -2*min_dist
            idx += 1
    
    return jac


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 115s safety margin
    
    x0 = _initial_guess()
    
    # Bounds for each variable
    # x_i, y_i in [0, 1], r_i in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0, 1))      # x_i
    for i in range(N):
        bounds.append((0, 1))      # y_i
    for i in range(N):
        bounds.append((0.01, 0.5)) # r_i
    
    # Constraint function and its Jacobian
    n_constraints = 5*N + N*(N-1)//2
    constraint = NonlinearConstraint(
        _pairwise_constraints,
        0, np.inf,
        jac=_jacobian_constraints
    )
    
    # Try multiple restarts with different initializations
    best_x = x0.copy()
    best_obj = _objective(x0)
    
    # Initial solve
    try:
        res = minimize(
            _objective,
            x0,
            method='trust-constr',
            jac=_gradient_objective,
            bounds=bounds,
            constraints=[constraint],
            options={'maxiter': 500, 'gtol': 1e-8, 'xtol': 1e-10}
        )
        if res.success or (res.fun < best_obj):
            if _pairwise_constraints(res.x).min() > -1e-6:  # Feasibility check
                best_x = res.x.copy()
                best_obj = res.fun
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    
    # If time remains, try a few more restarts with perturbed initializations
    if elapsed < deadline - 30:
        rng = np.random.default_rng(42)
        for restart in range(3):
            if time.time() > deadline - 15:
                break
            
            # Perturb initial guess
            x_pert = x0.copy()
            x_pert[:N] += rng.normal(0, 0.05, N)
            x_pert[N:2*N] += rng.normal(0, 0.05, N)
            x_pert[2*N:] *= (1 + rng.normal(0, 0.1, N))
            x_pert = np.clip(x_pert, 0.01, 0.99)
            x_pert[2*N:] = np.clip(x_pert[2*N:], 0.01, 0.5)
            
            try:
                res = minimize(
                    _objective,
                    x_pert,
                    method='trust-constr',
                    jac=_gradient_objective,
                    bounds=bounds,
                    constraints=[constraint],
                    options={'maxiter': 400, 'gtol': 1e-8, 'xtol': 1e-10}
                )
                if (res.fun < best_obj) and (_pairwise_constraints(res.x).min() > -1e-6):
                    best_x = res.x.copy()
                    best_obj = res.fun
            except Exception:
                pass
    
    # Final polish: run SLSQP on the best result found
    if time.time() < deadline - 5:
        try:
            # Convert to SLSQP format
            def slsqp_objective(x):
                return -np.sum(x[2*N:])
            
            def slsqp_constraints(x):
                cons = []
                x_c = x[:N]
                y_c = x[N:2*N]
                r = x[2*N:]
                
                # Boundary
                for i in range(N):
                    cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*N+i]})
                    cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[i] - x[2*N+i]})
                    cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[N+i] - x[2*N+i]})
                    cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[N+i] - x[2*N+i]})
                
                # Pairwise
                for i in range(N):
                    for j in range(i+1, N):
                        cons.append({
                            'type': 'ineq',
                            'fun': lambda x, i=i, j=j: 
                                (x[i]-x[j])**2 + (x[N+i]-x[N+j])**2 - (x[2*N+i]+x[2*N+j])**2
                        })
                return cons
            
            slsqp_bounds = [(0, 1)]*N + [(0, 1)]*N + [(0.01, 0.5)]*N
            
            res = minimize(
                slsqp_objective,
                best_x,
                method='SLSQP',
                bounds=slsqp_bounds,
                constraints=slsqp_constraints(),
                options={'maxiter': 300, 'ftol': 1e-12}
            )
            
            if res.fun < best_obj and _pairwise_constraints(res.x).min() > -1e-6:
                best_x = res.x.copy()
                best_obj = res.fun
        except Exception:
            pass
    
    # Format output
    x_c = best_x[:N]
    y_c = best_x[N:2*N]
    r = best_x[2*N:]
    
    result = np.column_stack([x_c, y_c, r])
    return result
# EVOLVE-BLOCK-END
