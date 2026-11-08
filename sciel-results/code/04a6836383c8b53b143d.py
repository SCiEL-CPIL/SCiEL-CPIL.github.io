"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    n = N
    # Initial guess: grid placement
    side = int(np.ceil(np.sqrt(n)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < n:
                positions.append([(i + 0.5) / side, (j + 0.5) / side])
    
    # Initial radii: small equal radii
    r0 = 0.5 / side
    radii = np.full(n, r0)
    
    # Variables: x_1, y_1, r_1, x_2, y_2, r_2, ...
    # Total variables: 3 * n
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = positions[i][0]
        x0[3*i+1] = positions[i][1]
        x0[3*i+2] = radii[i]
    
    # Bounds: x in [0,1], y in [0,1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((1e-6, 0.5)) # r
    
    def objective_and_grad(z):
        # z: (3*n,)
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        # Objective: minimize negative sum of radii
        obj = -np.sum(r)
        
        # Gradient of objective
        grad = np.zeros(3 * n)
        grad[2::3] = -1.0
        
        # Constraints:
        # 1. Circle-boundary: x_i >= r_i, x_i + r_i <= 1, y_i >= r_i, y_i + r_i <= 1
        #    These are handled by bounds partially, but let's add explicit constraints for safety
        #    Actually, bounds handle x in [0,1] and r in [0,0.5], but we need x >= r and x <= 1-r
        #    Let's add these as inequality constraints: g(z) >= 0
        
        # For each circle i:
        # x_i - r_i >= 0
        # 1 - x_i - r_i >= 0
        # y_i - r_i >= 0
        # 1 - y_i - r_i >= 0
        
        # 2. Circle-circle: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        #    g_ij = (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        
        # We'll use a penalty approach or add constraints.
        # With L-BFGS-B, we can't easily add many nonlinear constraints.
        # Instead, let's use a penalty method or reformulate.
        
        # Alternative: Use a barrier/penalty for violations
        # Let's add a large penalty for constraint violations to the objective
        
        penalty = 0.0
        penalty_grad = np.zeros(3 * n)
        
        # Boundary constraints (redundant with bounds but let's be safe)
        # Actually, let's just rely on bounds for x,y in [0,1] and r > 0
        # And add circle-circle constraints as penalty
        
        # Circle-circle constraints
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dr = r[i] + r[j]
                dist_sq = dx*dx + dy*dy
                if dist_sq < 1e-12:
                    dist = 1e-6
                    # Gradient w.r.t. x_i, y_i: -2*dx/dist, -2*dy/dist
                    # Gradient w.r.t. x_j, y_j: 2*dx/dist, 2*dy/dist
                    # Gradient w.r.t. r_i, r_j: -2*dr
                    # Penalty: (dr^2 - dist_sq) if dist_sq < dr^2
                    # Let's use a smooth penalty
                    viol = dr*dr - dist_sq
                    if viol > 0:
                        penalty += 1000 * viol * viol
                        # Gradient of penalty w.r.t. variables
                        # d/dx_i [1000 * (dr^2 - dist_sq)^2] = 1000 * 2 * viol * d/dr^2(-dist_sq)
                        # d(dist_sq)/dx_i = -2*dx
                        # So d(-dist_sq)/dx_i = 2*dx
                        # d/dr^2 = -d(dist_sq)/dr^2 ... wait
                        # Let f = (dr^2 - dist_sq)^2
                        # df/dx_i = 2*(dr^2 - dist_sq) * (-d(dist_sq)/dx_i) = 2*viol * (2*dx) = 4*viol*dx
                        # df/dx_j = 2*viol * (-2*dy... wait d(dist_sq)/dx_j = 2*dx_j - 2*x_i... no
                        # dist_sq = (x_i - x_j)^2 + (y_i - y_j)^2
                        # d(dist_sq)/dx_i = 2*(x_i - x_j) = 2*dx
                        # d(dist_sq)/dx_j = -2*(x_i - x_j) = -2*dx
                        # So d(-dist_sq)/dx_i = -2*dx
                        # df/dx_i = 2*viol * (-2*dx) = -4*viol*dx
                        # Wait, let me recompute.
                        # f = (A - B)^2 where A = dr^2, B = dist_sq
                        # df/dx_i = 2*(A-B) * d(A-B)/dx_i = 2*viol * (0 - dB/dx_i) = 2*viol * (-2*dx) = -4*viol*dx
                        # df/dx_j = 2*viol * (0 - dB/dx_j) = 2*viol * (2*dx) = 4*viol*dx
                        # df/dr_i = 2*viol * (2*r_i - 0) = 4*viol*r_i  (since d(A)/dr_i = 2*r_i)
                        # df/dr_j = 4*viol*r_j
                        
                        penalty_grad[3*i] += 1000 * (-4 * viol * dx)
                        penalty_grad[3*j] += 1000 * (4 * viol * dx)
                        penalty_grad[3*i+2] += 1000 * (4 * viol * r[i])
                        penalty_grad[3*j+2] += 1000 * (4 * viol * r[j])
                else:
                    dist = np.sqrt(dist_sq)
                    viol = dr*dr - dist_sq
                    if viol > 0:
                        penalty += 1000 * viol * viol
                        # Same gradients as above
                        penalty_grad[3*i] += 1000 * (-4 * viol * dx)
                        penalty_grad[3*j] += 1000 * (4 * viol * dx)
                        penalty_grad[3*i+1] += 1000 * (-4 * viol * dy)
                        penalty_grad[3*j+1] += 1000 * (4 * viol * dy)
                        penalty_grad[3*i+2] += 1000 * (4 * viol * r[i])
                        penalty_grad[3*j+2] += 1000 * (4 * viol * r[j])
        
        total_obj = obj + penalty
        total_grad = grad + penalty_grad
        
        return total_obj, total_grad
    
    # Use L-BFGS-B
    result = minimize(
        lambda z: objective_and_grad(z)[0],
        x0,
        jac=lambda z: objective_and_grad(z)[1],
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    # Extract result
    z = result.x
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]
    
    # Ensure circles are inside the square (clip if needed)
    # x >= r, x <= 1-r, y >= r, y <= 1-r
    # If violated, adjust r or position
    for i in range(n):
        # Ensure r is positive
        r[i] = max(r[i], 1e-6)
        # Ensure x, y within [r, 1-r]
        x[i] = np.clip(x[i], r[i], 1 - r[i])
        y[i] = np.clip(y[i], r[i], 1 - r[i])
    
    # Build output array
    result_array = np.zeros((n, 3))
    for i in range(n):
        result_array[i] = [x[i], y[i], r[i]]
    
    return result_array

# EVOLVE-BLOCK-END
