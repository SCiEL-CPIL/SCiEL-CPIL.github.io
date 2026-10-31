"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    # Initial guess: Grid
    side = int(np.ceil(np.sqrt(N)))
    # Create a grid of points
    x_grid, y_grid = np.meshgrid(np.linspace(0.5/side, 1 - 0.5/side, side), 
                                 np.linspace(0.5/side, 1 - 0.5/side, side))
    x_init = x_grid.flatten()[:N]
    y_init = y_grid.flatten()[:N]
    
    # Initial radii: small enough to not overlap in grid
    # Grid spacing is 1/side. Max radius approx 0.5/side
    r_init = np.full(N, 0.5/side * 0.9)
    
    # Flatten to 1D vector [x1, y1, r1, x2, y2, r2, ...]
    z0 = np.zeros(3*N)
    for i in range(N):
        z0[3*i] = x_init[i]
        z0[3*i+1] = y_init[i]
        z0[3*i+2] = r_init[i]
        
    def objective(z):
        # Maximize sum of radii -> Minimize negative sum
        return -np.sum(z[2::3])
        
    def objective_grad(z):
        grad = np.zeros_like(z)
        grad[2::3] = -1.0
        return grad
        
    # Constraints
    # Inequality constraints: c(z) >= 0
    
    def constraint_boundary(z):
        # x_i - r_i >= 0
        # 1 - x_i - r_i >= 0
        # y_i - r_i >= 0
        # 1 - y_i - r_i >= 0
        # r_i >= 0
        
        # Extract
        xs = z[0::3]
        ys = z[1::3]
        rs = z[2::3]
        
        c1 = xs - rs
        c2 = 1.0 - xs - rs
        c3 = ys - rs
        c4 = 1.0 - ys - rs
        c5 = rs
        
        return np.concatenate([c1, c2, c3, c4, c5])
        
    def constraint_boundary_jac(z):
        # Jacobian of boundary constraints
        # Rows correspond to constraints in order:
        # 0..N-1: xs - rs
        # N..2N-1: 1 - xs - rs
        # 2N..3N-1: ys - rs
        # 3N..4N-1: 1 - ys - rs
        # 4N..5N-1: rs
        
        jac = np.zeros((5*N, 3*N))
        
        # c1: xs - rs
        # d/dx = 1, d/dy = 0, d/dr = -1
        for i in range(N):
            jac[i, 3*i] = 1.0
            jac[i, 3*i+2] = -1.0
            
        # c2: 1 - xs - rs
        # d/dx = -1, d/dy = 0, d/dr = -1
        for i in range(N):
            jac[N+i, 3*i] = -1.0
            jac[N+i, 3*i+2] = -1.0
            
        # c3: ys - rs
        # d/dx = 0, d/dy = 1, d/dr = -1
        for i in range(N):
            jac[2*N+i, 3*i+1] = 1.0
            jac[2*N+i, 3*i+2] = -1.0
            
        # c4: 1 - ys - rs
        # d/dx = 0, d/dy = -1, d/dr = -1
        for i in range(N):
            jac[3*N+i, 3*i+1] = -1.0
            jac[3*N+i, 3*i+2] = -1.0
            
        # c5: rs
        # d/dx = 0, d/dy = 0, d/dr = 1
        for i in range(N):
            jac[4*N+i, 3*i+2] = 1.0
            
        return jac

    def constraint_pairwise(z):
        # (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
        xs = z[0::3]
        ys = z[1::3]
        rs = z[2::3]
        
        c = np.zeros(N*(N-1)//2)
        idx = 0
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dr = rs[i] + rs[j]
                c[idx] = dx*dx + dy*dy - dr*dr
                idx += 1
        return c

    def constraint_pairwise_jac(z):
        # Jacobian of pairwise constraints
        # Num constraints: M = N*(N-1)/2
        # Each constraint k corresponds to pair (i, j)
        
        M = N*(N-1)//2
        jac = np.zeros((M, 3*N))
        
        idx = 0
        for i in range(N):
            for j in range(i+1, N):
                xi = z[3*i]
                yi = z[3*i+1]
                ri = z[3*i+2]
                
                xj = z[3*j]
                yj = z[3*j+1]
                rj = z[3*j+2]
                
                dx = xi - xj
                dy = yi - yj
                dr = ri + rj
                
                # c = dx^2 + dy^2 - dr^2
                # d/dxi = 2*dx
                # d/dyi = 2*dy
                # d/dri = -2*dr
                # d/dxj = -2*dx
                # d/dyj = -2*dy
                # d/drj = -2*dr
                
                jac[idx, 3*i] = 2*dx
                jac[idx, 3*i+1] = 2*dy
                jac[idx, 3*i+2] = -2*dr
                
                jac[idx, 3*j] = -2*dx
                jac[idx, 3*j+1] = -2*dy
                jac[idx, 3*j+2] = -2*dr
                
                idx += 1
                
        return jac

    constraints = [
        {'type': 'ineq', 'fun': constraint_boundary, 'jac': constraint_boundary_jac},
        {'type': 'ineq', 'fun': constraint_pairwise, 'jac': constraint_pairwise_jac}
    ]
    
    # Bounds for variables to keep them in reasonable range
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((1e-6, 0.5)) # r

    # Run optimization
    # trust-constr is good for nonlinear constraints
    try:
        res = minimize(objective, z0, method='trust-constr', jac=objective_grad,
                       constraints=constraints, bounds=bounds,
                       options={'maxiter': 500, 'gtol': 1e-8, 'verbose': 0})
        
        if res.success or res.fun < objective(z0):
            z_opt = res.x
        else:
            z_opt = z0
            
    except Exception:
        z_opt = z0
        
    # Extract results
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = z_opt[3*i]
        result[i, 1] = z_opt[3*i+1]
        result[i, 2] = z_opt[3*i+2]
        
    return result
# EVOLVE-BLOCK-END
