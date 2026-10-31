import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    # Variables: x_1, y_1, r_1, ..., x_N, y_N, r_N
    # Total variables: 3 * N
    
    def objective(z):
        # z is shape (3*N,)
        # r_i are at indices 2*i + 2
        radii = z[2::3]
        return -np.sum(radii)
        
    def objective_grad(z):
        grad = np.zeros_like(z)
        # d/dz (-sum(r_i))
        # r_i is at 2*i + 2
        grad[2::3] = -1.0
        return grad

    # Constraints
    # 1. Boundary:
    # x_i >= r_i  => x_i - r_i >= 0
    # 1 - x_i >= r_i => 1 - x_i - r_i >= 0
    # y_i >= r_i => y_i - r_i >= 0
    # 1 - y_i >= r_i => 1 - y_i - r_i >= 0
    
    # 2. Non-overlap:
    # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    # g_ij = (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    
    # We will use NonlinearConstraint for all constraints to simplify, 
    # or split linear and nonlinear. 
    # trust-constr accepts a list of constraints.
    
    # Let's define the nonlinear constraint function for non-overlap.
    def non_overlap_cons(z):
        cons = []
        for i in range(N):
            for j in range(i + 1, N):
                xi = z[3*i]
                yi = z[3*i + 1]
                ri = z[3*i + 2]
                xj = z[3*j]
                yj = z[3*j + 1]
                rj = z[3*j + 2]
                val = (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
                cons.append(val)
        return np.array(cons)
        
    def non_overlap_jac(z):
        # Jacobian shape: (N*(N-1)/2, 3*N)
        jac = np.zeros((N*(N-1)//2, 3*N))
        k = 0
        for i in range(N):
            for j in range(i + 1, N):
                xi = z[3*i]
                yi = z[3*i + 1]
                ri = z[3*i + 2]
                xj = z[3*j]
                yj = z[3*j + 1]
                rj = z[3*j + 2]
                
                # d/dxi
                jac[k, 3*i] = 2 * (xi - xj)
                # d/dyi
                jac[k, 3*i + 1] = 2 * (yi - yj)
                # d/dri
                jac[k, 3*i + 2] = -2 * (ri + rj)
                
                # d/dxj
                jac[k, 3*j] = 2 * (xj - xi)
                # d/dyj
                jac[k, 3*j + 1] = 2 * (yj - yi)
                # d/drj
                jac[k, 3*j + 2] = -2 * (ri + rj)
                
                k += 1
        return jac

    # Linear constraints for boundaries
    # We can define them as LinearConstraint(A, lb, ub)
    # x_i - r_i >= 0
    # -x_i - r_i >= -1
    # y_i - r_i >= 0
    # -y_i - r_i >= -1
    
    A = np.zeros((4*N, 3*N))
    lb = np.zeros(4*N)
    ub = np.ones(4*N)
    
    for i in range(N):
        # x_i - r_i >= 0
        A[4*i, 3*i] = 1.0
        A[4*i, 3*i + 2] = -1.0
        lb[4*i] = 0.0
        ub[4*i] = np.inf
        
        # -x_i - r_i >= -1  => x_i + r_i <= 1
        A[4*i+1, 3*i] = -1.0
        A[4*i+1, 3*i + 2] = -1.0
        lb[4*i+1] = -1.0
        ub[4*i+1] = np.inf
        
        # y_i - r_i >= 0
        A[4*i+2, 3*i + 1] = 1.0
        A[4*i+2, 3*i + 2] = -1.0
        lb[4*i+2] = 0.0
        ub[4*i+2] = np.inf
        
        # -y_i - r_i >= -1 => y_i + r_i <= 1
        A[4*i+3, 3*i + 1] = -1.0
        A[4*i+3, 3*i + 2] = -1.0
        lb[4*i+3] = -1.0
        ub[4*i+3] = np.inf

    # Bounds for variables
    # x, y in [0, 1]
    # r in [0.01, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.01, 0.5)) # r

    constraints = [
        {'type': 'ineq', 'fun': non_overlap_cons, 'jac': non_overlap_jac},
        # Linear constraints are handled by bounds? No, they are linear inequalities.
        # trust-constr accepts LinearConstraint objects.
        # Let's use the dictionary format which is more flexible if jac is not provided for linear?
        # Actually, for linear constraints, we can just use 'type': 'ineq' with fun and jac.
        # But LinearConstraint is more efficient.
        # Let's stick to the dictionary format for simplicity, or use scipy's LinearConstraint class.
    ]
    
    # Using scipy.optimize.LinearConstraint
    from scipy.optimize import LinearConstraint
    
    lin_constraint = LinearConstraint(A, lb, ub)
    nonlin_constraint = {'type': 'ineq', 'fun': non_overlap_cons, 'jac': non_overlap_jac}
    
    all_constraints = [lin_constraint, nonlin_constraint]
    
    # Initial guesses
    def get_initial_guesses():
        guesses = []
        # 1. Grid
        side = int(np.ceil(np.sqrt(N)))
        # Create grid centers
        xs = np.linspace(0.5/side, 1 - 0.5/side, side)
        ys = np.linspace(0.5/side, 1 - 0.5/side, side)
        grid_pts = []
        for i in range(side):
            for j in range(side):
                grid_pts.append((xs[i], ys[j]))
        
        z0 = np.zeros(3*N)
        for i in range(N):
            x, y = grid_pts[i]
            z0[3*i] = x
            z0[3*i + 1] = y
            z0[3*i + 2] = 0.5/side - 0.01 # Initial radius
        guesses.append(z0)
        
        # 2. Random
        for _ in range(5):
            z0 = np.random.uniform(0.1, 0.9, size=N*2)
            r0 = np.full(N, 0.05)
            z0 = np.zeros(3*N)
            z0[0::3] = np.random.uniform(0.1, 0.9, N)
            z0[1::3] = np.random.uniform(0.1, 0.9, N)
            z0[2::3] = 0.05
            guesses.append(z0)
            
        return guesses

    best_val = -np.inf
    best_z = None
    
    start_time = time.time()
    time_limit = 100.0 # Leave margin for evaluation overhead
    
    for z0 in get_initial_guesses():
        if time.time() - start_time > time_limit:
            break
            
        try:
            res = minimize(
                objective, 
                z0, 
                method='trust-constr', 
                jac=objective_grad,
                bounds=bounds,
                constraints=all_constraints,
                options={'maxiter': 500, 'gtol': 1e-8, 'xtol': 1e-8, 'disp': False}
            )
            
            if res.success:
                # Check feasibility manually to be safe
                z_res = res.x
                # Check non-overlap
                feasible = True
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((z_res[3*i]-z_res[3*j])**2 + (z_res[3*i+1]-z_res[3*j+1])**2)
                        if dist < z_res[3*i+2] + z_res[3*j+2] - 1e-6:
                            feasible = False
                            break
                    if not feasible: break
                
                if feasible:
                    val = np.sum(z_res[2::3])
                    if val > best_val:
                        best_val = val
                        best_z = z_res
        except Exception as e:
            pass

    if best_z is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        xs = np.linspace(0.5/side, 1 - 0.5/side, side)
        ys = np.linspace(0.5/side, 1 - 0.5/side, side)
        grid_pts = []
        for i in range(side):
            for j in range(side):
                grid_pts.append((xs[i], ys[j]))
        
        z0 = np.zeros(3*N)
        for i in range(N):
            x, y = grid_pts[i]
            z0[3*i] = x
            z0[3*i + 1] = y
            z0[3*i + 2] = 0.5/side - 0.01
        best_z = z0

    # Format output: (x, y, radius)
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_z[3*i]
        result[i, 1] = best_z[3*i + 1]
        result[i, 2] = best_z[3*i + 2]
        
    return result
