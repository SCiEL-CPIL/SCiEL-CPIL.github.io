import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_value = -np.inf
    best_config = None

    # Initial configurations
    side = int(np.ceil(np.sqrt(N)))
    grid_cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01) for i in range(side) for j in range(side)]
    inits = [np.array(grid_cells[:N])]
    
    # Add a few random inits
    for _ in range(3):
        np.random.seed(np.random.randint(10000))
        xs = np.random.rand(N)
        ys = np.random.rand(N)
        rs = np.full(N, 0.1) # Start with reasonable radius
        inits.append(np.column_stack((xs, ys, rs)))

    def objective(z):
        r = z[2::3]
        return -np.sum(r)

    def gradient(z):
        # Gradient of -sum(r)
        grad = np.zeros_like(z)
        grad[2::3] = -1.0
        return grad

    # Constraints
    # We will define a single function that returns the list of constraint values
    # SLSQP expects a list of dicts with 'fun' and 'type'
    
    def make_constraints():
        cons = []
        # Boundary constraints
        # x - r >= 0
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[i*3] - z[i*3+2], 'jac': lambda z, i=i: np.array([1.0, 0.0, -1.0, 0.0, 0.0, 0.0])}) # This jac is wrong, needs to be full vector
            # Better to define jac properly
        # This manual constraint definition is tedious and error-prone.
        # Let's use a single constraint function that returns an array of values for all constraints.
        # SLSQP supports this? No, it expects a list of constraints.
        # However, we can define a function that returns all constraint values and use a wrapper.
        # Or just define them individually.
        
        # Let's define the constraints as a single array-returning function and use a helper.
        pass

    # Alternative: Use L-BFGS-B with penalty, which is easier to vectorize.
    # The previous L-BFGS-B approach was good. Let's stick to it but refine.
    
    def fast_objective(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        obj = -np.sum(r)
        
        eps = 1e-4
        def smooth_max(u):
            return 0.5 * (u + np.sqrt(u*u + eps*eps))
            
        # Boundary
        v_left = r - x
        v_right = r - (1.0 - x)
        v_bottom = r - y
        v_top = r - (1.0 - y)
        
        pen = np.sum(smooth_max(v_left)**2) + np.sum(smooth_max(v_right)**2) + \
              np.sum(smooth_max(v_bottom)**2) + np.sum(smooth_max(v_top)**2)
        
        # Circle-Circle
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx*dx + dy*dy + 1e-12)
        r_sum = r[:, None] + r[None, :]
        
        v = r_sum - d
        idx = np.triu_indices(N, 1)
        v_upper = v[idx]
        pen += np.sum(smooth_max(v_upper)**2)
        
        weight = 500.0 # Increased weight to enforce constraints more strictly
        return obj + weight * pen

    for init in inits:
        if time.time() > deadline:
            break
            
        # Use L-BFGS-B
        res = minimize(fast_objective, init, method='L-BFGS-B', 
                       bounds=[(0, 1)] * (2*N) + [(0, 0.5)] * N,
                       options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.fun < best_value:
            z = res.x
            x = z[0::3]
            y = z[1::3]
            r = z[2::3]
            
            # Check validity with strict tolerance
            valid = True
            for i in range(N):
                if x[i] < r[i] - 1e-5 or x[i] > 1 - r[i] + 1e-5 or \
                   y[i] < r[i] - 1e-5 or y[i] > 1 - r[i] + 1e-5:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        d = np.sqrt(dx*dx + dy*dy)
                        if d < r[i] + r[j] - 1e-5:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                val = np.sum(r)
                if val > best_value:
                    best_value = val
                    best_config = np.column_stack((x, y, r))

    if best_config is None:
        # Fallback
        return np.array(inits[0])
        
    return best_config
