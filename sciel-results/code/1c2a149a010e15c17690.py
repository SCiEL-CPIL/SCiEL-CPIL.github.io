import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Objective and constraints for SLSQP
    # x = [h_0, ..., h_{N-1}, t]
    
    def objective(x):
        return x[-1] # t
        
    def constraints(x):
        h = x[:-1]
        t = x[-1]
        cons = []
        # Equality: sum h = N/2
        cons.append({'type': 'eq', 'fun': lambda x: np.sum(x[:-1]) - N/2})
        # Inequalities: overlap_k <= t
        for k in range(N):
            # Calculate overlap for shift k
            # Sum_{i=0}^{N-1-k} h_i (1 - h_{i+k})
            # Using vectorization
            # h_i is x[i], h_{i+k} is x[i+k]
            # We can compute all overlaps at once if we want, but constraints need individual functions
            # To speed up, we can compute the max overlap in the objective? No, epigraph needs individual constraints.
            # However, SLSQP evaluates constraints one by one or in a vector?
            # It's better to provide a single constraint function that returns a vector if possible?
            # scipy minimize constraints can be a list of dicts.
            # Each dict has a 'fun'.
            # We can define the 'fun' for each k.
            
            # Optimization: Pre-compute indices?
            # For k=0: sum(h * (1-h))
            # For k=1: sum(h[:-1] * (1-h[1:]))
            # ...
            
            # This loop inside constraints is slow.
            # Alternative: Use a single constraint function that returns an array?
            # scipy.optimize.minimize with SLSQP accepts constraints as a list.
            # It does NOT accept a single function returning an array of inequalities directly in the standard dict format for SLSQP?
            # Actually, you can pass a list of constraints.
            # To make it faster, we can try to reduce the number of constraints?
            # Or use a penalty method?
            # Or use `minimize` with `method='SLSQP'` and define the constraints carefully.
            
            # Let's try to vectorize the constraint check in a single function that returns a vector of violations.
            # But SLSQP expects a list of constraint dicts.
            
            # Maybe we can just use a penalty approach in the objective?
            # Minimize t + penalty * max(0, overlap_k - t)?
            # This is non-smooth.
            
            # Let's stick to SLSQP with list of constraints.
            # N=100 constraints. This might be slow.
            # 100 constraints x 100 variables.
            # Hessian approximation might be expensive.
            
            # Alternative: Use `scipy.optimize.minimize` with `method='SLSQP'` is standard.
            # We can try to limit the number of restarts.
            
            pass # Placeholder
            
    # We need to implement the constraints properly.
    
    def make_constraint(k):
        def con(x):
            h = x[:-1]
            t = x[-1]
            if k == 0:
                overlap = np.sum(h * (1 - h))
            else:
                overlap = np.sum(h[:-k] * (1 - h[k:]))
            return overlap - t
        return {'type': 'ineq', 'fun': con} # Note: ineq means fun >= 0. So t - overlap >= 0.
        # Wait, SLSQP 'ineq' means fun(x) >= 0.
        # We want overlap <= t => t - overlap >= 0.
        
    # This creates N constraint functions.
    
    # Let's implement the full loop.
    
    best_h = np.full(N, 0.5)
    best_score = 25.0 # Seed value
    
    def evaluate(h):
        # Calculate max overlap
        max_ov = 0
        for k in range(N):
            if k == 0:
                ov = np.sum(h * (1 - h))
            else:
                ov = np.sum(h[:-k] * (1 - h[k:]))
            if ov > max_ov:
                max_ov = ov
        return max_ov
        
    # Initializations
    inits = []
    
    # 1. Random
    for _ in range(5):
        h_init = np.random.rand(N)
        # Normalize to sum 50
        h_init = h_init / np.sum(h_init) * 50
        h_init = np.clip(h_init, 0, 1)
        # Re-normalize after clip? SLSQP will handle it, but good start helps.
        inits.append(h_init)
        
    # 2. Sine
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    h_sine = 0.5 + 0.5 * np.sin(x)
    h_sine = h_sine / np.sum(h_sine) * 50
    inits.append(h_sine)
    
    # 3. Gaussian
    center = N/2
    x_idx = np.arange(N)
    sigma = 10
    h_gauss = np.exp(-((x_idx - center)**2) / (2*sigma**2))
    h_gauss = h_gauss / np.sum(h_gauss) * 50
    inits.append(h_gauss)
    
    # 4. Two blocks (1s at edges)
    h_two = np.zeros(N)
    h_two[:25] = 1
    h_two[75:] = 1
    inits.append(h_two)
    
    # Run SLSQP for each init
    for h_init in inits:
        if time.time() > deadline - 10: break
        
        x0 = np.append(h_init, 25.0) # t init
        
        constraints = []
        constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x[:-1]) - N/2})
        
        for k in range(N):
            constraints.append(make_constraint(k))
            
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=[(0,1)]*N + [(0, N)], constraints=constraints, options={'maxiter': 500, 'ftol': 1e-6})
            if res.success:
                h_res = res.x[:-1]
                # Project to sum 50 exactly
                diff = 50 - np.sum(h_res)
                h_res = h_res + diff / N
                h_res = np.clip(h_res, 0, 1)
                # Re-project
                h_res = h_res / np.sum(h_res) * 50
                
                ov = evaluate(h_res)
                if ov < best_score:
                    best_score = ov
                    best_h = h_res
        except Exception as e:
            pass
            
    return best_h
