import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s safety margin
    
    # Initial guess: uniform 0.5
    h0 = np.full(N, 0.5)
    t0 = 25.0 # Baseline score
    
    # We optimize h and t
    # x = [h_0, ..., h_{N-1}, t]
    
    def objective(x):
        return x[-1] # Minimize t
    
    def constraints(x):
        h = x[:-1]
        t = x[-1]
        
        cons = []
        
        # Sum constraint: sum(h) - N/2 = 0
        cons.append({'type': 'eq', 'fun': lambda h: np.sum(h) - N/2, 'jac': lambda h: np.ones(N)})
        
        # Overlap constraints: C(k) <= t  =>  C(k) - t <= 0
        # C(k) = sum_i h_i (1 - h_{i+k})
        # We need to compute C(k) for all k.
        # This is expensive inside the loop if done naively.
        # But N=100, so N^2 = 10000 ops. Fast enough.
        
        for k in range(N):
            # Shift h by k
            # h_shifted = np.roll(h, -k) ? No, linear shift with zero padding.
            # The problem says "shifted 1-h".
            # Usually in such contexts, if not specified, periodic is assumed for spectral analysis,
            # BUT the "overlap" definition usually implies the integral of product.
            # If periodic: sum_i h_i (1-h_{i+k mod N})
            # If linear: sum_{i=0}^{N-k-1} h_i (1-h_{i+k})
            
            # Let's check the seed score.
            # Seed h=0.5.
            # Periodic: C(k) = sum 0.25 = 25. Max 25.
            # Linear: C(k) = 0.25 (N-k). Max 25 (at k=0).
            # Both give 25.
            
            # Which one is intended?
            # "integral 1" on [0,2].
            # If we shift, do we wrap around?
            # Usually "overlap of h with shifted 1-h" in the context of "minimum overlap" problems (like C5 bound)
            # refers to the cyclic autocorrelation or the linear one?
            # The C5 bound in coding theory often involves cyclic shifts.
            # Let's assume periodic (cyclic) shifts first, as it's more symmetric and common in these "Erdos" type problems.
            # If I use linear, the k=0 term is dominant.
            
            # Let's try to solve for Cyclic first.
            # C(k) = sum_{i=0}^{N-1} h_i (1 - h_{(i+k)%N})
            
            # For efficiency, we can precompute the structure or just compute directly.
            # Since N is small, direct computation is fine.
            
            # However, defining a lambda for each constraint in a list is slow.
            # We should define the constraint functions once or compute the gradient.
            
            pass # Placeholder
        
        return cons

    # This approach of returning a list of constraints dynamically is slow for scipy.
    # Better to define the functions explicitly or use a single constraint function that returns a vector.
    
    def constraint_funcs(x):
        h = x[:-1]
        t = x[-1]
        
        # Equality: sum(h) - 50 = 0
        c_eq = np.sum(h) - 50
        
        # Inequalities: C(k) - t <= 0
        c_ineq = np.zeros(N)
        for k in range(N):
            h_shifted = np.roll(h, -k)
            # C(k) = sum h_i (1 - h_shifted_i)
            c_ineq[k] = np.sum(h * (1 - h_shifted)) - t
            
        return c_eq, c_ineq

    # SLSQP requires separate constraint dicts.
    # We can define them.
    
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x[:-1]) - N/2, 'jac': lambda x: np.append(np.ones(N), 0)}]
    
    # Add inequality constraints
    # We need to be careful with the Jacobian.
    
    def jac_ineq(k, x):
        h = x[:-1]
        t = x[-1]
        # C(k) = sum_i h_i - sum_i h_i h_{i+k}
        # d/dh_j C(k) = 1 - (h_{j-k} + h_{j+k})  (for cyclic)
        # d/dt C(k) = -1
        
        jac_h = np.zeros(N)
        # h_i appears in term i as h_i (coeff 1) and in term i-k as h_{i-k} (coeff -1 from -h_i h_{i-k}?)
        # C(k) = sum_i h_i - sum_i h_i h_{i+k}
        # Term for h_j:
        # From sum h_i: 1
        # From -sum h_i h_{i+k}:
        #   If i=j: -h_{j+k}
        #   If i+k=j => i=j-k: -h_{j-k}
        # So d/dh_j = 1 - h_{j+k} - h_{j-k}
        
        h_shifted_k = np.roll(h, -k)
        h_shifted_neg_k = np.roll(h, k)
        
        jac_h = 1 - h_shifted_k - h_shifted_neg_k
        jac_t = -1
        
        return np.append(jac_h, jac_t)

    for k in range(N):
        cons.append({
            'type': 'ineq',
            'fun': lambda x, k=k: np.sum(x[:-1] * (1 - np.roll(x[:-1], -k))) - x[-1],
            'jac': lambda x, k=k: jac_ineq(k, x)
        })

    # Bounds
    bounds = [(0, 1) for _ in range(N)] + [(0, None)] # t >= 0
    
    # Initial point
    x0 = np.append(h0, t0)
    
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 1000, 'ftol': 1e-9})
    
    if res.success:
        h_opt = res.x[:-1]
        # Ensure sum is exactly N/2
        h_opt = np.clip(h_opt, 0, 1)
        # Adjust sum
        diff = N/2 - np.sum(h_opt)
        if abs(diff) > 1e-6:
            # Distribute difference
            # Simple addition
            h_opt += diff / N
            h_opt = np.clip(h_opt, 0, 1)
            # Re-normalize if clipping changed sum
            # Iterate a few times
            for _ in range(5):
                diff = N/2 - np.sum(h_opt)
                h_opt += diff / N
                h_opt = np.clip(h_opt, 0, 1)
        
        return h_opt
    else:
        return h0
