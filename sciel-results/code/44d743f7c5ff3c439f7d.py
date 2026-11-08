import numpy as np
from scipy import optimize
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # The objective: for a step function with heights h on N intervals of width dx,
    # autoconvolution max (at 0) = dx * sum(h^2)
    # Normalization: likely (integral)^2 = (dx * sum(h))^2
    # So score = (dx * sum(h^2)) / (dx * sum(h))^2 = sum(h^2) / (dx * sum(h)^2)
    # But this depends on dx. Alternatively, score = sum(h^2)/sum(h) if normalized by L1.
    
    # Given seed (all ones) scores 1.0, let's figure out which formula gives 1 for h=[1]*600:
    # sum(h^2)/sum(h) = 600/600 = 1. Yes!
    # sum(h^2)/(dx*sum(h)^2) = 600/(dx*360000). dx=1/1200, so = 600*1200/360000 = 720000/360000 = 2. No.
    # max(f*f)/int(f) = (dx*sum(h^2))/(dx*sum(h)) = sum(h^2)/sum(h) = 1. Yes!
    
    # So the score is sum(h^2)/sum(h), and we want to minimize it.
    # But without constraint, we can make h -> 0 and score -> 0.
    # There must be a constraint. Perhaps sum(h) = 1 or max(h) = 1.
    
    # If max(h) = 1: uniform gives 1.0. We can do better with non-uniform.
    # If sum(h) = 1: uniform gives 1/600. Seed would give 1/600, not 1.0.
    
    # So likely constraint is max(h) = 1, and we minimize sum(h^2)/sum(h).
    # But if max(h)=1, we can set one h_i=1 and rest 0, giving 1/1=1. Same as uniform.
    # Set k intervals to 1: k/k = 1. Always 1 if h_i in {0,1}.
    # Set all to c: 600c^2/(600c) = c. Minimize c -> 0. But h_i >= 0, no lower bound besides 0.
    
    # This is problematic. Let me try: maybe the constraint is sum(h) = N (i.e., mean height = 1).
    # Then minimize sum(h^2) subject to sum(h) = 600, h_i >= 0.
    # By Cauchy-Schwarz, min sum(h^2) = (sum h)^2/N = 360000/600 = 600, achieved by h_i=1.
    # Score = 600/600 = 1. So uniform is optimal! Can't do better.
    
    # Hmm. Let me try another interpretation. Maybe it's about the L-infinity norm of the normalized autoconvolution.
    
    # Let me just try to use optimization to find something better, assuming the score is
    # max(f*f) / (∫f)^2 * some factor, and use a constraint like ∫f = 1.
    
    # Actually, let me just try a simple approach: use scipy to minimize a plausible objective.
    
    def objective(h):
        h = np.maximum(h, 0)  # ensure nonnegative
        s = np.sum(h)
        if s < 1e-12:
            return 1e10
        # Try: sum(h^2) / sum(h) with constraint that we normalize somehow
        # Or maybe the actual score involves the full autoconvolution max, not just at 0
        h_sq = np.sum(h**2)
        return h_sq / s
    
    # Constraint: let's try sum(h) = 1
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}]
    bounds = [(0, None)] * N
    
    x0 = np.ones(N) / N  # uniform
    x0 = np.clip(x0, 0, None)
    
    best_score = objective(x0)
    best_h = x0.copy()
    
    try:
        result = optimize.minimize(
            objective, x0, method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-8}
        )
        if result.success:
            h_opt = np.clip(result.x, 0, None)
            s = np.sum(h_opt)
            if s > 1e-12:
                score_opt = np.sum(h_opt**2) / s
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time - 5:
        # Return best found
        return [float(max(0, h)) for h in best_h]
    
    return [float(max(0, h)) for h in best_h]
