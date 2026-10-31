import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    def compute_overlaps(h):
        """
        Compute the overlap for all shifts k=0..N-1.
        Overlap_k = sum_{i=0}^{N-1-k} h_i (1 - h_{i+k})
        """
        overlaps = np.zeros(N)
        for k in range(N):
            if k == 0:
                overlaps[k] = np.sum(h * (1 - h))
            else:
                overlaps[k] = np.sum(h[:-k] * (1 - h[k:]))
        return overlaps

    def objective_and_grad(x, beta, lam):
        h = x
        overlaps = compute_overlaps(h)
        
        # LogSumExp: (1/beta) * log(sum(exp(beta * overlaps)))
        # To avoid overflow, subtract max
        max_ov = np.max(overlaps)
        lse = max_ov + np.log(np.sum(np.exp(beta * (overlaps - max_ov)))) / beta
        
        # Penalty for sum constraint
        sum_h = np.sum(h)
        penalty = lam * (sum_h - N/2.0)**2
        
        val = lse + penalty
        
        # Gradient
        # d/dh_i of LSE:
        # p_k = exp(beta * (overlap_k - max_ov)) / sum(exp(...))
        # grad_lse_i = sum_k p_k * d(overlap_k)/dh_i
        
        w = np.exp(beta * (overlaps - max_ov))
        sum_w = np.sum(w)
        if sum_w == 0:
            p = np.ones(N) / N
        else:
            p = w / sum_w
            
        grad_h = np.zeros(N)
        
        # For each shift k, contribution to grad_h[i] is p[k] * (1 - h[i+k]) for the h[i] term
        # and p[k] * (-h[i-k]) for the h[i] term coming from h[i-k](1-h[i])
        
        # Vectorized approach:
        # For fixed k, d(overlap_k)/dh = (1 - shift(h, k)) - shift(h, -k)
        # where shift(h, k)_i = h_{i+k} if i+k < N else 0
        # and shift(h, -k)_i = h_{i-k} if i-k >= 0 else 0
        
        # We can loop over k since N=100 is small, and vectorize over i.
        for k in range(N):
            if p[k] < 1e-12: continue
            # d(overlap_k)/dh
            # Term 1: (1 - h_{i+k}) for i in 0..N-1-k
            # Term 2: -h_{i-k} for i in k..N-1
            
            d_ov = np.zeros(N)
            if k == 0:
                # d/dh_i (h_i - h_i^2) = 1 - 2h_i
                d_ov[:] = 1 - 2*h
            else:
                # h_i (1 - h_{i+k}) -> deriv wrt h_i is (1 - h_{i+k})
                d_ov[:N-k] += (1 - h[k:])
                # h_{i-k} (1 - h_i) -> deriv wrt h_i is -h_{i-k}
                d_ov[k:] += (-h[:N-k])
            
            grad_h += p[k] * d_ov
        
        # grad_lse = grad_h (as derived above)
        
        # Gradient of penalty: 2 * lam * (sum_h - N/2)
        grad_penalty = 2 * lam * (sum_h - N/2) * np.ones(N)
        
        grad = grad_h + grad_penalty
        
        return val, grad

    def project_sum(h, target=None):
        if target is None:
            target = N / 2.0
        diff = target - np.sum(h)
        if abs(diff) > 1e-12:
            h = h + diff / N
            h = np.clip(h, 0.0, 1.0)
            # Re-project if clip changed sum
            diff2 = target - np.sum(h)
            if abs(diff2) > 1e-12:
                # Simple iterative projection
                for _ in range(10):
                    h = h + diff2 / N
                    h = np.clip(h, 0.0, 1.0)
                    diff2 = target - np.sum(h)
                    if abs(diff2) < 1e-12:
                        break
        return h

    best_h = np.full(N, 0.5)
    best_score = float('inf')
    
    # Initializations
    inits = [np.full(N, 0.5)]
    
    # Random inits
    for _ in range(5):
        h_init = np.random.rand(N)
        h_init = project_sum(h_init)
        inits.append(h_init)
        
    # Sine init
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    h_sine = 0.5 + 0.5 * np.sin(x)
    h_sine = project_sum(h_sine)
    inits.append(h_sine)

    # Continuation parameters
    betas = [10.0, 50.0, 100.0, 500.0, 1000.0]
    lam = 100.0 # Penalty weight
    
    for h_init in inits:
        if time.time() > deadline - 20:
            break
            
        current_h = h_init.copy()
        
        for beta in betas:
            if time.time() > deadline - 10:
                break
                
            res = minimize(
                objective_and_grad, 
                current_h, 
                method='L-BFGS-B', 
                jac=True, 
                bounds=[(0, 1)] * N,
                args=(beta, lam),
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if res.success:
                current_h = res.x
            
            # Project to ensure sum is exactly N/2
            current_h = project_sum(current_h)
            
            # Evaluate true max overlap (not LSE)
            overlaps = compute_overlaps(current_h)
            true_max = np.max(overlaps)
            
            if true_max < best_score:
                best_score = true_max
                best_h = current_h.copy()

    return best_h
