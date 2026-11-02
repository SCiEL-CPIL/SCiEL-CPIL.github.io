"""
Combined program: Direct optimization of step heights using soft-min cross-correlation.
Strategy:
1. Model the problem as maximizing the minimum cross-correlation R(k) = sum h[i]h[i+k].
   Minimizing max overlap is equivalent to maximizing min correlation because Overlap(k) = 1 - Correlation(k) 
   (given mean(h)=0.5 and mean(1-h)=0.5, and circular shifts).
   Actually, Overlap(k) = sum h[i](1-h[i+k]) = sum h[i] - sum h[i]h[i+k] = N/2 - Correlation(k).
   So minimizing max Overlap is equivalent to minimizing (N/2 - min Correlation), i.e., maximizing min Correlation.
   
2. Use a smooth surrogate: soft_min_corr = -1/beta * log(sum(exp(-beta * corr_k))).
   Objective to minimize: 1 - soft_min_corr (or just -soft_min_corr, scaling doesn't matter for argmin).
   We minimize -soft_min_corr.

3. Optimize h directly in [0,1]^N with sum constraint.
   Use L-BFGS-B with bounds [0,1].
   Project to sum=N/2 after each step or use a penalty/augmented Lagrangian? 
   The inspiration uses a projection step. We will use a projection step inside the objective/gradient 
   or simply rely on L-BFGS-B bounds and a final projection? 
   L-BFGS-B handles bounds well. The sum constraint is a linear equality. 
   A common trick is to optimize h in a subspace or use a penalty. 
   However, the inspiration's `_project_to_simplex` is effective. 
   We will implement an objective that projects h to the simplex (sum=N/2, 0<=h<=1) and then computes the cost.
   But projecting inside the objective makes the gradient discontinuous.
   Better approach: 
   Use L-BFGS-B on h with bounds [0,1]. 
   To enforce sum=N/2, we can use a soft penalty or adjust the gradient.
   Alternatively, parameterize h = 0.5 + v, where sum(v)=0. 
   But v must keep h in [0,1].
   
   Let's stick to the inspiration's approach which scored 1.25:
   Optimize h directly.
   Objective: f(h) = -soft_min_corr(h).
   Gradient: computed analytically for the soft-min part.
   Constraint: sum(h) = N/2.
   
   To handle the constraint efficiently within L-BFGS-B (which doesn't handle equality constraints natively),
   we can use a penalty method or project after each iteration.
   The inspiration used `_project_to_simplex` inside the loop? No, it used it to initialize and check results.
   Actually, standard L-BFGS-B with bounds [0,1] will drift in sum.
   We can add a large penalty term: lambda * (sum(h) - N/2)^2.
   Or, we can use the fact that the optimal solution likely lies on the boundary or has specific structure.
   
   Let's try a robust approach:
   1. Multiple restarts with diverse initializations.
   2. Use L-BFGS-B to minimize -soft_min_corr(h) + mu * (sum(h) - N/2)^2.
   3. Increase mu over iterations to enforce constraint.
   4. Final projection to ensure exact sum.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

# EVOLVE-BLOCK-START

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_correlations(h):
    """Compute circular cross-correlations R(k) = sum_i h[i] * h[(i+k)%N] for k=0..N-1."""
    n = len(h)
    # Use FFT for efficient computation? 
    # R(k) = ifft(fft(h) * conj(fft(h)))
    # This gives circular correlation.
    H = np.fft.fft(h)
    power = H * np.conj(H)
    corrs = np.real(np.fft.ifft(power))
    # The correlation is usually defined with a normalization. 
    # The task defines overlap as sum. 
    # R(k) = sum h[i] h[i+k].
    # So corrs from FFT are exactly what we need.
    return corrs

def _soft_min_corr(h, beta):
    """
    Compute soft-min of correlations.
    soft_min = -1/beta * log(sum(exp(-beta * corr_k)))
    """
    corrs = _compute_correlations(h)
    # We want to maximize min(corr), so we minimize -min(corr).
    # soft_min is an approximation of min(corr).
    # To avoid overflow, shift by max(corr).
    c_max = np.max(corrs)
    if c_max < -100:
        return c_max
    exp_vals = np.exp(-beta * (corrs - c_max))
    log_sum = np.log(np.sum(exp_vals))
    soft_min = c_max - log_sum / beta
    return soft_min, corrs

def _objective_and_grad(h, beta, mu):
    """
    Objective: -soft_min_corr(h) + mu * (sum(h) - N/2)^2
    Gradient: -d(soft_min)/dh + 2*mu*(sum(h)-N/2)*1
    """
    n = len(h)
    corrs = _compute_correlations(h)
    
    # Soft min calculation
    c_max = np.max(corrs)
    if c_max < -100:
        soft_min = c_max
        weights = np.full(n, 1.0/n) # Flat if all very negative? No, just return 0 grad?
        # If all are very negative, gradient is small.
        grad_soft_min = np.zeros(n)
    else:
        exp_vals = np.exp(-beta * (corrs - c_max))
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp # These are the weights for the min (inverse of max weights)
        soft_min = c_max - np.log(sum_exp) / beta
        
        # Gradient of soft_min w.r.t. h
        # d(soft_min)/d(h) = sum_k weights_k * d(corr_k)/d(h)
        # corr_k = sum_i h[i] h[(i+k)%N]
        # d(corr_k)/d(h[j]) = h[(j+k)%N] + h[(j-k)%N]
        
        grad_soft_min = np.zeros(n)
        for k in range(n):
            if weights[k] < 1e-15:
                continue
            # Contribution from corr_k
            # d(corr_k)/d(h) = roll(h, -k) + roll(h, k)
            dk_dh = np.roll(h, -k) + np.roll(h, k)
            grad_soft_min += weights[k] * dk_dh
        
        # d(soft_min)/d(h) is the sum above.
        # Note: The derivative of the log-sum-exp term involves the weights.
        # Let's verify the gradient.
        # f = -1/beta log(sum exp(-beta c_k))
        # df/dc_k = exp(-beta c_k) / sum(exp(-beta c_j)) = weights_k
        # dc_k/dh = roll(h, -k) + roll(h, k)
        # So grad_h = sum_k weights_k * (roll(h, -k) + roll(h, k))
        # This looks correct.

    obj = -soft_min + mu * (np.sum(h) - N/2.0)**2
    
    # Gradient of penalty term
    sum_h = np.sum(h)
    grad_penalty = 2.0 * mu * (sum_h - N/2.0) * np.ones(n)
    
    grad = -grad_soft_min + grad_penalty
    
    return obj, grad

def _project_to_simplex(v, target_sum):
    """Project v onto {x : 0 <= x <= 1, sum(x) = target_sum}."""
    n = len(v)
    
    # First project onto {x >= 0, sum = target_sum}
    # This is a standard water-filling problem.
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # Find rho such that u[rho] > (cssv[rho] - target_sum) / (rho+1)
    # rho is the index of the largest element that is above the water level.
    # Condition: u * arange(1, n+1) > cssv - target_sum
    rho_candidates = np.nonzero(u * np.arange(1, n + 1) > (cssv - target_sum))[0]
    if len(rho_candidates) == 0:
        rho = n - 1
    else:
        rho = rho_candidates[-1]
    
    theta = (cssv[rho] - target_sum) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    
    # Clip to [0, 1]
    w = np.clip(w, 0, 1)
    
    # Adjust to ensure sum = target_sum exactly
    # The water filling ensures sum >= target_sum if possible, but clipping to 1 might break it.
    # Iterate to fix sum.
    for _ in range(20):
        diff = target_sum - np.sum(w)
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            # Need to increase sum. Increase elements that are < 1.
            mask = w < 1.0
            cnt = np.sum(mask)
            if cnt == 0:
                break
            w[mask] += diff / cnt
        else:
            # Need to decrease sum. Decrease elements that are > 0.
            mask = w > 0.0
            cnt = np.sum(mask)
            if cnt == 0:
                break
            w[mask] += diff / cnt
        w = np.clip(w, 0, 1)
    
    return w

def construct():
    deadline = time.time() + 145.0
    best_h = np.full(N, 0.5)
    best_score = np.inf # We want to minimize max overlap, which is N/2 - min_corr.
    # Let's track min_corr to maximize it.
    best_min_corr = -np.inf
    
    def get_score(h):
        corrs = _compute_correlations(h)
        min_corr = np.min(corrs)
        # Max overlap = N/2 - min_corr
        return N/2.0 - min_corr

    current_score = get_score(best_h)
    best_score = current_score
    best_min_corr = N/2.0 - current_score

    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    restarts = []
    restarts.append(np.full(N, 0.5))
    
    # Block structures
    for frac in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        k = int(N * frac)
        if k == 0 or k == N: continue
        # High block, low block
        val_high = (N/2.0) / k
        val_low = 0.0
        if val_high <= 1.0:
            h1 = np.zeros(N)
            h1[:k] = val_high
            restarts.append(h1)
            
        # Low block, high block
        val_low2 = (N/2.0) / (N-k)
        if val_low2 <= 1.0:
            h2 = np.ones(N) * 0.5
            h2[:k] = 0.0
            h2[k:] = (N/2.0) / (N-k)
            restarts.append(h2)
            
    # Alternating
    restarts.append(np.array([0.0 if i % 2 == 0 else 1.0 for i in range(N)]))
    restarts.append(np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)]))
    
    # Sinusoidal
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    for m in [1, 2, 3]:
        for amp in [0.3, 0.4, 0.45]:
            restarts.append(0.5 + amp * np.sin(2*np.pi*m*x))
            restarts.append(0.5 + amp * np.cos(2*np.pi*m*x))
            
    # Random
    for _ in range(10):
        restarts.append(rng.uniform(0, 1, N))
        
    # Project all restarts to valid space
    valid_restarts = []
    for h0 in restarts:
        h0 = _project_to_simplex(h0, N/2.0)
        valid_restarts.append(h0)
        
    # Optimization parameters
    betas = [20.0, 50.0, 100.0, 200.0]
    mus = [10.0, 100.0, 1000.0] # Penalty for sum constraint
    
    # Phase 1: Coarse optimization with lower beta and lower penalty
    for h0 in valid_restarts:
        if time.time() > deadline - 20.0:
            break
            
        current_h = h0.copy()
        
        # Annealing beta and mu
        for beta in betas:
            if time.time() > deadline - 10.0:
                break
                
            for mu in mus:
                if time.time() > deadline - 5.0:
                    break
                
                def obj_grad(h):
                    return _objective_and_grad(h, beta, mu)
                
                try:
                    res = minimize(
                        obj_grad,
                        current_h,
                        jac=True,
                        method='L-BFGS-B',
                        bounds=[(0, 1)] * N,
                        options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
                    )
                    # Project result to ensure exact sum and bounds
                    h_opt = _project_to_simplex(res.x, N/2.0)
                    score = get_score(h_opt)
                    
                    if score < best_score - 1e-12:
                        best_score = score
                        best_h = h_opt
                        current_h = h_opt
                        best_min_corr = N/2.0 - best_score
                        
                except Exception:
                    pass

    # Phase 2: Fine-tuning with high beta and high penalty
    if time.time() < deadline - 10.0:
        for beta in [100.0, 200.0, 500.0]:
            if time.time() > deadline - 5.0:
                break
            for mu in [1000.0, 10000.0]:
                if time.time() > deadline - 2.0:
                    break
                def obj_grad(h):
                    return _objective_and_grad(h, beta, mu)
                
                try:
                    res = minimize(
                        obj_grad,
                        best_h,
                        jac=True,
                        method='L-BFGS-B',
                        bounds=[(0, 1)] * N,
                        options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    h_opt = _project_to_simplex(res.x, N/2.0)
                    score = get_score(h_opt)
                    
                    if score < best_score - 1e-12:
                        best_score = score
                        best_h = h_opt
                        best_min_corr = N/2.0 - best_score
                        
                except Exception:
                    pass

    # Final check and projection
    best_h = _project_to_simplex(best_h, N/2.0)
    
    return best_h

# EVOLVE-BLOCK-END
