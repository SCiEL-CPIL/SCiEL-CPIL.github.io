import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _compute_cross_corr(h):
    """
    Compute linear cross-correlation of h with (1-h) for all non-zero shifts.
    Returns array of length 2*(N-1).
    """
    g = 1.0 - h
    # Use FFT for efficiency
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    G = np.fft.rfft(g, n=n_fft)
    c_full = np.fft.irfft(H * G, n=n_fft)
    
    # Extract relevant parts
    # c_full[0] is k=0
    # c_full[1:N] is k=1..N-1
    # c_full[N_fft:N_fft-(N-1):] is k=-(N-1)..-1
    c_pos = c_full[1:N]
    c_neg = c_full[n_fft-(N-1):n_fft]
    
    return np.concatenate([c_neg, c_pos])

def _get_max_overlap(h):
    """
    Return the maximum absolute overlap value.
    """
    corr = _compute_cross_corr(h)
    if len(corr) == 0:
        return 0.0
    return np.max(np.abs(corr))

def _project_to_feasible(h_raw):
    """
    Project h_raw to the set {h | 0 <= h_i <= 1, sum(h) = N/2}.
    Uses bisection on the Lagrange multiplier lambda for h_proj = clip(h_raw - lambda, 0, 1).
    """
    target_sum = N / 2.0
    
    # Initial clip to [0,1] to bound the lambda search space
    h_clipped = np.clip(h_raw, 0.0, 1.0)
    current_sum = np.sum(h_clipped)
    
    # If sum is already correct, return
    if abs(current_sum - target_sum) < 1e-12:
        return h_clipped
    
    # Define function sum(lambda) = sum(clip(h_raw - lambda, 0, 1))
    # This is a monotonically decreasing function of lambda.
    # We need to find lambda such that sum(lambda) = target_sum.
    
    # Bounds for lambda
    # If lambda is very large negative, clip(h - lambda, 0, 1) -> 1 for all, sum -> N
    # If lambda is very large positive, clip(h - lambda, 0, 1) -> 0 for all, sum -> 0
    lo = -1.0 - np.max(h_raw)
    hi = 1.0 + np.max(h_raw)
    
    # Bisection
    for _ in range(50):
        mid = (lo + hi) / 2.0
        h_mid = np.clip(h_raw - mid, 0.0, 1.0)
        s_mid = np.sum(h_mid)
        
        if abs(s_mid - target_sum) < 1e-12:
            break
        if s_mid > target_sum:
            # Need to decrease sum, so increase lambda
            lo = mid
        else:
            # Need to increase sum, so decrease lambda
            hi = mid
            
    lambda_star = (lo + hi) / 2.0
    h_proj = np.clip(h_raw - lambda_star, 0.0, 1.0)
    
    # Final correction for floating point errors to ensure sum is exactly N/2
    # Distribute the residual among elements not at bounds 0 or 1
    residual = target_sum - np.sum(h_proj)
    if abs(residual) > 1e-14:
        free_mask = (h_proj > 1e-9) & (h_proj < 1.0 - 1e-9)
        if np.any(free_mask):
            h_proj[free_mask] += residual / np.sum(free_mask)
            h_proj = np.clip(h_proj, 0.0, 1.0)
    
    return h_proj

def _objective_function(x, beta=100.0, lambda_sum=1000.0):
    """
    Objective function for optimization.
    x = [h_0, ..., h_{N-1}, t]
    Minimize t + lambda_sum * (sum(h) - N/2)^2 + smooth_max(overlaps - t)
    
    Using LogSumExp approximation for max:
    max(z) ~= (1/beta) * log(sum(exp(beta*z)))
    """
    h = x[:N]
    t = x[N]
    
    # Sum constraint penalty
    sum_h = np.sum(h)
    penalty = lambda_sum * (sum_h - N/2.0)**2
    
    # Overlaps
    corr = _compute_cross_corr(h)
    if len(corr) == 0:
        return t + penalty
    
    # Smooth max of (corr - t)
    # We want to penalize if any corr > t
    # max(0, corr - t) is the violation.
    # Approximate max(0, z) with softplus(z) = log(1 + exp(z))
    # But we need the MAX of the violations, not the sum.
    # Let v_i = corr_i - t.
    # We want to minimize t + max(0, max(v_i)).
    # This is equivalent to min(t, t + max(v_i)) which is t if max(v_i)<=0, else t+max(v_i).
    # Actually, just minimizing t subject to corr_i <= t for all i is the epigraph.
    # Smooth constraint: sum_i softplus(beta * (corr_i - t)) / N
    # This forces t to be at least the maximum corr_i (on average, smoothed).
    
    shifted = beta * (corr - t)
    # Clip to avoid overflow in exp
    shifted = np.clip(shifted, -50, 50)
    
    # LogSumExp trick for numerical stability
    max_shifted = np.max(shifted)
    if max_shifted > 50: # If max is very large, softplus is approx linear
        softplus_sum = np.sum(shifted - max_shifted) / len(corr) + max_shifted / beta
    else:
        # sum(log(1 + exp(z)))
        # log(1+exp(z)) is stable if z is not too large
        log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        # softplus is log(1+exp(x)). 
        # sum softplus(x_i) = sum log(1+exp(x_i))
        # This is NOT log(sum(exp(x_i))).
        # Correct softplus sum:
        # For large x, softplus(x) ~ x.
        # For small x, softplus(x) ~ log(2).
        # We can compute sum(softplus(z)) directly if z is bounded.
        sp = np.log1p(np.exp(shifted))
        softplus_sum = np.sum(sp) / (beta * len(corr))
        
    return t + penalty + softplus_sum

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin for 150s limit
    
    rng = np.random.default_rng(42)
    
    # Initial candidates
    candidates = []
    
    # 1. Constant
    candidates.append(np.full(N, 0.5))
    
    # 2. Alternating
    h_alt = np.zeros(N)
    h_alt[::2] = 0.5 + 0.1
    h_alt[1::2] = 0.5 - 0.1
    candidates.append(h_alt)
    
    # 3. Sinusoidal variations
    t = np.arange(N)
    for freq in range(1, 10):
        for phase in [0, np.pi/4, np.pi/2, np.pi]:
            for amp in [0.1, 0.2, 0.3, 0.4]:
                h_sin = 0.5 + amp * np.sin(2*np.pi*freq*t/N + phase)
                candidates.append(h_sin)
                
    # 4. Random
    for _ in range(20):
        candidates.append(rng.uniform(0, 1, N))
        
    # Evaluate initial candidates
    best_h = np.full(N, 0.5)
    best_score = float('inf')
    
    for h_cand in candidates:
        h_proj = _project_to_feasible(h_cand)
        score = _get_max_overlap(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj
    
    # Optimization Phase
    bounds_h = [(0.0, 1.0) for _ in range(N)]
    bounds_t = [(0.0, 2.0)] # t can be up to 2 (max possible overlap for N=100, h in [0,1] is bounded)
    bounds_full = bounds_h + bounds_t
    
    beta_schedule = [10, 50, 100, 200, 400]
    
    for beta in beta_schedule:
        if time.time() > deadline - 30.0:
            break
            
        # Restart from best_h
        x0 = np.concatenate([best_h, [best_score]])
        
        # L-BFGS-B
        try:
            res = minimize(
                _objective_function, 
                x0, 
                args=(beta, 1000.0),
                method='L-BFGS-B', 
                bounds=bounds_full,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_new = _project_to_feasible(res.x[:N])
            score_new = _get_max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
        
        # Multiple random restarts for lower beta
        if beta < 100:
            for _ in range(5):
                if time.time() > deadline - 30.0:
                    break
                h_rand = rng.uniform(0, 1, N)
                h_rand = _project_to_feasible(h_rand)
                x0 = np.concatenate([h_rand, [best_score]])
                try:
                    res = minimize(
                        _objective_function, 
                        x0, 
                        args=(beta, 1000.0),
                        method='L-BFGS-B', 
                        bounds=bounds_full,
                        options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    h_new = _project_to_feasible(res.x[:N])
                    score_new = _get_max_overlap(h_new)
                    if score_new < best_score:
                        best_score = score_new
                        best_h = h_new
                except Exception:
                    pass
                    
    # Coordinate Descent Polish
    # This helps in cases where the gradient method gets stuck or due to numerical issues
    if time.time() < deadline - 10.0:
        improved = True
        max_cd_iters = 50
        cd_iter = 0
        while improved and cd_iter < max_cd_iters and time.time() < deadline - 10.0:
            improved = False
            cd_iter += 1
            for i in range(N):
                old_val = best_h[i]
                # Try small perturbations
                for delta in [0.005, -0.005, 0.01, -0.01, 0.02, -0.02]:
                    trial_h = best_h.copy()
                    trial_h[i] = np.clip(old_val + delta, 0, 1)
                    trial_h = _project_to_feasible(trial_h)
                    score_trial = _get_max_overlap(trial_h)
                    if score_trial < best_score - 1e-12:
                        best_h = trial_h
                        best_score = score_trial
                        improved = True
                        break
    
    # Final Projection to ensure exactness
    best_h = _project_to_feasible(best_h)
    
    return best_h
