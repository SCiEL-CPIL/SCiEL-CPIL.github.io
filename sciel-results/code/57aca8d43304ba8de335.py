import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _calc_score(h):
    """
    Calculate the score (upper bound on C3) for a given step function h.
    Score = max_{k != 0} |(h*h)_k| / ||h||_2^2
    For binary +/- 1, ||h||_2^2 = N.
    """
    # Use FFT for speed
    # Autocorrelation is the inverse FFT of |FFT(h)|^2
    # We need to be careful with padding to avoid circular convolution artifacts
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    autocorr = np.fft.irfft(H * np.conjugate(H), n=n_fft)
    
    # autocorr[0] is lag 0. autocorr[1:N] are lags 1..N-1.
    # autocorr[N+1:] are negative lags (circularly wrapped), but for real signals, 
    # the aperiodic autocorrelation is symmetric.
    # The full autocorrelation vector for lags -(N-1)..(N-1) can be reconstructed.
    # However, for peak sidelobe, we just need the max absolute value of non-zero lags.
    # Due to symmetry, max |autocorr[1:N]| is sufficient if we consider the full support?
    # Actually, np.fft.irfft gives the circular autocorrelation.
    # If we zero-pad to 2N, the circular autocorrelation of length 2N corresponds to the linear autocorrelation
    # at lags 0, 1, ..., N-1 and -(N-1), ..., -1.
    # Specifically, autocorr[0] is lag 0.
    # autocorr[1:N] are lags 1 to N-1.
    # autocorr[N+1:2N] are lags -1 to -(N-1).
    # Since h is real, autocorr is real and symmetric. autocorr[k] = autocorr[2N-k].
    # So checking autocorr[1:N] covers all non-zero lags.
    
    sidelobes = np.abs(autocorr[1:N])
    peak = np.max(sidelobes)
    norm = np.sum(h**2)
    if norm == 0:
        return 0.0
    return peak / norm

def _soft_max(values, beta=100):
    """
    Smooth approximation of max.
    """
    max_val = np.max(values)
    if max_val == 0:
        return 0.0
    # log-sum-exp trick for stability
    # We want max |x_i|
    # Let y_i = |x_i|
    # max y_i approx 1/beta * log(sum(exp(beta * y_i)))
    # To avoid overflow, subtract max
    exp_vals = np.exp(beta * (values - max_val))
    return max_val + (1.0 / beta) * np.log(np.sum(exp_vals))

def _objective_continuous(x):
    """
    Objective function for continuous optimization.
    x is in [-1, 1]^N.
    We minimize the soft-max of the absolute autocorrelation sidelobes.
    """
    # Calculate autocorrelation via FFT
    n_fft = 2 * N
    H = np.fft.rfft(x, n=n_fft)
    autocorr = np.fft.irfft(H * np.conjugate(H), n=n_fft)
    
    # Sidelobes are lags 1 to N-1
    sidelobes = np.abs(autocorr[1:N])
    
    # We want to minimize the peak sidelobe.
    # Use soft-max to make it differentiable.
    # Beta controls the sharpness. Higher beta -> closer to max.
    # Start with moderate beta, maybe increase later? Or keep constant.
    # The norm ||x||^2 is variable. We should normalize?
    # The task asks for upper bound on C3.
    # If we scale x, autocorr scales by ||x||^2, and norm scales by ||x||^2.
    # So the ratio is scale-invariant.
    # However, in continuous relaxation, if x -> 0, the ratio is undefined/0.
    # We can add a penalty or just rely on the optimizer not going to 0.
    # Actually, if we minimize max(autocorr) / ||x||^2, it's homogeneous of degree 0.
    # L-BFGS-B might struggle with scale invariance.
    # Let's minimize just the soft-max of sidelobes, but constrain ||x||^2 to be close to N?
    # Or better, minimize soft_max(sidelobes) / (||x||^2 + epsilon).
    # But division makes gradient complex.
    
    # Alternative: Since the problem is scale invariant, we can fix the scale.
    # E.g., force ||x||_2^2 = N.
    # But minimize doesn't support equality constraints easily (SLSQP does, but is slower).
    # Let's try minimizing soft_max(sidelobes) directly.
    # If the optimizer shrinks x, the score goes to 0.
    # We need to prevent x from going to 0.
    # We can add a term that penalizes deviation from unit norm (scaled).
    # target_norm = N.
    # penalty = (||x||^2 - N)^2
    # Objective = soft_max(sidelobes) + lambda * penalty
    
    # Let's stick to the ratio if possible, or just soft_max with a lower bound on norm.
    # Actually, if we just minimize soft_max(sidelobes), the global minimum is 0 (x=0).
    # We must avoid x=0.
    # Let's use the ratio: J = soft_max(sidelobes) / (||x||^2 + 1e-6)
    
    norm_sq = np.sum(x**2)
    if norm_sq < 1e-6:
        return 1e10 # Penalize zero vector
    
    sm = _soft_max(sidelobes, beta=50)
    return sm / norm_sq

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_h = None
    best_score = float('inf')
    
    # 1. Continuous Relaxation
    # Initialize x randomly in [-1, 1]
    # Multiple restarts for continuous optimization
    num_restarts = 5
    
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Random initialization
        x0 = np.random.uniform(-1, 1, size=N)
        
        # Try to optimize
        try:
            res = minimize(
                _objective_continuous, 
                x0, 
                method='L-BFGS-B', 
                bounds=[(-1, 1)] * N,
                options={'maxiter': 200, 'disp': False}
            )
            x_opt = res.x
            
            # Threshold to get discrete h
            h = np.sign(x_opt)
            # Ensure no zeros (sign(0)=0)
            h[h == 0] = 1
            
            # Evaluate discrete score
            score = _calc_score(h)
            
            if score < best_score:
                best_score = score
                best_h = h.copy()
                
        except Exception:
            pass

    if best_h is None:
        # Fallback to random
        best_h = np.random.choice([-1, 1], size=N)
        best_score = _calc_score(best_h)
        
    # 2. Discrete Refinement (Hill Climbing)
    # Use the best continuous result as starting point for discrete search
    # This is often much faster and better than starting from random discrete.
    
    h = best_h
    current_score = best_score
    
    # Local search
    # We can speed up score calculation by using FFT
    # But for single bit flip, we can update autocorrelation incrementally?
    # Or just recompute FFT. N=400, FFT is very fast.
    
    # To allow more iterations, we can use a more aggressive local search.
    # Try flipping bits that are most "likely" to improve?
    # Or just random order?
    
    # Let's do a few passes of hill climbing.
    # If no improvement in a pass, perturb and continue.
    
    passes = 0
    while time.time() < deadline - 5:
        improved = False
        # Shuffle indices to avoid bias
        indices = np.random.permutation(N)
        
        for j in indices:
            if time.time() > deadline - 5:
                break
            
            old_val = h[j]
            h[j] = -h[j]
            new_score = _calc_score(h)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                h[j] = old_val
        
        passes += 1
        if not improved:
            # Local minimum
            # Perturb
            # Flip a few random bits
            num_perturb = 5 + passes
            for _ in range(num_perturb):
                j = np.random.randint(N)
                h[j] = -h[j]
            current_score = _calc_score(h)
            
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
            else:
                # If perturbation didn't help, we might be stuck.
                # Try a new continuous restart?
                # We have time, so let's try another continuous restart if we have budget.
                # But we already did 5. Let's just continue discrete search with larger perturbations.
                pass
        else:
            # Update best if improved
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
            # Reset passes? No, keep counting to increase perturbation size if stuck later.
            # But if we are improving, we are in a good basin.
            # Let's just continue.

    return best_h
