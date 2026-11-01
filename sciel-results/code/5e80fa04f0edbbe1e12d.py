import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _score_discrete(f):
    """
    Calculate the score for a discrete signed step function f (values +/- 1).
    Score is sum of squared autocorrelation sidelobes.
    """
    # Autocorrelation
    # r_k = sum f_i f_{i+k}
    # Using FFT for speed
    F = np.fft.fft(f, n=2*N)
    power = np.abs(F)**2
    r = np.fft.ifft(power).real
    r = np.round(r) # Should be integers
    # r[0] is the DC component (sum f^2 = N)
    # r[1:N] are positive lags
    # r[N:] are negative lags (symmetric)
    # We want sum_{k=1}^{N-1} r_k^2
    # Note: r from FFT gives circular autocorrelation.
    # For linear autocorrelation, we need to be careful.
    # If N=400, and we pad to 2N=800, the linear autocorrelation is in r[0:N].
    # r[0] = sum f_i^2 = N.
    # r[k] for k=1..N-1 is the linear autocorrelation at lag k.
    # The score is sum_{k=1}^{N-1} r[k]^2.
    return np.sum(r[1:N]**2)

def _score_continuous(x):
    """
    Calculate the score for a continuous vector x.
    Objective: sum_{k=1}^{N-1} (sum_{i=0}^{N-k-1} x_i x_{i+k})^2
    """
    # Use FFT for linear autocorrelation
    # Pad x to 2N to get linear convolution
    x_pad = np.zeros(2*N)
    x_pad[:N] = x
    F = np.fft.fft(x_pad, n=2*N)
    power = np.abs(F)**2
    r = np.fft.ifft(power).real
    # r[0] is sum x_i^2
    # r[1:N] are the lags
    return np.sum(r[1:N]**2)

def _grad_continuous(x):
    """
    Gradient of the continuous objective.
    F(x) = sum_k r_k^2
    dF/dx_j = 2 * sum_k r_k * (dr_k/dx_j)
    dr_k/dx_j = x_{j+k} (if j+k < N) + x_{j-k} (if j-k >= 0)
    """
    x_pad = np.zeros(2*N)
    x_pad[:N] = x
    F = np.fft.fft(x_pad, n=2*N)
    power = np.abs(F)**2
    r = np.fft.ifft(power).real
    r = r[1:N] # lags 1 to N-1
    
    # We need to compute sum_k r_k * (x_{j+k} + x_{j-k})
    # This is a convolution of r with x?
    # Let's construct the term T_j = sum_{k=1}^{N-1} r_k (x_{j+k} + x_{j-k})
    
    # Term 1: sum_k r_k x_{j+k}
    # This is a cross-correlation of r and x.
    # Let's use FFT or direct convolution.
    # Since N=400, direct is O(N^2) = 160k ops. Fast enough.
    
    # Vectorized approach:
    # We want grad[j] = 2 * ( sum_{k=1}^{N-1-j} r_k x_{j+k} + sum_{k=1}^{j} r_k x_{j-k} )
    
    # Let's compute conv = np.convolve(r, x, mode='full')?
    # No, r is length N-1.
    
    # Let's use a simple loop for clarity and correctness, or vectorize with indexing.
    # Given N=400, a Python loop over j (400 iterations) with vectorized inner sums is fine.
    
    grad = np.zeros(N)
    # Precompute r as 1-indexed for ease? No, 0-indexed r[0] corresponds to lag 1.
    # r_lag_k = r[k-1]
    
    for j in range(N):
        # Sum 1: k such that j+k < N => k < N-j. k ranges 1 to N-j-1.
        # Corresponds to r indices 0 to N-j-2.
        # x indices j+1 to N-1.
        if N - j > 1:
            s1 = np.dot(r[:N-j-1], x[j+1:N])
        else:
            s1 = 0.0
            
        # Sum 2: k such that j-k >= 0 => k <= j. k ranges 1 to j.
        # Corresponds to r indices 0 to j-1.
        # x indices j-1 down to 0.
        if j > 0:
            s2 = np.dot(r[:j], x[j-1::-1])
        else:
            s2 = 0.0
            
        grad[j] = 2.0 * (s1 + s2)
        
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_score = np.inf
    best_f = None
    
    # Initial best: random
    f = np.random.choice([-1, 1], N)
    best_f = f.copy()
    best_score = _score_discrete(f)
    
    # Strategy: Multiple restarts of Continuous Relaxation + Discrete Polish
    
    while time.time() < deadline:
        # 1. Continuous Relaxation
        # Random init in [-1, 1]
        x0 = np.random.uniform(-1, 1, N)
        
        # Run L-BFGS-B
        try:
            res = minimize(
                _score_continuous, 
                x0, 
                method='L-BFGS-B', 
                jac=_grad_continuous,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            x_opt = res.x
        except Exception:
            x_opt = x0
            
        # 2. Round to discrete
        f_rounded = np.sign(x_opt)
        # Handle zeros (unlikely with uniform init)
        f_rounded[f_rounded == 0] = 1.0
        
        # 3. Discrete Local Search (Hill Climbing)
        # We can do a few steps of SA or Hill Climbing
        f_current = f_rounded.copy()
        score_current = _score_discrete(f_current)
        
        # Simple Hill Climbing with random restarts or SA
        # Let's do a quick SA
        T = 10.0
        T_min = 0.1
        cooling = 0.95
        
        # Estimate number of iterations based on remaining time
        # Each iteration is O(N) in Python loop, which is slow.
        # Let's limit iterations to keep it fast.
        max_iters = 5000
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            # Pick random index
            j = np.random.randint(N)
            # Flip
            old_val = f_current[j]
            f_current[j] = -old_val
            
            # Calculate new score
            # Efficient update?
            # For now, just recalculate. N=400, FFT is fast.
            new_score = _score_discrete(f_current)
            delta = new_score - score_current
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                score_current = new_score
                if score_current < best_score:
                    best_score = score_current
                    best_f = f_current.copy()
            else:
                f_current[j] = old_val
                
            T *= cooling
            if T < T_min:
                break
        
        # If we still have time, we can try another restart
        if time.time() > deadline - 10:
            break

    return best_f
