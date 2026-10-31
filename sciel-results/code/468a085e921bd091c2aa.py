import numpy as np
import time
import scipy.signal as sig
import scipy.optimize as opt

N = 600
DEADLINE = 110.0 # Safety margin

def construct():
    start_time = time.time()
    
    # Initial guess: Nuttall window
    # Nuttall has very low sidelobes
    h = sig.get_window('nuttall', N)
    
    # Normalize to sum to 1 for numerical stability during optimization, 
    # though the ratio is scale invariant.
    h = h / np.sum(h)
    
    # Objective function
    def objective(u):
        # u are log-heights to ensure positivity
        h_curr = np.exp(u)
        # Compute autocorrelation
        # Linear autocorrelation
        # Use FFT for speed
        n_fft = 2 * N
        H = np.fft.rfft(h_curr, n=n_fft)
        R = np.fft.irfft(H * np.conj(H), n=n_fft)
        R = R[:N] # Keep first N lags
        
        R0 = R[0]
        if R0 < 1e-10:
            return 1e10
            
        # Sidelobes
        R_side = R[1:]
        
        # Smooth max approximation
        # logsumexp(x) = log(sum(exp(x)))
        # We want max(R_side / R0)
        # Let's minimize sum(exp(beta * R_side / R0)) to approximate max
        beta = 100.0
        scaled = beta * R_side / R0
        # Shift for numerical stability
        max_val = np.max(scaled)
        log_sum_exp = max_val + np.log(np.sum(np.exp(scaled - max_val)))
        
        return log_sum_exp

    # Initial log-heights
    u0 = np.log(h + 1e-9)
    
    # Optimize
    best_h = h
    best_score = 1.0
    
    # Use L-BFGS-B
    try:
        res = opt.minimize(objective, u0, method='L-BFGS-B', 
                           options={'maxiter': 100, 'maxfun': 1000, 'ftol': 1e-8})
        
        if time.time() - start_time < DEADLINE:
            h_opt = np.exp(res.x)
            # Check score
            R = np.fft.irfft(np.fft.rfft(h_opt, 2*N) * np.conj(np.fft.rfft(h_opt, 2*N)), 2*N)[:N]
            score = np.max(R[1:]) / R[0]
            if score < best_score:
                best_score = score
                best_h = h_opt
    except Exception as e:
        pass

    # If optimization didn't improve or failed, use the initial window
    # Calculate final score for the best candidate
    R = np.fft.irfft(np.fft.rfft(best_h, 2*N) * np.conj(np.fft.rfft(best_h, 2*N)), 2*N)[:N]
    final_score = np.max(R[1:]) / R[0]
    
    # The task asks for the sequence of heights.
    # The score is the upper bound on C1, which is this normalized max.
    # We return the heights.
    
    return list(best_h)
