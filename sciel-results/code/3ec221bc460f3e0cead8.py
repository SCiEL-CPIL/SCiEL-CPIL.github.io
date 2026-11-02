import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 400
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def _compute_ac(x):
    """
    Compute linear autoconvolution of x.
    Returns array of length 2N-1.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    fx = np.fft.rfft(x, n=n_fft)
    power = fx * np.conj(fx)
    ac = np.fft.irfft(power, n=n_fft)
    # The linear autoconvolution is in ac[:2*N-1]
    # However, due to FFT circularity, the "center" (lag 0) is at index 0 
    # only if we define correlation as sum x[n]x[n+k].
    # Let's verify indices.
    # np.fft.irfft returns time domain samples n=0..n_fft-1.
    # For linear convolution of x with x (reversed? no, autocorr is x with x reversed? 
    # Autocorr R[k] = sum x[n] x[n+k].
    # This is equivalent to convolution of x with x_reversed?
    # Conv(x, x_rev)[k] = sum x[i] x_rev[k-i] = sum x[i] x[k-i].
    # If x is symmetric, it's the same.
    # Standard autocorrelation via FFT:
    # 1. Zero pad x to length L >= 2N-1.
    # 2. X = FFT(x).
    # 3. P = |X|^2.
    # 4. R = IFFT(P).
    # R[k] gives the circular autocorrelation.
    # For zero-padded linear case, R[0] is sum x^2.
    # R[k] for k=1..N-1 is sum x[i]x[i+k].
    # R[k] for k=N..2N-2 is sum x[i]x[i-k] (negative lags).
    # Due to symmetry of autocorr, R[k] = R[2N-1-k] roughly?
    # Actually, the output of IFFT(|X|^2) for zero-padded real x is real and symmetric around 0.
    # The array R returned by irfft has length n_fft.
    # R[0] is the peak (lag 0).
    # R[1] ... R[N-1] are positive lags.
    # R[n_fft-N+1] ... R[n_fft-1] are negative lags (wrapped around).
    # We want the max absolute value of the side lobes.
    # Side lobes are all lags k != 0.
    # In the array R, index 0 is lag 0.
    # Indices 1 to N-1 are lags 1 to N-1.
    # Indices n_fft - (N-1) to n_fft - 1 correspond to lags -(N-1) to -1.
    # So we check R[1:N] and R[n_fft-N+1 : n_fft].
    
    # To be safe and simple, we can just take the max of the whole array excluding the peak?
    # But the peak is always the largest positive value (sum of squares).
    # So max_abs_side = max(abs(R[1:]), abs(R[n_fft-N+1:]))?
    # Actually, just taking max(abs(R)) excluding index 0 is safe if we know R[0] is the max.
    # For real signals, R[0] = sum x^2 >= |R[k]| for all k (Cauchy-Schwarz).
    # So R[0] is the global maximum in absolute value.
    # Thus, max_abs_side = max(abs(R[1:])). 
    # Wait, R has length n_fft. The linear autocorr is embedded.
    # The "non-zero" part of the linear autocorr is at indices 0..N-1 and n_fft-N+1..n_fft-1.
    # The middle part (N..n_fft-N) should be zero (or close to it due to numerical error).
    # So max(abs(R[1:])) might pick up numerical noise in the middle if N is small relative to n_fft?
    # n_fft is power of 2 >= 2N-1. For N=400, n_fft=1024.
    # The non-zero part is 0..399 and 625..1023.
    # The middle 400..624 is zero.
    # So max(abs(R[1:])) is safe.
    
    return R

def _score_from_ac(R, x):
    """
    Calculate the score: max_abs_side / (sum(x))^2.
    """
    # R[0] is the peak.
    # Side lobes are the rest.
    # We take the max absolute value of R excluding index 0.
    # But we must be careful: R contains the full FFT length.
    # The linear autocorr values are at specific indices.
    # As established, R[0] is lag 0.
    # R[1:N] are lags 1..N-1.
    # R[n_fft-N+1:] are lags -(N-1)..-1.
    # The rest are zero.
    # So max_side = max(np.max(np.abs(R[1:N])), np.max(np.abs(R[n_fft-N+1:])))
    
    n_fft = len(R)
    max_side = 0.0
    if N > 1:
        max_side = max(np.max(np.abs(R[1:N])), np.max(np.abs(R[n_fft-N+1:n_fft])))
    
    s = np.sum(x)
    denom = s * s
    if denom < 1e-9:
        return 1e10
    return max_side / denom

def _objective(x):
    """
    Objective for L-BFGS.
    We want to minimize max_abs_side / sum^2.
    This is non-smooth.
    Surrogate: minimize (sum of squares of side lobes) / sum^2 
    PLUS a penalty to keep x close to +/- 1.
    
    Let F(x) = (sum_{k!=0} R[k]^2) / (sum x)^2 + lambda * sum (x_i^2 - 1)^2
    The first term is smooth (except at sum=0).
    The second term pushes x to the sphere of radius sqrt(N).
    """
    R = _compute_ac(x)
    n_fft = len(R)
    
    # Extract side lobes
    if N > 1:
        side1 = R[1:N]
        side2 = R[n_fft-N+1:n_fft]
        sum_sq_side = np.sum(side1**2) + np.sum(side2**2)
    else:
        sum_sq_side = 0.0
        
    s = np.sum(x)
    denom = s * s
    if denom < 1e-9:
        denom = 1e-9
        
    # The ratio we want to minimize is roughly max_side / denom.
    # Minimizing sum_sq_side / denom is a proxy for minimizing peak side lobe,
    # as it penalizes all side lobes.
    term1 = sum_sq_side / denom
    
    # Penalty for not being +/- 1
    # We want x_i^2 = 1.
    penalty = np.sum((x**2 - 1.0)**2)
    
    return term1 + 10.0 * penalty

def _gradient(x):
    """
    Numerical gradient or analytic?
    L-BFGS can use numerical gradient, but it's slow.
    Let's try to provide an analytic gradient if possible, or use a fast numerical one.
    Given the complexity of the FFT derivative, numerical might be safer for implementation correctness.
    However, N=400. 400 function evaluations per gradient step is too slow for L-BFGS if we do many steps.
    
    Alternative: Use finite difference with complex step? Or just rely on scipy's numerical gradient?
    scipy.optimize.minimize with method='L-BFGS-B' uses numerical gradients by default if jac is None.
    It uses a forward difference approximation.
    Step size is automatic.
    For N=400, one gradient eval takes ~400 * (cost of _objective).
    Cost of _objective: FFT of size 1024. Very fast.
    So 400 FFTs is ~400 * 10us = 4ms.
    So one gradient step is ~4ms.
    We can afford ~1000 steps in 150s? 1000 * 4ms = 4s. Yes.
    """
    return None # Let scipy handle it

def construct():
    start_time = time.time()
    
    # 1. Initial guess: Random +/- 1
    # But L-BFGS works on continuous space.
    # Start with x = ones? Or random continuous?
    # Let's start with a random continuous vector, then project.
    # Or start with ones, which is a known local minimum for the "all same" case,
    # but we want to escape it.
    
    best_x_discrete = np.ones(N)
    best_score = _score_from_ac(_compute_ac(best_x_discrete), best_x_discrete)
    
    # Run L-BFGS from a few different random continuous initializations
    # We will run 3-5 restarts.
    
    num_restarts = 5
    max_iter_per_restart = 100
    
    for r in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Initialization
        if r == 0:
            x0 = np.ones(N)
        else:
            x0 = np.random.randn(N)
            # Normalize to have sum ~ N? Or just random.
            # The penalty term will push magnitude to 1.
            
        # Optimization
        try:
            res = minimize(
                _objective, 
                x0, 
                method='L-BFGS-B', 
                jac=None, # Numerical
                options={'maxiter': max_iter_per_restart, 'disp': False, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            x_opt = res.x
        except Exception:
            x_opt = x0
            
        # Project to discrete +/- 1
        x_discrete = np.sign(x_opt)
        x_discrete[x_discrete == 0] = 1
        
        # Evaluate discrete score
        score = _score_from_ac(_compute_ac(x_discrete), x_discrete)
        
        if score < best_score:
            best_score = score
            best_x_discrete = x_discrete
        
        # Local Polish: Single-bit flips
        # Try flipping each bit and see if it improves.
        # To speed up, we can update the autocorr incrementally?
        # Or just recompute. N=400, 400 flips * 400 evals = 160k ops? No, 400 evals of FFT.
        # 400 * 10us = 4ms. Very fast.
        
        improved = True
        while improved:
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            improved = False
            for i in range(N):
                if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                    break
                old_val = best_x_discrete[i]
                best_x_discrete[i] = -old_val
                new_score = _score_from_ac(_compute_ac(best_x_discrete), best_x_discrete)
                if new_score < best_score:
                    best_score = new_score
                    improved = True
                else:
                    best_x_discrete[i] = old_val
                    
    return best_x_discrete

# To ensure the function is named construct and returns the sequence
if __name__ == "__main__":
    # Simple test
    x = construct()
    print("Constructed sequence sum:", np.sum(x))
    R = _compute_ac(x)
    n_fft = len(R)
    max_side = max(np.max(np.abs(R[1:N])), np.max(np.abs(R[n_fft-N+1:n_fft])))
    s = np.sum(x)
    score = max_side / (s*s)
    print("Score:", score)
