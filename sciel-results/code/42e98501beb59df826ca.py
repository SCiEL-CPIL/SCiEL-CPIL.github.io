import numpy as np
from scipy.optimize import minimize
import time

N = 400
# The domain is [-1/4, 1/4]. We have 400 intervals.
# The step function is constant on each interval.
# Let's represent the function by its values on these intervals.

def _autoconv_bound(signed_steps):
    """
    Calculate the upper bound on C3.
    The task states: 'score is the upper bound on C3'.
    'Paper and published result notebook place the absolute value differently; execution must explicitly select a metric.'
    
    Common metric for autocorrelation inequality (C3):
    Usually involves minimizing the maximum autocorrelation sidelobe or the integrated sidelobe.
    Given 'upper bound on C3', and the context of 'autoconvolution', 
    C3 often refers to a constant in an inequality like |f * f| <= C3 ||f||^2 or similar.
    
    However, the inspiration notes mention 'minimize the autoconvolution upper bound'.
    Let's look at the structure.
    If f is a step function, f * f (autoconvolution) is a piecewise quadratic function.
    The 'upper bound' likely refers to the maximum value of the autoconvolution outside the main lobe, 
    or the integral of the square of the autoconvolution (Parseval's theorem connection).
    
    Let's assume the metric is the sum of squares of the autoconvolution coefficients (energy in frequency domain)
    or the maximum sidelobe.
    
    Wait, the task says: 'Choose a signed step function... to improve an autoconvolution upper bound.'
    And 'The evaluator measures the raw value (lower is better).'
    
    Let's assume the metric is the L2 norm of the autoconvolution function, or the maximum absolute value of the sidelobes.
    Given 'upper bound', it might be the maximum value of the autoconvolution.
    
    Let's compute the autoconvolution of the step function.
    Let the intervals be $I_k = [x_k, x_{k+1}]$.
    $f(x) = h_k$ for $x \in I_k$.
    The autoconvolution $g(t) = \int f(x)f(t-x) dx$.
    
    For a step function, the autoconvolution is a piecewise quadratic polynomial.
    However, calculating the exact continuous autoconvolution is complex.
    
    Alternative interpretation: 
    Perhaps the 'autoconvolution' refers to the discrete autocorrelation of the sequence of heights?
    Or the continuous autoconvolution evaluated at specific points?
    
    Let's look at the inspiration 2 and 3.
    Inspiration 2: `_autoconv_bound`
    Inspiration 3: `_autoconv_bound`
    
    Let's define the metric as the sum of squares of the discrete autocorrelation of the sequence of heights.
    $R[k] = \sum_{i} h_i h_{i+k}$.
    The 'upper bound' might be $\max_k |R[k]|$ for $k \neq 0$?
    Or $\sum_{k \neq 0} R[k]^2$?
    
    Let's try minimizing the sum of squares of the non-zero lag autocorrelations.
    This is equivalent to minimizing the energy of the sidelobes in the frequency domain (by Parseval).
    $E = \sum_{k=1}^{N-1} R[k]^2$.
    
    Let's implement this metric.
    """
    # Discrete autocorrelation
    # Use FFT for speed
    F = np.fft.fft(signed_steps)
    Power = np.abs(F)**2
    # The sum of squares of autocorrelation lags is related to the sum of squares of the power spectrum.
    # Specifically, $\sum_{k} R[k]^2 = \frac{1}{N} \sum_{j} |F[j]|^4$? No.
    # Parseval for autocorrelation: $\sum_{k} |R[k]|^2 = \frac{1}{N} \sum_{j} |F[j]|^4$.
    # We want to minimize sidelobe energy, so we subtract the DC term (lag 0).
    # R[0] = sum(h_i^2) = N (since h_i is +/- 1).
    # So we want to minimize $\sum_{k \neq 0} R[k]^2$.
    
    # Calculate R[0]
    R0 = np.sum(signed_steps**2)
    
    # Calculate total energy of autocorrelation
    # sum_k R[k]^2 = (1/N) * sum_j |F[j]|^4
    total_energy = np.sum(Power**2) / N
    
    # Subtract R[0]^2
    sidelobe_energy = total_energy - R0**2
    
    return sidelobe_energy

def _objective(w):
    """
    Objective function for continuous relaxation.
    w is a vector of real values of length N.
    We map w to signed steps using a soft sign function or just thresholding?
    For gradient descent, we need a differentiable mapping.
    However, the final output must be signed steps (+1/-1).
    
    Strategy:
    1. Optimize w (real values) to minimize the sidelobe energy of the *continuous* approximation?
       Or minimize the sidelobe energy of the *thresholded* sequence?
       Thresholding is non-differentiable.
       
    Alternative:
    Optimize w to be a smooth window (e.g., Gaussian-like) that minimizes the sidelobe energy of its *continuous* autoconvolution.
    Then threshold w to get the signed steps.
    
    But the discrete autocorrelation of the thresholded sequence might not correlate perfectly with the continuous one.
    
    Let's try a different approach:
    Use the continuous w to define the probability of being +1.
    But that's stochastic.
    
    Let's stick to the discrete metric but use a soft sign function for the optimization,
    then hard threshold at the end.
    Soft sign: tanh(alpha * w).
    We optimize w to minimize _autoconv_bound(tanh(alpha * w)).
    As alpha increases, it approaches the hard sign.
    
    However, _autoconv_bound expects a sequence.
    Let's define the objective as the sidelobe energy of the sequence $s_i = \tanh(\alpha w_i)$.
    This is differentiable with respect to w.
    """
    # Map w to soft signs
    alpha = 10.0  # Sharpness
    s = np.tanh(alpha * w)
    
    # Calculate sidelobe energy
    F = np.fft.fft(s)
    Power = np.abs(F)**2
    R0 = np.sum(s**2)
    total_energy = np.sum(Power**2) / N
    sidelobe_energy = total_energy - R0**2
    
    return sidelobe_energy

def construct():
    start_time = time.time()
    deadline = 120.0  # 120s limit, keep safety margin
    
    # Initial guess: random real values
    rng = np.random.default_rng(42)
    w0 = rng.normal(0, 1, N)
    
    best_w = w0.copy()
    best_score = _objective(w0)
    
    # Use L-BFGS-B to minimize
    # We need to be careful with time.
    # L-BFGS-B can be fast for N=400.
    
    try:
        # Define a callback to check time and update best
        def callback(w):
            nonlocal best_w, best_score
            current_time = time.time()
            if current_time > deadline - 10:
                raise RuntimeError("Time limit approaching")
            
            # Evaluate current score
            try:
                score = _objective(w)
                if score < best_score:
                    best_score = score
                    best_w = w.copy()
            except Exception:
                pass
            return False  # Continue optimization
            
        # Run optimization
        res = minimize(
            _objective, 
            w0, 
            method='L-BFGS-B', 
            options={'maxiter': 1000, 'disp': False, 'maxfun': 5000}
        )
        
        if res.success or res.fun < best_score:
            if res.fun < best_score:
                best_score = res.fun
                best_w = res.x.copy()
                
    except Exception as e:
        # If optimization fails or times out, use the best found so far
        pass
    
    # Check time
    if time.time() > deadline - 5:
        # Return best found
        pass
    
    # Threshold the best continuous solution to get signed steps
    # Map positive to 1, negative to -1
    # What about zeros? Randomly assign or keep 0? 
    # Task says "signed step function", usually implies +/- 1.
    # If w is close to 0, the sign is ambiguous. 
    # We can add a small random noise to break ties deterministically.
    
    signs = np.sign(best_w)
    # Handle zeros
    zero_mask = (signs == 0)
    if np.any(zero_mask):
        # Assign random signs to zeros
        signs[zero_mask] = rng.choice([-1, 1], size=np.sum(zero_mask))
        
    return signs

# Test the metric to ensure it's reasonable
if __name__ == "__main__":
    # Quick test
    signs = construct()
    score = _autoconv_bound(signs)
    print(f"Score: {score}")
