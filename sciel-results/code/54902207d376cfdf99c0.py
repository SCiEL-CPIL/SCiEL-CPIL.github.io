import numpy as np
from scipy.optimize import minimize
import time

N = 400

def construct():
    start_time = time.time()
    timeout = 140  # Leave 10s margin for 150s limit
    
    # Variables: Real and Imaginary parts of H[1] to H[N//2 - 1]
    # H[0] is fixed to 1 (sum h = 1)
    # H[N//2] is real, so it's one variable.
    # Total variables: (N//2 - 1) * 2 + 1
    
    n_vars = (N // 2 - 1) * 2 + 1
    # Indices for H[1] ... H[N//2 - 1]
    # Let's just define the vector x and map it to H
    
    def x_to_H(x):
        H = np.zeros(N, dtype=complex)
        H[0] = 1.0
        # H[N//2] is real
        H[N//2] = x[0]
        idx = 1
        for k in range(1, N // 2):
            H[k] = x[idx] + 1j * x[idx + 1]
            H[N - k] = x[idx] - 1j * x[idx + 1]
            idx += 2
        return H
    
    def H_to_h(H):
        return np.real(np.fft.ifft(H))
    
    def objective(x):
        H = x_to_H(x)
        h = H_to_h(H)
        # Check sum h
        # s = np.sum(h)
        # If s is close to 0, return large value
        # s should be 1.0 due to H[0]=1
        
        c = np.convolve(h, h)
        # Soft max
        beta = 50.0
        # c can be negative? Yes.
        # max(c) is what we want.
        # To avoid overflow in exp, subtract max
        c_max = np.max(c)
        # If c_max is very negative, exp might underflow, which is fine (max approx 0)
        # But we want the true max.
        # Using log-sum-exp:
        # max(c) approx (1/beta) * log(sum(exp(beta * c)))
        # To be stable:
        c_shifted = c - c_max
        sum_exp = np.sum(np.exp(beta * c_shifted))
        soft_max = c_max + (1.0 / beta) * np.log(sum_exp)
        
        val = 2.0 * N * soft_max
        return val
    
    # Initial guess: H[0]=1, others 0.
    # This corresponds to x = 0
    x0 = np.zeros(n_vars)
    
    # We can try a few random restarts or just one good run
    # Let's use L-BFGS-B
    
    best_x = x0
    best_val = objective(x0)
    
    # Optimization
    # Use a callback to check time
    def callback(xk):
        nonlocal best_val
        nonlocal best_x
        # Calculate current value
        val = objective(xk)
        if val < best_val:
            best_val = val
            best_x = xk
        if time.time() - start_time > timeout:
            raise RuntimeError("Timeout")
            
    try:
        res = minimize(objective, x0, method='L-BFGS-B', callback=callback, options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8})
    except RuntimeError:
        pass
    except Exception:
        pass
    
    # If we didn't improve, or if we want to try a different strategy:
    # The "flat c" target suggests we want H to have specific structure.
    # Let's just return the best found.
    
    H_best = x_to_H(best_x)
    h_best = H_to_h(H_best)
    
    # Ensure finite and sum != 0
    if not np.all(np.isfinite(h_best)) or np.sum(h_best) == 0:
        return np.ones(N)
        
    return h_best
