"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
from scipy.signal import fftconvolve

def _autoconv(x):
    """Compute autoconvolution of x (linear, full)."""
    return fftconvolve(x, x, mode='full')

def _c3prime(x):
    """Compute C3' = max|autoconv| / (sum(x))^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e12
    ac = _autoconv(x)
    return np.max(np.abs(ac)) / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    best_x = None
    best_score = np.inf
    
    # Try several initializations
    rng = np.random.RandomState(42)
    
    initializations = []
    
    # All ones
    initializations.append(np.ones(N))
    
    # Random signs
    initializations.append(np.sign(rng.randn(N)))
    
    # Block patterns
    for block_size in [10, 20, 40, 50, 80, 100]:
        pattern = np.zeros(N)
        for i in range(0, N, 2 * block_size):
            pattern[i:i+block_size] = 1
            if i + 2*block_size <= N:
                pattern[i+block_size:i+2*block_size] = -1
        # Trim to N
        pattern = pattern[:N]
        initializations.append(pattern)
    
    # Sine wave patterns
    for freq in [0.1, 0.2, 0.3, 0.5]:
        t = np.arange(N)
        initializations.append(np.sign(np.sin(2 * np.pi * freq * t)))
    
    # Alternating blocks
    for block_size in [5, 10, 20]:
        pattern = np.zeros(N)
        for i in range(0, N, block_size):
            val = 1 if (i // block_size) % 2 == 0 else -1
            pattern[i:min(i+block_size, N)] = val
        initializations.append(pattern)
    
    def _optimize(x0, max_iter=1000, tol=1e-8):
        """Optimize using L-BFGS-B with soft max approximation."""
        x0 = np.array(x0, dtype=float)
        
        # We want to minimize max|autoconv(x)| / (sum(x))^2
        # Use a smooth approximation: (1/beta) * log(sum(exp(beta*|ac_k|))) / (sum(x))^2
        # But this is complex. Instead, use auxiliary variable formulation.
        
        # Alternative: minimize t subject to |autoconv_k| <= t * (sum(x))^2 for all k
        # This is non-linear. Let's use a different approach.
        
        # Direct approach: minimize f(x) = max_k |ac_k(x)| / (sum(x))^2
        # Use subgradient or smooth approximation
        
        # Smooth approximation via log-sum-exp:
        # f_smooth(x) = (1/beta) * log(sum_k exp(beta * |ac_k(x)|)) / (sum(x))^2
        
        beta = 50.0
        
        def _objective_and_grad(x):
            s = np.sum(x)
            if abs(s) < 1e-10:
                return 1e12, np.zeros_like(x)
            
            ac = _autoconv(x)
            abs_ac = np.abs(ac)
            
            # Log-sum-exp for stability
            max_val = np.max(abs_ac)
            if max_val > 1e10:
                return 1e12, np.zeros_like(x)
            
            log_sum_exp = max_val + np.log(np.sum(np.exp(abs_ac - max_val)))
            f = log_sum_exp / beta / (s * s)
            
            # Gradient computation
            # d/dx_k of log_sum_exp = sum_j exp(beta*|ac_j|) * d|ac_j|/dx_k / sum_j exp(beta*|ac_j|)
            # d|ac_j|/dx_k = sign(ac_j) * d(ac_j)/dx_k
            # ac_j = sum_i x_i x_{j-i} (with appropriate bounds)
            # d(ac_j)/dx_k = x_{j-k} + x_{k-j} (if in range)
            
            exp_vals = np.exp(abs_ac - max_val)
            weights = exp_vals / np.sum(exp_vals)
            
            # Gradient of log_sum_exp w.r.t. x
            grad_lse = np.zeros_like(x)
            for k in range(N):
                # d(ac_j)/dx_k for j in [0, 2N-2]
                # ac_j = sum_{i} x_i x_{j-i} where 0<=i<N and 0<=j-i<N
                # d(ac_j)/dx_k = x_{j-k} + x_{k-j} if indices valid, else just one term
                # Actually ac_j = sum_{i=max(0,j-N+1)}^{min(N-1,j)} x_i x_{j-i}
                # d(ac_j)/dx_k = x_{j-k} (if 0<=j-k<N) + x_{k-j} (if 0<=k-j<N)
                # But if j-k == k-j (i.e., j=2k), it's 2*x_k... no, let's be careful.
                
                # Let's compute it properly.
                # ac[j] = sum_{i=0}^{N-1} x[i] * x[j-i] where x[m]=0 if m<0 or m>=N
                # d(ac[j])/dx[k] = x[j-k] + x[k-j] (but if j-k==k-j, i.e., j=2k, then it's 2*x[k]... 
                # wait, no: d/dx_k of x[i]*x[j-i] is delta_{ik}*x[j-i] + x[i]*delta_{j-i,k}
                # = x[j-k] (if k in [0,N-1] and j-k in [0,N-1]) + x[k-j] (if k in [0,N-1] and k-j in [0,N-1])
                # If j-k == k-j, i.e., j=2k, then both terms are x[k], so total is 2*x[k].
                
                term1 = 0.0
                if 0 <= j - k < N:
                    term1 = x[j - k]
                term2 = 0.0
                if 0 <= k - j < N:
                    term2 = x[k - j]
                
                # But we need to sum over j with weights
                pass
            
            # This O(N^2) gradient is too slow. Let's use FFT-based gradient.
            
            # Actually, let's just return a numerical gradient or use a simpler approach.
            # For now, let's just return the objective and use finite differences or a simpler optimizer.
            
            return f, None
        
        # Since analytical gradient is complex, let's use a different strategy:
        # Use coordinate descent with FFT acceleration
        
        x = x0.copy()
        current_obj = _c3prime(x)
        
        for it in range(max_iter):
            if time.time() > deadline - 10.0:
                break
            
            improved = False
            for k in range(N):
                # Try flipping x[k]
                old_val = x[k]
                x[k] = -old_val
                new_obj = _c3prime(x)
                if new_obj < current_obj:
                    current_obj = new_obj
                    improved = True
                else:
                    x[k] = old_val
            
            if not improved:
                break
        
        return x, current_obj
    
    # Try each initialization
    for x0 in initializations:
        if time.time() > deadline - 20.0:
            break
        
        x_opt, score = _optimize(x0, max_iter=50)
        
        if score < best_score:
            best_score = score
            best_x = x_opt.copy()
    
    # Final polish with more iterations if time permits
    if best_x is not None and time.time() < deadline - 30.0:
        x_final, score_final = _optimize(best_x, max_iter=200)
        if score_final < best_score:
            best_score = score_final
            best_x = x_final
    
    if best_x is None:
        return np.ones(N)
    
    # Ensure the result is a proper step function (values are +/-1)
    result = np.sign(best_x)
    # If all zeros, default to ones
    if np.all(result == 0):
        result = np.ones(N)
    
    return result
# EVOLVE-BLOCK-END
