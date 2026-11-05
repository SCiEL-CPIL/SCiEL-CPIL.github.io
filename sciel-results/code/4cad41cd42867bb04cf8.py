import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    def score(h):
        s = np.sum(h)
        if s == 0 or not np.isfinite(s):
            return float('inf')
        # Use FFT for convolution for speed during optimization
        # z = np.convolve(h, h)
        # For N=400, direct is fine, but FFT is safer for many iterations
        # Actually, for N=400, np.convolve is very fast.
        # But we need gradients. Let's use FFT based computation for the objective function
        # to allow vectorized gradient calculation if needed, or just use numerical 
        # gradients via finite differences if analytic is too complex?
        # L-BFGS-B can use analytic gradients.
        
        # Let's stick to a robust analytic gradient approach for the surrogate.
        pass

    # The objective is:
    # V(h) = abs(2 * N * max(z) / (sum(h))^2)
    # where z = convolve(h, h).
    
    # Since we want to minimize V, and V is scale invariant, we can fix sum(h) = 1.
    # Then V = 2 * N * max(z).
    # We want to minimize max(z) subject to sum(h)=1.
    
    # The function max(z) is convex. We can use a smooth surrogate:
    # f_beta(z) = (1/beta) * log(sum(exp(beta * z)))
    # This approximates max(z) for large beta.
    
    # To compute z and its gradient efficiently, we use FFT.
    # z[k] = sum_{i+j=k} h[i] h[j].
    # In frequency domain: Z(f) = H(f) * H(f) (with proper scaling).
    # Actually, it's easier to compute z in time domain for N=400?
    # N=400 is small. np.convolve is O(N^2) = 160,000 ops.
    # 1000 iterations * 160,000 = 160M ops. This is fast in Python/NumPy.
    # Let's use np.convolve for simplicity and correctness, and compute gradients analytically.
    
    def get_conv(h):
        return np.convolve(h, h)
    
    def objective_and_grad(h, beta):
        # h is normalized to sum=1 externally? 
        # No, let's handle normalization inside or fix it.
        # It's better to optimize h directly and normalize at the end?
        # No, scale invariance means we can fix sum(h)=1.
        # Let's project h to sum=1 at each step? Or just fix it by removing the mean?
        # If we fix sum(h)=1, the search space is N-1 dimensional.
        # L-BFGS-B handles equality constraints poorly.
        # Let's just optimize h and include the normalization in the objective?
        # V(h) = 2*N * max(z) / (sum(h))^2.
        # This is homogeneous of degree 0.
        # Gradient of a homogeneous function of degree 0 is orthogonal to h.
        # So we can just optimize h freely.
        
        z = get_conv(h)
        s = np.sum(h)
        
        # Surrogate for max(z)
        # Shift z to avoid overflow in exp
        z_max = np.max(z)
        z_shifted = z - z_max
        exp_z = np.exp(beta * z_shifted)
        sum_exp = np.sum(exp_z)
        
        # f = z_max + (1/beta) * log(sum_exp)
        f_val = z_max + (1.0 / beta) * np.log(sum_exp)
        
        # Value
        val = 2.0 * N * f_val / (s * s)
        
        # Gradient
        # df/dz_j = (1/beta) * exp(beta * (z_j - z_max)) / sum_exp = exp_z[j] / sum_exp
        # Note: d z_max / d z_j is 0 if z_j < z_max, 1 if z_j == z_max (subgradient).
        # The softmax surrogate handles this smoothly.
        # However, the term z_max in f_val is not differentiable.
        # Standard softmax approximation: f = (1/beta) log(sum(exp(beta z))).
        # This is always smooth. Let's use this instead of the shifted version with explicit max.
        # f = (1/beta) * log(sum(exp(beta * z)))
        
        # Let's re-calculate with pure softmax to ensure smoothness.
        # To prevent overflow, we can scale, but for N=400 and reasonable h, it might be okay.
        # Let's use the shifted version but be careful with the gradient of the shift.
        # Actually, the gradient of (1/beta) log(sum(exp(beta z))) is simply:
        # grad_f_z = exp(beta z) / sum(exp(beta z))
        # This is the softmax probability vector.
        
        # Let's compute this properly.
        # To avoid overflow, we use the max shift trick, but the gradient formula 
        # for the shifted version is the same: p_j = exp(beta(z_j - z_max)) / sum(exp(beta(z - z_max)))
        # Because the max shifts both numerator and denominator.
        
        p = exp_z / sum_exp
        
        # dV/dz_j = 2*N / (s*s) * p_j
        
        # Now, d z_k / d h_i.
        # z = convolve(h, h).
        # z_k = sum_{i+j=k} h_i h_j.
        # d z_k / d h_i = sum_{j=k-i} (delta_{i,j} h_j + h_i delta_{j,k-i}?)
        # d/dh_i (h_i h_j) = delta_{i,j} h_j + h_i delta_{i,j} = 2 h_j if i=j?
        # No.
        # z_k = sum_{i} h_i h_{k-i} (where h_x=0 for x<0 or x>=N).
        # d z_k / d h_m = sum_{i} (delta_{i,m} h_{k-i} + h_i delta_{k-i,m})
        # = h_{k-m} + h_{k-m} = 2 h_{k-m} (if indices valid).
        # Wait, if i=m, term is h_m h_{k-m}. Deriv is h_{k-m}.
        # If k-i=m => i=k-m, term is h_{k-m} h_m. Deriv is h_m.
        # So d z_k / d h_m = 2 h_{k-m} if 0 <= k-m < N, else 0.
        # Note: If k-m = m, i.e., 2m=k, we have h_m^2. Deriv is 2h_m.
        # The formula 2 h_{k-m} covers this.
        
        # So grad_h_m = sum_k (dV/dz_k) * (d z_k / d h_m)
        # = sum_k (2*N/s^2 * p_k) * (2 h_{k-m})
        # = (4*N/s^2) * sum_k p_k h_{k-m}
        
        # This is a convolution!
        # Let q = 2*N/s^2 * p.
        # Then grad_h = 2 * convolve(q, h) ??
        # sum_k p_k h_{k-m}.
        # Let j = k-m => k = j+m.
        # sum_j p_{j+m} h_j.
        # This is cross-correlation of p and h?
        # convolve(p, h)[m] = sum_j p_j h_{m-j}.
        # We want sum_j p_{j+m} h_j.
        # Let p_rev = p[::-1].
        # convolve(p_rev, h)[m + N - 1]?
        
        # Let's just compute it directly for N=400. It's O(N^2).
        # grad_h = np.zeros(N)
        # for m in range(N):
        #     # sum over k such that k-m is in [0, N-1]
        #     # k ranges from m to m+N-1
        #     # p is length 2N-1
        #     # h is length N
        #     # term is p[k] * h[k-m]
        #     # k-m is index in h. Let idx_h = k-m.
        #     # idx_h in [0, N-1] => k in [m, m+N-1]
        #     # p index is k.
        #     # sum_{idx_h=0}^{N-1} p[idx_h + m] * h[idx_h]
        
        # This is efficient enough.
        
        factor = 4.0 * N / (s * s)
        # p is length 2N-1
        # h is length N
        # grad_h[m] = factor * sum_{j=0}^{N-1} p[m+j] * h[j]
        
        # We can compute this using matrix multiplication or loops.
        # Since N=400, a loop is fine, or we can use FFT?
        # Let's use a simple loop for clarity and safety, it's fast enough.
        
        grad_h = np.zeros(N)
        for m in range(N):
            # p[m : m+N]
            grad_h[m] = factor * np.dot(p[m:m+N], h)
            
        return val, grad_h

    # Optimization Strategy
    # 1. Generate diverse initial points.
    # 2. Use L-BFGS-B to minimize the smooth surrogate.
    # 3. Increase beta over time to sharpen the max approximation.
    
    best_h = np.ones(N)
    best_val = score(best_h) # This uses the exact max, not surrogate.
    # We need a fast exact score for comparison.
    def exact_score(h):
        s = np.sum(h)
        if s == 0: return float('inf')
        z = np.convolve(h, h)
        return abs(2 * N * np.max(z) / (s * s))
    
    best_val = exact_score(best_h)
    
    rng = np.random.RandomState(42)
    
    # Initial candidates
    candidates = [
        np.ones(N),
        rng.randn(N),
        np.cos(2 * np.pi * np.arange(N) * 5.0 / N),
        np.sin(2 * np.pi * np.arange(N) * 5.0 / N),
        np.sign(rng.randn(N)),
        # Chirp
        np.cos(2 * np.pi * (np.arange(N)**2) / N)
    ]
    
    # Filter valid candidates
    valid_candidates = []
    for c in candidates:
        if abs(np.sum(c)) > 1e-6 and np.all(np.isfinite(c)):
            valid_candidates.append(c)
    
    if not valid_candidates:
        valid_candidates = [np.ones(N)]
    
    # Run optimization from each candidate
    # Budget: ~140s.
    # Let's allow 5-10 seconds per candidate.
    
    for c in valid_candidates:
        if time.time() > deadline - 20: # Keep 20s for final checks
            break
            
        h0 = c.copy()
        
        # Normalize to sum=1 to help scaling?
        # The objective is scale invariant, but gradients might scale.
        # Let's normalize h0 to sum=1.
        s0 = np.sum(h0)
        if s0 != 0:
            h0 = h0 / s0
            
        # Optimize with increasing beta
        # Start with beta=10, go up to 100.
        # Each run: minimize.
        
        current_h = h0.copy()
        
        for beta in [10, 20, 50, 100, 200]:
            if time.time() > deadline - 10:
                break
                
            # Define objective and gradient for L-BFGS-B
            def obj_func(h_flat, beta_val=beta):
                val, grad = objective_and_grad(h_flat, beta_val)
                return val, grad
                
            # L-BFGS-B expects f and grad separately or a function returning both?
            # scipy.optimize.minimize(fun, x0, jac=...)
            # If jac returns tuple, it's treated as (f, g).
            
            try:
                res = minimize(
                    objective_and_grad, 
                    current_h, 
                    method='L-BFGS-B', 
                    options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-6, 'disp': False}
                )
                current_h = res.x
            except Exception:
                pass
            
            # Check exact score
            val_exact = exact_score(current_h)
            if val_exact < best_val:
                best_val = val_exact
                best_h = current_h.copy()
                
    # Final local refinement on the best found
    # Do a few more steps with high beta if time permits
    if time.time() < deadline - 5:
        beta = 500
        def final_obj(h_flat):
            val, grad = objective_and_grad(h_flat, beta)
            return val, grad
            
        try:
            res = minimize(
                final_obj, 
                best_h, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            h_final = res.x
            val_final = exact_score(h_final)
            if val_final < best_val:
                best_h = h_final
                best_val = val_final
        except Exception:
            pass
            
    return best_h
