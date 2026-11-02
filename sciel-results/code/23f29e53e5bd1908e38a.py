import numpy as np
import time
import sys

N = 600
# Time budget: 150s limit, use 140s for safety
TIME_LIMIT = 140.0
START_TIME = time.time()

def compute_score(heights):
    """
    Computes the normalized maximum of the autoconvolution.
    Returns the score (lower is better).
    """
    if len(heights) != N:
        return float('inf')
    if np.any(heights < 0):
        return float('inf')
    
    # Normalize the step function. 
    # The task says "normalized maximum". 
    # Standard normalization for such problems usually involves the L1 norm or L2 norm.
    # Given the context of "autoconvolution" and "upper bound on C1", 
    # and the fact that uniform [1]*N scores 1.0, let's deduce the normalization.
    
    # If f is uniform 1 on N intervals, area = N * (1/N) = 1? No, interval length is 1/300.
    # Let's assume the heights are coefficients of the step function.
    # Step function f(x) = h_i on interval i.
    # Autoconvolution (f*f)(t) = integral f(s)f(t-s) ds.
    # Discrete approximation: sum_i h_i h_j * (interval_length) * (number of pairs) ...
    
    # Let's look at the seed: [1.0]*N gives score 1.0.
    # If we scale h by k, the autoconvolution scales by k^2.
    # If normalization is L1 norm: ||f||_1 = sum h_i * dx.
    # If normalization is L2 norm: ||f||_2^2 = sum h_i^2 * dx.
    
    # Let's test the L1 normalization hypothesis.
    # If f=1, ||f||_1 = 1 (assuming dx=1/N? No, domain is [-1/4, 1/4], length 1/2. 600 intervals. dx = 1/1200).
    # Wait, the domain is [-1/4, 1/4]. Length = 0.5.
    # N=600. dx = 0.5/600 = 1/1200.
    # If f=1, ||f||_1 = 600 * (1/1200) = 0.5.
    # Autoconvolution at t=0: integral f(s)f(-s) ds. For f=1, f(s)f(-s)=1 on support.
    # Support of f is [-0.25, 0.25]. f*f support is [-0.5, 0.5].
    # At t=0, integral is length of intersection of [s in -0.25,0.25] and [t-s in -0.25,0.25] -> s in [-0.25, 0.25].
    # Length is 0.5.
    # So (f*f)(0) = 0.5.
    # Normalized max: max(f*f) / ||f||_1^2 ?
    # 0.5 / (0.5)^2 = 0.5 / 0.25 = 2.0.
    # But seed score is 1.0.
    
    # Maybe normalization is different.
    # Perhaps "normalized" means max(f*f) / max(f)^2 ? No, that would be 1 for uniform.
    # Or max(f*f) / (sum h_i)^2 ?
    # Let's check the wording: "minimize the normalized maximum of its autoconvolution".
    # Often in these inequality problems (like Littlewood-Offord or similar), the normalization is by the L1 norm squared or similar to make it scale invariant.
    
    # Let's look at Inspiration 1: "Initial program... score 1.0182".
    # Inspiration 2: "score 1.0034".
    # Inspiration 3: "score 1.0009".
    # The seed [1.0]*N scores 1.0.
    
    # If the score is max(f*f) / ||f||_1^2, uniform gives 2.0.
    # If the score is max(f*f) / ||f||_2^2, uniform:
    # ||f||_2^2 = sum 1^2 * dx = 600 * (1/1200) = 0.5.
    # max(f*f) = 0.5.
    # 0.5 / 0.5 = 1.0.
    # This matches!
    
    # So the metric is: max(f*f) / ||f||_2^2.
    # Let's verify.
    # f is a step function with heights h_i on intervals of length dx.
    # ||f||_2^2 = dx * sum(h_i^2).
    # (f*f)(t) is a piecewise linear function.
    # We need to compute the maximum of f*f.
    
    # Efficient computation of f*f:
    # f*f is the convolution of two step functions.
    # It is a piecewise quadratic? No, convolution of two step functions is piecewise linear (trapezoids).
    # Actually, f*f(t) = integral f(s)f(t-s) ds.
    # Since f is constant on intervals, f*f is a sum of shifted rectangles, resulting in a piecewise linear function with slopes changing at the grid points.
    # The maximum will occur at a grid point or in between?
    # For non-negative f, f*f is unimodal? Not necessarily, but often is.
    # However, we can compute f*f exactly at the grid points and check midpoints if needed.
    # Actually, the convolution of two step functions with step size dx is a sequence of values at t_k = k*dx.
    # (f*f)(t_k) = dx * sum_i h_i h_{i+k} (with appropriate indices).
    # This is the discrete auto-correlation scaled by dx.
    # Let A_k = sum_i h_i h_{i+k}.
    # Then (f*f)(t_k) = dx * A_k.
    # The function f*f is linear between t_k and t_{k+1}.
    # The maximum of a linear function on an interval is at the endpoints.
    # So we only need to check the values at the grid points t_k.
    
    # Indices:
    # h is length N.
    # k ranges from -N+1 to N-1.
    # A_k = sum_{i} h[i] h[i+k] where indices are valid.
    
    h = np.array(heights, dtype=float)
    norm_sq = dx * np.sum(h**2)
    if norm_sq == 0:
        return float('inf')
        
    # Compute auto-correlation
    # Use FFT for speed? N=600, direct sum is 600*1200 = 720k ops. Very fast.
    # But let's use FFT to be safe and fast.
    # Zero pad to 2N.
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    auto_corr = np.fft.irfft(power, n=n_fft)
    
    # The auto_corr from FFT is circular. We need the linear part.
    # The linear auto-correlation A_k for k=0..N-1 is in auto_corr[0..N-1].
    # For k=-(N-1)..-1, it is in auto_corr[N..2N-1] (reversed/conjugate symmetric).
    # Since h is real, auto_corr is symmetric: A_{-k} = A_k.
    # So we just need max(auto_corr[0:N])?
    # Wait, A_k = sum h_i h_{i+k}.
    # The peak is usually at k=0 (central).
    # Let's just take the max of the first N elements.
    
    max_conv = dx * np.max(auto_corr[:N])
    
    score = max_conv / norm_sq
    return score

def construct():
    # Strategy: Simulated Annealing on a parametric basis.
    # We represent the step function h as a linear combination of basis functions.
    # To ensure non-negativity, we can use h = exp(basis) or h = basis^2 + epsilon.
    # Or simpler: optimize the coefficients of a cosine series directly, then project to non-negative.
    # However, projecting might break the smoothness.
    
    # Better: Represent h_i = c_0 + sum_{k=1}^K c_k cos(2*pi*k*i/N) + sum_{k=1}^K s_k sin(2*pi*k*i/N).
    # This guarantees a smooth periodic function. But our function is on a finite interval, not periodic.
    # We can use a windowed cosine series or just optimize the raw heights with constraints.
    
    # Alternative: Directly optimize the heights h using gradient descent on a smoothed objective.
    # Objective: J(h) = max_k (dx * A_k) / (dx * sum h_i^2).
    # This is non-differentiable due to max.
    # Use log-sum-exp approximation for max:
    # max(x) approx (1/T) log(sum(exp(T*x_i))).
    # Then J_smooth(h) = (1/T) log(sum_k exp(T * dx * A_k)) / (dx * sum h_i^2).
    # This is differentiable.
    
    # We will use L-BFGS-B from scipy.optimize to minimize J_smooth(h) with bounds h_i >= 0.
    
    start = time.time()
    
    # Initial guess: Uniform
    h0 = np.ones(N)
    
    # Temperature for log-sum-exp
    T = 100.0
    
    def objective(h):
        # h is flattened array of size N
        if np.any(h < 0):
            return 1e10
        norm_sq = dx * np.sum(h**2)
        if norm_sq < 1e-10:
            return 1e10
        
        # Compute auto-correlation
        n_fft = 2 * N
        F = np.fft.rfft(h, n=n_fft)
        power = F * np.conj(F)
        auto_corr = np.fft.irfft(power, n=n_fft)
        
        # We care about the linear auto-correlation part
        # Values are dx * A_k
        vals = dx * auto_corr[:N]
        
        # Log-sum-exp for max
        # max(vals) approx (1/T) * log(sum(exp(T*vals)))
        # To avoid overflow, subtract max
        v_max = np.max(vals)
        if v_max == 0:
            return 0.0
        log_sum_exp = v_max + (1.0/T) * np.log(np.sum(np.exp(T * (vals - v_max))))
        
        # Objective is log_sum_exp / norm_sq
        # Note: log_sum_exp has units of value^2? No, vals are values of convolution.
        # norm_sq is value^2 * length?
        # Let's check units.
        # h is height.
        # auto_corr is sum h_i h_j. Units: h^2.
        # vals = dx * auto_corr. Units: h^2 * length.
        # norm_sq = dx * sum h^2. Units: h^2 * length.
        # Ratio is dimensionless.
        
        return log_sum_exp / norm_sq

    # Gradient
    def gradient(h):
        if np.any(h < 0):
            return np.zeros_like(h)
        norm_sq = dx * np.sum(h**2)
        if norm_sq < 1e-10:
            return np.zeros_like(h)
        
        n_fft = 2 * N
        F = np.fft.rfft(h, n=n_fft)
        power = F * np.conj(F)
        auto_corr = np.fft.irfft(power, n=n_fft)
        vals = dx * auto_corr[:N]
        
        v_max = np.max(vals)
        exp_vals = np.exp(T * (vals - v_max))
        sum_exp = np.sum(exp_vals)
        log_sum_exp = v_max + (1.0/T) * np.log(sum_exp)
        
        obj = log_sum_exp / norm_sq
        
        # d(obj)/dh
        # obj = L / S, where L = log_sum_exp, S = norm_sq
        # dL/dh = (1/L_SE_sum) * sum_k exp(T*vals_k) * d(vals_k)/dh
        # dS/dh = 2 * dx * h
        
        # d(vals_k)/dh_i
        # vals_k = dx * sum_j h_j h_{j+k}
        # d(vals_k)/dh_i = dx * (h_{i+k} + h_{i-k}) (with boundary handling)
        
        # This gradient is complex to compute analytically with FFT.
        # Let's use numerical gradient or a simpler optimization.
        # Given time constraints, let's try a simpler local search or just return uniform if time is tight.
        # But we want to improve.
        
        # Let's use scipy.optimize.minimize with 'L-BFGS-B' and analytic gradient if possible, 
        # or finite differences if not. Finite differences on 600 vars is 600 evals per step. 
        # Each eval is ~1ms. 600ms per step. 100 steps = 60s. Feasible.
        
        # For now, let's implement a simple gradient-free local search (Nelder-Mead is too slow for 600D).
        # Let's try coordinate descent?
        # Or just use the analytic gradient approximation.
        
        # Actually, let's stick to the plan: use scipy.optimize.minimize with method='L-BFGS-B'
        # and provide the analytic gradient.
        
        # d(vals_k)/dh_i:
        # vals_k = dx * sum_j h_j h_{j+k}
        # Let's define h_ext with zeros.
        # d/dh_i (sum_j h_j h_{j+k}) = h_{i+k} + h_{i-k}
        
        # We can compute this efficiently.
        # h_padded = np.zeros(2*N)
        # h_padded[:N] = h
        # d_vals = dx * (np.roll(h, -k) + np.roll(h, k)) ... for each k?
        # This is O(N^2).
        
        # Let's just use finite differences for the gradient in the optimization step.
        # It's safer and less code.
        
        # However, L-BFGS-B with finite differences might be slow.
        # Let's try a different approach: 
        # Since we suspect the optimal shape is not uniform, let's try a few parametric shapes
        # and pick the best one.
        
        # Shapes:
        # 1. Uniform (Score ~1.0)
        # 2. Triangular (h_i = 1 - |2i/N - 1|)
        # 3. Gaussian (h_i = exp(-((i - N/2)^2) / (2 * sigma^2)))
        # 4. Cosine (h_i = 0.5 * (1 + cos(2*pi*i/N)))
        
        # Let's evaluate these.
        
        candidates = []
        
        # Uniform
        candidates.append(np.ones(N))
        
        # Triangular
        i = np.arange(N)
        tri = 1.0 - np.abs(2.0*i/N - 1.0)
        candidates.append(tri)
        
        # Gaussian
        sigma = N / 4.0
        gauss = np.exp(-((i - N/2.0)**2) / (2.0 * sigma**2))
        candidates.append(gauss)
        
        # Cosine
        cos_shape = 0.5 * (1.0 + np.cos(2.0 * np.pi * i / N))
        candidates.append(cos_shape)
        
        # Sinc-like (not non-negative everywhere, so clip)
        # sinc is oscillatory, skip.
        
        best_h = None
        best_score = float('inf')
        
        for h_cand in candidates:
            # Ensure non-negative
            h_cand = np.maximum(h_cand, 0)
            s = compute_score(h_cand)
            if s < best_score:
                best_score = s
                best_h = h_cand
        
        # Now, try to refine the best candidate with a few steps of gradient descent 
        # using a smoothed objective and finite differences.
        
        if best_h is not None:
            h_opt = best_h.copy()
            
            # Simple gradient descent with line search
            lr = 0.01
            for step in range(20):
                if time.time() - start > TIME_LIMIT - 10:
                    break
                    
                # Compute gradient via central differences
                grad = np.zeros_like(h_opt)
                eps = 1e-4
                for j in range(0, N, 10): # Sparse gradient to save time
                    h_plus = h_opt.copy()
                    h_minus = h_opt.copy()
                    h_plus[j] += eps
                    h_minus[j] -= eps
                    if h_minus[j] < 0:
                        h_minus[j] = 0
                        grad[j] = (objective(h_plus) - objective(h_opt)) / eps
                    else:
                        grad[j] = (objective(h_plus) - objective(h_minus)) / (2*eps)
                
                # Update
                h_opt -= lr * grad
                h_opt = np.maximum(h_opt, 0)
                
                # Normalize to keep scale (objective is scale invariant, but helps stability)
                # Actually, scale invariance means we can fix sum(h)=1 or max(h)=1.
                # Let's fix sum(h) = 1.
                if np.sum(h_opt) > 0:
                    h_opt = h_opt / np.sum(h_opt) * N # Keep average 1? No, just normalize.
                    # Better: fix L1 norm to 1?
                    # The score is scale invariant. We can fix sum(h)=N to keep heights O(1).
                    h_opt = h_opt * (N / np.sum(h_opt))
                    
                s = compute_score(h_opt)
                if s < best_score:
                    best_score = s
                    best_h = h_opt.copy()
                else:
                    # Revert if worse
                    h_opt = best_h.copy()
                    break # Or continue with smaller lr? Let's break to save time.
                    
        return list(best_h)

# Pre-compute dx
dx = 0.5 / N

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(compute_score(res))
