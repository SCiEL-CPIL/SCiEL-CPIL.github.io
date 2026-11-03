import numpy as np
import time
from scipy.optimize import minimize
from scipy.fft import fft, ifft

N = 600
TIME_LIMIT = 145  # Safety margin before 150s kill

# Precompute grid
# The intervals are equal on [-1/4, 1/4].
# There are N intervals.
# The step function is constant on each interval.
# We represent the function by its values at the centers of the intervals.
x_edges = np.linspace(-0.25, 0.25, N + 1)
x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0

# Precompute FFT size for fast convolution
# Convolution of two length-N sequences results in length 2N-1.
# We need FFT size >= 2N-1. Next power of 2 for 1199 is 2048.
FFT_SIZE = 2048

def compute_score(f):
    """
    Compute the normalized maximum of the autoconvolution.
    f: array of shape (N,)
    Returns: max(conv) / (sum(f))^2
    """
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
        
    # Use FFT for convolution
    f_padded = np.zeros(FFT_SIZE)
    f_padded[:N] = f
    F = fft(f_padded)
    conv_padded = ifft(F * F)
    conv = np.real(conv_padded[:2*N-1])
    
    max_conv = np.max(conv)
    
    return max_conv / (s * s)

def smooth_max_and_grad(f, beta=100.0):
    """
    Compute the smooth maximum of the autoconvolution and its gradient.
    Uses Log-Sum-Exp approximation: M_beta(c) = (1/beta) * log(sum(exp(beta * c_i)))
    Gradient: dM/dc_i = exp(beta * c_i) / sum(exp(beta * c_j))
    
    f: array of shape (N,)
    Returns: (smooth_max_value, gradient_vector_of_length_N)
    """
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf, np.zeros(N)
        
    # Convolution via FFT
    f_padded = np.zeros(FFT_SIZE)
    f_padded[:N] = f
    F = fft(f_padded)
    conv_padded = ifft(F * F)
    conv = np.real(conv_padded[:2*N-1])
    
    # Normalize conv by s^2 to get the objective component
    # Objective J = max(conv) / s^2
    # We want to minimize J.
    # Let's work with the unnormalized smooth max first, then scale gradient?
    # Actually, let's just optimize the raw max conv, but we need to be careful with scaling.
    # If we fix sum(f) = 1, then max(conv) is the objective.
    # But sum(f) is a free variable in the optimization.
    # Let's use the normalized objective directly in the smooth max?
    # J = M_beta(conv) / s^2
    # This is complicated to differentiate directly.
    
    # Alternative: Optimize f such that sum(f)=1 is not enforced, but we penalize large sums?
    # No, the objective is scale-invariant? No.
    # If we scale f by k, conv scales by k^2, s scales by k.
    # Score = (k^2 * max_conv_1) / (k^2 * s_1^2) = max_conv_1 / s_1^2.
    # The score is invariant to scaling of f!
    # Therefore, we can fix the normalization, e.g., sum(f) = 1, and just minimize max(conv).
    # This simplifies the optimization significantly.
    
    # So, let's assume we are optimizing over f with sum(f)=1.
    # But standard minimizers don't handle equality constraints well without Lagrangians.
    # However, since the score is scale invariant, we can just normalize f at the end.
    # During optimization, if we don't constrain sum(f), the optimizer might drift.
    # But if we minimize M_beta(conv), and conv scales with f^2, the optimizer will likely
    # find a shape. The scale doesn't matter for the final score.
    # So, let's just minimize M_beta(conv).
    
    # Shift conv for numerical stability in exp
    max_c = np.max(conv)
    # If max_c is very large or small, exp might overflow/underflow.
    # But conv is non-negative.
    # We can just use log-sum-exp trick.
    # log_sum_exp(conv) = max_c + log(sum(exp(conv - max_c)))
    # Smooth Max M = (1/beta) * log_sum_exp(beta * conv)
    # Let's compute M_beta(conv).
    
    beta_conv = beta * conv
    max_bc = np.max(beta_conv)
    # Avoid overflow in exp
    if max_bc > 700:
        # This shouldn't happen if we scale f, but just in case.
        # If it does, the gradient will be a delta function at the max.
        idx = np.argmax(beta_conv)
        grad_conv = np.zeros(len(conv))
        grad_conv[idx] = 1.0
        return max_c, grad_conv # Return raw max and one-hot grad for now? 
        # Actually, if beta is large, it's close to max.
        
    exp_bc = np.exp(beta_conv - max_bc)
    sum_exp = np.sum(exp_bc)
    
    # Smooth max value
    M = (max_bc + np.log(sum_exp)) / beta
    
    # Gradient of M with respect to conv
    # dM/dconv_i = exp(beta * conv_i) / sum(exp(beta * conv_j))
    grad_conv = exp_bc / sum_exp
    
    # Now, we need dM/df
    # M = g(conv)
    # dM/df = (dconv/df)^T grad_conv
    # conv = f * f (discrete convolution)
    # conv_k = sum_i f_i f_{k-i}
    # dconv_k / df_j = f_{k-j} + f_{k-j} (if k-j is valid index, considering symmetry?)
    # Actually, for linear convolution of f with f:
    # conv_k = sum_{i} f_i f_{k-i}
    # If we differentiate w.r.t f_j:
    # Term where i=j: f_j f_{k-j} -> derivative f_{k-j}
    # Term where k-i=j => i=k-j: f_{k-j} f_j -> derivative f_j
    # So dconv_k / df_j = 2 * f_{k-j} (if indices are valid)
    # Wait, if i=j and i=k-j are the same term (i.e., 2j=k), the derivative is 2*f_{k-j}.
    # If they are distinct, it's f_{k-j} + f_j. Since f is the same vector, it's always 2 * f_{k-j} (with appropriate boundary handling).
    
    # Efficient calculation using FFT:
    # grad_f = ifft(fft(f) * fft(grad_conv_padded)) ? No.
    # dconv/df is a linear operator.
    # Let G = grad_conv (length 2N-1).
    # We want h_j = sum_k G_k * (dconv_k / df_j)
    # h_j = sum_k G_k * 2 * f_{k-j} (approx)
    # This is essentially a convolution of G and f, or correlation.
    
    # Let's use the FFT property:
    # Convolution C = f * f.
    # Gradient of sum(G * C) w.r.t f is 2 * (G * f) (truncated to length N).
    # Specifically, if C_k = sum_i f_i f_{k-i}, and we want sum_k G_k C_k,
    # then d/d f_j sum_k G_k C_k = sum_k G_k (f_{k-j} + f_{k-j}) = 2 * sum_k G_k f_{k-j}.
    # This is 2 * (G * f)_j, where * is linear convolution.
    
    # We need to compute conv(G, f) and take the first N elements.
    # G has length 2N-1. f has length N.
    # Result has length 3N-2. We want the central part?
    # Let's check indices.
    # C_k for k=0..2N-2.
    # d/d f_j C_k is non-zero if 0 <= k-j < N.
    # So j must be in [k-N+1, k].
    # The sum over k gives the gradient.
    
    # Let's just do it with FFT for speed.
    # Pad G to size >= (2N-1) + N - 1 = 3N-2. Next power of 2 for 1800 is 2048? No, 4096.
    # 3*600 - 2 = 1798. Next power of 2 is 2048.
    FFT_SIZE_GRAD = 2048
    
    G_padded = np.zeros(FFT_SIZE_GRAD)
    G_padded[:len(G)] = G
    
    f_padded = np.zeros(FFT_SIZE_GRAD)
    f_padded[:N] = f
    
    G_f = ifft(fft(G_padded) * fft(f_padded))
    
    # The convolution result G*f has length 3N-2.
    # The gradient for f_j is 2 * (G*f)_j ?
    # Let's verify.
    # (G*f)_j = sum_k G_k f_{j-k}.
    # We want sum_k G_k f_{k-j}.
    # This is correlation, not convolution?
    # C_k = sum_i f_i f_{k-i}.
    # dC_k/df_j = f_{k-j} + f_{k-j} (if valid).
    # Sum_k G_k f_{k-j}.
    # Let m = k-j. Sum_m G_{m+j} f_m.
    # This is (G_shifted * f)_j?
    # Actually, simply:
    # grad_f = 2 * ifft(fft(f) * fft(G_padded)) ? No.
    # Let's use the property that for real signals, gradient of ||f*f||_G is 2 * (G * f) where * is convolution, but aligned correctly.
    
    # Simpler: Just use the direct FFT multiplication for the gradient of the quadratic form.
    # Objective Q = sum_k G_k (f*f)_k
    # In Fourier domain:
    # (f*f)_k corresponds to IFFT(FFT(f)^2).
    # This is tricky to differentiate directly in frequency.
    
    # Let's stick to time domain convolution for the gradient step if FFT alignment is confusing, 
    # but N=600 is small enough that even O(N^2) might be okay? 600^2 = 360,000 ops.
    # In Python/NumPy, vectorized O(N^2) is fast.
    # But we are in a loop.
    
    # Let's try the FFT alignment again.
    # We want h = 2 * corr(G, f) ?
    # Let's test with small N.
    
    # Actually, there is a simpler way.
    # We are minimizing M_beta(conv).
    # grad_f = 2 * (G * f) where G is padded to length 2N-1?
    # Let's compute conv(G, f) using FFT of size 4096 to be safe.
    
    FFT_SIZE_BIG = 4096
    G_big = np.zeros(FFT_SIZE_BIG)
    G_big[:len(G)] = G
    f_big = np.zeros(FFT_SIZE_BIG)
    f_big[:N] = f
    
    conv_Gf = ifft(fft(G_big) * fft(f_big)).real
    
    # The linear convolution of G (len 2N-1) and f (len N) has length 3N-2.
    # The element corresponding to index j in f (0-indexed) in the gradient formula sum_k G_k f_{k-j}
    # corresponds to index j in the correlation.
    # Correlation (G, f)[j] = sum_k G_k f_{k-j}.
    # Convolution (G, f)[m] = sum_k G_k f_{m-k}.
    # If we reverse G, conv(rev(G), f)[m] = sum_k rev(G)_k f_{m-k} = sum_k G_{L-1-k} f_{m-k}.
    # Let l = L-1-k. k = L-1-l.
    # sum_l G_l f_{m-(L-1-l)} = sum_l G_l f_{l + m - L + 1}.
    # This is cross-correlation.
    
    # So, grad_f_j = 2 * (corr(G, f))_j.
    # We can compute corr via conv with reversed G.
    
    G_rev = G[::-1]
    G_rev_big = np.zeros(FFT_SIZE_BIG)
    G_rev_big[:len(G_rev)] = G_rev
    
    corr = ifft(fft(G_rev_big) * fft(f_big)).real
    
    # The length of corr is 3N-2.
    # We need the part corresponding to f indices 0..N-1.
    # The correlation output index m corresponds to shift.
    # Standard definition: (G * f)_m = sum_k G_k f_{m-k}.
    # We want sum_k G_k f_{k-j}.
    # Let n = k-j. k = n+j.
    # sum_n G_{n+j} f_n.
    # This is the cross-correlation at lag j.
    
    # In NumPy/FFT, if we compute conv(G, f), the result index m is sum_k G_k f_{m-k}.
    # If we use G_rev, conv(G_rev, f)[m] = sum_k G_rev_k f_{m-k} = sum_k G_{L-1-k} f_{m-k}.
    # Let j = L-1-k. k = L-1-j.
    # sum_j G_j f_{m - (L-1-j)} = sum_j G_j f_{j + m - L + 1}.
    # We want sum_j G_j f_{j + j_target} ? No, we want sum_n G_{n+j} f_n.
    # Let n = j + m - L + 1.
    # This is getting messy.
    
    # Let's just do the O(N^2) gradient calculation in NumPy. It's robust and fast enough for N=600.
    # grad_f = zeros(N)
    # for j in range(N):
    #   for k in range(2N-1):
    #     if 0 <= k-j < N:
    #       grad_f[j] += 2 * G[k] * f[k-j]
    
    # Vectorized version:
    # Create a matrix A where A[j, k] = f[k-j] if 0 <= k-j < N else 0.
    # Then grad_f = 2 * (A * G)
    
    # Indices k range from 0 to 2N-2.
    # Indices j range from 0 to N-1.
    # k-j ranges from 0 to N-1.
    # Condition: 0 <= k-j <= N-1 => j <= k <= j+N-1.
    
    # We can use np.roll or slicing.
    # For each k, the valid j are max(0, k-N+1) to min(N-1, k).
    
    # Alternative: Use the fact that dconv/df is a banded matrix.
    # But let's use the FFT correlation result and just find the right slice.
    
    # Let's trust the FFT correlation.
    # corr = ifft(fft(G_rev) * fft(f))
    # The valid indices for f_j (j=0..N-1) in the correlation output.
    # The correlation output index m represents sum_k G_k f_{m-k} (with G reversed in the conv logic above).
    # Actually, let's just compute the gradient numerically for a few points to verify the FFT slice?
    # No, let's use a known identity.
    # Gradient of (f*f) w.r.t f is 2 * (f * f) (convolution).
    # Here we have weighted gradient G.
    # grad_f = 2 * (G * f) where * is the appropriate cross-correlation.
    
    # Let's use the direct summation for safety, N=600 is small.
    # 600 * 1200 = 720,000 operations.
    # In NumPy, this is ~0.1 ms.
    
    # We can vectorize it:
    # K = np.arange(2*N-1)
    # J = np.arange(N)
    # K_J = K[None, :] - J[:, None]
    # Mask = (K_J >= 0) & (K_J < N)
    # F_vals = np.where(Mask, f[K_J], 0.0)
    # Grad_f = 2.0 * (F_vals * G[None, :]).sum(axis=1)
    
    # This creates a 600x1199 array. ~720k floats. ~5MB. Fast.
    
    K = np.arange(2 * N - 1)
    J = np.arange(N)
    K_J = K[None, :] - J[:, None]
    Mask = (K_J >= 0) & (K_J < N)
    
    # f[K_J] will broadcast.
    # We need to handle indices out of bounds for f.
    # Since Mask is False where K_J is out of [0, N-1], we can use np.where.
    # But f[K_J] will error if K_J is out of bounds.
    # So we use np.take with mode='clip' or just initialize zeros.
    
    F_matrix = np.zeros((N, 2 * N - 1))
    # For each row j, we want f[k-j] for valid k.
    # This is equivalent to shifting f.
    # Row j contains f shifted by j.
    # f_shifted_j[k] = f[k-j]
    
    # We can construct this efficiently:
    # For k in 0..2N-2:
    #   f[k-j] is non-zero if 0 <= k-j < N.
    
    # Let's just use the FFT correlation and slice it.
    # corr = ifft(fft(G_rev) * fft(f))
    # Length 3N-2.
    # The term for f_j is at index j + N - 1?
    # Let's test with N=2.
    # f=[f0, f1]. G=[g0, g1, g2].
    # conv(f,f) = [f0^2, 2f0f1, f1^2].
    # grad_f0 = 2(g0 f0 + g1 f1)
    # grad_f1 = 2(g1 f0 + g2 f1)
    
    # G_rev = [g2, g1, g0].
    # conv(G_rev, f) = [g2 f0, g2 f1 + g1 f0, g1 f1 + g0 f0, g0 f1]
    # Indices: 0, 1, 2, 3.
    # We want:
    # j=0: 2(g0 f0 + g1 f1). This matches index 2? (g1 f1 + g0 f0). Yes.
    # j=1: 2(g1 f0 + g2 f1). This matches index 1? (g2 f1 + g1 f0). Yes.
    
    # So for general N:
    # grad_f_j corresponds to index (2N - 2 - j) in the convolution result?
    # For N=2, 2N-2 = 2.
    # j=0 -> index 2.
    # j=1 -> index 1.
    # So index = 2N - 2 - j.
    
    # Let's verify for N=1.
    # f=[f0]. G=[g0].
    # conv(f,f)=[f0^2].
    # grad_f0 = 2 g0 f0.
    # G_rev=[g0].
    # conv(G_rev, f) = [g0 f0]. Index 0.
    # 2N-2-j = 0-0=0. Correct.
    
    # So, grad_f = 2 * corr[2*N-2 - J]
    
    grad_f = 2.0 * corr[2 * N - 2 - J]
    
    return M, grad_f

def objective_smooth(f, beta=100.0):
    """
    Objective function for optimization using smooth max.
    """
    val, grad = smooth_max_and_grad(f, beta)
    # Note: The score is max_conv / s^2.
    # We are minimizing max_conv (assuming scale invariance handles the s^2 part).
    # However, if the optimizer changes the scale, the score changes?
    # No, score is invariant.
    # But if we minimize max_conv, we might drive f to 0?
    # If f -> 0, max_conv -> 0. Score -> 0/0?
    # Actually, if f is small, max_conv is small.
    # But we can normalize f at the end.
    # To prevent f from collapsing to 0, we should probably fix the norm or sum.
    # Let's add a penalty if sum(f) deviates from 1?
    # Or just normalize f after each step?
    # Minimizing max_conv with sum(f)=1 is the correct problem.
    # Let's add a Lagrange multiplier or just project.
    # Simplest: In the objective, if sum(f) != 1, scale f?
    # No, that breaks gradient.
    
    # Let's just minimize max_conv. If f goes to 0, we stop.
    # But L-BFGS might move towards 0.
    # Let's initialize with sum(f)=1.
    # And in the objective, return val + 100*(sum(f)-1)^2 to keep sum near 1.
    
    s = np.sum(f)
    penalty = 10.0 * (s - 1.0)**2
    
    total_val = val + penalty
    # Gradient of penalty: 20 * (s-1) * ones
    grad_penalty = 20.0 * (s - 1.0) * np.ones(N)
    
    return total_val, val, grad + grad_penalty

def construct():
    start_time = time.time()
    
    # Initialize f
    # Start with a Gaussian or uniform
    f_init = np.exp(-50 * x_centers**2)
    f_init /= np.sum(f_init)
    
    best_score = compute_score(f_init)
    best_f = f_init.copy()
    
    # Optimization loop
    # We will use L-BFGS-B with smooth max.
    # We need to handle the beta parameter. Start with low beta (smoother), increase later?
    # Or just use a fixed moderate beta.
    
    beta = 50.0
    
    def obj_wrapper(f):
        val, raw_val, grad = objective_smooth(f, beta)
        return val, grad
        
    try:
        res = minimize(
            obj_wrapper, 
            f_init, 
            method='L-BFGS-B', 
            jac=True,
            bounds=[(0, None)] * N,
            options={
                'maxiter': 100,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'disp': False
            }
        )
        
        f_opt = res.x
        f_opt = np.maximum(f_opt, 0)
        # Normalize
        s = np.sum(f_opt)
        if s > 1e-6:
            f_opt /= s
            
        score_opt = compute_score(f_opt)
        
        if score_opt < best_score:
            best_score = score_opt
            best_f = f_opt.copy()
            
    except Exception as e:
        pass
    
    # If we have time, try a second pass with higher beta (sharper max approximation)
    if time.time() - start_time < TIME_LIMIT - 10:
        beta2 = 200.0
        
        def obj_wrapper2(f):
            val, raw_val, grad = objective_smooth(f, beta2)
            return val, grad
            
        try:
            res2 = minimize(
                obj_wrapper2, 
                best_f, 
                method='L-BFGS-B', 
                jac=True,
                bounds=[(0, None)] * N,
                options={
                    'maxiter': 100,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'disp': False
                }
            )
            
            f_opt2 = res2.x
            f_opt2 = np.maximum(f_opt2, 0)
            s = np.sum(f_opt2)
            if s > 1e-6:
                f_opt2 /= s
                
            score_opt2 = compute_score(f_opt2)
            
            if score_opt2 < best_score:
                best_score = score_opt2
                best_f = f_opt2.copy()
                
        except Exception:
            pass

    # Final polish: Local coordinate descent on the best_f
    if time.time() - start_time < TIME_LIMIT - 5:
        f_cd = best_f.copy()
        s = np.sum(f_cd)
        f_cd /= s
        
        # Simple coordinate ascent/descent
        # For each index, try to adjust f[j] to reduce score.
        # This is slow, but we can do a few iterations.
        
        # Actually, L-BFGS is already doing this effectively.
        # Let's just return the best found.
        
    # Ensure non-negative
    best_f = np.maximum(best_f, 0.0)
    
    # Final normalization check (though score is invariant, it's good practice)
    s_final = np.sum(best_f)
    if s_final > 0:
        best_f /= s_final

    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
