import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    deadline = time.time() + 145  # Safety margin for 150s limit
    
    # Strategy: 
    # The objective is V = 2*N*max(conv(h,h)) / sum(h)^2.
    # By scale invariance, we can fix sum(h) = 1.
    # Then V = 2*N * max(conv(h,h)).
    # We want to minimize the peak of the autocorrelation.
    # The autocorrelation is the inverse FFT of |H|^2.
    # To minimize the peak of the inverse FFT, we want |H|^2 to be as flat as possible.
    # However, h must be real and positive (to keep sum large and avoid cancellation).
    # We parameterize h using a low-dimensional set of Fourier coefficients 
    # with a specific envelope that ensures positivity and smoothness.
    
    # Let's use a parameterization h_k = exp(a_k) where a_k is a smooth function.
    # Or better: h_k = softplus(x_k) where x_k is generated from a few basis functions.
    
    # Actually, a very effective approach for "flat spectrum" is to use 
    # a window function multiplied by a random phase or specific structure.
    
    # Let's try optimizing a vector of coefficients c for a basis of 
    # low-frequency cosine terms, but with a non-linear mapping to ensure h > 0.
    
    # Basis: Cosine waves on [0, N-1]
    # h[i] = 1 + sum_{k=1}^M c_k * cos(2*pi*k*i/N)
    # To ensure h > 0, we can use a larger offset or a non-linear map.
    # Let's use h[i] = exp(sum c_k * phi_k[i])
    # This guarantees h > 0.
    # The sum(h) will vary, but we normalize h by its sum to fix sum=1.
    # So effectively we are optimizing the shape.
    
    M = 10  # Number of frequency components
    
    # Precompute basis functions phi_k[i] = cos(2*pi*k*i/N)
    # We also include a linear term to allow asymmetry? 
    # Cosine basis is symmetric. Autocorrelation of a symmetric function is symmetric.
    # The max is likely at lag 0. 
    # For lag 0, conv(0) = sum(h^2).
    # We want to minimize sum(h^2) / sum(h)^2 * 2N.
    # This is minimized when h is constant (uniform), giving 2N * (N*(1/N)^2) / 1 = 2.
    # Wait, the seed is 2.0. 
    # Can we go lower than 2?
    # Cauchy-Schwarz: (sum h)^2 <= N * sum(h^2).
    # So sum(h^2) / sum(h)^2 >= 1/N.
    # Therefore 2*N * sum(h^2)/sum(h)^2 >= 2.
    # The minimum possible value is 2.
    # The seed achieves 2.0.
    # Is it possible to get < 2?
    # The formula is abs(2 * N * max(conv) / sum(h)^2).
    # The max is taken over ALL lags.
    # For a uniform h, conv is a triangle with peak at 0.
    # max(conv) = sum(h^2).
    # So value = 2.
    # To get value < 2, we would need max(conv) < sum(h)^2 / (2N).
    # But max(conv) >= conv(0) = sum(h^2).
    # And sum(h^2) >= sum(h)^2 / N.
    # So max(conv) >= sum(h)^2 / N.
    # Thus value >= 2*N * (sum(h)^2/N) / sum(h)^2 = 2.
    # So 2 is the theoretical lower bound.
    # The seed already achieves this bound (or very close to it).
    # Therefore, it is mathematically impossible to score better than 1.0 (which corresponds to value 2.0) 
    # UNLESS the evaluator's definition of max or sum differs slightly or there's a floating point nuance.
    # However, the prompt says "lower is better" and seed is 2.0.
    # If the bound is strict, we cannot improve.
    # BUT, the inspirations show scores > 1.0 (e.g., 1.2897).
    # This implies my derivation that max(conv) is always conv(0) is WRONG.
    # max(conv) is the maximum over ALL lags.
    # For a non-uniform h, the peak might shift or the shape might change.
    # Wait, conv(0) = sum(h_i^2).
    # Is it possible that for some h, max(conv) < sum(h^2)? No, max is always >= any element, including conv(0).
    # So max(conv) >= sum(h^2).
    # So value >= 2N * sum(h^2) / sum(h)^2 >= 2.
    # So how did they get 1.28?
    # Maybe I misread the formula.
    # "value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)"
    # If h has negative values, sum(h) can be small, making the denominator small.
    # But if sum(h) is small, the ratio explodes.
    # If h is all positive, sum(h) is large.
    # Let's re-read carefully.
    # "Choose real, possibly negative step heights".
    # If we allow negative values, we can make sum(h) large while keeping sum(h^2) small?
    # No, (sum h)^2 <= N sum(h^2) holds for ANY real h.
    # So sum(h^2)/sum(h)^2 >= 1/N is always true.
    # So 2N * sum(h^2)/sum(h)^2 >= 2 is always true.
    # Since max(conv) >= conv(0) = sum(h^2), the value is ALWAYS >= 2.
    # So the score seed/value <= 1.0.
    # How did inspiration 1 get 1.2897?
    # Score = seed / value.
    # If score > 1, then value < seed.
    # Seed value is 2.0.
    # So value < 2.0.
    # This contradicts the math.
    # Unless... np.convolve(h, h) is NOT autocorrelation in the standard sense?
    # np.convolve(h, h) with mode 'full' gives linear convolution.
    # For real h, conv(h, h) is the autocorrelation only if one is reversed.
    # Autocorrelation is usually <h, shift(h)> = sum h_i h_{i+k}.
    # Convolution is sum h_i h_{k-i}.
    # If h is symmetric, they are the same.
    # If h is NOT symmetric, conv(h, h) is NOT the autocorrelation.
    # The problem says "autoconvolution".
    # It uses np.convolve(h, h).
    # So it is linear convolution of h with itself.
    # Let c_k = sum_{i} h_i h_{k-i}.
    # c_0 = sum h_i^2.
    # c_N-1 = h_0 h_{N-1}.
    # The max is over all c_k.
    # Is it possible that max(c_k) < c_0?
    # Yes, if h is not symmetric, the peak of the convolution might not be at the center (lag 0 of autocorrelation corresponds to index N-1 in full conv? No).
    # Let's check indices.
    # np.convolve([a, b], [a, b]) = [a^2, ab+ba, b^2] = [a^2, 2ab, b^2].
    # For N=2, indices 0, 1, 2.
    # c_0 = a^2. c_1 = 2ab. c_2 = b^2.
    # If a=1, b=-1. h=[1, -1]. sum=0. Invalid.
    # If a=1, b=0.5. sum=1.5. sum_sq=1.25.
    # c = [1, 1, 0.25]. max=1.
    # Value = 2*2*1 / 1.5^2 = 4/2.25 = 1.77 < 2.
    # AHA!
    # The Cauchy-Schwarz bound applies to sum(h^2)/sum(h)^2.
    # But max(conv) is NOT necessarily sum(h^2).
    # sum(h^2) is just ONE element of the convolution (the first one, index 0).
    # If the function is not symmetric, the peak of the convolution can be LOWER than the sum of squares!
    # Wait, for positive h, is conv(0) always the max?
    # c_k = sum h_i h_{k-i}.
    # If h > 0, then c_k is a sum of positive terms.
    # c_0 = sum h_i^2.
    # c_k for k>0 involves cross terms.
    # For a=1, b=0.5, c_0=1, c_1=1, c_2=0.25. Max is 1.
    # Value 1.77.
    # So we CAN go below 2!
    # The strategy is to make the convolution peak as flat as possible, or shifted away from the high-energy lags?
    # Actually, we want to minimize max(c_k).
    # For positive h, c_0 = sum h^2.
    # If we make h very "spiky", sum h^2 is large.
    # If we make h uniform, sum h^2 is minimal for fixed sum, BUT the convolution is a triangle with peak at center.
    # For uniform h, c_k is max at k=N-1 (center of full conv? No, full conv of N and N has length 2N-1. Center is N-1).
    # For uniform, c_{N-1} = N * (1/N)^2 * N? No.
    # h_i = 1/N.
    # c_k = sum (1/N)(1/N) = (N-|k-(N-1)|)/N^2.
    # Max is at k=N-1, value = N/N^2 = 1/N.
    # Value = 2N * (1/N) / (1)^2 = 2.
    # So uniform gives 2.
    # My example a=1, b=0.5 gave 1.77.
    # Let's check N=2, h=[1, 0.5].
    # sum = 1.5. sum^2 = 2.25.
    # conv = [1, 1, 0.25]. max = 1.
    # Value = 2*2*1 / 2.25 = 1.77.
    # So non-uniform positive h can beat uniform!
    # Why? Because the "peak" of the convolution for uniform is spread out, but for [1, 0.5], the peak is 1.
    # Wait, for uniform N=2, h=[0.5, 0.5]. sum=1.
    # conv = [0.25, 0.5, 0.25]. max = 0.5.
    # Value = 2*2*0.5 / 1 = 2.
    # For [1, 0.5], we normalized? No, I didn't normalize in the manual check.
    # Let's normalize [1, 0.5] to sum 1: h = [2/3, 1/3].
    # sum = 1.
    # conv = [4/9, 4/9, 1/9]. max = 4/9.
    # Value = 2*2*(4/9)/1 = 16/9 = 1.77.
    # So [2/3, 1/3] is better than [1/2, 1/2].
    # The key is to make the convolution as "flat" as possible or minimize its max.
    # This is related to "Low Autocorrelation Sequences" or "Golay Complementary Pairs" but for real continuous values.
    # We want to minimize max_k c_k subject to sum h = 1.
    # This is a convex optimization problem if we fix the sign?
    # c_k are linear in h? No, quadratic.
    # But we can use gradient descent.
    
    # Strategy:
    # 1. Start with a random positive h.
    # 2. Normalize to sum 1.
    # 3. Compute conv.
    # 4. Find the index of the max.
    # 5. Use gradient descent to reduce that specific c_k (and neighbors) while keeping sum h = 1.
    # Since the objective is non-smooth (max), we can use a surrogate: minimize sum(c_k^2) or max(c_k) with smoothing.
    
    # Better: Minimize the peak using a "soft-max" or just gradient descent on the peak element.
    
    # Let's use scipy.optimize.minimize with a custom objective.
    # Variables: h (length N).
    # Constraint: sum(h) = 1.
    # To ensure h > 0, we can parameterize h = exp(x) / sum(exp(x)).
    # This is the softmax parameterization.
    
    # Objective: 2*N * max(conv(h))
    # (Since sum h = 1).
    
    # We need the gradient of max(conv(h)) w.r.t h.
    # Let m = argmax(conv(h)).
    # grad_h (conv_m) = conv(grad_m, h) roughly?
    # conv_m = sum_i h_i h_{m-i}.
    # d(conv_m)/d(h_j) = h_{m-j} + h_{m-j} (if m-j != j) or 2h_j.
    # Essentially, the gradient of the k-th convolution coefficient w.r.t h is a shifted version of h.
    
    # We can use automatic differentiation or manual gradient.
    # Given time constraints, let's try a simple iterative projection method or L-BFGS-B on the log-domain.
    
    # Let's try L-BFGS-B on x where h = softmax(x).
    
    def softmax(x):
        x_max = np.max(x)
        e = np.exp(x - x_max)
        return e / np.sum(e)
    
    def objective_and_grad(x):
        h = softmax(x)
        # Compute conv
        # Use FFT for speed
        n_fft = 1 << (2 * N - 1).bit_length()
        H = np.fft.rfft(h, n_fft)
        Z = np.fft.irfft(H * np.conj(H), n_fft)[:2*N-1]
        
        # Z is the circular autocorrelation? No, linear conv if padded.
        # Z[k] is the k-th coefficient of conv(h, h).
        
        max_val = np.max(Z)
        max_idx = np.argmax(Z)
        
        # Gradient of Z[max_idx] w.r.t h
        # Z[k] = sum_{i} h[i] h[k-i]
        # dZ[k]/dh[j] = h[k-j] + h[k-j] (symmetry)
        # Let's compute the gradient vector g_h.
        # g_h[j] = h[k-j] + h[k-j] ??
        # Let's derive: Z[k] = sum_{i=0}^{N-1} h[i] h[k-i] (with h out of bounds = 0).
        # dZ[k]/dh[j] = term where i=j: h[k-j] + term where k-i=j => i=k-j: h[k-j].
        # So dZ[k]/dh[j] = 2 * h[k-j] (if indices valid).
        
        # We need grad_x.
        # h = softmax(x).
        # dh_j/dx_i = h_j (delta_{ji} - h_i).
        
        # This is getting complex to code manually robustly.
        # Let's use a finite difference gradient or just minimize the objective with L-BFGS-B 
        # which approximates gradients.
        
        # To help L-BFGS-B, we can smooth the max.
        # Softmax of Z with high temperature?
        # Or just minimize Z[max_idx] + sum of neighbors.
        
        # Let's just minimize the max value using a surrogate:
        # f(x) = np.max(Z) + 0.1 * np.sum(Z**2)  # Encourage flatness
        
        return np.max(Z) + 0.1 * np.sum(Z**2)
        
    # Initial guess: random positive
    rng = np.random.default_rng(42)
    x0 = rng.normal(0, 1, N)
    
    # We want to optimize x.
    # Constraint: None, because softmax ensures sum=1 and h>0.
    
    # We will run L-BFGS-B.
    
    best_x = x0
    best_val = float('inf')
    
    # Multiple restarts
    for i in range(5):
        if time.time() > deadline - 10:
            break
            
        if i > 0:
            x0 = rng.normal(0, 1, N)
            
        try:
            res = minimize(
                objective_and_grad, 
                x0, 
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            h_res = softmax(res.x)
            # Calculate true score
            n_fft = 1 << (2 * N - 1).bit_length()
            H = np.fft.rfft(h_res, n_fft)
            Z = np.fft.irfft(H * np.conj(H), n_fft)[:2*N-1]
            true_val = 2 * N * np.max(Z) / np.sum(h_res)**2
            
            if true_val < best_val:
                best_val = true_val
                best_x = res.x
        except Exception:
            continue
            
    # Final refinement on best_x
    if time.time() < deadline - 5:
        try:
            res_final = minimize(
                objective_and_grad, 
                best_x, 
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            h_final = softmax(res_final.x)
            n_fft = 1 << (2 * N - 1).bit_length()
            H = np.fft.rfft(h_final, n_fft)
            Z = np.fft.irfft(H * np.conj(H), n_fft)[:2*N-1]
            final_val = 2 * N * np.max(Z) / np.sum(h_final)**2
            
            if final_val < best_val:
                best_val = final_val
                best_x = res_final.x
        except Exception:
            pass
            
    h_out = softmax(best_x)
    
    # Sanity check
    if not np.all(np.isfinite(h_out)) or np.sum(h_out) == 0:
        return np.ones(N)
        
    return h_out
