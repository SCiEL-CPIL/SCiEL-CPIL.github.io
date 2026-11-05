import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Precompute FFT basis
    # We represent h as a linear combination of real-valued Fourier modes.
    # h[n] = sum_{k=0}^{N-1} a_k * exp(2j * pi * k * n / N)
    # To ensure h is real, we can use a real-valued basis:
    # h[n] = c_0 + sum_{k=1}^{N//2} (c_k * cos(2*pi*k*n/N) + s_k * sin(2*pi*k*n/N))
    # However, for optimization, it is often easier to work with complex coefficients 
    # and enforce hermitian symmetry, or just use a real basis matrix.
    # Let's use a real basis matrix M of shape (N, N).
    # M[n, k] = cos(2*pi*k*n/N) for k=1..N//2, sin for k=N//2+1..N-1, and 1 for k=0.
    
    # Actually, a simpler and very effective approach for "flat" autoconvolution 
    # is to target h such that its power spectrum is flat (white noise) 
    # but with a specific DC component to maximize the ratio.
    # 
    # Let's stick to optimizing the coefficients of a Fourier series directly.
    # Let h = M @ coeffs, where M is the DCT-II or similar orthogonal basis.
    # Using DCT-II is nice because it provides a real, orthogonal basis.
    
    # Construct DCT-II matrix
    # DCT-II definition: X_k = sum_{n=0}^{N-1} x_n * cos(pi * (2n+1) * k / (2N))
    # We want to generate x from coefficients.
    # x_n = sum_{k=0}^{N-1} c_k * cos(pi * (2n+1) * k / (2N))
    
    n_idx = np.arange(N)
    k_idx = np.arange(N)
    # Create the basis matrix
    # B[n, k] = cos(pi * (2n+1) * k / (2N))
    B = np.cos(np.pi * (2 * n_idx[:, None] + 1) * k_idx[None, :] / (2 * N))
    
    # Normalize rows/columns? 
    # For optimization, scaling matters. 
    # If we use orthogonal basis, the mapping is invertible.
    # Let's just use the raw basis and let the optimizer handle scaling.
    # However, to avoid huge gradients, we might want to scale.
    # Let's stick to the standard DCT-II forward transform matrix (unnormalized).
    
    def coeffs_to_h(coeffs):
        return B @ coeffs

    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        c = np.convolve(h, h)
        return abs(2.0 * N * np.max(c) / (s * s))

    def spectral_objective_and_grad(coeffs):
        """
        Objective in coefficient space.
        h = B @ coeffs
        J = 2*N*max(c)/S^2
        
        We use a soft-max for max(c).
        Gradient chain rule:
        dJ/dcoeffs = dJ/dh @ (B.T)
        
        dJ/dh is computed as in the previous strategy, but we can potentially
        use a higher temperature or different smoothing if the spectral basis
        helps navigate the landscape.
        
        Actually, a key insight: 
        If we restrict coeffs to only low frequencies, we get smooth h.
        If we allow all, we get arbitrary h.
        
        Let's try a different structural change:
        Instead of soft-max on c, we can minimize the variance of c 
        subject to a fixed mean? 
        No, J depends on max(c).
        
        Let's try a different optimization method: 
        Use the fact that min max(c) is related to the minimum L-infinity norm 
        of the correlation.
        
        Alternative Strategy: 
        Randomized Restart with Spectral Annealing.
        1. Start with random high-frequency coefficients (noise).
        2. Anneal the temperature of the soft-max from low to high? 
           No, usually we want to start with smooth (low temp soft-max approx mean) 
           and refine to sharp (high temp soft-max approx max).
           
        Let's implement the gradient step carefully.
        """
        h = coeffs_to_h(coeffs)
        s = np.sum(h)
        if abs(s) < 1e-12:
            return 1e10, np.zeros_like(coeffs)
        
        c = np.convolve(h, h)
        
        # Soft max parameters
        # We want to approximate max(c).
        # Using a fixed temperature can be tricky.
        # Let's use a temperature that adapts or is set high enough to be smooth
        # but low enough to capture the peak.
        # For N=400, c has 799 points.
        # Let's try t=50.0.
        t = 50.0
        
        c_max = np.max(c)
        u = t * (c - c_max)
        exp_u = np.exp(u)
        Z = np.sum(exp_u)
        p = exp_u / Z
        
        C_soft = c_max + np.log(Z) / t
        
        # Gradient of C_soft w.r.t h
        grad_C = np.zeros(N)
        
        # Efficient gradient calculation for soft-max of convolution
        # grad_C[m] = sum_j p_j * (h_{m+L_j} + h_{m-L_j})
        # This is equivalent to:
        # grad_C = convolve(p, h) + convolve(p[::-1], h) ?
        # Let's check indices.
        # c_j = sum_i h_i h_{i-L_j}.
        # dc_j/dh_m = h_{m-L_j} + h_{m+L_j}.
        # Sum_j p_j h_{m-L_j} is a convolution of p and h.
        # Specifically, (p * h)[m] = sum_j p_j h_{m-j}.
        # If we define p_k corresponding to lag L_k, this matches.
        
        # np.convolve(p, h, mode='full')?
        # p has length 2N-1. h has length N.
        # The result has length 3N-2.
        # We need the central part corresponding to m in [0, N-1].
        
        # Let's verify with small N.
        # L_j ranges from -(N-1) to N-1.
        # j = L + N - 1.
        # p is indexed by j.
        
        # conv_p_h = np.convolve(p, h, mode='full')
        # conv_p_h[k] = sum_j p_j h_{k-j}.
        # We want sum_j p_j h_{m-L_j}.
        # Let k = m. sum_j p_j h_{m-j}.
        # Here j is the index in p. But p_j corresponds to lag L_j = j - (N-1).
        # So h_{m - (j - (N-1))} = h_{m - j + N - 1}.
        # This shifts the index.
        
        # It is safer to use the slicing method or explicit loops if N is small, 
        # but N=400 is small enough for vectorized slicing.
        
        # Let's use the slicing method from the current program but ensure it's correct.
        for L in range(-(N-1), N):
            j = L + N - 1
            w = p[j]
            if w < 1e-10:
                continue
            
            if L >= 0:
                if N > L:
                    grad_C[:N-L] += w * h[L:]
                if L < N:
                    grad_C[L:] += w * h[:N-L]
            else:
                Lp = -L
                if N > Lp:
                    grad_C[:N-Lp] += w * h[Lp:]
                if Lp < N:
                    grad_C[Lp:] += w * h[:N-Lp]
        
        # Objective J = 2*N*C_soft / S^2
        # dJ/dh = 2*N * ( (grad_C) / S - 2*C_soft / S^2 )
        grad_J_h = (2.0 * N / s) * grad_C - (4.0 * N * C_soft / (s * s))
        
        # Chain rule: dJ/dcoeffs = B.T @ grad_J_h
        grad_J_coeffs = B.T @ grad_J_h
        
        val = 2.0 * N * C_soft / (s * s)
        
        return val, grad_J_coeffs

    # Initializations in coefficient space
    # 1. DC only (constant h)
    # 2. Random low freq
    # 3. Random high freq
    
    starts = []
    
    # DC only: coeffs[0] = 1, rest 0
    c0 = np.zeros(N)
    c0[0] = 1.0
    starts.append(c0)
    
    # Low frequency noise
    rng = np.random.RandomState(42)
    c_low = rng.randn(N) * 0.1
    c_low[0] = 1.0
    starts.append(c_low)
    
    # High frequency noise
    c_high = rng.randn(N) * 0.1
    c_high[0] = 1.0
    starts.append(c_high)
    
    # "Flat spectrum" target: 
    # We want h to be such that its autocorrelation is flat.
    # This implies h is a random phase sequence? 
    # In coefficient space, if we set all coeffs to have similar magnitude, 
    # h will be "noise-like".
    c_flat = rng.randn(N)
    c_flat[0] = np.sqrt(np.sum(c_flat[1:]**2)) # Balance DC
    starts.append(c_flat)
    
    h_best = np.ones(N)
    val_best = compute_value(h_best)
    
    for c0 in starts:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                spectral_objective_and_grad,
                c0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'maxls': 50, 'disp': False, 'ftol': 1e-8}
            )
            c_res = res.x
            h_res = coeffs_to_h(c_res)
            val_res = compute_value(h_res)
            if val_res < val_best:
                val_best = val_res
                h_best = h_res.copy()
        except Exception:
            pass
            
    return h_best
