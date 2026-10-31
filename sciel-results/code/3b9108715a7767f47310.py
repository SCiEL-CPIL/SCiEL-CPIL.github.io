"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _local_sign_flip(h, max_passes=5, deadline=None):
    """Local optimization: try flipping each element's sign, keep if improves."""
    h = h.copy()
    best_val = _score(h)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]  # revert
        if not improved:
            break
    return h, best_val

def _gradient_descent(h, deadline):
    """
    Gradient descent on the smooth objective.
    The objective is max|ac(k)| / (sum h)^2.
    We approximate the gradient of the numerator (max abs ac) and denominator.
    Since max is non-smooth, we use a soft-max approximation or simply
    treat the argmax index as fixed for a few steps (subgradient).
    
    Actually, a simpler approach for signed step functions with continuous relaxation:
    Let's relax h to be real-valued, then threshold at the end.
    Objective f(h) = max_k |ac(k)| / (sum h)^2.
    Let M = max_k |ac(k)|, let k* be the argmax.
    Let I = sum h.
    f = M / I^2.
    df/dh_i = (dM/dh_i * I^2 - M * 2 * I * dI/dh_i) / I^4
             = (dM/dh_i * I - 2 * M * h_i) / I^3
    
    dI/dh_i = 1.
    dM/dh_i depends on which k* is the max and the sign of ac(k*).
    ac(k) = sum_j h_j h_{j+k} (with appropriate boundaries).
    dac(k)/dh_i = h_{i-k} + h_{i+k} (roughly, depending on index validity).
    If ac(k*) > 0, dM/dh_i = dac(k*)/dh_i.
    If ac(k*) < 0, dM/dh_i = -dac(k*)/dh_i.
    
    We will perform a few steps of this.
    """
    h = h.copy()
    best_val = _score(h)
    
    # Precompute FFT size
    size = 2 * N - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    # We need to compute ac and its gradient efficiently.
    # ac = ifft(fft(h)^2)
    # grad_ac_k = ifft(fft(h) * fft(grad_h_k)) ? No, that's for product rule.
    # ac(k) = sum_j h_j h_{j+k}.
    # Let's just use the subgradient approach with FFT for the value,
    # and explicit convolution for the gradient of the specific lag k*.
    
    for step in range(200):
        if time.time() > deadline:
            break
        
        # Compute current ac
        H = np.fft.rfft(h, fsize)
        ac = np.fft.irfft(H * H, fsize)[:size]
        
        # Find max absolute value
        abs_ac = np.abs(ac)
        k_star = np.argmax(abs_ac)
        M = abs_ac[k_star]
        sign_ac = np.sign(ac[k_star])
        
        I = np.sum(h)
        if abs(I) < 1e-9:
            break
            
        # Gradient of M w.r.t h
        # M = |ac(k_star)| = sign_ac * ac(k_star)
        # ac(k_star) = sum_{j=0}^{N-1-k_star} h_j h_{j+k_star}
        # d ac(k_star) / d h_i:
        # Term 1: j=i -> h_{i+k_star} (if i+k_star < N)
        # Term 2: j+k_star=i -> h_{i-k_star} (if i-k_star >= 0)
        
        grad_M = np.zeros(N)
        # Vectorized gradient computation for the specific lag
        # Indices where j=i is valid: i in [0, N-1-k_star]
        # Indices where j+k_star=i is valid: i in [k_star, N-1]
        
        if k_star == 0:
            # ac(0) = sum h_i^2. d/dh_i = 2 h_i.
            # But wait, ac(0) is always positive and max.
            # If k_star=0, M = sum h^2.
            # grad_M = sign_ac * 2 * h = 2 * h (since sign is +1).
            grad_M = 2.0 * h
        else:
            # h_{i+k_star}
            idx1 = np.arange(N - k_star)
            grad_M[idx1] += sign_ac * h[idx1 + k_star]
            
            # h_{i-k_star}
            idx2 = np.arange(k_star, N)
            grad_M[idx2] += sign_ac * h[idx2 - k_star]
            
        # Gradient of f = M / I^2
        # df/dh_i = (grad_M_i * I - 2 * M * h_i) / I^3
        grad_f = (grad_M * I - 2.0 * M * h) / (I ** 3)
        
        # Gradient descent step
        # Adaptive step size
        norm_grad = np.linalg.norm(grad_f)
        if norm_grad < 1e-9:
            break
        
        # Scale step size to move a reasonable amount
        # We want to change h by roughly O(1) in some directions
        lr = 0.1 / norm_grad * np.max(np.abs(h))
        
        h_new = h - lr * grad_f
        
        # Threshold to {-1, 1} or keep continuous?
        # The task asks for "signed step heights". Usually implies discrete or continuous?
        # "sequence of signed step heights"
        # The seed program used discrete {-1, 1}.
        # The inspiration 1 used discrete.
        # The finding says "Gradient descent on smooth max-conv objective may find non-constant sequences... Measured: 1 -> 1.048".
        # This suggests continuous relaxation is allowed and beneficial, OR that the discrete search found something better.
        # Given "signed step heights", continuous values are likely allowed.
        # However, the denominator is (sum h)^2. If we scale h, the score is invariant.
        # So we can fix the norm or sum.
        # Let's keep h continuous but normalize it to have sum = N (or some fixed value) to prevent drift?
        # No, the score is scale invariant. So we can just normalize h to unit L2 norm or sum=1.
        # Let's normalize h such that sum(h) = N to keep values in a similar range.
        s_new = np.sum(h_new)
        if abs(s_new) > 1e-9:
            h_new = h_new * (N / s_new)
        
        val_new = _score(h_new)
        
        if val_new < best_val:
            best_val = val_new
            h = h_new
        else:
            # Revert and maybe reduce learning rate?
            # For simplicity, just revert.
            pass
            
    return h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    best_h = np.ones(N)
    best_val = _score(best_h)
    
    np.random.seed(42)
    
    # Strategy:
    # 1. Start with ones(N).
    # 2. Perform gradient descent on the continuous relaxation.
    # 3. Perform local sign flips on the result.
    # 4. Try multiple random initializations + gradient descent + local sign flips.
    
    # Phase 1: Gradient descent from ones(N)
    h_grad, val_grad = _gradient_descent(best_h, deadline)
    if val_grad < best_val:
        best_val = val_grad
        best_h = h_grad
    
    # Phase 2: Local sign flips on the gradient result
    h_opt, val_opt = _local_sign_flip(best_h, max_passes=5, deadline=deadline)
    if val_opt < best_val:
        best_val = val_opt
        best_h = h_opt
    
    # Phase 3: Random restarts with gradient descent
    n_restarts = 20
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        if r % 2 == 0:
            h = np.random.uniform(-1, 1, size=N)
        else:
            h = np.random.choice([-1.0, 1.0], size=N)
            
        # Ensure sum is not too small
        s = np.sum(h)
        if abs(s) < N * 0.1:
            if s < 0:
                h = -h
            # Add bias
            h = h + 0.5
            h = np.sign(h)
            
        # Gradient descent
        h_grad, val_grad = _gradient_descent(h, deadline)
        
        # Local sign flips
        h_opt, val_opt = _local_sign_flip(h_grad, max_passes=3, deadline=deadline)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
            
    return best_h

# EVOLVE-BLOCK-END
