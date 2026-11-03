"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_autoconv(h):
    """Compute linear autoconvolution of h."""
    size = 2 * N - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _score_normalized(h):
    """
    Compute the objective: max|autoconv| / (sum h)^2.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _compute_autoconv(h)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def _optimize_continuous(h_init, deadline):
    """
    Optimize h using gradient descent on the continuous domain.
    Objective: minimize max|autoconv| subject to sum(h)=1.
    """
    h = h_init.copy()
    # Normalize initial h to sum=1
    s = np.sum(h)
    if abs(s) < 1e-9:
        h = np.random.randn(N)
        s = np.sum(h)
    h /= s
    
    # Hyperparameters
    lr = 0.01
    momentum = 0.9
    v = np.zeros(N)
    current_val = _score_normalized(h)
    
    while time.time() < deadline:
        # Compute autoconvolution and find max abs index
        ac = _compute_autoconv(h)
        k_star = np.argmax(np.abs(ac))
        val_ac = ac[k_star]
        
        if abs(val_ac) < 1e-12:
            break
            
        sign_ac = np.sign(val_ac)
        
        # Gradient of |A_{k_star}| w.r.t h
        # A_k = sum_i h_i h_{k-i}
        # d A_k / d h_j = h_{k-j} + h_{j-k} (where indices are valid)
        # We need to construct the vector g such that g_j = sign(A_k) * (h_{k-j} + h_{j-k})
        
        g = np.zeros(N)
        
        # Term 1: h_{k-j}
        # For j in [0, N-1], index k-j must be in [0, N-1]
        # k-j >= 0 => j <= k
        # k-j < N => j > k-N
        idx1 = np.arange(max(0, k_star - N + 1), min(N, k_star + 1))
        if len(idx1) > 0:
            vals1 = h[k_star - idx1]
            g[idx1] += sign_ac * vals1
            
        # Term 2: h_{j-k}
        # For j in [0, N-1], index j-k must be in [0, N-1]
        # j-k >= 0 => j >= k
        # j-k < N => j < k+N
        idx2 = np.arange(max(0, k_star), min(N, k_star + N))
        if len(idx2) > 0:
            vals2 = h[idx2 - k_star]
            g[idx2] += sign_ac * vals2
            
        # Update with momentum
        v = momentum * v + lr * g
        h -= v
        
        # Project to sum(h) = 1
        s = np.sum(h)
        if abs(s) < 1e-9:
            h = np.random.randn(N)
            s = np.sum(h)
        h -= (s - 1.0) / N
        
        # Calculate score for tracking
        new_val = _score_normalized(h)
        if new_val < current_val:
            current_val = new_val
        else:
            # If it worsens significantly, maybe reduce learning rate or revert?
            # For simplicity, we just continue. The max function is non-smooth,
            # so oscillation is expected.
            pass
            
    return h, current_val

def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin
    
    best_val = np.inf
    best_h = np.ones(N) / N
    
    # Initial baseline
    best_val = _score_normalized(best_h)
    
    # Generate diverse initializations
    inits = []
    
    # 1. Random Gaussian
    rng = np.random.default_rng(42)
    inits.append(rng.randn(N))
    
    # 2. Thue-Morse sequence (known for good spectral properties)
    tm = np.ones(N)
    for i in range(1, N):
        # Count trailing zeros or use bit trick
        # tm[i] = -tm[i//2] if i is odd? No, standard Thue-Morse: t(n) = parity of popcount(n)
        # t(n) = t(n//2) if n even, -t(n//2) if n odd? 
        # Actually, t(n) = (-1)^(number of 1s in binary n)
        # Let's just generate it simply
        pass
    
    # Simpler Thue-Morse generation
    t = np.array([1])
    while len(t) < N:
        t = np.concatenate([t, -t])
    inits.append(t[:N])
    
    # 3. Sine wave
    x = np.arange(N)
    inits.append(np.sin(2 * np.pi * 5 * x / N))
    
    # 4. Another random
    inits.append(rng.randn(N))
    
    # 5. Block pattern
    block = 10
    pat = np.tile([1]*block + [-1]*block, N // (2*block) + 1)
    inits.append(pat[:N])
    
    for h_init in inits:
        if time.time() > deadline - 5:
            break
            
        h_opt, val = _optimize_continuous(h_init, deadline=deadline - 2)
        
        if val < best_val:
            best_val = val
            best_h = h_opt
            
    return best_h

# EVOLVE-BLOCK-END
