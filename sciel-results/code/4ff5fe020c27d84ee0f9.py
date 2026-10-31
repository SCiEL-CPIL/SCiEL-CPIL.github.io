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

def _thue_morse_sequence(n):
    """Generate Thue-Morse sequence of length n as +/-1."""
    # Thue-Morse: t(n) = parity of number of 1s in binary representation of n
    # Map 0 -> +1, 1 -> -1
    seq = np.zeros(n)
    for i in range(n):
        # Count set bits
        x = i
        count = 0
        while x:
            count += x & 1
            x >>= 1
        seq[i] = 1.0 if count % 2 == 0 else -1.0
    return seq

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
    Gradient descent on the smooth objective with soft-max penalty.
    Objective: minimize max_k |ac(k)| subject to sum(h) = N.
    """
    h = h.copy()
    # Normalize to sum = N
    s = np.sum(h)
    if abs(s) > 1e-9:
        h = h * (N / s)
    best_val = _score(h)
    
    size = 2 * N - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    lr = 0.05
    
    for step in range(300):
        if time.time() > deadline:
            break
        
        # Compute ac
        H = np.fft.rfft(h, fsize)
        ac = np.fft.irfft(H * H, fsize)[:size]
        
        # Find k* = argmax |ac|
        abs_ac = np.abs(ac)
        k_star = np.argmax(abs_ac)
        sign_ac = np.sign(ac[k_star])
        
        grad_M = np.zeros(N)
        if k_star == 0:
            grad_M = 2.0 * h
        else:
            idx1 = np.arange(N - k_star)
            grad_M[idx1] += sign_ac * h[idx1 + k_star]
            idx2 = np.arange(k_star, N)
            grad_M[idx2] += sign_ac * h[idx2 - k_star]
        
        h_new = h - lr * grad_M
        
        # Normalize to sum = N
        s_new = np.sum(h_new)
        if abs(s_new) > 1e-9:
            h_new = h_new * (N / s_new)
        else:
            h_new = h
        
        val_new = _score(h_new)
        
        if val_new < best_val:
            best_val = val_new
            h = h_new
            lr *= 0.99
        else:
            lr *= 0.95
            
    return h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    best_h = None
    best_val = 1e10
    
    np.random.seed(42)
    
    # Phase 1: Thue-Morse sequence
    h_tm = _thue_morse_sequence(N)
    val_tm = _score(h_tm)
    if val_tm < best_val:
        best_val = val_tm
        best_h = h_tm.copy()
    
    # Phase 2: Local sign flips on Thue-Morse
    h_opt, val_opt = _local_sign_flip(h_tm, max_passes=10, deadline=deadline)
    if val_opt < best_val:
        best_val = val_opt
        best_h = h_opt
    
    # Phase 3: Gradient descent from Thue-Morse
    h_grad, val_grad = _gradient_descent(h_opt, deadline)
    if val_grad < best_val:
        best_val = val_grad
        best_h = h_grad
    
    # Phase 4: More local sign flips
    h_opt2, val_opt2 = _local_sign_flip(h_grad, max_passes=5, deadline=deadline)
    if val_opt2 < best_val:
        best_val = val_opt2
        best_h = h_opt2
    
    # Phase 5: Random restarts with gradient descent
    n_restarts = 20
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        if r % 3 == 0:
            h = np.random.uniform(-1, 1, size=N)
        elif r % 3 == 1:
            h = np.random.choice([-1.0, 1.0], size=N)
        else:
            # Perturbed Thue-Morse
            h = _thue_morse_sequence(N)
            flip_mask = np.random.random(N) < 0.1
            h[flip_mask] *= -1
            
        # Ensure sum is not too small
        s = np.sum(h)
        if abs(s) < N * 0.1:
            if s < 0:
                h = -h
            h = h + 0.5
            h = np.sign(h)
            
        # Gradient descent
        h_grad, val_grad = _gradient_descent(h, deadline)
        
        # Local sign flips
        h_opt, val_opt = _local_sign_flip(h_grad, max_passes=3, deadline=deadline)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
            
    # Fallback if nothing found
    if best_h is None:
        best_h = np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
