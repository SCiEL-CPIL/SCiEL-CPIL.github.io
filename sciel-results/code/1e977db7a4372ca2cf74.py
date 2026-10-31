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
    ac = np.fft.irfft(H * H, fsize)[:size]
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _thue_morse_sequence(n):
    """Generate Thue-Morse sequence of length n as +/-1."""
    seq = np.zeros(n)
    for i in range(n):
        x = i
        count = 0
        while x:
            count += x & 1
            x >>= 1
        seq[i] = 1.0 if count % 2 == 0 else -1.0
    return seq

def _local_sign_flip_incremental(h, max_passes=20, deadline=None):
    """
    Local optimization with incremental autoconvolution update.
    When flipping h[i] to -h[i], the change in autoconvolution at lag k is:
    delta_ac[k] = -4 * h[i] * h[i+k] (for k>=0, within bounds)
    This allows O(N) update instead of O(N log N) FFT per step.
    """
    h = h.copy()
    size = 2 * N - 1
    ac = np.convolve(h, h, mode='full')
    abs_ac = np.abs(ac)
    best_val = _score(h)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            
            # Compute delta in autoconvolution if we flip h[i]
            # ac_new[k] = ac[k] - 4*h[i]*h[i+k] for k in [0, N-1-i]
            #            ac[k] - 4*h[i]*h[i-k] for k in [i+1, 2N-2]
            # More precisely, flipping h[i] to -h[i] changes ac by:
            # delta_ac[k] = -2*h[i]*(h[i+k] + h[i-k]) for valid indices
            # Actually: ac[k] = sum_j h[j]*h[j+k]
            # After flip: ac_new[k] = ac[k] - 2*h[i]*h[i+k] - 2*h[i]*h[i-k] (where defined)
            # But h[i+k] and h[i-k] terms... let's be more careful.
            # ac[k] = sum_{j=0}^{N-1-k} h[j]*h[j+k]
            # The terms involving h[i]: j=i gives h[i]*h[i+k], and j+k=i gives h[i-k]*h[i] (if i-k>=0)
            # So delta_ac[k] = -2*h[i]*h[i+k] (if 0<=i+k<N) - 2*h[i-k]*h[i] (if 0<=i-k<N-k... wait)
            # Let me just compute it directly for correctness, but use the structure.
            
            # For efficiency, compute the delta for all k
            # delta_ac[k] for k in [0, 2N-2]:
            # Term 1: j=i in sum, so h[i]*h[i+k], valid when 0<=i+k<N, i.e., k <= N-1-i
            # Term 2: j+k=i in sum, so h[i-k]*h[i], valid when 0<=i-k and i-k <= N-1-k, i.e., i-k>=0 and i<=N-1
            # So delta_ac[k] = -2*h[i]*h[i+k] (if i+k < N) - 2*h[i-k]*h[i] (if i-k >= 0 and i-k+k = i < N, always true)
            # Wait, for term 2: j = i-k, so h[j]*h[j+k] = h[i-k]*h[i], valid when 0 <= i-k <= N-1-k
            # i-k >= 0 means k <= i
            # i-k <= N-1-k means i <= N-1, always true
            # So: delta_ac[k] = -2*h[i]*h[i+k] (if k <= N-1-i) - 2*h[i-k]*h[i] (if k <= i)
            
            # For simplicity and correctness, let's compute delta for relevant k values
            # Actually, for a cleaner approach, let's just compute the full delta array
            delta = np.zeros(size)
            # k ranges from 0 to 2N-2
            # Term 1: for k in [0, N-1-i], delta[k] += -2*h[i]*h[i+k]
            if i + (N-1-i) >= 0:  # always true
                k1_end = N - i  # k from 0 to N-1-i inclusive
                if k1_end > 0:
                    delta[:k1_end] += -2.0 * h[i] * h[i:i+k1_end]
            # Term 2: for k in [0, i], delta[k] += -2*h[i-k]*h[i]
            if i >= 0:
                k2_end = i + 1  # k from 0 to i inclusive
                delta[:k2_end] += -2.0 * h[i] * h[i-k2_end+1:i+1][::-1]
            
            ac_new = ac + delta
            abs_ac_new = np.abs(ac_new)
            max_abs_new = np.max(abs_ac_new)
            s_val = np.sum(h)
            val_new = max_abs_new / (s_val * s_val)
            
            if val_new < best_val:
                # Accept the flip
                h[i] = -h[i]
                ac = ac_new
                abs_ac = abs_ac_new
                best_val = val_new
                improved = True
            # else: don't flip, ac stays the same
        
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
    
    best_h = np.ones(N)
    best_val = _score(best_h)
    
    np.random.seed(42)
    
    # Phase 1: Thue-Morse sequence
    h_tm = _thue_morse_sequence(N)
    val_tm = _score(h_tm)
    if val_tm < best_val:
        best_val = val_tm
        best_h = h_tm.copy()
    
    # Phase 2: Local sign flips on Thue-Morse with incremental update
    h_opt, val_opt = _local_sign_flip_incremental(h_tm, max_passes=15, deadline=deadline)
    if val_opt < best_val:
        best_val = val_opt
        best_h = h_opt
    
    # Phase 3: Gradient descent from Thue-Morse
    h_grad, val_grad = _gradient_descent(h_opt, deadline)
    if val_grad < best_val:
        best_val = val_grad
        best_h = h_grad
    
    # Phase 4: More local sign flips with incremental update
    h_opt2, val_opt2 = _local_sign_flip_incremental(h_grad, max_passes=10, deadline=deadline)
    if val_opt2 < best_val:
        best_val = val_opt2
        best_h = h_opt2
    
    # Phase 5: Random restarts with gradient descent and incremental local search
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
        
        # Local sign flips with incremental update
        h_opt, val_opt = _local_sign_flip_incremental(h_grad, max_passes=5, deadline=deadline)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
    
    # Final polish: more incremental local search on the best solution
    if time.time() < deadline - 5.0:
        h_final, val_final = _local_sign_flip_incremental(best_h, max_passes=10, deadline=deadline)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
    
    return best_h
# EVOLVE-BLOCK-END
