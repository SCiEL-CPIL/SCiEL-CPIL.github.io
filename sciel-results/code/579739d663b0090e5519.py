import numpy as np
import time

N = 400
DEADLINE = 145.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    C3 = max_t |autoconv(t)| where autoconv is the convolution of f with itself.
    """
    M = 2 * N  # fine grid points
    dx = 0.5 / M
    
    # Vectorized construction of fine grid
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N-1)
    f_fine = signed_step[i_indices]
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    c3 = np.max(np.abs(autoconv))
    
    return c3

def _compute_c3_soft(signed_step, beta=50.0):
    """
    Compute a smooth surrogate of C3 using log-sum-exp.
    This is differentiable and helps gradient-based optimization.
    """
    M = 2 * N
    dx = 0.5 / M
    
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N-1)
    f_fine = signed_step[i_indices]
    
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    # Soft max of |autoconv|
    abs_autoconv = np.abs(autoconv)
    scaled = beta * abs_autoconv
    max_val = np.max(scaled)
    log_sum_exp = max_val + np.log(np.sum(np.exp(scaled - max_val)))
    soft_c3 = log_sum_exp / beta
    
    return soft_c3

def _soft_c3_from_autoconv(autoconv, beta=50.0):
    """Compute soft C3 from precomputed autoconvolution."""
    abs_autoconv = np.abs(autoconv)
    scaled = beta * abs_autoconv
    max_val = np.max(scaled)
    log_sum_exp = max_val + np.log(np.sum(np.exp(scaled - max_val)))
    return log_sum_exp / beta

def _compute_autoconv(signed_step):
    """Compute autoconvolution for a signed step function."""
    M = 2 * N
    dx = 0.5 / M
    
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N-1)
    f_fine = signed_step[i_indices]
    
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    return autoconv

def _gradient_soft_c3(x, beta=50.0):
    """
    Compute analytic gradient of soft C3 with respect to continuous vector x.
    x is in [-1, 1]^N.
    """
    M = 2 * N
    dx = 0.5 / M
    
    # Map x to fine grid
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N-1)
    f_fine = x[i_indices]
    
    # Autoconvolution
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    # Gradient of log-sum-exp of |autoconv|
    abs_autoconv = np.abs(autoconv)
    scaled = beta * abs_autoconv
    max_val = np.max(scaled)
    
    # Stable log-sum-exp and weights
    exp_scaled = np.exp(scaled - max_val)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp
    
    # d(soft_c3)/d(autoconv_k) = beta * weights_k * sign(autoconv_k)
    grad_autoconv = beta * weights * np.sign(autoconv)
    
    # Chain rule: d(autoconv)/d(f_fine)
    # Using FFT for linear correlation requires zero padding.
    fft_grad_auto_2M = np.fft.fft(grad_autoconv, n=2*M)
    fft_f_2M = np.fft.fft(f_fine, n=2*M)
    grad_f_fine = np.fft.ifft(fft_grad_auto_2M * np.conj(fft_f_2M)).real[:M] * (2 * dx)
    
    # Map back to x
    grad_x = np.zeros(N)
    np.add.at(grad_x, i_indices, grad_f_fine)
    
    return grad_x

def _multi_flip_polish(current_step, current_c3, best_step, best_c3, max_time):
    """
    Local search with multi-flip moves (2-3 simultaneous flips) to escape
    shallow local minima.
    """
    no_improve = 0
    patience = 500
    rng = np.random.default_rng(12345)
    
    while time.time() < max_time:
        if no_improve >= patience:
            break
        
        # Choose 2 or 3 random indices to flip
        n_flips = rng.choice([2, 3], p=[0.7, 0.3])
        idxs = rng.choice(N, size=n_flips, replace=False)
        
        old_vals = current_step[idxs].copy()
        current_step[idxs] = -current_step[idxs]
        
        new_c3 = _compute_c3(current_step)
        
        if new_c3 < current_c3:
            current_c3 = new_c3
            if current_c3 < best_c3:
                best_c3 = current_c3
                best_step = current_step.copy()
            no_improve = 0
        else:
            current_step[idxs] = old_vals
            no_improve += 1
    
    return current_step, current_c3, best_step, best_c3

def construct():
    start_time = time.time()
    
    # Initialize with all ones
    best_step = np.ones(N, dtype=np.int8)
    best_c3 = _compute_c3(best_step)
    
    rng = np.random.default_rng(42)
    
    # Try random initializations to find a good starting point
    for trial in range(15):
        if time.time() - start_time > DEADLINE - 15.0:
            break
        random_step = rng.choice([-1, 1], size=N).astype(np.int8)
        c3_val = _compute_c3(random_step)
        if c3_val < best_c3:
            best_step = random_step.copy()
            best_c3 = c3_val
    
    # Continuous Optimization Phase
    x = best_step.astype(np.float64)
    
    beta = 50.0
    step_size = 0.1
    
    # Adaptive step size gradient descent
    for grad_iter in range(800):
        if time.time() - start_time > DEADLINE - 25.0:
            break
        
        grad = _gradient_soft_c3(x, beta=beta)
        
        norm_grad = np.linalg.norm(grad)
        if norm_grad < 1e-8:
            break
            
        current_step_size = step_size / (1.0 + 0.005 * grad_iter)
        
        x_new = x - current_step_size * grad
        x_new = np.clip(x_new, -1.0, 1.0)
        
        if grad_iter % 8 == 0:
             soft_val_new = _compute_c3_soft(x_new, beta=beta)
             soft_val_old = _compute_c3_soft(x, beta=beta)
             if soft_val_new > soft_val_old:
                 step_size *= 0.5
                 continue
             else:
                 x = x_new
                 cand = np.sign(x)
                 cand[cand == 0] = 1
                 cand_c3 = _compute_c3(cand)
                 if cand_c3 < best_c3:
                     best_c3 = cand_c3
                     best_step = cand.astype(np.int8)
                     step_size = 0.1
        else:
             x = x_new
    
    # Final Hard Thresholding
    final_step = np.sign(x)
    final_step[final_step == 0] = 1
    final_step = final_step.astype(np.int8)
    final_c3 = _compute_c3(final_step)
    
    if final_c3 < best_c3:
        best_c3 = final_c3
        best_step = final_step.copy()
    
    # Discrete Polish Phase with multi-flip moves
    polish_deadline = time.time() + (DEADLINE - 5.0 - (time.time() - start_time))
    polish_deadline = min(polish_deadline, start_time + DEADLINE - 5.0)
    
    current_step = best_step.copy()
    current_c3 = best_c3
    
    current_step, current_c3, best_step, best_c3 = _multi_flip_polish(
        current_step, current_c3, best_step, best_c3, polish_deadline
    )
    
    # Single-flip polish with remaining time
    if time.time() < start_time + DEADLINE - 3.0:
        no_improve = 0
        patience = 300
        while time.time() < start_time + DEADLINE - 3.0:
            if no_improve >= patience:
                break
            idx = rng.integers(N)
            old_val = current_step[idx]
            current_step[idx] = -old_val
            new_c3 = _compute_c3(current_step)
            if new_c3 < current_c3:
                current_c3 = new_c3
                if current_c3 < best_c3:
                    best_c3 = current_c3
                    best_step = current_step.copy()
                no_improve = 0
            else:
                current_step[idx] = old_val
                no_improve += 1
    
    return best_step
