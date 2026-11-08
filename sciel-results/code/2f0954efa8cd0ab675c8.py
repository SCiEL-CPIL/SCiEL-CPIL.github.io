import numpy as np
import time

N = 400
DEADLINE = 145.0

def _compute_autoconv(signed_step):
    """
    Compute the autoconvolution array for a signed step function.
    """
    M = 2 * N
    dx = 0.5 / M
    
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N-1)
    f_fine = signed_step[i_indices].astype(np.float64)
    
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    return autoconv

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound: max |autoconv|
    """
    autoconv = _compute_autoconv(signed_step)
    return np.max(np.abs(autoconv))

def _compute_soft_c3(signed_step, beta=20.0):
    """
    Compute a smooth surrogate for max |autoconv| using log-sum-exp.
    This is differentiable and guides the search more effectively.
    """
    autoconv = _compute_autoconv(signed_step)
    abs_autoconv = np.abs(autoconv)
    
    max_val = np.max(abs_autoconv)
    if max_val < 1e-15:
        return 0.0
    
    # Soft max via log-sum-exp for numerical stability
    shifted = abs_autoconv - max_val
    # Use beta to control sharpness; higher beta = closer to hard max
    exp_vals = np.exp(beta * shifted)
    soft_max = max_val + np.log(np.sum(exp_vals) / len(abs_autoconv)) / beta
    
    return soft_max

def _compute_autoconv_delta(signed_step, flip_idx):
    """
    Compute the change in autoconv when flipping sign at flip_idx.
    Uses linearity: if we flip position i, the new f = old_f - 2*old_f[i]*e_i
    So new_autoconv = old_autoconv - 4*old_f[i]*(old_f * e_i) * dx
    where (old_f * e_i) is the convolution of f with a delta at i.
    """
    M = 2 * N
    dx = 0.5 / M
    
    old_autoconv = _compute_autoconv(signed_step)
    
    # The contribution of position i to the autoconvolution
    # f_new = f - 2*f[i]*delta_i
    # autoconv_new = (f - 2*f[i]*delta_i) * (f - 2*f[i]*delta_i)
    #              = f*f - 4*f[i]*(f*delta_i) + 4*f[i]^2*(delta_i*delta_i)
    # Since delta_i*delta_i is a delta at 0, and f[i]^2 = 1:
    # autoconv_new = autoconv_old - 4*f[i]*(f*delta_i) + 4*delta_0
    
    # Compute f * delta_i: this is just f shifted by i positions in the fine grid
    # In the fine grid, position i corresponds to indices [i*M//N, (i+1)*M//N)
    start_fine = i_fine_start(flip_idx, N, M)
    end_fine = start_fine + M // N
    
    f_fine = signed_step.astype(np.float64)
    # Create delta_i in fine grid
    delta_i = np.zeros(M, dtype=np.float64)
    delta_i[start_fine:end_fine] = 1.0 / (M // N)  # normalized so integral is 1
    
    # Convolution f * delta_i via FFT
    fft_f = np.fft.fft(f_fine, n=2*M)
    fft_delta = np.fft.fft(delta_i, n=2*M)
    conv_fd = np.fft.ifft(fft_f * fft_delta).real * dx
    
    # delta_0 in fine grid (centered)
    delta_0 = np.zeros(M, dtype=np.float64)
    center = M // 2
    delta_0[center] = 1.0 / (M // N)
    
    f_val = signed_step[flip_idx]
    
    # autoconv_new = autoconv_old - 4*f_val*conv_fd + 4*delta_0*dx
    # But we need to be careful with the scaling
    new_autoconv = old_autoconv - 4.0 * f_val * conv_fd
    
    return new_autoconv

def i_fine_start(i, N, M):
    """Get the start index in the fine grid for coarse interval i."""
    return i * M // N

def construct():
    start_time = time.time()
    
    # Initialize with all ones (the seed)
    best_step = np.ones(N, dtype=np.int8)
    best_c3 = _compute_c3(best_step)
    
    np.random.seed(42)
    
    # Try a few random initializations
    for trial in range(10):
        if time.time() - start_time > DEADLINE - 15.0:
            break
        random_step = np.random.choice([-1, 1], size=N).astype(np.int8)
        c3_val = _compute_c3(random_step)
        if c3_val < best_c3:
            best_step = random_step.copy()
            best_c3 = c3_val
    
    # Phase 1: Local search using soft surrogate for guidance
    # Use soft C3 to evaluate moves, but track exact C3 for the best
    max_iterations = 80000
    no_improve_count = 0
    patience = 5000
    
    current_soft_c3 = _compute_soft_c3(best_step, beta=20.0)
    
    for iteration in range(max_iterations):
        if time.time() - start_time > DEADLINE - 20.0:
            break
        
        # Single flip move
        idx = np.random.randint(N)
        
        old_val = best_step[idx]
        best_step[idx] = -old_val
        
        # Compute new soft C3
        new_soft_c3 = _compute_soft_c3(best_step, beta=20.0)
        
        if new_soft_c3 < current_soft_c3:
            current_soft_c3 = new_soft_c3
            # Also check exact C3
            exact_c3 = _compute_c3(best_step)
            if exact_c3 < best_c3:
                best_c3 = exact_c3
            no_improve_count = 0
        else:
            best_step[idx] = old_val
            no_improve_count += 1
        
        if no_improve_count >= patience:
            if time.time() - start_time > DEADLINE - 25.0:
                break
            # Perturbation: flip a random subset
            mask = np.random.random(N) < 0.15
            perturbation = np.random.choice([-1, 1], size=N).astype(np.int8)
            candidate = best_step.copy()
            candidate[mask] = perturbation[mask]
            
            cand_soft = _compute_soft_c3(candidate, beta=20.0)
            cand_exact = _compute_c3(candidate)
            
            if cand_soft < current_soft_c3 or cand_exact < best_c3:
                best_step = candidate
                current_soft_c3 = cand_soft
                if cand_exact < best_c3:
                    best_c3 = cand_exact
            no_improve_count = 0
    
    # Phase 2: Final polish with exact C3 evaluation
    # Do a few rounds of exact single-flip local search
    polish_patience = 2000
    polish_no_improve = 0
    polish_max_iter = 20000
    
    for p_iter in range(polish_max_iter):
        if time.time() - start_time > DEADLINE - 5.0:
            break
        
        idx = np.random.randint(N)
        old_val = best_step[idx]
        best_step[idx] = -old_val
        
        new_c3 = _compute_c3(best_step)
        
        if new_c3 < best_c3:
            best_c3 = new_c3
            polish_no_improve = 0
        else:
            best_step[idx] = old_val
            polish_no_improve += 1
        
        if polish_no_improve >= polish_patience:
            # Try a small perturbation
            if time.time() - start_time > DEADLINE - 10.0:
                break
            mask = np.random.random(N) < 0.1
            perturbation = np.random.choice([-1, 1], size=N).astype(np.int8)
            candidate = best_step.copy()
            candidate[mask] = perturbation[mask]
            cand_c3 = _compute_c3(candidate)
            if cand_c3 < best_c3:
                best_step = candidate
                best_c3 = cand_c3
            polish_no_improve = 0
    
    return best_step
