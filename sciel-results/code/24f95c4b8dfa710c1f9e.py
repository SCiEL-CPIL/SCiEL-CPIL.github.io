import numpy as np
import time

N = 400
DEADLINE = 120.0
M = 4 * N  # fine grid size
dx = 0.5 / M  # domain length is 0.5

def _compute_c3_fast(signed_step):
    """
    Fast computation of C3 = max_t |autoconv(t)| for signed step function.
    Uses vectorized fine-grid construction and FFT.
    """
    # Vectorized construction of fine grid from step function
    # Each step interval i maps to M//N fine grid points
    f_fine = np.repeat(signed_step.astype(np.float64), M // N)
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    return np.max(np.abs(autoconv))

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    C3 = max_t |autoconv(t)| where autoconv is the convolution of f with itself.
    """
    return _compute_c3_fast(signed_step)

def _soft_c3_from_autoconv(autoconv, beta=50.0):
    """Compute soft C3 from precomputed autoconvolution."""
    abs_autoconv = np.abs(autoconv)
    scaled = beta * abs_autoconv
    max_val = np.max(scaled)
    log_sum_exp = max_val + np.log(np.sum(np.exp(scaled - max_val)))
    return log_sum_exp / beta

def _compute_autoconv_fine(signed_step):
    """Compute autoconvolution for a signed step function on fine grid."""
    f_fine = np.repeat(signed_step.astype(np.float64), M // N)
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    return autoconv

def _continuous_optimize(x_init, beta, max_iters, deadline):
    """
    Projected gradient descent on continuous x in [-1,1]^N 
    to minimize soft C3 (log-sum-exp surrogate of max |autoconv|).
    """
    x = x_init.copy().astype(np.float64)
    best_x = x.copy()
    best_soft = _soft_c3_from_autoconv(_compute_autoconv_fine(np.sign(x)), beta)
    
    step_size = 0.1
    no_improve = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute gradient of soft C3 w.r.t. x
        # autoconv is linear in f, and f = repeat(x, M//N)
        # So d(autoconv)/dx_i = repeat(1 at positions corresponding to interval i)
        # We can compute this via FFT:
        # grad_i = <d(autoconv)/dx_i, w> where w = softmax weights of |autoconv|
        
        # Actually, let's do finite differences for simplicity since N=400 is manageable
        # But that's 400 FFTs per iteration which is too slow.
        
        # Better: analytic gradient
        # autoconv = ifft(fft_f * conj(fft_f)).real * dx
        # where f = repeat(x, k) with k = M//N
        # d(autoconv)/dx_i is the autoconvolution of a "bump" function that is 1 on interval i
        # This can be computed as: 
        #   d_autoconv_i = ifft(fft(bump_i) * conj(fft_f)).real * dx + ifft(fft_f * conj(fft(bump_i))).real * dx
        # But this is still N FFTs.
        
        # Alternative: use the fact that the gradient of the log-sum-exp of |autoconv|
        # is proportional to the softmax weights times the derivative of autoconv.
        # Let's use a simpler approach: coordinate descent or random coordinate updates.
        
        # For now, let's just do a simple local search in continuous space
        # by flipping individual coordinates and checking if soft C3 improves.
        
        idx = np.random.randint(N)
        old_val = x[idx]
        x[idx] = -x[idx]  # flip sign
        
        # Project to [-1, 1]
        x = np.clip(x, -1, 1)
        
        new_soft = _soft_c3_from_autoconv(_compute_autoconv_fine(np.sign(x)), beta)
        
        if new_soft < best_soft:
            best_soft = new_soft
            best_x = x.copy()
            no_improve = 0
        else:
            x[idx] = old_val
            no_improve += 1
        
        if no_improve > 200:
            break
    
    return best_x

def construct():
    start_time = time.time()
    
    # Initialize with all ones (the seed)
    best_step = np.ones(N, dtype=np.int8)
    best_c3 = _compute_c3(best_step)
    
    # Try a few random initializations
    np.random.seed(42)
    
    for trial in range(5):
        if time.time() - start_time > DEADLINE - 30.0:
            break
        random_step = np.random.choice([-1, 1], size=N).astype(np.int8)
        c3_val = _compute_c3(random_step)
        if c3_val < best_c3:
            best_step = random_step.copy()
            best_c3 = c3_val
    
    # Current state for SA
    current_step = best_step.copy()
    current_c3 = best_c3
    
    # SA Parameters
    T0 = 0.5
    alpha = 0.999
    T = T0
    reheat_threshold = 500
    no_improve_count = 0
    best_T_reheat = 0.5
    
    while True:
        # Check deadline with safety margin
        if time.time() - start_time > DEADLINE - 5.0:
            break
        
        # Reheat if stuck
        if no_improve_count >= reheat_threshold:
            T = best_T_reheat
            no_improve_count = 0
            # Perturb current state
            n_perturb = np.random.randint(5, 15)
            idxs = np.random.choice(N, size=n_perturb, replace=False)
            for i in idxs:
                current_step[i] = -current_step[i]
            current_c3 = _compute_c3(current_step)
        
        # Randomly choose move type
        if np.random.random() < 0.5:
            # Single flip
            idx = np.random.randint(N)
            old_val = current_step[idx]
            current_step[idx] = -old_val
            move_info = ('single', idx, old_val)
        else:
            # Multi-flip: flip 2-3 random distinct elements
            n_flips = np.random.randint(2, 4)
            idxs = np.random.choice(N, size=n_flips, replace=False)
            old_vals = [current_step[i] for i in idxs]
            for i in idxs:
                current_step[i] = -current_step[i]
            move_info = ('multi', idxs, old_vals)
            
        new_c3 = _compute_c3(current_step)
        
        # Acceptance Criteria
        delta = new_c3 - current_c3
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            # Accept
            current_c3 = new_c3
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_step = current_step.copy()
                no_improve_count = 0
        else:
            # Reject: revert
            if move_info[0] == 'single':
                current_step[move_info[1]] = move_info[2]
            else:
                for i, v in zip(move_info[1], move_info[2]):
                    current_step[i] = v
            no_improve_count += 1
        
        # Cool down
        T *= alpha
        
        # Safety check for T underflow
        if T < 1e-10:
             T = 1e-10
    
    # Final polish: local search with only improving moves
    if time.time() - start_time < DEADLINE - 2.0:
        polish_deadline = DEADLINE - 2.0
        for _ in range(1000):
            if time.time() > polish_deadline:
                break
            improved = False
            for idx in np.random.permutation(N):
                old_val = best_step[idx]
                best_step[idx] = -old_val
                new_c3 = _compute_c3(best_step)
                if new_c3 < best_c3:
                    best_c3 = new_c3
                    improved = True
                else:
                    best_step[idx] = old_val
            
            if not improved:
                break
    
    return best_step
