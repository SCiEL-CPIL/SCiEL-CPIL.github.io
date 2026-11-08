import numpy as np
import time
from scipy.optimize import minimize

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

def _compute_soft_max(signed_step, beta=50.0):
    """
    Compute a smooth surrogate for max |autoconv| using log-sum-exp.
    This provides a differentiable approximation that guides the search.
    """
    M = 2 * N
    dx = 0.5 / M
    
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N-1)
    f_fine = signed_step[i_indices]
    
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    abs_autoconv = np.abs(autoconv)
    
    # Soft max via log-sum-exp for numerical stability
    max_val = np.max(abs_autoconv)
    if max_val < 1e-10:
        return 0.0
    shifted = abs_autoconv - max_val
    soft_max = max_val + np.log(np.sum(np.exp(beta * shifted)) / len(abs_autoconv)) / beta
    
    return soft_max

def construct():
    start_time = time.time()
    
    # Initialize with all ones (the seed)
    best_step = np.ones(N, dtype=np.int8)
    best_c3 = _compute_c3(best_step)
    
    np.random.seed(42)
    
    # Try random initializations
    for trial in range(8):
        if time.time() - start_time > DEADLINE - 10.0:
            break
        random_step = np.random.choice([-1, 1], size=N).astype(np.int8)
        c3_val = _compute_c3(random_step)
        if c3_val < best_c3:
            best_step = random_step.copy()
            best_c3 = c3_val
    
    # Local search using smooth surrogate for guidance
    max_iterations = 100000
    no_improve_count = 0
    patience = 2000
    beta = 50.0
    
    # Track both exact and soft objectives
    best_soft = _compute_soft_max(best_step, beta)
    
    for iteration in range(max_iterations):
        if time.time() - start_time > DEADLINE - 5.0:
            break
        
        # Favor single flips for efficiency
        r = np.random.random()
        if r < 0.8:
            num_flips = 1
        elif r < 0.95:
            num_flips = 2
        else:
            num_flips = 3
        
        if num_flips == 1:
            idxs = [np.random.randint(N)]
        elif num_flips == 2:
            idxs = list(np.random.choice(N, size=2, replace=False))
        else:
            idxs = list(np.random.choice(N, size=3, replace=False))
        
        old_vals = best_step[idxs].copy()
        
        for idx in idxs:
            best_step[idx] = -best_step[idx]
        
        # Use soft max for quick comparison during search
        new_soft = _compute_soft_max(best_step, beta)
        
        if new_soft < best_soft:
            # Accept based on soft objective, update exact
            new_c3 = _compute_c3(best_step)
            if new_c3 < best_c3:
                best_c3 = new_c3
            else:
                # Soft improved but exact didn't - still accept to explore
                pass
            best_soft = new_soft
            no_improve_count = 0
        else:
            for i, idx in enumerate(idxs):
                best_step[idx] = old_vals[i]
            no_improve_count += 1
        
        if no_improve_count >= patience:
            if time.time() - start_time > DEADLINE - 10.0:
                break
            # Perturbation: flip a random subset
            perturbation = np.random.choice([-1, 1], size=N).astype(np.int8)
            mask = np.random.random(N) < 0.2
            candidate = best_step.copy()
            candidate[mask] = perturbation[mask]
            c3_val = _compute_c3(candidate)
            if c3_val < best_c3:
                best_step = candidate
                best_c3 = c3_val
                best_soft = _compute_soft_max(best_step, beta)
            no_improve_count = 0
    
    # Final polish: do a few more exact evaluations around the best point
    if time.time() - start_time < DEADLINE - 10.0:
        # Try flipping each individual bit and see if it improves
        for idx in range(N):
            if time.time() - start_time > DEADLINE - 5.0:
                break
            old_val = best_step[idx]
            best_step[idx] = -best_step[idx]
            new_c3 = _compute_c3(best_step)
            if new_c3 < best_c3:
                best_c3 = new_c3
            else:
                best_step[idx] = old_val
    
    return best_step
