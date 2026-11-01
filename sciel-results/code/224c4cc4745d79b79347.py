"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_autocorr(h):
    """
    Compute linear aperiodic autocorrelation using FFT.
    Returns array of length 2*N.
    """
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    return ac


def _gradient_autocorr(h, ac):
    """
    Compute gradient of each sidelobe ac[k] w.r.t. h using FFT.
    Returns array grad where grad[k, i] = d(ac[k])/dh[i]
    """
    # ac[k] = sum_j h[j] * h[j+k]
    # d(ac[k])/dh[i] = h[i+k] (if i+k < N) + h[i-k] (if i-k >= 0)
    # This is a convolution-like structure we can compute efficiently
    # For each k, the gradient is a shifted version of h
    # grad_k = h[k:] (padded) + h[:-k] (padded)
    
    # We'll compute this in a vectorized way for all k at once
    # Actually, let's use the fact that the Jacobian has a Toeplitz-like structure
    # d(ac[k])/dh[i] = h[i+k] + h[i-k]
    
    # For efficiency with N=400, we can build the full gradient matrix
    # But that's 400x400 = 160k entries, which is fine
    
    grad_matrix = np.zeros((N-1, N))  # grad_matrix[k-1, i] = d(ac[k])/dh[i]
    
    for k in range(1, N):
        # d(ac[k])/dh[i] = h[i+k] if i+k < N, else 0
        #                  + h[i-k] if i-k >= 0, else 0
        if k < N:
            grad_matrix[k-1, :N-k] += h[k:]
            grad_matrix[k-1, k:] += h[:N-k]
    
    return grad_matrix


def _max_sidelobe(ac):
    """
    Compute max absolute value of sidelobes.
    """
    # Due to symmetry, max |ac[1:N]| is sufficient
    return np.max(np.abs(ac[1:N]))


def _soft_max_sidelobe(ac, beta=50.0):
    """
    Smooth approximation of max |ac[1:N]| using log-sum-exp.
    """
    sidelobes = np.abs(ac[1:N])
    max_val = np.max(sidelobes)
    if max_val < 1e-10:
        return max_val
    
    shifted = beta * (sidelobes - max_val)
    # Clip to avoid overflow in exp (though shifted <= 0)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_val + log_sum / beta


def _objective_and_gradient_v2(x, beta=50.0):
    """
    Improved objective and gradient using FFT-based gradient computation.
    """
    h = x
    ac = _compute_autocorr(h)
    sidelobes_abs = np.abs(ac[1:N])
    
    max_val = np.max(sidelobes_abs)
    if max_val < 1e-10:
        return max_val, np.zeros(N)
        
    shifted = beta * (sidelobes_abs - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    obj = max_val + np.log(sum_exp) / beta
    
    weights = beta * exp_shifted / sum_exp
    
    # Compute gradient using the structure: d(ac[k])/dh[i] = h[i+k] + h[i-k]
    grad = np.zeros(N)
    
    for k in range(1, N):
        w_k = weights[k-1]
        if w_k < 1e-15:
            continue
        
        sign_k = np.sign(ac[k])
        c_k = sign_k * w_k
        
        if c_k == 0:
            continue
            
        # Vectorized update
        grad[0:N-k] += c_k * h[k:N]
        grad[k:N] += c_k * h[0:N-k]
        
    return obj, grad


def _objective_and_gradient(x, beta=50.0):
    """
    Objective: Smooth approximation of max sidelobe.
    Gradient: Analytic.
    """
    h = x
    ac = _compute_autocorr(h)
    sidelobes_abs = np.abs(ac[1:N])
    
    # Soft max value
    max_val = np.max(sidelobes_abs)
    if max_val < 1e-10:
        return max_val, np.zeros(N)
        
    shifted = beta * (sidelobes_abs - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    obj = max_val + np.log(sum_exp) / beta
    
    # Gradient of soft max w.r.t. sidelobes
    # d/dx_i (soft_max) = sum_k w_k * d|ac[k]|/dx_i
    # where w_k = exp(beta * (ac[k] - max)) / sum_exp = beta * exp_shifted[k] / sum_exp
    # Note: d|ac[k]|/dx_i = sign(ac[k]) * d(ac[k])/dx_i
    # d(ac[k])/dx_i = h[i+k] + h[i-k] (if valid)
    
    weights = beta * exp_shifted / sum_exp
    
    grad = np.zeros(N)
    
    for k in range(1, N):
        w_k = weights[k-1]
        if w_k < 1e-15:
            continue
        
        # ac[k] = sum_j h[j] h[j+k]
        # d(ac[k])/dx_i = h[i+k] + h[i-k] (if valid)
        # sign(ac[k]) * d(ac[k])/dx_i
        
        sign_k = np.sign(ac[k])
        c_k = sign_k * w_k
        
        if c_k == 0:
            continue
            
        # Vectorized update
        # Indices i such that i+k < N => i < N-k
        # grad[0:N-k] += c_k * h[k:N]
        # Indices i such that i-k >= 0 => i >= k
        # grad[k:N] += c_k * h[0:N-k]
        
        grad[0:N-k] += c_k * h[k:N]
        grad[k:N] += c_k * h[0:N-k]
        
    return obj, grad


def _local_search(h, max_iterations=10000, time_limit=5.0, use_sa=False, T0=0.01):
    """
    Fast local search using vectorized incremental autocorrelation updates for bit flips.
    Optionally uses Simulated Annealing (SA) to escape local minima.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    best_val = current_val
    best_h = h.copy()
    
    if use_sa and current_val > 1e-10:
        T = T0
    else:
        T = 0.0

    # Precompute index arrays for vectorized updates
    k_indices = np.arange(1, N)
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Try flipping a random bit
        i = np.random.randint(N)
        old_h_i = h[i]
        
        # Vectorized computation of delta for all k
        # delta_k = h[i+k] (if i+k < N) + h[i-k] (if i-k >= 0)
        i_plus_k = i + k_indices
        i_minus_k = i - k_indices
        
        mask_plus = i_plus_k < N
        mask_minus = i_minus_k >= 0
        
        delta = np.zeros(N - 1)
        if np.any(mask_plus):
            delta[mask_plus] += h[i_plus_k[mask_plus]]
        if np.any(mask_minus):
            delta[mask_minus] += h[i_minus_k[mask_minus]]
        
        # ac[k] changes by -2 * old_h_i * delta_k
        ac_new = ac.copy()
        ac_new[1:N] += -2.0 * old_h_i * delta
        
        new_val = _max_sidelobe(ac_new)
        
        if new_val < current_val:
            # Accept improvement
            current_val = new_val
            h[i] = -old_h_i  # Apply the flip
            ac = ac_new
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            # Check for acceptance in SA
            accepted = False
            if T > 1e-9:
                delta_val = new_val - current_val
                if delta_val < 0:
                    accepted = True
                else:
                    prob = np.exp(-delta_val / T)
                    if np.random.rand() < prob:
                        accepted = True
            
            if accepted:
                current_val = new_val
                h[i] = -old_h_i  # Apply the flip
                ac = ac_new
            else:
                # Revert: h[i] stays as old_h_i, ac stays as ac
                pass
        
        if T > 1e-9:
            T *= 0.9995  # Cooling factor

    return best_h, best_val


def _local_search_multi(h, max_iterations=10000, time_limit=5.0, use_sa=False, T0=0.01, num_flips=2):
    """
    Local search that flips multiple bits at once for better escape from local minima.
    Uses incremental update for efficiency.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    best_val = current_val
    best_h = h.copy()
    
    if use_sa and current_val > 1e-10:
        T = T0
    else:
        T = 0.0

    k_indices = np.arange(1, N)
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Try flipping multiple random bits
        indices = np.random.choice(N, size=num_flips, replace=False)
        old_values = h[indices].copy()
        h[indices] = -h[indices]
        
        # Incremental update: for each flipped index, compute delta
        ac_new = ac.copy()
        for idx in indices:
            old_val = old_values[indices.tolist().index(idx)]
            i_plus_k = idx + k_indices
            i_minus_k = idx - k_indices
            
            mask_plus = i_plus_k < N
            mask_minus = i_minus_k >= 0
            
            delta = np.zeros(N - 1)
            if np.any(mask_plus):
                delta[mask_plus] += h[i_plus_k[mask_plus]]
            if np.any(mask_minus):
                delta[mask_minus] += h[i_minus_k[mask_minus]]
            
            ac_new[1:N] += -2.0 * old_val * delta
        
        new_val = _max_sidelobe(ac_new)
        
        if new_val < current_val:
            current_val = new_val
            ac = ac_new
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            accepted = False
            if T > 1e-9:
                delta_val = new_val - current_val
                if delta_val < 0:
                    accepted = True
                else:
                    prob = np.exp(-delta_val / T)
                    if np.random.rand() < prob:
                        accepted = True
            
            if accepted:
                current_val = new_val
                ac = ac_new
            else:
                h[indices] = old_values
        
        if T > 1e-9:
            T *= 0.9995  # Cooling factor

    return best_h, best_val


def _tanh_objective_and_gradient(z, beta=50.0):
    """
    Objective and gradient for tanh-mapped variables.
    z are unconstrained real variables, h = tanh(z) in [-1, 1].
    """
    h = np.tanh(z)
    ac = _compute_autocorr(h)
    sidelobes_abs = np.abs(ac[1:N])
    
    max_val = np.max(sidelobes_abs)
    if max_val < 1e-10:
        return max_val, np.zeros(N)
    
    shifted = beta * (sidelobes_abs - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    obj = max_val + np.log(sum_exp) / beta
    
    # Gradient of soft max w.r.t. sidelobes
    weights = beta * exp_shifted / sum_exp
    
    # Gradient w.r.t. h
    grad_h = np.zeros(N)
    for k in range(1, N):
        w_k = weights[k-1]
        if w_k < 1e-15:
            continue
        sign_k = np.sign(ac[k])
        c_k = sign_k * w_k
        if c_k == 0:
            continue
        grad_h[0:N-k] += c_k * h[k:N]
        grad_h[k:N] += c_k * h[0:N-k]
    
    # Chain rule: dz_i = dh_i * (1 - h_i^2)
    grad_z = grad_h * (1.0 - h**2)
    
    return obj, grad_z


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Continuous Relaxation with Tanh Mapping
    # Try multiple random initializations with different betas and strategies
    num_continuous_starts = 8
    betas = [20.0, 40.0, 80.0, 160.0]
    
    for i in range(num_continuous_starts):
        if time.time() >= deadline - 60.0:  # Reserve 60s for final polish
            break
            
        rng = np.random.default_rng(1000 + i)
        
        # Different initialization strategies
        strategy = i % 4
        if strategy == 0:
            # Binary initialization
            h_init = rng.choice([-1.0, 1.0], size=N)
        elif strategy == 1:
            # Block pattern: alternating blocks of +1 and -1
            block_size = N // 8
            h_init = np.ones(N)
            for b in range(8):
                if b % 2 == 1:
                    start = b * block_size
                    end = start + block_size if b < 7 else N
                    h_init[start:end] = -1.0
        elif strategy == 2:
            # Sparse initialization
            h_init = np.zeros(N)
            nonzero_idx = rng.choice(N, size=N//3, replace=False)
            h_init[nonzero_idx] = rng.choice([-1.0, 1.0], size=len(nonzero_idx))
        else:
            # Smooth initialization with small noise
            h_init = rng.normal(0, 0.3, size=N)
            h_init = np.sign(h_init) * np.clip(np.abs(h_init), 0.1, 1.0)
        
        # arctanh of ±1 is infinite, so clip
        h_clipped = np.clip(h_init, -0.999, 0.999)
        z0 = np.arctanh(h_clipped)
        
        beta = betas[i % len(betas)]
        
        def obj_grad(z, beta=beta):
            return _tanh_objective_and_gradient(z, beta)
        
        # Use L-BFGS-B on unconstrained z
        try:
            res = minimize(obj_grad, z0, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-7, 'maxls': 30})
        except Exception:
            continue
        
        # Map back to h
        h_cont = np.tanh(res.x)
        
        # Round to binary with threshold
        threshold = 0.1
        h_bin = np.where(h_cont > threshold, 1.0, np.where(h_cont < -threshold, -1.0, 0.0))
        zero_mask = (h_bin == 0)
        if np.any(zero_mask):
            h_bin[zero_mask] = rng.choice([-1.0, 1.0], size=np.sum(zero_mask))
            
        # Evaluate binary solution
        ac = _compute_autocorr(h_bin)
        val = _max_sidelobe(ac)
        
        if val < best_val:
            best_val = val
            best_h = h_bin.copy()
            
        # Quick local search refinement on this candidate
        if time.time() < deadline - 45.0:
            h_ref, val_ref = _local_search(h_bin.copy(), max_iterations=5000, time_limit=4.0)
            if val_ref < best_val:
                best_val = val_ref
                best_h = h_ref

    # Phase 2: Refine best solution with local search (single-bit)
    if best_h is not None and time.time() < deadline - 30.0:
        h_ref, val_ref = _local_search(best_h.copy(), max_iterations=20000, time_limit=15.0)
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref

    # Phase 2b: Multi-bit local search for better escape
    if best_h is not None and time.time() < deadline - 20.0:
        h_ref, val_ref = _local_search_multi(best_h.copy(), max_iterations=15000, time_limit=10.0, num_flips=2)
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref

    # Phase 2c: Another round of single-bit local search after multi-bit
    if best_h is not None and time.time() < deadline - 10.0:
        h_ref, val_ref = _local_search(best_h.copy(), max_iterations=20000, time_limit=8.0)
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref

    # Phase 3: Final Discrete Search with Simulated Annealing
    remaining_time = deadline - time.time()
    if remaining_time > 5.0 and best_h is not None:
        # Use SA to potentially escape the local minimum found by greedy local search
        T0 = max(0.005, best_val * 0.03) 
        h_final, val_final = _local_search(best_h.copy(), max_iterations=500000, time_limit=remaining_time - 2.0, use_sa=True, T0=T0)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
            
    if best_h is None:
        return np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
