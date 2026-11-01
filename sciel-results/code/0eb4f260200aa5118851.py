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
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_val + log_sum / beta


def _epigraph_objective_and_gradient(x, beta=50.0):
    """
    Epigraph formulation: minimize t subject to soft_max_sidelobe(h) <= t.
    We use a smooth penalty: f(x) = t + lambda * max(0, soft_max - t)^2
    But for simplicity, we directly optimize soft_max with auxiliary variable.
    
    Actually, let's use a different approach: minimize t + mu * (soft_max - t)^2
    where mu is a large penalty parameter. This encourages t >= soft_max.
    
    For gradient-based optimization, we can use:
    f(x) = soft_max_sidelobe(h)  (original approach)
    
    But let's try a better smooth approximation using the log-sum-exp trick
    on both positive and negative parts.
    """
    h = x
    ac = _compute_autocorr(h)
    sidelobes = ac[1:N]
    
    # For each sidelobe, we want to minimize |ac[k]|
    # soft_max of |ac[k]| is not smooth at ac[k]=0
    # Instead, use: sum_k exp(2*beta*|ac[k]|)^(1/(2*beta*N))
    # Or use the trick: |x| ≈ sqrt(x^2 + eps)
    
    # Let's use a smoother approach: minimize sum of exp(beta * ac[k]^2)^(1/beta)
    # This is the "soft L-infinity" norm based on squares
    
    # Actually, let's stick with the original soft_max but make it more robust
    sidelobes_abs = np.abs(sidelobes)
    
    max_val = np.max(sidelobes_abs)
    if max_val < 1e-10:
        return max_val, np.zeros(N)
        
    # Use a stable log-sum-exp
    shifted = beta * (sidelobes_abs - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    obj = max_val + np.log(sum_exp) / beta
    
    # Gradient computation
    weights = beta * exp_shifted / sum_exp
    
    grad = np.zeros(N)
    
    for k in range(1, N):
        w_k = weights[k-1]
        if w_k < 1e-15:
            continue
        
        sign_k = np.sign(sidelobes[k-1])
        c_k = sign_k * w_k
        
        if c_k == 0:
            continue
            
        grad[0:N-k] += c_k * h[k:N]
        grad[k:N] += c_k * h[0:N-k]
        
    return obj, grad


def _local_search_fast(h, max_iterations=10000, time_limit=5.0, use_sa=False, T0=0.01):
    """
    Fast local search with incremental autocorrelation update.
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

    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        i = np.random.randint(N)
        old_h_i = h[i]
        new_h_i = -old_h_i
        
        # Incremental update of autocorrelation
        # ac[k] = sum_j h[j]*h[j+k]
        # When we flip h[i], the change in ac[k] is:
        # delta_ac[k] = new_h_i*old_h_i*(h[i+k] + h[i-k]) for valid indices
        # Since new_h_i = -old_h_i, delta_ac[k] = -old_h_i^2 * (h[i+k] + h[i-k])
        # = -(h[i+k] + h[i-k])
        
        delta_ac = np.zeros(2*N)
        
        # For k > 0: ac[k] changes by -(h[i+k] + h[i-k]) if both indices valid
        for k in range(1, N):
            change = 0.0
            if i + k < N:
                change += h[i+k]
            if i - k >= 0:
                change += h[i-k]
            delta_ac[k] = -change
            delta_ac[2*N-k] = -change  # symmetry
        
        ac_new = ac + delta_ac
        new_val = _max_sidelobe(ac_new)
        
        if new_val < current_val:
            # Accept improvement
            h[i] = new_h_i
            ac = ac_new
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            # Check for acceptance in SA
            accepted = False
            if T > 1e-9:
                delta = new_val - current_val
                if delta < 0:
                    accepted = True
                else:
                    prob = np.exp(-delta / T)
                    if np.random.rand() < prob:
                        accepted = True
            
            if accepted:
                h[i] = new_h_i
                ac = ac_new
                current_val = new_val
            else:
                # Revert
                h[i] = old_h_i
        
        if T > 1e-9:
            T *= 0.9995

    return best_h, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Continuous Relaxation with Multiple Strategies
    num_continuous_starts = 8
    betas = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
    
    for i in range(num_continuous_starts):
        if time.time() >= deadline - 40.0:
            break
            
        rng = np.random.default_rng(42 + i * 137)
        
        # Different initialization strategies
        if i < 2:
            # Binary initialization
            x0 = rng.choice([-1.0, 1.0], size=N)
        elif i < 4:
            # Near-binary with noise
            x0 = rng.choice([-1.0, 1.0], size=N) * 0.9 + rng.normal(0, 0.1, size=N)
            x0 = np.clip(x0, -1.0, 1.0)
        elif i < 6:
            # Continuous random
            x0 = rng.uniform(-1.0, 1.0, size=N)
        else:
            # Start from best found so far with perturbation
            if best_h is not None:
                x0 = best_h.copy()
                mask = rng.random(N) < 0.2
                x0[mask] = rng.choice([-1.0, 1.0], size=np.sum(mask))
            else:
                x0 = rng.choice([-1.0, 1.0], size=N)
        
        beta = betas[i % len(betas)]
        
        def obj_grad(x, beta=beta):
            return _epigraph_objective_and_gradient(x, beta)
        
        try:
            res = minimize(obj_grad, x0, method='L-BFGS-B', 
                           bounds=[(-1.0, 1.0)] * N, 
                           options={'maxiter': 30, 'ftol': 1e-9, 'gtol': 1e-7, 'maxls': 10, 'disp': False})
        except Exception:
            continue
        
        # Round to binary using multiple strategies
        candidates = []
        
        # Strategy 1: sign
        h_bin1 = np.sign(res.x)
        zero_mask = (h_bin1 == 0)
        if np.any(zero_mask):
            h_bin1[zero_mask] = rng.choice([-1.0, 1.0], size=np.sum(zero_mask))
        candidates.append(h_bin1)
        
        # Strategy 2: threshold at 0 with random tie-breaking
        h_bin2 = np.where(res.x > 0, 1.0, -1.0)
        candidates.append(h_bin2)
        
        # Strategy 3: probabilistic rounding
        probs = 0.5 * (1 + res.x)  # map [-1,1] to [0,1]
        h_bin3 = np.where(rng.random(N) < probs, 1.0, -1.0)
        candidates.append(h_bin3)
        
        for h_bin in candidates:
            ac = _compute_autocorr(h_bin)
            val = _max_sidelobe(ac)
            
            if val < best_val:
                best_val = val
                best_h = h_bin.copy()
            
            # Quick local search refinement
            if time.time() < deadline - 25.0:
                h_ref, val_ref = _local_search_fast(h_bin.copy(), max_iterations=500, time_limit=0.5)
                if val_ref < best_val:
                    best_val = val_ref
                    best_h = h_ref

    # Phase 2: Thorough local search on best solution
    if best_h is not None and time.time() < deadline - 15.0:
        h_ref, val_ref = _local_search_fast(best_h.copy(), max_iterations=20000, time_limit=10.0)
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref

    # Phase 3: Final SA polish
    remaining_time = deadline - time.time()
    if remaining_time > 5.0 and best_h is not None:
        T0 = max(0.005, best_val * 0.03)
        h_final, val_final = _local_search_fast(best_h.copy(), max_iterations=50000, time_limit=remaining_time - 1.0, use_sa=True, T0=T0)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
            
    if best_h is None:
        return np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
