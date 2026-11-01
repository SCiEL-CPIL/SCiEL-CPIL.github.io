import numpy as np
import time

N = 400

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

def _epigraph_objective_and_gradient(x, t, beta=50.0):
    """
    Epigraph formulation: minimize t subject to |ac[k]| <= t for all k in 1..N-1.
    Smoothed via log-sum-exp on the constraint violations.
    Returns (objective, grad_x, grad_t).
    """
    h = x
    ac = _compute_autocorr(h)
    sidelobes_abs = np.abs(ac[1:N])
    
    # Smooth max of (sidelobes_abs - t)
    violations = sidelobes_abs - t
    max_violation = np.max(violations)
    
    if max_violation < -1e-10:
        # All constraints satisfied, objective is t
        return t, np.zeros(N), np.ones()
    
    shifted = beta * (violations - max_violation)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    smooth_max = max_violation + np.log(sum_exp) / beta
    
    # Objective: t + max(0, smooth_max) approximated smoothly
    # Use softplus-like: t + smooth_relu(smooth_max)
    # For simplicity, use t + smooth_max when positive, else t
    # Actually, the epigraph objective is: t + max(0, max_k (|ac[k]| - t))
    # Smooth: t + softplus(beta * smooth_max) / beta... let's use a simpler approach
    
    # Direct smoothed epigraph: minimize t + (1/beta)*log(sum(exp(beta*(|ac[k]|-t))))
    # This is smooth and approximates t + max(0, max_k(|ac[k]|-t))
    # But this can be negative. Let's use:
    # f(x,t) = t + (1/beta)*log(sum(exp(beta*(|ac[k]|-t))))
    
    obj = t + np.log(sum_exp) / beta
    
    weights = beta * exp_shifted / sum_exp  # sum to 1
    
    grad_x = np.zeros(N)
    for k in range(1, N):
        w_k = weights[k-1]
        if w_k < 1e-12:
            continue
        
        sign_k = np.sign(ac[k])
        c_k = sign_k * w_k
        
        if c_k == 0:
            continue
            
        grad_x[0:N-k] += c_k * h[k:N]
        grad_x[k:N] += c_k * h[0:N-k]
    
    grad_t = 1.0 - np.sum(weights)  # = 1.0 - 1.0 = 0? No.
    # d/dt of (1/beta)*log(sum(exp(beta*(s_k - t)))) = (1/beta)*sum(-beta*exp(beta*(s_k-t)))/sum(exp(...)) = -1
    # So grad_t = 1 - 1 = 0? That's not right.
    # f = t + (1/beta)*log(sum(exp(beta*(s_k - t))))
    # df/dt = 1 + (1/beta)*sum(-beta*exp(beta*(s_k-t)))/sum(exp(...)) = 1 - 1 = 0
    # Hmm, that means t doesn't appear in the gradient? Let me re-derive.
    # Actually this formulation has the issue that t cancels out. Let me use a different approach.
    
    # Better: use the original soft-max objective but with t as a threshold
    # Minimize t subject to soft constraints. Use augmented Lagrangian style.
    # Or simply: f = t + penalty where penalty = sum of soft violations
    
    # Let's go back to the simpler log-sum-exp max objective, which works well.
    # The epigraph approach needs more care. For now, use the working soft-max.
    grad_t = 0.0  # placeholder, we'll use the simpler approach
    
    return obj, grad_x, grad_t

def _objective_and_gradient(x, beta=50.0):
    """
    Objective: Smooth approximation of max sidelobe.
    Gradient: Analytic.
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
    
    grad = np.zeros(N)
    
    for k in range(1, N):
        w_k = weights[k-1]
        if w_k < 1e-12:
            continue
        
        sign_k = np.sign(ac[k])
        c_k = sign_k * w_k
        
        if c_k == 0:
            continue
            
        grad[0:N-k] += c_k * h[k:N]
        grad[k:N] += c_k * h[0:N-k]
        
    return obj, grad

def _polish_discrete(h, max_iterations=50000, time_limit=10.0):
    """
    High-precision polish using exact nonsmooth objective.
    Uses greedy coordinate descent with look-ahead.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    best_val = current_val
    best_h = h.copy()
    
    # Greedy: find the flip that gives the largest improvement
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Try all positions and pick the best flip
        best_delta = 0.0
        best_i = -1
        
        for i in range(N):
            old_h_i = h[i]
            
            # Compute delta in autocorrelation from flipping h[i]
            delta_ac = np.zeros(2*N)
            for k in range(1, N):
                delta = 0.0
                if i + k < N:
                    delta -= 2.0 * old_h_i * h[i + k]
                if i - k >= 0:
                    delta -= 2.0 * h[i - k] * old_h_i
                delta_ac[k] = delta
                delta_ac[2*N - k] = delta
            
            ac_new = ac + delta_ac
            new_val = _max_sidelobe(ac_new)
            delta = new_val - current_val
            
            if delta < best_delta:
                best_delta = delta
                best_i = i
        
        if best_i < 0 or best_delta >= -1e-15:
            break
        
        # Apply the best flip
        i = best_i
        old_h_i = h[i]
        h[i] = -old_h_i
        
        delta_ac = np.zeros(2*N)
        for k in range(1, N):
            delta = 0.0
            if i + k < N:
                delta -= 2.0 * old_h_i * h[i + k]
            if i - k >= 0:
                delta -= 2.0 * h[i - k] * old_h_i
            delta_ac[k] = delta
            delta_ac[2*N - k] = delta
        
        ac = ac + delta_ac
        current_val += best_delta
        
        if current_val < best_val:
            best_val = current_val
            best_h = h.copy()
        
        # Early termination if we've reached a very good value
        if best_val < 0.01:
            break
    
    return best_h, best_val

def _local_search(h, max_iterations=10000, time_limit=5.0, use_sa=False, T0=0.01, parallel_tempering=False):
    """
    Fast local search using incremental autocorrelation updates.
    Optionally uses Simulated Annealing (SA) or Parallel Tempering to escape local minima.
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
    
    # For parallel tempering, we maintain multiple replicas at different temperatures
    if parallel_tempering:
        num_replicas = 5
        temperatures = [0.01 * (0.5 ** i) for i in range(num_replicas)]
        # replicas: list of (h, ac, val, T)
        replicas = []
        for i in range(num_replicas):
            h_rep = h.copy()
            ac_rep = _compute_autocorr(h_rep)
            val_rep = _max_sidelobe(ac_rep)
            replicas.append([h_rep, ac_rep, val_rep, temperatures[i]])
        
        for _ in range(max_iterations):
            if time.time() - start_time > time_limit:
                break
            
            # For each replica, try a flip
            for r in range(num_replicas):
                h_rep = replicas[r][0]
                ac_rep = replicas[r][1]
                val_rep = replicas[r][2]
                T_rep = replicas[r][3]
                
                i = np.random.randint(N)
                old_h_i = h_rep[i]
                h_rep[i] = -old_h_i
                
                # Incremental update
                delta_ac = np.zeros(2*N)
                for k in range(1, N):
                    delta = 0.0
                    if i + k < N:
                        delta -= 2.0 * old_h_i * h_rep[i + k]
                    if i - k >= 0:
                        delta -= 2.0 * h_rep[i - k] * old_h_i
                    delta_ac[k] = delta
                    delta_ac[2*N - k] = delta
                
                ac_new = ac_rep + delta_ac
                new_val = _max_sidelobe(ac_new)
                
                # Accept/reject
                accepted = False
                if T_rep > 1e-9:
                    delta_val = new_val - val_rep
                    if delta_val < 0:
                        accepted = True
                    else:
                        prob = np.exp(-delta_val / T_rep)
                        if np.random.rand() < prob:
                            accepted = True
                
                if accepted:
                    replicas[r][1] = ac_new
                    replicas[r][2] = new_val
                else:
                    h_rep[i] = old_h_i
                
                # Update best
                if replicas[r][2] < best_val:
                    best_val = replicas[r][2]
                    best_h = replicas[r][0].copy()
            
            # Swap replicas with different temperatures (Metropolis criterion)
            for r in range(num_replicas - 1):
                if np.random.rand() < 0.1:  # 10% chance to attempt swap
                    # Compute energy difference
                    delta_E = replicas[r+1][2] - replicas[r][2]
                    # Accept swap with probability min(1, exp(-delta_E / (T_high - T_low)))
                    # But since we're minimizing, and T_high > T_low:
                    # We want to swap if it helps the lower temp replica
                    # Standard PT: accept swap with prob min(1, exp(-(E1 - E2)/(T1 - T2)))
                    # where T1 > T2, E1 is energy of replica 1 (higher T), E2 of replica 2
                    # Here r+1 has lower T than r (since temperatures decrease with index)
                    # Wait, temperatures = [0.01, 0.005, 0.0025, ...] so index 0 has highest T
                    # So r has higher T than r+1
                    # delta_E = E_{r+1} - E_r
                    # Accept with prob min(1, exp(-delta_E / (T_r - T_{r+1})))
                    T_diff = replicas[r][3] - replicas[r+1][3]
                    if T_diff > 1e-10:
                        if delta_E < 0:
                            accept_swap = True
                        else:
                            accept_swap = np.random.rand() < np.exp(-delta_E / T_diff)
                    else:
                        accept_swap = False
                    
                    if accept_swap:
                        # Swap h, ac, val, T
                        h_r, ac_r, val_r, T_r = replicas[r]
                        h_rp1, ac_rp1, val_rp1, T_rp1 = replicas[r+1]
                        replicas[r] = [h_rp1, ac_rp1, val_rp1, T_r]
                        replicas[r+1] = [h_r, ac_r, val_r, T_rp1]
                        
                        if val_rp1 < best_val:
                            best_val = val_rp1
                            best_h = h_rp1.copy()
                        if val_r < best_val:
                            best_val = val_r
                            best_h = h_r.copy()
            
            # Cool down
            for r in range(num_replicas):
                if replicas[r][3] > 1e-9:
                    replicas[r][3] *= 0.999
        
        return best_h, best_val
    
    # Standard SA / Hill climbing
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        i = np.random.randint(N)
        old_h_i = h[i]
        h[i] = -old_h_i
        
        delta_ac = np.zeros(2*N)
        for k in range(1, N):
            delta = 0.0
            if i + k < N:
                delta -= 2.0 * old_h_i * h[i + k]
            if i - k >= 0:
                delta -= 2.0 * h[i - k] * old_h_i
            delta_ac[k] = delta
            delta_ac[2*N - k] = delta
        
        ac_new = ac + delta_ac
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
                h[i] = old_h_i
        
        if T > 1e-9:
            T *= 0.999

    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Continuous Relaxation with Smooth Objective
    num_continuous_starts = 20
    betas = [15.0, 30.0, 50.0, 80.0, 120.0, 200.0]
    
    # Generate diverse starting points
    starts = []
    for i in range(num_continuous_starts):
        if i < 5:
            # Random ±1
            x0 = np.random.choice([-1.0, 1.0], size=N)
        elif i < 10:
            # Random continuous in [-1, 1]
            x0 = np.random.uniform(-1.0, 1.0, size=N)
        elif i < 15:
            # Structured: alternating blocks
            block_size = 4 + (i - 10) * 2
            x0 = np.zeros(N)
            for j in range(N):
                x0[j] = 1.0 if (j // block_size) % 2 == 0 else -1.0
        else:
            # Structured: sinusoidal
            freq = 2 + (i - 15) * 3
            x0 = np.sign(np.sin(2 * np.pi * freq * np.arange(N) / N))
            x0[x0 == 0] = 1.0
        starts.append(x0)
    
    for i, x0 in enumerate(starts):
        if time.time() >= deadline - 30.0:
            break
            
        beta = betas[i % len(betas)]
        
        def obj_grad(x, beta=beta):
            return _objective_and_gradient(x, beta)
        
        try:
            from scipy.optimize import minimize
            res = minimize(obj_grad, x0, method='L-BFGS-B', 
                           bounds=[(-1.0, 1.0)] * N, 
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-9, 'maxls': 40, 'maxfun': 1000})
        except Exception:
            continue
        
        h_bin = np.sign(res.x)
        zero_mask = (h_bin == 0)
        if np.any(zero_mask):
            h_bin[zero_mask] = np.random.choice([-1.0, 1.0], size=np.sum(zero_mask))
            
        ac = _compute_autocorr(h_bin)
        val = _max_sidelobe(ac)
        
        if val < best_val:
            best_val = val
            best_h = h_bin.copy()
            
        if best_h is not None and time.time() < deadline - 15.0:
            h_ref, val_ref = _local_search(best_h.copy(), max_iterations=3000, time_limit=3.0)
            if val_ref < best_val:
                best_val = val_ref
                best_h = h_ref
    
    # Phase 2: Final Discrete Search with Parallel Tempering
    remaining_time = deadline - time.time()
    if remaining_time > 8.0 and best_h is not None:
        # Use parallel tempering for better exploration
        h_final, val_final = _local_search(best_h.copy(), max_iterations=100000, time_limit=max(remaining_time - 5.0, 2.0), 
                                           use_sa=True, T0=0.05, parallel_tempering=True)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
    
    # Phase 3: High-precision polish with exact objective
    remaining_time = deadline - time.time()
    if remaining_time > 3.0 and best_h is not None:
        h_polished, val_polished = _polish_discrete(best_h.copy(), max_iterations=20000, time_limit=max(remaining_time - 1.0, 1.0))
        if val_polished < best_val:
            best_val = val_polished
            best_h = h_polished
    
    if best_h is None:
        return np.ones(N)
        
    return best_h
