import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 118.0  # Safety margin before 120s limit

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

def _soft_max_sidelobe(ac, beta=10.0):
    """
    Smooth approximation to max |sidelobe| using log-sum-exp.
    """
    sidelobes = np.abs(ac[1:N])
    max_val = np.max(sidelobes)
    if max_val < 1e-10:
        return max_val
    # log-sum-exp trick for numerical stability
    shifted = beta * (sidelobes - max_val)
    log_sum = np.log(np.sum(np.exp(shifted))) / beta
    return max_val + log_sum

def _objective_and_gradient(x, beta=10.0):
    """
    Objective: Soft-max of sidelobes (smooth surrogate for max).
    Gradient: Analytic via chain rule using FFT-based convolution.
    """
    h = x
    ac = _compute_autocorr(h)
    sidelobes_abs = np.abs(ac[1:N])
    
    # Soft-max objective
    max_val = np.max(sidelobes_abs)
    if max_val < 1e-10:
        return max_val, np.zeros(N)
    
    shifted = beta * (sidelobes_abs - max_val)
    exp_shifted = np.exp(shifted)
    log_sum = np.log(np.sum(exp_shifted))
    obj = max_val + log_sum / beta
    
    # Weights for each sidelobe in the soft-max
    weights = exp_shifted / np.sum(exp_shifted)
    signs = np.sign(ac[1:N])
    
    # Gradient: d/dh_j sum_k w_k * sign(ac_k) * ac_k
    # ac_k = sum_i h_i * h_{i+k} for k>0
    # d(ac_k)/d(h_j) = h_{j-k} + h_{j+k} (with boundary handling)
    # Use FFT to compute the gradient efficiently
    # grad_j = sum_k w_k * sign(ac_k) * (h_{j-k} + h_{j+k})
    
    # Create a kernel from weighted signs
    kernel = np.zeros(2*N)
    kernel[1:N] = weights * signs
    kernel[2*N-1 : 2*N-1+N-1] = weights * signs  # symmetric part for negative lags
    
    # convolve h with kernel to get gradient
    n_fft = 4 * N
    F_h = np.fft.rfft(h, n=n_fft)
    F_kernel = np.fft.rfft(kernel, n=n_fft)
    grad = np.fft.irfft(F_h * F_kernel, n=n_fft)[:N]
    
    return obj, grad

def _local_search(h, max_iterations=10000, time_limit=5.0):
    """
    Fast local search using incremental autocorrelation updates with simulated annealing.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    
    # Simulated annealing parameters
    T0 = current_val * 0.1 if current_val > 1e-10 else 0.01
    cooling_rate = 0.999
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Try flipping a random bit
        i = np.random.randint(N)
        old_h_i = h[i]
        h[i] = -old_h_i
        
        # Incrementally update autocorrelation
        # ac[k] = sum_j h[j] * h[j+k]
        # When h[i] flips from old to new (old + 2*old = 2*old since new = -old)
        # delta_ac[k] = 2*old_h_i * (h[i-k] + h[i+k]) for valid indices
        delta_ac = np.zeros(2*N)
        
        # For k > 0: ac[k] changes by 2*old_h_i * (h[i-k] if i-k>=0 else 0 + h[i+k] if i+k<N else 0)
        for k in range(1, N):
            delta = 0.0
            if i - k >= 0:
                delta += h[i-k]
            if i + k < N:
                delta += h[i+k]
            delta_ac[k] = 2.0 * old_h_i * delta
            delta_ac[2*N - k] = delta_ac[k]  # symmetric
        
        ac_new = ac + delta_ac
        new_val = _max_sidelobe(ac_new)
        
        delta_obj = new_val - current_val
        if delta_obj < 0:
            # Accept improvement
            current_val = new_val
            ac = ac_new
        else:
            # Simulated annealing: accept with probability exp(-delta/T)
            T = T0 * (cooling_rate ** _)
            if T > 1e-15 and np.random.random() < np.exp(-delta_obj / T):
                current_val = new_val
                ac = ac_new
            else:
                # Revert
                h[i] = old_h_i
                # ac remains unchanged (old ac)
                
    return h, current_val

def _parallel_tempering(h_starts, num_reps=5, time_limit=10.0):
    """
    Parallel tempering: run multiple chains at different temperatures in parallel.
    h_starts: list of initial configurations
    Returns the best configuration found.
    """
    start_time = time.time()
    best_h = None
    best_val = float('inf')
    
    # Initialize chains with different temperatures
    T_base = 0.01
    T_factors = [1.0, 2.0, 5.0, 10.0, 20.0]  # Different temperatures
    
    # Create initial configurations
    chains = []
    for i, h0 in enumerate(h_starts):
        T = T_base * T_factors[i % len(T_factors)]
        chains.append({
            'h': h0.copy(),
            'T': T,
            'val': _max_sidelobe(_compute_autocorr(h0))
        })
    
    cooling_rate = 0.9995
    swap_interval = 100
    
    iteration = 0
    while time.time() - start_time < time_limit:
        # Check deadline
        if time.time() - start_time > time_limit - 1.0:
            break
        
        # Perform moves for each chain
        for chain in chains:
            h = chain['h']
            T = chain['T']
            
            # Try flipping a random bit
            i = np.random.randint(N)
            old_h_i = h[i]
            h[i] = -old_h_i
            
            new_val = _max_sidelobe(_compute_autocorr(h))
            delta = new_val - chain['val']
            
            if delta < 0:
                chain['val'] = new_val
            else:
                if T > 1e-15 and np.random.random() < np.exp(-delta / T):
                    chain['val'] = new_val
                else:
                    h[i] = old_h_i
        
        # Swap configurations between adjacent temperature levels
        if iteration % swap_interval == 0:
            for i in range(len(chains) - 1):
                # Try swapping h between chain i and i+1
                h1, h2 = chains[i]['h'], chains[i+1]['h']
                v1, v2 = chains[i]['val'], chains[i+1]['val']
                T1, T2 = chains[i]['T'], chains[i+1]['T']
                
                # Metropolis criterion for swap
                delta = T1 * v2 + T2 * v1 - T1 * v1 - T2 * v2
                if delta < 0:
                    # Swap
                    chains[i]['h'], chains[i+1]['h'] = h2, h1
                    chains[i]['val'], chains[i+1]['val'] = v2, v1
                else:
                    if np.random.random() < np.exp(-delta / max(T1, T2)):
                        chains[i]['h'], chains[i+1]['h'] = h2, h1
                        chains[i]['val'], chains[i+1]['val'] = v2, v1
        
        # Update best
        for chain in chains:
            if chain['val'] < best_val:
                best_val = chain['val']
                best_h = chain['h'].copy()
        
        iteration += 1
        
        # Cool down all chains
        for chain in chains:
            chain['T'] *= cooling_rate
    
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Generate diverse initializations
    initializations = []
    
    # All ones (seed)
    initializations.append(np.ones(N))
    
    # Random binary
    for _ in range(5):
        initializations.append(np.random.choice([-1.0, 1.0], size=N))
    
    # Structured patterns
    # Alternating
    initializations.append(np.tile([-1.0, 1.0], N // 2))
    
    # Block patterns
    block_size = N // 8
    for offset in range(0, block_size, block_size // 4 if block_size >= 4 else 1):
        pattern = np.zeros(N)
        for i in range(N):
            block_idx = i // block_size
            pattern[i] = 1.0 if block_idx % 2 == 0 else -1.0
        if offset > 0:
            pattern = np.roll(pattern, offset)
        initializations.append(pattern)
    
    # Sine-based patterns
    for freq in [1, 2, 4]:
        x = np.linspace(0, 2 * np.pi * freq, N)
        initializations.append(np.sign(np.sin(x)))
    
    # Phase 2: Continuous relaxation with soft-max objective
    num_continuous_starts = min(8, len(initializations))
    
    for i in range(num_continuous_starts):
        if time.time() >= deadline - 30.0:
            break
            
        # Initialize x near the binary pattern
        x0 = initializations[i].copy() * 0.85
        
        # Use L-BFGS-B with soft-max objective, adaptive beta
        try:
            # Start with moderate beta, can increase for sharper approximation
            beta_start = 5.0
            res = minimize(_objective_and_gradient, x0, method='L-BFGS-B', 
                           bounds=[(-1.0, 1.0)] * N, 
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8, 'maxls': 30},
                           args=(beta_start,))
        except Exception:
            continue
        
        # If we have time, do a second pass with higher beta for sharper max approximation
        if time.time() < deadline - 25.0:
            try:
                res2 = minimize(_objective_and_gradient, res.x, method='L-BFGS-B', 
                               bounds=[(-1.0, 1.0)] * N, 
                               options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10, 'maxls': 20},
                               args=(20.0,))
                res = res2
            except Exception:
                pass
        
        # Round to binary
        h_bin = np.sign(res.x)
        # Handle zeros
        zero_mask = (h_bin == 0)
        if np.any(zero_mask):
            h_bin[zero_mask] = np.random.choice([-1.0, 1.0], size=np.sum(zero_mask))
            
        # Evaluate binary solution
        ac = _compute_autocorr(h_bin)
        val = _max_sidelobe(ac)
        
        if val < best_val:
            best_val = val
            best_h = h_bin.copy()
            
        # Refine the best binary solution found so far using local search
        if best_h is not None:
            if time.time() < deadline - 20.0:
                h_ref, val_ref = _local_search(best_h.copy(), max_iterations=3000, time_limit=3.0)
                if val_ref < best_val:
                    best_val = val_ref
                    best_h = h_ref
    
    # Phase 3: Parallel tempering for global exploration
    if time.time() < deadline - 15.0 and best_h is not None:
        # Use best_h and some other initializations for parallel tempering
        pt_starts = [best_h.copy()]
        for h in initializations[1:4]:
            pt_starts.append(h.copy())
        
        remaining_time = deadline - time.time() - 5.0
        h_pt, val_pt = _parallel_tempering(pt_starts, time_limit=remaining_time)
        
        if h_pt is not None and val_pt < best_val:
            best_val = val_pt
            best_h = h_pt
    
    # Phase 4: Final discrete search with simulated annealing
    remaining_time = deadline - time.time()
    if remaining_time > 5.0 and best_h is not None:
        # Use the best_h as starting point for a longer local search
        h_final, val_final = _local_search(best_h.copy(), max_iterations=200000, time_limit=remaining_time - 2.0)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
            
    if best_h is None:
        return np.ones(N)
        
    return best_h
