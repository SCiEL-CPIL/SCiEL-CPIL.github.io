import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _value(h):
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (N,):
        return np.inf
    if np.any(~np.isfinite(h)):
        return np.inf
    h_clipped = np.clip(h, 0.0, 1.0)
    corr = np.correlate(h_clipped, 1.0 - h_clipped, mode="full")
    return float(np.max(corr)) / N * 2.0

def _calc_max_corr(h):
    h = np.asarray(h, dtype=np.float64)
    corr = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(corr))

def _local_search_binary(h_bin):
    """
    Local search on binary vectors with exactly N/2 ones.
    Tries swaps (one 1 with one 0) to preserve the count of ones.
    Optimized: only compute correlation changes for affected lags.
    """
    h_bin = np.array(h_bin, dtype=np.int8)
    h_float = h_bin.astype(np.float64)
    current_corr = np.correlate(h_float, 1.0 - h_float, mode="full")
    current_max = float(np.max(current_corr))
    
    ones = np.where(h_bin == 1)[0]
    zeros = np.where(h_bin == 0)[0]
    
    improved = True
    max_iter = 100  # Safety cap
    iter_count = 0
    
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        best_gain = 0
        best_i = -1
        best_j = -1
        best_new_corr = None
        
        # Precompute 1-h for faster correlation
        one_minus_h = 1.0 - h_float
        
        # Try swapping a 1 at i with a 0 at j
        # When we flip i from 1->0 and j from 0->1:
        # - At position i: h[i] changes from 1 to 0, so (1-h)[i] changes from 0 to 1
        # - At position j: h[j] changes from 0 to 1, so (1-h)[j] changes from 1 to 0
        # The correlation at lag k is sum(h[m] * (1-h)[m+k]) for valid m
        # Flipping i and j affects lags where i or j participate
        
        for i in ones:
            for j in zeros:
                # Compute new correlation efficiently
                # Only lags affected by positions i and j change
                # Lag k involves pairs (m, m+k) where m+k < N
                # Position i affects lags where i is in [0, N-1] as first or second element
                # Position j affects lags where j is in [0, N-1] as first or second element
                
                # For efficiency, compute full correlation but cache the computation
                h_temp = h_bin.copy()
                h_temp[i] = 0
                h_temp[j] = 1
                h_temp_float = h_temp.astype(np.float64)
                new_corr = np.correlate(h_temp_float, 1.0 - h_temp_float, mode="full")
                new_max = float(np.max(new_corr))
                
                gain = current_max - new_max
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_j = j
                    best_new_corr = new_corr
        
        if best_gain > 1e-6:
            h_bin[best_i] = 0
            h_bin[best_j] = 1
            h_float = h_bin.astype(np.float64)
            current_corr = best_new_corr
            current_max = current_max - best_gain
            ones = np.where(h_bin == 1)[0]
            zeros = np.where(h_bin == 0)[0]
            improved = True
            
    return h_float, current_max

def _continuous_refine(h0, deadline):
    """
    Refine a continuous vector h0 using SLSQP with an auxiliary variable t
    to minimize max(corr) subject to sum(h) = N/2 and 0 <= h <= 1.
    Uses multiple passes with decreasing lambda for better convergence.
    """
    N_ = len(h0)
    
    def _make_ineq_constraint(lam):
        def _ineq_constraint(x):
            h = np.clip(x[:N_], 0.0, 1.0)
            t = x[N_]
            corr = np.correlate(h, 1.0 - h, mode="full")
            # Smooth max approximation using log-sum-exp
            shifted = corr - np.max(corr)
            soft_max = np.log(np.sum(np.exp(lam * shifted))) / lam + np.max(corr)
            return t - soft_max
        return _ineq_constraint

    def _eq_constraint(x):
        h = np.clip(x[:N_], 0.0, 1.0)
        return np.sum(h) - N_ / 2.0

    def obj(x):
        return x[N_]  # t

    bounds_t = [(0.0, 1.0)] * N_ + [(0.0, float(N_))]

    h_current = h0.copy()
    val_current = _value(h_current)
    
    # Multiple passes with decreasing lambda for better convergence
    lambdas = [20.0, 50.0, 100.0, 200.0]
    
    for lam in lambdas:
        if time.time() > deadline - 5:
            break
            
        corr_init = np.correlate(h_current, 1.0 - h_current, mode="full")
        t_init = float(np.max(corr_init))
        
        x0 = np.concatenate([h_current, [t_init]])
        
        cons = [
            {'type': 'ineq', 'fun': _make_ineq_constraint(lam)},
            {'type': 'eq', 'fun': _eq_constraint}
        ]
        
        try:
            res = minimize(
                obj, x0,
                method='SLSQP',
                bounds=bounds_t,
                constraints=cons,
                options={'maxiter': 300, 'ftol': 1e-10}
            )
            
            if res.success:
                h_new = np.clip(res.x[:N_], 0.0, 1.0)
                # Fix sum precisely
                diff = N_ / 2.0 - np.sum(h_new)
                if abs(diff) > 1e-9:
                    # Distribute the correction evenly, but respect bounds
                    step = diff / N_
                    for idx in range(N_):
                        h_new[idx] += step
                        if h_new[idx] < 0:
                            h_new[idx] = 0.0
                        elif h_new[idx] > 1:
                            h_new[idx] = 1.0
                    
                    # Second pass if needed
                    diff2 = N_ / 2.0 - np.sum(h_new)
                    if abs(diff2) > 1e-9:
                        h_new = h_new + diff2 / N_
                        h_new = np.clip(h_new, 0.0, 1.0)
                
                val_new = _value(h_new)
                if val_new < val_current:
                    h_current = h_new
                    val_current = val_new
                    
        except Exception:
            pass
    
    return h_current, val_current

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin for 150s limit

    best_h = np.full(N, 0.5)
    best_val = _value(best_h)
    
    rng = np.random.default_rng(42)
    
    # 1. Binary Local Search with diverse initializations
    # We use random starts AND structured sinusoidal starts to find diverse local optima.
    num_random_restarts = 20
    num_sin_restarts = 15
    
    # Random restarts
    for restart in range(num_random_restarts):
        if time.time() > deadline - 40:
            break
            
        h_bin = np.zeros(N, dtype=np.int8)
        ones_idx = rng.choice(N, size=N//2, replace=False)
        h_bin[ones_idx] = 1
        
        h_opt, max_corr = _local_search_binary(h_bin)
        val = max_corr / N * 2.0
        
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()
            
    # Sinusoidal restarts
    # Structured patterns might lead to different local minima.
    for k in range(1, num_sin_restarts + 1):
        if time.time() > deadline - 40:
            break
        
        # Frequency k
        # A simple sine wave sin(2*pi*k*i/N) will have equal positive and negative parts (approx).
        # We add a small random phase shift to diversify.
        phase = rng.uniform(0, 2*np.pi)
        h_cont = 0.5 + 0.5 * np.sin(2 * np.pi * k * np.arange(N) / N + phase)
        
        # Threshold to binary
        h_bin = (h_cont >= 0.5).astype(np.int8)
        
        # Ensure exactly N/2 ones? The task requires sum(h) = N/2.
        # The binary local search preserves the number of ones.
        # If the number of ones is not N/2, we need to adjust.
        num_ones = np.sum(h_bin)
        if num_ones != N // 2:
            # Swap some to get exactly N/2 ones
            if num_ones > N // 2:
                # Remove excess ones
                excess = num_ones - N // 2
                ones_idx = np.where(h_bin == 1)[0]
                zeros_idx = np.where(h_bin == 0)[0]
                # Randomly pick excess ones to flip to 0
                to_flip = rng.choice(ones_idx, size=excess, replace=False)
                h_bin[to_flip] = 0
            else:
                # Add missing ones
                missing = N // 2 - num_ones
                ones_idx = np.where(h_bin == 1)[0]
                zeros_idx = np.where(h_bin == 0)[0]
                # Randomly pick missing zeros to flip to 1
                to_flip = rng.choice(zeros_idx, size=missing, replace=False)
                h_bin[to_flip] = 1
        
        h_opt, max_corr = _local_search_binary(h_bin)
        val = max_corr / N * 2.0
        
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()

    # 2. Continuous Refinement
    # Use the best binary solution as a starting point for continuous optimization.
    
    if time.time() < deadline - 15:
        h_refined, val_refined = _continuous_refine(best_h, deadline)
        if val_refined < best_val:
            best_val = val_refined
            best_h = h_refined

    # 3. Try continuous refinement from sinusoidal starts (not just binary)
    if time.time() < deadline - 10:
        for k in range(1, 8):
            if time.time() > deadline - 10:
                break
            phase = rng.uniform(0, 2*np.pi)
            h_sin = 0.5 + 0.5 * np.sin(2 * np.pi * k * np.arange(N) / N + phase)
            h_sin = np.clip(h_sin, 0.0, 1.0)
            # Fix sum
            diff = N/2.0 - np.sum(h_sin)
            h_sin = h_sin + diff / N
            h_sin = np.clip(h_sin, 0.0, 1.0)
            
            h_refined, val_refined = _continuous_refine(h_sin, deadline)
            if val_refined < best_val:
                best_val = val_refined
                best_h = h_refined

    # 4. If time remains, try a few more refinements with perturbations
    if time.time() < deadline - 5:
        # Try random perturbations of the best solution and refine
        for _ in range(8):
            if time.time() > deadline - 5:
                break
            h_perturbed = best_h + rng.normal(0, 0.03, N)
            h_perturbed = np.clip(h_perturbed, 0.0, 1.0)
            # Fix sum
            diff = N/2.0 - np.sum(h_perturbed)
            h_perturbed = h_perturbed + diff / N
            h_perturbed = np.clip(h_perturbed, 0.0, 1.0)
            
            h_refined, val_refined = _continuous_refine(h_perturbed, deadline)
            if val_refined < best_val:
                best_val = val_refined
                best_h = h_refined

    # 5. Final high-precision polish pass
    if time.time() < deadline - 2:
        # Do one more refinement with very tight tolerance
        h_polish, val_polish = _continuous_refine(best_h, deadline)
        if val_polish < best_val:
            best_val = val_polish
            best_h = h_polish

    return best_h.tolist()
