import numpy as np
import time
import sys

N = 50

def _compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * sup|h*h|)"""
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 0.0
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _log_ratio(h):
    """Compute log(ratio) for stability. Returns -inf if invalid."""
    h = np.asarray(h, dtype=float)
    # Enforce non-negativity for the objective
    h = np.maximum(h, 0.0)
    
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return -np.inf
        
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return -np.inf
        
    return np.log(l2_sq) - np.log(l1) - np.log(sup_conv)

def _nelder_mead(objective, x0, max_iter=200, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5, tol=1e-12):
    """
    Standard Nelder-Mead simplex optimization.
    objective: function to minimize.
    x0: initial point.
    """
    n = len(x0)
    
    # Initialize simplex
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    
    for i in range(n):
        simplex[i + 1] = x0
        if x0[i] != 0:
            simplex[i + 1][i] = x0[i] * 1.05 + 0.001
        else:
            simplex[i + 1][i] = 0.01
            
    # Evaluate objective
    f = np.array([objective(v) for v in simplex])
    
    # Sort simplex by objective value
    idx = np.argsort(f)
    simplex = simplex[idx]
    f = f[idx]
    
    c_alpha = alpha
    c_gamma = gamma
    c_rho = rho
    c_sigma = sigma
    
    for i in range(max_iter):
        # Check convergence
        if np.max(f) - np.min(f) < tol:
            break
            
        # Centroid of all points except the worst
        x_centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_reflected = x_centroid + c_alpha * (x_centroid - simplex[-1])
        f_reflected = objective(x_reflected)
        
        if f_reflected < f[0]:
            # Expansion
            x_expanded = x_centroid + c_gamma * (x_reflected - x_centroid)
            f_expanded = objective(x_expanded)
            
            if f_expanded < f_reflected:
                simplex[-1] = x_expanded
                f[-1] = f_expanded
            else:
                simplex[-1] = x_reflected
                f[-1] = f_reflected
        elif f_reflected < f[-2]:
            # Simple reflection
            simplex[-1] = x_reflected
            f[-1] = f_reflected
        else:
            # Contraction
            x_contracted = x_centroid + c_rho * (simplex[-1] - x_centroid)
            f_contracted = objective(x_contracted)
            
            if f_contracted < f[-1]:
                simplex[-1] = x_contracted
                f[-1] = f_contracted
            else:
                # Shrink
                for j in range(1, n + 1):
                    simplex[j] = simplex[0] + c_sigma * (simplex[j] - simplex[0])
                    f[j] = objective(simplex[j])
        
        # Re-sort
        idx = np.argsort(f)
        simplex = simplex[idx]
        f = f[idx]
        
    # Return best point
    return simplex[0], f[0]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # 1. Generate candidate supports (Sidon-like sets)
    # A Sidon set minimizes the number of repeated sums, which helps keep convolution peaks low.
    # We will test a few different Sidon sets and random sparse sets.
    
    supports = []
    
    # Known small Sidon sets in [0, N)
    # 1. Quadratic residues / simple patterns
    supports.append(list(range(0, 10))) # Dense start
    supports.append([0, 1, 3, 7, 12, 18, 25, 33, 42]) # Sparse Sidon-ish
    
    # Generate random Sidon-like sets
    np.random.seed(42)
    for k in range(20):
        S = set()
        while len(S) < 12:
            x = np.random.randint(0, N)
            # Check Sidon property (sums are unique)
            is_sidon = True
            for y in S:
                if x + y in S: # This is a simplified check, true Sidon requires pair sums
                    pass
            # For simplicity, just pick random distinct indices
            if x not in S:
                S.add(x)
        supports.append(list(S))
        
    # 2. For each support, run Nelder-Mead on the amplitudes
    
    for S in supports:
        if time.time() > deadline - 10.0:
            break
            
        k = len(S)
        
        # Initial guess: uniform amplitudes
        h0 = np.zeros(N)
        for idx in S:
            h0[idx] = 1.0 / k
            
        # Objective: minimize negative log ratio
        def obj(h):
            return -_log_ratio(h)
            
        # Run Nelder-Mead
        # Limit iterations to ensure we finish in time
        # 50 dimensions is high for NM, but we are only optimizing non-zero coords?
        # No, NM operates on the full vector. To be efficient, we should project.
        # However, NM on full 50D is slow.
        # Strategy: Optimize only the amplitudes in S, keeping others 0.
        # We can create a wrapper that maps amplitudes -> h.
        
        def obj_amplitudes(ampl):
            h = np.zeros(N)
            for j, idx in enumerate(S):
                h[idx] = max(0.0, ampl[j])
            return -_log_ratio(h)
            
        x0_ampl = np.ones(k) / k
        
        # Run NM on the reduced space
        try:
            x_opt_ampl, f_opt = _nelder_mead(obj_amplitudes, x0_ampl, max_iter=500)
            h_opt = np.zeros(N)
            for j, idx in enumerate(S):
                h_opt[idx] = max(0.0, x_opt_ampl[j])
            
            score = _compute_ratio(h_opt)
            
            if score > best_score:
                best_score = score
                best_h = h_opt.copy()
                
        except Exception as e:
            pass
            
    # 3. Refinement phase: Use the best support found so far, but allow NM to move support slightly?
    # NM on full 50D is expensive. Instead, do a few rounds of "Support Swap" + NM.
    
    # If we have time, try to refine the best solution by allowing small changes to all coordinates
    # using a coarser NM or just a few random perturbations.
    
    if time.time() < deadline - 20.0 and best_score > 0:
        # Try a few random perturbations of the best solution
        for _ in range(10):
            h_perturbed = best_h + np.random.normal(0, 0.05, N)
            h_perturbed = np.maximum(h_perturbed, 0)
            
            # Quick NM on full vector? Too slow.
            # Instead, just evaluate.
            score = _compute_ratio(h_perturbed)
            if score > best_score:
                best_score = score
                best_h = h_perturbed.copy()

    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(_compute_ratio(h))
