import numpy as np
import time
from scipy.optimize import minimize

N = 50
_DEADLINE = 115.0  # Safety margin

def _compute_ratio(h):
    """
    Computes the ratio: ||h||_2^2 / (||h||_1 * ||h * h||_infty)
    Assumes h is non-negative.
    """
    h = np.asarray(h, dtype=float)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    l_inf = np.max(conv)
    if l_inf == 0:
        return 0.0
    return l2_sq / (l1 * l_inf)

def _objective_complex(g_flat):
    """
    Objective to minimize: negative of the ratio.
    g_flat is a real vector of length 2*N representing complex g.
    h = |g|^2.
    We normalize h to L1=1 inside the objective to handle scale invariance
    and focus on shape optimization.
    """
    g = g_flat.reshape(N, 2)
    h = g[:, 0]**2 + g[:, 1]**2
    
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    h_norm = h / l1
    
    l2_sq = np.sum(h_norm**2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    l_inf = np.max(conv)
    
    if l_inf == 0:
        return 0.0
    
    # The ratio for normalized h is l2_sq / l_inf
    # We want to maximize this, so minimize negative
    return -l2_sq / l_inf

def _find_sidon_set(n, target_size=10):
    """
    Greedily finds a Sidon set of size target_size in [0, n).
    """
    S = [0]
    used_sums = set([0])
    while len(S) < target_size:
        found = False
        for i in range(1, n):
            if i in S:
                continue
            valid = True
            new_sums = set()
            for j in S:
                s = i + j
                if s in used_sums:
                    valid = False
                    break
                new_sums.add(s)
            if valid:
                S.append(i)
                used_sums.update(new_sums)
                found = True
                break
        if not found:
            break
    return S

def _optimize_shape(h_init, time_limit):
    """
    Optimizes the shape starting from h_init using complex parameterization.
    """
    # Convert h_init to g_init
    # h_init = |g_init|^2 => g_init = sqrt(h_init) (real part), 0 (imag part)
    # To avoid zero gradients at 0, we add a small epsilon or use random phase
    g_real = np.sqrt(h_init + 1e-10)
    g_imag = np.zeros(N)
    
    # Add a bit of random phase to escape symmetry if needed?
    # For now, stick to real initialization.
    g_flat = np.concatenate([g_real, g_imag])
    
    start_time = time.time()
    
    # Use L-BFGS-B
    # Note: maxiter and maxfun are limited to respect time_limit
    try:
        result = minimize(
            _objective_complex,
            g_flat,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 1000}
        )
        
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            # Timeout, return initial or last best? 
            # scipy minimize usually returns the best found so far or the last point.
            # We'll just use the result.x
            pass
            
        g_opt = result.x.reshape(N, 2)
        h_opt = g_opt[:, 0]**2 + g_opt[:, 1]**2
        
        # Normalize to L1=1
        l1 = np.sum(h_opt)
        if l1 > 0:
            h_opt = h_opt / l1
        else:
            h_opt = h_init / np.sum(h_init)
            
        return h_opt
        
    except Exception:
        # Fallback to initial
        l1 = np.sum(h_init)
        if l1 > 0:
            return h_init / l1
        return np.ones(N) / N

def construct():
    start_time = time.time()
    
    best_h = np.ones(N) / N
    best_score = _compute_ratio(best_h)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Uniform
    initializations.append(np.ones(N))
    
    # 2. Delta (single spike)
    h_delta = np.zeros(N)
    h_delta[N//2] = 1.0
    initializations.append(h_delta)
    
    # 3. Sidon sets
    # Try a few different Sidon sets of different sizes
    for size in [6, 8, 10]:
        S = _find_sidon_set(N, target_size=size)
        h_sidon = np.zeros(N)
        for idx in S:
            h_sidon[idx] = 1.0
        initializations.append(h_sidon)
        
    # 4. Smooth windows
    x = np.linspace(-1, 1, N)
    initializations.append(np.exp(-x**2 / 0.2))  # Gaussian
    initializations.append(0.5 + 0.5 * np.cos(np.pi * x))  # Cosine
    initializations.append(1 - np.abs(x))  # Triangle
    
    # 5. Random sparse (to break symmetry)
    rng = np.random.RandomState(42)
    for _ in range(3):
        h_rand = np.zeros(N)
        indices = rng.choice(N, size=8, replace=False)
        h_rand[indices] = 1.0
        initializations.append(h_rand)
        
    # 6. Alternating
    h_alt = np.ones(N)
    h_alt[::2] = 0.0
    initializations.append(h_alt)
    
    # Optimize each initialization
    # Estimate time per optimization
    # We have ~115s. Let's allow ~10s per run.
    # 12 initializations * 10s = 120s. Too many.
    # Let's prioritize the most promising ones.
    
    # Priority: Sidon sets (sparse, high potential), Smooth (stable), Uniform (baseline)
    # We'll run all, but with a time check.
    
    time_per_opt = 8.0
    
    for i, h0 in enumerate(initializations):
        elapsed = time.time() - start_time
        if elapsed + time_per_opt > _DEADLINE:
            break
            
        h_opt = _optimize_shape(h0, time_per_opt)
        
        # Compute score on the normalized result
        score = _compute_ratio(h_opt)
        
        if score > best_score:
            best_score = score
            best_h = h_opt
            
    # Final polish: run a few perturbations on the best solution found
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 10:
        for pert_seed in [100, 200, 300]:
            elapsed = time.time() - start_time
            if elapsed + 5 > _DEADLINE:
                break
            rng_local = np.random.RandomState(pert_seed)
            # Perturb in g-space
            g_best_real = np.sqrt(best_h + 1e-10)
            g_best_imag = np.zeros(N)
            
            pert_real = g_best_real + rng_local.randn(N) * 0.1
            pert_imag = rng_local.randn(N) * 0.1
            
            h_pert = pert_real**2 + pert_imag**2
            h_pert = h_pert / np.sum(h_pert)
            
            h_opt = _optimize_shape(h_pert, 5.0)
            score = _compute_ratio(h_opt)
            if score > best_score:
                best_score = score
                best_h = h_opt
                
    return list(best_h)
