"""Program for signed step heights minimizing max|autoconv|/(sum h)^2.

Model-and-optimize approach: parameterize h via a compact set of basis functions
and directly optimize the parameters against the true objective (max absolute
autoconvolution / (sum h)^2) using derivative-free optimization.

The key insight is that the objective is scale-invariant (scaling h by c scales
both numerator and denominator by c^2), so we can fix sum(h)=1 without loss of
generality. This reduces the problem to optimizing N-1 free parameters.

We use a low-dimensional parameterization: h = sum_i a_i * phi_i where phi_i are
smooth basis functions (Gaussian bumps, sines, cosines, etc.). This keeps the
search space small while allowing sufficient flexibility.
"""

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, basinhopping
from scipy.signal import fftconvolve

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac[:size]))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _soft_max_abs(ac, beta=50.0):
    """
    Soft approximation of max|ac| using log-sum-exp for differentiability.
    Returns a smooth surrogate that approximates max|ac|.
    """
    abs_ac = np.abs(ac)
    # Use log-sum-exp for numerical stability
    max_val = np.max(abs_ac)
    if max_val < 1e-12:
        return 0.0
    shifted = abs_ac - max_val
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return max_val + log_sum

def _score_smooth(h, beta=50.0):
    """
    Smooth surrogate for the objective: soft_max|autoconv| / (sum h)^2.
    This is differentiable and can be used with gradient-based optimizers.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    
    soft_max = _soft_max_abs(ac, beta)
    return soft_max / (s * s)

def _objective_from_params(params, basis, t0, deadline):
    """Objective function for optimizer: returns score or inf if deadline passed."""
    if time.time() > deadline:
        return 1e10
    h = np.dot(params, basis)
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _smooth_objective_from_params(params, basis, t0, deadline, beta=50.0):
    """Smooth objective function for gradient-based optimizer."""
    if time.time() > deadline:
        return 1e10
    h = np.dot(params, basis)
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    
    soft_max = _soft_max_abs(ac, beta)
    return soft_max / (s * s)

def _make_basis_functions(n, n_basis=25):
    """
    Create a set of basis functions for parameterizing h.
    Returns a 2D array of shape (n_basis, N).
    """
    x = np.arange(n, dtype=float)
    basis = []
    
    # Constant function (needed for sum normalization)
    basis.append(np.ones(n))
    
    # Low-frequency cosines
    for k in range(1, 8):
        basis.append(np.cos(2 * np.pi * k * x / n))
    
    # Low-frequency sines
    for k in range(1, 8):
        basis.append(np.sin(2 * np.pi * k * x / n))
    
    # Gaussian bumps at different positions
    for center_frac in [0.2, 0.4, 0.6, 0.8]:
        mu = center_frac * n
        sigma = n / 8
        basis.append(np.exp(-(x - mu)**2 / (2 * sigma**2)))
    
    # Cosine window
    basis.append(np.cos(np.pi * x / n))
    
    # Quadratic polynomial
    basis.append((x - n/2)**2 / (n**2))
    
    # Linear ramp
    basis.append(x / n - 0.5)
    
    # Higher frequency cosine
    for k in [10, 15, 20]:
        basis.append(np.cos(2 * np.pi * k * x / n))
    
    return np.array(basis[:n_basis])

def _local_search_perturb(h, max_passes=5, deadline=None, step_size=0.05):
    """
    Local optimization: try small perturbations to each element, keep if improves score.
    Uses coordinate descent style updates.
    """
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(777)
    
    # Precompute FFT size
    size = 2 * N - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Random order for better exploration
        idx_order = rng.permutation(N)
        for i in idx_order:
            if deadline is not None and time.time() > deadline:
                break
            old_val = h[i]
            # Try several perturbation sizes with random direction
            best_local = old_val
            best_local_val = best_val
            for delta in [step_size, -step_size, 2*step_size, -2*step_size, 
                         0.5*step_size, -0.5*step_size]:
                h[i] = old_val + delta
                new_val = _score(h)
                if new_val < best_local_val:
                    best_local_val = new_val
                    best_local = h[i]
            if best_local != old_val:
                h[i] = best_local
                best_val = best_local_val
                improved = True
            else:
                h[i] = old_val  # revert if no improvement
        if not improved:
            break
    return h, best_val

def _local_search_flip(h, max_passes=3, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    """
    h = h.copy()
    best_val = _score(h)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]  # revert
        if not improved:
            break
    return h, best_val

def _gaussian_perturbation(h, scale=0.1, deadline=None, max_iters=200):
    """
    Simulated annealing style search with Gaussian perturbations.
    """
    h_curr = h.copy()
    h_best = h.copy()
    curr_val = _score(h_curr)
    best_val = curr_val
    rng = np.random.default_rng(999)
    
    T_start = 0.5
    T_end = 0.01
    n_iter = max_iters
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        T = T_start * (T_end / T_start) ** (it / max(1, n_iter - 1))
        
        # Generate perturbation
        perturb = rng.standard_normal(N) * scale
        h_trial = h_curr + perturb
        
        trial_val = _score(h_trial)
        
        # Metropolis acceptance
        if trial_val < curr_val or rng.random() < np.exp(-(trial_val - curr_val) / max(T, 1e-10)):
            h_curr = h_trial
            curr_val = trial_val
            if curr_val < best_val:
                best_val = curr_val
                h_best = h_curr.copy()
        
        # Gradually reduce perturbation scale
        scale *= 0.995
    
    return h_best, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s limit

    best_score = np.inf
    best_h = np.ones(N) / N  # Baseline: constant sequence
    
    # Evaluate baseline
    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()
    
    # Create basis functions for parameterization
    n_basis = 25
    basis = _make_basis_functions(N, n_basis)
    
    # Phase 1: Global optimization with differential evolution on compact parameterization
    if time.time() < deadline - 50:
        # Use differential evolution for global search
        # Parameters are the coefficients in the basis expansion
        # Normalize so that sum(h) = 1 by fixing first coefficient
        
        bounds = [(-2.0, 2.0)] * n_basis
        
        def de_objective(params):
            if time.time() > deadline - 45:
                return 1e10
            h = np.dot(params, basis)
            s = np.sum(h)
            if abs(s) < 1e-9:
                return 1e10
            return _autoconv_max_abs(h) / (s * s)
        
        try:
            res_de = differential_evolution(
                de_objective, bounds, 
                seed=42, maxiter=40, tol=1e-5,
                popsize=15, mutation=(0.3, 1.2), recombination=0.7,
                polish=False  # We'll do our own polishing
            )
            if res_de.fun < best_score:
                h_de = np.dot(res_de.x, basis)
                # Normalize
                s_de = np.sum(h_de)
                if abs(s_de) > 1e-6:
                    h_de = h_de / s_de
                val_de = _score(h_de)
                if val_de < best_score:
                    best_score = val_de
                    best_h = h_de.copy()
        except Exception:
            pass
    
    # Phase 2: L-BFGS-B refinement with smooth surrogate and multiple initializations
    if time.time() < deadline - 30:
        # Generate diverse initializations
        inits = []
        
        # From best so far (parameterize in basis)
        h_curr = best_h.copy()
        s_curr = np.sum(h_curr)
        if abs(s_curr) > 1e-6:
            h_curr = h_curr / s_curr
        
        # Project onto basis to get initial parameters
        if len(basis.shape) == 2:
            # Least squares projection
            params_init, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)
            inits.append(params_init)
        
        # Random initializations
        rng = np.random.default_rng(123)
        for i in range(10):
            params_rand = rng.standard_normal(n_basis) * 0.5
            inits.append(params_rand)
        
        # Structured initializations
        # Emphasize constant term (good starting point)
        for amp in [0.1, 0.3, 0.5, 0.7, 1.0]:
            params_struct = np.zeros(n_basis)
            params_struct[0] = 1.0  # Constant term
            params_struct[1] = amp * 0.1  # Small first cosine
            inits.append(params_struct)
        
        # Try smooth surrogate optimization first
        for params_init in inits[:6]:
            if time.time() > deadline - 25:
                break
            
            try:
                # First optimize with smooth surrogate using BFGS
                def smooth_obj(params):
                    if time.time() > deadline - 22:
                        return 1e10
                    h = np.dot(params, basis)
                    s = np.sum(h)
                    if abs(s) < 1e-9:
                        return 1e10
                    return _score_smooth(h, beta=80.0)
                
                res_smooth = minimize(
                    smooth_obj, params_init, method='BFGS',
                    options={'maxiter': 300, 'disp': False, 'gtol': 1e-7}
                )
                
                # Then refine with the true objective using L-BFGS-B
                h_smooth = np.dot(res_smooth.x, basis)
                s_smooth = np.sum(h_smooth)
                if abs(s_smooth) > 1e-6:
                    h_smooth = h_smooth / s_smooth
                    # Re-project to get refined initial parameters
                    params_refined, _, _, _ = np.linalg.lstsq(basis.T, h_smooth, rcond=None)
                
                def lbfgs_obj(params):
                    if time.time() > deadline - 20:
                        return 1e10
                    h = np.dot(params, basis)
                    s = np.sum(h)
                    if abs(s) < 1e-9:
                        return 1e10
                    return _autoconv_max_abs(h) / (s * s)
                
                res_lbfgs = minimize(
                    lbfgs_obj, params_refined, method='L-BFGS-B',
                    options={'maxiter': 300, 'disp': False, 'gtol': 1e-8}
                )
                h_opt = np.dot(res_lbfgs.x, basis)
                s_opt = np.sum(h_opt)
                if abs(s_opt) > 1e-6:
                    h_opt = h_opt / s_opt
                    val_opt = _score(h_opt)
                    if val_opt < best_score:
                        best_score = val_opt
                        best_h = h_opt.copy()
            except Exception:
                pass
        
        # Also try direct L-BFGS-B with true objective on remaining inits
        for params_init in inits[6:]:
            if time.time() > deadline - 15:
                break
            
            try:
                def lbfgs_obj(params):
                    if time.time() > deadline - 12:
                        return 1e10
                    h = np.dot(params, basis)
                    s = np.sum(h)
                    if abs(s) < 1e-9:
                        return 1e10
                    return _autoconv_max_abs(h) / (s * s)
                
                res_lbfgs = minimize(
                    lbfgs_obj, params_init, method='L-BFGS-B',
                    options={'maxiter': 200, 'disp': False, 'gtol': 1e-6}
                )
                h_opt = np.dot(res_lbfgs.x, basis)
                s_opt = np.sum(h_opt)
                if abs(s_opt) > 1e-6:
                    h_opt = h_opt / s_opt
                    val_opt = _score(h_opt)
                    if val_opt < best_score:
                        best_score = val_opt
                        best_h = h_opt.copy()
            except Exception:
                pass
    
    # Phase 3: Basin hopping for further refinement
    if time.time() < deadline - 15:
        try:
            # Get current best in parameter space
            h_curr = best_h.copy()
            s_curr = np.sum(h_curr)
            if abs(s_curr) > 1e-6:
                h_curr = h_curr / s_curr
            params_curr, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)
            
            def basin_obj(params):
                if time.time() > deadline - 10:
                    return 1e10
                h = np.dot(params, basis)
                s = np.sum(h)
                if abs(s) < 1e-9:
                    return 1e10
                return _autoconv_max_abs(h) / (s * s)
            
            res_bh = basinhopping(
                basin_obj, params_curr, 
                niter=25, T=0.1, stepsize=0.3,
                minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 150, 'gtol': 1e-7}},
                seed=42
            )
            if res_bh.fun < best_score:
                h_bh = np.dot(res_bh.x, basis)
                s_bh = np.sum(h_bh)
                if abs(s_bh) > 1e-6:
                    h_bh = h_bh / s_bh
                    val_bh = _score(h_bh)
                    if val_bh < best_score:
                        best_score = val_bh
                        best_h = h_bh.copy()
        except Exception:
            pass
    
    # Phase 4: Local search refinement on the full N-dimensional space
    if time.time() < deadline - 10:
        # Sign flip search
        h_flip, val_flip = _local_search_flip(best_h, max_passes=4, deadline=deadline-7)
        if val_flip < best_score:
            best_score = val_flip
            best_h = h_flip.copy()
        
        # Perturbation search
        if time.time() < deadline - 5:
            h_pert, val_pert = _local_search_perturb(best_h, max_passes=6, deadline=deadline-3, step_size=0.03)
            if val_pert < best_score:
                best_score = val_pert
                best_h = h_pert.copy()
    
    # Phase 5: Final Gaussian perturbation / simulated annealing
    if time.time() < deadline - 3:
        h_sa, val_sa = _gaussian_perturbation(best_h, scale=0.05, deadline=deadline-1, max_iters=400)
        if val_sa < best_score:
            best_score = val_sa
            best_h = h_sa.copy()
    
    # Final high-precision polish: try to refine with very tight tolerance
    if time.time() < deadline - 2:
        try:
            h_curr = best_h.copy()
            s_curr = np.sum(h_curr)
            if abs(s_curr) > 1e-6:
                h_curr = h_curr / s_curr
            params_curr, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)
            
            def final_obj(params):
                if time.time() > deadline - 1:
                    return 1e10
                h = np.dot(params, basis)
                s = np.sum(h)
                if abs(s) < 1e-9:
                    return 1e10
                return _autoconv_max_abs(h) / (s * s)
            
            res_final = minimize(
                final_obj, params_curr, method='L-BFGS-B',
                options={'maxiter': 500, 'disp': False, 'gtol': 1e-10}
            )
            h_final = np.dot(res_final.x, basis)
            s_final = np.sum(h_final)
            if abs(s_final) > 1e-6:
                h_final = h_final / s_final
                val_final = _score(h_final)
                if val_final < best_score:
                    best_score = val_final
                    best_h = h_final.copy()
        except Exception:
            pass
    
    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N
    
    return best_h

# EVOLVE-BLOCK-END
