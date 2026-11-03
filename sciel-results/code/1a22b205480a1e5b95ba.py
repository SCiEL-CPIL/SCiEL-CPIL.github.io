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

Additionally, we exploit the spectral structure of the autoconvolution:
autoconvolution in frequency domain is |H(f)|^2, so max|ac| is related to
the peak of the power spectrum. We use a spectral surrogate that directly
optimizes the ratio of spectral peak to (sum h)^2, which is smoother and
more amenable to gradient-based optimization.
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

def _autoconv_full(h):
    """Compute full autoconvolution and its max absolute value."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    return ac, np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _score_and_ac(h):
    """Compute score and full autoconvolution for further analysis."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10, None
    ac, max_abs = _autoconv_full(h)
    return max_abs / (s * s), ac

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

def _spectral_surrogate(h, beta=30.0):
    """
    Spectral surrogate: approximates max|ac|/(sum h)^2 using the power spectrum.
    The autoconvolution is the IFFT of |H|^2. Its max abs is bounded by the
    L1 norm of |H|^2 / fsize. We use a soft-max over the power spectrum as
    a smooth proxy for the time-domain max.
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
    power = np.abs(H) ** 2
    
    # The time-domain autoconvolution is ifft(power). Its max abs is at most
    # sum(power)/fsize. We use a soft-max over power/fsize as a surrogate.
    # Actually, ac = ifft(power)[:size], so max|ac| <= sum(power)/fsize.
    # A better surrogate: soft-max of the actual ac values.
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

def _make_basis_functions(n, n_basis=32):
    """
    Create a set of basis functions for parameterizing h.
    Returns a 2D array of shape (n_basis, N).
    """
    x = np.arange(n, dtype=float)
    basis = []
    
    # Constant function (needed for sum normalization)
    basis.append(np.ones(n))
    
    # Low-frequency cosines
    for k in range(1, 10):
        basis.append(np.cos(2 * np.pi * k * x / n))
    
    # Low-frequency sines
    for k in range(1, 10):
        basis.append(np.sin(2 * np.pi * k * x / n))
    
    # Gaussian bumps at different positions with varied widths
    for center_frac in [0.15, 0.3, 0.5, 0.7, 0.85]:
        mu = center_frac * n
        sigma = n / 10
        basis.append(np.exp(-(x - mu)**2 / (2 * sigma**2)))
    
    # Cosine window
    basis.append(np.cos(np.pi * x / n))
    
    # Quadratic polynomial
    basis.append((x - n/2)**2 / (n**2))
    
    # Linear ramp
    basis.append(x / n - 0.5)
    
    # Higher frequency cosines
    for k in [12, 16, 20, 25]:
        basis.append(np.cos(2 * np.pi * k * x / n))
    
    # Cubic polynomial for more shape flexibility
    basis.append(((x - n/2) / (n/2))**3)
    
    return np.array(basis[:n_basis])

def _local_search_perturb(h, max_passes=5, deadline=None, step_size=0.05):
    """
    Local optimization: try small perturbations to each element, keep if improves score.
    Uses coordinate descent style updates with adaptive step size.
    """
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(777)
    
    current_step = step_size
    no_improve_count = 0
    
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
            for delta in [current_step, -current_step, 0.5*current_step, -0.5*current_step]:
                h[i] = old_val + delta
                new_val = _score(h)
                if new_val < best_local_val:
                    best_local_val = new_val
                    best_local = h[i]
            if best_local != old_val:
                h[i] = best_local
                best_val = best_local_val
                improved = True
                no_improve_count = 0
            else:
                h[i] = old_val  # revert if no improvement
                no_improve_count += 1
        # Adapt step size
        if not improved:
            break
        if no_improve_count > N // 2:
            current_step *= 0.7
        elif improved:
            current_step = min(current_step * 1.1, step_size)
    return h, best_val

def _local_search_flip(h, max_passes=3, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    Uses randomized order and early termination.
    """
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(555)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        idx_order = rng.permutation(N)
        for i in idx_order:
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
    Uses adaptive temperature and multiple restarts within time budget.
    """
    h_curr = h.copy()
    h_best = h.copy()
    curr_val = _score(h_curr)
    best_val = curr_val
    rng = np.random.default_rng(999)
    
    T_start = 0.5
    T_end = 0.005
    n_iter = max_iters
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        T = T_start * (T_end / T_start) ** (it / max(1, n_iter - 1))
        
        # Generate perturbation with occasional larger jumps
        if rng.random() < 0.05:
            perturb = rng.standard_normal(N) * scale * 3
        else:
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
        scale *= 0.997
    
    return h_best, best_val

def _peak_suppression_search(h, deadline=None, max_iters=100):
    """
    Targeted search that tries to reduce the maximum autoconvolution peak.
    Identifies the location of the max peak and applies counter-perturbations.
    """
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(888)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        ac, max_abs = _autoconv_full(h)
        if ac is None:
            break
        
        # Find the index of the maximum absolute autoconvolution value
        peak_idx = np.argmax(np.abs(ac))
        
        # The autoconvolution at index k is sum_j h[j]*h[k-j]
        # To reduce the peak at position k, we can try adjusting h values
        # that contribute to this sum
        k = peak_idx
        # Elements that contribute: h[j] for j where 0 <= j < N and 0 <= k-j < N
        valid_j = [j for j in range(N) if 0 <= k - j < N]
        
        if not valid_j:
            continue
        
        # Try perturbing a random subset of contributing elements
        n_perturb = min(5, len(valid_j))
        perturbed_idx = rng.choice(valid_j, size=n_perturb, replace=False)
        
        h_trial = h.copy()
        for j in perturbed_idx:
            # Try to reduce the contribution at the peak
            # If ac[k] > 0, decrease h[j] or h[k-j]; if ac[k] < 0, increase them
            sign = np.sign(ac[k])
            delta = rng.uniform(0.01, 0.05) * (-sign)
            h_trial[j] += delta
            # Also try the complementary element
            kj = k - j
            if 0 <= kj < N and kj != j:
                h_trial[kj] += delta * 0.5
        
        trial_val = _score(h_trial)
        if trial_val < best_val:
            best_val = trial_val
            h = h_trial.copy()
    
    return h, best_val

def _spectral_optimization(h, basis, params, deadline, beta_schedule=None):
    """
    Optimize using spectral surrogate with adaptive beta.
    Returns improved params and score.
    """
    if beta_schedule is None:
        beta_schedule = [30.0, 60.0, 120.0, 250.0]
    
    current_params = params.copy()
    
    for beta in beta_schedule:
        if time.time() > deadline:
            break
        
        def obj(params, b=beta):
            if time.time() > deadline:
                return 1e10
            h = np.dot(params, basis)
            s = np.sum(h)
            if abs(s) < 1e-9:
                return 1e10
            return _score_smooth(h, beta=b)
        
        try:
            res = minimize(
                obj, current_params, method='L-BFGS-B',
                options={'maxiter': 300, 'disp': False, 'gtol': 1e-9}
            )
            current_params = res.x
        except Exception:
            pass
    
    return current_params

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s limit

    best_score = np.inf
    best_h = np.ones(N) / N  # Baseline: constant sequence
    
    # Evaluate baseline
    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()
    
    # Create basis functions for parameterization
    n_basis = 32
    basis = _make_basis_functions(N, n_basis)
    
    # Phase 1: Global optimization with differential evolution on compact parameterization
    phase1_deadline = deadline - 50
    if time.time() < phase1_deadline:
        bounds = [(-2.0, 2.0)] * n_basis
        
        def de_objective(params):
            if time.time() > phase1_deadline:
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
                polish=False
            )
            if res_de.fun < best_score:
                h_de = np.dot(res_de.x, basis)
                s_de = np.sum(h_de)
                if abs(s_de) > 1e-6:
                    h_de = h_de / s_de
                val_de = _score(h_de)
                if val_de < best_score:
                    best_score = val_de
                    best_h = h_de.copy()
        except Exception:
            pass
    
    # Phase 2: Spectral surrogate optimization with adaptive beta
    phase2_deadline = deadline - 25
    if time.time() < phase2_deadline:
        # Generate diverse initializations
        inits = []
        
        h_curr = best_h.copy()
        s_curr = np.sum(h_curr)
        if abs(s_curr) > 1e-6:
            h_curr = h_curr / s_curr
        
        params_init, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)
        inits.append(params_init)
        
        rng = np.random.default_rng(123)
        for i in range(10):
            params_rand = rng.standard_normal(n_basis) * 0.5
            inits.append(params_rand)
        
        for amp in [0.2, 0.5, 0.8, 1.0]:
            params_struct = np.zeros(n_basis)
            params_struct[0] = 1.0
            params_struct[1] = amp * 0.1
            inits.append(params_struct)
        
        # Spectral surrogate optimization with multiple beta values
        for params_init in inits[:10]:
            if time.time() > phase2_deadline - 5:
                break
            
            try:
                # Use adaptive beta schedule: start smooth, get sharper
                improved_params = _spectral_optimization(
                    best_h, basis, params_init, phase2_deadline - 3,
                    beta_schedule=[30.0, 60.0, 120.0, 250.0, 500.0]
                )
                
                # Final refinement with true objective
                def lbfgs_obj(params):
                    if time.time() > phase2_deadline - 1:
                        return 1e10
                    h = np.dot(params, basis)
                    s = np.sum(h)
                    if abs(s) < 1e-9:
                        return 1e10
                    return _autoconv_max_abs(h) / (s * s)
                
                res_lbfgs = minimize(
                    lbfgs_obj, improved_params, method='L-BFGS-B',
                    options={'maxiter': 300, 'disp': False, 'gtol': 1e-9}
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
    
    # Phase 3: Basin hopping
    phase3_deadline = deadline - 12
    if time.time() < phase3_deadline:
        try:
            h_curr = best_h.copy()
            s_curr = np.sum(h_curr)
            if abs(s_curr) > 1e-6:
                h_curr = h_curr / s_curr
            params_curr, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)
            
            def basin_obj(params):
                if time.time() > phase3_deadline:
                    return 1e10
                h = np.dot(params, basis)
                s = np.sum(h)
                if abs(s) < 1e-9:
                    return 1e10
                return _autoconv_max_abs(h) / (s * s)
            
            res_bh = basinhopping(
                basin_obj, params_curr, 
                niter=25, T=0.1, stepsize=0.3,
                minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 150, 'gtol': 1e-8}},
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
    
    # Phase 4: Peak suppression search
    phase4_deadline = deadline - 8
    if time.time() < phase4_deadline:
        h_peak, val_peak = _peak_suppression_search(best_h, deadline=phase4_deadline, max_iters=100)
        if val_peak < best_score:
            best_score = val_peak
            best_h = h_peak.copy()
    
    # Phase 5: Local search refinement
    phase5_deadline = deadline - 3
    if time.time() < phase5_deadline:
        h_flip, val_flip = _local_search_flip(best_h, max_passes=3, deadline=phase5_deadline - 2)
        if val_flip < best_score:
            best_score = val_flip
            best_h = h_flip.copy()
        
        if time.time() < phase5_deadline - 1:
            h_pert, val_pert = _local_search_perturb(best_h, max_passes=4, deadline=phase5_deadline - 0.5, step_size=0.03)
            if val_pert < best_score:
                best_score = val_pert
                best_h = h_pert.copy()
    
    # Phase 6: Final simulated annealing polish
    if time.time() < deadline - 1:
        h_sa, val_sa = _gaussian_perturbation(best_h, scale=0.03, deadline=deadline - 0.5, max_iters=350)
        if val_sa < best_score:
            best_score = val_sa
            best_h = h_sa.copy()
    
    # Final high-precision polish with very tight tolerance
    if time.time() < deadline - 0.5:
        try:
            h_curr = best_h.copy()
            s_curr = np.sum(h_curr)
            if abs(s_curr) > 1e-6:
                h_curr = h_curr / s_curr
            params_curr, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)
            
            def final_obj(params):
                if time.time() > deadline - 0.2:
                    return 1e10
                h = np.dot(params, basis)
                s = np.sum(h)
                if abs(s) < 1e-9:
                    return 1e10
                return _autoconv_max_abs(h) / (s * s)
            
            res_final = minimize(
                final_obj, params_curr, method='L-BFGS-B',
                options={'maxiter': 500, 'disp': False, 'gtol': 1e-12}
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
