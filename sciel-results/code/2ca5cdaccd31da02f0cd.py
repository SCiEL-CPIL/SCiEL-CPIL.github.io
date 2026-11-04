"""Second autocorrelation inequality: spectral parameterization approach."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
d = 1.0 / (2 * N)  # interval width

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin below 150s limit

    def _time_left():
        return deadline - time.time()

    def _autoconv_peak(h):
        """Compute ||h*h||_inf for step function heights h."""
        ac = np.correlate(h, h, mode='full')
        return np.max(ac)

    def _ratio(h):
        """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
        h = np.asarray(h, dtype=float)
        h = np.maximum(0, h)
        sum_sq = np.sum(h**2)
        sum_h = np.sum(h)
        if sum_h < 1e-15 or sum_sq < 1e-30:
            return 0.0
        peak = _autoconv_peak(h)
        if peak < 1e-30:
            return 0.0
        return sum_sq / (d * sum_h * peak)

    def _smooth_max(x, alpha=200.0):
        """Smooth approximation of max using log-sum-exp."""
        x_max = np.max(x)
        if x_max < 1e-30:
            return 1e-30
        shifted = alpha * (x - x_max)
        sum_exp = np.sum(np.exp(shifted))
        if sum_exp < 1e-300:
            sum_exp = 1e-300
        return x_max + np.log(sum_exp) / alpha

    def _spectral_to_heights(params):
        """
        Convert spectral parameters to time-domain heights.
        params = [m1, m2, ..., mK, p1, p2, ..., pK, scale]
        where m_i are magnitudes (log-scale), p_i are phases, scale is overall.
        
        h = |IDFT(spectral)|^2 where spectral has K active bins.
        """
        K = len(params) // 3
        mags = np.exp(params[:K])  # Exponentiate to get positive magnitudes
        phases = params[K:2*K]
        scale = np.exp(params[2*K]) if len(params) > 2*K else 1.0
        
        # Create sparse spectral vector
        spec = np.zeros(N, dtype=complex)
        # Place active bins at strategic positions
        # Use symmetric bins around DC for real-valued output
        if K == 1:
            spec[0] = mags[0] * np.exp(1j * phases[0])
        elif K == 2:
            spec[0] = mags[0] * np.exp(1j * phases[0])
            spec[N//2] = mags[1] * np.exp(1j * phases[1])
        elif K == 3:
            spec[0] = mags[0] * np.exp(1j * phases[0])
            spec[N//4] = mags[1] * np.exp(1j * phases[1])
            spec[3*N//4] = mags[2] * np.exp(1j * phases[2])
        elif K == 4:
            spec[0] = mags[0] * np.exp(1j * phases[0])
            spec[N//4] = mags[1] * np.exp(1j * phases[1])
            spec[N//2] = mags[2] * np.exp(1j * phases[2])
            spec[3*N//4] = mags[3] * np.exp(1j * phases[3])
        else:
            # For larger K, use evenly spaced bins
            bins = np.linspace(0, N-1, K, dtype=int)
            for i, b in enumerate(bins):
                spec[b] = mags[i] * np.exp(1j * phases[i])
        
        # Compute time-domain heights as squared magnitude of IDFT
        h = np.abs(np.fft.ifft(spec))**2
        h = h * scale
        return np.maximum(0, h)

    def _spectral_objective(params):
        """Negative ratio for minimization."""
        h = _spectral_to_heights(params)
        sum_h = np.sum(h)
        if sum_h < 1e-15:
            return 1e10
        ratio = _ratio(h)
        if ratio < 1e-15:
            return 1e10
        return -np.log(ratio)  # Minimize -log(ratio) = maximize log(ratio)

    def _optimize_spectral(K, h_init, maxiter=300, maxfun=5000):
        """Optimize spectral parameters for given K active bins."""
        # Initialize from h_init
        H = np.fft.fft(h_init)
        H_mag = np.abs(H)
        H_phase = np.angle(H)
        
        # Find top K magnitude bins
        if K == 1:
            bin_idx = [0]
        elif K == 2:
            bin_idx = [0, N//2]
        elif K == 3:
            bin_idx = [0, N//4, 3*N//4]
        elif K == 4:
            bin_idx = [0, N//4, N//2, 3*N//4]
        else:
            bin_idx = np.linspace(0, N-1, K, dtype=int).tolist()
        
        # Initialize parameters
        mags_init = []
        phases_init = []
        for b in bin_idx:
            m = H_mag[b]
            if m < 1e-10:
                m = 1.0
            mags_init.append(np.log(m))
            phases_init.append(H_phase[b])
        
        # Estimate scale
        h_from_spec = _spectral_to_heights(mags_init + phases_init + [0.0])
        s_h = np.sum(h_from_spec)
        s_init = np.sum(h_init)
        if s_h > 1e-10 and s_init > 1e-10:
            scale_init = np.log(s_init / s_h)
        else:
            scale_init = 0.0
        
        params0 = mags_init + phases_init + [scale_init]
        
        # Bounds: magnitudes in log space [-5, 5], phases [-pi, pi], scale [-3, 3]
        bounds = [(-5, 5)] * K + [(-np.pi, np.pi)] * K + [(-3, 3)]
        
        try:
            res = minimize(
                _spectral_objective, params0, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'maxfun': maxfun, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            h_opt = _spectral_to_heights(res.x)
            return h_opt, _ratio(h_opt)
        except Exception:
            return None, 0.0

    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # Single spike
    h = np.zeros(N)
    h[N//2] = 1.0
    candidates.append(h.copy())
    
    # Gaussian shapes
    t = np.arange(N)
    for center in [N//4, N//2, 3*N//4]:
        for width in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            h = np.exp(-0.5 * ((t - center) / width)**2)
            candidates.append(h.copy())
    
    # Bimodal
    for gap in [5, 10, 15, 20]:
        for w in [1.0, 2.0, 3.0]:
            h1 = np.exp(-0.5 * ((t - (N//2 - gap)) / w)**2)
            h2 = np.exp(-0.5 * ((t - (N//2 + gap)) / w)**2)
            candidates.append((h1 + h2).copy())
    
    # Random smooth
    rng = np.random.default_rng(42)
    for _ in range(10):
        h = np.abs(rng.normal(1.0, 0.5, N))
        kernel = np.array([1, 2, 3, 2, 1]) / 9.0
        h = np.convolve(h, kernel, mode='same')
        candidates.append(h.copy())
    
    # Uniform
    candidates.append(np.ones(N).copy())
    
    # Evaluate all candidates
    best_h = candidates[0].copy()
    best_score = _ratio(best_h)
    for h in candidates:
        s = _ratio(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()

    # Phase 2: Spectral optimization from multiple K values and starts
    K_values = [1, 2, 3, 4]
    start_indices = [0, len(candidates)//4, len(candidates)//2, 3*len(candidates)//4, len(candidates)-1]
    
    for K in K_values:
        if _time_left() < 20:
            break
        for idx in start_indices:
            if _time_left() < 15:
                break
            h0 = candidates[idx].copy()
            h_opt, s_opt = _optimize_spectral(K, h0)
            if h_opt is not None and s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # Phase 3: Refine best with more iterations
    if _time_left() > 10:
        for K in [2, 3, 4]:
            if _time_left() < 8:
                break
            h_opt, s_opt = _optimize_spectral(K, best_h, maxiter=500, maxfun=10000)
            if h_opt is not None and s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # Phase 4: Coordinate descent polish
    if _time_left() > 5:
        h = best_h.copy()
        for iteration in range(50):
            if _time_left() < 3:
                break
            improved = False
            for i in range(N):
                old = h[i]
                for mult in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 5.0]:
                    h[i] = max(0.0, old * mult) if old > 0 else mult
                    s = _ratio(h)
                    if s > best_score + 1e-15:
                        best_score = s
                        best_h = h.copy()
                        improved = True
                    else:
                        h[i] = old
                        break
                h[i] = old
            if not improved:
                break

    # Ensure nonnegativity and return
    best_h = np.maximum(0, best_h)
    if np.sum(best_h) < 1e-15:
        best_h = np.zeros(N)
        best_h[N // 2] = 1.0

    return list(best_h)

# EVOLVE-BLOCK-END
