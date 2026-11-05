"""Sparse rectangular pulse superposition with direct convex optimization of amplitudes."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 145.0  # Safety margin before 150s kill

def _eval_value(h):
    """Compute the objective value (lower is better)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(~np.isfinite(h)):
        return np.inf
    sh = float(np.sum(h))
    if abs(sh) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    max_conv = float(np.max(conv))
    val = abs(2.0 * N * max_conv / (sh * sh))
    return val

def _pulse_response(center, width, h):
    """Compute the contribution of a rectangular pulse at center with given width to conv."""
    # Rectangular pulse: h[i] = 1 for i in [center - width//2, center + width//2]
    start = max(0, int(center) - width // 2)
    end = min(N, int(center) + width // 2 + 1)
    pulse = np.zeros(N)
    pulse[start:end] = 1.0
    return np.convolve(pulse, pulse)

def _build_pulse_basis(pulse_specs):
    """
    Build a basis matrix where each row is the autoconvolution response of one pulse.
    pulse_specs: list of (center, width) tuples
    Returns: (B, pulse_masks) where B[k] is the conv response of pulse k,
             pulse_masks[k] is the binary mask of pulse k
    """
    n_pulses = len(pulse_specs)
    B = np.zeros((n_pulses, 2 * N - 1))
    masks = []
    for k, (center, width) in enumerate(pulse_specs):
        start = max(0, int(center) - width // 2)
        end = min(N, int(center) + width // 2 + 1)
        mask = np.zeros(N)
        mask[start:end] = 1.0
        masks.append(mask)
        B[k] = np.convolve(mask, mask)
    return B, masks

def _pulse_objective(params, B, masks):
    """
    Objective for pulse amplitude optimization.
    params: [a_0, a_1, ..., a_{k-1}] amplitudes
    B: autoconv responses, masks: pulse masks
    """
    k = len(masks)
    a = params[:k]
    # h = sum a_i * mask_i
    h = np.zeros(N)
    for i in range(k):
        h += a[i] * masks[i]
    
    sh = np.sum(h)
    if abs(sh) < 1e-12:
        return 1e10
    
    # max_conv = max over conv = max over B @ a (since conv is linear in h)
    conv_vals = B @ a
    max_conv = np.max(conv_vals)
    
    val = abs(2.0 * N * max_conv / (sh * sh))
    if not np.isfinite(val):
        return 1e10
    return val

def _pulse_grad(params, B, masks):
    """Gradient of pulse objective."""
    k = len(masks)
    a = params[:k]
    h = np.zeros(N)
    for i in range(k):
        h += a[i] * masks[i]
    
    sh = np.sum(h)
    if abs(sh) < 1e-12:
        return np.zeros_like(params)
    
    conv_vals = B @ a
    max_conv = np.max(conv_vals)
    
    # Find which conv index is the max
    max_idx = np.argmax(conv_vals)
    
    # d(max_conv)/da_j = B[j, max_idx]
    d_max_da = B[:, max_idx]
    
    # d(sh)/da_j = sum(mask_j)
    d_sh_da = np.array([np.sum(m) for m in masks])
    
    # val = |2N * max_conv / sh^2|
    # Let f = 2N * max_conv / sh^2
    # df/da_j = 2N * (d_max_da_j * sh^2 - max_conv * 2 * sh * d_sh_da_j) / sh^4
    #         = 2N * (d_max_da_j * sh - 2 * max_conv * d_sh_da_j) / sh^3
    
    numerator = 2.0 * N * (d_max_da * sh - 2.0 * max_conv * d_sh_da)
    denominator = sh ** 3
    df_da = numerator / denominator
    
    val = abs(2.0 * N * max_conv / (sh * sh))
    
    if 2.0 * N * max_conv / (sh * sh) >= 0:
        grad = df_da
    else:
        grad = -df_da
    
    return grad

def construct():
    t0 = time.time()
    best_h = np.ones(N)
    best_val = _eval_value(best_h)  # 2.0
    
    def elapsed():
        return time.time() - t0
    
    rng = np.random.RandomState(42)
    
    # --- Strategy 1: Sparse rectangular pulse superposition ---
    # Try various configurations of non-overlapping or partially overlapping pulses
    
    pulse_configs = []
    
    # Configuration A: 2 pulses of equal width, separated
    for width in [50, 100, 150]:
        for sep in [width, width + 20, width + 50, N // 2]:
            c1 = sep // 2
            c2 = N - 1 - sep // 2
            pulse_configs.append([(c1, width), (c2, width)])
    
    # Configuration B: 3 pulses
    for width in [40, 60, 80]:
        for offset in [0, 10, 20]:
            c1 = width // 2 + offset
            c2 = N // 2
            c3 = N - 1 - width // 2 - offset
            pulse_configs.append([(c1, width), (c2, width), (c3, width)])
    
    # Configuration C: 4 pulses
    for width in [30, 50, 70]:
        positions = [N // 8, N // 4, 3 * N // 4, 7 * N // 8]
        pulse_configs.append([(p, width) for p in positions])
    
    # Configuration D: 5 pulses
    for width in [30, 40]:
        positions = [N // 10, N // 5, N // 2, 4 * N // 5, 9 * N // 10]
        pulse_configs.append([(p, width) for p in positions])
    
    # Configuration E: Asymmetric widths
    for w1, w2 in [(30, 70), (50, 50), (70, 30), (40, 60), (60, 40)]:
        pulse_configs.append([(N // 4, w1), (3 * N // 4, w2)])
    
    # Configuration F: Random pulse placements
    for _ in range(30):
        n_pulses = rng.randint(2, 6)
        specs = []
        for _ in range(n_pulses):
            width = rng.randint(20, 120)
            center = rng.randint(width // 2, N - width // 2)
            specs.append((center, width))
        pulse_configs.append(specs)
    
    # --- Optimize each pulse configuration ---
    for specs in pulse_configs:
        if elapsed() > DEADLINE - 20:
            break
        
        # Check for overlaps - if pulses overlap, the linearity assumption breaks
        # We'll still try, but the basis approach is approximate for overlapping pulses
        B, masks = _build_pulse_basis(specs)
        k = len(specs)
        
        # Initial guess: alternating signs
        a0 = np.zeros(k)
        for i in range(k):
            a0[i] = (-1) ** i
        
        # Also try all positive
        a0_pos = np.ones(k)
        
        for a_init in [a0, a0_pos]:
            if elapsed() > DEADLINE - 10:
                break
            
            try:
                result = minimize(
                    lambda x: _pulse_objective(x, B, masks),
                    a_init,
                    jac=lambda x: _pulse_grad(x, B, masks),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 400, 'gtol': 1e-8}
                )
                
                if elapsed() > DEADLINE - 5:
                    break
                
                a_opt = result.x
                h_candidate = np.zeros(N)
                for i in range(k):
                    h_candidate += a_opt[i] * masks[i]
                
                v = _eval_value(h_candidate)
                if v < best_val and np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-10:
                    best_val = v
                    best_h = h_candidate.copy()
                    
            except Exception:
                pass
    
    # --- Strategy 2: Direct optimization with smooth surrogate from best pulse result ---
    if elapsed() < DEADLINE - 15:
        h0 = best_h.copy()
        s0 = np.sum(h0)
        if abs(s0) < 1e-10:
            h0 = h0 + 0.01
        
        # Smooth surrogate using log-sum-exp
        def _smooth_obj(x, beta=100.0):
            h = x
            s = np.sum(h)
            if abs(s) < 1e-10:
                return 1e10
            conv = np.convolve(h, h)
            max_c = np.max(conv)
            shifted = conv - max_c
            log_sum_exp = max_c + np.log(np.sum(np.exp(beta * shifted))) / beta
            val = abs(2.0 * N * log_sum_exp / (s * s))
            if not np.isfinite(val):
                return 1e10
            return val
        
        def _smooth_grad_fn(x, beta=100.0):
            h = x
            s = np.sum(h)
            if abs(s) < 1e-10:
                return np.zeros_like(h)
            
            conv = np.convolve(h, h)
            max_c = np.max(conv)
            shifted = conv - max_c
            unnormalized_weights = np.exp(beta * shifted)
            weight_sum = np.sum(unnormalized_weights)
            weights = unnormalized_weights / weight_sum
            
            grad_lse = np.zeros(N)
            for i in range(N):
                grad_lse[i] = 2.0 * np.dot(weights[i:i+N], h)
            
            lse = max_c + np.log(weight_sum) / beta
            f = 2.0 * N * lse / (s * s)
            
            df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)
            
            if f >= 0:
                grad = df_dh
            else:
                grad = -df_dh
            
            return grad
        
        # Polish with different betas
        for beta in [80.0, 120.0, 150.0]:
            if elapsed() > DEADLINE - 5:
                break
            
            try:
                res = minimize(
                    lambda x, b=beta: _smooth_obj(x, b),
                    h0,
                    jac=lambda x, b=beta: _smooth_grad_fn(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'maxfun': 600, 'gtol': 1e-9}
                )
                
                if elapsed() > DEADLINE - 3:
                    break
                
                h_candidate = res.x
                v = _eval_value(h_candidate)
                if v < best_val and np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-10:
                    best_val = v
                    best_h = h_candidate.copy()
                    h0 = best_h.copy()
            except Exception:
                pass
    
    # --- Strategy 3: Additional diverse initializations and quick L-BFGS refinement ---
    if elapsed() < DEADLINE - 20:
        t = np.linspace(0, 1, N, endpoint=False)
        i_arr = np.arange(N)
        
        extra_inits = []
        
        # Cosine-based with varying frequencies
        for freq in [1, 2, 3, 5, 7, 10, 15]:
            extra_inits.append(np.cos(2 * np.pi * freq * t))
            extra_inits.append(np.sin(2 * np.pi * freq * t))
        
        # Bump profiles
        for center in [N//4, N//2, 3*N//4]:
            for width in [30, 50, 80, 120]:
                h_bump = np.exp(-((i_arr - center)**2) / (2 * width**2))
                extra_inits.append(h_bump)
                extra_inits.append(-h_bump)
        
        # Double bumps
        for width in [40, 60, 80]:
            h_double = np.exp(-((i_arr - N//4)**2) / (2 * width**2)) - np.exp(-((i_arr - 3*N//4)**2) / (2 * width**2))
            extra_inits.append(h_double)
        
        # Random smooth
        for _ in range(10):
            h_r = rng.uniform(-1, 1, N)
            from scipy.ndimage import gaussian_filter1d
            h_r = gaussian_filter1d(h_r, sigma=rng.uniform(5, 25))
            extra_inits.append(h_r)
        
        # Evaluate and pick top
        init_vals = []
        for h_init in extra_inits:
            v = _eval_value(h_init)
            if np.isfinite(v):
                init_vals.append((v, h_init))
        init_vals.sort(key=lambda x: x[0])
        
        top_extra = [h for _, h in init_vals[:10]]
        
        for h_init in top_extra:
            if elapsed() > DEADLINE - 5:
                break
            
            s = np.sum(h_init)
            if abs(s) < 1e-10:
                h_init = h_init + 0.05
                s = np.sum(h_init)
            
            try:
                res = minimize(
                    lambda x: _smooth_obj(x, 100.0),
                    h_init,
                    jac=lambda x: _smooth_grad_fn(x, 100.0),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 400, 'gtol': 1e-8}
                )
                
                if elapsed() > DEADLINE - 3:
                    break
                
                h_candidate = res.x
                v = _eval_value(h_candidate)
                if v < best_val and np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-10:
                    best_val = v
                    best_h = h_candidate.copy()
            except Exception:
                pass
    
    # Final safety
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
    
    return best_h
