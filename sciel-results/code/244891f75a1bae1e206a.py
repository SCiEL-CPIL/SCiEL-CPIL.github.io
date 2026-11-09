"""Refined program for Laguerre double roots (m=12, n=1) - structural redesign."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _precompute_laguerre_coeffs(alpha, max_deg=50):
    """Precompute coefficient arrays for L_0^alpha, L_2^alpha, ..., L_50^alpha."""
    coeffs_list = []
    for d in range(0, max_deg + 1, 2):
        c = np.zeros(d + 1)
        for k in range(d + 1):
            # binom(d+alpha, d-k) * (-1)^k / k!
            # Use log-gamma for numerical stability
            from math import lgamma, factorial
            # binom(d+alpha, d-k) = Gamma(d+alpha+1) / (Gamma(d-k+1) * Gamma(k+1))
            log_num = lgamma(d + alpha + 1)
            log_den1 = lgamma(d - k + 1)
            log_den2 = lgamma(k + 1)
            log_binom = log_num - log_den1 - log_den2
            if log_binom < 700:
                binom_val = math.exp(log_binom)
            else:
                binom_val = 0.0
            c[k] = binom_val * ((-1) ** k) / factorial(k)
        coeffs_list.append(c)
    return coeffs_list

def _eval_poly_and_deriv(coeffs, x):
    """Evaluate polynomial and derivative at x using Horner's method."""
    val = 0.0
    deriv = 0.0
    for k in range(len(coeffs) - 1, -1, -1):
        deriv = deriv * x + val
        val = val * x + coeffs[k]
    return val, deriv

def _fast_score(z_list, lag_coeffs):
    """
    Fast numerical score: B/(2*pi).
    Returns None if system is singular or invalid.
    """
    z_sorted = sorted(z_list)
    n_z = 12
    n_pols = 26
    
    # Build 26x26 system
    A = np.zeros((26, 26))
    b = np.zeros(26)
    
    # Row 0: g(0) = 0
    for j in range(26):
        val, _ = _eval_poly_and_deriv(lag_coeffs[j], 0.0)
        A[0, j] = val
    b[0] = 0.0
    
    # Row 1: g'(0) = 1
    for j in range(26):
        _, deriv = _eval_poly_and_deriv(lag_coeffs[j], 0.0)
        A[1, j] = deriv
    b[1] = 1.0
    
    # Rows 2..25: g(z_i)=0, g'(z_i)=0
    for i in range(n_z):
        zi = z_sorted[i]
        row_g = 2 + 2 * i
        row_gp = 3 + 2 * i
        for j in range(26):
            val, deriv = _eval_poly_and_deriv(lag_coeffs[j], zi)
            A[row_g, j] = val
            A[row_gp, j] = deriv
        b[row_g] = 0.0
        b[row_gp] = 0.0
    
    # Check condition number
    try:
        cond = np.linalg.cond(A)
        if cond > 1e15:
            return None
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    except Exception:
        return None
    
    # Build g(x) as standard polynomial
    g_coeffs = np.zeros(51)
    for j in range(26):
        for k in range(len(lag_coeffs[j])):
            g_coeffs[k] += c[j] * lag_coeffs[j][k]
    
    # Build denominator: x * prod_i (x - z_i)^2
    denom = np.array([0.0, 1.0])  # x
    for zi in z_sorted:
        factor = np.array([zi * zi, -2.0 * zi, 1.0])
        denom = np.convolve(denom, factor)
    
    # Polynomial long division: g / denom
    deg_g = len(g_coeffs) - 1
    deg_d = len(denom) - 1
    
    if deg_g < deg_d:
        return 0.0
    
    q_coeffs = np.zeros(deg_g - deg_d + 1)
    rem = g_coeffs.copy()
    
    for idx in range(deg_g - deg_d, -1, -1):
        if abs(rem[-1]) < 1e-14:
            break
        lc = rem[-1] / denom[-1]
        q_coeffs[idx] = lc
        # Subtract lc * x^idx * denom from rem
        for k in range(len(denom)):
            pos = idx + k
            if pos < len(rem):
                rem[pos] -= lc * denom[k]
        # Trim
        while len(rem) > 0 and abs(rem[-1]) < 1e-14:
            rem = rem[:-1]
        if len(rem) <= deg_d:
            break
    
    # Find real roots of q
    try:
        roots = np.roots(q_coeffs)
    except Exception:
        return None
    
    real_roots = [float(r.real) for r in roots if abs(r.imag) < 1e-8]
    
    if not real_roots:
        return 0.0
    
    # Evaluate q at each positive real root with sign check
    def eval_q(xv):
        val = 0.0
        for k in range(len(q_coeffs) - 1, -1, -1):
            val = val * xv + q_coeffs[k]
        return val
    
    max_r = 0.0
    eps = 1e-12
    for r in real_roots:
        if r <= 0:
            continue
        qm = eval_q(r - eps)
        qp = eval_q(r + eps)
        if qm * qp < 0:
            if r > max_r:
                max_r = r
    
    B = max(0.0, max_r)
    return B / (2.0 * math.pi)

def _validate_z(z_list):
    """Check validity: 12 positive, finite, pairwise distinct."""
    if len(z_list) != 12:
        return False
    for z in z_list:
        if not math.isfinite(z) or z <= 0:
            return False
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(z_list[i] - z_list[j]) < 1e-10:
                return False
    return True

def _generate_candidate(log_scale, spread, offset):
    """Generate a candidate z configuration."""
    z = []
    for i in range(12):
        val = offset + spread * (log_scale ** i)
        if val <= 0:
            val = offset + 0.01
        z.append(val)
    return z

def construct():
    """Return 12 double-root locations z_1..z_12."""
    deadline = time.time() + 590.0  # 630s kill, keep 40s margin
    lag_coeffs = _precompute_laguerre_coeffs(-0.5, 50)
    
    best_z = None
    best_score = float('inf')
    
    # Phase 1: Try diverse structural configurations
    configs = []
    
    # Geometric with various parameters
    for scale in [1.1, 1.2, 1.3, 1.5, 2.0]:
        for offset in [0.1, 0.5, 1.0, 2.0]:
            for spread in [0.5, 1.0, 2.0]:
                configs.append((scale, spread, offset))
    
    # Clustered configurations: some z close together, others far
    for base in [1.0, 2.0, 5.0]:
        for gap in [0.1, 0.5, 1.0]:
            z = []
            for i in range(12):
                if i < 6:
                    z.append(base + i * gap)
                else:
                    z.append(base + 6 * gap + (i - 6) * gap * 5)
            if _validate_z(z):
                configs.append(('clustered', z))
    
    # Bessel J0 zeros squared
    def bessel_zero_J0(k):
        x = (k - 0.25) * math.pi
        if x < 0.1:
            return 2.4048255576957728
        return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)
    
    z_bessel = [bessel_zero_J0(i)**2 for i in range(1, 13)]
    if _validate_z(z_bessel):
        configs.append(('bessel', z_bessel))
    
    # Evenly spaced
    z_even = [1.0 + i * 1.5 for i in range(12)]
    if _validate_z(z_even):
        configs.append(('even', z_even))
    
    # Evaluate all configs
    for cfg in configs:
        if time.time() > deadline - 30:
            break
        
        if isinstance(cfg, tuple) and len(cfg) == 3 and all(isinstance(x, float) for x in cfg):
            z = _generate_candidate(*cfg)
        elif isinstance(cfg, tuple) and len(cfg) == 2:
            z = cfg[1]
        else:
            continue
        
        if not _validate_z(z):
            continue
        
        try:
            score = _fast_score(z, lag_coeffs)
            if score is not None and score < best_score:
                best_score = score
                best_z = list(z)
        except Exception:
            continue
    
    # Phase 2: Simulated annealing refinement from best
    if best_z is not None and time.time() < deadline - 60:
        z_current = list(best_z)
        current_score = best_score
        T = 0.1  # initial temperature
        
        while time.time() < deadline - 30:
            T *= 0.95
            if T < 1e-4:
                break
            
            # Random perturbation
            z_trial = list(z_current)
            i = np.random.randint(0, 12)
            delta = np.random.randn() * T * max(1.0, z_current[i])
            z_trial[i] = z_current[i] + delta
            
            if not _validate_z(z_trial):
                continue
            
            try:
                score_trial = _fast_score(z_trial, lag_coeffs)
            except Exception:
                continue
            
            if score_trial is None:
                continue
            
            if score_trial < current_score:
                z_current = z_trial
                current_score = score_trial
                if current_score < best_score:
                    best_score = current_score
                    best_z = list(z_trial)
            else:
                # Accept with probability exp(-delta/T)
                delta_score = score_trial - current_score
                if delta_score < 0:
                    z_current = z_trial
                    current_score = score_trial
                elif np.random.random() < math.exp(-delta_score / max(T, 1e-10)):
                    z_current = z_trial
                    current_score = score_trial
    
    # Phase 3: Local coordinate descent from best
    if best_z is not None and time.time() < deadline - 20:
        z_local = list(best_z)
        local_score = best_score
        step_sizes = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005]
        
        for ss in step_sizes:
            if time.time() > deadline - 15:
                break
            improved = True
            while improved and time.time() < deadline - 15:
                improved = False
                for i in range(12):
                    for direction in [-1, 1]:
                        z_try = list(z_local)
                        z_try[i] = z_local[i] + direction * ss
                        if not _validate_z(z_try):
                            continue
                        try:
                            s = _fast_score(z_try, lag_coeffs)
                        except Exception:
                            continue
                        if s is not None and s < local_score - 1e-12:
                            z_local = z_try
                            local_score = s
                            if local_score < best_score:
                                best_score = local_score
                                best_z = list(z_local)
                            improved = True
                            break
    
    if best_z is None:
        # Fallback: evenly spaced
        return [1.0 + i * 2.0 for i in range(12)]
    
    return best_z

# EVOLVE-BLOCK-END
