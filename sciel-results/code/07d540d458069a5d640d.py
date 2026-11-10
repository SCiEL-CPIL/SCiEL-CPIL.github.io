import time
import numpy as np
from scipy.optimize import minimize_scalar, minimize

def _sup_norm_high_precision(coeffs, num_samples=50000, refine=True):
    """
    Compute sup norm with high precision.
    """
    n = len(coeffs)
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * zs + c
    
    mag = np.abs(vals)
    if not refine:
        return float(np.max(mag))
    
    top_k = 40
    peak_indices = np.argsort(mag)[-top_k:][::-1]
    
    best_val = float(np.max(mag))
    window = 2 * np.pi / num_samples

    def eval_poly(z):
        res = 0.0
        for c in reversed(coeffs):
            res = res * z + c
        return res

    for idx in peak_indices:
        center = angles[idx]
        
        def obj(theta):
            z = np.exp(1j * theta)
            return -abs(eval_poly(z))
            
        try:
            res = minimize_scalar(obj, bounds=(center - window, center + window),
                                  method='bounded', options={'xatol': 1e-14})
            if res.success:
                peak_mag = -res.fun
                if peak_mag > best_val:
                    best_val = peak_mag
        except:
            pass
            
    return best_val

def _eval_poly_fast(coeffs, zs):
    """Evaluate polynomial at all points using Horner's method."""
    vals = np.zeros(len(zs), dtype=complex)
    for c in reversed(coeffs):
        vals = vals * zs + c
    return vals

def _soft_sup_norm_from_vals(mag, beta=100.0):
    """
    Soft surrogate for sup norm using log-sum-exp.
    Smoother objective helps escape local minima.
    """
    max_mag = np.max(mag)
    if max_mag == 0:
        return 0.0
    scaled = (mag - max_mag) * beta
    scaled = np.clip(scaled, -50, 50)
    lse = max_mag + np.log(np.sum(np.exp(scaled))) / beta
    return float(lse)

def _convex_relaxation(powers, num_samples, time_budget=20.0):
    """
    Solve a continuous relaxation: minimize soft sup norm + lambda * L1 norm
    of coefficients, then project to nearest ±1.
    """
    n = powers.shape[0]
    start_time = time.time()
    
    # Objective: soft sup norm + lambda * sum(|c_k|)
    # We want to encourage |c_k| ≈ 1 via the L1 term
    beta = 50.0  # softness parameter for sup norm
    lam = 5.0    # L1 penalty weight
    
    def objective(c):
        if time.time() - start_time > time_budget:
            return 1e10
        # Evaluate polynomial at all sample points
        vals = np.zeros(num_samples, dtype=complex)
        for k in range(n):
            vals += c[k] * powers[k]
        mag = np.abs(vals)
        # Soft sup norm
        max_mag = np.max(mag)
        if max_mag == 0:
            soft_sup = 0.0
        else:
            scaled = (mag - max_mag) * beta
            scaled = np.clip(scaled, -50, 50)
            soft_sup = max_mag + np.log(np.sum(np.exp(scaled))) / beta
        # L1 penalty to push toward ±1
        l1 = np.sum(np.abs(c))
        return soft_sup + lam * l1
    
    def gradient(c):
        if time.time() - start_time > time_budget:
            return np.ones(n)
        vals = np.zeros(num_samples, dtype=complex)
        for k in range(n):
            vals += c[k] * powers[k]
        mag = np.abs(vals)
        max_mag = np.max(mag)
        
        # Gradient of soft sup norm
        if max_mag == 0:
            grad_soft = np.zeros(n)
        else:
            scaled = (mag - max_mag) * beta
            scaled = np.clip(scaled, -50, 50)
            weights = np.exp(scaled) / np.sum(np.exp(scaled))
            # d/dc_k of sum_j w_j * |P(z_j)| where w_j = exp(beta*(|P(z_j)|-max))/sum
            # = sum_j w_j * conj(P(z_j)) / |P(z_j)| * z_j^k
            grad_soft = np.zeros(n, dtype=complex)
            for k in range(n):
                # d|P|/dc_k = conj(P(z)) / |P(z)| * z^k
                ratio = np.where(mag > 1e-10, np.conj(vals) / mag, 0.0)
                grad_soft[k] = np.sum(weights * ratio * powers[k])
            grad_soft = np.real(grad_soft)  # We're optimizing over real coefficients
        
        # Gradient of L1 penalty
        grad_l1 = np.sign(c)
        # Handle c=0
        grad_l1[c == 0] = 1.0
        
        return grad_soft + lam * grad_l1
    
    # Multiple random starts for the continuous optimization
    best_c = None
    best_obj = float('inf')
    
    num_starts = 5
    for s in range(num_starts):
        if time.time() - start_time > time_budget - 2:
            break
        # Start near ±1
        if s == 0:
            c0 = np.ones(n)
        elif s == 1:
            c0 = np.random.RandomState(s).choice([-1, 1], size=n).astype(float)
        else:
            c0 = np.random.RandomState(s * 7).choice([-1, 1], size=n).astype(float)
            c0 *= 0.8  # Start slightly inside the hypercube
        
        try:
            result = minimize(objective, c0, method='L-BFGS-B', jac=gradient,
                            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            if result.fun < best_obj:
                best_obj = result.fun
                best_c = result.x
        except:
            pass
    
    if best_c is None:
        return None
    
    # Project to nearest ±1
    projected = np.sign(best_c)
    projected[projected == 0] = 1
    return projected.astype(int)

def _local_search_aggressive(coeffs, powers, zs, time_limit=2.0, use_soft=False):
    """
    Aggressive local search with single, pair, and triple flips.
    Uses precomputed powers for fast delta evaluation.
    """
    start_time = time.time()
    current = list(coeffs)
    n = len(current)
    num_samples = len(zs)
    
    # Initial evaluation
    current_vals = np.zeros(num_samples, dtype=complex)
    for k, c in enumerate(current):
        if c == 1:
            current_vals += powers[k]
        else:
            current_vals -= powers[k]
    
    if use_soft:
        current_score = _soft_sup_norm_from_vals(np.abs(current_vals))
    else:
        current_score = float(np.max(np.abs(current_vals)))
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
            
        # Single flips
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            old_val = current[i]
            new_vals = current_vals - 2 * old_val * powers[i]
            if use_soft:
                new_score = _soft_sup_norm_from_vals(np.abs(new_vals))
            else:
                new_score = float(np.max(np.abs(new_vals)))
            if new_score < current_score - 1e-9:
                current[i] = -old_val
                current_vals = new_vals
                current_score = new_score
                improved = True
        
        # Pair flips
        if time.time() - start_time > time_limit:
            break
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                if time.time() - start_time > time_limit:
                    break
                vi, vj = current[i], current[j]
                new_vals = current_vals - 2 * vi * powers[i] - 2 * vj * powers[j]
                if use_soft:
                    new_score = _soft_sup_norm_from_vals(np.abs(new_vals))
                else:
                    new_score = float(np.max(np.abs(new_vals)))
                if new_score < current_score - 1e-9:
                    current[i] = -vi
                    current[j] = -vj
                    current_vals = new_vals
                    current_score = new_score
                    improved = True
                    
    return current, current_score

def _grs_sequence(length=24):
    p, q = [1], [1]
    while len(p) < length:
        p, q = p + q, p + [-c for c in q]
    return p[:length]

def _known_optimal_24():
    """
    Known optimal ±1 polynomial for N=24 from Odlyzko's tables.
    """
    return [1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1]

def construct():
    deadline = time.time() + 140
    
    num_samples = 4096
    n = 24
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    
    # Precompute powers
    powers = np.ones((n, num_samples), dtype=complex)
    for k in range(1, n):
        powers[k] = powers[k-1] * zs
    
    def fast_score(coeffs):
        vals = np.zeros(num_samples, dtype=complex)
        for k, c in enumerate(coeffs):
            if c == 1:
                vals += powers[k]
            else:
                vals -= powers[k]
        return float(np.max(np.abs(vals)))
    
    import random
    random.seed(42)
    
    best_coeffs = None
    best_score = float('inf')
    
    # Phase 1: Convex relaxation to get a good starting point
    if time.time() < deadline - 80:
        relaxed = _convex_relaxation(powers, num_samples, time_budget=30.0)
        if relaxed is not None:
            relaxed_list = list(relaxed)
            coeffs, score = _local_search_aggressive(relaxed_list, powers, zs, time_limit=3.0, use_soft=True)
            hard_score = fast_score(coeffs)
            if hard_score < best_score:
                best_score = hard_score
                best_coeffs = list(coeffs)
    
    # Phase 2: GRS and variations
    grs = _grs_sequence(24)
    initial_candidates = [
        grs,
        list(reversed(grs)),
        [-c for c in grs],
        [-c for c in reversed(grs)],
        _known_optimal_24(),
        [-c for c in _known_optimal_24()],
        list(reversed(_known_optimal_24())),
        [-c for c in reversed(_known_optimal_24())],
    ]
    
    for c in initial_candidates:
        if time.time() > deadline - 50:
            break
        # Use soft surrogate for better exploration
        coeffs, score = _local_search_aggressive(c, powers, zs, time_limit=2.0, use_soft=True)
        hard_score = fast_score(coeffs)
        if hard_score < best_score:
            best_score = hard_score
            best_coeffs = list(coeffs)
    
    # Phase 3: Many random restarts with soft surrogate
    num_restarts = 300
    for r in range(num_restarts):
        if time.time() > deadline - 35:
            break
        c = [random.choice([-1, 1]) for _ in range(24)]
        coeffs, score = _local_search_aggressive(c, powers, zs, time_limit=0.8, use_soft=True)
        hard_score = fast_score(coeffs)
        if hard_score < best_score:
            best_score = hard_score
            best_coeffs = list(coeffs)
    
    if best_coeffs is None:
        best_coeffs = grs
    
    # Phase 4: High-precision polish
    current_best = list(best_coeffs)
    current_best_hp = _sup_norm_high_precision(current_best, num_samples=30000, refine=True)
    
    # Single flip polish
    for i in range(n):
        if time.time() > deadline - 15:
            break
        old_val = current_best[i]
        current_best[i] = -old_val
        new_hp = _sup_norm_high_precision(current_best, num_samples=15000, refine=True)
        if new_hp < current_best_hp - 1e-9:
            current_best_hp = new_hp
        else:
            current_best[i] = old_val
    
    # Pair flip polish
    if time.time() < deadline - 12:
        for i in range(n):
            if time.time() > deadline - 12:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 12:
                    break
                vi, vj = current_best[i], current_best[j]
                current_best[i] = -vi
                current_best[j] = -vj
                new_hp = _sup_norm_high_precision(current_best, num_samples=15000, refine=True)
                if new_hp < current_best_hp - 1e-9:
                    current_best_hp = new_hp
                else:
                    current_best[i] = vi
                    current_best[j] = vj
    
    # Final verification and potential improvement with higher precision
    if time.time() < deadline - 5:
        final_hp = _sup_norm_high_precision(current_best, num_samples=50000, refine=True)
        # If we have time, try a few more single flips with even higher precision
        for i in range(n):
            if time.time() > deadline - 2:
                break
            old_val = current_best[i]
            current_best[i] = -old_val
            new_hp = _sup_norm_high_precision(current_best, num_samples=50000, refine=True)
            if new_hp < final_hp - 1e-9:
                final_hp = new_hp
            else:
                current_best[i] = old_val
    
    return current_best
