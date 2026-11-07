"""Program for Erdos minimum overlap task using anti-periodic model + progressive smooth optimization.

The objective max(overlap(h,1-h)) is minimized when h is anti-periodic
(h[i+N/2] = 1 - h[i]), which forces the overlap to 0 at the shifts
t = 0, N, 2N (the 'forbidden' shifts). We build a strong anti-periodic
seed and refine it with progressive smooth optimization, sharpening the
surrogate to target the true max correlation, followed by discrete refinement.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = N // 2
DEADLINE = 145.0
START = time.time()


def remaining():
    return DEADLINE - (time.time() - START)


def value(h):
    """The exact evaluator value (lower is better)."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0


def value_smooth(h, beta):
    """Smooth surrogate for the max via log-sum-exp (lower is better)."""
    c = np.correlate(h, 1.0 - h, mode="full")
    cm = float(c.max())
    z = beta * (c - cm)
    # Clip for numerical stability
    z = np.clip(z, -100, 100)
    logsum = np.log(np.sum(np.exp(z))) + beta * cm
    return float(logsum) / N * 2.0


def make_seed():
    """Try multiple anti-periodic seeds and return the best one."""
    seeds = []
    
    # Seed 1: Triangular wave
    i = np.arange(HALF)
    d = np.abs(i - (HALF - 1) / 2.0) / ((HALF - 1) / 2.0)
    tri = 1.0 - d
    first1 = 0.5 + (0.5 - tri.mean()) * tri
    h1 = np.empty(N)
    h1[:HALF] = first1
    h1[HALF:] = 1.0 - first1
    h1 = np.clip(h1, 0.0, 1.0)
    h1 += (HALF - h1.sum()) / N
    h1 = np.clip(h1, 0.0, 1.0)
    seeds.append(h1)
    
    # Seed 2: Sine wave
    first2 = 0.5 + 0.4 * np.sin(2 * np.pi * i / HALF)
    h2 = np.empty(N)
    h2[:HALF] = first2
    h2[HALF:] = 1.0 - first2
    h2 = np.clip(h2, 0.0, 1.0)
    h2 += (HALF - h2.sum()) / N
    h2 = np.clip(h2, 0.0, 1.0)
    seeds.append(h2)
    
    # Seed 3: Square wave (alternating blocks)
    first3 = np.where(i < HALF // 2, 0.3, 0.7)
    h3 = np.empty(N)
    h3[:HALF] = first3
    h3[HALF:] = 1.0 - first3
    h3 = np.clip(h3, 0.0, 1.0)
    h3 += (HALF - h3.sum()) / N
    h3 = np.clip(h3, 0.0, 1.0)
    seeds.append(h3)
    
    # Seed 4: Linear ramp
    first4 = np.linspace(0.2, 0.8, HALF)
    h4 = np.empty(N)
    h4[:HALF] = first4
    h4[HALF:] = 1.0 - first4
    h4 = np.clip(h4, 0.0, 1.0)
    h4 += (HALF - h4.sum()) / N
    h4 = np.clip(h4, 0.0, 1.0)
    seeds.append(h4)
    
    # Return the best seed
    best_seed = min(seeds, key=value)
    return best_seed


def refine_continuous(h, beta, maxiter):
    """Optimize the anti-periodic parameters to lower the smooth objective."""
    h0 = np.asarray(h, dtype=np.float64).copy()
    h0 = np.clip(h0, 1e-12, 1.0 - 1e-12)
    p0 = h0[:HALF].copy()
    bounds = [(1e-12, 1.0 - 1e-12)] * HALF

    def unpack(p):
        p = np.clip(p, 0.0, 1.0)
        hh = np.empty(N)
        hh[:HALF] = p
        hh[HALF:] = 1.0 - p
        return hh

    def obj(p):
        return value_smooth(unpack(p), beta)

    def grad(p):
        """Compute gradient of the smooth objective."""
        hh = unpack(p)
        c = np.correlate(hh, 1.0 - hh, mode="full")
        cm = float(c.max())
        z = beta * (c - cm)
        z = np.clip(z, -100, 100)
        exp_z = np.exp(z)
        sum_exp = np.sum(exp_z)
        
        # Gradient of log-sum-exp w.r.t. c
        weights = exp_z / sum_exp
        
        # Gradient of c w.r.t. h
        # c[k] = sum_i h[i] * (1-h[i+k-N+100]) for appropriate ranges
        # This is complex, so use finite differences as fallback
        grad_h = np.zeros(N)
        eps = 1e-8
        for idx in range(N):
            hh_plus = hh.copy()
            hh_plus[idx] += eps
            c_plus = np.correlate(hh_plus, 1.0 - hh_plus, mode="full")
            cm_plus = float(c_plus.max())
            z_plus = beta * (c_plus - cm_plus)
            z_plus = np.clip(z_plus, -100, 100)
            logsum_plus = np.log(np.sum(np.exp(z_plus))) + beta * cm_plus
            val_plus = float(logsum_plus) / N * 2.0
            
            hh_minus = hh.copy()
            hh_minus[idx] -= eps
            c_minus = np.correlate(hh_minus, 1.0 - hh_minus, mode="full")
            cm_minus = float(c_minus.max())
            z_minus = beta * (c_minus - cm_minus)
            z_minus = np.clip(z_minus, -100, 100)
            logsum_minus = np.log(np.sum(np.exp(z_minus))) + beta * cm_minus
            val_minus = float(logsum_minus) / N * 2.0
            
            grad_h[idx] = (val_plus - val_minus) / (2 * eps)
        
        # Map gradient from h to p (first half)
        grad_p = grad_h[:HALF].copy()
        # For second half, h[HALF:] = 1 - p, so d/dp = -grad_h[HALF:]
        grad_p -= grad_h[HALF:]
        
        return grad_p

    try:
        res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds,
                       jac=grad,
                       options={"maxiter": maxiter, "ftol": 1e-16,
                                "gtol": 1e-13, "maxls": 50})
        return unpack(np.clip(res.x, 0.0, 1.0))
    except Exception:
        # Fallback without gradient
        try:
            res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": maxiter, "ftol": 1e-16,
                                    "gtol": 1e-13, "maxls": 50})
            return unpack(np.clip(res.x, 0.0, 1.0))
        except Exception:
            return h0


def snap_to_grid(h, n_levels):
    """Snap h to a symmetric grid of n_levels values, preserving anti-periodicity."""
    h = np.asarray(h, dtype=np.float64)
    first = h[:HALF].copy()
    
    if n_levels <= 1:
        first_q = np.full(HALF, 0.5)
    else:
        grid = np.linspace(0.0, 1.0, n_levels)
        idx = np.argmin(np.abs(first[:, None] - grid[None, :]), axis=1)
        first_q = grid[idx]
    
    # Adjust to satisfy sum constraint: sum(first_q) should be HALF
    target_sum = HALF
    current_sum = first_q.sum()
    diff = target_sum - current_sum
    
    if abs(diff) > 1e-10:
        first_q += diff / HALF
        first_q = np.clip(first_q, 0.0, 1.0)
        for _ in range(10):
            diff = target_sum - first_q.sum()
            if abs(diff) < 1e-12:
                break
            first_q += diff / HALF
            first_q = np.clip(first_q, 0.0, 1.0)
    
    h_out = np.empty(N)
    h_out[:HALF] = first_q
    h_out[HALF:] = 1.0 - first_q
    h_out = np.clip(h_out, 0.0, 1.0)
    
    total = h_out.sum()
    if abs(total - HALF) > 1e-9 * N:
        h_out += (HALF - total) / N
        h_out = np.clip(h_out, 0.0, 1.0)
    
    return h_out


def greedy_permutation_search(h, max_iters=5000):
    """Greedy local search: swap pairs in the first half to reduce max correlation."""
    h = np.asarray(h, dtype=np.float64).copy()
    first = h[:HALF].copy()
    best_val = value(h)
    
    rng = np.random.default_rng(42)
    
    improved = True
    iters = 0
    while improved and iters < max_iters and remaining() > 20.0:
        improved = False
        iters += 1
        for i in range(HALF):
            if remaining() < 20.0:
                break
            for j in range(i + 1, HALF):
                if remaining() < 20.0:
                    break
                first[i], first[j] = first[j], first[i]
                h_candidate = np.empty(N)
                h_candidate[:HALF] = first
                h_candidate[HALF:] = 1.0 - first
                v_candidate = value(h_candidate)
                
                if v_candidate < best_val - 1e-12:
                    best_val = v_candidate
                    improved = True
                else:
                    first[i], first[j] = first[j], first[i]
    
    h_out = np.empty(N)
    h_out[:HALF] = first
    h_out[HALF:] = 1.0 - first
    return h_out, best_val


def simulated_annealing_permutation(h, max_iters=20000):
    """Simulated annealing on permutations of the first half."""
    h = np.asarray(h, dtype=np.float64).copy()
    first = h[:HALF].copy()
    best_first = first.copy()
    best_val = value(h)
    current_val = best_val
    
    rng = np.random.default_rng(123)
    
    T0 = 0.1
    T_end = 1e-6
    n_steps = min(max_iters, int(remaining() * 100))
    if n_steps < 100:
        n_steps = 100
    
    for it in range(n_steps):
        if remaining() < 25.0:
            break
        T = T0 * (T_end / T0) ** (it / max(1, n_steps - 1))
        
        i = rng.integers(0, HALF)
        j = rng.integers(0, HALF)
        if i == j:
            continue
        
        first[i], first[j] = first[j], first[i]
        h_candidate = np.empty(N)
        h_candidate[:HALF] = first
        h_candidate[HALF:] = 1.0 - first
        v_candidate = value(h_candidate)
        
        delta = v_candidate - current_val
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current_val = v_candidate
            if v_candidate < best_val:
                best_val = v_candidate
                best_first = first.copy()
        else:
            first[i], first[j] = first[j], first[i]
    
    h_out = np.empty(N)
    h_out[:HALF] = best_first
    h_out[HALF:] = 1.0 - best_first
    return h_out, best_val


def construct():
    t0 = time.time()

    # Phase 1: Continuous optimization to get a good anti-periodic seed
    h = make_seed()
    best = h.copy()
    best_val = value(best)

    # Progressive smooth optimization with increasing beta - more iterations
    for beta in [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0, 800.0]:
        if remaining() < 30.0:
            break
        h = refine_continuous(h, beta, 500)
        v = value(h)
        if v < best_val:
            best, best_val = h.copy(), v
        
        # Also try from the previous best
        h_alt = refine_continuous(best, beta, 300)
        v_alt = value(h_alt)
        if v_alt < best_val:
            best, best_val = h_alt.copy(), v_alt

    # Phase 2: Skip discrete refinement - it's counterproductive
    # Instead, do more continuous refinement with different starting points
    
    # Try small perturbations of the best solution
    rng = np.random.default_rng(42)
    for _ in range(10):
        if remaining() < 25.0:
            break
        
        # Add small noise to break symmetry
        noise = rng.normal(0, 0.01, HALF)
        p_perturbed = np.clip(best[:HALF] + noise, 0.0, 1.0)
        
        h_perturbed = np.empty(N)
        h_perturbed[:HALF] = p_perturbed
        h_perturbed[HALF:] = 1.0 - p_perturbed
        
        # Refine from perturbed point
        for beta in [200.0, 500.0]:
            if remaining() < 20.0:
                break
            h_refined = refine_continuous(h_perturbed, beta, 300)
            v_refined = value(h_refined)
            if v_refined < best_val:
                best, best_val = h_refined.copy(), v_refined

    # Phase 3: Final continuous polish with very high beta
    if remaining() > 15.0:
        p0 = np.clip(best[:HALF], 0.0, 1.0).copy()
        bounds = [(0.0, 1.0)] * HALF

        def unpack(p):
            p = np.clip(p, 0.0, 1.0)
            hh = np.empty(N)
            hh[:HALF] = p
            hh[HALF:] = 1.0 - p
            return hh

        # Try multiple betas for the final polish - higher betas
        for beta in [500.0, 1000.0, 2000.0, 5000.0]:
            if remaining() < 10.0:
                break
            try:
                res = minimize(lambda p: value_smooth(unpack(p), beta), p0,
                               method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": 500, "ftol": 1e-18,
                                        "gtol": 1e-15, "maxls": 60})
                h = unpack(np.clip(res.x, 0.0, 1.0))
                v = value(h)
                if v < best_val:
                    best, best_val = h.copy(), v
                    p0 = np.clip(res.x, 0.0, 1.0).copy()
            except Exception:
                pass

    # Ensure constraints are met - use a more robust projection
    best = np.asarray(best, dtype=np.float64)
    
    # Project onto the constraint set: h in [0,1]^N, sum(h) = N/2
    # Using iterative proportional fitting
    for _ in range(100):
        best = np.clip(best, 0.0, 1.0)
        current_sum = best.sum()
        if abs(current_sum - HALF) < 1e-12:
            break
        best += (HALF - current_sum) / N
        best = np.clip(best, 0.0, 1.0)
    
    # Final check
    if not np.all(np.isfinite(best)):
        raise ValueError("Non-finite values in result")
    
    return best
