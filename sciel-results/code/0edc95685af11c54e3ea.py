import numpy as np
import random
import time
# EVOLVE-BLOCK-START

# --- Precomputation ---
_N_SAMPLES = 2048
_Z = np.exp(2j * np.pi * np.arange(_N_SAMPLES) / _N_SAMPLES)

# Precompute z^k for k=0..23 at all sample points
_z_powers = np.ones((24, _N_SAMPLES), dtype=complex)
for _k in range(1, 24):
    _z_powers[_k] = _z_powers[_k - 1] * _Z

# Precompute real and imag parts for faster dot products in local search
_zp_real = _z_powers.real
_zp_imag = _z_powers.imag


def _eval_poly(coeffs, z):
    """Evaluate polynomial P(z) = sum c_k z^k at complex z using Horner's method."""
    result = 0.0
    for c in reversed(coeffs):
        result = result * z + c
    return result


def _eval_all(coeffs, num_samples=_N_SAMPLES):
    """Evaluate P at all sample points. Returns complex array."""
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * _Z[:num_samples] + c
    return vals


def _refine_peak(coeffs, center, window):
    """
    Refine the peak of |P(e^{i theta})| near 'center' within +/- window.
    Uses a combination of grid search and Newton's method.
    """
    # 1. Coarse grid search to locate the peak more precisely
    n_grid = 100
    thetas = np.linspace(center - window, center + window, n_grid)
    zs = np.exp(1j * thetas)
    
    # Evaluate P(z) for all thetas efficiently? 
    # Horner's rule in a loop is O(N * n_grid). N=24, n_grid=100. 2400 ops. Fast.
    mags = np.zeros(n_grid)
    for k in range(n_grid):
        mags[k] = abs(_eval_poly(coeffs, zs[k]))
    
    max_idx = np.argmax(mags)
    
    # 2. Parabolic interpolation or Newton's method
    # Let's use Newton's method on f(theta) = |P(e^{i theta})|^2
    # f'(theta) = 2 Re( P'(e^{i theta}) * i * e^{i theta} * conj(P(e^{i theta})) )
    # This requires P'(z).
    # Alternatively, use a local quadratic fit.
    
    # If max_idx is at boundary, we might need to expand, but window is small.
    # Let's use a fine grid + bisection on the derivative if possible, 
    # or just a very fine grid.
    # Given the small window (2*pi/4096 ~ 0.0015), a fine grid of 1000 points is cheap.
    
    # Let's try a simple high-resolution grid + parabolic fit
    if max_idx == 0 or max_idx == n_grid - 1:
        # Peak is at edge, try expanding or just return edge
        # For safety, just return the best grid point
        return thetas[max_idx], mags[max_idx]
        
    # Parabolic interpolation
    x0, x1, x2 = thetas[max_idx-1], thetas[max_idx], thetas[max_idx+1]
    y0, y1, y2 = mags[max_idx-1], mags[max_idx], mags[max_idx+1]
    
    # Vertex of parabola through (x0,y0), (x1,y1), (x2,y2)
    # x_v = x1 - 0.5 * ( (y0-y2)/(y0-2*y1+y2) * (x0-x2) )
    denom = y0 - 2*y1 + y2
    if abs(denom) < 1e-12:
        theta_best = x1
        mag_best = y1
    else:
        theta_best = x1 - 0.5 * ((y0 - y2) / denom) * (x0 - x2)
        # Ensure theta_best is within [x0, x2]
        if theta_best < x0 or theta_best > x2:
            theta_best = x1
            mag_best = y1
        else:
            mag_best = abs(_eval_poly(coeffs, np.exp(1j * theta_best)))
            
    return theta_best, mag_best


def _sup_norm_refined(coeffs, num_samples=4096):
    """
    Compute sup norm with golden-section refinement of top peaks.
    """
    n = len(coeffs)
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)

    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * zs + c

    mag = np.abs(vals)

    top_k = 10
    if num_samples > top_k:
        peak_indices = np.argsort(mag)[-top_k:][::-1]
    else:
        peak_indices = np.arange(num_samples)

    best_val = float(np.max(mag))
    window = 2 * np.pi / num_samples

    for idx in peak_indices:
        center = angles[idx] % (2 * np.pi)
        
        # Handle wrap-around if needed, but window is small so it's fine
        theta_peak, mag_peak = _refine_peak(coeffs, center, window)
        
        if mag_peak > best_val:
            best_val = mag_peak

    return best_val


def _score_fast(coeffs, num_samples=_N_SAMPLES):
    """Fast score: max |P(z)| / sqrt(n) using precomputed roots."""
    n = len(coeffs)
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * _Z[:num_samples] + c
    return float(np.max(np.abs(vals))) / np.sqrt(n)


def _local_search(coeffs, time_limit=3.0, num_samples=_N_SAMPLES):
    """
    Local search with single, pair, and triple flips using fast precomputed evaluation.
    Returns (coeffs, score).
    """
    start = time.time()
    current = list(coeffs)
    n = len(current)
    zp = _z_powers[:n]
    zp_r = _zp_real[:n]
    zp_i = _zp_imag[:n]

    # Precompute current values at all sample points
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(current):
        vals = vals * _Z[:num_samples] + c
    vals_r = vals.real
    vals_i = vals.imag
    current_score_sq = float(np.max(vals_r**2 + vals_i**2))

    improved = True
    while improved:
        improved = False
        if time.time() - start > time_limit:
            break

        # Single flips
        for i in range(n):
            if time.time() - start > time_limit:
                break
            re_part = vals_r * zp_r[i] + vals_i * zp_i[i]
            delta = -4.0 * current[i] * re_part + 4.0
            new_mag_sq = (vals_r**2 + vals_i**2) + delta
            new_max_sq = float(np.max(new_mag_sq))
            if new_max_sq < current_score_sq - 1e-12:
                current[i] = -current[i]
                vals = vals + 2.0 * current[i] * zp[i]
                vals_r = vals.real
                vals_i = vals.imag
                current_score_sq = new_max_sq
                improved = True

        # Pair flips
        if time.time() - start > time_limit:
            break
        for i in range(n):
            if time.time() - start > time_limit:
                break
            for j in range(i + 1, n):
                if time.time() - start > time_limit:
                    break
                new_vals = vals - 2.0 * current[i] * zp[i] - 2.0 * current[j] * zp[j]
                new_max_sq = float(np.max(new_vals.real**2 + new_vals.imag**2))
                
                if new_max_sq < current_score_sq - 1e-12:
                    current[i] = -current[i]
                    current[j] = -current[j]
                    vals = new_vals
                    vals_r = vals.real
                    vals_i = vals.imag
                    current_score_sq = new_max_sq
                    improved = True
        
        # Triple flips (only if time remains)
        if time.time() - start > time_limit - 0.5:
            continue
        for i in range(n):
            if time.time() - start > time_limit:
                break
            for j in range(i + 1, n):
                if time.time() - start > time_limit:
                    break
                for k in range(j + 1, n):
                    if time.time() - start > time_limit:
                        break
                    new_vals = vals - 2.0 * current[i] * zp[i] - 2.0 * current[j] * zp[j] - 2.0 * current[k] * zp[k]
                    new_max_sq = float(np.max(new_vals.real**2 + new_vals.imag**2))
                    
                    if new_max_sq < current_score_sq - 1e-12:
                        current[i] = -current[i]
                        current[j] = -current[j]
                        current[k] = -current[k]
                        vals = new_vals
                        vals_r = vals.real
                        vals_i = vals.imag
                        current_score_sq = new_max_sq
                        improved = True

    current_score = np.sqrt(current_score_sq) / np.sqrt(n)
    return current, current_score


def _grs_sequence(length=24):
    """Generate Golay-Rudin-Shapiro sequence truncated to given length."""
    p, q = [1], [1]
    while len(p) < length:
        p, q = p + q, p + [-c for c in q]
    return p[:length]


def _all_reversals(coeffs):
    """Generate all reversals of the coefficient sequence."""
    n = len(coeffs)
    results = [list(coeffs)]
    for k in range(1, n):
        rev = coeffs[:k] + list(reversed(coeffs[k:]))
        results.append(list(rev))
    return results


def _all_shifts_and_reversals(coeffs):
    """Generate all cyclic shifts and their reversals."""
    n = len(coeffs)
    variants = []
    for shift in range(n):
        rotated = coeffs[shift:] + coeffs[:shift]
        variants.append(list(rotated))
        variants.append(list(reversed(rotated)))
    return variants


def _soft_max_score(coeffs, num_samples=_N_SAMPLES, beta=10.0):
    """
    Soft surrogate for max |P(z)| using log-sum-exp.
    This provides a smooth objective that approximates the max for large beta.
    """
    n = len(coeffs)
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * _Z[:num_samples] + c
    mag = np.abs(vals)
    # log-sum-exp for numerical stability
    m = np.max(mag)
    log_sum = m + np.log(np.sum(np.exp((mag - m) * beta)))
    # Approximate max: (1/beta) * log(sum(exp(beta * x)))
    soft_max = log_sum / beta
    return soft_max / np.sqrt(n)


def _local_search_soft(coeffs, time_limit=3.0, num_samples=_N_SAMPLES, beta=10.0):
    """
    Local search using soft-max surrogate for smoother landscape.
    """
    start = time.time()
    current = list(coeffs)
    n = len(current)
    zp = _z_powers[:n]

    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(current):
        vals = vals * _Z[:num_samples] + c
    current_score = _soft_max_score(current, num_samples, beta)

    improved = True
    while improved:
        improved = False
        if time.time() - start > time_limit:
            break

        # Single flips
        for i in range(n):
            if time.time() - start > time_limit:
                break
            old_c = current[i]
            current[i] = -old_c
            # Recompute vals incrementally
            vals = vals - 2.0 * old_c * zp[i]
            new_score = _soft_max_score(current, num_samples, beta)
            if new_score < current_score - 1e-12:
                current_score = new_score
                improved = True
            else:
                current[i] = old_c
                vals = vals + 2.0 * old_c * zp[i]

        # Pair flips
        if time.time() - start > time_limit:
            break
        for i in range(n):
            if time.time() - start > time_limit:
                break
            for j in range(i + 1, n):
                if time.time() - start > time_limit:
                    break
                ci, cj = current[i], current[j]
                current[i] = -ci
                current[j] = -cj
                vals = vals - 2.0 * ci * zp[i] - 2.0 * cj * zp[j]
                new_score = _soft_max_score(current, num_samples, beta)
                if new_score < current_score - 1e-12:
                    current_score = new_score
                    improved = True
                else:
                    current[i] = ci
                    current[j] = cj
                    vals = vals + 2.0 * ci * zp[i] + 2.0 * cj * zp[j]

    # Return with hard max score for consistency
    final_score = _score_fast(current, num_samples)
    return current, final_score


def _simulated_annealing(coeffs, time_limit=5.0, num_samples=_N_SAMPLES):
    """
    Simulated Annealing to escape local minima.
    """
    start = time.time()
    current = list(coeffs)
    n = len(current)
    zp = _z_powers[:n]

    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(current):
        vals = vals * _Z[:num_samples] + c
    
    current_score = _score_fast(current, num_samples)
    best_coeffs = list(current)
    best_score = current_score
    
    T = 0.1 # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.95
    
    while time.time() - start < time_limit and T > T_min:
        # Random flip
        i = random.randint(0, n - 1)
        old_c = current[i]
        current[i] = -old_c
        vals = vals - 2.0 * old_c * zp[i]
        
        new_score = _score_fast(current, num_samples)
        delta = new_score - current_score
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_coeffs = list(current)
        else:
            # Revert
            current[i] = old_c
            vals = vals + 2.0 * old_c * zp[i]
            
        T *= cooling_rate
        
    return best_coeffs, best_score


def construct():
    deadline = time.time() + 115  # Safety margin before 150s kill

    # Initialize with GRS and its variants
    grs = _grs_sequence(24)
    
    # Generate diverse initial candidates
    candidates = []
    
    # GRS and its reversals
    candidates.extend(_all_reversals(grs))
    
    # GRS shifts and reversals
    candidates.extend(_all_shifts_and_reversals(grs))
    
    # Negated versions
    for cand in list(candidates):
        candidates.append([-c for c in cand])
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(list(c))
    
    best_coeffs = None
    best_score = float('inf')

    # Phase 1: Quick screening with fast scoring
    for cand in unique_candidates:
        if time.time() > deadline - 40:
            break
        score = _score_fast(cand)
        if score < best_score:
            best_score = score
            best_coeffs = list(cand)

    # Phase 2: Local search from top candidates (hard max)
    scored = [(c, _score_fast(c)) for c in unique_candidates]
    scored.sort(key=lambda x: x[1])
    
    for c, s in scored[:15]:
        if time.time() > deadline - 35:
            break
        coeffs, score = _local_search(c, time_limit=2.0)
        if score < best_score:
            best_score = score
            best_coeffs = list(coeffs)

    # Phase 3: Soft-max local search from best and top candidates
    for c, s in scored[:8]:
        if time.time() > deadline - 30:
            break
        coeffs, score = _local_search_soft(c, time_limit=2.0, beta=10.0)
        if score < best_score:
            best_score = score
            best_coeffs = list(coeffs)

    # Phase 4: Simulated Annealing restarts
    while time.time() < deadline - 20:
        coeffs = [random.choice([-1, 1]) for _ in range(24)]
        coeffs, score = _simulated_annealing(coeffs, time_limit=3.0)
        if score < best_score:
            best_score = score
            best_coeffs = list(coeffs)
        
        # Occasionally restart from best with perturbation
        if random.random() < 0.3:
            perturbed = list(best_coeffs)
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, 23)
                perturbed[idx] = -perturbed[idx]
            coeffs, score = _simulated_annealing(perturbed, time_limit=3.0)
            if score < best_score:
                best_score = score
                best_coeffs = list(coeffs)

    # Phase 5: Final polish with local search
    if time.time() < deadline - 5:
        coeffs, score = _local_search(best_coeffs, time_limit=min(5.0, deadline - time.time() - 2))
        if score < best_score:
            best_score = score
            best_coeffs = list(coeffs)

    # Final Selection: Compare top candidates using refined score
    final_score = _sup_norm_refined(best_coeffs, num_samples=4096) / np.sqrt(24)

    return best_coeffs

# EVOLVE-BLOCK-END
