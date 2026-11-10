"""
Flattest plus-minus-one polynomials with 24 coefficients (minimum sup norm).
Combines fast precomputed-evaluation local search (single + pair + triple flips)
with custom golden-section refinement for the final score.
"""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START

# --- Precomputation ---
_N_SAMPLES = 4096
_Z = np.exp(2j * np.pi * np.arange(_N_SAMPLES) / _N_SAMPLES)

# Precompute z^k for k=0..23 at all sample points
_z_powers = np.ones((24, _N_SAMPLES), dtype=complex)
for _k in range(1, 24):
    _z_powers[_k] = _z_powers[_k - 1] * _Z


def _eval_poly(coeffs, z):
    """Evaluate polynomial P(z) = sum c_k z^k at complex z using Horner's method."""
    result = 0.0
    for c in reversed(coeffs):
        result = result * z + c
    return result


def _golden_section_max(f, a, b, tol=1e-14, max_iter=200):
    """
    Golden-section search for the maximum of f on [a, b].
    Returns (x_max, f_max).
    """
    gr = (np.sqrt(5.0) - 1.0) / 2.0  # golden ratio ~0.618
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    
    for _ in range(max_iter):
        if b - a < tol:
            break
        if fc < fd:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = f(d)
        else:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = f(c)
    
    x_max = (a + b) / 2.0
    f_max = f(x_max)
    return x_max, f_max


def _sup_norm_refined(coeffs, num_samples=8192):
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

    top_k = 16
    if num_samples > top_k:
        peak_indices = np.argsort(mag)[-top_k:][::-1]
    else:
        peak_indices = np.arange(num_samples)

    best_val = float(np.max(mag))
    window = 2 * np.pi / num_samples

    for idx in peak_indices:
        center = angles[idx] % (2 * np.pi)

        def obj(theta):
            z = np.exp(1j * theta)
            return abs(_eval_poly(coeffs, z))

        lower = center - window
        upper = center + window

        try:
            _, peak_mag = _golden_section_max(obj, lower, upper, tol=1e-14, max_iter=150)
            if peak_mag > best_val:
                best_val = peak_mag
        except Exception:
            pass

    return best_val


def _score_fast(coeffs, num_samples=_N_SAMPLES):
    """Fast score: max |P(z)| / sqrt(n) using precomputed roots."""
    n = len(coeffs)
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * _Z[:num_samples] + c
    return float(np.max(np.abs(vals))) / np.sqrt(n)


def _local_search(coeffs, time_limit=3.0, num_samples=_N_SAMPLES, max_triple=50):
    """
    Local search with single, pair, and limited triple flips.
    Uses fast precomputed evaluation.
    Returns (coeffs, score).
    """
    start = time.time()
    current = list(coeffs)
    n = len(current)
    zp = _z_powers[:n]

    # Precompute current values at all sample points
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(current):
        vals = vals * _Z[:num_samples] + c
    current_score = float(np.max(np.abs(vals))) / np.sqrt(n)

    improved = True
    while improved:
        improved = False
        if time.time() - start > time_limit:
            break

        # Single flips
        for i in range(n):
            if time.time() - start > time_limit:
                break
            delta = -4.0 * current[i] * np.real(vals * np.conj(zp[i])) + 4.0
            new_mag_sq = np.abs(vals) ** 2 + delta
            new_max = float(np.max(new_mag_sq))
            new_score = np.sqrt(new_max) / np.sqrt(n)
            if new_score < current_score - 1e-12:
                current[i] = -current[i]
                vals = vals - 2.0 * current[i] * zp[i]
                current_score = new_score
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
                new_max = float(np.max(np.abs(new_vals) ** 2))
                new_score = np.sqrt(new_max) / np.sqrt(n)
                if new_score < current_score - 1e-12:
                    current[i] = -current[i]
                    current[j] = -current[j]
                    vals = new_vals
                    current_score = new_score
                    improved = True

        # Triple flips (limited for speed)
        if time.time() - start > time_limit:
            break
        triple_count = 0
        for i in range(n):
            if time.time() - start > time_limit or triple_count >= max_triple:
                break
            for j in range(i + 1, n):
                if time.time() - start > time_limit or triple_count >= max_triple:
                    break
                for k in range(j + 1, n):
                    if time.time() - start > time_limit or triple_count >= max_triple:
                        break
                    new_vals = vals - 2.0 * current[i] * zp[i] - 2.0 * current[j] * zp[j] - 2.0 * current[k] * zp[k]
                    new_max = float(np.max(np.abs(new_vals) ** 2))
                    new_score = np.sqrt(new_max) / np.sqrt(n)
                    if new_score < current_score - 1e-12:
                        current[i] = -current[i]
                        current[j] = -current[j]
                        current[k] = -current[k]
                        vals = new_vals
                        current_score = new_score
                        improved = True
                        triple_count += 1

    return current, current_score


def _tabu_search(coeffs, time_limit=3.0, num_samples=_N_SAMPLES, tabu_tenure=8, max_iter=200):
    """
    Tabu search to escape local minima.
    """
    start = time.time()
    current = list(coeffs)
    n = len(current)
    zp = _z_powers[:n]

    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(current):
        vals = vals * _Z[:num_samples] + c
    current_score = float(np.max(np.abs(vals))) / np.sqrt(n)
    
    best_coeffs = list(current)
    best_score = current_score

    tabu = [0] * n  # iteration when each position became tabu
    it = 0

    while it < max_iter:
        if time.time() - start > time_limit:
            break
        it += 1

        best_move = None
        best_move_score = current_score
        best_move_indices = None

        # Evaluate all single flips
        for i in range(n):
            if it - tabu[i] < tabu_tenure:
                continue  # tabu
            delta = -4.0 * current[i] * np.real(vals * np.conj(zp[i])) + 4.0
            new_mag_sq = np.abs(vals) ** 2 + delta
            new_max = float(np.max(new_mag_sq))
            new_score = np.sqrt(new_max) / np.sqrt(n)
            if new_score < best_move_score:
                best_move_score = new_score
                best_move = i
                best_move_indices = [i]

        # Evaluate some pair flips
        for i in range(n):
            for j in range(i + 1, n):
                if it - tabu[i] < tabu_tenure or it - tabu[j] < tabu_tenure:
                    continue
                new_vals = vals - 2.0 * current[i] * zp[i] - 2.0 * current[j] * zp[j]
                new_max = float(np.max(np.abs(new_vals) ** 2))
                new_score = np.sqrt(new_max) / np.sqrt(n)
                if new_score < best_move_score:
                    best_move_score = new_score
                    best_move = (i, j)
                    best_move_indices = [i, j]

        if best_move is None:
            break

        # Apply best move
        if isinstance(best_move, int):
            current[best_move] = -current[best_move]
            vals = vals - 2.0 * current[best_move] * zp[best_move]
            tabu[best_move] = it
        else:
            i, j = best_move
            current[i] = -current[i]
            current[j] = -current[j]
            vals = vals - 2.0 * current[i] * zp[i] - 2.0 * current[j] * zp[j]
            tabu[i] = it
            tabu[j] = it

        current_score = best_move_score
        if current_score < best_score:
            best_score = current_score
            best_coeffs = list(current)

    return best_coeffs, best_score


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


def construct():
    deadline = time.time() + 115  # Safety margin before 150s kill

    # Initialize with GRS and its variants
    grs = _grs_sequence(24)
    
    # Generate diverse initial candidates
    candidates = []
    candidates.extend(_all_reversals(grs))
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
        if time.time() > deadline - 50:
            break
        score = _score_fast(cand)
        if score < best_score:
            best_score = score
            best_coeffs = list(cand)

    # Phase 2: Local search from top candidates (hard max with triple flips)
    scored = [(c, _score_fast(c)) for c in unique_candidates]
    scored.sort(key=lambda x: x[1])
    
    for c, s in scored[:20]:
        if time.time() > deadline - 45:
            break
        coeffs, score = _local_search(c, time_limit=2.0, max_triple=100)
        if score < best_score:
            best_score = score
            best_coeffs = list(coeffs)

    # Phase 3: Tabu search from top candidates
    for c, s in scored[:10]:
        if time.time() > deadline - 35:
            break
        coeffs, score = _tabu_search(c, time_limit=2.5)
        if score < best_score:
            best_score = score
            best_coeffs = list(coeffs)

    # Phase 4: Random restarts with local search
    while time.time() < deadline - 20:
        coeffs = [random.choice([-1, 1]) for _ in range(24)]
        coeffs, score = _local_search(coeffs, time_limit=3.0, max_triple=50)
        if score < best_score:
            best_score = score
            best_coeffs = list(coeffs)

    # Phase 5: Perturbation + local search from best
    if time.time() < deadline - 12:
        for _ in range(30):
            if time.time() > deadline - 8:
                break
            perturbed = list(best_coeffs)
            n_flips = random.randint(1, 4)
            for _ in range(n_flips):
                idx = random.randint(0, 23)
                perturbed[idx] = -perturbed[idx]
            coeffs, score = _local_search(perturbed, time_limit=1.5, max_triple=30)
            if score < best_score:
                best_score = score
                best_coeffs = list(coeffs)

    # Phase 6: Tabu search from best
    if time.time() < deadline - 6:
        coeffs, score = _tabu_search(best_coeffs, time_limit=min(4.0, deadline - time.time() - 3))
        if score < best_score:
            best_score = score
            best_coeffs = list(coeffs)

    # Final refinement of the best candidate
    final_score = _sup_norm_refined(best_coeffs, num_samples=8192) / np.sqrt(24)

    return best_coeffs

# EVOLVE-BLOCK-END
