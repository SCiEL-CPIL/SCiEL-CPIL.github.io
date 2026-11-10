import time
import numpy as np
import random

_START_TIME = None

def _eval_poly_vec(coeffs, angles):
    """Evaluate P(z) = sum c_k z^k at z = exp(i*angle) for array of angles."""
    n = len(coeffs)
    # z^k for k=0..n-1, shape (len(angles), n)
    # Use broadcasting: angles[:,None] * k[None,:]
    k = np.arange(n)
    phases = np.exp(1j * np.outer(angles, k))
    return phases @ coeffs

def _sup_norm(coeffs, n_samples=4096):
    """Compute sup norm of P(z) over |z|=1 via dense sampling + golden-section refinement."""
    angles = 2 * np.pi * np.arange(n_samples) / n_samples
    vals = _eval_poly_vec(coeffs, angles)
    mag = np.abs(vals)
    
    # Find local maxima in the sampled magnitudes
    # Use a simple approach: find top peaks and refine
    # Get indices of top values
    top_k = min(16, n_samples)
    top_idx = np.argpartition(mag, -top_k)[-top_k:]
    
    best = 0.0
    for idx in top_idx:
        # Golden-section refinement around this angle
        a = (idx - 1) * 2 * np.pi / n_samples
        b = (idx + 1) * 2 * np.pi / n_samples
        # Handle wrap-around
        if a < 0:
            a += 2 * np.pi
        if b > 2 * np.pi:
            b -= 2 * np.pi
        
        # If wrap-around, just use a small window
        if b < a:
            # Handle wrap: split into two intervals
            # For simplicity, just use the center
            center = angles[idx]
            best = max(best, abs(_eval_poly_vec(coeffs, [center])[0]))
            continue
        
        # Golden section search
        gr = (np.sqrt(5) - 1) / 2  # golden ratio conjugate
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = abs(_eval_poly_vec(coeffs, [c])[0])
        fd = abs(_eval_poly_vec(coeffs, [d])[0])
        
        for _ in range(50):
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = abs(_eval_poly_vec(coeffs, [c])[0])
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = abs(_eval_poly_vec(coeffs, [d])[0])
        
        mid = (a + b) / 2
        val = abs(_eval_poly_vec(coeffs, [mid])[0])
        best = max(best, val)
    
    return best

def _score(coeffs):
    """Compute the score: sup norm / sqrt(24)."""
    return _sup_norm(coeffs) / np.sqrt(24)

def _gr_sequence(n):
    """Generate Golay-Rudin-Shapiro sequence of length n."""
    p, q = [1], [1]
    while len(p) < n:
        p, q = p + q, p + [-c for c in q]
    return p[:n]

def _local_search(coeffs, max_iters=5000, time_limit=None):
    """Local search: flip individual coefficients to reduce sup norm."""
    global _START_TIME
    best_coeffs = list(coeffs)
    best_score = _score(best_coeffs)
    n = len(best_coeffs)
    
    rng = random.Random(42)
    order = list(range(n))
    
    iter_count = 0
    while iter_count < max_iters:
        # Check time
        if time_limit is not None and time.time() - _START_TIME > time_limit:
            break
        
        improved = False
        rng.shuffle(order)
        for i in order:
            if time_limit is not None and time.time() - _START_TIME > time_limit:
                break
            
            # Try flipping coefficient i
            old_val = best_coeffs[i]
            best_coeffs[i] = -old_val
            new_score = _score(best_coeffs)
            
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                best_coeffs[i] = old_val
        
        iter_count += 1
        if not improved:
            # Try random restarts from neighborhood
            # Pick a few random flips
            for _ in range(3):
                if time_limit is not None and time.time() - _START_TIME > time_limit:
                    break
                idx = rng.randint(0, n - 1)
                best_coeffs[idx] = -best_coeffs[idx]
                new_score = _score(best_coeffs)
                if new_score < best_score:
                    best_score = new_score
                else:
                    best_coeffs[idx] = -best_coeffs[idx]
    
    return best_coeffs, best_score

def construct():
    global _START_TIME
    _START_TIME = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
    n = 24
    best_coeffs = None
    best_score = float('inf')
    
    # Generate multiple initial candidates
    candidates = []
    
    # 1. Golay-Rudin-Shapiro
    gr = _gr_sequence(n)
    candidates.append(gr)
    
    # 2. All +1s (baseline)
    candidates.append([1] * n)
    
    # 3. Alternating
    candidates.append([1 if i % 2 == 0 else -1 for i in range(n)])
    
    # 4. Random candidates
    rng = random.Random(12345)
    for _ in range(20):
        candidates.append([rng.choice([1, -1]) for _ in range(n)])
    
    # 5. Known good patterns - try some structured ones
    # Barker-like: +1 -1 -1 +1 -1 -1 +1 (repeated)
    bark = [1, -1, -1, 1, -1, -1, 1]
    candidates.append((bark * 4)[:n])
    
    # 6. Another pattern
    pat = [1, 1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1]
    candidates.append((pat * 2)[:n])
    
    # Evaluate all candidates
    for cand in candidates:
        if len(cand) != n:
            continue
        s = _score(cand)
        if s < best_score:
            best_score = s
            best_coeffs = list(cand)
    
    # Local search from the best candidate
    if best_coeffs is not None:
        remaining_time = DEADLINE - (time.time() - _START_TIME)
        if remaining_time > 5:
            best_coeffs, best_score = _local_search(best_coeffs, max_iters=2000, time_limit=DEADLINE)
    
    # Try local search from a few other good candidates
    for cand in candidates[:5]:
        if time.time() - _START_TIME > DEADLINE - 10:
            break
        s = _score(cand)
        if s < best_score + 0.05:  # Only if close to best
            remaining_time = DEADLINE - (time.time() - _START_TIME)
            if remaining_time > 5:
                c2, s2 = _local_search(cand, max_iters=1000, time_limit=DEADLINE)
                if s2 < best_score:
                    best_score = s2
                    best_coeffs = c2
    
    # Final check
    if time.time() - _START_TIME > DEADLINE:
        pass  # Just return what we have
    
    if best_coeffs is None:
        best_coeffs = [1] * n
    
    return best_coeffs
