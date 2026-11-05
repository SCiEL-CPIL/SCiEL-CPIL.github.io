import numpy as np
import time

def _eval_at(c, t):
    """Evaluate |P(exp(2j*pi*t))| using Horner's method."""
    z = np.exp(2j * np.pi * t)
    acc = 0j
    for coeff in reversed(c):
        acc = acc * z + coeff
    return np.abs(acc)

def _peak(c):
    """
    Estimate the max |P(z)| on the unit circle.
    Uses dense FFT sampling and golden-section refinement of top peaks.
    """
    n = len(c)
    N = 8192  # Increased resolution for safety
    cf = np.zeros(N, dtype=complex)
    cf[:n] = c
    # F[k] = sum_j c_j exp(2*pi*i*j*k/N) = P(exp(2*pi*i*k/N))
    F = np.fft.ifft(cf) * N
    mags = np.abs(F)
    
    # Find top 10 peaks
    top_k = 10
    if N < top_k:
        top_k = N
    idx = np.argpartition(mags, -top_k)[-top_k:]
    
    best_peak = mags[idx].max()
    
    # Refine each top peak
    for k in idx:
        # Search window around k/N
        # k is in [0, N-1]
        lo = max(0.0, (k - 5) / N)
        hi = min(1.0, (k + 6) / N)
        
        # Golden section maximization
        gr = (np.sqrt(5) - 1) / 2
        a, b = lo, hi
        c1 = b - gr * (b - a)
        c2 = a + gr * (b - a)
        f1 = _eval_at(c, c1)
        f2 = _eval_at(c, c2)
        
        for _ in range(50):  # ~50 iterations is enough for 1e-9 accuracy in this range
            if f1 < f2:
                a = c1
                c1 = c2
                f1 = f2
                c2 = a + gr * (b - a)
                f2 = _eval_at(c, c2)
            else:
                b = c2
                c2 = c1
                f2 = f1
                c1 = b - gr * (b - a)
                f1 = _eval_at(c, c1)
        
        v = max(f1, f2)
        if v > best_peak:
            best_peak = v
            
    return best_peak

def _local_search(c, deadline):
    """
    Perform local search by flipping coefficients to minimize the peak.
    """
    current_peak = _peak(c)
    improved = True
    c = c.copy()
    
    while improved and time.time() < deadline:
        improved = False
        best_flip = -1
        best_flip_peak = current_peak
        
        for i in range(len(c)):
            if time.time() > deadline:
                break
            c[i] = -c[i]
            p_trial = _peak(c)
            if p_trial < best_flip_peak:
                best_flip_peak = p_trial
                best_flip = i
            else:
                c[i] = -c[i]  # revert
                
        if best_flip >= 0:
            # Apply the best flip
            if best_flip != -1:
                # We reverted all, so flip the best one again
                c[best_flip] = -c[best_flip]
            current_peak = best_flip_peak
            improved = True
        else:
            # No improvement found in this pass
            break
            
    return c, current_peak

def construct():
    start = time.time()
    deadline = start + 145.0  # Safety margin
    
    n = 24
    
    # Initial candidates
    candidates = []
    
    # 1. GRS (Golomb-Rudin) style construction
    p, q = [1], [1]
    while len(p) < n:
        p, q = p + q, p + [-c for c in q]
    candidates.append(np.array(p[:n], dtype=float))
    
    # 2. Random restarts
    rng = np.random.default_rng(42)
    num_randoms = 50
    for _ in range(num_randoms):
        candidates.append(rng.choice([-1.0, 1.0], size=n))
        
    # 3. Known good patterns (if any were hardcoded, but we rely on search)
    # Add a few specific patterns that are often good
    # e.g. alternating, blocks, etc.
    candidates.append(np.array([1, -1] * 12, dtype=float))
    candidates.append(np.array([1, 1, -1, -1] * 6, dtype=float))
    candidates.append(np.array([1, -1, -1] * 8, dtype=float))

    best_c = None
    best_score = float('inf')
    
    for c in candidates:
        if time.time() > deadline - 10:
            break
            
        # Run local search for this candidate
        c_opt, peak_opt = _local_search(c, deadline - 5)
        
        if peak_opt < best_score:
            best_score = peak_opt
            best_c = c_opt
            
        # If we are very close to the theoretical optimum, we might stop early?
        # Theoretical optimum is ~1.2583758 * sqrt(24) ~ 6.161
        # If best_score < 6.162, it's likely optimal.
        if best_score < 6.162:
            break

    if best_c is None:
        # Fallback
        best_c = candidates[0]
        
    return [int(x) for x in best_c]
