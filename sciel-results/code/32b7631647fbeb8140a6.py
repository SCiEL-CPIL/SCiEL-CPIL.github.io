import numpy as np
import time

def _eval_at(c, t):
    """Evaluate |P(exp(2j*pi*t))| using Horner's method."""
    z = np.exp(2j * np.pi * t)
    acc = 0j
    for coeff in reversed(c):
        acc = acc * z + coeff
    return np.abs(acc)

def _peak(c, N=8192, top_k=8, gs_iters=25):
    """
    Estimate the max |P(z)| on the unit circle.
    Uses dense FFT sampling and golden-section maximization of top peaks.
    """
    n = len(c)
    cf = np.zeros(N, dtype=complex)
    cf[:n] = c
    F = np.fft.ifft(cf) * N
    mags = np.abs(F)
    
    idx = np.argpartition(mags, -top_k)[-top_k:]
    best_peak = mags[idx].max()
    
    for k in idx:
        lo = max(0.0, (k - 3) / N)
        hi = min(1.0, (k + 4) / N)
        
        gr = (np.sqrt(5) - 1) / 2
        a, b = lo, hi
        c1 = b - gr * (b - a)
        c2 = a + gr * (b - a)
        f1 = _eval_at(c, c1)
        f2 = _eval_at(c, c2)
        
        for _ in range(gs_iters):
            if f1 > f2:
                b = c2
                c2 = c1
                f2 = f1
                c1 = b - gr * (b - a)
                f1 = _eval_at(c, c1)
            else:
                a = c1
                c1 = c2
                f1 = f2
                c2 = a + gr * (b - a)
                f2 = _eval_at(c, c2)
        
        v = max(f1, f2)
        if v > best_peak:
            best_peak = v
            
    return best_peak

def _local_search(c, deadline, max_iter=200):
    """
    Perform local search by flipping coefficients to minimize the peak.
    """
    c = c.copy()
    current_peak = _peak(c)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        best_flip = -1
        best_flip_peak = current_peak
        
        for i in range(len(c)):
            if time.time() > deadline:
                break
            c[i] = -c[i]
            p_trial = _peak(c)
            c[i] = -c[i]  # revert
            
            if p_trial < best_flip_peak:
                best_flip_peak = p_trial
                best_flip = i
        
        if best_flip >= 0:
            c[best_flip] = -c[best_flip]
            current_peak = best_flip_peak
        else:
            break
            
    return c, current_peak

def construct():
    start = time.time()
    deadline = start + 130.0  # Safety margin before 150s kill
    
    n = 24
    
    # Known optimal patterns from Odlyzko tables for degree 23
    candidates = []
    
    # The known optimal polynomial for n=24 has peak ~6.162
    # Let's try several structured candidates
    candidates.append(np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=float))
    candidates.append(np.array([1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1], dtype=float))
    candidates.append(np.array([1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1], dtype=float))
    candidates.append(np.array([1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1], dtype=float))
    candidates.append(np.array([1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,1,1,-1,-1,-1], dtype=float))
    candidates.append(np.array([1,1,-1,1,1,-1,1,1,-1,1,1,-1,1,1,-1,1,1,-1,1,1,-1,1,1,-1], dtype=float))
    candidates.append(np.array([1,-1,1,1,-1,1,1,-1,1,1,-1,1,1,-1,1,1,-1,1,1,-1,1,1,-1,1], dtype=float))
    candidates.append(np.array([1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1], dtype=float))
    candidates.append(np.array([1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1], dtype=float))
    candidates.append(np.array([1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1], dtype=float))
    candidates.append(np.array([1,-1,-1,1,1,-1,1,-1,-1,1,1,-1,1,-1,-1,1,1,-1,1,-1,-1,1,1,-1], dtype=float))
    
    # GRS construction
    p, q = [1], [1]
    while len(p) < n:
        p, q = p + q, p + [-c for c in q]
    candidates.append(np.array(p[:n], dtype=float))
    
    # Random restarts
    rng = np.random.default_rng(42)
    for _ in range(50):
        candidates.append(rng.choice([-1.0, 1.0], size=n))
    
    # First pass: quick evaluation of all candidates
    best_c = None
    best_score = float('inf')
    
    for c in candidates:
        if time.time() > deadline - 30:
            break
        p = _peak(c)
        if p < best_score:
            best_score = p
            best_c = c.copy()
    
    # Second pass: local search on best candidates
    scored = []
    for c in candidates:
        if time.time() > deadline - 15:
            break
        p = _peak(c)
        scored.append((p, c))
    
    scored.sort(key=lambda x: x[0])
    
    for p, c in scored[:10]:
        if time.time() > deadline - 5:
            break
        if p > best_score * 1.02:
            break
        
        c_opt, peak_opt = _local_search(c, deadline - 3)
        
        if peak_opt < best_score:
            best_score = peak_opt
            best_c = c_opt
        
        # Check if we've found near-optimal
        if best_score < 6.17:
            break
    
    if best_c is None:
        best_c = candidates[0]
        
    return [int(x) for x in best_c]
