import numpy as np

N = 50

def _ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h)
    if np.max(h) < 1e-12:
        return 0.0
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    
    # Compute autoconvolution peak
    n_fft = 1 << (2 * N - 1).bit_length()
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h
    H = np.fft.rfft(h_pad)
    cc = np.fft.irfft(H * H.conj(), n=n_fft)
    sup_cc = np.max(cc[:N])
    
    if sup_cc < 1e-12:
        return 0.0
        
    return l2_sq / (l1 * sup_cc)

def construct():
    best_h = None
    best_score = -1.0
    
    # Candidates: Delta, Boxcar, Random
    candidates = []
    
    # Delta at center
    h_delta = np.zeros(N)
    h_delta[N // 2] = 1.0
    candidates.append(h_delta)
    
    # Boxcar
    h_box = np.ones(N)
    candidates.append(h_box)
    
    # Random
    np.random.seed(42)
    h_rand = np.random.rand(N)
    candidates.append(h_rand)
    
    # Evaluate initial candidates
    for h in candidates:
        # Normalize to Sup=1 for evaluation? 
        # The task says "nonnegative heights". 
        # If I return h, the evaluator scores it.
        # My local search should optimize the shape.
        # The ratio is scale-invariant? No, R(c*h) = R(h)/c.
        # So I should normalize h to Sup=1 before scoring in my search,
        # and return the normalized h.
        # This assumes the evaluator also normalizes or that Sup=1 is the constraint.
        # Given the seed [1.0]*N, Sup=1 is the most consistent assumption.
        
        h_norm = h / np.max(h) if np.max(h) > 0 else h
        s = _ratio(h_norm)
        if s > best_score:
            best_score = s
            best_h = h_norm.copy()
            
    # Local search
    # We want to maximize the ratio.
    # We operate on Sup-normalized vectors.
    h_current = best_h
    score_current = best_score
    
    for _ in range(5000):
        # Pick a random index
        idx = np.random.randint(N)
        # Perturb
        h_trial = h_current.copy()
        # Add a small random value, clip to non-negative
        h_trial[idx] += np.random.randn() * 0.1
        h_trial = np.maximum(h_trial, 0.0)
        
        # Normalize to Sup=1
        max_val = np.max(h_trial)
        if max_val < 1e-12:
            continue
        h_trial = h_trial / max_val
        
        s = _ratio(h_trial)
        if s > score_current:
            score_current = s
            h_current = h_trial
            best_score = s
            best_h = h_current
    
    return list(best_h)
