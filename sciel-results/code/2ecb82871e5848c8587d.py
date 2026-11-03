"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _construct_gcp_pair(n, time_limit):
    """
    Construct a Golay Complementary Pair (a, b) of length n.
    We search for binary sequences (+1/-1) such that a*a + b*b = delta.
    This implies a*a and b*b are 'noise' with peak 1, and their sum is 1 at 0 and 0 elsewhere.
    If we set h = a + b, then h*h = a*a + b*b + 2(a*b) = delta + 2(a*b).
    The peak of h*h is roughly 1 + 2*max|a*b|.
    For random a,b, max|a*b| ~ sqrt(n). So peak ~ 2*sqrt(n).
    Sum(h) = Sum(a) + Sum(b).
    If we can make Sum(a) and Sum(b) large, the ratio improves.
    However, GCP construction is hard. We use a heuristic search.
    """
    best_a = np.ones(n)
    best_b = np.ones(n)
    best_val = float('inf')
    
    # Start with random or simple patterns
    candidates = [
        (np.ones(n), np.ones(n)),
        (np.ones(n), np.ones(n) * (-1)**np.arange(n)),
        (np.random.choice([-1, 1], n), np.random.choice([-1, 1], n))
    ]
    
    # Simple local search to find a GCP-like pair
    # Objective: minimize max |a*a + b*b - delta|
    # But we actually want to minimize score of h = a+b.
    # Let's just search for a, b that minimize score(a+b).
    
    t_start = time.time()
    
    for _ in range(50): # 50 restarts
        a = np.random.choice([-1, 1], n)
        b = np.random.choice([-1, 1], n)
        
        h = a + b
        current_score = _score(h)
        
        improved = True
        while improved and (time.time() - t_start) < time_limit:
            improved = False
            # Try flipping one element in a
            idx = np.random.randint(n)
            old_val = a[idx]
            a[idx] = -old_val
            h = a + b
            new_score = _score(h)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                a[idx] = old_val
            
            # Try flipping one element in b
            idx = np.random.randint(n)
            old_val = b[idx]
            b[idx] = -old_val
            h = a + b
            new_score = _score(h)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                b[idx] = old_val
                
        h = a + b
        s = _score(h)
        if s < best_val:
            best_val = s
            best_a = a.copy()
            best_b = b.copy()
            
    return best_a + best_b, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 115.0
    
    # Baseline: All ones
    h_const = np.ones(N)
    best_h = h_const
    best_score = _score(h_const)
    
    # Strategy: Golay Complementary Pair Construction
    # We allocate 100 seconds to this.
    t_limit = 100.0
    h_gcp, score_gcp = _construct_gcp_pair(N, t_limit)
    
    if score_gcp < best_score:
        best_h = h_gcp
        best_score = score_gcp
        
    # If we have time left, try a few other structured candidates?
    # Maybe a "chirp" pair?
    if time.time() < deadline - 5.0:
        # Try a chirp based h
        # h[n] = exp(i * pi * n^2 / N) is complex.
        # Real part: cos(pi * n^2 / N)
        # Let's try sum of two chirps with different rates to cancel peaks?
        n = np.arange(N)
        h_chirp1 = np.cos(np.pi * n**2 / N)
        h_chirp2 = np.cos(np.pi * n**2 / (N+1))
        h_chirp_sum = h_chirp1 + h_chirp2
        s = _score(h_chirp_sum)
        if s < best_score:
            best_score = s
            best_h = h_chirp_sum
            
    return best_h

# EVOLVE-BLOCK-END
