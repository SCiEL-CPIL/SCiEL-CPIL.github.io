"""
Strategy: Sparse, real-valued amplitude optimization.
Instead of forcing x to be ±1, we treat x as a real-valued sequence.
We use the gauge freedom sum(x) = 1 to maximize the denominator.
We minimize the numerator max(|autoconv(x)|) using coordinate descent.
This allows the optimizer to find sparse, low-magnitude structures
that dense binary searches miss, potentially lowering the ratio.
"""

import numpy as np
from numpy.fft import fft, ifft
import time

N = 400

def _autoconv(x):
    """Compute linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    size2 = 1
    while size2 < size:
        size2 *= 2
    F = fft(x, size2)
    A = ifft(F * np.conj(F)).real
    return A[:size]

def _score(x):
    """Compute max(|autoconv|) / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    A = _autoconv(x)
    m = np.max(np.abs(A))
    return m / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # We will work with normalized sequences where sum(x) = 1.
    # The score is then just max(|autoconv(x)|).
    # Because of homogeneity: score(c*x) = max(|c^2 autoconv(x)|) / (c^2 sum(x)^2) = score(x).
    # So we can fix sum(x) = 1 without loss of generality.

    best_x = np.ones(N) / N  # Uniform distribution, sum=1
    best_score = _score(best_x)

    def try_candidate(x):
        nonlocal best_x, best_score
        # Normalize to sum=1
        s = np.sum(x)
        if abs(s) < 1e-12:
            return
        x_norm = x / s
        sc = _score(x_norm)
        if sc < best_score:
            best_score = sc
            best_x = x_norm.copy()

    # Phase 1: Deterministic sparse initializations
    # Try various sparsity levels
    rng = np.random.default_rng(42)
    
    sparsity_levels = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    
    for sp in sparsity_levels:
        if time.time() > deadline:
            break
        # Random sparse vectors
        for _ in range(50):
            if time.time() > deadline:
                break
            x = np.zeros(N)
            mask = rng.random(N) < sp
            x[mask] = rng.choice([-1.0, 1.0], size=np.sum(mask))
            # Add some noise to break symmetry
            x += rng.normal(0, 0.1, N)
            try_candidate(x)
            
        # Structured sparse: blocks
        if sp < 0.5:
            # Try periodic sparse patterns
            for period in [2, 3, 4, 5, 10, 20]:
                if time.time() > deadline:
                    break
                # Create a sparse pattern in one period
                # e.g., for period 4, sp 0.25 -> one non-zero
                num_nonzero = max(1, int(period * sp))
                pattern = np.zeros(period)
                indices = rng.choice(period, size=num_nonzero, replace=False)
                pattern[indices] = 1.0
                x = np.tile(pattern, N // period + 1)[:N]
                try_candidate(x)

    # Phase 2: Coordinate Descent on Real-Valued Amplitudes
    # We optimize x to minimize max(|autoconv(x)|) subject to sum(x)=1.
    # We do this by iteratively adjusting individual x_i.
    
    # Start from best candidate found so far
    x = best_x.copy()
    current_score = best_score
    
    # Pre-compute FFT size
    size = 2 * N - 1
    size2 = 1
    while size2 < size:
        size2 *= 2
    
    # Helper to update score efficiently?
    # Full FFT per coordinate step is O(N log N). N=400, size2=1024.
    # 400 steps * 1024 log 1024 ~ 400 * 10k ops ~ 4M ops per full pass.
    # Python loop overhead is high. Let's try batch updates or fewer passes.
    
    # Instead of single coordinate descent, let's do "block" coordinate descent
    # or just simple random coordinate descent with a time limit.
    
    # We'll use a simple loop:
    # 1. Pick a random index i
    # 2. Try changing x_i to a new value v
    # 3. If score improves, accept
    
    # To make it faster, we can pre-compute the contribution of x_i to the autoconvolution.
    # autoconv_k = sum_j x_j x_{j+k}.
    # Changing x_i affects terms where j=i or j+k=i.
    
    # However, implementing an efficient incremental update is complex.
    # Given N=400, a full FFT takes ~0.1ms in numpy.
    # 10000 updates * 0.1ms = 1 second. We have 140s.
    # So we can afford ~100,000 updates.
    
    # Let's do a simple random coordinate descent.
    # We need to maintain sum(x)=1.
    # If we change x_i to x_i + delta, we must subtract delta from some other x_j?
    # Or we can just change x_i and then renormalize the whole vector?
    # Renormalizing the whole vector changes all x_j, which breaks the local update logic.
    
    # Better approach:
    # Fix the sum constraint by defining x_N = 1 - sum(x_0...x_{N-1}).
    # Then we optimize x_0...x_{N-1} freely.
    # Score is computed on the full vector.
    
    # Let's implement this.
    x_free = x[:N-1].copy()
    x_full = np.zeros(N)
    x_full[:N-1] = x_free
    x_full[N-1] = 1.0 - np.sum(x_free)
    
    # Initial score
    sc = _score(x_full)
    if sc < best_score:
        best_score = sc
        best_x = x_full.copy()
    
    # Coordinate descent on x_free
    # We will try to minimize max(|autoconv|)
    
    # To speed up, we can use a smaller number of iterations but with larger steps?
    # Or just many small steps.
    
    # Let's try a "greedy" approach:
    # For each i, find the best x_i that minimizes the score.
    # This is a 1D optimization problem.
    # We can sample a few values for x_i.
    
    # Given time constraints, let's do random perturbations of the free variables.
    
    steps = 0
    while time.time() < deadline - 10.0:
        # Batch of random updates
        for _ in range(100):
            if time.time() > deadline - 10.0:
                break
            i = rng.integers(0, N-1)
            # Try a small random change
            delta = rng.normal(0, 0.01)
            x_free_new = x_free.copy()
            x_free_new[i] += delta
            
            # Construct full vector
            x_full_new = np.zeros(N)
            x_full_new[:N-1] = x_free_new
            x_full_new[N-1] = 1.0 - np.sum(x_free_new)
            
            sc_new = _score(x_full_new)
            
            if sc_new < sc - 1e-12:
                x_free = x_free_new
                x_full = x_full_new
                sc = sc_new
                if sc < best_score:
                    best_score = sc
                    best_x = x_full.copy()
        steps += 100
        
        # Occasionally, try a larger jump (restart-like)
        if steps % 1000 == 0:
             # Mix with a random sparse vector
             if rng.random() < 0.1:
                 x_new = np.zeros(N)
                 mask = rng.random(N) < 0.1
                 x_new[mask] = rng.choice([-1.0, 1.0], size=np.sum(mask))
                 s = np.sum(x_new)
                 if abs(s) > 1e-6:
                     x_new = x_new / s
                     x_free_new = x_new[:N-1]
                     x_full_new = x_new
                     sc_new = _score(x_full_new)
                     if sc_new < sc:
                         x_free = x_free_new
                         x_full = x_full_new
                         sc = sc_new
                         if sc < best_score:
                             best_score = sc
                             best_x = x_full.copy()

    # Final check
    return best_x

