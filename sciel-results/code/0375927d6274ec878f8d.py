"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max_abs(a):
    """Compute max absolute value of linear autoconvolution of a."""
    # Linear autoconvolution via FFT
    n = len(a)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fa = np.fft.rfft(a, size)
    conv = np.fft.irfft(fa * fa, size)
    # Linear autoconvolution is the first 2n-1 values
    return np.max(np.abs(conv[:2 * n - 1]))

def _score(a):
    """Compute the ratio max|autoconv| / (sum a)^2. Lower is better."""
    s = np.sum(a)
    if s == 0:
        return np.inf
    return _autoconv_max_abs(a) / (s * s)

def _local_search(a, max_iters, deadline):
    """Local search with single-flip and pair-flip moves."""
    a = a.copy()
    current_score = _score(a)
    best_a = a.copy()
    best_score = current_score
    n = len(a)
    rng = np.random.default_rng()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Single flip
        i = rng.integers(n)
        a[i] = -a[i]
        new_score = _score(a)
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_a = a.copy()
        else:
            a[i] = -a[i]  # revert
        
        # Pair flip (every few iterations)
        if it % 3 == 0:
            i = rng.integers(n)
            j = rng.integers(n)
            if i != j:
                a[i] = -a[i]
                a[j] = -a[j]
                new_score = _score(a)
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_a = a.copy()
                else:
                    a[i] = -a[i]
                    a[j] = -a[j]
    
    return best_a, best_score

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    rng = np.random.default_rng(42)
    
    best_a = np.ones(N)
    best_score = 1.0  # baseline
    
    # Generate diverse initial patterns
    patterns = []
    
    # Pure random ±1
    for _ in range(5):
        patterns.append(rng.choice([-1.0, 1.0], size=N))
    
    # Sparse patterns: mostly 1s with some -1s
    for frac in [0.1, 0.2, 0.3]:
        a = np.ones(N)
        n_neg = int(N * frac)
        idx = rng.choice(N, size=n_neg, replace=False)
        a[idx] = -1.0
        patterns.append(a)
    
    # Block patterns: alternating blocks
    for block_size in [10, 20, 50]:
        a = np.zeros(N)
        for i in range(N):
            a[i] = 1.0 if (i // block_size) % 2 == 0 else -1.0
        patterns.append(a)
    
    # Run local search from each pattern
    iters_per_pattern = max(100, int(20000 / len(patterns)))
    
    for p in patterns:
        if time.time() > deadline:
            break
        a, score = _local_search(p, iters_per_pattern, deadline)
        if score < best_score:
            best_score = score
            best_a = a
    
    # Final polish on the best pattern with more iterations
    if time.time() < deadline - 10:
        a, score = _local_search(best_a, 50000, deadline)
        if score < best_score:
            best_score = score
            best_a = a
    
    return best_a
# EVOLVE-BLOCK-END
