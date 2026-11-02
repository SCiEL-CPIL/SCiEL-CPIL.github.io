import time
import math
import numpy as np
import random

def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    if not U or len(U) < 2:
        return 0.0
    arr = np.array(U, dtype=np.int64)
    if arr[0] != 0:
        arr = np.concatenate(([0], arr))
    arr = np.unique(arr)
    n = len(arr)
    mx = int(arr.max())
    if mx == 0:
        return 0.0
    
    # Size of sumset U+U: values in [0, 2*mx]
    # Size of diffset U-U: values in [-mx, mx]
    N = 2 * mx + 1
    
    # Efficient counting via FFT
    sig = np.zeros(N, dtype=np.float64)
    sig[arr] = 1.0
    
    # Sumset: convolution of sig with sig
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
        
    F = np.fft.rfft(sig, n=fft_size)
    
    # Sumset
    S_sum = np.fft.irfft(F * F, n=fft_size)
    sumset_size = np.sum(S_sum[:N] > 1e-6) # Sums range 0..2*mx which is 0..N-1
    
    # Diffset
    flip_sig = np.zeros(N, dtype=np.float64)
    flip_sig[:] = sig[::-1]
    
    F_flip = np.fft.rfft(flip_sig, n=fft_size)
    S_diff = np.fft.irfft(F * F_flip, n=fft_size)
    
    diffset_size = np.sum(S_diff > 1e-6)
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
        
    denom = np.log(2 * mx + 1)
    if denom == 0:
        return 0.0
        
    return 1.0 + np.log(ratio) / denom

def _generate_initial_set(size, max_val):
    """
    Generate a 2D lattice-like set.
    U = { i*B + j | i in [0, rows), j in [0, cols) }
    We choose B large enough to avoid overlap between rows in sums?
    Actually, for sumset minimization, we want the set to be "thin".
    A lattice {i*B + j} has sumset size ~ 2*rows*B + 2*cols.
    Difference set size ~ 2*rows*B + 2*cols.
    Ratio is ~1. This is bad.
    
    We need a set where |U+U| < |U-U|.
    This happens if the set is "convex" or "logarithmic" in some sense?
    Actually, for U = {0, 1, 2, ..., n}, |U+U| = 2n+1, |U-U| = 2n+1. Ratio 1.
    
    Let's look at the paper's constructions. Size 2003.
    If we use a geometric progression U = {0, 1, M, M^2, ...}, sums are unique?
    If M is large enough, sums of two elements are unique (Sidon-like).
    If sums are unique, |U+U| = n(n+1)/2.
    Differences are also unique? |U-U| = n(n-1).
    Ratio ~ (n-1)/(n+1) ~ 1. Still not great.
    
    Wait, the score is 1 + log(ratio)/log(2M+1).
    If ratio is close to 1, log(ratio) is close to 0, score ~ 1.
    To get score > 1, we need ratio > 1.
    So |U-U| > |U+U|.
    
    When is |U-U| > |U+U|?
    For U = {0, 1, ..., n}, they are equal.
    For U = {0, 1, 3, 7, 15, ...} (2^k - 1), let's check.
    U = {0, 1, 3}.
    U+U = {0, 1, 2, 3, 4, 6}. Size 6.
    U-U = {-3, -2, -1, 0, 1, 2, 3}. Size 7.
    Ratio 7/6 > 1.
    Score = 1 + log(7/6)/log(7).
    log(1.166) ~ 0.15. log(7) ~ 1.95.
    Score ~ 1 + 0.077 = 1.077.
    
    The current program gets 1.0117.
    The inspirations get 1.0281.
    
    The strategy of "prefix sums of primes" is essentially a Sidon set construction.
    For a Sidon set, |U+U| is minimal (n(n+1)/2).
    |U-U| is maximal? No, for Sidon sets, differences are not necessarily unique.
    But if we use a "weakly Sidon" or specific structure, maybe we can maximize ratio.
    
    Let's try a different structure: **Bullseye / Radial**.
    Or **Simulated Annealing on a small set**.
    
    Since the time limit is 120s, and FFT is fast, we can do local search.
    Start with a set like {0, 1, 3, 7, 15, ...} (powers of 2 minus 1).
    Then perform random moves:
    1. Add a random new element x.
    2. Remove a random element x (keep 0).
    3. Move a random element x to x + delta or x - delta.
    
    We need to be careful with max(U). If max(U) grows too large, denominator grows.
    So we should constrain max(U) to a reasonable range, e.g., < 100,000.
    
    Let's implement a simple Simulated Annealing.
    """
    # Start with powers of 2 minus 1: 0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023...
    # Let's start with a few elements.
    U = [0, 1, 3, 7, 15, 31, 63, 127, 255, 511]
    return U

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initialize with a geometric-like set
    U = [0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383]
    U = np.array(U, dtype=np.int64)
    
    current_score = _score_set(U)
    best_set = U.copy()
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 50.0
    T_min = 0.1
    cooling_rate = 0.99
    
    # We will iterate until time runs out
    # Each step: propose a move, calculate new score, accept if better or with prob exp((new-old)/T)
    
    # To speed up, we can limit the size of U.
    # The paper mentions size 2003.
    # Let's see if we can grow the set.
    
    # Moves:
    # 1. Add a random element in [1, 2*max(U)]
    # 2. Remove a random element (not 0)
    # 3. Perturb a random element
    
    # We need to keep U sorted and unique for FFT?
    # _score_set handles sorting/uniquing.
    
    # However, FFT size depends on max(U).
    # If max(U) is ~ 10^5, N = 2*10^5+1. FFT size ~ 2^19 = 524288.
    # This is fast.
    
    # Let's try to grow the set size to ~ 100-500 first, then optimize.
    
    # Initial growth: add elements that maximize score incrementally?
    # Or just random growth.
    
    # Let's stick to SA.
    
    # Random seed for reproducibility in local testing, but random in eval?
    # random.seed(42)
    
    while time.time() < deadline:
        T = max(T_min, T * cooling_rate)
        
        # Choose a move
        move_type = random.randint(0, 2)
        new_U = U.copy()
        
        if move_type == 0:
            # Add element
            if len(new_U) < 500: # Cap size to avoid huge FFTs and collisions
                # Sample from [1, 2*max]
                max_val = int(new_U.max())
                x = random.randint(1, 2 * max_val + 100)
                if x not in new_U:
                    new_U = np.append(new_U, x)
                else:
                    continue
        elif move_type == 1:
            # Remove element
            if len(new_U) > 5:
                idx = random.randint(1, len(new_U) - 1) # Don't remove 0
                new_U = np.delete(new_U, idx)
        else:
            # Perturb
            if len(new_U) > 1:
                idx = random.randint(1, len(new_U) - 1)
                old_val = new_U[idx]
                # Small perturbation
                delta = random.randint(-10, 10)
                new_val = old_val + delta
                if new_val > 0 and new_val not in new_U:
                    new_U[idx] = new_val
                else:
                    continue
        
        new_score = _score_set(new_U)
        
        if new_score > current_score:
            U = new_U
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = U.copy()
        else:
            # Accept with probability
            delta_score = new_score - current_score
            if delta_score < 0:
                prob = math.exp(delta_score / T)
                if random.random() < prob:
                    U = new_U
                    current_score = new_score
            else:
                # Should not happen if check above, but safe
                U = new_U
                current_score = new_score
        
        # If we are stuck for a long time, maybe increase T?
        # Or just let it cool.
        
    return list(best_set)

if __name__ == "__main__":
    s = construct()
    print(f"Result: {s[:10]}... Size: {len(s)}, Max: {max(s)}, Score: {_score_set(s)}")
