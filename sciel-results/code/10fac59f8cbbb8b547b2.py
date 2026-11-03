"""Rewritten program: signed step functions minimizing max |autoconv| / (sum)^2."""
import time
import numpy as np

N = 400
START_TIME = time.time()
TIME_LIMIT = 140  # safety margin before 150s kill

# Precompute FFT indices for fast autocorrelation
# For linear autocorrelation of a length-N sequence:
# We embed in length >= 2N-1, use FFT, multiply by conjugate, inverse FFT
_FFT_N = 1 << (2 * N - 1).bit_length()
_freqs = np.fft.fftfreq(_FFT_N)

def _autocorr(x):
    """Compute linear autocorrelation of x (length N). Returns array length 2N-1."""
    X = np.fft.fft(x, _FFT_N)
    acf = np.fft.ifft(X * np.conj(X)).real
    return acf[:2 * N - 1]

def _score(x):
    """
    Score = max |c_k| for k != 0, divided by (sum x)^2.
    If sum x is zero, return a large penalty (or inf).
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    acf = _autocorr(x)
    # Exclude lag 0 (k=0)
    max_abs = max(np.max(np.abs(acf[:N - 1])), np.max(np.abs(acf[N:])))
    return max_abs / (s * s)

def _flip(x, idx):
    """Flip sign of x[idx], return new array (or modify in place with undo)."""
    x[idx] = -x[idx]

def _local_search(x, max_iters=5000):
    """Greedy local search: flip bits that reduce score."""
    cur_score = _score(x)
    improved = True
    iters = 0
    while improved and iters < max_iters and (time.time() - START_TIME) < TIME_LIMIT:
        improved = False
        # Try random subset of positions to save time
        test_indices = np.random.choice(N, size=min(N, 50), replace=False)
        best_gain = 0
        best_idx = -1
        for idx in test_indices:
            x[idx] = -x[idx]
            new_score = _score(x)
            gain = cur_score - new_score
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
            x[idx] = -x[idx]  # undo
        if best_idx >= 0:
            x[best_idx] = -x[best_idx]
            cur_score = _score(x)
            improved = True
        iters += 1
    return cur_score

def construct():
    best_x = np.ones(N, dtype=np.float64)
    best_score = _score(best_x)
    
    deadline = time.time() + TIME_LIMIT - 5  # leave 5s margin
    
    # Multiple restarts with different initializations
    n_restarts = 20
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate initial sequence
        if r == 0:
            x = np.ones(N, dtype=np.float64)
        elif r == 1:
            x = np.random.choice([-1.0, 1.0], size=N)
        elif r == 2:
            # Random with bias
            x = np.random.choice([-1.0, 1.0], size=N)
            # Make sum non-zero by flipping a few
            s = np.sum(x)
            if abs(s) < N * 0.1:
                idxs = np.random.choice(N, size=10, replace=False)
                x[idxs] = 1.0
        elif r == 3:
            # Low-frequency dominant
            t = np.linspace(0, 2 * np.pi, N)
            x = np.sign(np.sin(t) + 0.1 * np.random.randn(N))
            x[x == 0] = 1.0
        elif r == 4:
            # High-frequency
            t = np.linspace(0, 20 * np.pi, N)
            x = np.sign(np.sin(t) + 0.1 * np.random.randn(N))
            x[x == 0] = 1.0
        elif r == 5:
            # Block structure
            x = np.ones(N)
            block_len = 10
            n_blocks = N // block_len
            signs = np.random.choice([-1.0, 1.0], size=n_blocks)
            for b in range(n_blocks):
                x[b * block_len:(b + 1) * block_len] = signs[b]
        elif r == 6:
            # Threading-like: alternating with noise
            x = np.tile([1.0, -1.0], N // 2)
            flip_mask = np.random.rand(N) < 0.2
            x[flip_mask] *= -1
        elif r == 7:
            # Random permutation of +1/-1
            x = np.ones(N)
            x[N // 4:] = -1.0
            np.random.shuffle(x)
        else:
            x = np.random.choice([-1.0, 1.0], size=N)
        
        # Ensure sum is not too small
        s = np.sum(x)
        if abs(s) < N * 0.05:
            # Flip some to increase sum
            pos_mask = x > 0
            neg_mask = x < 0
            if np.sum(pos_mask) < N // 2:
                # Add more +1s
                neg_idx = np.where(neg_mask)[0]
                if len(neg_idx) > 0:
                    n_flip = min(len(neg_idx), N // 10)
                    idxs = np.random.choice(neg_idx, size=n_flip, replace=False)
                    x[idxs] = 1.0
            else:
                pos_idx = np.where(pos_mask)[0]
                if len(pos_idx) > 0:
                    n_flip = min(len(pos_idx), N // 10)
                    idxs = np.random.choice(pos_idx, size=n_flip, replace=False)
                    x[idxs] = -1.0
        
        # Local search from this initialization
        cur_score = _local_search(x, max_iters=3000)
        
        if cur_score < best_score:
            best_score = cur_score
            best_x = x.copy()
        
        # Simulated annealing refinement
        if time.time() > deadline:
            break
        
        x = best_x.copy()
        T = 0.1
        cooling = 0.99
        sa_iters = 2000
        sa_improved = False
        
        for it in range(sa_iters):
            if time.time() > deadline:
                break
            if T < 1e-6:
                break
            idx = np.random.randint(N)
            old_val = x[idx]
            x[idx] = -x[idx]
            new_score = _score(x)
            delta = new_score - best_score
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                best_score = new_score
                best_x = x.copy()
                sa_improved = True
            else:
                x[idx] = old_val
            T *= cooling
        
        # Final local search on best found
        x = best_x.copy()
        final_score = _local_search(x, max_iters=2000)
        if final_score < best_score:
            best_score = final_score
            best_x = x.copy()
    
    return best_x
