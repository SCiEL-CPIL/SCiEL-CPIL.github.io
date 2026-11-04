import numpy as np
import time

N = 400
TIME_LIMIT = 110.0  # Safety margin

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    s = np.zeros(n, dtype=np.int8)
    s[0] = 1
    for i in range(1, n):
        # Number of 1s in binary representation of i
        s[i] = 1 - (bin(i).count('1') % 2)
    return s

def _compute_autocorr(s):
    """Compute autocorrelation A_k for k=1..N-1."""
    N = len(s)
    A = np.zeros(N)
    for k in range(1, N):
        A[k] = np.dot(s[:-k], s[k:])
    return A

def _max_sidelobe(A):
    """Return max |A_k| for k=1..N-1."""
    return np.max(np.abs(A[1:]))

def _sum_sq(A):
    """Return sum of A_k^2 for k=1..N-1."""
    return np.sum(A[1:]**2)

def construct():
    start_time = time.time()
    
    def time_ok():
        return time.time() - start_time < TIME_LIMIT - 2
    
    # Strategy:
    # 1. Initialize with Thue-Morse (known low autoconvolution peak)
    # 2. Greedy local search using delta method for speed
    # 3. Multiple restarts with random perturbations of Thue-Morse
    # 4. Track best by max_sidelobe (the actual score metric)
    
    best_score = float('inf')
    best_seq = None
    
    # Pre-compute Thue-Morse base
    tm = _thue_morse(N)
    
    def greedy_search(s, max_passes=200):
        """Greedy local search minimizing sum_sq, tracking max_sidelobe."""
        A = _compute_autocorr(s)
        current_sum_sq = _sum_sq(A)
        
        for _pass in range(max_passes):
            if not time_ok():
                break
            
            improved = False
            # Randomize the order of indices to avoid bias
            indices = np.random.permutation(N)
            
            for i in indices:
                if not time_ok():
                    break
                
                si = s[i]
                
                # Compute delta_A for all k efficiently
                # delta_A[k] = -2 * si * (s[i+k] if i+k<N else 0 + s[i-k] if i-k>=0 else 0)
                delta_sum_sq = 0.0
                deltas = np.zeros(N, dtype=np.float64)
                
                # Vectorized computation of deltas
                # For k where i+k < N: contribution is -2*si*s[i+k]
                # For k where i-k >= 0: contribution is -2*si*s[i-k]
                
                # k ranges from 1 to N-1
                # For plus part: k such that i+k < N => k < N-i
                # For minus part: k such that i-k >= 0 => k <= i
                
                # Build the delta array
                # k = 1, 2, ..., N-1
                # plus contribution exists for k in [1, N-1-i]
                # minus contribution exists for k in [1, i]
                
                if N - 1 - i > 0:
                    # k from 1 to N-1-i
                    k_plus = np.arange(1, N - i)
                    deltas[k_plus] += -2.0 * si * s[i + k_plus]
                
                if i > 0:
                    # k from 1 to i
                    k_minus = np.arange(1, i + 1)
                    deltas[k_minus] += -2.0 * si * s[i - k_minus]
                
                # Compute delta_sum_sq = sum(2*A[k]*deltas[k] + deltas[k]^2)
                mask = deltas != 0
                if np.any(mask):
                    delta_sum_sq = np.sum(2.0 * A[mask] * deltas[mask] + deltas[mask]**2)
                
                if delta_sum_sq < -1e-9:
                    # Accept the flip
                    s[i] = -si
                    A += deltas
                    current_sum_sq += delta_sum_sq
                    improved = True
            
            if not improved:
                break
        
        return s, A, current_sum_sq
    
    # Restart 0: Pure Thue-Morse
    s = tm.copy().astype(np.float64)
    s, A, sum_sq = greedy_search(s)
    max_sl = _max_sidelobe(A)
    score = max_sl / (N - 1)
    
    if score < best_score:
        best_score = score
        best_seq = s.copy()
    
    # Restarts: Perturbed Thue-Morse
    n_restarts = 30
    for r in range(n_restarts):
        if not time_ok():
            break
        
        # Create perturbation: flip a random subset of positions
        s = tm.copy().astype(np.float64)
        n_flips = np.random.randint(5, 40)
        flip_indices = np.random.choice(N, size=n_flips, replace=False)
        s[flip_indices] *= -1
        
        s, A, sum_sq = greedy_search(s, max_passes=100)
        max_sl = _max_sidelobe(A)
        score = max_sl / (N - 1)
        
        if score < best_score:
            best_score = score
            best_seq = s.copy()
    
    # If we still have time, do random restarts
    if time_ok():
        for r in range(10):
            if not time_ok():
                break
            
            s = np.random.choice([1.0, -1.0], N)
            s, A, sum_sq = greedy_search(s, max_passes=50)
            max_sl = _max_sidelobe(A)
            score = max_sl / (N - 1)
            
            if score < best_score:
                best_score = score
                best_seq = s.copy()
    
    if best_seq is None:
        return np.ones(N)
    
    return best_seq
