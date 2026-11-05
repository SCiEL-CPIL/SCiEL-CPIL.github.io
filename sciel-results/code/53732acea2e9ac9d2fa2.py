import numpy as np
import time

N = 400

def _autoconvolve_fft(h):
    """Compute the full linear autoconvolution using FFT."""
    # Pad to 2*N - 1 for linear convolution
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    # The linear autoconvolution is in acf[0:N] and acf[fft_size:N-1] (wrapped)
    # Actually, for linear convolution of length N with itself, result length is 2N-1.
    # np.fft.irfft returns fft_size elements.
    # The first N elements are lags 0 to N-1.
    # The remaining elements are lags N to fft_size-1, which correspond to negative lags.
    # We only care about lags 0 to N-1 for the max absolute value (symmetric).
    return acf[:N]

def _score_from_acf(acf):
    """Compute the score (max abs acf for lags 1..N-1) from a precomputed acf array."""
    # acf[0] is the sum of squares, which is constant for +/-1 sequences.
    # We look at lags 1 to N-1.
    return np.max(np.abs(acf[1:N]))

def _local_search_fast(h, time_limit):
    """
    Fast local search using O(N) updates to the autoconvolution array.
    """
    h = np.array(h, dtype=np.float64)
    acf = _autoconvolve_fft(h)
    current_max = _score_from_acf(acf)
    
    start_time = time.time()
    improved = True
    
    # Pre-compute indices for neighbor access
    # We need h[i+k] and h[i-k] for k in 1..N-1
    # This is effectively a sliding window sum.
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        best_gain = 0
        best_idx = -1
        best_new_max = current_max
        best_delta_acf = None
        
        # Try flipping each index
        # Optimization: We can randomize the order to break symmetries
        indices = np.random.permutation(N)
        
        for i in indices:
            # Calculate delta for flipping h[i]
            # h[i] becomes -h[i]
            # Change in acf[k]:
            # acf[k] = sum_j h[j] h[j+k]
            # Terms involving h[i]:
            # 1. j=i: h[i] h[i+k] -> -h[i] h[i+k] => change -2 h[i] h[i+k]
            # 2. j+k=i => j=i-k: h[i-k] h[i] -> h[i-k] (-h[i]) => change -2 h[i-k] h[i]
            
            # We need to construct the delta vector for lags 0..N-1
            # delta_acf[k] = -2 * h[i] * (h[(i+k)%N] + h[(i-k)%N]) 
            # Wait, linear convolution, not circular.
            # h is defined on [0, N-1].
            # For lag k (0 <= k < N):
            # acf[k] = sum_{j=0}^{N-1-k} h[j] h[j+k]
            
            # If we flip h[i]:
            # It affects terms where j=i (if i <= N-1-k) and j+k=i (if i-k >= 0).
            
            # Let's construct the delta vector efficiently.
            # delta[k] = 0 for most k.
            # Only k such that i is in [0, N-1-k] or i-k is in [0, N-1] are affected.
            # This means k is in [0, N-1-i] (for j=i) and k is in [i, N-1] (for j=i-k).
            
            # Actually, it's easier to just compute the full delta vector using vectorized ops on h.
            # But doing this for every i is O(N^2).
            
            # Let's use the FFT update method instead, it's O(N log N) but constant is low?
            # No, O(N) time domain update is better if we can vectorize the delta creation.
            
            # Let's try a different vectorized approach for the delta:
            # We know h. We create a copy h_new = h.copy(); h_new[i] = -h_new[i].
            # Then we compute the difference in acf? No, that's O(N log N).
            
            # Let's stick to the O(N) manual update but optimized.
            # For a given i, the non-zero deltas are at lags k where the pair (j, j+k) involves i.
            # Pairs: (i, i+k) for k=0..N-1-i. And (i-k, i) for k=1..i.
            # Note: acf[k] is symmetric, acf[k] = acf[-k]. But we store 0..N-1.
            # The term h[i]h[i+k] contributes to acf[k].
            # The term h[i-k]h[i] contributes to acf[k] as well?
            # acf[k] = sum_{j} h[j]h[j+k].
            # If j=i, term is h[i]h[i+k].
            # If j+k=i => j=i-k, term is h[i-k]h[i].
            # So for lag k, the change is -2 h[i] h[i+k] (if i+k < N) AND -2 h[i-k] h[i] (if i-k >= 0).
            
            # Let's build the delta array for this specific i.
            # This is still O(N) per i.
            
            # Is there a faster way?
            # Maybe just recompute the score using FFT for the candidate?
            # N=400. FFT of size 1024.
            # 400 flips * 400 iterations * 1024 log 1024 ~ 400 * 400 * 10000 = 1.6e9 ops. Too slow in Python loops.
            
            # The O(N) update is 400 * 400 * 400 = 64e6 ops. Much better.
            
            # Let's implement the O(N) update carefully.
            
            delta = np.zeros(N)
            hi = h[i]
            
            # Contribution from j=i: h[i]h[i+k] for k=0 to N-1-i
            # k ranges from 0 to N-1-i
            # delta[k] += -2 * hi * h[i+k]
            if N - 1 - i >= 0:
                delta[0 : N-i] += -2 * hi * h[i : N]
            
            # Contribution from j=i-k: h[i-k]h[i] for k=1 to i
            # Here lag is k. j = i-k.
            # k ranges from 1 to i.
            # delta[k] += -2 * h[i-k] * hi
            if i > 0:
                # h[i-k] for k=1..i is h[i-1], h[i-2]...h[0]
                # We need to add this to delta[1..i]
                # h[i-1::-1] gives h[i-1], h[i-2]...
                delta[1 : i+1] += -2 * h[i-1::-1] * hi
            
            # Update acf
            new_acf = acf + delta
            new_max = _score_from_acf(new_acf)
            
            if new_max < best_new_max - 1e-9:
                best_new_max = new_max
                best_idx = i
                best_delta_acf = delta
                best_gain = current_max - new_max
            
            # Revert acf (it wasn't modified in place, we used new_acf)
            # No revert needed for acf itself, we just didn't assign it.
            # But we need to check if we should accept the flip.
            # If we accept, we must update h and acf.
            
        if best_idx != -1:
            # Apply the best flip
            h[best_idx] = -h[best_idx]
            acf = acf + best_delta_acf
            current_max = best_new_max
            improved = True

    return h, current_max

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +/-1 values."""
    seq = [1]
    while len(seq) < n:
        seq += [-x for x in seq]
    return np.array(seq[:n], dtype=np.float64)

def _legendre_sequence(N, p):
    """Generate a Legendre sequence of length N using prime p."""
    seq = np.zeros(N)
    for i in range(N):
        x = i + 1
        if x < p:
            # Legendre symbol
            if x % p == 0:
                seq[i] = 0
            else:
                seq[i] = 1 if pow(x, (p-1)//2, p) == 1 else -1
        else:
            seq[i] = 1 # Fallback
    # Ensure no zeros if possible, or handle them. 
    # For p=401, N=400, x is 1..400, all < p.
    return seq

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_h = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # 1. Generate diverse initial candidates
    candidates = []
    
    # Thue-Morse
    candidates.append(_thue_morse(N))
    
    # Legendre (p=401)
    candidates.append(_legendre_sequence(N, 401))
    
    # Random
    for _ in range(20):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # 2. Evaluate and sort candidates
    scored_candidates = []
    for h in candidates:
        acf = _autoconvolve_fft(h)
        s = _score_from_acf(acf)
        scored_candidates.append((s, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # 3. Refine top candidates with fast local search
    # We will try the top 5 candidates
    for i, (s, h) in enumerate(scored_candidates[:5]):
        if time.time() > deadline - 20:
            break
        time_limit = deadline - 10
        h_refined, s_refined = _local_search_fast(h, time_limit)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
    # 4. If time remains, do random restarts with local search
    while time.time() < deadline - 10:
        h_init = rng.choice([-1.0, 1.0], size=N)
        # Give each restart a smaller time budget to allow more restarts
        time_limit = min(deadline - 5, time.time() + 5)
        h_refined, s_refined = _local_search_fast(h_init, time_limit)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
    if best_h is None:
        best_h = np.ones(N)
        
    return best_h

# Test
if __name__ == "__main__":
    h = construct()
    acf = _autoconvolve_fft(h)
    score = _score_from_acf(acf)
    print(f"Score: {score}")
