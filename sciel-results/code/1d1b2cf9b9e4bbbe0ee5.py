"""
Strategy: Number-theoretic initialization + Vectorized Greedy Refinement.

The current program uses a slow, scalar local search that gets stuck in local minima 
early on. This strategy:
1. Initializes using a Legendre symbol sequence modulo a prime near N. These sequences 
   have known low additive energy and pseudorandom properties, providing a much better 
   starting point than random noise.
2. Uses a vectorized greedy algorithm. Instead of checking each flip one by one (O(N) 
   evaluations per pass), we precompute the autocorrelation vector R. 
   Flipping a bit at index i changes the objective function (max absolute autocorrelation) 
   in a predictable way. We can approximate the impact of all possible flips simultaneously 
   using NumPy operations, allowing us to evaluate all N positions in O(N) time per pass 
   with high constant factors, or even batch multiple passes.
   
   Specifically, the change in autocorrelation R[k] when flipping f[i] is:
   delta_R[k] = -2 * f[i] * (f[i+k] + f[i-k])  (with boundary handling)
   We want to minimize max(|R[k]|). 
   A greedy heuristic: find the flip that most reduces the current peak.
   To do this efficiently, we can compute the "potential" reduction for each flip.
   Since max is non-differentiable, we focus on the current peak indices.
   If the peak is at lag k*, flipping i helps if it reduces |R[k*]|.
   We can vectorize the calculation of how much each flip reduces the current peak magnitude.
"""

import numpy as np
import time

N = 400

def _legendre_symbol(n, p):
    """Compute Legendre symbol (n/p)."""
    if n % p == 0:
        return 0
    # Euler's criterion
    return pow(n, (p - 1) // 2, p)

def _init_legendre(N):
    """
    Initialize sequence with Legendre symbols for a prime p close to N.
    We need a sequence of length N. We can take the first N non-zero residues 
    or simply map indices 1..N to Legendre symbols mod p, where p is the 
    smallest prime >= N+1 to ensure we have enough distinct non-zero residues 
    if needed, or just use a large prime and map 0..N-1.
    
    Actually, standard Legendre sequences are defined on Z_p. 
    We want a length-N sequence. 
    Let p be a prime. We define f[i] = (i/p) for i=1..N? 
    If N < p, we just take the first N values. 
    We should ensure the sum is not zero if possible, but signed step functions 
    usually allow any sum. The task says "signed step function", implying +/- 1.
    Legendre symbol is 0, 1, -1. We must map 0 to +/-1. 
    Let's pick the smallest prime p >= N. 
    For i in 0..N-1, f[i] = legendre(i+1, p) if i+1 < p else ... 
    Actually, simpler: pick a prime p. Generate the Legendre sequence of length p. 
    Truncate to N. Replace 0s with 1s (or -1s). 
    """
    # Find smallest prime >= N
    def is_prime(num):
        if num < 2:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True

    p = N
    while not is_prime(p):
        p += 1
        
    # Generate Legendre symbols for 1..p
    # f[i] for i in 0..N-1 corresponds to residue i+1 mod p
    seq = []
    for i in range(N):
        val = _legendre_symbol(i + 1, p)
        if val == 0:
            # This shouldn't happen for i+1 < p, but if N == p, i+1 can be p.
            # If i+1 == p, Legendre is 0. Map to 1.
            val = 1
        seq.append(float(val))
    
    return np.array(seq)

def _compute_autocorr(f):
    """
    Compute linear autocorrelation R[k] for k=0..N-1.
    R[k] = sum_{i=0}^{N-1-k} f[i] * f[i+k]
    Using FFT for speed.
    """
    n = len(f)
    size = 1
    while size < 2 * n:
        size *= 2
    F = np.fft.rfft(f, size)
    R = np.fft.irfft(F * np.conj(F), size)
    return R[:n]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # 1. Initialize with Legendre sequence
    f = _init_legendre(N)
    
    # 2. Initial Evaluation
    R = _compute_autocorr(f)
    current_peak = np.max(np.abs(R[1:]))
    
    # 3. Vectorized Greedy Refinement
    # We will perform passes where we try to flip bits to reduce the peak.
    # To be fast, we identify the "active" lags (those near the peak).
    # However, a full vectorized greedy that considers all lags is complex.
    # A simpler, robust vectorized approach:
    # In each pass, calculate the potential benefit of flipping each i.
    # Benefit is defined by how much it reduces the current maximum absolute autocorrelation.
    # Let M = max(|R[k]| for k>0). Let K_peak be the set of k where |R[k]| == M.
    # Flipping i changes R[k] by delta_k = -2 * f[i] * (f[i+k] + f[i-k]).
    # We want to minimize max(|R[k] + delta_k|).
    # Approximation: Focus on the current peak lags.
    # If we can reduce the value at the peak lag, we likely reduce the global max.
    
    improved = True
    passes = 0
    max_passes = 1000 # Cap to avoid infinite loops if stuck
    
    while improved and time.time() < deadline - 5.0 and passes < max_passes:
        improved = False
        passes += 1
        
        # Identify peak lags
        abs_R = np.abs(R[1:])
        max_val = np.max(abs_R)
        # Get indices of lags that are close to the max (within 1% or so) to handle ties/noise
        # For strictness, let's just use the exact max indices
        peak_lags = np.where(abs_R == max_val)[0] + 1
        
        if len(peak_lags) == 0:
            break
            
        # Calculate delta_R for all positions i and all peak lags k
        # delta_R[i, j] = -2 * f[i] * (f[i + k_j] + f[i - k_j])
        # We need to handle boundaries.
        
        # Vectorized calculation of the change in autocorrelation for each flip i
        # and each peak lag k.
        
        # Let's compute the impact on the specific peak lags.
        # If we flip i, R[k] becomes R[k] - 2*f[i]*(f[i+k] + f[i-k]).
        # We want to know if this reduces |R[k]|.
        
        # Since N is small (400), we can actually afford to compute the full new R 
        # for the *best* candidate found in a batch, but finding the best candidate 
        # is the bottleneck.
        
        # Alternative: Simulated Annealing with vectorized proposal?
        # Or just a very fast scalar greedy that is optimized in C via NumPy?
        
        # Let's try a "Block Flip" strategy or just a highly optimized single-bit greedy.
        # The previous program did single-bit greedy but was slow because it recomputed 
        # the full FFT for every single bit flip.
        
        # Optimization: 
        # Instead of recomputing FFT for every flip, we can update R directly.
        # But R is length N. Updating R for a flip at i takes O(N) time.
        # So one pass of N flips takes O(N^2). 
        # N=400, N^2 = 160,000 operations. This is very fast in Python if vectorized?
        # No, a loop of 400 iterations doing vector operations of length 400 is fine.
        
        # Let's implement a fast greedy pass:
        # For each i in 0..N-1:
        #   Calculate delta_R (vector of length N)
        #   Calculate new_R = R + delta_R
        #   Calculate new_peak = max(|new_R[1:]|)
        #   If new_peak < current_peak, accept flip, update R and f.
        
        # This is O(N^2) per pass. 160k ops per pass. 
        # 1000 passes = 160M ops. Might be too slow in pure Python loops.
        # We need to vectorize the "try all flips" step.
        
        # Can we vectorize the "try all flips" step?
        # We want to find the i that minimizes max(|R + delta_i|).
        # delta_i[k] = -2 * f[i] * (f[i+k] + f[i-k])
        
        # Let's construct a matrix Delta where Delta[i, k] is the change in R[k] 
        # if we flip f[i].
        # Delta[i, k] = -2 * f[i] * (f[i+k] + f[i-k])
        
        # We can compute this matrix in O(N^2) using broadcasting.
        # Then we have NewR[i, k] = R[k] + Delta[i, k].
        # NewPeak[i] = max(|NewR[i, 1:]|).
        # We pick i* = argmin(NewPeak).
        # If NewPeak[i*] < current_peak, we apply the flip.
        # This takes O(N^2) memory and time per pass, but all in NumPy.
        # 400x400 matrix is 160k floats. 160k * 8 bytes = 1.28 MB. Trivial.
        # Operations: ~160k * few ops. Very fast.
        
        # Let's implement this.
        
        # 1. Construct Delta matrix
        # Indices i: 0..N-1, k: 0..N-1
        i_idx = np.arange(N)
        k_idx = np.arange(N)
        
        # We need f[i+k] and f[i-k].
        # Use np.roll for circular? No, we want linear (zero-padded).
        # f[i+k] is valid if 0 <= i+k < N.
        # f[i-k] is valid if 0 <= i-k < N.
        
        # Create index arrays for f[i+k]
        # M_ik = i + k
        # If M_ik < N, val = f[M_ik], else 0.
        
        # This is hard to do directly with simple broadcasting without masks.
        # But we can use np.where.
        
        M_add = i_idx[:, None] + k_idx[None, :]
        M_sub = i_idx[:, None] - k_idx[None, :]
        
        # f_add[i, k] = f[i+k] if valid else 0
        # f_sub[i, k] = f[i-k] if valid else 0
        
        # To get f[i+k], we can use an array of indices, but we need to handle out of bounds.
        # Trick: Create a padded f_pad of length 2N.
        f_pad = np.concatenate([np.zeros(N), f])
        
        # For M_add:
        # Valid if M_add < N.
        # Index into f_pad? 
        # If we use f_pad, f[i+k] is at index i+k. 
        # But we want 0 if i+k >= N.
        # So we can just index f_pad[M_add] and then mask where M_add >= N.
        # Wait, f_pad is length 2N. Indices 0..2N-1.
        # If M_add is in 0..N-1, it's valid.
        
        # Actually, simpler:
        # f_add_vals = np.where(M_add < N, f[M_add % N] ... no, modulo is wrong.
        
        # Let's use a lookup table approach or just mask.
        # f_add_raw = f_pad[M_add] # This works if M_add < 2N. Max M_add is 2N-2.
        # But we want 0 for M_add >= N.
        f_add = np.where(M_add < N, f_pad[M_add], 0.0)
        
        # For M_sub:
        # Valid if M_sub >= 0.
        # f[i-k]. 
        # We can reverse f? 
        # Or just mask.
        # f_sub_raw = f_pad[N + M_sub] ? 
        # If i-k = j, then j = i-k. 
        # We want f[j]. 
        # If we index f_pad with N + M_sub, that's N + i - k.
        # This doesn't directly give f[i-k].
        
        # Let's just use f directly with masks.
        # f_sub = np.where(M_sub >= 0, f[M_sub], 0.0) # Fails if M_sub < 0.
        
        # Safe indexing:
        # Clip indices to 0..N-1 for access, then mask.
        M_sub_safe = np.clip(M_sub, 0, N-1)
        f_sub = np.where(M_sub >= 0, f[M_sub_safe], 0.0)
        
        M_add_safe = np.clip(M_add, 0, N-1)
        f_add = np.where(M_add < N, f[M_add_safe], 0.0)
        
        # Delta[i, k] = -2 * f[i] * (f[i+k] + f[i-k])
        # f[i] is shape (N, 1)
        f_i = f[:, None]
        
        Delta = -2.0 * f_i * (f_add + f_sub)
        
        # NewR[i, k] = R[k] + Delta[i, k]
        # R is shape (N,), broadcast to (N, N)
        NewR = R[None, :] + Delta
        
        # We only care about k > 0.
        # NewR[:, 0] is irrelevant for the peak (k=0 is DC, we skip it in max).
        # Actually, the task says "upper bound on C3", usually max |R(k)| for k != 0.
        # So we take max over k=1..N-1.
        
        NewAbsR = np.abs(NewR[:, 1:])
        NewPeaks = np.max(NewAbsR, axis=1)
        
        # Find the best flip
        best_i = np.argmin(NewPeaks)
        best_peak = NewPeaks[best_i]
        
        if best_peak < current_peak - 1e-12:
            # Apply the flip
            f[best_i] = -f[best_i]
            R = R + Delta[best_i, :]
            current_peak = best_peak
            improved = True
        else:
            # No improvement found in this pass with this heuristic.
            # We might be stuck. 
            # To escape, we can try a random restart or a larger perturbation.
            # But for now, let's break or try a few random flips.
            # Given the vectorized speed, we can do many passes.
            # If we don't improve, we break.
            break

    # Final check: ensure we return a valid array
    return f

# EVOLVE-BLOCK-END
