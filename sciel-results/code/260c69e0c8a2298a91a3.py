"""
Strategy: Ising Spin Chain Minimization with Long-Range Interactions.

The task is to find a signed step function f (vector of ±1) of length N=400
that minimizes the upper bound on C3, which is related to the peak sidelobe
level (PSL) of the autoconvolution.

The current program uses Simulated Annealing with random block flips.
This new approach frames the problem as finding the ground state of an
Ising spin chain with long-range interactions. The "energy" is the PSL,
but we use a gradient-like heuristic derived from the autoconvolution
structure to guide flips. Specifically, flipping a spin at index i changes
the autoconvolution at all lags. We approximate the impact of a flip
by looking at the current autoconvolution values. If the autoconvolution
at a lag is large and positive, flipping spins that contribute positively
to that lag (i.e., where f[i] == f[i+lag]) will reduce the peak.

We implement a "Correlated Flip" strategy:
1. Compute the linear autoconvolution c.
2. Identify the peak magnitude M = max(|c[1:]|).
3. For each lag k where |c[k]| is close to M, determine which pairs (i, i+k)
   are "aligned" (f[i]*f[i+k] == 1) and contributing to the peak.
4. Flip spins that are involved in the most "aligned" pairs at high-magnitude lags.
5. Use Simulated Annealing to accept/reject these correlated moves.
"""

import numpy as np
import time

N = 400

def _linear_autoconv(f):
    """
    Compute the linear autoconvolution of f.
    Returns array of length 2*N - 1.
    """
    # FFT method for speed
    size = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(f, n=size)
    conv = np.fft.irfft(F * F, n=size)[:2 * N - 1]
    return conv

def _score(f):
    """
    Score is the peak sidelobe level (PSL) of the linear autoconvolution.
    Lower is better.
    """
    conv = _linear_autoconv(f)
    # The zero-lag term is conv[N-1] (center of the convolution for even length? 
    # Wait. Linear convolution of two sequences of length N has length 2N-1.
    # The peak (lag 0) is at index N-1.
    # Side lobes are all other indices.
    if N == 1:
        return 0.0
    # Exclude the main peak at N-1
    # We want max abs of the rest.
    # Create a mask
    mask = np.ones(len(conv), dtype=bool)
    mask[N-1] = False
    return np.max(np.abs(conv[mask]))

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We stop at 140s.
    deadline = start_time + 140.0
    
    rng = np.random.RandomState(42)
    
    # Initialize
    # Start with a random sequence, or maybe a structured one like a chirp?
    # Random is a safe baseline, but let's try a few random starts.
    best_f = rng.choice([-1.0, 1.0], size=N)
    best_score = _score(best_f)
    
    current_f = best_f.copy()
    current_score = best_score
    
    # Temperature for Simulated Annealing
    T = 1.0
    T_min = 0.001
    cooling_rate = 0.999
    
    # Pre-compute indices for lags
    # We focus on the top K lags with highest magnitude to guide flips
    top_k_lags = 50
    
    while time.time() < deadline - 5.0:
        # 1. Compute current autoconvolution
        conv = _linear_autoconv(current_f)
        
        # 2. Identify high-magnitude lags (side lobes)
        # We need to find lags k (1 <= k < N) where |conv[N-1+k]| or |conv[N-1-k]| is large.
        # Due to symmetry, we just look at the positive lags part of the conv array.
        # Indices in conv: 0 to 2N-2.
        # Lag 0 is at N-1.
        # Lag k > 0 corresponds to conv[N-1+k] and conv[N-1-k].
        
        # Extract side lobe magnitudes
        # We'll just look at the whole array excluding center
        side_lobe_indices = np.arange(2*N - 1)
        side_lobe_indices = side_lobe_indices[side_lobe_indices != N-1]
        
        # Get top K magnitudes
        if len(side_lobe_indices) > top_k_lags:
            # We want the indices in the original conv array that have the highest abs values
            # Sort side_lobe_indices by abs(conv[idx])
            sorted_indices = side_lobe_indices[np.argsort(np.abs(conv[side_lobe_indices]))][::-1][:top_k_lags]
        else:
            sorted_indices = side_lobe_indices
            
        # 3. Determine which flips would reduce these peaks
        # For a lag k (distance), the contribution to conv[N-1+k] is sum(f[i] * f[i+k] for i=0..N-1-k)
        # Flipping f[j] changes the sign of terms where j is involved in a pair.
        # If f[j] * f[j+k] == 1, they contribute +1. Flipping one makes it -1 (change -2).
        # If f[j] * f[j+k] == -1, they contribute -1. Flipping one makes it +1 (change +2).
        
        # We want to reduce |conv[lag]|.
        # If conv[lag] > 0, we want to change +1 to -1 (flip aligned pairs).
        # If conv[lag] < 0, we want to change -1 to +1 (flip anti-aligned pairs).
        
        # Let's accumulate a "pressure" for each spin j.
        # pressure[j] represents how much flipping j would help reduce the total peak energy.
        pressure = np.zeros(N)
        
        for idx in sorted_indices:
            # idx is an index in the conv array.
            # Determine the lag k
            if idx > N - 1:
                k = idx - (N - 1)
            else:
                k = (N - 1) - idx
                
            if k == 0:
                continue
                
            val = conv[idx]
            # We want to flip pairs (i, i+k) such that they contribute to val.
            # Pairs are (i, i+k) for i in 0..N-1-k.
            # Contribution is f[i] * f[i+k].
            # If val > 0, we prefer f[i]*f[i+k] = -1. So if it's 1, flip one of them.
            # If val < 0, we prefer f[i]*f[i+k] = 1. So if it's -1, flip one of them.
            
            # Let's just add weight to spins involved in "bad" pairs.
            # Bad pair: sign(f[i]*f[i+k]) == sign(val)
            # We can't easily flip "one of them" without breaking symmetry or double counting.
            # Let's just flip a random spin involved in a bad pair with probability proportional to |val|.
            
            # To keep it simple and fast, let's pick one random lag from the top few
            # and flip a random spin involved in a conflicting pair.
            pass
            
        # Simpler Heuristic Move:
        # Pick a random high-magnitude lag.
        # Find a random pair (i, i+k) that is "aligned" with the sign of the lag value.
        # Flip one of the spins in that pair.
        
        if len(sorted_indices) > 0:
            # Weighted random selection of a lag index based on magnitude
            # Actually, just pick the top one or a random one from top K
            # Let's pick a random one from the top K to explore
            idx = sorted_indices[rng.randint(len(sorted_indices))]
            k = abs(idx - (N - 1))
            if k == 0 or k >= N:
                # Fallback: random flip
                j = rng.randint(N)
                new_f = current_f.copy()
                new_f[j] *= -1.0
            else:
                val = conv[idx]
                # We want to flip a pair that contributes with the same sign as val.
                # Pairs (i, i+k) for i in 0..N-1-k
                # We need f[i]*f[i+k] to have the same sign as val.
                # Let's pick a random i in range(0, N-k)
                i = rng.randint(0, N - k)
                j1 = i
                j2 = i + k
                
                prod = current_f[j1] * current_f[j2]
                # If prod * val > 0, they are aligned with the peak. Flipping one reduces the peak magnitude.
                # If prod * val < 0, flipping one increases the peak magnitude (bad).
                # So we only flip if they are aligned.
                if prod * val > 0:
                    # Flip one of them randomly
                    if rng.rand() < 0.5:
                        j = j1
                    else:
                        j = j2
                    new_f = current_f.copy()
                    new_f[j] *= -1.0
                else:
                    # If not aligned, flipping increases error. 
                    # Instead, let's try a random flip elsewhere or skip.
                    # Let's try a random flip to escape local minima.
                    j = rng.randint(N)
                    new_f = current_f.copy()
                    new_f[j] *= -1.0
        
        else:
            # No side lobes? Unlikely. Random flip.
            j = rng.randint(N)
            new_f = current_f.copy()
            new_f[j] *= -1.0
            
        new_score = _score(new_f)
        
        delta = new_score - current_score
        if delta < 0 or (T > T_min and rng.rand() < np.exp(-delta / T)):
            current_f = new_f
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_f = current_f.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Final Polish: Greedy single flip search
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        for i in range(N):
            if time.time() > deadline - 1.0:
                break
            old_val = best_f[i]
            best_f[i] = -old_val
            score_new = _score(best_f)
            if score_new < best_score:
                best_score = score_new
                improved = True
            else:
                best_f[i] = old_val

    return best_f

# EVOLVE-BLOCK-END
