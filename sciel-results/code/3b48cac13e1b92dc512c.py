import time
import hashlib
import math

# Global state
_start_time = None
_deadline = 280.0  # Safety margin before 300s/330s kill
_accepted_vectors = []
_n_cache = None
_ft_cache = {}  # Cache for Fourier Transform magnitudes: key=tuple(vector), val=float(sparsity_score)

def _get_fourier_sparsity(vec_tuple, n):
    """
    Computes a proxy for the sparsity of the DFT over Z_3^n.
    
    For a vector x in {0,1,2}^n, the DFT is defined as:
    F(k) = sum_{j=0}^{n-1} omega^{k_j x_j} for k in {0,1,2}^n
    where omega is a primitive 3rd root of unity.
    
    The magnitude squared |F(k)|^2 can be computed.
    Sparsity here is defined as the number of non-zero (or significant) frequency components.
    However, computing the full DFT is O(3^n), which is too slow.
    
    Instead, we use a structural proxy:
    Vectors with "regular" patterns (e.g., constant, or low entropy) tend to have
    sparser spectra. 
    Specifically, if a vector has many identical coordinates, its spectrum is more concentrated.
    
    A better proxy for "low complexity" in this context, which correlates with
    being part of a structured cap set (like a subspace or coset), is the
    "rank" or "entropy" of the coordinate distribution.
    
    However, the directive specifically asks for "minimizing the number of non-zero
    coefficients in their discrete Fourier transform".
    
    Since we cannot compute the full DFT, we approximate the spectral energy concentration.
    Vectors that are "simple" (e.g., all 0s, all 1s, or periodic) have high energy in
    few frequency bins.
    
    Let's use a different structural proxy that is computationally cheap but captures
    "algebraic simplicity":
    The number of distinct values in the vector is a lower bound on complexity.
    But we need something that differentiates vectors with the same distinct count.
    
    Let's look at the "autocorrelation" or "self-similarity".
    
    Actually, a key property of cap sets is that they avoid lines.
    Lines in F_3^n are defined by x, x+d, x+2d.
    
    Let's reconsider the "spectral" aspect.
    If we treat the vector as a signal, a "sparse spectrum" means the signal is a sum of
    few sinusoids. In Z_3, the "sinusoids" are the characters of the group.
    
    For a vector v, the "Fourier coefficients" are related to the sums of omega^{v_i k_i}.
    
    Let's implement a heuristic that approximates spectral sparsity:
    1. Count the frequency of each symbol (0, 1, 2).
    2. Vectors with highly imbalanced symbol frequencies (e.g., mostly 0s) have
       energy concentrated in the "zero frequency" component (the mean).
    3. Vectors with balanced frequencies (1/3 each) have energy spread out.
    
    Wait, the directive says "minimizing the number of non-zero coefficients".
    A constant vector (all 0s) has only one non-zero coefficient in its DFT (the DC term).
    A vector with random values has many non-zero coefficients.
    
    So, we should prioritize vectors that look more "constant" or "structured".
    This means favoring vectors with low entropy or high autocorrelation.
    
    Let's define a "Complexity Score" based on the entropy of the coordinate distribution.
    Lower entropy = more structured = sparser spectrum.
    
    Additionally, we can look at the "runs" of identical values.
    
    Let's combine:
    1. Entropy of symbol counts.
    2. A hash for tie-breaking.
    
    This is a structural change from "Sum of Squares" (which favors 1s/2s) to
    "Spectral Sparsity Proxy" (which favors low entropy/regularity).
    """
    if vec_tuple in _ft_cache:
        return _ft_cache[vec_tuple]
        
    # Calculate entropy of the coordinate distribution
    counts = [0, 0, 0]
    for v in vec_tuple:
        counts[v] += 1
        
    total = len(vec_tuple)
    if total == 0:
        score = 0.0
    else:
        entropy = 0.0
        for c in counts:
            if c > 0:
                p = c / total
                entropy -= p * math.log2(p)
        
        # Max entropy for n>0 is log2(3). Min is 0.
        # We want to MINIMIZE complexity, so lower entropy is better (higher priority).
        # So we invert the entropy for the score.
        # Score = MaxEntropy - Entropy
        max_entropy = math.log2(3)
        score = max_entropy - entropy
        
    # Add a small hash for tie-breaking to ensure determinism and spread
    # Using a fast hash
    h = 0
    base = 31
    for val in vec_tuple:
        h = (h * base + val) % (10**9 + 7)
    
    # The hash should be small relative to the entropy difference
    hash_score = (h / (10**9 + 7)) * 0.01
    
    final_score = score + hash_score
    
    _ft_cache[vec_tuple] = final_score
    return final_score

def priority(vector, n):
    global _start_time, _deadline, _accepted_vectors, _n_cache, _ft_cache
    
    if _start_time is None:
        _start_time = time.time()
        
    current_time = time.time()
    if current_time - _start_time > _deadline:
        # Fallback to a very fast static score if time is running out
        # Just return the hash to avoid any computation overhead
        h = 0
        base = 31
        for val in vector:
            h = (h * base + val) % (10**9 + 7)
        return float(h)

    vec_tuple = tuple(vector)
    
    # Check if n has changed to reset cache if necessary (though n is usually fixed per run)
    if _n_cache != n:
        _n_cache = n
        _ft_cache.clear()
        
    # Get the spectral sparsity proxy score
    # Higher score = Lower Complexity = Better
    score = _get_fourier_sparsity(vec_tuple, n)
    
    return float(score)
