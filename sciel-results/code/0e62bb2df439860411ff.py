import time
import math
import zlib

_start_time = time.time()
_DEADLINE = _start_time + 280.0

# Precompute roots of unity for Z3
# omega = exp(2*pi*i/3) = -0.5 + i*sqrt(3)/2
# omega^2 = -0.5 - i*sqrt(3)/2
# omega^0 = 1
# We need to compute DFT over Z3^n.
# The DFT coefficient for frequency k is sum_{x in support} omega^{<k,x>}
# However, the "support" here is the single vector v itself? 
# No, the vector v is a point in {0,1,2}^n.
# The "function" represented by v is the delta function at v.
# The DFT of the delta function at v is 1 for all frequencies? No.
# Let's rethink the spectral model.
# A cap set is a subset S of G=Z3^n with no 3-term AP.
# The indicator function of S has Fourier transform.
# But we are adding points one by one.
# The priority function takes a candidate vector v.
# We want to choose v such that it doesn't create an AP with existing points.
# This is hard to know without the set.
# The strategy suggests: "prioritizing vectors based on the magnitude of their discrete Fourier transform coefficients... penalizing high-frequency components".
# This implies we treat the vector v itself as a signal? Or a pattern?
# If v is just a point, its "spectrum" is trivial.
# Perhaps the strategy means: look at the structure of v itself.
# Vectors with "regular" patterns (e.g., all 0s, or 010101) have strong low-frequency components?
# Or maybe: A vector v defines a direction. If we look at the line passing through 0 and v, 
# it contains 0, v, 2v. This is an AP.
# So any v is part of an AP with 0 and 2v.
# The greedy algorithm will likely pick 0 early (or not, depending on priority).
# If we pick 0, we must avoid all v and 2v.
# 
# Let's interpret "spectral problem" as analyzing the frequency content of the vector v 
# as a sequence of symbols.
# High-frequency components in a sequence of 0,1,2 might correspond to rapid changes.
# Low-frequency components correspond to smoothness.
# APs in Z3^n are linear structures. 
# Vectors that are "random" (high entropy, high frequency) might be safer 
# because they are less likely to align with existing linear structures?
# Or conversely, vectors that are "smooth" (low frequency) might be more structured 
# and thus more likely to form APs?
# Actually, random vectors are less likely to form APs with other random vectors 
# than structured ones?
# Let's try to compute a simple frequency-based score for the vector v.
# We can map the vector to a complex signal and compute its power spectrum.
# But n can be up to 30? 300 seconds limit.
# Computing DFT of length n is O(n log n). 
# But we are doing this for every candidate. 
# The number of candidates is 3^n. We can't evaluate all. 
# The evaluator calls priority() for a specific vector. 
# So we just compute the score for that one vector.
# 
# How to compute "magnitude of DFT coefficients"?
# Let's treat the vector v = (v_0, ..., v_{n-1}) as a signal.
# We can compute the DFT of this signal.
# However, the values are 0,1,2. 
# Let's center them: x_i = v_i - 1. So x_i in {-1, 0, 1}.
# Compute DFT: X_k = sum_{j=0}^{n-1} x_j * omega^{jk} where omega = exp(-2*pi*i/n).
# This is the standard DFT.
# High frequency = large k.
# We want to penalize high-frequency components? 
# "Penalizing high-frequency components that correlate with the formation of arithmetic progressions."
# This is a bit vague. 
# Let's try: Score = - (sum of |X_k|^2 for k > n/2) / (sum of |X_k|^2)
# This measures the "roughness" or "high-frequency content".
# If we penalize high frequency, we prefer smooth vectors.
# Smooth vectors in {0,1,2}^n are those with long runs of same value or gradual changes.
# Example: (0,0,0,1,1,1) is smoother than (0,1,0,1,0,1).
# Does smoothness help cap sets?
# A cap set in Z3^n is a set with no 3 distinct points in a line.
# Lines are affine subspaces of dimension 1.
# Vectors that are "smooth" might be less likely to lie on lines with other smooth vectors?
# Actually, structured vectors are more likely to lie on lines.
# (0,0,0) and (1,1,1) and (2,2,2) are on a line.
# (0,1,2) and (1,2,0) and (2,0,1) are on a line.
# These are "structured".
# Random vectors are less likely to be on lines with each other.
# So maybe we should PREFER high frequency (randomness)?
# The prompt says "penalizing high-frequency components". 
# Maybe the idea is that high-frequency noise creates local APs?
# Or maybe the "spectral problem" refers to the eigenvalues of the adjacency matrix?
# 
# Let's look at the inspiration: "target central, diverse vectors".
# Central vectors have low Hamming weight? Or balanced counts?
# 
# Let's try a different spectral approach:
# Compute the autocorrelation of the vector?
# Or simply: Count the number of "local" APs within the vector itself?
# A triple (i, j, k) with i,j,k distinct indices? No, indices are not in Z3.
# 
# Let's stick to the DFT idea but interpret it carefully.
# If we view the vector as a function on Z_n, its Fourier transform tells us about its periodicity.
# Vectors with strong low-frequency components are periodic or smooth.
# Vectors with high-frequency components are "noise".
# Cap sets are "independent" sets. Noise is more independent.
# So maybe we should REWARD high frequency?
# The prompt says "penalizing high-frequency components". 
# Let's follow the prompt. Maybe the logic is:
# High-frequency components in the *set* (not the vector) cause APs?
# But we only have the vector.
# 
# Alternative interpretation:
# The "vector" is a point. 
# Maybe we should look at the vector's coordinates as a polynomial?
# 
# Let's try a simpler proxy for "spectral structure":
# 1. Compute the Hamming weight (count of non-zeros).
# 2. Compute the "energy" of the first few Fourier modes.
# 
# Let's implement a fast DFT-like score.
# Since n is small (likely < 20 for the search to be feasible in 300s? 
# 3^20 is huge. The greedy algorithm picks one by one.
# The total number of calls to priority might be large, but n is fixed per evaluation.
# 
# Let's compute the power spectrum of the centered vector.
# We will use numpy if available, otherwise pure python.
# Numpy is allowed.
# 
# Strategy:
# 1. Center vector: x = v - 1
# 2. Compute FFT of x.
# 3. Compute power spectrum P = |FFT(x)|^2.
# 4. Normalize P.
# 5. Calculate "high-frequency energy" HFE = sum(P[k] for k in [n/2, n)) / sum(P).
# 6. Score = -HFE (Penalize high frequency).
# 7. Add a small random tie-breaker.
# 
# Why penalize high frequency?
# Hypothesis: Vectors that are "smooth" (low frequency) are more "aligned" with the grid structure, 
# but perhaps the greedy algorithm works better if it picks "smooth" vectors first 
# to establish a skeleton, then fills in? 
# Or maybe the opposite. 
# Let's try penalizing high frequency as instructed.
# 
# Wait, if I penalize high frequency, I am selecting smooth vectors.
# Smooth vectors like (0,0,0,0...) or (0,0,1,1...)
# (0,0,0,0) is the zero vector.
# (0,0,1,1) has an AP? No, it's a single vector.
# 
# Let's also consider the "center" idea.
# Vectors with counts close to n/3 for each symbol are "central".
# Let's combine:
# Score = -HFE + lambda * Centrality
# Centrality = 1 - (sum (c_i - n/3)^2)
# 
# Let's implement this.

import numpy as np

def priority(vector, n):
    if time.time() > _DEADLINE:
        return 0.0

    # Convert to numpy array
    v = np.array(vector, dtype=np.float64)
    
    # Center the values: map 0,1,2 to -1,0,1
    x = v - 1.0
    
    # If n is small, FFT is cheap. If n is large, it might be expensive.
    # But n is typically small in these combinatorial searches (n <= 10-15).
    try:
        # Compute FFT
        fft_x = np.fft.fft(x)
        power = np.abs(fft_x) ** 2
        
        # Total power
        total_power = np.sum(power)
        if total_power < 1e-9:
            # Zero vector (all 1s) or constant
            # Constant vectors have only DC component.
            # HFE = 0.
            hfe = 0.0
        else:
            # High frequency is the second half of the spectrum
            # For real input, the spectrum is symmetric.
            # Indices 0 to n-1.
            # High freq: n//2 to n
            hfe = np.sum(power[n//2:]) / total_power
            
    except Exception:
        hfe = 0.5 # Default to medium if FFT fails
    
    # Calculate Centrality
    # Counts of 0, 1, 2
    c0 = np.sum(v == 0)
    c1 = np.sum(v == 1)
    c2 = np.sum(v == 2)
    
    # Ideal count is n/3
    ideal = n / 3.0
    var = (c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2
    # Max variance is when one count is n, others 0. var = 2*n^2 + n^2 = 3*n^2?
    # (n-0)^2 + (0-0)^2 + (0-0)^2 = n^2. 
    # Wait, if c0=n, c1=0, c2=0.
    # (n - n/3)^2 + (0 - n/3)^2 + (0 - n/3)^2
    # = (2n/3)^2 + (-n/3)^2 + (-n/3)^2
    # = 4n^2/9 + n^2/9 + n^2/9 = 6n^2/9 = 2n^2/3.
    # So max var is 2n^2/3.
    # Normalize var to [0, 1]
    max_var = 2 * n * n / 3.0
    if max_var > 0:
        norm_var = var / max_var
    else:
        norm_var = 0.0
    
    centrality = 1.0 - norm_var
    
    # Combine scores
    # We penalize HFE (so we want low HFE)
    # We reward Centrality (we want high Centrality)
    
    # Weights: 
    # Let's try equal weight.
    score = -hfe + centrality
    
    # Tie-breaker: Hash
    h = 0
    for val in vector:
        h = (h * 37 + val) & 0xFFFFFFFF
    score += (h / 4294967296.0) * 0.1 # Small perturbation
    
    return float(score)
