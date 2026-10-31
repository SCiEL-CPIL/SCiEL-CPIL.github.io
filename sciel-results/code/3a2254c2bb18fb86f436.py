import time
import numpy as np

# Global state for safety and potential future use, though this strategy is static
_start_time = time.time()
_DEADLINE = 290.0  # Safety margin

def priority(vector, n):
    """
    Priority function based on spectral sparsity in Z_3^n.
    
    The hypothesis is that vectors with "simple" spectral representations
    (few non-zero Fourier coefficients) correspond to low-complexity algebraic
    structures that are less likely to form 3-term arithmetic progressions
    with other similarly structured vectors, or at least provide a better
    ordering for the greedy algorithm to discover large cap sets.
    
    We compute the Discrete Fourier Transform (DFT) of the vector treated as
    a function on Z_3^n. However, a full DFT of size 3^n is expensive.
    Instead, we can interpret the vector v in {0,1,2}^n as an element of
    the group Z_3^n. The characters of Z_3^n are indexed by k in Z_3^n.
    chi_k(v) = omega^{sum v_i k_i} where omega = exp(2*pi*i/3).
    
    Computing the full DFT (size 3^n) for each vector is O(3^n * n), which is
    too slow for n=10 (59049 * 10 * 59049 ~ 3.5e10).
    
    Alternative Interpretation:
    Perhaps "spectral sparsity" refers to the sparsity of the vector itself
    in a transformed basis? Or maybe we can approximate the spectral complexity
    using a smaller set of test frequencies?
    
    Let's reconsider the cost. If n is small (e.g., n<=6), 3^n is small.
    But for n=10, it's large.
    
    A different spectral approach:
    Map the vector v to a complex vector z in C^n where z_i = omega^{v_i}.
    Then compute the "energy" or "variance" of this complex vector?
    This is O(n).
    
    However, the prompt specifically suggests "minimizing the number of non-zero
    coefficients in their discrete Fourier transform over Z_3^n".
    
    If we cannot afford the full DFT, we can use a **Randomized Fourier Test**.
    We sample a fixed set of M frequency vectors k_j in Z_3^n.
    We compute the character values chi_{k_j}(v).
    If the vector is "spectrally simple", its projection onto random frequencies
    might have specific statistical properties?
    
    Actually, there is a simpler structural proxy for "spectral simplicity" in Z_3^n:
    Vectors that are constant on cosets of large subgroups have simple spectra.
    For example, vectors that depend on only a few coordinates (sparse support in
    coordinate sense) have a DFT that is supported on the dual subgroup.
    
    Let's try a **Coordinate Sparsity** heuristic combined with **Value Simplicity**.
    A vector that uses only {0, 1} (no 2s) is "binary".
    A vector that is constant is "trivial".
    
    But the directive says "different design family... spectral".
    
    Let's implement a **Truncated/Approximate Spectral Score** using a fixed
    set of random frequency vectors.
    
    Score = - (Number of non-zero character values among M random frequencies)
    
    Why? If a vector has a sparse spectrum, it has few non-zero Fourier coefficients.
    Sampling M frequencies, the expected number of non-zero hits is proportional
    to the spectral support size.
    
    Pre-compute M random frequency vectors.
    For each input vector v, compute chi_k(v) for k in K.
    Count how many are "non-zero" (i.e., not 1.0 + 0j, which corresponds to 0 phase).
    Actually, chi_k(v) = 1 if sum v_i k_i == 0 mod 3.
    So "non-zero coefficient" in the sense of the DFT being non-zero everywhere
    unless the function is a character? No.
    
    Let's look at the DFT of the *indicator* function of the set {v}?
    No, we score a single vector.
    
    Let's stick to the **Randomized Character Test**.
    We define "spectral complexity" as the number of distinct character values
    observed, or the "entropy" of the character values.
    
    Better:
    Compute the sum of squares of the character values? They are all 1.
    
    Let's try **Variance of Character Values** over a set of random frequencies.
    If the vector is "random", the character values are uniformly distributed in {1, omega, omega^2}.
    The variance will be high.
    If the vector is "simple" (e.g., constant 0), the character value is always 1.
    The variance is 0.
    
    So: **Minimize Variance of Character Values**.
    
    Steps:
    1. Pre-define a set of R random frequency vectors k_1, ..., k_R in Z_3^n.
    2. For a given vector v, compute c_j = sum(v_i * k_j_i) mod 3 for j=1..R.
    3. Map c_j to complex values z_j = exp(2*pi*i*c_j/3).
    4. Compute variance of z_j.
    5. Priority = -Variance.
    
    This is O(R*n). If R=100, n=10, that's 1000 ops per vector.
    59049 vectors * 1000 = 6e7 ops. Very fast in NumPy if vectorized.
    """
    
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        return 0.0
    
    # Pre-computed random frequencies
    # We need a deterministic set. Use a fixed seed.
    # We'll generate them on the first call or at module load?
    # Module load is better to avoid first-call latency.
    
    global _random_freqs
    if not hasattr(priority, '_random_freqs'):
        # Generate 200 random frequency vectors in Z_3^n
        # We don't know n at module load, so we must generate on first call
        # and cache by n.
        priority._random_freqs = {}
        
    if n not in priority._random_freqs:
        np.random.seed(42 + n)  # Deterministic per n
        # 200 random vectors in {0,1,2}^n
        priority._random_freqs[n] = np.random.randint(0, 3, size=(200, n))
        
    freqs = priority._random_freqs[n]
    
    v_arr = np.array(vector, dtype=np.int64)
    
    # Compute dot products mod 3
    # (200, n) dot (n,) -> (200,)
    dots = np.dot(freqs, v_arr) % 3
    
    # Map to complex roots of unity
    # 0 -> 1+0j, 1 -> -0.5+0.866j, 2 -> -0.5-0.866j
    # We can use np.exp(2j * np.pi * dots / 3.0)
    # But for speed, we can map to integers and use variance of the real/imag parts?
    # Or just use the values directly.
    
    # Variance of the complex numbers.
    # Var(z) = E[|z|^2] - |E[z]|^2 = 1 - |Mean(z)|^2
    # So minimizing Variance is equivalent to Maximizing |Mean(z)|^2.
    
    mean_z = np.mean(np.exp(2j * np.pi * dots / 3.0))
    magnitude_sq = np.abs(mean_z) ** 2
    
    # Higher magnitude_sq means lower variance, meaning "simpler" spectrum.
    # We want to prioritize these.
    # So priority = magnitude_sq.
    
    # Add a small tie-breaker based on Hamming weight to avoid ties
    # Vectors with lower Hamming weight are "sparser" in coordinate space.
    hamming_weight = np.sum(v_arr != 0)
    
    # Combine: Primary is spectral simplicity, secondary is coordinate sparsity.
    # Scale spectral score to [0, 1].
    # Max possible magnitude_sq is 1.
    
    score = magnitude_sq * 100.0 - hamming_weight * 0.1
    
    return float(score)
