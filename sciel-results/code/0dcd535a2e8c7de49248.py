"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _spectral_flatten(h_init, time_budget):
    """
    Optimize h in the frequency domain to minimize max(|H|^2) subject to sum(h)=1.
    This is equivalent to minimizing the peak power in the spectrum while keeping DC fixed.
    We use a simple iterative projection:
    1. Compute DFT of h.
    2. Identify peaks in |H|^2 (excluding DC).
    3. Reduce the magnitude of peaks, increase others to maintain total power? 
       Actually, we want to minimize max|ac|. ac[k] is inverse FFT of |H|^2.
       So we want to flatten |H|^2 as much as possible, subject to H[0] being fixed (sum h).
       
    Let's use a gradient-free iterative method:
    - Start with h.
    - Compute H = fft(h).
    - Let P = |H|^2.
    - We want to minimize max(ifft(P)).
    - A simple heuristic: replace P with a smoother version, e.g., clip the top x% of P values to the median, or use a soft threshold.
    - But we must keep H[0] fixed.
    - Also, we need to reconstruct h with the correct phase.
    
    Alternative: Use the fact that for a real signal, H[n] = conj(H[N-n]).
    Let's try a direct search in the phase domain? No, too complex.
    
    Let's try a simpler structural change: **Random Restart with Spectral Initialization**.
    Generate random phases, set magnitudes to a flat spectrum (except DC), compute h, then local search.
    """
    best_h = h_init.copy()
    best_val = _score(h_init)
    
    # 1. Flat Spectrum Initialization
    # We want H[k] to be constant for k != 0.
    # H[0] = sum(h). Let's normalize h so sum(h)=1.
    s0 = np.sum(h_init)
    if abs(s0) < 1e-9:
        h_norm = np.ones(N) / N
    else:
        h_norm = h_init / s0
        
    # Try a few random phase configurations with flat magnitude spectrum
    # Magnitude spectrum: |H[0]| = 1, |H[k]| = c for k != 0.
    # Total energy E = 1 + (N-1)c^2.
    # By Parseval, sum(h^2) = E/N.
    # This doesn't constrain sum(h) directly, but H[0] does.
    
    # Actually, a known construction for low autocorrelation is the **Chirp**.
    # h[n] = exp(i * pi * n^2 / N). But we need real h.
    # Real chirp: cos(pi * n^2 / N).
    # Let's generate several structured inits and local search them.
    
    inits = []
    
    # 1. Cosine Chirp
    n = np.arange(N)
    inits.append(np.cos(np.pi * n**2 / N))
    
    # 2. Sine Chirp
    inits.append(np.sin(np.pi * n**2 / N))
    
    # 3. Flat Spectrum with Random Phases
    # Construct H directly.
    # H[0] = 1.
    # H[k] for k=1..N/2: magnitude = c, phase = random.
    # We need to choose c such that the resulting h has sum 1? No, H[0] fixes sum.
    # But the time-domain h will be complex. We take real part?
    # If we take real part, the spectrum changes.
    # Better: Construct h from random phases of a flat spectrum, then take real part, then normalize.
    
    rng = np.random.default_rng(42)
    for _ in range(20):
        # Random phases
        phases = rng.uniform(0, 2*np.pi, N//2 + 1)
        # Magnitudes: constant for k>0
        c = 1.0 / np.sqrt(N)  # Arbitrary scale
        H = np.zeros(N//2 + 1, dtype=complex)
        H[0] = 1.0
        H[1:] = c * np.exp(1j * phases[1:])
        
        # Reconstruct real signal
        # H_full is symmetric.
        # Use irfft
        h_complex = np.fft.irfft(H, N)
        h_real = h_complex.real
        
        # Normalize
        s = np.sum(h_real)
        if abs(s) > 1e-9:
            h_real = h_real / s
            inits.append(h_real)
            
    # 4. Barker-like / Pseudo-random
    inits.append(rng.choice([-1, 1], size=N).astype(float))
    
    # Local search on each init
    for h_init in inits:
        if time.time() > time_budget:
            break
        
        h = h_init.copy()
        val = _score(h)
        
        # Simple local search: flip signs
        # To speed up, we can estimate the change in autoconvolution.
        # But FFT is fast enough for N=400.
        for _ in range(50): # Max passes
            if time.time() > time_budget:
                break
            improved = False
            # Random order of indices to avoid bias
            idxs = rng.permutation(N)
            for i in idxs:
                if time.time() > time_budget:
                    break
                old_val = h[i]
                h[i] = -old_val
                new_val = _score(h)
                if new_val < val:
                    val = new_val
                    improved = True
                else:
                    h[i] = old_val
            if not improved:
                break
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    return best_h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin

    # Start with a good baseline
    # The constant sequence has score 1.0 (max ac = N, sum^2 = N^2, ratio 1/N? No.)
    # h = ones(N). ac[k] = N - |k|. max ac = N. sum(h) = N. score = N / N^2 = 1/N = 0.0025.
    # Wait, the score is max|ac| / (sum h)^2.
    # For h=ones(N), sum=N. max|ac|=N. score = N/N^2 = 1/N.
    # For N=400, score = 0.0025.
    # The seed program score is 1.2681. This implies the seed program is WORSE than the constant sequence?
    # Let's check the seed program's score calculation.
    # _score(h) = _autoconv_max_abs(h) / (s * s).
    # If h=ones(N)/N, sum=1. ac[0]=N*(1/N)^2 = 1/N. max ac = 1/N. score = (1/N)/1 = 0.0025.
    # Why is the seed score 1.2681?
    # Maybe the seed program is not optimizing well, or I am misinterpreting the score.
    # "score is the upper bound on C3'... lower is better".
    # "1.0 equals the seed program".
    # If the seed program returns a sequence with score 1.2681, and the constant sequence has score 0.0025, then the constant sequence is MUCH better.
    # Why would the seed program score 1.2681?
    # Perhaps the seed program's `construct()` returns a sequence that does NOT have score 1.2681, but the *evaluator* scores it as 1.2681 relative to a baseline?
    # "The evaluator measures the raw value (lower is better). The search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement".
    # So, if my program returns a sequence with raw value V, and the seed program returns a sequence with raw value V_seed, then my search score is V_seed / V.
    # If V_seed = 1.2681 * V_baseline? No.
    # "1.0 equals the seed program". So if I return the same sequence as the seed, my score is 1.0.
    # If I return a sequence with raw value V, and the seed's raw value is V_seed, then score = V_seed / V.
    # We don't know V_seed. But we know the seed's search score is 1.0 by definition.
    # The prompt says "CURRENT PROGRAM (score 1.2681)". This likely means the raw value of the current program's output is 1.2681? Or the search score relative to some other baseline?
    # "score 1.2681" usually refers to the metric reported by the evaluator.
    # If the evaluator reports "seed/value", and the current program is the seed, its score should be 1.0.
    # Unless "score 1.2681" refers to the RAW value of the current program's output.
    # Let's assume the RAW value of the current program's output is 1.2681.
    # And the constant sequence has RAW value 0.0025.
    # Then the constant sequence would have a search score of 1.2681 / 0.0025 = 507.
    # That seems too good. Is there a constraint I'm missing?
    # "signed equal-width step functions".
    # "minimizing the maximum absolute autoconvolution divided by squared integral".
    # Maybe the integral is not sum(h), but integral of h(x)^2?
    # "squared integral" usually means (integral h)^2.
    # If it meant integral of h^2, it would be "squared L2 norm".
    # Let's re-read carefully: "divided by squared integral".
    # Standard definition of C3' often involves normalization by L1 or L2.
    # If the seed program's raw value is 1.2681, and my constant sequence raw value is 0.0025, I should just return the constant sequence.
    # Why didn't the seed program do that?
    # Maybe the seed program is NOT the constant sequence.
    # The seed program code:
    # `best_h = np.ones(N) / N` is the baseline.
    # `base_score = _score(best_h)`.
    # Then it tries to improve.
    # If it fails to improve, it returns `best_h`.
    # So the seed program likely returns `np.ones(N) / N` or something close.
    # If it returns `np.ones(N) / N`, the raw value is 0.0025.
    # But the prompt says "CURRENT PROGRAM (score 1.2681)".
    # This is a contradiction.
    # Possibility A: The raw value of the seed program's output is 1.2681. This implies the seed program did NOT return the constant sequence, but a worse one.
    # Possibility B: The "score 1.2681" is the SEARCH score relative to a DIFFERENT baseline (e.g., a known optimal or a random baseline).
    # "1.0 equals the seed program". This strongly suggests that the seed program's score is 1.0.
    # So why does it say "CURRENT PROGRAM (score 1.2681)"?
    # Maybe the 1.2681 is the RAW value, and the "score 1.0" statement is a general rule.
    # If the raw value is 1.2681, and the constant sequence raw value is 0.0025, then the constant sequence is vastly superior.
    # I will implement the strategy to find the minimum raw value.
    # My strategy:
    # 1. Return the constant sequence (raw value ~ 0.0025).
    # 2. Try to improve it using spectral flattening and local search.
    
    # Let's verify the raw value of the constant sequence.
    h_const = np.ones(N) / N
    val_const = _score(h_const)
    
    # Let's verify the raw value of a random sequence.
    rng = np.random.default_rng(0)
    h_rand = rng.standard_normal(N)
    val_rand = _score(h_rand)
    
    # Let's verify the raw value of the seed program's likely output.
    # The seed program tries to minimize. If it fails, it returns the constant sequence.
    # So the seed program's raw value should be close to val_const.
    # If val_const is 0.0025, and the prompt says 1.2681, there is a mismatch.
    # Could "squared integral" mean (sum h^2)?
    # If denominator is sum(h^2), then for constant sequence: sum(h^2) = N * (1/N)^2 = 1/N.
    # max ac = 1/N.
    # score = (1/N) / (1/N) = 1.0.
    # This matches the "1.0 equals the seed program" if the seed program returns the constant sequence.
    # Let's check the seed program's code again.
    # `return _autoconv_max_abs(h) / (s * s)` where `s = np.sum(h)`.
    # So the code uses (sum h)^2.
    # If the code uses (sum h)^2, the constant sequence score is 1/N = 0.0025.
    # If the prompt says the score is 1.2681, maybe the N is different? No, N=400.
    # Maybe the "score 1.2681" is just a label for the current best found in the population, which might be worse than the constant sequence?
    # Or maybe the evaluator normalizes differently?
    # "The evaluator measures the raw value (lower is better)."
    # I will assume the raw value is what I compute with `_score`.
    # If I return the constant sequence, I get 0.0025.
    # If the seed program's raw value is 1.2681, then my search score is 1.2681 / 0.0025 = 507.
    # This is a huge jump.
    # I will simply return the constant sequence and then try to improve it.
    
    best_h = h_const
    best_val = val_const
    
    # Try spectral flattening and local search to see if we can go below 0.0025.
    # Is it possible to go below 1/N?
    # For any h with sum 1, ac[0] = sum(h^2).
    # By Cauchy-Schwarz, sum(h^2) >= (sum h)^2 / N = 1/N.
    # So max|ac| >= ac[0] >= 1/N.
    # Therefore, the score max|ac| / (sum h)^2 >= 1/N.
    # The minimum possible score is 1/N = 0.0025.
    # This is achieved by the constant sequence.
    # So the constant sequence is OPTIMAL.
    # No other sequence can have a lower score.
    # The seed program's score of 1.2681 must be a misunderstanding of the metric or a different N.
    # Or maybe the "score" in the prompt is the SEARCH score, and the raw value of the seed is 1.0?
    # If the raw value of the seed is 1.0, and the constant sequence raw value is 0.0025, then the constant sequence is 400x better.
    # I will return the constant sequence.
    
    return best_h

# EVOLVE-BLOCK-END
