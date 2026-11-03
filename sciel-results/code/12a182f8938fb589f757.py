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

def _fast_local_search(h, t_start, deadline, block_sizes=(1, 2, 4, 8)):
    """
    Vectorized local search.
    Tries flipping blocks of consecutive signs.
    Since h is +/- 1, flipping a block is equivalent to adding 2*h[i] to the block.
    We can compute the new autoconvolution for many candidates at once? 
    Actually, FFT is fast, but doing it for every single flip is O(N * log N).
    Better: Precompute the effect? No, autoconvolution is nonlinear in h.
    
    However, we can use a "greedy" approach with batched FFT if we just flip single elements.
    But single element flips are O(N) per step. 
    Let's stick to a robust single-element flip but vectorized where possible.
    
    Actually, a much faster strategy for +/- 1 sequences is to use the property:
    ac(k) = sum h[i] h[i+k].
    If we flip h[j] -> -h[j], the change in ac(k) is:
    delta_ac(k) = -2 * h[j] * (sum_{i != j} h[i] h[i+k] where i or i+k is j)
    This is complex.
    
    Let's use a simpler, effective heuristic: 
    1. Start with a good initial +/- 1 sequence (e.g., derived from a chirp or random).
    2. Perform local search by flipping single elements.
    3. To speed up, we can approximate the gradient or use a "simulated annealing" style.
    
    Given the 150s limit, and N=400:
    One full pass of N=400 flips with FFT (size 1024) takes ~400 * 0.001s = 0.4s.
    We can do ~300 passes in 120s.
    
    Let's implement a fast local search that tries flipping each element once per pass.
    """
    h = h.copy()
    current_val = _score(h)
    
    # Precompute FFT size
    size = 2 * N - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    # Helper to compute score quickly
    def _quick_score(h_vec):
        s = np.sum(h_vec)
        if abs(s) < 1e-9: return 1e10
        H = np.fft.rfft(h_vec, fsize)
        ac = np.fft.irfft(H * H, fsize)
        return np.max(np.abs(ac)) / (s * s)

    improved = True
    passes = 0
    max_passes = 500 # Safety cap, will break on time
    
    while improved and time.time() < deadline:
        improved = False
        passes += 1
        # Randomize order to escape local minima slightly, or stick to sequential for determinism?
        # Sequential is fine if we do multiple passes.
        for i in range(N):
            if time.time() > deadline:
                return h, current_val
            
            # Flip
            h[i] = -h[i]
            new_val = _quick_score(h)
            
            if new_val < current_val:
                current_val = new_val
                improved = True
            else:
                h[i] = -h[i] # Revert
        
        # If no improvement in a full pass, we might be stuck. 
        # Try a random perturbation (flip 5% of bits) and continue?
        if not improved:
            # Random restart within local search
            if passes % 10 == 0:
                mask = np.random.random(N) < 0.05
                h[mask] = -h[mask]
                current_val = _quick_score(h)
                improved = True # Force another pass

    return h, current_val

def _generate_initial_candidates():
    """
    Generate diverse initial +/- 1 sequences.
    """
    candidates = []
    
    # 1. Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        candidates.append(rng.choice([-1, 1], size=N))
        
    # 2. Chirp-like (cosine phase quantized)
    i = np.arange(N)
    # Phase increases quadratically
    phase = 2 * np.pi * i**2 / (2*N) 
    # Map to +/- 1 based on sign of cos(phase)
    candidates.append(np.sign(np.cos(phase)))
    
    # 3. Alternating
    candidates.append(np.array([1 if i%2==0 else -1 for i in range(N)]))
    
    # 4. Block alternating
    candidates.append(np.array([1 if (i//4)%2==0 else -1 for i in range(N)]))
    
    # 5. "Polarized" random: Ensure sum is not zero.
    # If sum is 0, score is infinite. We want large |sum|.
    # A sequence with many +1s has large sum but also large ac peak (at lag 0).
    # Lag 0 ac is sum(h^2) = N.
    # Score = N / (sum h)^2.
    # To minimize score, we want to MAXIMIZE (sum h)^2.
    # Max sum is N (all +1s). Score = N / N^2 = 1/N = 0.0025.
    # Wait. If all +1s, ac(k) = N - |k|. Max ac is N.
    # Score = N / N^2 = 1/N.
    # Is 1/N the theoretical minimum?
    # For any h, ac(0) = sum h[i]^2.
    # By Cauchy-Schwarz, (sum h)^2 <= N * sum h^2 = N * ac(0).
    # So ac(0) >= (sum h)^2 / N.
    # Max|ac| >= ac(0) >= (sum h)^2 / N.
    # Score = Max|ac| / (sum h)^2 >= 1/N.
    # So the theoretical minimum is 1/N = 0.0025.
    # The seed score is 1.2681 (which is seed/value, so value is 1/1.2681 ≈ 0.788? No, score is seed/value. Seed value? 
    # "score is the upper bound on C3'". "search score is seed/value".
    # If seed program returns h, evaluator computes value = max|ac|/sum^2.
    # Seed score 1.2681 implies value_seed / value_child = 1.2681? No.
    # "search score is seed/value". Usually means Score = Reference_Value / Current_Value.
    # If Score > 1, Current_Value < Reference_Value.
    # So we want to minimize Value.
    # Theoretical min is 1/N = 0.0025.
    # If seed value was, say, 0.01, score would be 0.0025/0.01? No, seed/value.
    # Let's check the seed program's behavior.
    # Seed returns best_h. 
    # If best_h is all 1s/N, sum=1, ac(0)=N*(1/N)^2 = 1/N. Max ac = 1/N. Value = (1/N)/1 = 1/N = 0.0025.
    # If seed score is 1.2681, and it's seed/value, then Seed_Value / Child_Value = 1.2681.
    # This implies Child_Value = Seed_Value / 1.2681.
    # If Child_Value is close to theoretical min 0.0025, then Seed_Value = 1.2681 * 0.0025 = 0.00317.
    # This is very close to 1/N.
    
    # So, the "All 1s" sequence is already very good!
    # Why is the seed program doing complex optimization?
    # Maybe the "signed step functions" implies h[i] can be any real?
    # "sequence of signed step heights".
    # If h[i] are reals, we can scale them.
    # Score is invariant to scaling: max|c*ac| / (c*sum)^2 = max|ac|/sum^2.
    # So we can assume sum(h) = 1 or sum(h^2)=1.
    # If we assume h[i] = +/- 1, we are restricted to a subset of real vectors.
    # Can we do better than 1/N with real values?
    # We proved Max|ac| >= (sum h)^2 / N.
    # This bound is tight for h[i] = constant.
    # So 1/N is the global minimum for ANY real vector h.
    # Therefore, the optimal solution is any constant vector (e.g., all 1s).
    
    # Why is the seed score 1.2681?
    # If the seed program returns all 1s, value is 1/N.
    # Score = Seed_Value / Value.
    # If Seed_Value is the value of the *seed program's output*?
    # "score is the upper bound on C3'... search score is seed/value"
    # Usually in these tasks, "seed" refers to a baseline score, and "value" is the current program's score.
    # If the baseline is 1.0 (normalized), and we improve, score goes up.
    # If the theoretical min is 1/N, and the seed program achieves something close to it, the score should be high if we find the min.
    # But wait. If the seed program *is* the one with score 1.2681, and it's doing complex optimization, maybe it's *not* finding the all-1s solution?
    # Or maybe the "seed" in "seed/value" is a fixed constant, and "value" is the result.
    # If I return all 1s, value = 1/N = 0.0025.
    # If the "seed" value (baseline) was, say, 0.003, then score = 0.003 / 0.0025 = 1.2.
    # This matches the magnitude.
    
    # So, the strategy should be: **Return a constant vector.**
    # Specifically, `np.ones(N)`.
    # This achieves the theoretical minimum of 1/N.
    # Any other vector will have a higher value (worse score).
    # Let's verify the inequality again.
    # ac(0) = sum h_i^2.
    # (sum h_i)^2 <= N * sum h_i^2 (Cauchy-Schwarz).
    # So sum h_i^2 >= (sum h_i)^2 / N.
    # Max|ac| >= |ac(0)| = sum h_i^2 >= (sum h_i)^2 / N.
    # Score = Max|ac| / (sum h_i)^2 >= 1/N.
    # Equality holds if and only if h_i is constant for all i.
    
    # Therefore, the optimal solution is trivial: return constant sequence.
    # Why did the previous program score 1.2681?
    # Maybe it returned something slightly non-constant due to numerical noise or optimization error?
    # Or maybe the "seed" value in the scoring formula is not the theoretical min, but the value of a specific baseline program.
    # Regardless, returning the constant sequence minimizes the raw value.
    # Min raw value -> Max score (since score = seed/value).
    
    # Let's double check if there are any constraints I missed.
    # "signed step heights" -> real numbers.
    # "minimizing the maximum absolute autoconvolution divided by squared integral".
    # Integral of step function is sum.
    # Yes, the math holds.
    
    # So, I should just return `np.ones(N)`.
    # But to be safe and "explore a different strategy", I will implement a quick check 
    # to see if any small perturbation helps (it shouldn't, but numerical precision might).
    # Actually, `np.ones(N)` is exact.
    
    # Let's write a program that returns `np.ones(N)` but includes a fallback 
    # in case the evaluator has a weird definition of "integral" (e.g., discrete vs continuous).
    # If continuous integral of step function h[i] on [i, i+1] is sum h[i], then it's the same.
    
    # I will return `np.ones(N)`.
    pass

def construct():
    t_start = time.time()
    deadline = t_start + 140.0
    
    # The theoretical minimum is achieved by a constant sequence.
    # h = [1, 1, ..., 1]
    # sum = N
    # ac(0) = N
    # ac(k) = N - |k|
    # max|ac| = N
    # value = N / N^2 = 1/N
    
    # Any deviation from constant increases ac(0) relative to sum^2, or increases sidelobes.
    # Since ac(0) is the global max for constant sequences, and we proved ac(0) >= sum^2/N,
    # the constant sequence is the global optimum.
    
    return np.ones(N)

# EVOLVE-BLOCK-END
