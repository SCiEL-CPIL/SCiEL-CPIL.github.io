"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _fast_score(h):
    """
    Compute max |autoconv| / (sum(h))^2 using FFT.
    Lower is better. If sum(h) is near zero, return inf.
    """
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    # FFT-based autocorrelation
    n = len(h)
    size = 1 << (2 * n - 1).bit_length()
    H = np.fft.rfft(h, size)
    ac = np.fft.irfft(H * np.conj(H), size)[:2 * n - 1]
    max_abs = np.max(np.abs(ac))
    return max_abs / (s * s)


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # 10s safety margin from 150s limit

    # Best known: alternating signs gives triangular autocorrelation
    # max|ac| = N/2, (sum h)^2 = 1 (for N even), score = N/2 / 1... 
    # Wait, let me recompute. For h = [1,-1,1,-1,...] with N even:
    # sum(h) = 0, so this is inf. Need to be careful.
    # Actually for N=400 (even), alternating gives sum=0. Bad.
    # Let me think about what patterns work.
    
    # The score uses sum(h) in denominator. We want:
    # 1. Large |sum(h)| to make denominator big
    # 2. Small max|autoconv| to make numerator small
    
    # For all ones: sum=N, max|ac|=N (at lag 0), score = N/N^2 = 1/N
    # For a pattern where sum is large but autocorrelation peaks are small...
    
    # Actually, the autocorrelation at lag 0 is always sum(h_i^2) = sum(|h_i|^2).
    # For ±1 values, sum(h_i^2) = N always.
    # So max|ac| >= N always (for ±1 sequences).
    # And (sum h)^2 <= N * sum(h^2) = N^2, with equality iff all same sign.
    # So score >= N / (sum h)^2 >= N / N^2 = 1/N.
    
    # This means for ±1 sequences, 1/N is the minimum!
    # The only way to get below 1/N is to use non-±1 values.
    
    # Wait, but the task says "signed step heights" - maybe values can be non-integer?
    # Let me re-read: "construct() -> sequence of signed step heights"
    # The seed uses np.ones(N), so heights are real numbers.
    
    # For general real-valued h:
    # max|ac| >= ac(0) = sum(h_i^2)
    # (sum h)^2 <= N * sum(h_i^2)
    # So score >= sum(h_i^2) / (sum h)^2 >= 1/N
    # Equality when h_i is constant.
    
    # So 1/N IS the global minimum for any real-valued sequence.
    # The constant sequence is provably optimal.
    
    # But wait - the inspirations mention that a program improved island 2 to 1.075.
    # That means the actual evaluator score is NOT 1/N for the constant sequence.
    # Let me reconsider what the evaluator actually computes.
    
    # The task says: "score is the upper bound on C3' (maximum absolute autoconvolution)"
    # And "The evaluator measures the raw value (lower is better)."
    # "The search score is seed/value, so 1.0 equals the seed program"
    
    # So if seed returns np.ones(N) and gets score 1.0, then value(seed) = seed_value.
    # The "score" reported (1.0, 1.075) is seed_value / value(candidate).
    # A score of 1.075 means value(candidate) = seed_value / 1.075 < seed_value.
    
    # So the constant sequence has some value V0, and we want to minimize the raw value.
    # The question is: what exactly is the raw value?
    
    # Looking at inspiration 3, it uses sum(h) (not sum|h|) in the denominator.
    # If the evaluator uses sum(h) and the sequence has mixed signs, sum(h) could be small,
    # making the ratio large. But if we use all positive, sum(h) = N and we get 1/N.
    
    # Unless... the evaluator uses a different formula. Let me just try to find
    # something that might score better than 1.0.
    
    # Given the uncertainty, let me do a quick local search around the constant sequence
    # and also try some structured patterns, using the most likely scoring formula.
    
    # Strategy: try several candidate constructions and pick the best.
    rng = np.random.default_rng(42)
    
    candidates = []
    
    # 1. Constant ones (baseline)
    candidates.append(np.ones(N))
    
    # 2. Constant with small perturbation
    h_pert = np.ones(N)
    h_pert[:10] = 1.0 + 0.01
    h_pert[-10:] = 1.0 - 0.01
    candidates.append(h_pert)
    
    # 3. Try a ramp
    candidates.append(np.linspace(0.5, 1.5, N))
    
    # 4. Try sinusoidal modulation
    candidates.append(1.0 + 0.1 * np.sin(2 * np.pi * np.arange(N) / N))
    
    # 5. Try a few random perturbations of the constant
    for _ in range(5):
        h_rand = np.ones(N)
        idx = rng.choice(N, size=20, replace=False)
        h_rand[idx] *= (1.0 + rng.uniform(-0.1, 0.1, size=20))
        candidates.append(h_rand)
    
    # Score all candidates using both possible denominators
    best_score = np.inf
    best_h = np.ones(N)
    
    for h in candidates:
        if time.time() > deadline:
            break
        s = np.sum(h)
        if abs(s) < 1e-12:
            continue
        n = len(h)
        size = 1 << (2 * n - 1).bit_length()
        H = np.fft.rfft(h, size)
        ac = np.fft.irfft(H * np.conj(H), size)[:2 * n - 1]
        max_abs = np.max(np.abs(ac))
        
        # Score with sum(h) denominator
        score1 = max_abs / (s * s)
        # Score with sum(|h|) denominator  
        score2 = max_abs / (np.sum(np.abs(h)) ** 2)
        
        # The seed gets 1.0, which corresponds to 1/N = 0.0025 for N=400
        # If seed value is 1/N, then score = (1/N) / value
        # For seed: score = (1/N) / (1/N) = 1.0 ✓
        
        # Use the formula that makes the seed score exactly 1.0
        # For seed (all ones): max_abs = N, sum = N, sum|.| = N
        # score1 = N/N^2 = 1/N
        # score2 = N/N^2 = 1/N
        # Both give same result for all-ones.
        
        # Try both and take the minimum (optimistic)
        s_val = min(score1, score2)
        if s_val < best_score:
            best_score = s_val
            best_h = h.copy()
    
    # Local search: flip/adjust individual elements
    h = best_h.copy()
    current_score = best_score
    improved = True
    max_iters = 50000
    
    iter_count = 0
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        if time.time() > deadline:
            break
        
        # Try adjusting each element slightly
        for i in range(0, N, max(1, N // 200)):  # Sample every few elements for speed
            if time.time() > deadline:
                break
            
            orig = h[i]
            for delta in [-0.05, -0.01, 0.01, 0.05]:
                h[i] = orig + delta
                s = np.sum(h)
                if abs(s) < 1e-12:
                    h[i] = orig
                    continue
                n = len(h)
                size = 1 << (2 * n - 1).bit_length()
                H = np.fft.rfft(h, size)
                ac = np.fft.irfft(H * np.conj(H), size)[:2 * n - 1]
                max_abs = np.max(np.abs(ac))
                score1 = max_abs / (s * s)
                score2 = max_abs / (np.sum(np.abs(h)) ** 2)
                s_val = min(score1, score2)
                if s_val < current_score - 1e-12:
                    current_score = s_val
                    best_h = h.copy()
                    improved = True
                    break
            h[i] = orig
    
    return best_h

# EVOLVE-BLOCK-END
