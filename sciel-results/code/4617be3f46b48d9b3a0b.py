import numpy as np
import time

# ---- Task constants (interface) ----
N = 50
DELTA = 0.5 / N
_XC = -0.25 + (np.arange(N) + 0.5) * DELTA

_DEADLINE = 140.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _ratio(h):
    """Raw objective: L2^2 / (L1 * max_autoconv)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < 0):
        return -np.inf
    l1 = np.sum(h)
    if l1 <= 0.0:
        return 0.0
    l2_sq = np.dot(h, h)
    if l2_sq <= 0.0:
        return 0.0
    conv = np.convolve(h, h, mode="full")
    m = np.max(conv)
    if m <= 0.0:
        return 0.0
    return l2_sq / (l1 * m)

def _find_sidon_set(N, max_k=10, rng=None):
    """
    Find a Sidon set in [0, N) of size up to max_k.
    Uses a greedy randomized approach.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    best_set = []
    
    for _ in range(100):
        S = []
        sums = set()
        candidates = list(range(N))
        rng.shuffle(candidates)
        
        for c in candidates:
            if len(S) >= max_k:
                break
            conflict = False
            # Check c + c
            if c + c in sums:
                conflict = True
            else:
                # Check c + s for s in S
                for s in S:
                    if c + s in sums:
                        conflict = True
                        break
                if not conflict:
                    S.append(c)
                    sums.add(c + c)
                    for s in S:
                        sums.add(c + s)
        
        if len(S) > len(best_set):
            best_set = S[:]
            if len(best_set) >= max_k:
                break
                
    return best_set

def construct():
    np.random.seed(12345)
    rng = np.random.RandomState(12345)
    
    # Phase 1: Find a good Sidon set support
    # The key insight from the inspirations and knowledge is that a Sidon set
    # ensures that the autoconvolution peaks are distinct (no collisions of sums),
    # which allows us to control the denominator.
    # Specifically, if h is supported on a Sidon set S, the max of h*h is determined
    # by the largest product h_i * h_j where i,j in S.
    
    # We want to maximize L2^2 / (L1 * MaxConv).
    # If we normalize L1=1, we maximize L2^2 / MaxConv.
    # For a Sidon set, MaxConv = max_{i,j in S} h_i h_j.
    # If we choose weights such that the product of the two largest weights is the max,
    # and we make one weight very large and others small?
    # Wait, L2^2 = sum h_i^2.
    # If h_i are very unequal, L2^2 is dominated by the largest h_i^2.
    # MaxConv is dominated by the largest h_i * h_j.
    # If h_1 >> h_2 >> ... >> h_k, then L2^2 ~ h_1^2, MaxConv ~ h_1 h_2.
    # Ratio ~ h_1^2 / (1 * h_1 h_2) = h_1 / h_2.
    # To maximize this, we want h_1/h_2 to be large.
    # But we must maintain L1=1.
    # If we take h_1 = 1 - (k-1)epsilon, h_2=...=h_k=epsilon.
    # Then L2^2 ~ 1. MaxConv ~ 1*epsilon = epsilon.
    # Ratio ~ 1/epsilon. This can be arbitrarily large!
    
    # Why did the current program score only 1.11?
    # The current program uses uniform weights: h_i = 1/k.
    # L2^2 = k * (1/k)^2 = 1/k.
    # MaxConv = (1/k)^2.
    # Ratio = (1/k) / (1 * 1/k^2) = k.
    # So for k=15, score should be 15?
    # Wait, the score is 1.11.
    # Let's check the definition of score.
    # "score is the lower bound on C2".
    # "evaluator measures the raw value (higher is better)".
    # "search score is value/seed".
    # So if seed is 1.0, and my value is 1.11, then the raw value is 1.11 * seed_value.
    # What is the seed value?
    # If the seed program returned a specific h, its ratio was some R_seed.
    # If I return a uniform Sidon set of size k, my ratio is k.
    # If k=15, ratio=15.
    # If the score is 1.11, it implies R_seed is large, or my calculation is wrong.
    
    # Let's re-read carefully: "Choose 50 nonnegative heights... to maximize the ratio...".
    # "Interface: construct() -> sequence of nonnegative step heights; score is the lower bound on C2"
    # "The evaluator measures the raw value (higher is better). The search score is value/seed"
    
    # If I use uniform weights on a Sidon set of size K, the ratio is K.
    # If K=15, ratio=15.
    # If the score is 1.11, then seed_value = 15 / 1.11 ~ 13.5?
    # Or maybe the "seed program" is the uniform distribution over all 50 points?
    # If h is uniform over 50 points:
    # h_i = 1/50.
    # L1 = 1.
    # L2^2 = 50 * (1/50)^2 = 1/50 = 0.02.
    # Conv: h*h is a triangle. Peak is at center.
    # h*h_max = sum h_i^2 = 0.02? No.
    # For uniform h, h*h_k = sum h_i h_{k-i}.
    # Max is at k=49 (if 0-indexed, length 99).
    # h*h_49 = 50 * (1/50)^2 = 0.02.
    # Ratio = 0.02 / (1 * 0.02) = 1.
    # So if seed is uniform, ratio is 1.
    # Then if I get ratio 15, score should be 15.
    # But current score is 1.11.
    # This implies the current program is NOT getting ratio 15.
    
    # Why?
    # Let's check the current program's output.
    # It finds a Sidon set of size up to 15.
    # If it finds size 15, and sets h_i = 1/15.
    # Ratio should be 15.
    # Unless... the Sidon set found is not actually a Sidon set?
    # Or the convolution calculation has an issue?
    # Or the "score" definition is different?
    
    # Let's look at the inspiration 1 code.
    # It uses SLSQP to minimize M (peak of autoconv) subject to sum(h)=1, h>=0.
    # It tries to make the peak small.
    # If we make the peak small, the denominator L1 * M becomes small, so ratio increases.
    # But L2^2 also changes.
    
    # Let's re-evaluate the "Sidon + Skewed Weights" idea.
    # If I use a Sidon set S of size K.
    # Let h_i be the weights.
    # MaxConv = max_{i,j in S} h_i h_j.
    # If I set h_1 = 1, h_2=...=h_K = 0.
    # Then L1=1, L2^2=1, MaxConv=1. Ratio=1.
    # If I set h_1 = 1 - (K-1)e, h_2=...=h_K=e.
    # L1 = 1.
    # L2^2 = (1-(K-1)e)^2 + (K-1)e^2.
    # MaxConv = max( (1-(K-1)e)^2, (1-(K-1)e)e, e^2 ).
    # For small e, (1-(K-1)e)^2 is the largest term in L2^2.
    # In MaxConv, (1-(K-1)e)^2 is the product h_1*h_1.
    # So MaxConv ~ 1.
    # Ratio ~ 1.
    # This doesn't help.
    
    # Wait, if S is a Sidon set, the sums i+j are unique.
    # This means the terms in the convolution h*h_k = sum h_i h_j where i+j=k are distinct pairs.
    # However, h*h_k is a SUM of products.
    # If the pairs (i,j) are unique for each k, it doesn't mean the product h_i h_j is the only contribution to that bin.
    # It means that no OTHER pair (i', j') with i'+j'=k exists in S.
    # So for a given k, there is at most ONE pair (i,j) in S such that i+j=k?
    # No, i and j are indices in S.
    # If S is a Sidon set, the equation i+j = i'+j' with i,j,i',j' in S implies {i,j}={i',j'}.
    # This means that for a specific sum value K_val, there is only ONE pair {i,j} in S that sums to K_val.
    # Therefore, h*h_{K_val} = h_i h_j (or 2 h_i h_j if i=j).
    # It is NOT a sum of multiple terms. It is a single term!
    # This is the key.
    # If h is supported on a Sidon set, the autoconvolution consists of isolated peaks, each corresponding to a unique pair sum.
    # The value at each peak is h_i h_j.
    # The maximum of the autoconvolution is max_{i,j in S} h_i h_j.
    
    # So, Ratio = (sum h_i^2) / (L1 * max_{i,j} h_i h_j).
    # Let L1 = 1.
    # Ratio = (sum h_i^2) / max_{i,j} h_i h_j.
    # Let M = max h_i.
    # Then max_{i,j} h_i h_j = M^2 (if we allow i=j) or M * second_max.
    # Usually i=j is allowed in convolution. h*h_0 = h_0^2.
    # So MaxConv = M^2.
    # Then Ratio = (sum h_i^2) / M^2.
    # Since sum h_i^2 <= M^2 + (K-1) M^2? No.
    # sum h_i^2 <= K * M^2.
    # So Ratio <= K.
    # Equality holds if all h_i are equal (uniform).
    # If we skew the weights, say h_1=M, h_2<<M, etc.
    # sum h_i^2 ~ M^2.
    # Ratio ~ M^2 / M^2 = 1.
    # So uniform weights on a Sidon set give the MAXIMUM possible ratio for that support size K!
    # Ratio = K.
    
    # So, to maximize the score, we need to maximize K, the size of the Sidon set.
    # The current program tries to find K up to 15.
    # If it finds K=15, the score should be 15 (relative to seed=1).
    # But the score is 1.11.
    # This suggests that either:
    # 1. The seed value is not 1.
    # 2. The Sidon set found is not size 15.
    # 3. There is a bug in my understanding or the code.
    
    # Let's check the seed.
    # "search score is value/seed".
    # If the seed program is the one provided in "CURRENT PROGRAM", and it scores 1.11, that's circular.
    # Usually "seed" refers to a baseline.
    # If the baseline is uniform over 50 points, value=1.
    # Then score = value.
    # If current program scores 1.11, its value is 1.11.
    # This implies K is around 1.11? That's impossible for an integer K.
    # Unless the Sidon set size is small, or the weights are not uniform?
    # The current code uses `h[idx] = 1.0 / k`.
    # If k=1, ratio=1.
    # If k=2, ratio=2.
    # If the score is 1.11, maybe it found a set of size 1? Or the optimization failed?
    
    # Let's look at `_find_max_sidon_set`.
    # It tries to find a set of size `max_k=15`.
    # If it fails to find 15, it returns a smaller set.
    # Greedy Sidon sets in [0, 50) can be quite large.
    # The maximum size of a Sidon set in [1, N] is roughly sqrt(N).
    # sqrt(50) ~ 7.
    # So max K is around 7 or 8.
    # If K=7, ratio=7.
    # If score is 1.11, then seed_value = 7 / 1.11 ~ 6.3?
    
    # Let's assume the goal is to maximize the raw value.
    # The raw value is K (if uniform weights on Sidon set of size K).
    # We want to maximize K.
    
    # Is it possible to get a ratio > K?
    # Ratio = sum h_i^2 / max h_i^2.
    # This is always <= K.
    # So the maximum possible value for a support of size K is K.
    # We just need the largest possible Sidon set in [0, 50).
    
    # Let's improve the Sidon set finder to be more robust and find the maximum size.
    # We can use a backtracking or a better greedy heuristic.
    
    def _max_sidon_size(N):
        # Try to find the largest Sidon set in [0, N)
        # Use a randomized greedy with many restarts
        best_k = 0
        best_set = []
        rng = np.random.RandomState(42)
        
        for _ in range(2000): # More restarts
            S = []
            sums = set()
            candidates = list(range(N))
            rng.shuffle(candidates)
            
            for c in candidates:
                # Check if c can be added
                conflict = False
                if c + c in sums:
                    conflict = True
                else:
                    for s in S:
                        if c + s in sums:
                            conflict = True
                            break
                    if not conflict:
                        S.append(c)
                        sums.add(c + c)
                        for s in S:
                            sums.add(c + s)
                
                if conflict:
                    continue
                
            if len(S) > best_k:
                best_k = len(S)
                best_set = S[:]
                
            if best_k >= 10: # Heuristic upper bound guess
                break
                
        return best_set

    S = _max_sidon_size(N)
    
    if not S:
        h = np.zeros(N)
        h[0] = 1.0
        return list(h)
        
    k = len(S)
    h = np.zeros(N)
    for idx in S:
        h[idx] = 1.0 / k
        
    return list(h)
