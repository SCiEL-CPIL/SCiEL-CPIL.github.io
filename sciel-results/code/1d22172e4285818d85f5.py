"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import combinations

N = 50

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115.0
    best_val = -1.0
    best_h = [1.0] + [0.0] * (N - 1)
    
    # The objective is to maximize R = ||h||_2^2 / (||h||_1 * ||h * h||_inf)
    # Note: The task says "squared L2 norm to L1 norm times supremum norm of the autoconvolution".
    # Usually this is written as ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    # Let's verify with seed [1]*50.
    # h = [1]*50
    # ||h||_2^2 = 50
    # ||h||_1 = 50
    # (h*h) is a triangular sequence: 1, 2, ..., 50, ..., 2, 1. Max is 50.
    # R = 50 / (50 * 50) = 1/50 = 0.02.
    # The prompt says seed scores 0.0000 in the "CURRENT PROGRAM" section but the search score is value/seed.
    # Wait, the prompt says "score is the lower bound on C2". And "score 0.0000" for current program.
    # But the knowledge says: "Measured: 1.133 -> 1.173". This suggests the values are > 1.
    # Let's re-read carefully: "maximize the ratio of squared L2 norm to L1 norm times supremum norm".
    # If the formula is ||h||_2^2 / (||h||_1 * ||h*h||_inf), then for [1]*50 it is 0.02.
    # If the formula is ||h||_2^2 * ||h||_1 / ||h*h||_inf? No.
    # Let's look at the "Sidon set" finding.
    # "Finding the maximum Sidon set (size 8) instead of a small greedy one increases the discrete ratio J from ~1.1 to 4".
    # This implies the values are indeed > 1.
    # How can the ratio be > 1?
    # R = ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    # For h=[1,0,...,0]: ||h||_2^2=1, ||h||_1=1, ||h*h||_inf=1. R=1.
    # For h=[1,1]: ||h||_2^2=2, ||h||_1=2, ||h*h||_inf=2 (conv is [1,2,1]). R = 2/(2*2) = 0.5.
    # For h=[1,0,1]: ||h||_2^2=2, ||h||_1=2, ||h*h||_inf=1 (conv is [1,0,2,0,1]). R = 2/(2*1) = 1.
    # So sparse vectors give R=1. Dense vectors give R<1.
    # The Sidon set mention suggests that if we use a Sidon set of indices where h_i=1, the collision in convolution is minimized.
    # A Sidon set B is a set of integers such that all pairwise sums b_i+b_j are distinct (or have unique representation).
    # If B is a Sidon set, then the autoconvolution h*h has all non-zero coefficients equal to 1 (except for the center which is |B|).
    # Wait, the autoconvolution at lag 0 is sum h_i^2. If h_i in {0,1}, sum h_i^2 = |B|.
    # The other lags are the number of pairs (i,j) such that i-j = k.
    # For a Sidon set, each difference i-j appears at most once (for i!=j) and zero appears |B| times.
    # So ||h*h||_inf = max(|B|, 1) = |B|.
    # Then R = |B| / (|B| * |B|) = 1/|B|.
    # This would mean larger Sidon sets give SMALLER ratios.
    # This contradicts the finding "increases ... from ~1.1 to 4".
    
    # Let's re-evaluate the formula.
    # "ratio of squared L2 norm to L1 norm times supremum norm"
    # Could it be (||h||_2^2 * ||h||_1) / ||h*h||_inf?
    # Seed [1]*50: (50 * 50) / 50 = 50.
    # Single [1]: (1 * 1) / 1 = 1.
    # This doesn't match the ~1.1 to 4 range well if we assume heights are 1.
    
    # Let's look at the "Second autocorrelation inequality".
    # This usually refers to: ||h||_2^4 >= ||h||_1^2 ||h*h||_inf / N ? No.
    # A common inequality is ||h||_2^2 / ||h||_1 >= C * ||h*h||_inf ?
    
    # Let's trust the "Sidon set" hint and the measured values 1.133 -> 1.173.
    # If Sidon sets improve the score, and Sidon sets are sparse, maybe the score is INVERSELY related to collisions?
    # Or maybe the heights are not binary?
    
    # Let's try to interpret "score is the lower bound on C2".
    # C2 might be a constant in an inequality like:
    # ||h||_2^2 >= C2 * ||h||_1 * ||h*h||_inf / something?
    
    # Let's assume the objective function is:
    # Value = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # And the "Sidon set" comment is about a DIFFERENT metric or I am misinterpreting "increases".
    # If J goes from 1.1 to 4, and Sidon sets have high J, then J must be high for sparse sets?
    # No, we calculated R=1 for sparse, R=0.02 for dense.
    # So sparse is BETTER for R.
    # If the seed [1]*50 scores 0.02, and we want to maximize, we should go sparse.
    # But the current program scores 0.0000?
    # "CURRENT PROGRAM (score 0.0000)" -> This might mean the SCORE (value/seed) is 0.
    # Or the VALUE is 0?
    # If value is 0, maybe the formula yields 0 for [1]*50? Unlikely.
    # Maybe the evaluator subtracts a baseline?
    
    # Let's look at the "BET" from the previous failed attempt.
    # "Using a single non-zero height ... achieving the theoretical upper bound of 1.0, which is 5x the seed value of 0.2."
    # This implies the seed value was 0.2.
    # Why 0.2?
    # If seed is [1]*50, and value is 0.2.
    # My calc gave 0.02. Factor of 10 difference.
    # Maybe N is in the denominator? Or numerator?
    # If Value = N * ||h||_2^2 / (||h||_1 * ||h*h||_inf), then 50 * 0.02 = 1.0.
    # If Value = ||h||_2^2 / (||h||_1 * ||h*h||_inf / N), same thing.
    
    # Let's assume the metric is:
    # Score = ||h||_2^2 / (||h||_1 * ||h*h||_inf / N) = N * ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # For [1]*50: 50 * 50 / (50 * 50) = 1.0.
    # For [1,0,...,0]: 50 * 1 / (1 * 1) = 50.
    # This would be a huge jump.
    
    # However, the knowledge says: "Measured: 1.133 -> 1.173".
    # This suggests the values are around 1.
    # If the max possible is 50, why are we stuck at 1.1?
    # Maybe the heights are not restricted to {0,1}?
    # Or maybe the formula is different.
    
    # Let's reconsider the "Sidon set" finding.
    # "Finding the maximum Sidon set (size 8) instead of a small greedy one increases the discrete ratio J from ~1.1 to 4".
    # This explicitly states that using a Sidon set increases the ratio.
    # My analysis of R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) showed that Sidon sets (sparse) give R=1/|B|, which decreases as size increases.
    # This is a contradiction.
    # Therefore, the metric J must be INVERSE of what I assumed, or I am calculating ||h*h||_inf wrong.
    # ||h*h||_inf for a Sidon set of size K with h_i=1 is K (at lag 0).
    # Is it possible the metric is ||h||_2^2 * ||h||_1 / ||h*h||_inf ?
    # Sidon set size K: K * K / K = K.
    # So J=K.
    # If K=8, J=8.
    # If "small greedy" gave 1.1, maybe it was a dense set?
    # Dense set [1]*50: 50 * 50 / 50 = 50.
    # This doesn't fit "1.1 to 4".
    
    # What if the metric is ||h||_2^2 / ||h*h||_inf ? (ignoring L1)
    # Sidon K: K / K = 1.
    # Dense: 50 / 50 = 1.
    
    # What if the metric is (||h||_2^2 / ||h||_1) / (||h*h||_inf / N) ?
    # This is the same as N * ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    # Sidon K: N * K / (K * K) = N/K.
    # If N=50, K=8, J = 50/8 = 6.25.
    # If K=50, J=1.
    # This fits the "increases" trend! Larger Sidon sets (smaller K? No, "maximum Sidon set size 8" vs "small greedy").
    # Wait, a "maximum Sidon set" in [0, N) usually means the LARGEST possible size.
    # If we go from a "small greedy" (small K) to "maximum" (large K), J = N/K decreases.
    # But the finding says J INCREASES from 1.1 to 4.
    # So "maximum Sidon set" must mean something else, or the formula is different.
    
    # Let's look at the phrasing: "Finding the maximum Sidon set (size 8) ... increases ... from ~1.1 to 4".
    # This implies the previous solution was NOT a maximum Sidon set, and was smaller/worse.
    # So the new solution IS a size 8 Sidon set, and it is BETTER.
    # So J(Sidon 8) = 4.
    # J(Other) = 1.1.
    # If J = N/K, then 50/8 = 6.25 != 4.
    # If J = K? 8 != 4.
    # If J = ||h||_2^2 / ||h||_1 ? No, that's 1 for binary.
    
    # Let's try to find a formula that gives ~4 for a Sidon set of size 8 in N=50.
    # Maybe the heights are not 1?
    # If we optimize heights, we might get better values.
    
    # Given the ambiguity, I will implement a local search that optimizes the ratio:
    # R(h) = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # And I will assume the evaluator's "value" is proportional to this, or I will try to maximize this and see.
    # However, the previous failed attempt returned [1,0,...,0] and was rejected/not admitted.
    # The rejection reason was "empty", which is odd.
    # The feedback says "source-validation result".
    
    # Let's try a different structural change: Use a randomized hill-climbing approach to maximize the ratio R = ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    # I will start with a sparse vector (Sidon-like) and perturb it.
    
    def score(h):
        h = np.array(h, dtype=np.float64)
        if np.all(h == 0):
            return -1.0
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        # Linear convolution
        conv = np.convolve(h, h, mode='full')
        l_inf = np.max(conv)
        if l_inf == 0:
            return -1.0
        return l2_sq / (l1 * l_inf)

    # Initial candidate: A Sidon set.
    # Greedy Sidon set construction for N=50.
    def greedy_sidon_set(n, k):
        S = []
        used_sums = set()
        for i in range(n):
            if len(S) == k:
                break
            # Check if i can be added
            ok = True
            for s in S:
                if i + s in used_sums:
                    ok = False
                    break
            if ok:
                S.append(i)
                for s in S:
                    used_sums.add(i + s)
        return S

    # Let's try a few random initializations and hill climb.
    best_val = -1.0
    best_h = [1.0] + [0.0] * (N - 1)
    
    # Try Sidon sets of various sizes
    for k in range(1, 15):
        if time.time() > deadline:
            break
        S = greedy_sidon_set(N, k)
        h = [0.0] * N
        for idx in S:
            h[idx] = 1.0
        val = score(h)
        if val > best_val:
            best_val = val
            best_h = h[:]

    # Hill climbing with random restarts
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        
        # Start from best_h or random sparse
        if restart == 0:
            h = best_h[:]
        else:
            # Random sparse vector
            k = random.randint(1, 10)
            indices = random.sample(range(N), k)
            h = [0.0] * N
            for idx in indices:
                h[idx] = 1.0
        
        cur_val = score(h)
        if cur_val > best_val:
            best_val = cur_val
            best_h = h[:]
        
        # Local search: perturb one or two elements
        for step in range(1000):
            if time.time() > deadline - 2:
                break
            i = random.randint(0, N - 1)
            # Change h[i]
            old_val = h[i]
            # Try setting to 0, or 1, or random small value
            choices = [0.0, 1.0, random.uniform(0.1, 1.0)]
            if old_val > 0:
                choices.append(0.0)
            h[i] = random.choice(choices)
            
            new_val = score(h)
            if new_val > cur_val:
                cur_val = new_val
                if new_val > best_val:
                    best_val = new_val
                    best_h = h[:]
            else:
                # Revert with some probability (simulated annealing style, but simple)
                if random.random() > 0.5:
                    h[i] = old_val
                # else keep the worse move? No, that's bad. Just revert.
                h[i] = old_val
    
    return best_h
# EVOLVE-BLOCK-END
