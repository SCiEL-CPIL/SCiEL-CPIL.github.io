"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint
import itertools

N = 50
# Equal intervals on [-1/4, 1/4] means 50 points.
# The problem asks to maximize: (||h||_2^2) / (||h||_1 * ||h*h||_inf)
# where h is a vector of 50 nonnegative heights.
# Note: The intervals are "equal intervals of [-1/4, 1/4]".
# Usually this implies a grid. Let's assume the values are just the 50 numbers.
# The autoconvolution is likely the discrete convolution or continuous approximation.
# Given "step heights", it's likely a piecewise constant function.
# Let's assume the standard discrete autoconvolution (circular or linear?).
# "supremum norm of the autoconvolution".
# If it's step functions on an interval, the convolution is a continuous function.
# However, often in these coding tasks, it simplifies to discrete vectors.
# Let's look at the seed: [1.0] * 50.
# If h = [1, 1, ..., 1]:
# ||h||_2^2 = 50
# ||h||_1 = 50
# Autoconvolution of a box of length 50 is a triangle (if linear) or similar.
# If discrete linear convolution: h*h has length 99.
# Peak is at the center, value 50.
# Ratio = 50 / (50 * 50) = 1/50 = 0.02.
# Wait, the score is 1.0 for the seed.
# The task says "score is the lower bound on C2".
# And "evaluator measures the raw value (higher is better). The search score is value/seed".
# So if seed score is 1.0, the raw value of the seed is some constant V_seed.
# We need to maximize V.
#
# Let's re-read carefully: "maximize the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution."
# Let R(h) = (||h||_2^2) / (||h||_1 * ||h*h||_inf).
#
# Let's test the seed [1]*50 with discrete linear convolution.
# h = ones(50)
# L2_sq = 50
# L1 = 50
# conv = np.convolve(h, h) -> [1, 2, ..., 50, ..., 2, 1]
# max(conv) = 50
# R = 50 / (50 * 50) = 0.02.
#
# What if it's circular convolution?
# conv_circ = np.fft.ifft(np.fft.fft(h)**2).real
# For h=ones(50), FFT is [50, 0, ..., 0]. Square is [2500, 0...]. IFFT is [50, 0, ..., 0].
# max = 50.
# R = 50 / (50 * 50) = 0.02.
#
# Is it possible the norms are scaled by the interval width?
# "Choose 50 nonnegative heights on equal intervals of [-1/4, 1/4]"
# Total length = 0.5.
# 50 intervals. Width dx = 0.5 / 50 = 0.01.
#
# L1 norm of the function f(x): sum(h_i * dx)
# L2 norm of the function f(x): sqrt(sum(h_i^2 * dx)) -> L2_sq = sum(h_i^2 * dx)
#
# Autoconvolution f*f.
# Supremum norm ||f*f||_inf.
# For a step function, the convolution is a piecewise quadratic (or linear if we approximate differently?).
# Actually, the convolution of two step functions is a piecewise quadratic function.
# However, often in these problems, "autoconvolution" refers to the discrete operation on the coefficients, or the peak of the continuous convolution.
#
# Let's check the ratio scaling.
# If we scale h by c:
# L2_sq scales by c^2 * dx
# L1 scales by c * dx
# Convolution scales by c^2 * dx (area of overlap) ... wait.
# (c f) * (c f) = c^2 (f*f).
# So ||c f * c f||_inf = c^2 ||f*f||_inf.
# Ratio R(c h) = (c^2 L2_sq) / (c L1 * c^2 ConvPeak) = (1/c) * (L2_sq / (L1 * ConvPeak)).
# So R is inversely proportional to the scale.
# To maximize R, we should make h as large as possible?
# But h is "nonnegative heights". Is there a bound on the height?
# "Choose 50 nonnegative heights...". No explicit bound like h_i <= 1.
# If there is no bound, we can make h_i -> infinity, and R -> infinity?
# Wait. R = L2_sq / (L1 * ConvPeak).
# If h -> c h, R -> R/c.
# So larger h gives SMALLER R.
# We want to MAXIMIZE R.
# So we want SMALL h?
# But h must be nonnegative. If h -> 0, R -> infinity?
# 0/0 is undefined.
# Let's check the limit.
# If all h_i are small, the ratio is scale invariant?
# No, R(c h) = R(h) / c.
# So smaller c means larger R.
# This implies the problem must have a constraint I'm missing, or my interpretation of the norms is wrong.
#
# "maximize the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution"
# Maybe it's (||h||_2^2) / (||h||_1 * ||h||_inf)? No, "autoconvolution".
#
# Let's re-read the seed score.
# Seed: [1.0] * 50. Score 1.0.
# If the raw value is 0.02, then V_seed = 0.02.
# If I choose h = [0.01] * 50.
# L2_sq = 50 * 0.0001 * 0.01 = 0.00005
# L1 = 50 * 0.01 * 0.01 = 0.005
# ConvPeak (discrete) = 50 * 0.0001 = 0.005.
# Wait, discrete conv of h*h:
# h_i = 0.01.
# conv_k = sum h_i h_{k-i} = 50 * 0.01 * 0.01 = 0.005.
# R = 0.00005 / (0.005 * 0.005) = 0.00005 / 0.000025 = 2.
# This is 100 times larger than 0.02.
# So if I scale down by 50, I get 50 times the ratio?
# R(c h) = R(h) / c.
# If c = 1/50, R = 50 * R(1) = 50 * 0.02 = 1.0.
#
# So if I can choose arbitrarily small heights, the score goes to infinity.
# There MUST be a constraint.
# "Choose 50 nonnegative heights on equal intervals of [-1/4, 1/4]"
# Maybe the heights are probabilities? Sum to 1?
# Or maybe the L2 norm is normalized?
# Or maybe the "heights" are bounded by 1?
# If h_i <= 1, then the maximum R is achieved by the smallest possible h.
# But if h_i can be arbitrarily close to 0, R -> infinity.
#
# Let's look at the "seed program" again.
# `return [1.0] * N`
# If the constraint was h_i <= 1, the seed is at the boundary.
# If the constraint was sum(h_i) = 1, the seed is invalid (sum=50).
# If the constraint was ||h||_1 = 1, the seed is invalid.
#
# Is it possible the ratio is defined differently?
# "ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution"
# Could it be: (||h||_2^2) / (||h||_1) * ||h*h||_inf ?
# Then R(c h) = (c^2 L2_sq / c L1) * c^2 Conv = c^3 R.
# Then large h is better.
# If h_i <= 1 is the constraint, we maximize by setting h_i = 1.
# Seed is [1, 1, ..., 1].
# If this were the case, the seed would be optimal (or close to it) for uniform distribution?
# Let's check if a non-uniform distribution is better for this metric.
# R = (L2_sq / L1) * ConvPeak.
# For [1]*50:
# L2_sq = 50, L1 = 50, ConvPeak = 50.
# R = (1) * 50 = 50.
#
# If I pick a spike: h = [1, 0, ..., 0].
# L2_sq = 1, L1 = 1, ConvPeak = 1.
# R = 1 * 1 = 1.
# Much worse.
#
# If I pick [1, 1, 0, ..., 0].
# L2_sq = 2, L1 = 2, ConvPeak = 2.
# R = 1 * 2 = 2.
#
# So for this metric, spreading out is better.
# The uniform distribution [1, 1, ..., 1] maximizes L2_sq/L1 (which is max h_i) and ConvPeak (which is sum h_i^2).
# Actually, L2_sq/L1 = (sum h^2) / (sum h). This is maximized by making the distribution as "spiky" as possible?
# No, (sum h^2) / (sum h) is the weighted average of h. It is maximized by putting all mass in one bin (if bounded) or making values large.
# But ConvPeak = max (sum h_i h_{i+k}).
#
# Let's reconsider the first interpretation: R = L2_sq / (L1 * ConvPeak).
# If there is a constraint like ||h||_1 = 1.
# Then R = L2_sq / ConvPeak.
# For [1/50]*50:
# L2_sq = 50 * (1/50)^2 = 1/50.
# ConvPeak = 50 * (1/50)^2 = 1/50.
# R = (1/50) / (1/50) = 1.
#
# For spike [1, 0, ...]:
# L2_sq = 1.
# ConvPeak = 1.
# R = 1.
#
# For [1/2, 1/2, 0...]:
# L2_sq = 1/4 + 1/4 = 1/2.
# ConvPeak = 1/4 + 1/4 + 1/4 = 3/4?
# Conv: [0.25, 0.5, 0.25]. Max = 0.5.
# R = 0.5 / 0.5 = 1.
#
# It seems R=1 is a lower bound?
# The task says "score is the lower bound on C2".
# And "maximize the ratio".
#
# Let's assume the constraint is ||h||_1 = 1 (probability distribution).
# Then we want to maximize L2_sq / ConvPeak.
# This is equivalent to minimizing ConvPeak / L2_sq.
#
# If we use the Sidon set idea from the inspiration:
# "maximizing the ratio by minimizing the peak autocorrelation relative to energy"
# Energy = L2_sq. Peak Autoconv = ConvPeak.
# So yes, we want to minimize ConvPeak / L2_sq.
#
# The inspiration says: "Replacing local optimization with combinatorial search for a Sidon set... yielding a score proportional to the set size (approx 3.5-4) instead of ~1.1"
# This suggests the score can be > 1.
#
# Let's verify the metric with the Sidon set idea.
# If h is the indicator of a Sidon set S (scaled to sum to 1).
# h_i = 1/|S| if i in S, 0 otherwise.
# L2_sq = |S| * (1/|S|)^2 = 1/|S|.
# ConvPeak = max_k sum_{i in S} h_i h_{i+k}.
# = max_k (1/|S|^2) * (number of pairs i,j in S such that j-i=k).
# Let A(k) be the number of such pairs.
# ConvPeak = max_k A(k) / |S|^2.
# Ratio R = L2_sq / ConvPeak = (1/|S|) / (max_k A(k) / |S|^2) = |S| / max_k A(k).
#
# For a Sidon set, the number of representations of any difference is small.
# Specifically, for a Sidon set, each difference is represented at most twice (or once if we consider ordered pairs and k!=0, plus the trivial k=0).
# For k=0, A(0) = |S|.
# So ConvPeak includes the k=0 term?
# "Autoconvolution" usually includes the peak at lag 0.
# If k=0 is included, A(0) = |S|.
# Then R = |S| / |S| = 1.
# This would mean the score is always 1 if k=0 is included.
#
# BUT, the inspiration says "minimizing the peak autocorrelation".
# Often "autocorrelation" in signal processing contexts for these problems might refer to the off-peak values, or the problem definition excludes the main lobe?
# Or maybe the "supremum norm of the autoconvolution" refers to the continuous convolution of the step functions, where the peak at 0 is handled differently?
#
# Let's check the continuous convolution of a step function.
# If h is a step function with height H on a set of intervals.
# The convolution f*f at lag 0 is the integral of f^2.
# The peak of the continuous convolution is at lag 0.
# So ||f*f||_inf = (f*f)(0) = ||f||_2^2 (continuous).
# If so, R = ||f||_2^2 / (||f||_1 * ||f||_2^2) = 1 / ||f||_1.
# If ||f||_1 = 1, R = 1.
# This contradicts the "improvement" to 1.173.
#
# Therefore, the "supremum norm of the autoconvolution" must NOT be the value at lag 0.
# Or, the "autoconvolution" is defined as the circular convolution, and we look at the max?
# Or, the problem implies the max of the *side lobes*?
# Or, the "autoconvolution" is the discrete convolution, and we are looking for the max value *excluding* the center?
#
# Let's look at the phrasing: "supremum norm of the autoconvolution".
# If it's the discrete convolution c = h*h.
# c[0] = sum h_i^2 = L2_sq.
# If the max is c[0], then R = L2_sq / (L1 * L2_sq) = 1/L1.
# Again, if L1=1, R=1.
#
# However, if the max is attained at some k != 0, and that value is smaller than c[0], then R > 1.
# This happens if the signal is "flat" in frequency domain? No.
# It happens if the signal is a "Barker sequence" or "Sidon set".
# For a Sidon set, the off-diagonal correlations are small.
# Specifically, for a Sidon set, the number of pairs with difference k is small for k != 0.
# The max off-diagonal correlation is bounded by ~sqrt(N) or similar?
# Actually, for a Sidon set of size K in [1, N], the max number of pairs with difference k is small.
#
# If the problem asks for the max over ALL k (including 0), and the max is at 0, then R=1.
# If the problem asks for the max over k != 0, then we can get R > 1.
# Given the score 1.173 > 1, it is highly likely that the peak at lag 0 is EXCLUDED, or the metric is defined such that the main lobe is not the bottleneck.
#
# Wait, look at the seed score again.
# Seed: [1, 1, ..., 1].
# If we exclude k=0:
# h = ones(50).
# Conv (linear): [1, 2, ..., 50, ..., 1].
# c[0] (if 0-indexed start) corresponds to lag 0?
# np.convolve(ones(50), ones(50))
# Index 0: 1*1 = 1.
# Index 49: 50.
# Index 98: 1.
# The max is 50 at index 49 (lag 0 in centered view).
# If we exclude the center, the next highest is 49.
# R = L2_sq / (L1 * 49) = 50 / (50 * 49) = 1/49 = 0.02.
# This is < 1.
#
# What if the "autoconvolution" is the CIRCULAR convolution?
# h = ones(50).
# Circular conv: all values are 50.
# Max is 50.
# R = 50 / (50 * 50) = 0.02.
#
# This is confusing. Let's look at the "Sidon set" inspiration again.
# "yielding a score proportional to the set size (approx 3.5-4)"
# If R = K / max_off_diag, and for a Sidon set max_off_diag is small (e.g., 1 or 2), then R ~ K.
# If K=50, R could be 25 or 50.
#
# Let's assume the metric is:
# R = ||h||_2^2 / (||h||_1 * max_{k \neq 0} |h*h[k]|)
# And we are constrained to ||h||_1 = 1.
#
# If h is a Sidon set indicator scaled to sum 1:
# ||h||_2^2 = 1/K.
# max_{k \neq 0} |h*h[k]| = max_{k \neq 0} A(k) / K^2.
# For a perfect Sidon set, A(k) is 1 or 2 for all k != 0.
# So max off-diag = 2/K^2 (or 1/K^2).
# R = (1/K) / (2/K^2) = K/2.
# If K=50, R=25.
#
# This fits the "improvement" narrative. The seed [1]*50 has a huge off-diagonal peak (49), so R is small.
# A Sidon set has small off-diagonal peaks, so R is large.
#
# So the goal is:
# 1. Choose a subset S of indices of size K.
# 2. Set h_i = 1/K if i in S, else 0.
# 3. Maximize K / max_{k \neq 0} A(k).
#
# To maximize this, we want a large K and a small max A(k).
# A(k) is the number of pairs (i, j) in S such that j - i = k.
#
# We can search for such a set.
#
# Algorithm:
# 1. Start with a random set or greedy construction.
# 2. Use local search (swap elements) to minimize max_{k \neq 0} A(k).
# 3. Keep K as large as possible? Or fix K=50?
# The task says "Choose 50 nonnegative heights".
# It does not say "exactly 50 non-zero heights".
# But if we add zeros, K decreases.
# R = K / max_A.
# If we reduce K, R might decrease.
# Usually, larger K is better if max_A grows slowly.
# For random sets, max_A ~ sqrt(K). So R ~ sqrt(K).
# So we want K as large as possible.
# Can we have K=50?
# A Sidon set of size 50 in a domain of size 50?
# The maximum size of a Sidon set in [1, N] is ~sqrt(N).
# For N=50, max Sidon size is ~7.
# So we cannot have K=50 with small max_A.
#
# If we use K=7 (Sidon set), R ~ 7/2 = 3.5.
# If we use K=50 (all ones), max_A = 49. R = 50/49 ~ 1.02.
#
# So the optimal strategy is likely to use a sparse set (Sidon-like) with K ~ 7-10.
#
# Let's implement a search for a Sidon set of size K in the domain [0, 49].
# We can try different K values.
#
# Plan:
# 1. Define a function to compute the score for a given binary mask (scaled to sum 1).
#    Score = (sum h^2) / (sum h * max_off_diag_conv)
#    Since sum h = 1, Score = sum h^2 / max_off_diag_conv.
#    For binary mask of size K:
#    sum h^2 = K * (1/K)^2 = 1/K.
#    max_off_diag_conv = max_{k \neq 0} (count_pairs_diff_k) / K^2.
#    Score = (1/K) / (max_count / K^2) = K / max_count.
#
# 2. Search for a set S that minimizes max_count.
#    We can use a greedy approach or local search.
#
# 3. Try K from 2 to 50.
#    For each K, find the best set.
#    Track the best score.
#
# Note: The "heights" are nonnegative. We can also use non-binary heights?
# The Sidon set argument suggests binary is optimal for this specific ratio structure if we fix the support.
#
# Let's write the code to find a good Sidon set.

def compute_score_for_mask(mask):
    """
    mask: binary array of length N.
    Returns the score R.
    """
    K = np.sum(mask)
    if K == 0:
        return 0.0
    h = mask.astype(float) / K
    # L2_sq
    l2_sq = np.sum(h**2)
    # L1
    l1 = np.sum(h) # should be 1
    # Autoconvolution (linear)
    conv = np.convolve(h, h)
    # Exclude lag 0.
    # The center of the convolution is at index K-1.
    # We want to exclude the peak at lag 0.
    # In linear conv of length K with itself, the result has length 2K-1.
    # The peak is at index K-1.
    # We take the max of the rest.
    # If K=1, conv has length 1. No off-diagonal. Score is infinity?
    # If K=1, h=[1]. conv=[1]. max_off_diag is undefined.
    # Let's assume K>=2.
    
    # Identify the index of the main lobe.
    # For h with support size K, the main lobe is at the center.
    # Actually, np.convolve(h, h) aligns the end of the first with the start of the second?
    # No, standard linear convolution.
    # If h is sparse, the main lobe is where the overlap is maximal.
    # For a set S, the max overlap is at lag 0 (difference 0).
    # In the conv array, the value at lag 0 corresponds to index len(S)-1?
    # Let's just find the max value and subtract it? No, we want the max of the REST.
    
    # To be safe, let's compute the autocorrelation explicitly for all lags.
    # lags from -(K-1) to K-1.
    # c[k] = sum h[i] h[i+k]
    
    # Efficient way: FFT
    # But N is small (50).
    
    max_off_diag = 0.0
    for k in range(1, K):
        # Lag k
        # sum h[i] h[i+k] for i=0 to N-1-k
        # Using numpy slicing
        val = np.dot(h[:-k], h[k:])
        if val > max_off_diag:
            max_off_diag = val
    for k in range(1, K):
        # Lag -k
        val = np.dot(h[k:], h[:-k])
        if val > max_off_diag:
            max_off_diag = val
            
    if max_off_diag == 0:
        return float('inf')
        
    return l2_sq / (l1 * max_off_diag)

def find_best_sidon(N, max_k=20):
    best_score = 0.0
    best_mask = None
    
    for K in range(2, max_k + 1):
        # Try to find a good set of size K
        # Greedy initialization
        # Start with a random set
        best_local_score = 0.0
        best_local_mask = None
        
        for _ in range(10): # 10 restarts
            # Random subset
            indices = np.random.choice(N, K, replace=False)
            mask = np.zeros(N)
            mask[indices] = 1.0
            
            # Local search: swap one element
            improved = True
            while improved:
                improved = False
                current_score = compute_score_for_mask(mask)
                
                # Try swapping each element in the set with each element not in the set
                in_set = np.where(mask > 0)[0]
                out_set = np.where(mask == 0)[0]
                
                best_swap_score = current_score
                best_swap = None
                
                for i in in_set:
                    for j in out_set:
                        # Create new mask
                        new_mask = mask.copy()
                        new_mask[i] = 0
                        new_mask[j] = 1
                        s = compute_score_for_mask(new_mask)
                        if s > best_swap_score:
                            best_swap_score = s
                            best_swap = (i, j)
                            
                if best_swap:
                    i, j = best_swap
                    mask[i] = 0
                    mask[j] = 1
                    improved = True
                    
            if best_local_mask is None or current_score > best_local_score:
                best_local_score = current_score
                best_local_mask = mask.copy()
                
        if best_local_score > best_score:
            best_score = best_local_score
            best_mask = best_local_mask.copy()
            
    return best_score, best_mask

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 140  # 120s limit, keep margin
    
    # Try to find a good Sidon set
    score, mask = find_best_sidon(N, max_k=15)
    
    # Convert mask to heights
    if mask is not None:
        K = np.sum(mask)
        if K > 0:
            heights = mask / K
        else:
            heights = [1.0] * N
    else:
        heights = [1.0] * N
        
    # Check time
    if time.time() - start_time > timeout:
        # Return seed if too slow
        return [1.0] * N
        
    return list(heights)
# EVOLVE-BLOCK-END
