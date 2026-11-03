import numpy as np

def construct():
    N = 50
    DELTA = 0.5 / N

    def _score_raw(h):
        h = np.asarray(h, dtype=np.float64)
        l1 = np.sum(h)
        if l1 == 0:
            return 0.0
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        autoconv_max = DELTA * np.max(h_conv)
        if autoconv_max == 0:
            return 0.0
        return (l2_sq / (l1 * autoconv_max))

    # Strategy: Use a Sidon set (B_2 sequence) with uniform weights.
    # For a Sidon set, all pairwise sums i+j (i<=j) are distinct.
    # The autoconvolution at lag k corresponds to the sum of weights for pairs with difference k.
    # With uniform weights 1/K on a Sidon set, the peak of the autoconvolution is determined by the "self" terms (i=i).
    # There are K such terms, each contributing (1/K)^2 = 1/K^2 to the unnormalized convolution.
    # So max_conv (unnormalized) = K * (1/K^2) = 1/K.
    # Wait, the convolution sums h[i]*h[j] where j-i = k.
    # For k=0, sum is sum_i h[i]^2 = K * (1/K)^2 = 1/K.
    # For k != 0, the number of pairs with difference k is at most 1 (for Sidon sets? No, Sidon means sums are distinct, not differences).
    # Actually, a Sidon set (B_2 set) means all sums i+j (i<=j) are distinct.
    # This implies that the representation of any integer as a sum of two elements (not necessarily distinct) is unique.
    # The autoconvolution peak is max_k sum_i h[i]h[i+k].
    # If we use a Sidon set, does it minimize the peak?
    # Let's check a simpler property.
    # If we use a set where all differences i-j are distinct (a "Golomb ruler" type property for differences), then for any non-zero lag k, there is at most one pair (i, i+k).
    # Then the peak is at k=0, which is sum h[i]^2 = 1/K.
    # The L2 norm squared is 1/K. L1 is 1.
    # Score = (1/K) / (1 * DELTA * 1/K) = 1/DELTA = N/0.5 = 2N = 100.
    #
    # If we use uniform distribution over all N points:
    # h[i] = 1/N.
    # L2^2 = N * (1/N)^2 = 1/N.
    # L1 = 1.
    # Autoconvolution at k=0 is 1/N.
    # Autoconvolution at k=1 is (N-1)/N^2.
    # Max is at k=0, which is 1/N.
    # Score = (1/N) / (1 * DELTA * 1/N) = 1/DELTA = 100.
    #
    # So both uniform and a "Golomb ruler" style set give 100?
    # Let's re-read the task. "maximize the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution".
    # Ratio = ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    # Note: The autoconvolution defined in the code is `np.convolve(h, h, mode='full')`.
    # This is the standard linear convolution.
    # The "supremum norm of the autoconvolution" in the code is `DELTA * np.max(h_conv)`.
    # Why DELTA?
    # The integral of the convolution of two step functions of height h_i and h_j over width DELTA is h_i*h_j*DELTA.
    # So the continuous autoconvolution peak is approximately DELTA * max(discrete_conv).
    #
    # Let's check if we can beat 100.
    # The inequality is ||h||_2^2 >= C * ||h||_1 * ||h*h||_inf ?
    # Actually, the task says "score is the lower bound on C2".
    # And "maximize the ratio".
    # So we want to maximize R = ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    #
    # Consider h being a single spike. h[0]=1, others 0.
    # L2^2 = 1. L1 = 1.
    # Convolution is a spike at 0 with height 1.
    # Max conv = 1.
    # Score = 1 / (1 * DELTA * 1) = 1/DELTA = 100.
    #
    # Consider h uniform. Score = 100.
    #
    # Is it possible to get > 100?
    # Let's try h[0]=a, h[1]=b.
    # L2^2 = a^2+b^2. L1 = a+b.
    # Conv: [a^2, ab, b^2]. Max is max(a^2, ab, b^2).
    # Assume a>=b>=0. Max is a^2.
    # Score = (a^2+b^2) / ((a+b) * DELTA * a^2) = (1 + (b/a)^2) / ((1 + b/a) * DELTA).
    # Let x = b/a. f(x) = (1+x^2)/(1+x).
    # f'(x) = [2x(1+x) - (1+x^2)] / (1+x)^2 = (2x+2x^2 - 1 - x^2)/(1+x)^2 = (x^2+2x-1)/(1+x)^2.
    # Critical point at x = -1 + sqrt(2) approx 0.414.
    # f(0.414) = (1 + 0.171) / 1.414 = 1.171 / 1.414 = 0.828.
    # f(0) = 1. f(1) = 1.
    # So x=0 (single spike) or x=1 (uniform 2 spikes) gives 1.
    # So 2 spikes doesn't beat 1 spike.
    #
    # What if we use a Golomb ruler?
    # A Golomb ruler is a set of marks where all differences are distinct.
    # If indices form a Golomb ruler, then for any k != 0, there is at most one pair (i, i+k).
    # So conv[k] = h[i]h[i+k] for unique i.
    # conv[0] = sum h[i]^2.
    # If we set all h[i] = 1/K, then conv[0] = 1/K.
    # conv[k] = 1/K^2 for the unique pair.
    # Max conv is 1/K (since K>=1, 1/K >= 1/K^2).
    # L2^2 = 1/K. L1 = 1.
    # Score = (1/K) / (1 * DELTA * 1/K) = 1/DELTA = 100.
    #
    # It seems 100 is the limit for "spread out" signals where the peak is at k=0.
    # Can we make the peak at k=0 small?
    # To reduce conv[0] = sum h[i]^2, we need to spread h.
    # But spreading h increases conv[k] for k!=0.
    #
    # Let's look at the "Sidon" idea again.
    # Sidon set: sums i+j distinct.
    # This is related to the convolution of the *indicator* of the set.
    # If A is a Sidon set, the sumset A+A has size ~|A|^2.
    # This means the convolution of 1_A with 1_A has non-zero entries at ~|A|^2 distinct lags.
    # The values are 1 or 2 (if i!=j) or 1 (if i=j).
    # Actually, for Sidon, each sum is unique.
    # So (1_A * 1_A)[k] is 1 if there is a unique representation, 0 otherwise?
    # No, the number of representations.
    # Sidon means number of representations is <= 1 for k>0? No, for sums.
    #
    # Let's try to just use a Golomb ruler with uniform weights.
    # It gives 100.
    # The current program scores 1.17.
    # This suggests the current program is returning something bad.
    # My new program should return a Golomb ruler or Uniform, both of which give ~100.
    #
    # Let's implement a simple Golomb ruler finder or just return Uniform.
    # Uniform is safe and fast.
    #
    # Wait, is it possible that the "supremum norm of the autoconvolution" refers to the continuous L-infinity norm of the convolution of the *function*?
    # The function is a step function.
    # The convolution of two step functions is a trapezoid/triangle shape.
    # The peak of the continuous convolution is indeed DELTA * max(discrete_conv) if the steps are aligned?
    # Let's verify.
    # h(x) = sum h_i * 1_{[x_i, x_i+DELTA)}(x).
    # (h*h)(t) = integral h(x)h(t-x) dx.
    # If t is in [x_i, x_i+DELTA), then h(x) is h_i for x in [x_i, x_i+DELTA).
    # The overlap length is DELTA.
    # So (h*h)(t) = h_i^2 * DELTA.
    # If t is such that it overlaps two steps?
    # The steps are adjacent.
    # If t = x_j, and x_j = x_i + k*DELTA.
    # Then the overlap of [x_i, x_i+DELTA) and [x_j-t, x_j-t+DELTA) ...
    # Actually, the convolution of rectangular pulses is triangular.
    # The peak of the convolution of two identical rectangular pulses of height H and width W is H^2 W.
    # Here, we have a sum of pulses.
    # (sum h_i rect_i) * (sum h_j rect_j) = sum_{i,j} h_i h_j (rect_i * rect_j).
    # The peak of rect_i * rect_j is h_i h_j DELTA.
    # If i=j, peak is h_i^2 DELTA.
    # If i != j, the peaks might overlap or be separated.
    # If the supports are disjoint, the peaks add?
    # The support of rect_i * rect_j is [x_i, x_j+DELTA) (assuming x_i < x_j).
    # Width is (x_j - x_i) + DELTA.
    # If we use a Golomb ruler, the differences are distinct.
    # The peaks of the cross terms h_i h_j DELTA occur at the center of the interval between i and j.
    # These centers are distinct for a Golomb ruler.
    # The self terms h_i^2 DELTA occur at x_i.
    # Do the cross-term peaks overlap with self-term peaks?
    # A cross-term peak for (i,j) is at (x_i+x_j)/2.
    # A self-term peak is at x_k.
    # If (x_i+x_j)/2 = x_k, then x_i+x_j = 2x_k.
    # This means the ruler is not a "Sidon" set in the sense of distinct averages?
    # No, Golomb ruler only guarantees distinct differences.
    # It allows x_i+x_j = 2x_k (arithmetic progressions).
    # If there is an AP, the cross term peak overlaps with a self term peak.
    # Then the peak value is h_i^2 DELTA + h_j^2 DELTA + 2 h_i h_j DELTA ?
    # No, the convolution of rect_i and rect_j is a triangle.
    # At the midpoint, the value is h_i h_j DELTA.
    # The convolution of rect_k and rect_k is a triangle with peak h_k^2 DELTA.
    # If they overlap, the values add.
    # So peak = h_k^2 DELTA + h_i h_j DELTA.
    # To avoid this, we want a set with no 3-term arithmetic progressions.
    # A "3-free set" or "Sidon set" (B_2 sequence) has no solutions to x_i+x_j = x_k+x_l unless {i,j}={k,l}.
    # This implies no 3-term AP (since 2x_k = x_i+x_j would be a duplicate sum 2x_k = x_i+x_j).
    # So a Sidon set ensures that the peaks of the autoconvolution (from self terms) and cross-terms do not coincide.
    # In that case, the global max is max( max_i h_i^2, max_{i<j} h_i h_j ) * DELTA.
    # If we use uniform weights 1/K on a Sidon set of size K:
    # max self = 1/K^2.
    # max cross = 1/K^2.
    # So max conv = 1/K^2 * DELTA.
    # L2^2 = K * (1/K)^2 = 1/K.
    # L1 = 1.
    # Score = (1/K) / (1 * DELTA * 1/K^2) = K / DELTA = 2NK.
    #
    # THIS IS THE KEY!
    # If the set is Sidon (no 3-term AP, distinct sums), the peaks don't overlap.
    # The max conv is 1/K^2 (times DELTA).
    # The L2^2 is 1/K.
    # The ratio is K/DELTA = 2NK.
    #
    # For Uniform (N points), it's NOT Sidon.
    # Many sums collide.
    # Peak is at 0: sum h_i^2 = 1/N.
    # Ratio is (1/N) / (DELTA * 1/N) = 1/DELTA = 2N.
    #
    # For Sidon set of size K: Ratio = 2NK.
    # Since K > 1, this is better than Uniform!
    # Max K for N=50?
    # Sidon sets in [1, N] have size ~ sqrt(N).
    # sqrt(50) is approx 7.
    # So K=7 gives 2*50*7 = 700.
    # K=8 gives 800.
    #
    # So the strategy is:
    # 1. Find a large Sidon set in {0, ..., 49}.
    # 2. Return uniform weights on this set.
    #
    # I will implement a greedy search for a Sidon set.
    # Start with empty set.
    # Try adding 0, 1, 2, ...
    # Check if adding x creates a duplicate sum.
    # Sums are x+y for x,y in set (x<=y).
    #
    # Let's code this.

    def _find_sidon_set(n, max_size=10):
        # Greedy approach
        # We want to maximize size.
        # We can try to construct it.
        
        # A known good construction for small n is just greedy.
        # Let's try to find the largest possible.
        
        # Backtracking search might be better for optimality, but greedy is fast.
        # Let's do a simple greedy first.
        
        def is_sidon(s):
            sums = set()
            for i in range(len(s)):
                for j in range(i, len(s)):
                    val = s[i] + s[j]
                    if val in sums:
                        return False
                    sums.add(val)
            return True

        # Greedy: try to add smallest possible numbers
        # This might not give the largest set.
        # Better: try to fill as many as possible.
        
        # Let's use a randomized greedy or just a simple backtracking for small K.
        
        def backtrack(start, current, used_sums):
            # Try to add more elements
            best = current[:]
            
            for i in range(start, n):
                # Check if i can be added
                valid = True
                for s in current:
                    if s + i in used_sums:
                        valid = False
                        break
                if not valid:
                    continue
                if i + i in used_sums:
                    continue
                
                # Add i
                current.append(i)
                new_sums = set()
                for s in current[:-1]:
                    new_sums.add(s + i)
                new_sums.add(i + i)
                
                used_sums.update(new_sums)
                
                res = backtrack(i + 1, current, used_sums)
                if len(res) > len(best):
                    best = res[:]
                
                # Undo
                used_sums.difference_update(new_sums)
                current.pop()
            
            return best

        return backtrack(0, [], set())

    sidon_indices = _find_sidon_set(N, 10)
    
    # Fallback if something goes wrong
    if not sidon_indices:
        sidon_indices = list(range(N))
        
    K = len(sidon_indices)
    h = np.zeros(N)
    for idx in sidon_indices:
        h[idx] = 1.0 / K
        
    return list(h)
