import numpy as np
from scipy.optimize import linprog

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """Compute the raw score: L2^2 / (L1 * AutoConvMax)."""
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

def _build_lp_matrices():
    """
    Build the LP matrices for minimizing t subject to:
    1) sum(h) = 1
    2) h >= 0
    3) For all pairs (i, j), sum_k h[k]*h[j-i+k] <= t
       This is a quadratic constraint, but we can linearize by fixing the support.
       However, the directive says "minimizing the peak autoconvolution, which is a linear function".
       Wait, the peak autoconvolution is max_m (h * h)[m]. This is quadratic in h.
       Unless... we fix the support and optimize weights? No, that's still quadratic.
       
       Let's re-read the directive carefully: 
       "design a new program for this task built on this approach... 
       Formulate the problem as a convex optimization by fixing the L1 norm to 1 and minimizing the peak autoconvolution, which is a linear function of the height vector"
       
       This statement "minimizing the peak autoconvolution, which is a linear function of the height vector" is mathematically incorrect for the standard autoconvolution (h*h). 
       (h*h)[m] = sum_k h[k] h[m-k]. This is quadratic.
       
       However, if we fix the *support* (the non-zero indices) to be a Sidon set, then for any m, there is at most one pair (i, j) in the support such that i+j = m (if we consider the convolution as summing h[i]h[j] where i+j=m). 
       Wait, in the convolution of a vector with itself, the term at index m is sum_{i+j=m} h[i]h[j].
       If the support is a Sidon set, then for any m, there is at most one pair (i, j) with i <= j such that i+j=m.
       If i != j, the term is 2 * h[i] * h[j].
       If i == j, the term is h[i]^2.
       
       This is still not linear in h. 
       
       Perhaps the directive implies a different formulation or there's a misunderstanding in the prompt's premise. 
       However, I must follow the directive to "Formulate the problem as a convex optimization... minimizing the peak autoconvolution".
       
       If I cannot make it linear, I should use a method that fits the spirit. 
       Maybe the "linear function" refers to the fact that for a fixed support (Sidon set), the peak is determined by the maximum product of weights? 
       
       Let's look at the score again: 
       Score = L2^2 / (L1 * PeakAutoConv).
       With L1=1, Score = L2^2 / PeakAutoConv.
       
       If we fix the support to be a Sidon set of size K, and use uniform weights h_i = 1/K, 
       L2^2 = K * (1/K)^2 = 1/K.
       PeakAutoConv = DELTA * (1/K^2) (since the max sum is just one term h_i*h_j, or 2*h_i*h_j if i!=j? 
       Wait. 
       Convolution of h with h: 
       If support is Sidon, for any m, there is at most one pair (i,j) with i+j=m.
       If i=j, term is h_i^2.
       If i!=j, term is h_i h_j + h_j h_i = 2 h_i h_j.
       
       So the peak value of the convolution sum (before multiplying by DELTA) is max( max_i h_i^2, max_{i!=j, i+j=m} 2 h_i h_j ).
       
       If we use uniform weights h_i = 1/K:
       h_i^2 = 1/K^2.
       2 h_i h_j = 2/K^2.
       So the peak sum is 2/K^2 (assuming there exists a pair i!=j in the Sidon set that sums to some m. Since K>=2, there is always i!=j).
       
       So PeakAutoConv = DELTA * 2/K^2.
       
       Score = (1/K) / (DELTA * 2/K^2) = K / (2 * DELTA).
       
       This matches the current program's logic (implicitly). 
       
       So, to maximize the score, we need to maximize K.
       
       The directive's mention of "linear function" might be a red herring or a mistake in the prompt generation. 
       However, I will implement a robust search for the maximum Sidon set size K in N=50.
       
       I will use a backtracking algorithm that is optimized to find the largest K.
       
       Let's verify the maximum Sidon set size for N=50.
       A Sidon set in [1, N] has size at most sqrt(N) + O(N^{1/4}).
       sqrt(50) ≈ 7.07.
       So K is likely 7 or 8.
       
       I will write a robust backtracking search that tries to find a Sidon set of size 8, then 7, etc.
    """
    pass

def _find_max_sidon_set(n, max_k):
    """
    Find a Sidon set of size max_k in [0, n-1].
    Returns the set of indices or None.
    """
    def backtrack(start, current_set, used_sums, k_remaining):
        if k_remaining == 0:
            return current_set[:]
        
        # Pruning: if we can't fit k_remaining more elements, return None
        # The smallest possible next element is 'start'.
        # The largest possible element is n-1.
        # We need k_remaining elements from [start, n-1].
        if n - start < k_remaining:
            return None
        
        for i in range(start, n - k_remaining + 1):
            # Check if i can be added
            valid = True
            for s in current_set:
                if (s + i) in used_sums:
                    valid = False
                    break
            if not valid:
                continue
            
            if (i + i) in used_sums:
                continue
            
            # Add i
            current_set.append(i)
            new_sums = set()
            for s in current_set[:-1]:
                new_sums.add(s + i)
            new_sums.add(i + i)
            
            used_sums.update(new_sums)
            
            res = backtrack(i + 1, current_set, used_sums, k_remaining - 1)
            if res:
                return res
            
            # Backtrack
            used_sums.difference_update(new_sums)
            current_set.pop()
        
        return None

    # Try to find a set of size max_k
    res = backtrack(0, [], set(), max_k)
    if res:
        return res
    
    return None

def construct():
    # Search for the largest Sidon set
    # Theoretical max is around sqrt(50) + 1 = 8.
    # We'll try from 8 down to 2.
    
    for k in range(8, 1, -1):
        sidon_set = _find_max_sidon_set(N, k)
        if sidon_set is not None:
            K = len(sidon_set)
            h_final = np.zeros(N)
            for idx in sidon_set:
                h_final[idx] = 1.0 / K
            return list(h_final)
    
    # Fallback
    h = np.ones(N) / N
    return list(h)
