"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a set of distinct integers in 1..256 such that subset sums
    with different cardinalities are distinct.
    
    We use an arithmetic progression {a, a+1, ..., a+n-1} with n=31 and a=226.
    This set is {226, 227, ..., 256}.
    
    For an arithmetic progression, the set of subset sums of cardinality k, S_k,
    forms a contiguous interval of integers [min_k, max_k].
    The condition that S_k and S_{k+1} are disjoint is equivalent to max_k < min_{k+1}.
    
    min_k = sum of k smallest elements = k*a + k(k-1)/2
    max_k = sum of k largest elements = k*a + k(n-1) - k(k-1)/2
    
    We require max_k < min_{k+1} for all k=1..n-1.
    max_k < min_{k+1}  <=>  k*a + k(n-1) - k(k-1)/2 < (k+1)*a + k(k+1)/2
    <=>  k(n-1) - k(k-1)/2 < a + k(k+1)/2
    <=>  k(n-1) - k^2 < a
    
    The maximum of f(k) = k(n-1) - k^2 for k in 1..n-1 occurs near k=(n-1)/2.
    For n=31, max f(k) is at k=15: 15*30 - 225 = 225.
    So we need a > 225.
    We also need a+n-1 <= 256 => a <= 256-30 = 226.
    So a=226 is the unique valid value for n=31.
    
    For n=32, max f(k) is at k=15 or 16: 15*31 - 225 = 240.
    We need a > 240.
    But a <= 256-31 = 225.
    Contradiction. So n=32 is not possible with AP.
    Given the density of subset sums, n=32 is likely impossible for any set.
    """
    return list(range(226, 257))

# EVOLVE-BLOCK-END
