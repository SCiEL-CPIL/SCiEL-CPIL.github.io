"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a set of distinct integers in 1..256 such that subset sums
    with different cardinalities are distinct.
    
    We use an arithmetic progression {A, A+1, ..., A+n-1}.
    The condition for disjoint sum sets is that the minimum sum of k elements
    is strictly greater than the maximum sum of k-1 elements.
    For an AP with difference 1, this requires A > (n-1)(n-k) roughly.
    Specifically, A > max_{k} (k-1)(n-k).
    The maximum of (k-1)(n-k) is at k = (n+1)/2.
    For n=31, max is 15*15 = 225. So A > 225. A=226.
    Max element is 226 + 30 = 256.
    This fits in 1..256.
    
    For n=32, max is 16*16 = 256. So A > 256. A=257.
    Max element is 257 + 31 = 288 > 256.
    So n=31 is the maximum for this family.
    """
    return list(range(226, 257))

# EVOLVE-BLOCK-END
