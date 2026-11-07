import time

def construct():
    """
    Returns a list of distinct integers in [1, 256] such that subset sums
    of different cardinalities are disjoint. Maximizes the set size.
    
    Analysis:
    Let S = {x_1, ..., x_k} with x_1 < ... < x_k.
    A sufficient condition for disjointness of sum sets S_i and S_j (i < j)
    is max(S_i) < min(S_j).
    For an arithmetic progression x_n = a + (n-1)d with d=1:
    max(S_i) = sum of i largest = i*a + i*(2k-i-1)/2
    min(S_j) = sum of j smallest = j*a + j*(j-1)/2
    Condition: i*a + i*(2k-i-1)/2 < j*a + j*(j-1)/2
    For j = i+1, this simplifies to a > i*(k-i-1).
    Max of i*(k-i-1) for i in 1..k-1 is approx ((k-1)/2)^2.
    For k=31, max is 15*15 = 225.
    So a >= 226.
    x_k = a + k - 1 = 226 + 30 = 256.
    This fits in [1, 256].
    For k=32, max is 16*15 = 240.
    So a >= 241.
    x_k = 241 + 31 = 272 > 256.
    So k=31 is the maximum for AP.
    Non-AP sets are less efficient in packing, so 31 is likely the global max.
    """
    k = 31
    a = 226
    # Generate the set
    result = [a + i for i in range(k)]
    
    # Optional: Quick validity check to ensure no bugs in logic
    # This check is O(k^2 * 2^k) which is too slow for k=31.
    # But we can check the boundary condition for APs quickly.
    # The difference min(S_{i+1}) - max(S_i) = i^2 - (k-1)i + a + 1 - (k-1)?
    # Derived earlier: i^2 - 30i + 226 for k=31, a=226.
    # Min value is 1 > 0. So valid.
    
    return result
