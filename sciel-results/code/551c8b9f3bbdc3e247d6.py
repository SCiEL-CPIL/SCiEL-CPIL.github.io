# EVOLVE-BLOCK-START
def construct():
    n = 12
    # The minimal largest element for n distinct positive integers with distinct subset sums
    # is 2^(n-1). This is achieved by the powers of 2: 1, 2, 4, ..., 2^(n-1).
    # Proof: Let a_1 < ... < a_n. To avoid duplicate subset sums, a_k must not be a difference
    # of two subset sums of {a_1, ..., a_{k-1}}. The maximum difference is sum(a_1...a_{k-1}).
    # If we choose a_k > sum(a_1...a_{k-1}), it is safe. This leads to a_k >= 2^(k-1).
    # The powers of 2 satisfy this with equality.
    return [1 << i for i in range(n)]
# EVOLVE-BLOCK-END
