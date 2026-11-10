"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Returns a list of distinct integers in 1..256 such that subset sums
    with different cardinalities are distinct.
    
    We use the arithmetic progression {226, 227, ..., 256} of size 31.
    This set satisfies max(sum of k largest) < min(sum of k+1 smallest)
    for all k, ensuring the condition.
    
    Proof of optimality:
    For n=32, let x_1 < ... < x_32.
    We require Sum(15 largest) < Sum(16 smallest).
    Let D = Sum(x_1..x_16) - Sum(x_18..x_32). We need D > 0.
    The maximum possible D is achieved by an arithmetic progression with d=1.
    For AP x_i = a + i - 1:
    Sum(16 smallest) = 16a + 120.
    Sum(15 largest) = 15a + 360.
    D = a - 240.
    D > 0 implies a > 240, so a >= 241.
    Then x_32 = a + 31 >= 272, which exceeds 256.
    Thus, no set of size 32 exists in 1..256.
    Size 31 is achievable with a=226 (x_31 = 256, D = 226 - 225 = 1 > 0).
    """
    return [226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256]

# EVOLVE-BLOCK-END
