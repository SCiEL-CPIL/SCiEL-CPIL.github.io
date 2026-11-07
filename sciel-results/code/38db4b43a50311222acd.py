import time

# EVOLVE-BLOCK-START
def construct():
    # The optimal set size is 31.
    # We use an arithmetic progression with difference 1.
    # Condition for disjoint subset sums of different cardinalities:
    # min_sum(k) > max_sum(k-1) for all k.
    # For x_i = A + (i-1), this requires A > (k-1)(n-k).
    # For n=31, max of (k-1)(31-k) is at k=16: 15*15=225.
    # So A >= 226.
    # x_31 = 226 + 30 = 256, which is within 1..256.
    
    k = 31
    x_1 = 226
    result = [x_1 + i for i in range(k)]
    
    # Optional: Verify validity to ensure no bugs in logic
    # (Skipped for speed as it's deterministic and proven)
    
    return result
# EVOLVE-BLOCK-END
