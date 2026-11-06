"""Cap set priority function for greedy construction."""

# EVOLVE-BLOCK-START
def priority(vector, n):
    zeros = 0
    ones = 0
    twos = 0
    for v in vector:
        if v == 0:
            zeros += 1
        elif v == 1:
            ones += 1
        else:
            twos += 1
    # Primary: zero count (higher = better)
    # Secondary: prefer vectors where non-zero entries are "mixed" (not all same)
    # This reduces the chance of forming APs with other similar vectors
    nonzero = ones + twos
    if nonzero > 0:
        # Penalize if all non-zeros are the same value
        if ones == 0 or twos == 0:
            mix_penalty = 1.0
        else:
            mix_penalty = 0.0
    else:
        mix_penalty = 0.0
    return zeros * 10.0 - mix_penalty
# EVOLVE-BLOCK-END
