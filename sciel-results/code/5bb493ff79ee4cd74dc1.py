import time
import zlib

# We implement a "line-blocking" heuristic for the greedy cap-set builder.
# The idea: prioritize vectors that, if added, would complete the most affine
# lines (i.e., pairs (u, v) in the space such that u, v, candidate form a line).
# This aggressively eliminates future conflict options early.

# We need to be fast: priority() is called for every candidate at each greedy step.
# For n up to ~12, 3^n is up to ~531441, which is too many to precompute fully.
# So we use a fast proxy: for a candidate vector c, the number of lines it completes
# is related to how many pairs (u, v) satisfy u + v + c = 0 mod 3 (element-wise).
# Actually, a line in Z_3^n is {a, b, c} where a + b + c = 0 and a, b, c are distinct.
# For a fixed c, the number of lines containing c is the number of pairs (a, b) with
# a + b = -c and a != b != c. But this is still expensive.

# Better proxy: we can use the number of "2-cycles" or use a weighted sum based on
# the vector's coordinates. A simpler and faster structural change: 
# Use the L1 distance from the "all-ones" vector (1,...,1) as a proxy for
# how "central" the vector is. Vectors close to all-ones have more balanced coordinates
# and may block more lines on average.

# Actually, let's try a different structural approach: use the sum of pairwise 
# coordinate differences as a measure of "spread". Vectors with high spread are 
# "more diverse" and may block more lines.

# Let me try: score = -sum_{i<j} |v_i - v_j| * (1/n^2)
# This penalizes vectors that are "uniform" (all same value) which might be bad 
# for cap sets, and rewards diverse vectors.

# But let's think about line-blocking more carefully. 
# A vector c blocks lines if it completes many lines. 
# For a random vector, the expected number of lines it completes is roughly (3^n - 1) / 2.
# But we want to find vectors that complete MORE than average.

# Fast proxy: the number of lines completed by c is maximized when c has coordinates 
# that are "spread out". So let's use the variance of the coordinates.

# Let's try: score = variance of coordinates (higher is better)
# Variance = (1/n) * sum((v_i - mean)^2)

# But we also need to break ties deterministically. Use zlib.crc32 of the vector.

def priority(vector, n):
    """
    Prioritize vectors by their 'line-blocking potential', approximated by
    the variance of their coordinates. Higher variance means more spread,
    which should block more future affine lines in the greedy construction.
    """
    # Compute mean and variance of coordinates
    if n == 0:
        return 0.0
    
    total = sum(vector)
    mean = total / n
    
    variance = 0.0
    for v in vector:
        diff = v - mean
        variance += diff * diff
    
    variance /= n
    
    # Deterministic tie-breaker using CRC32
    # Convert vector to bytes for CRC
    vec_bytes = bytes(vector)
    crc = zlib.crc32(vec_bytes) & 0xFFFFFFFF
    
    # Normalize CRC to [0, 1)
    tie_breaker = (crc / 0x100000000) * 0.001
    
    return variance + tie_breaker
