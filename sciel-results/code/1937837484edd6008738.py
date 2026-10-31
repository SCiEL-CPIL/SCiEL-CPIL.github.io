import hashlib

# EVOLVE-BLOCK-START
def priority(vector, n):
    # Use a deterministic hash to create a pseudo-random priority.
    # This breaks the lexicographic bias and often leads to larger cap sets
    # in greedy construction by distributing selections more evenly.
    vec_str = ''.join(map(str, vector))
    h = hashlib.md5(vec_str.encode('utf-8')).hexdigest()
    # Take first 8 hex digits for a 32-bit integer
    val = int(h[:8], 16)
    # Normalize to [0, 1)
    return val / (16 ** 8)
# EVOLVE-BLOCK-END
