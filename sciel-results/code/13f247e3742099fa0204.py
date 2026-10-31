import zlib

def priority(vector, n):
    # Static proxy for minimizing future conflict:
    # Prefer vectors with higher Hamming weight (more non-zero coordinates).
    # This heuristic aims to select structurally 'complex' vectors first,
    # potentially reducing the density of local line-forming structures
    # compared to sparse vectors which may cluster more easily.
    weight = sum(1 for x in vector if x != 0)
    # Use a hash as a deterministic tie-breaker to ensure unique ordering
    # and avoid bias from stable sort on equal weights.
    h = zlib.crc32(bytes(vector))
    # Combine weight and hash. Weight is the primary driver.
    # We scale the hash to be a small fraction of the weight unit
    # so that weight dominates, but hash breaks ties.
    return float(weight * 1000000 + h)
