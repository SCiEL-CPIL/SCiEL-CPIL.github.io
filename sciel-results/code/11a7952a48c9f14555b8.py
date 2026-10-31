import zlib

def priority(vector, n):
    # Convert tuple to bytes for hashing
    # Vector elements are 0, 1, 2
    # Heuristic: Prioritize vectors with fewer zeros (more "dense" or "off-center")
    # to avoid saturating the central, high-connectivity points early.
    count_zeros = 0
    for v in vector:
        if v == 0:
            count_zeros += 1
    return -float(count_zeros)
