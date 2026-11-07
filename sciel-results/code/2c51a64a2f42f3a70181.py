def construct():
    n = 16
    result = []
    # Iterate over all unique pairs
    for u in range(n):
        for v in range(u + 1, n):
            # Keep only edges between the two partitions {0..7} and {8..15}
            if (u < 8) ^ (v < 8):
                result.append([u, v])
    return result
