"""Program for the cap set task; improved EVOLVE-BLOCK."""

# EVOLVE-BLOCK-START
def priority(vector, n):
    # Use a deterministic, structured priority based on the vector's sum and a
    # secondary hash component to break ties. The sum-based priority encourages
    # the greedy algorithm to select vectors with similar "weight" first, which
    # can lead to more regular cap set structures. The secondary hash component
    # (using a simple polynomial hash) breaks ties in a way that is different
    # from CRC32, potentially exploring different configurations.
    s = sum(vector)
    # A simple hash to break ties deterministically
    h = 0
    for i, v in enumerate(vector):
        h = (h * 31 + v * (i + 1)) & 0xFFFFFFFF
    # Combine sum and hash: prioritize lower sums, then use hash for tie-breaking
    # Scale the sum to dominate, but allow hash to influence within same sum
    return float(s * 1000000 + h)
# EVOLVE-BLOCK-END
