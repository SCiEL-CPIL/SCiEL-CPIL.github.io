import hashlib

# EVOLVE-BLOCK-START
def priority(vector, n):
    # Use a structural heuristic based on the sum of coordinates.
    # Prioritizing vectors with lower sums (closer to the origin)
    # often allows for a denser packing in the initial stages of
    # greedy construction, as these vectors are less likely to form
    # arithmetic progressions with each other compared to random selection.
    return sum(vector)
# EVOLVE-BLOCK-END
