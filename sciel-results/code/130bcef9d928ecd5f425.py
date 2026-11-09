"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def priority(vector, n):
    # Prioritize vectors with more 1s (central vectors first).
    # Central vectors as middle terms of 3-APs block many pairs early,
    # constraining the space so more total vectors can be added.
    # Tiebreak: higher sum of coordinates.
    count_ones = sum(1 for x in vector if x == 1)
    total_sum = sum(vector)
    return count_ones * 1000.0 + total_sum
# EVOLVE-BLOCK-END
