"""Cap set greedy builder with coordinate-balance priority heuristic."""
# EVOLVE-BLOCK-START
def priority(vector, n):
    """
    Assign priority based on coordinate balance.
    Vectors with more 1s (closer to the center) are given higher priority,
    as they participate in more potential 3-term APs and are more constrained.
    Placing them first forces the greedy algorithm to be more selective,
    potentially yielding a larger cap set.
    """
    count_ones = sum(1 for v in vector if v == 1)
    return float(count_ones)
# EVOLVE-BLOCK-END
