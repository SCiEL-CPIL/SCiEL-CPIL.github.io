"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import zlib

# EVOLVE-BLOCK-START
def priority(vector, n):
    # Approach: Combinatorial Structure Priority
    # Instead of randomizing with a hash, we prioritize vectors based on
    # their composition of 1s and 2s.
    # This encourages the greedy algorithm to fill the space in layers of
    # increasing "distance" from the zero vector in terms of symbol counts.
    # This can help avoid early clustering of collinear points that often
    # occurs with lexicographic or simple sum-based orderings.
    n1 = 0
    n2 = 0
    for x in vector:
        if x == 1:
            n1 += 1
        elif x == 2:
            n2 += 1
    # We want to interleave 1s and 2s to avoid lines like (1,1,1) and (2,2,2)
    # being added too close together if they are in the same layer.
    # A simple linear combination with a large weight for n1 ensures
    # vectors with more 1s are generally prioritized over those with fewer,
    # but within the same n1 count, we order by n2.
    # Let's try a specific mix: (n1 * 3 + n2) might be too coarse.
    # Let's use a hash of the counts to break ties? No, keep it deterministic and structural.
    # Let's try: priority = n1 * n + n2
    return float(n1 * 1000 + n2)
# EVOLVE-BLOCK-END
