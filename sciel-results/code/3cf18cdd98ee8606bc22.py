def priority(vector, n):
    # Calculate the base-3 value of the vector with digits reversed.
    # This creates a permutation that is deterministic but "scrambles" the
    # lexicographic locality, potentially helping the greedy algorithm
    # avoid early collisions that would occur in a standard lex order.
    val = 0
    for i in range(n):
        val = val * 3 + vector[n - 1 - i]
    return float(val)
