def priority(vector, n):
    # Heuristic: Minimize "saturation" by preferring vectors with lower Hamming weight
    # (fewer non-zero elements). This is a proxy for lower geometric density/conflict potential.
    weight = sum(1 for x in vector if x != 0)
    # Return negative weight so that lower weight (higher priority) comes first in descending sort.
    return -float(weight)
