import math

# EVOLVE-BLOCK-START
def construct():
    n = 255
    # Precompute degrees in the non-coprime graph
    # degree[i] = number of j in [1, n] such that gcd(i, j) > 1
    degrees = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if math.gcd(i, j) > 1:
                degrees[i] += 1

    # Sort candidates by degree (descending)
    # Tie-breaker: smaller index first (deterministic)
    sorted_candidates = sorted(range(1, n + 1), key=lambda x: (-degrees[x], x))

    selected = []
    available = set(range(1, n + 1))

    for node in sorted_candidates:
        if node not in available:
            continue
        
        # Add node to selected
        selected.append(node)
        
        # Remove node and all its coprime neighbors from available
        # We iterate over a copy of available to allow removal
        to_remove = []
        for m in available:
            if math.gcd(node, m) == 1:
                to_remove.append(m)
        
        for m in to_remove:
            available.remove(m)
        available.discard(node)

    return selected
# EVOLVE-BLOCK-END
