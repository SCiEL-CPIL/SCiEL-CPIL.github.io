"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    deadline = time.time() + 85
    mask = (1 << 201) - 1  # bits 0..200

    def greedy(order):
        reachable = 1  # bit 0 set (empty subset sums to 0)
        included = []
        for x in order:
            new_reach = reachable | ((reachable << x) & mask)
            if (new_reach >> 200) & 1:
                continue  # adding x would create subset sum 200
            reachable = new_reach
            included.append(x)
        return included

    best = []

    # Structured orders
    structured = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
        list(range(100, 0, -2)) + list(range(99, 0, -2)),
        list(range(1, 101, 2)) + list(range(100, 0, -2)),
    ]

    for order in structured:
        result = greedy(order)
        if len(result) > len(best):
            best = result

    # Random orders for diverse exploration
    random.seed(42)
    for i in range(200000):
        if time.time() > deadline:
            break
        order = list(range(1, 101))
        random.shuffle(order)
        result = greedy(order)
        if len(result) > len(best):
            best = result

    return best
# EVOLVE-BLOCK-END
