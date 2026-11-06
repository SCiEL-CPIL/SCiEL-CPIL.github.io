"""Size-efficient sorting network search on 18 wires.
Starts from a correct baseline and prunes redundant comparators using the
0-1 principle (verified over all 2^18 Boolean inputs) under a hard deadline.
"""
import time
import random

N = 18
M = 1 << N
DEADLINE = time.time() + 80.0  # safety margin under the 90s kill

# EVOLVE-BLOCK-START
def _odd_even():
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def _is_sorted_all(net):
    """0-1 principle check over all 2^N Boolean inputs."""
    for mask in range(M):
        wires = [(mask >> i) & 1 for i in range(N)]
        for (a, b) in net:
            x, y = wires[a], wires[b]
            if x > y:
                wires[a], wires[b] = y, x
        for i in range(N - 1):
            if wires[i] > wires[i + 1]:
                return False
    return True


def construct():
    """Returns a valid comparator list, pruned as far as the budget allows."""
    net = _odd_even()
    n = len(net)
    rng = random.Random(20240611)

    # Pass 1: greedy forward prune (cheap, deterministic).
    for idx in range(n):
        if time.time() > DEADLINE:
            break
        cand = net[:idx] + net[idx + 1:]
        if _is_sorted_all(cand):
            net = cand
            n -= 1

    # Pass 2: randomized prune with occasional double-removal.
    while time.time() < DEADLINE:
        idx = rng.randrange(n)
        try_k = 2 if (n > 2 and rng.random() < 0.3) else 1
        try_k = min(try_k, n - 1)
        cand = net[:idx] + net[idx + try_k:]
        if _is_sorted_all(cand):
            net = cand
            n -= try_k
        if n <= 70:
            break
    return net
# EVOLVE-BLOCK-END
