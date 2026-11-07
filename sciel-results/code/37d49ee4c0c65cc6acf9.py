"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 16}

def construct():
    start_time = time.time()
    deadline = start_time + 85

    n = P['n']
    half = n // 2
    A = list(range(half))
    B = list(range(half, n))

    edges = []
    for u in A:
        if time.time() > deadline:
            break
        for v in B:
            if time.time() > deadline:
                break
            edges.append([int(u), int(v)])

    return edges
# EVOLVE-BLOCK-END
