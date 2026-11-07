"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 16}

def construct():
    n = P['n']
    half = n // 2
    edges = []
    # Pre-calculate size for minor optimization
    # K_{half, half} has half * half edges
    for u in range(half):
        for v in range(half, n):
            edges.append([int(u), int(v)])
    return edges
# EVOLVE-BLOCK-END
