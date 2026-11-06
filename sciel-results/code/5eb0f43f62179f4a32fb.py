"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 40

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sparse pancyclic graph on N=40 vertices.
    Strategy: Hamiltonian cycle + 19 chords from vertex 0.
    Total edges: 59. Score: 19.
    """
    # Build edges: Hamiltonian cycle + chords (0,k) for k in 2..20
    edges = set()
    for i in range(N):
        a, b = i, (i + 1) % N
        edges.add((min(a, b), max(a, b)))
    for k in range(2, 21):
        edges.add((0, k))
    edges = sorted(edges)

    # Build cycles of every length 3..40
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            # Full Hamiltonian cycle
            cycle = list(range(N))
        elif L <= 21:
            # Use chord (0, L-1): cycle 0-1-...-(L-1)-0
            k = L - 1
            cycle = list(range(k + 1))
        else:
            # Use chord (0, 41-L): cycle 0-(41-L)-(42-L)-...-39-0
            k = 41 - L
            cycle = [0] + list(range(k, N))
        cycles.append(cycle)

    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
