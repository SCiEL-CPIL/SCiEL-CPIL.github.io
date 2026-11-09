"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    # Build a sparse pancyclic graph on 40 vertices.
    # Base: Hamiltonian rim (40 edges).
    # Chords: 19 chords from vertex 0 to vertices 2..20.
    # Total edges: 59. Score = 59 - 40 = 19.
    # Cycle certificates:
    #   Length k (3 <= k <= 21): 0, 1, 2, ..., k-1, 0  (uses chord (0, k-1))
    #   Length k (22 <= k <= 39): 0, 39, 38, ..., (40-k+1), 0  (uses chord (0, 40-k+1))
    #   Length 40: Hamiltonian cycle 0, 1, 2, ..., 39, 0

    edges = set()

    # Hamiltonian rim
    for i in range(N):
        j = (i + 1) % N
        edges.add((min(i, j), max(i, j)))

    # Chords from 0 to vertices 2 through 20
    for j in range(2, 21):
        edges.add((0, j))

    edge_list = sorted(edges)

    # Build cycle certificates for every length 3..40
    cycles = []
    for k in range(3, N + 1):
        if k <= 21:
            # Cycle: 0, 1, 2, ..., k-1, 0
            cycle = list(range(k))
        elif k < N:
            # Cycle: 0, 39, 38, ..., (N-k+1), 0
            start = N - k + 1
            cycle = [0] + list(range(N - 1, start - 1, -1))
        else:
            # Hamiltonian cycle: 0, 1, 2, ..., 39, 0
            cycle = list(range(N))
        cycles.append(cycle)

    return {'edges': edge_list, 'cycles': cycles}
# EVOLVE-BLOCK-END
