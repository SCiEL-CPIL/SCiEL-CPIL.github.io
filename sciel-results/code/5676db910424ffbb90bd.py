"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    # Build a "fan" plus a few chords: vertex 0 connects to all others, and there is
    # a path through vertices 1..39.  We add a few chords to allow shorter cycles.
    # The cycles for each length k are constructed explicitly.
    edges_set = set()
    edges_list = []
    def add_edge(a, b):
        a, b = sorted((a, b))
        if a != b and (a, b) not in edges_set:
            edges_set.add((a, b))
            edges_list.append((a, b))

    # Star edges from 0
    for i in range(1, N):
        add_edge(0, i)

    # Path edges
    for i in range(1, N - 1):
        add_edge(i, i + 1)

    # Chords to enable short cycles
    add_edge(0, 3)
    add_edge(1, 3)
    add_edge(0, 4)
    add_edge(1, 4)
    add_edge(0, 5)
    add_edge(1, 5)
    add_edge(2, 5)

    cycles = []

    # Build cycles of length k = 3..40
    # For k = 3: triangle 0-1-3-0
    cycles.append([0, 1, 3])

    # For k = 4: 0-1-2-3-0
    cycles.append([0, 1, 2, 3])

    # For k = 5: 0-1-2-3-4-0
    cycles.append([0, 1, 2, 3, 4])

    # For k = 6: 0-1-2-3-4-5-0
    cycles.append([0, 1, 2, 3, 4, 5])

    # For k = 7..40: use the path plus 0
    for k in range(7, N + 1):
        # 0 - 1 - 2 - ... - (k-1) - 0
        cyc = [0] + list(range(1, k))
        cycles.append(cyc)

    return {'edges': edges_list, 'cycles': cycles}
# EVOLVE-BLOCK-END
