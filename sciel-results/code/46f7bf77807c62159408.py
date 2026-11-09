import time

N = 40

def construct():
    t0 = time.time()
    limit = t0 + 50.0

    # Build a pancyclic graph: Hamiltonian cycle + minimal chords
    # Ring edges form cycle of length 40
    # Chords from hub vertex create pairs of cycle lengths

    edge_set = set()

    # Hamiltonian cycle: 0-1-2-...-39-0
    for i in range(N):
        a, b = i, (i + 1) % N
        edge_set.add((min(a, b), max(a, b)))

    # Select chords to cover cycle lengths 3..39
    # Chord (0, k) yields cycles of length k+1 and N-k+1
    # Need union over chosen k of {k+1, 41-k} to cover [3, 39]
    # Optimal: k in [2, 20], giving 19 chords

    hub = 0
    chord_targets = list(range(2, 21))

    for k in chord_targets:
        edge_set.add((hub, k))

    # Verify deadline
    if time.time() > limit:
        pass  # construction is deterministic and fast

    # Build cycle certificates
    # Map each required length to a valid cycle

    cyc = {}

    # Length 40: the full ring
    cyc[40] = list(range(N))

    # For each chord k, assign the two cycle lengths it creates
    for k in chord_targets:
        len_fwd = k + 1
        len_bwd = N - k + 1

        if len_fwd not in cyc:
            # Forward path: 0, 1, 2, ..., k
            cyc[len_fwd] = list(range(k + 1))

        if len_bwd not in cyc:
            # Backward path: 0, 39, 38, ..., k
            cyc[len_bwd] = [0] + list(range(N - 1, k - 1, -1))

    # Assemble output in order: lengths 3, 4, ..., 40
    ordered_cycles = []
    for L in range(3, 41):
        if L in cyc:
            ordered_cycles.append(cyc[L])
        else:
            raise RuntimeError(f"No cycle found for length {L}")

    # Finalize edge list
    final_edges = sorted((min(a, b), max(a, b)) for (a, b) in edge_set)

    return {'edges': final_edges, 'cycles': ordered_cycles}
