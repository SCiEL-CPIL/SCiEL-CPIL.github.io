import time

N = 40

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 59.0

    # Base Hamiltonian cycle edges: 0-1-2-...-39-0
    edges = set()
    for i in range(N):
        edges.add((min(i, (i+1)%N), max(i, (i+1)%N)))

    # Chords from vertex 0 to vertices 2..20 (19 chords)
    # Total edges: 40 + 19 = 59
    for k in range(2, 21):
        edges.add((0, k))

    edge_list = sorted(list(edges))

    # Build cycles of every length 3..40
    cycle_by_length = {}

    # Length 40: Hamiltonian cycle
    cycle_by_length[40] = list(range(N))

    for k in range(2, 21):
        # Cycle 1: 0 -> 1 -> 2 -> ... -> k -> 0
        # Uses edges (0,1), (1,2), ..., (k-1,k), (0,k)
        # Length: k + 1
        len1 = k + 1
        cycle_by_length[len1] = list(range(k+1))

        # Cycle 2: 0 -> 39 -> 38 -> ... -> k -> 0
        # Uses edges (0,39), (39,38), (38,37), ..., (k+1,k), (0,k)
        # Length: (N - k) + 1 = 41 - k
        len2 = N - k + 1
        cycle_by_length[len2] = [0] + list(range(N-1, k-1, -1))

    # Verify all lengths 3..40 are covered
    final_cycles = []
    for L in range(3, N+1):
        if L in cycle_by_length:
            final_cycles.append(cycle_by_length[L])
        else:
            # Fallback: should never happen, but use Hamiltonian cycle truncated
            # This would be invalid, so we must ensure coverage
            raise RuntimeError(f"Missing cycle of length {L}")

    # Safety check: verify all edges exist for all cycles
    edge_set = set(edge_list)
    for c in final_cycles:
        for i in range(len(c)):
            u, v = c[i], c[(i+1) % len(c)]
            if (min(u, v), max(u, v)) not in edge_set:
                raise RuntimeError(f"Missing edge ({u},{v}) in cycle of length {len(c)}")

    return {'edges': edge_list, 'cycles': final_cycles}
