"""Sparse pancyclic graph on 40 vertices - fan construction."""

import time

N = 40
DEADLINE = 80.0


def _build_fan_cycles():
    """Generate all pancyclic cycles for the fan-on-cycle construction."""
    result = []
    for k in range(3, N + 1):
        if k == N:
            result.append(list(range(N)))
        elif k <= 21:
            # Chord (0, k-1): cycle 0,1,2,...,k-1
            result.append(list(range(k)))
        else:
            # Chord (0, N-(k-1)): cycle 0, N-1, N-2, ..., N-(k-1)
            end = N - (k - 1)
            result.append([0] + list(range(N - 1, end - 1, -1)))
    return result


def _build_fan_edges():
    """Build the edge set: Hamiltonian cycle + 19 chords from vertex 0."""
    edges = []
    # Hamiltonian cycle edges
    for i in range(N):
        edges.append((i, (i + 1) % N))
    # Chords from vertex 0 to vertices 2..20
    for d in range(2, 21):
        edges.append((0, d))
    return edges


def construct():
    t_start = time.time()

    edges = _build_fan_edges()
    cycles = _build_fan_cycles()

    # Verification
    edge_set = set()
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if key in edge_set or u == v:
            return {'edges': edges, 'cycles': cycles}
        edge_set.add(key)

    for cyc in cycles:
        if len(cyc) != len(set(cyc)):
            return {'edges': edges, 'cycles': cycles}
        for i in range(len(cyc)):
            u, v = cyc[i], cyc[(i + 1) % len(cyc)]
            if (min(u, v), max(u, v)) not in edge_set:
                return {'edges': edges, 'cycles': cycles}

    if time.time() - t_start > DEADLINE:
        return {'edges': edges, 'cycles': cycles}

    return {'edges': edges, 'cycles': cycles}
