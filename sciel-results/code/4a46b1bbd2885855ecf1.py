import time

N = 40

def construct():
    # Circular difference set model (Ring + Chords)
    # Base: Cycle 0-1-...-39-0
    edges = []
    edges_set = set()

    def add_edge(a, b):
        if a == b:
            return
        if a > b:
            a, b = b, a
        if (a, b) not in edges_set:
            edges_set.add((a, b))
            edges.append((a, b))

    # Add Ring edges
    for i in range(N):
        add_edge(i, (i + 1) % N)

    # Add Chords to cover cycle lengths 3..39
    # Chord (0, d) has distance d.
    # Covers lengths d+1 and 41-d.
    # We need to cover 3..39.
    # d=2 covers 3, 39
    # d=3 covers 4, 38
    # ...
    # d=19 covers 20, 22
    # d=20 covers 21
    # So we need chords (0, d) for d in 2..20.
    
    chords = []
    for d in range(2, 21):
        add_edge(0, d)
        chords.append(d)

    cycles = []
    for L in range(3, N + 1):
        if L == N:
            # The ring itself
            cycles.append(list(range(N)))
        else:
            # Find a chord that covers L
            # L = d + 1 => d = L - 1
            # L = 41 - d => d = 41 - L
            found = False
            for d in chords:
                if d + 1 == L:
                    # Cycle 0, 1, ..., d, 0
                    # Wait, chord is (0, d).
                    # Path 0-1-...-d.
                    # Cycle: 0, 1, ..., d, 0.
                    # Length d+1.
                    cycles.append(list(range(d + 1)) + [0])
                    found = True
                    break
                elif 41 - d == L:
                    # Cycle 0, d, d+1, ..., 39, 0?
                    # Chord (0, d).
                    # Path d, d+1, ..., 39, 0.
                    # Cycle: 0, d, d+1, ..., 39, 0.
                    # Length: 1 (0-d) + (39-d) (d..39) + 1 (39-0) = 41-d.
                    cycle = [0] + list(range(d, N))
                    cycles.append(cycle)
                    found = True
                    break
            if not found:
                # Fallback if logic fails (should not happen)
                # Use a simple star+path fallback
                e2 = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
                c2 = [list(range(k)) for k in range(3, N + 1)]
                return {'edges': e2, 'cycles': c2}

    return {'edges': edges, 'cycles': cycles}
