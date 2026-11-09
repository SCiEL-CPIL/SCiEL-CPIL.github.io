"""Sparse pancyclic graphs on 40 vertices: single-hub with optimized chord set."""
import time

N = 40


def _verify_cyc(cyc, edge_set):
    """Check a cycle: distinct vertices, all edges present."""
    k = len(cyc)
    if len(set(cyc)) != k:
        return False
    for i in range(k):
        u, v = cyc[i], cyc[(i + 1) % k]
        if (min(u, v), max(u, v)) not in edge_set:
            return False
    return True


def construct():
    t0 = time.time()
    DL = t0 + 70.0

    # ---- Single-hub construction with optimized chord set ----
    # Rim: 40 edges (cycle 0-1-2-...-39-0)
    edges = set()
    for i in range(N):
        a, b = i, (i + 1) % N
        edges.add((min(a, b), max(a, b)))

    # Chord set K: 15 chords from hub 0
    # K = {2,3,4,5,6,7,8,9,10,11,12,13,14,15,24}
    # This covers all cycle lengths 3..39 via:
    #   - Single chord k: cycle length k+1 (forward) or 41-k (backward)
    #   - Double chord |k1-k2|: cycle length |k1-k2|+2
    K = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 24]
    K_set = set(K)
    for k in K:
        edges.add((0, k))

    # Total edges: 40 rim + 15 chords = 55 edges
    # Objective: 55 - 40 = 15

    # Build and verify cycles for lengths 3..40
    cset = set(edges)
    cycs = []
    valid = True

    for length in range(3, N + 1):
        if length == N:
            c = list(range(N))
        else:
            found = False

            # Try single chord: L = k+1 => k = L-1
            k1 = length - 1
            if k1 in K_set:
                c = list(range(k1 + 1))
                found = True

            # Try single chord: L = 41-k => k = 41-L
            if not found:
                k2 = 41 - length
                if k2 in K_set:
                    c = [0] + list(range(N - 1, k2 - 1, -1))
                    found = True

            # Try double chord: L = 2 + |k1 - k2| => |k1-k2| = L-2
            if not found:
                diff = length - 2
                for u in K:
                    v1 = u + diff
                    v2 = u - diff
                    if v1 in K_set and v1 != u:
                        # Cycle 0-u-(u+1)-...-v1-0
                        c = [0] + list(range(u, v1 + 1))
                        found = True
                        break
                    elif v2 in K_set and v2 != u:
                        # Cycle 0-v2-(v2+1)-...-u-0
                        c = [0] + list(range(v2, u + 1))
                        found = True
                        break

            if not found:
                valid = False
                break

        if not _verify_cyc(c, cset):
            valid = False
            break
        cycs.append(c)

    if valid:
        best_edges = sorted(edges)
        best_cycs = cycs
    else:
        # Fallback to 57-edge construction
        rim2 = set()
        for i in range(N):
            a, b = i, (i + 1) % N
            rim2.add((min(a, b), max(a, b)))

        fan_chords = set()
        for d in range(2, 19):
            fan_chords.add((0, d))

        bridge2 = (2, 21)
        e58 = frozenset(rim2 | fan_chords | {bridge2})

        best_edges = sorted(e58)
        best_cycs = []
        for k in range(3, N + 1):
            if k == N:
                c = list(range(N))
            elif k <= 19:
                c = list(range(k))
            elif k == 20:
                c = list(range(2, 22))
            elif k == 21:
                c = [0, 2] + list(range(21, 40))
            elif k == 22:
                c = [2, 1, 0] + list(range(39, 20, -1))
            else:
                dp = 41 - k
                c = [0] + list(range(39, dp - 1, -1))
            best_cycs.append(c)

    # Final validation
    final_es = set(best_edges)
    assert len(final_es) == len(best_edges), "Duplicate edges"
    for c in best_cycs:
        assert _verify_cyc(c, final_es), f"Invalid cycle: {c}"

    return {'edges': best_edges, 'cycles': best_cycs}
