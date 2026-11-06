import time
import random

N = 40
START_TIME = time.time()
DEADLINE = 80  # safety margin before 90s limit

def elapsed():
    return time.time() - START_TIME

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 75  # safety margin

    # Base cycle: 0-1-2-...-39-0 (40 edges)
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))

    # Hub 0 chords: cover cycle lengths 3..20 (chords (0,k) for k=2..19)
    # Cycle of length L = k+1: 0-1-2-...-k-0
    for k in range(2, 20):
        edges.add((0, k))

    # Hub 20 chords: cover cycle lengths 3..12 (shifted)
    # Cycle: 20-21-22-...-k-20 (length k-19)
    # For length L, need chord (20, 19+L) ... but 19+L may exceed 39
    # Actually: cycle 20-21-...-k-20 has length k-20+1 = k-19
    # For L in 3..12: k = 19+L, so k in 22..31
    for L in range(3, 13):
        k = 19 + L
        if k < N:
            edges.add((20, k))

    # Cross chords for longer cycles (lengths 21..39)
    # Using hub 0 with reverse path: 0-39-38-...-k-0 (length 41-k)
    # For L in 21..39: k = 41-L, so k in 2..20
    # These chords (0,k) for k=2..20 are already added above (k=2..19)
    # Need (0,20) for L=21
    edges.add((0, 20))

    # Now generate cycles for each length 3..40
    cycles = {}
    edge_list = list(edges)

    def edge_exists(u, v):
        return (min(u, v), max(u, v)) in edges

    for L in range(3, N + 1):
        if L == N:
            cycles[L] = list(range(N))
            continue

        found = False
        # Try chord from hub 0, forward path: 0-1-2-...-k-0, length k+1
        k = L - 1
        if 2 <= k <= 38 and edge_exists(0, k):
            cyc = list(range(k + 1))
            valid = all(edge_exists(cyc[i], cyc[(i + 1) % (k + 1)]) for i in range(k + 1))
            if valid:
                cycles[L] = cyc
                found = True

        # Try chord from hub 0, reverse path: 0-39-38-...-k-0, length 41-k
        if not found:
            k2 = 41 - L
            if 2 <= k2 <= 38 and edge_exists(0, k2):
                cyc = [0] + list(range(39, k2 - 1, -1))
                valid = all(edge_exists(cyc[i], cyc[(i + 1) % L]) for i in range(L))
                if valid:
                    cycles[L] = cyc
                    found = True

        # Try chord from hub 20: 20-21-...-k-20, length k-19
        if not found:
            k3 = 19 + L
            if 20 <= k3 <= 39 and edge_exists(20, k3):
                cyc = list(range(20, k3 + 1))
                valid = all(edge_exists(cyc[i], cyc[(i + 1) % L]) for i in range(L))
                if valid:
                    cycles[L] = cyc
                    found = True

        # Try chord from hub 20, reverse: 20-19-18-...-k-20, length 20-k+1 = 21-k
        if not found:
            k4 = 21 - L
            if 0 <= k4 <= 19 and edge_exists(20, k4):
                cyc = [20] + list(range(19, k4 - 1, -1))
                valid = all(edge_exists(cyc[i], cyc[(i + 1) % L]) for i in range(L))
                if valid:
                    cycles[L] = cyc
                    found = True

        if not found:
            # Fallback: add necessary chord
            k = L - 1
            if 2 <= k <= 38:
                edges.add((0, k))
                cycles[L] = list(range(k + 1))
            else:
                k2 = 41 - L
                edges.add((0, k2))
                cycles[L] = [0] + list(range(39, k2 - 1, -1))

    # Verify all cycles
    def verify_all():
        for L in range(3, N + 1):
            cyc = cycles.get(L)
            if cyc is None or len(cyc) != L or len(set(cyc)) != L:
                return False
            for i in range(L):
                u, v = cyc[i], cyc[(i + 1) % L]
                if not edge_exists(u, v):
                    return False
        return True

    if not verify_all():
        # Fallback to known-good construction
        edges = set()
        for i in range(N):
            edges.add((i, (i + 1) % N))
        for k in range(2, 21):
            edges.add((0, k))
        cycles = {}
        for L in range(3, N + 1):
            if L == N:
                cycles[L] = list(range(N))
            else:
                k = L - 1
                if 2 <= k <= 20:
                    cycles[L] = list(range(k + 1))
                else:
                    k2 = 41 - L
                    cycles[L] = [0] + list(range(39, k2 - 1, -1))

    # Prune edges not used by any cycle
    used_edges = set()
    for L in range(3, N + 1):
        cyc = cycles[L]
        for i in range(L):
            u, v = cyc[i], cyc[(i + 1) % L]
            used_edges.add((min(u, v), max(u, v)))

    edges = used_edges

    # Second pass: try to find alternative cycles that use fewer edges
    # For each cycle, try to see if we can use a different path
    # This is a local search: try swapping one chord for another

    # Try to reduce edge count by finding cycles that share edges more
    # For now, just ensure we have the minimal set
    edge_list = list(edges)

    # Check if we can remove any edge
    changed = True
    while changed and elapsed() < deadline:
        changed = False
        for e in list(edge_list):
            trial_edges = set(edge_list)
            trial_edges.discard(e)
            # Check if all cycles still valid
            valid = True
            for L in range(3, N + 1):
                cyc = cycles[L]
                for i in range(L):
                    u, v = cyc[i], cyc[(i + 1) % L]
                    if (min(u, v), max(u, v)) not in trial_edges:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                edge_list = [x for x in edge_list if x != e]
                changed = True
                break

    return {'edges': edge_list, 'cycles': [cycles[L] for L in range(3, N + 1)]}
# EVOLVE-BLOCK-END
