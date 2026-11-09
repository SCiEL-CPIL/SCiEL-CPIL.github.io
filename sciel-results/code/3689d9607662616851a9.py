import time
import random

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Start with the known 59-edge fan construction
    # Then try to remove chords and replace with other edges to maintain pancyclicity

    # Base: Hamiltonian cycle 0-1-2-...-39-0
    base_edges = set()
    for i in range(N):
        u, v = i, (i + 1) % N
        if u > v:
            u, v = v, u
        base_edges.add((u, v))

    # Fan chords: 0-k for k in 2..20
    fan_chords = set()
    for k in range(2, 21):
        u, v = 0, k
        if u > v:
            u, v = v, u
        fan_chords.add((u, v))

    all_edges = base_edges | fan_chords

    def get_cycles_from_edges(edge_set):
        """Try to find cycles of all lengths 3..40 in the given edge set."""
        edge_set = set(edge_set)
        adj = {}
        for u, v in edge_set:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        cycles = {}

        # For each length L from 3 to 40, try to find a cycle
        for L in range(3, N + 1):
            found = False
            # Try different starting points and patterns
            # For the fan structure, cycles are:
            # - 0,1,2,...,k,0 (length k+1) for chord k
            # - 0,39,38,...,k,0 (length 41-k) for chord k
            # - Combinations of chords

            # Try to find cycle using BFS/DFS for each length
            # This is expensive, so use heuristics first

            # Heuristic: try the fan-based cycles first
            if L == N:
                # Hamiltonian cycle
                cycle = list(range(N))
                valid = all((cycle[i], cycle[(i+1)%N]) in edge_set or (cycle[(i+1)%N], cycle[i]) in edge_set for i in range(N))
                if valid:
                    cycles[L] = cycle
                    found = True

            if not found:
                # Try chord-based cycles
                for k in range(2, N - 1):
                    # Forward: 0,1,...,k,0
                    if k + 1 == L:
                        cycle = list(range(k + 1))
                        valid = True
                        for i in range(len(cycle)):
                            u, v = cycle[i], cycle[(i+1) % len(cycle)]
                            if (u, v) not in edge_set and (v, u) not in edge_set:
                                valid = False
                                break
                        if valid:
                            cycles[L] = cycle
                            found = True
                            break

                    # Backward: 0,39,38,...,k,0
                    if N - k + 1 == L:
                        cycle = [0] + list(range(N - 1, k - 1, -1))
                        valid = True
                        for i in range(len(cycle)):
                            u, v = cycle[i], cycle[(i+1) % len(cycle)]
                            if (u, v) not in edge_set and (v, u) not in edge_set:
                                valid = False
                                break
                        if valid:
                            cycles[L] = cycle
                            found = True
                            break

            if not found:
                # Try more complex cycles using multiple chords
                # For now, mark as not found
                cycles[L] = None

        return cycles

    # Try to remove chords one by one and see if we can maintain pancyclicity
    # with replacement edges

    current_edges = set(all_edges)

    # Try removing each chord and replacing with a non-fan edge
    # to see if we can reduce total edge count

    # For each chord 0-k, try removing it and adding an edge that helps
    # maintain the cycles that depended on it

    # Actually, let's try a different approach: start with fewer chords
    # and see if we can complete the graph

    # Try with chords 0-k for k in 2..m for various m
    # and see what's missing

    best_edges = current_edges
    best_cycles = get_cycles_from_edges(current_edges)

    # Check if all cycles are found
    def is_complete(cycles):
        for L in range(3, N + 1):
            if L not in cycles or cycles[L] is None:
                return False
        return True

    if is_complete(best_cycles):
        # Try to reduce
        for k in range(2, 21):
            if time.time() > deadline:
                break

            test_edges = set(current_edges)
            # Remove chord 0-k
            chord = (0, k) if 0 < k else (k, 0)
            if chord in test_edges:
                test_edges.remove(chord)

            # Try to find replacement: add an edge that helps with missing cycles
            # The cycles that used chord k were:
            # - length k+1: 0,1,...,k,0
            # - length 41-k: 0,39,...,k,0

            # Try adding edges that can form these cycles differently
            # For length k+1, we need a path from 0 to k of length k-1 not using chord k
            # For length 41-k, we need a path from 0 to k of length 40-k not using chord k

            # Try various replacement edges
            for u in range(N):
                if time.time() > deadline:
                    break
                for v in range(u + 1, N):
                    if (u, v) in test_edges:
                        continue
                    # Skip rim edges and fan chords (already in base)
                    if v == (u + 1) % N or u == (v + 1) % N:
                        continue
                    if u == 0 and 2 <= v <= 20:
                        continue
                    if v == 0 and 2 <= u <= 20:
                        continue

                    test_edges.add((u, v))
                    cycles = get_cycles_from_edges(test_edges)
                    if is_complete(cycles) and len(test_edges) < len(best_edges):
                        best_edges = test_edges
                        best_cycles = cycles
                    test_edges.remove((u, v))

            # If we found a better solution, update current
            if len(best_edges) < len(current_edges):
                current_edges = best_edges

            if time.time() > deadline:
                break

    # Final check
    final_cycles = get_cycles_from_edges(best_edges)

    # If not complete, fall back to the known good solution
    if not is_complete(final_cycles):
        final_cycles = get_cycles_from_edges(all_edges)
        best_edges = all_edges

    # Build the result
    edge_list = sorted(list(best_edges))
    cycle_list = [final_cycles[L] for L in range(3, N + 1)]

    return {'edges': edge_list, 'cycles': cycle_list}
