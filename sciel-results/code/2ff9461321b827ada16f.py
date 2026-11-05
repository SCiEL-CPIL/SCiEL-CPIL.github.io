import time
import random

N = 40

def _validate(edges, cycles):
    """
    Validate the graph and cycles.
    - Edges are undirected, simple, on vertices 0..N-1
    - Cycles are vertex-lists of lengths 3..N in that order
    - Each cycle has distinct vertices
    - Each cycle's edges exist in the edge set
    """
    n = N
    # Build edge set for O(1) lookup
    edge_set = set()
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n) or u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in edge_set:
            return False
        edge_set.add((u, v))

    if len(cycles) != n - 2:  # lengths 3..n
        return False

    for i, cycle in enumerate(cycles):
        L = i + 3
        if len(cycle) != L:
            return False
        # Distinct vertices
        if len(set(cycle)) != L:
            return False
        # All edges in cycle must exist
        for j in range(L):
            u = cycle[j]
            v = cycle[(j + 1) % L]
            if u > v:
                u, v = v, u
            if (u, v) not in edge_set:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s/90s limit

    # ------------------------------------------------------------------
    # Strategy: Start with the proven 59-edge construction (Hamiltonian
    # cycle + 19 chords from vertex 0). Then attempt local search to
    # find a 58-edge pancyclic graph by trying to eliminate one chord
    # while replacing lost cycle coverage with alternative cycles.
    #
    # The current best uses:
    #   - 40 Hamiltonian cycle edges
    #   - 19 chords: (0,2), (0,3), ..., (0,20)
    # Total: 59 edges. Score contribution: 59 - 40 = 19.
    #
    # If we can get 58 edges, score = 58/19 ≈ 3.05 (vs 1.947).
    #
    # Approach: Try removing each chord (0,d) and see if all cycle
    # lengths can still be covered by remaining edges. Use BFS/DFS
    # to find cycles of required lengths in the reduced graph.
    # ------------------------------------------------------------------

    # Base construction: Hamiltonian cycle + chords
    ham_edges = [(i, (i + 1) % N) for i in range(N)]
    chord_edges = [(0, d) for d in range(2, 21)]
    base_edges = ham_edges + chord_edges

    # Base cycles (always valid with base_edges)
    def make_base_cycles():
        cycles = []
        for L in range(3, N + 1):
            if L == N:
                cycles.append(list(range(N)))
            elif L <= 21:
                cycles.append(list(range(L)))
            else:
                end_vertex = N + 1 - L
                cycles.append([0] + list(range(N - 1, end_vertex - 1, -1)))
        return cycles

    base_cycles = make_base_cycles()

    # Verify base construction
    if not _validate(base_edges, base_cycles):
        return {'edges': base_edges, 'cycles': base_cycles}

    best_edges = base_edges
    best_cycles = base_cycles
    best_edge_count = len(base_edges)

    # Attempt to remove chords one at a time and find alternative cycles
    # For each chord (0, d) that we remove, we lose:
    #   - Cycle of length d+1 (forward: 0->1->...->d->0)
    #   - Cycle of length N+1-d (backward: 0->N-1->...->N+1-d->0) if d != N+1-d
    #   - If d == N+1-d (i.e., d=20), we lose one cycle of length 21
    # We need to find these cycles using other edges.

    # Precompute adjacency for base graph
    def build_adj(edges):
        adj = [[] for _ in range(N)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    # Find a simple cycle of exact length L in graph, avoiding a forbidden edge
    # Returns list of vertices or None
    def find_cycle_length_L(adj, L, forbidden_edge=None):
        """
        Find a simple cycle of exactly L vertices in the graph defined by adj.
        If forbidden_edge is given, the cycle must not use that edge.
        Uses DFS with pruning. Returns vertex list or None.
        """
        if L < 3 or L > N:
            return None

        fe = None
        if forbidden_edge is not None:
            u, v = forbidden_edge
            if u > v:
                u, v = v, u
            fe = (u, v)

        # Try starting from vertex 0 to reduce symmetry
        # Actually, try a few starting vertices
        for start in [0, 1, 2]:
            # DFS: path from start, trying to close cycle of length L
            # State: (current_node, path, visited_set)
            # Use iterative DFS with pruning

            def dfs(start):
                # Path starts with [start], we need L-1 more vertices to complete
                # Then check edge back to start
                stack = [(start, [start])]
                visited_in_path = {start}

                # This DFS can explode; limit depth by L
                # Use a set of (node, frozenset(path)) - too expensive
                # Instead, simple DFS with backtracking, limited by time

                # Better: use a recursive approach with early termination
                path = [start]
                visited = {start}

                def recurse(current, depth):
                    if depth == L:
                        # Check if edge (current, start) exists
                        if start in adj[current]:
                            if fe is not None and (min(current, start), max(current, start)) == fe:
                                return None
                            return list(path)
                        return None
                    for neighbor in adj[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            path.append(neighbor)
                            result = recurse(neighbor, depth + 1)
                            if result is not None:
                                return result
                            path.pop()
                            visited.discard(neighbor)
                    return None

                return recurse(start, 1)

            result = dfs(start)
            if result is not None:
                return result

        return None

    # Try removing each chord and see if we can still cover all cycle lengths
    for chord_idx in range(len(chord_edges)):
        if time.time() > deadline:
            break

        chord = chord_edges[chord_idx]
        u, v = chord
        d = max(u, v)  # d is in 2..20

        # Edges without this chord
        trial_edges = [e for i, e in enumerate(base_edges) if i != N + chord_idx]
        # base_edges = ham_edges (indices 0..39) + chord_edges (indices 40..58)
        # chord_idx maps to base_edges index N + chord_idx

        # Determine which cycle lengths were covered ONLY by this chord
        # Chord (0,d) directly gives cycles of length d+1 and N+1-d
        lengths_at_risk = set()
        if d + 1 <= N:
            lengths_at_risk.add(d + 1)
        if N + 1 - d >= 3 and N + 1 - d != d + 1:
            lengths_at_risk.add(N + 1 - d)

        # For each at-risk length, try to find an alternative cycle
        trial_adj = build_adj(trial_edges)
        valid = True
        trial_cycles = []

        for L in range(3, N + 1):
            if time.time() > deadline:
                valid = False
                break

            if L not in lengths_at_risk:
                # Use base cycle
                trial_cycles.append(base_cycles[L - 3])
            else:
                # Need to find alternative cycle of length L
                alt_cycle = find_cycle_length_L(trial_adj, L, forbidden_edge=chord)
                if alt_cycle is None:
                    valid = False
                    break
                trial_cycles.append(alt_cycle)

        if valid:
            # Verify the trial solution
            if _validate(trial_edges, trial_cycles):
                if len(trial_edges) < best_edge_count:
                    best_edges = trial_edges
                    best_cycles = trial_cycles
                    best_edge_count = len(trial_edges)
                    # We found a better solution; can stop early
                    break

    return {'edges': best_edges, 'cycles': best_cycles}
