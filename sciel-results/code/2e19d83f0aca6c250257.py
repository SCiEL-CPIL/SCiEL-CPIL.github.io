import time
import random

N = 40

def _validate(edges, cycles):
    n = N
    edge_set = set()
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n) or u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in edge_set:
            return False
        edge_set.add((u, v))

    if len(cycles) != n - 2:
        return False

    for i, cycle in enumerate(cycles):
        L = i + 3
        if len(cycle) != L:
            return False
        if len(set(cycle)) != L:
            return False
        for j in range(L):
            u = cycle[j]
            v = cycle[(j + 1) % L]
            if u > v:
                u, v = v, u
            if (u, v) not in edge_set:
                return False
    return True

def _build_adj(edges):
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _find_cycle(adj, L, forbidden=None, start_vertices=None):
    """Find a simple cycle of length L in graph adj, avoiding forbidden edge if given."""
    if L < 3 or L > N:
        return None
    fe = None
    if forbidden is not None:
        u, v = forbidden
        if u > v:
            u, v = v, u
        fe = (u, v)
    
    if start_vertices is None:
        start_vertices = [0, 1, 2, 3, 4]
    
    for start in start_vertices:
        path = [start]
        visited = {start}
        
        def recurse(current, depth):
            if depth == L:
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
        
        result = recurse(start, 1)
        if result is not None:
            return result
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # ------------------------------------------------------------------
    # Strategy: Use a circular difference set / Golomb ruler inspired approach.
    # Instead of a single hub (vertex 0) with chords, we use a Hamiltonian cycle
    # and add chords to cover all cycle lengths 3..39.
    #
    # The current seed uses a "fan" from vertex 0 with chords (0,2)..(0,20).
    # This creates cycles of length d+1 and 41-d for each chord d.
    # Total edges: 40 (Hamiltonian) + 19 (chords) = 59.
    #
    # We attempt to find a set of chords (not necessarily all from vertex 0)
    # such that:
    # 1. Every cycle length 3..39 can be formed.
    # 2. The number of chords is minimized.
    #
    # Key insight: A chord (u, v) on the Hamiltonian cycle creates two cycles
    # of lengths (v-u+1) and (N-(v-u)+1) (assuming u < v).
    # We need to cover all lengths 3..39.
    #
    # We use a greedy approach to select chords that cover the most uncovered
    # lengths, then use local search to try to reduce the count further.
    # ------------------------------------------------------------------

    # Base Hamiltonian cycle
    ham_edges = [(i, (i + 1) % N) for i in range(N)]
    
    # All possible chords (excluding Hamiltonian edges)
    # Chord (u, v) with u < v covers lengths (v-u+1) and (N-(v-u)+1)
    # We only need chords where the shorter arc length is between 2 and 19
    # (since length 1 is a self-loop, not a chord, and length 20 covers 20 and 21)
    
    all_chords = []
    for u in range(N):
        for v in range(u + 2, N):  # Skip adjacent vertices (u, u+1)
            # Also skip the wrap-around edge (39, 0) which is part of Hamiltonian cycle
            if u == 0 and v == N - 1:
                continue
            all_chords.append((u, v))

    # For each chord, determine which cycle lengths it covers
    chord_coverage = {}
    for chord in all_chords:
        u, v = chord
        d = v - u  # Distance along the cycle
        # Cycle lengths: d+1 and N-d+1
        l1 = d + 1
        l2 = N - d + 1
        # Only consider lengths 3..39
        lengths = set()
        if 3 <= l1 <= 39:
            lengths.add(l1)
        if 3 <= l2 <= 39:
            lengths.add(l2)
        chord_coverage[chord] = lengths

    # Greedy selection of chords to cover all lengths 3..39
    needed_lengths = set(range(3, 40))
    selected_chords = []
    covered = set()

    while len(covered) < len(needed_lengths):
        if time.time() > deadline:
            break
        best_chord = None
        best_new_coverage = -1
        for chord in all_chords:
            if chord in [c for c in selected_chords]:
                continue
            new_cov = chord_coverage[chord] - covered
            if len(new_cov) > best_new_coverage:
                best_new_coverage = len(new_cov)
                best_chord = chord
        
        if best_chord is None:
            break
        
        selected_chords.append(best_chord)
        covered.update(chord_coverage[best_chord])

    # Build edges
    edges = ham_edges + selected_chords
    
    # Build cycles for validation
    def build_cycles_from_chords(edges):
        adj = _build_adj(edges)
        cycles = []
        for L in range(3, N + 1):
            if L == N:
                # Hamiltonian cycle
                cycles.append(list(range(N)))
            else:
                # Find a cycle of length L
                # Try using the chords
                found = False
                for chord in selected_chords:
                    u, v = chord
                    d = v - u
                    l1 = d + 1
                    l2 = N - d + 1
                    if L == l1:
                        # Cycle: u, u+1, ..., v, u
                        cycle_verts = list(range(u, v + 1))
                        if len(cycle_verts) == L and len(set(cycle_verts)) == L:
                            cycles.append(cycle_verts)
                            found = True
                            break
                    elif L == l2:
                        # Cycle: v, v+1, ..., N-1, 0, ..., u, v
                        cycle_verts = list(range(v, N)) + list(range(0, u + 1))
                        if len(cycle_verts) == L and len(set(cycle_verts)) == L:
                            cycles.append(cycle_verts)
                            found = True
                            break
                if not found:
                    # Fallback: search for cycle
                    cycle = _find_cycle(adj, L)
                    if cycle is None:
                        return None
                    cycles.append(cycle)
        return cycles

    cycles = build_cycles_from_chords(edges)
    
    if cycles is None or not _validate(edges, cycles):
        # Fallback to seed construction
        chord_edges = [(0, d) for d in range(2, 21)]
        edges = ham_edges + chord_edges
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
        cycles = make_base_cycles()
    
    # Local search: Try to remove chords if possible
    best_edges = edges
    best_cycles = cycles
    best_edge_count = len(edges)

    for chord_idx in range(len(selected_chords)):
        if time.time() > deadline:
            break

        chord = selected_chords[chord_idx]
        trial_edges = [e for i, e in enumerate(edges) if i != N + chord_idx]
        
        # Determine which lengths are at risk
        lengths_at_risk = chord_coverage[chord]
        
        trial_adj = _build_adj(trial_edges)
        valid = True
        trial_cycles = []

        for L in range(3, N + 1):
            if time.time() > deadline:
                valid = False
                break

            if L not in lengths_at_risk:
                # Use existing cycle if it doesn't use the removed chord
                # Simplified: just try to find a cycle
                trial_cycles.append(None)  # Placeholder
            else:
                # Need to find alternative cycle
                alt_cycle = _find_cycle(trial_adj, L, forbidden_edge=chord)
                if alt_cycle is None:
                    valid = False
                    break
                trial_cycles[L-3] = alt_cycle
        
        if valid:
            # Rebuild all cycles properly
            for L in range(3, N + 1):
                if L == N:
                    trial_cycles[L-3] = list(range(N))
                else:
                    if L in lengths_at_risk:
                        # Already found
                        pass
                    else:
                        # Find cycle in trial graph
                        cycle = _find_cycle(trial_adj, L)
                        if cycle is None:
                            valid = False
                            break
                        trial_cycles[L-3] = cycle
            
            if valid and _validate(trial_edges, trial_cycles):
                if len(trial_edges) < best_edge_count:
                    best_edges = trial_edges
                    best_cycles = trial_cycles
                    best_edge_count = len(trial_edges)
                    break

    return {'edges': best_edges, 'cycles': best_cycles}
