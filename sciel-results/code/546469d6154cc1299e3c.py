import numpy as np
import time
from collections import deque

def _build_base_matrix(n, edges):
    """Build the unsigned adjacency structure (unit weights)."""
    A = np.zeros((n, n), dtype=float)
    for (u, v) in edges:
        A[u, v] = 1.0
        A[v, u] = 1.0
    return A

def _spectral_radius_fast(signs, n, edges, base_A, chord_indices, chord_positions):
    """
    Compute spectral radius efficiently by updating only chord entries.
    signs: array of +1/-1 for all edges
    base_A: the base matrix with all 1s
    chord_indices: list of edge indices that are chords
    chord_positions: for each chord, (u, v) tuple
    """
    A = base_A.copy()
    # Set all edges to +1 first (base_A already has 1s)
    # Now apply signs: for edges with sign -1, negate
    # For tree edges, sign is always +1 in our parameterization
    # For chords, sign depends on the mask
    for k, edge_idx in enumerate(chord_indices):
        if signs[edge_idx] == -1:
            u, v = chord_positions[k]
            A[u, v] = -1.0
            A[v, u] = -1.0
    eigvals = np.linalg.eigvalsh(A)
    return float(np.max(np.abs(eigvals)))

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)

    # Build base matrix
    base_A = _build_base_matrix(n, edges)

    # For prism graphs, the structure is:
    # - Top cycle: vertices 0..m-1, edges (i, (i+1)%m)
    # - Bottom cycle: vertices m..2m-1, edges (m+i, m+(i+1)%m)
    # - Matching: edges (i, m+i) for i in 0..m-1
    #
    # Total edges: 3m
    #
    # The 2-lift is connected iff the product of signs around any fundamental
    # cycle that goes through the matching is -1 (i.e., the lift doesn't split
    # into two disconnected components).
    #
    # For the prism, we can parameterize signs by:
    # - s_top[i]: sign of top cycle edge (i, (i+1)%m)
    # - s_bot[i]: sign of bottom cycle edge (m+i, m+(i+1)%m)
    # - s_match[i]: sign of matching edge (i, m+i)
    #
    # The lift is connected iff the product of signs around the "ladder" cycle
    # (going around top, down matching, around bottom backwards, up matching)
    # is -1. More precisely, for the lift to be connected, we need:
    # prod(s_top) * prod(s_bot) * prod(s_match)^2 != +1 in a way that disconnects.
    #
    # Actually, for a 2-lift of a connected graph, it's connected iff there exists
    # a cycle in the base graph where the product of signs is -1.
    #
    # For the prism, we can enumerate all 2^(3m) sign assignments, but that's too many.
    # Instead, we can use the fact that the spectral radius depends on the pattern
    # of signs, and we can use a smarter search.

    # Identify edge types
    top_edges = []
    bot_edges = []
    match_edges = []
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_edges.append(i)
        elif u >= m and v >= m:
            bot_edges.append(i)
        else:
            match_edges.append(i)

    num_top = len(top_edges)
    num_bot = len(bot_edges)
    num_match = len(match_edges)

    # For m=4,5,7,9: num_top=num_bot=num_match=m, total=3m
    # Total sign assignments: 2^(3m)
    # For m=4: 2^12=4096, m=5: 2^15=32768, m=7: 2^21=2M, m=9: 2^27=134M
    #
    # We need a smarter approach. Let's use the fact that the spectral radius
    # of the signed adjacency matrix can be computed efficiently, and use
    # a combination of exhaustive search for small m and heuristic search for larger m.

    # Strategy:
    # 1. For small m (m<=5), do exhaustive search over all 2^(3m) sign assignments
    # 2. For larger m, use a heuristic: try to make the signed graph as "anti-bipartite"
    #    as possible, which tends to minimize spectral radius.

    best_radius = float('inf')
    best_signs = None

    # Helper: check if lift is connected
    def is_lift_connected(signs):
        parent = list(range(2 * n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry
        
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                union(2*u, 2*v)
                union(2*u+1, 2*v+1)
            else:
                union(2*u, 2*v+1)
                union(2*u+1, 2*v)
        
        root = find(0)
        return all(find(v) == root for v in range(2 * n))

    # Helper: compute spectral radius from signs
    def compute_radius(signs):
        A = base_A.copy()
        for i, (u, v) in enumerate(edges):
            if signs[i] == -1:
                A[u, v] = -1.0
                A[v, u] = -1.0
        eigvals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(eigvals)))

    # Phase 1: Try structured patterns
    # Pattern A: All matching edges -1, cycles +1
    # This makes the lift a "twisted" prism, which is connected.
    signs = np.ones(num_edges, dtype=int)
    for idx in match_edges:
        signs[idx] = -1
    if is_lift_connected(signs):
        r = compute_radius(signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()

    # Pattern B: Alternating signs on matching
    for phase in range(2):
        signs = np.ones(num_edges, dtype=int)
        for j, idx in enumerate(match_edges):
            if (j + phase) % 2 == 0:
                signs[idx] = -1
        if is_lift_connected(signs):
            r = compute_radius(signs)
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()

    # Pattern C: Make top and bottom cycles have alternating signs
    for top_phase in range(2):
        for bot_phase in range(2):
            for match_phase in range(2):
                signs = np.ones(num_edges, dtype=int)
                for j, idx in enumerate(top_edges):
                    if (j + top_phase) % 2 == 0:
                        signs[idx] = -1
                for j, idx in enumerate(bot_edges):
                    if (j + bot_phase) % 2 == 0:
                        signs[idx] = -1
                for j, idx in enumerate(match_edges):
                    if (j + match_phase) % 2 == 0:
                        signs[idx] = -1
                if is_lift_connected(signs):
                    r = compute_radius(signs)
                    if r < best_radius:
                        best_radius = r
                        best_signs = signs.copy()

    # Phase 2: Local search from best solution
    if best_signs is not None:
        current_signs = best_signs.copy()
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(num_edges):
                if time.time() > deadline:
                    break
                # Try flipping sign i
                trial = current_signs.copy()
                trial[i] = -trial[i]
                if is_lift_connected(trial):
                    r = compute_radius(trial)
                    if r < best_radius:
                        best_radius = r
                        current_signs = trial
                        best_signs = trial.copy()
                        improved = True

    # Phase 3: Random search with connectivity check
    rng = np.random.default_rng(42)
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        signs = rng.choice([-1, 1], size=num_edges).astype(int)
        if is_lift_connected(signs):
            r = compute_radius(signs)
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
        if attempts > 200000:
            break

    # Final: if no valid solution found, use a known valid pattern
    if best_signs is None:
        # All matching edges -1 is always valid for prism graphs
        best_signs = np.ones(num_edges, dtype=int)
        for idx in match_edges:
            best_signs[idx] = -1

    return best_signs
