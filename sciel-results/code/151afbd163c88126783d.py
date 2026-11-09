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

def _is_lift_connected(signs, n, edges):
    """Check if the 2-lift is connected using union-find."""
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

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)

    # Build base matrix
    base_A = _build_base_matrix(n, edges)

    # Find spanning tree and identify chords
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))

    visited = [False] * n
    tree_edge_set = set()
    visited[0] = True
    dq = deque([0])
    while dq:
        u = dq.popleft()
        for v, idx in adj[u]:
            if not visited[v]:
                visited[v] = True
                tree_edge_set.add(idx)
                dq.append(v)

    chords = [i for i in range(num_edges) if i not in tree_edge_set]
    num_chords = len(chords)
    chord_positions = [edges[i] for i in chords]

    # Precompute chord positions for fast access
    chord_u = np.array([edges[i][0] for i in chords], dtype=int)
    chord_v = np.array([edges[i][1] for i in chords], dtype=int)

    best_radius = float('inf')
    best_mask = 0
    best_signs = None

    # Phase 1: Exhaustive search over all 2^num_chords chord sign assignments
    # (tree edges fixed at +1, which is a valid parameterization)
    total_masks = 1 << num_chords
    for mask in range(1, total_masks):
        if time.time() > deadline:
            break
        
        # Build signs array
        signs = np.ones(num_edges, dtype=int)
        for k in range(num_chords):
            if mask & (1 << k):
                signs[chords[k]] = -1
        
        r = _spectral_radius_fast(signs, n, edges, base_A, chords, chord_positions)
        
        if r < best_radius:
            best_radius = r
            best_mask = mask
            best_signs = signs.copy()

    # Phase 2: Local search - flip individual chord signs to see if we can improve
    # Start from best_mask and try flipping each chord
    if best_mask > 0:
        current_mask = best_mask
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for k in range(num_chords):
                if time.time() > deadline:
                    break
                # Try flipping chord k
                trial_mask = current_mask ^ (1 << k)
                if trial_mask == 0:
                    continue  # Would be disconnected
                signs = np.ones(num_edges, dtype=int)
                for j in range(num_chords):
                    if trial_mask & (1 << j):
                        signs[chords[j]] = -1
                r = _spectral_radius_fast(signs, n, edges, base_A, chords, chord_positions)
                if r < best_radius - 1e-12:
                    best_radius = r
                    current_mask = trial_mask
                    best_mask = trial_mask
                    best_signs = signs.copy()
                    improved = True

    # Phase 2b: Double-flip local search
    if time.time() < deadline and num_chords >= 2:
        current_mask = best_mask
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for k in range(num_chords):
                if time.time() > deadline:
                    break
                for l in range(k + 1, num_chords):
                    if time.time() > deadline:
                        break
                    # Try flipping chords k and l
                    trial_mask = current_mask ^ (1 << k) ^ (1 << l)
                    if trial_mask == 0:
                        continue
                    signs = np.ones(num_edges, dtype=int)
                    for j in range(num_chords):
                        if trial_mask & (1 << j):
                            signs[chords[j]] = -1
                    r = _spectral_radius_fast(signs, n, edges, base_A, chords, chord_positions)
                    if r < best_radius - 1e-12:
                        best_radius = r
                        current_mask = trial_mask
                        best_mask = trial_mask
                        best_signs = signs.copy()
                        improved = True

    # Phase 3: Random restarts - try random sign assignments (not just tree+chord param)
    # This explores the full space of 2^num_edges sign assignments
    rng = np.random.default_rng(42)
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Random sign assignment
        signs = rng.choice([-1, 1], size=num_edges).astype(int)
        
        # Quick connectivity check using union-find on the lift
        if not _is_lift_connected(signs, n, edges):
            continue
            
        r = _spectral_radius_fast(signs, n, edges, base_A, chords, chord_positions)
        if r < best_radius - 1e-12:
            best_radius = r
            best_signs = signs.copy()
        
        if attempts > 100000:
            break

    # Phase 4: Final polish - more aggressive local search from best_signs
    if best_signs is not None and time.time() < deadline:
        current_signs = best_signs.copy()
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(num_edges):
                if time.time() > deadline:
                    break
                # Try flipping edge i
                trial_signs = current_signs.copy()
                trial_signs[i] = -trial_signs[i]
                
                if not _is_lift_connected(trial_signs, n, edges):
                    continue
                    
                r = _spectral_radius_fast(trial_signs, n, edges, base_A, chords, chord_positions)
                if r < best_radius - 1e-12:
                    best_radius = r
                    current_signs = trial_signs
                    best_signs = trial_signs.copy()
                    improved = True

    # If we found a solution, use it
    if best_signs is not None:
        return best_signs

    # Fallback: single -1 on first chord
    signs = np.ones(num_edges, dtype=int)
    if chords:
        signs[chords[0]] = -1

    return signs
