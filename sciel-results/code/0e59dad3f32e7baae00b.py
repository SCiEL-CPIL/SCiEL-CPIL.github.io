import numpy as np
import time
from collections import deque

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 50  # Safety margin

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)

    # Identify edge types
    c1_edges = []
    c2_edges = []
    m_edges = []

    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            c1_edges.append(i)
        elif u >= m and v >= m:
            c2_edges.append(i)
        else:
            m_edges.append(i)

    # Precompute edge endpoints
    edge_u = np.array([u for u, v in edges], dtype=int)
    edge_v = np.array([v for u, v in edges], dtype=int)

    # Precompute lift adjacency contributions
    # lift_nodes = 2 * n
    # For each edge i, store the 4 pairs of lift nodes for sign +1 and sign -1
    lift_adj_plus = [None] * num_edges
    lift_adj_minus = [None] * num_edges
    
    for i in range(num_edges):
        u = edge_u[i]
        v = edge_v[i]
        u0, u1 = u * 2, u * 2 + 1
        v0, v1 = v * 2, v * 2 + 1
        
        # Sign +1: (u,0)-(v,0) and (u,1)-(v,1)
        lift_adj_plus[i] = [(u0, v0), (v0, u0), (u1, v1), (v1, u1)]
        
        # Sign -1: (u,0)-(v,1) and (u,1)-(v,0)
        lift_adj_minus[i] = [(u0, v1), (v1, u0), (u1, v0), (v0, u1)]

    def evaluate_signs(signs):
        # signs is a numpy array of 1s and -1s
        # 1. Check connectivity of the lift
        lift_n = 2 * n
        # Instead of building full adjacency list, we can do BFS directly using the precomputed edges
        # But we need to know which edges are "active" based on signs.
        # Actually, all edges are always present in the lift, just the connections differ.
        # So we need to traverse the lift graph defined by the current signs.
        
        # To make BFS fast, let's build the adjacency list once per evaluation?
        # Or, we can precompute the "structure" and just switch neighbors?
        # The neighbors of a node in the lift depend on the signs of incident base edges.
        # This is complex to precompute dynamically.
        # Given n <= 18, lift_n <= 36. Building adj list is O(E_lift) = O(4 * E_base) = O(12 * 3m) ~ O(360) ops.
        # This is very fast.
        
        adj_lift = [[] for _ in range(lift_n)]
        for i in range(num_edges):
            if signs[i] == 1:
                for u_l, v_l in lift_adj_plus[i]:
                    adj_lift[u_l].append(v_l)
            else:
                for u_l, v_l in lift_adj_minus[i]:
                    adj_lift[u_l].append(v_l)
        
        visited = [False] * lift_n
        queue = deque([0])
        visited[0] = True
        while queue:
            u = queue.popleft()
            for v in adj_lift[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        
        if not all(visited):
            return float('inf')
            
        # 2. Compute spectral radius
        A = np.zeros((n, n), dtype=float)
        # Vectorized fill
        A[edge_u, edge_v] = signs
        A[edge_v, edge_u] = signs
        
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))

    best_radius = float('inf')
    best_signs = None

    def try_signs(signs):
        nonlocal best_radius, best_signs
        r = evaluate_signs(signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()

    # Helper to get cycle edge order
    def get_cycle_edges_indices(edge_indices, offset=0):
        pos_map = {}
        for idx in edge_indices:
            u, v = edges[idx]
            u = u - offset
            v = v - offset
            if (u + 1) % m == v:
                pos = u
            elif (v + 1) % m == u:
                pos = v
            else:
                continue
            pos_map[pos] = idx
        return sorted(pos_map.items())

    c1_ordered = get_cycle_edges_indices(c1_edges, 0)
    c2_ordered = get_cycle_edges_indices(c2_edges, m)
    m_ordered = sorted(m_edges, key=lambda i: edges[i][0] if edges[i][0] < m else edges[i][1])

    # Generate initial candidates
    candidates = []
    
    # Pattern 1: Alt C1, Alt C2, -1 M
    s1 = np.ones(num_edges, dtype=int)
    for pos, idx in c1_ordered: s1[idx] = 1 if pos % 2 == 0 else -1
    for pos, idx in c2_ordered: s1[idx] = 1 if pos % 2 == 0 else -1
    for idx in m_edges: s1[idx] = -1
    candidates.append(s1)
    
    # Pattern 2: Alt C1, Anti-Alt C2, -1 M
    s2 = s1.copy()
    for pos, idx in c2_ordered: s2[idx] = -1 if pos % 2 == 0 else 1
    candidates.append(s2)
    
    # Pattern 3: Alt C1, Alt C2, +1 M
    s3 = s1.copy()
    for idx in m_edges: s3[idx] = 1
    candidates.append(s3)
    
    # Pattern 4: All +1 C1, All +1 C2, Alt M
    s4 = np.ones(num_edges, dtype=int)
    for pos, idx in enumerate(m_ordered): s4[idx] = 1 if pos % 2 == 0 else -1
    candidates.append(s4)
    
    # Pattern 5: Alt C1, Alt C2, Alt M
    s5 = s1.copy()
    for pos, idx in enumerate(m_ordered): s5[idx] = 1 if pos % 2 == 0 else -1
    candidates.append(s5)
    
    # Evaluate initial candidates
    for s in candidates:
        try_signs(s)

    # Local Search
    current_signs = best_signs.copy()
    current_radius = best_radius

    # Phase 1: Single Flip Hill Climb
    max_passes = 20
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(num_edges):
            if time.time() > deadline:
                break
            current_signs[i] *= -1
            r = evaluate_signs(current_signs)
            if r < current_radius - 1e-9:
                current_radius = r
                best_radius = r
                best_signs = current_signs.copy()
                improved = True
            else:
                current_signs[i] *= -1
        if not improved:
            break

    # Phase 2: Pairwise Flip Search
    # Try flipping pairs of edges (i, j)
    # To avoid redundant checks, we can iterate i < j
    # We restart the pairwise search from the current best
    current_signs = best_signs.copy()
    current_radius = best_radius
    
    # Limit pairwise search to a few passes
    max_pair_passes = 5
    for _ in range(max_pair_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(num_edges):
            if time.time() > deadline:
                break
            for j in range(i + 1, num_edges):
                if time.time() > deadline:
                    break
                # Flip i and j
                current_signs[i] *= -1
                current_signs[j] *= -1
                r = evaluate_signs(current_signs)
                if r < current_radius - 1e-9:
                    current_radius = r
                    best_radius = r
                    best_signs = current_signs.copy()
                    improved = True
                    # Restart pairwise search from new best?
                    # Or just continue? Continuing might miss other pairs from the new state.
                    # Let's break out to restart the pass from the new state for thoroughness
                    # But that might be slow. Let's just revert and continue if not improved.
                    # If improved, we keep the flip and continue the inner loop?
                    # No, flipping back j and i is hard if we moved on.
                    # Standard 2-opt: if improvement, accept and restart the search (or at least the current pass).
                    # Let's accept and break out to restart the pass.
                    current_signs = best_signs.copy()
                    break
                else:
                    # Revert
                    current_signs[i] *= -1
                    current_signs[j] *= -1
            if improved:
                break
        if not improved:
            break

    # Phase 3: Final Single Flip Polish
    current_signs = best_signs.copy()
    current_radius = best_radius
    for _ in range(10):
        if time.time() > deadline:
            break
        improved = False
        for i in range(num_edges):
            if time.time() > deadline:
                break
            current_signs[i] *= -1
            r = evaluate_signs(current_signs)
            if r < current_radius - 1e-9:
                current_radius = r
                best_radius = r
                best_signs = current_signs.copy()
                improved = True
            else:
                current_signs[i] *= -1
        if not improved:
            break

    return best_signs
