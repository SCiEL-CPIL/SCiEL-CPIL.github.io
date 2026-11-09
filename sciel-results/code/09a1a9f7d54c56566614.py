import numpy as np
import time
from collections import deque

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)

    # Identify edge types
    c1_edges = []  # edges in first cycle (0..m-1)
    c2_edges = []  # edges in second cycle (m..2m-1)
    m_edges = []   # matching edges

    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            c1_edges.append(i)
        elif u >= m and v >= m:
            c2_edges.append(i)
        else:
            m_edges.append(i)

    # Precompute edge endpoints as arrays for fast indexing
    edge_u = np.array([u for u, v in edges], dtype=np.int32)
    edge_v = np.array([v for u, v in edges], dtype=np.int32)

    # Build adjacency structure for lift connectivity check
    # For each edge, precompute the lift adjacency pairs
    lift_adj_plus = []
    lift_adj_minus = []
    for i in range(num_edges):
        u = edge_u[i]
        v = edge_v[i]
        u0, u1 = u * 2, u * 2 + 1
        v0, v1 = v * 2, v * 2 + 1
        lift_adj_plus.append(((u0, v0), (v0, u0), (u1, v1), (v1, u1)))
        lift_adj_minus.append(((u0, v1), (v1, u0), (u1, v0), (v0, u1)))

    def is_connected_lift(signs):
        """Check if the 2-lift is connected using BFS"""
        lift_n = 2 * n
        visited = np.zeros(lift_n, dtype=bool)
        queue = deque([0])
        visited[0] = True
        count = 1

        while queue:
            u = queue.popleft()
            for i in range(num_edges):
                if edge_u[i] == u // 2 or edge_v[i] == u // 2:
                    # Check if this edge is incident to vertex u//2 in the base graph
                    # Determine the lift vertex
                    base_u = u // 2
                    lift_bit = u % 2

                    if edge_u[i] == base_u:
                        other_base = edge_v[i]
                    elif edge_v[i] == base_u:
                        other_base = edge_u[i]
                    else:
                        continue

                    sign = signs[i]
                    if sign == 1:
                        other_lift = other_base * 2 + lift_bit
                    else:
                        other_lift = other_base * 2 + (1 - lift_bit)

                    if not visited[other_lift]:
                        visited[other_lift] = True
                        count += 1
                        queue.append(other_lift)

        return count == lift_n

    def compute_spectral_radius(signs):
        """Compute spectral radius of signed adjacency matrix"""
        A = np.zeros((n, n), dtype=np.float64)
        A[edge_u, edge_v] = signs.astype(np.float64)
        A[edge_v, edge_u] = signs.astype(np.float64)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))

    def evaluate(signs):
        """Full evaluation: connectivity + spectral radius"""
        if not is_connected_lift(signs):
            return float('inf')
        return compute_spectral_radius(signs)

    # Get ordered cycle edges for pattern generation
    def get_ordered_cycle_edges(edge_indices, offset):
        pos_map = {}
        for idx in edge_indices:
            u, v = edges[idx]
            u_local = u - offset
            v_local = v - offset
            if (u_local + 1) % m == v_local:
                pos_map[u_local] = idx
            elif (v_local + 1) % m == u_local:
                pos_map[v_local] = idx
        return sorted(pos_map.items())

    c1_ordered = get_ordered_cycle_edges(c1_edges, 0)
    c2_ordered = get_ordered_cycle_edges(c2_edges, m)
    m_ordered = sorted(m_edges, key=lambda i: min(edges[i][0], edges[i][1]))

    # Generate initial candidates
    candidates = []

    # Pattern: alternating signs on cycles
    for c1_alt in [0, 1]:  # 0: start with +1, 1: start with -1
        for c2_alt in [0, 1]:
            for m_sign in [-1, 1]:
                s = np.ones(num_edges, dtype=np.int32)
                for pos, idx in c1_ordered:
                    s[idx] = 1 if (pos + c1_alt) % 2 == 0 else -1
                for pos, idx in c2_ordered:
                    s[idx] = 1 if (pos + c2_alt) % 2 == 0 else -1
                for idx in m_edges:
                    s[idx] = m_sign
                candidates.append(s.copy())

    # All same sign
    candidates.append(np.ones(num_edges, dtype=np.int32))
    candidates.append(-np.ones(num_edges, dtype=np.int32))

    # Mixed patterns
    for i in range(min(5, len(m_ordered))):
        s = np.ones(num_edges, dtype=np.int32)
        s[m_ordered[i]] = -1
        candidates.append(s.copy())

    # Evaluate all candidates
    best_radius = float('inf')
    best_signs = None

    for s in candidates:
        r = evaluate(s)
        if r < best_radius:
            best_radius = r
            best_signs = s.copy()

    if best_signs is None:
        # Fallback: try random
        rng = np.random.default_rng(3)
        for _ in range(100):
            s = rng.choice([-1, 1], size=num_edges).astype(np.int32)
            r = evaluate(s)
            if r < best_radius:
                best_radius = r
                best_signs = s.copy()

    if best_signs is None:
        return np.ones(num_edges, dtype=np.int32)

    # Simulated annealing
    current_signs = best_signs.copy()
    current_radius = best_radius

    # Precompute which edges are in which category for fast flip
    # Use a temperature schedule
    T_start = 1.0
    T_end = 0.01
    max_iterations = 50000

    # Time-based iteration limit
    time_per_iter = 0.0001  # target ~5s for SA
    max_iters_time = int(5.0 / time_per_iter)
    max_iters = min(max_iterations, max_iters_time)

    rng = np.random.default_rng(42)

    for it in range(max_iters):
        if time.time() > deadline - 10:
            break

        # Temperature schedule
        progress = it / max_iters
        T = T_start * (T_end / T_start) ** progress

        # Choose a random edge to flip
        i = rng.integers(0, num_edges)
        current_signs[i] *= -1

        new_radius = evaluate(current_signs)

        if new_radius < current_radius:
            current_radius = new_radius
            if new_radius < best_radius:
                best_radius = new_radius
                best_signs = current_signs.copy()
        else:
            # Metropolis criterion
            delta = new_radius - current_radius
            if delta < 0:
                # Accept (shouldn't happen since we check above)
                current_radius = new_radius
            else:
                # Accept with probability exp(-delta/T)
                if T > 0 and delta / T < 50:  # avoid underflow
                    prob = np.exp(-delta / T)
                    if rng.random() < prob:
                        current_radius = new_radius
                    else:
                        current_signs[i] *= -1  # revert
                else:
                    current_signs[i] *= -1  # revert

    # Final local search: single flips
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(num_edges):
            if time.time() > deadline - 5:
                break
            best_signs[i] *= -1
            r = evaluate(best_signs)
            if r < best_radius - 1e-10:
                best_radius = r
                improved = True
            else:
                best_signs[i] *= -1
        if not improved:
            break

    # Pairwise flips
    if time.time() < deadline - 3:
        for _ in range(50):
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(num_edges):
                if time.time() > deadline - 3:
                    break
                for j in range(i + 1, num_edges):
                    if time.time() > deadline - 3:
                        break
                    best_signs[i] *= -1
                    best_signs[j] *= -1
                    r = evaluate(best_signs)
                    if r < best_radius - 1e-10:
                        best_radius = r
                        improved = True
                        break
                    else:
                        best_signs[i] *= -1
                        best_signs[j] *= -1
                if improved:
                    break
            if not improved:
                break

    return best_signs
