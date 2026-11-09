import numpy as np
import time
from collections import deque


def _lift_connected_and_radius(signs, n, edges, m):
    """Return spectral radius if lift is connected 3-regular, else None."""
    N = 2 * n
    adj = [[] for _ in range(N)]
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        if s == 1:
            pairs = [(2*u, 2*v), (2*u+1, 2*v+1)]
        else:
            pairs = [(2*u, 2*v+1), (2*u+1, 2*v)]
        for a, b in pairs:
            adj[a].append(b)
            adj[b].append(a)
    seen = [False] * N
    seen[0] = True
    stack = [0]
    count = 1
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if not seen[y]:
                seen[y] = True
                count += 1
                stack.append(y)
    if count != N:
        return None
    A = np.zeros((n, n), dtype=float)
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        A[u, v] = s
        A[v, u] = s
    eigvals = np.linalg.eigvalsh(A)
    return float(np.max(np.abs(eigvals)))


def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)

    # Build adjacency
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))

    # Find spanning tree and chords
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

    best_radius = float('inf')
    best_signs = None

    # Guaranteed valid baseline
    base_signs = np.ones(num_edges, dtype=int)
    base_signs[chords[0]] = -1
    r = _lift_connected_and_radius(base_signs, n, edges, m)
    if r is not None:
        best_radius = r
        best_signs = base_signs.copy()

    # Phase-based gradient descent with 2-opt local search
    def _smooth_radius(phases):
        """Spectral radius of matrix with weights cos(phase_u - phase_v)."""
        A = np.zeros((n, n), dtype=float)
        for i, (u, v) in enumerate(edges):
            w = np.cos(phases[u] - phases[v])
            A[u, v] = w
            A[v, u] = w
        eigvals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(eigvals)))

    def _gradient(phases):
        """Gradient of smooth_radius w.r.t. phases using finite differences."""
        grad = np.zeros(n, dtype=float)
        eps = 1e-4
        for i in range(n):
            phases[i] += eps
            val_plus = _smooth_radius(phases)
            phases[i] -= 2 * eps
            val_minus = _smooth_radius(phases)
            phases[i] += eps
            grad[i] = (val_plus - val_minus) / (2 * eps)
        return grad

    def _phases_to_signs(phases):
        """Convert phases to signs."""
        signs = np.ones(num_edges, dtype=int)
        for i, (u, v) in enumerate(edges):
            val = np.cos(phases[u] - phases[v])
            if val < 0:
                signs[i] = -1
        return signs

    def _two_opt_local_search(signs):
        """Perform 2-opt local search: try flipping pairs of edges."""
        if time.time() > deadline:
            return signs, _lift_connected_and_radius(signs, n, edges, m)
        
        current_signs = signs.copy()
        current_r = _lift_connected_and_radius(current_signs, n, edges, m)
        if current_r is None:
            return signs, None
        
        improved = True
        max_rounds = 50
        for round_num in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            for i in range(num_edges):
                if time.time() > deadline:
                    break
                for j in range(i + 1, num_edges):
                    if time.time() > deadline:
                        break
                    # Try flipping edges i and j
                    current_signs[i] *= -1
                    current_signs[j] *= -1
                    r = _lift_connected_and_radius(current_signs, n, edges, m)
                    if r is not None and r < current_r - 1e-10:
                        current_r = r
                        improved = True
                    else:
                        # Revert
                        current_signs[i] *= -1
                        current_signs[j] *= -1
            if not improved:
                break
        
        return current_signs, current_r

    # Run gradient descent with multiple restarts
    rng = np.random.default_rng(42)
    num_restarts = 15
    max_iters = 300

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Initialize phases randomly
        phases = rng.uniform(0, 2 * np.pi, size=n)

        # Gradient descent with adaptive step size
        step_size = 0.1
        prev_radius = _smooth_radius(phases)

        for it in range(max_iters):
            if time.time() > deadline:
                break

            grad = _gradient(phases)

            # Update phases
            phases -= step_size * grad
            phases = phases % (2 * np.pi)

            new_radius = _smooth_radius(phases)

            if new_radius < prev_radius:
                prev_radius = new_radius
                step_size *= 1.05
            else:
                step_size *= 0.5
                phases = phases + step_size * grad
                phases = phases % (2 * np.pi)

            if step_size < 1e-6:
                break

        # Convert phases to signs and check validity
        signs = _phases_to_signs(phases)
        r = _lift_connected_and_radius(signs, n, edges, m)
        if r is not None:
            # Apply 2-opt local search
            signs, r = _two_opt_local_search(signs)
            if r is not None and r < best_radius:
                best_radius = r
                best_signs = signs.copy()

    # Also try exhaustive search over chords if we have time
    if time.time() < deadline - 5 and num_chords <= 12:
        for mask in range(1 << num_chords):
            if time.time() > deadline:
                break
            signs = np.ones(num_edges, dtype=int)
            for k, edge_idx in enumerate(chords):
                if mask & (1 << k):
                    signs[edge_idx] = -1
            r = _lift_connected_and_radius(signs, n, edges, m)
            if r is None:
                continue
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()

    # Safety: ensure we always return something valid
    if best_signs is None:
        best_signs = base_signs.copy()

    return best_signs
