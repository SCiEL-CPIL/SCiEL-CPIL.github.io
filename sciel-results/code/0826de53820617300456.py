"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def sign_edges(graph, budget):
    start_time = time.time()
    DEADLINE = 55  # Safety margin before 60s kill

    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m

    # Precompute edge structure
    edge_u = np.array([e[0] for e in edges], dtype=np.int32)
    edge_v = np.array([e[1] for e in edges], dtype=np.int32)

    def build_signed_adj(signs):
        """Build the 2m x 2m signed adjacency matrix."""
        A = np.zeros((n_vertices, n_vertices), dtype=np.float64)
        for idx in range(n_edges):
            s = signs[idx]
            u = edge_u[idx]
            v = edge_v[idx]
            A[u, v] = s
            A[v, u] = s
        return A

    def spectral_radius_exact(signs):
        """Compute exact spectral radius of signed adjacency matrix."""
        A = build_signed_adj(signs)
        eigvals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(eigvals)))

    def spectral_radius_power(signs, n_iter=50):
        """Fast power iteration estimate of spectral radius."""
        A = build_signed_adj(signs)
        x = np.random.default_rng(0).random(n_vertices)
        x /= np.linalg.norm(x)
        for _ in range(n_iter):
            x = A @ x
            norm = np.linalg.norm(x)
            if norm < 1e-15:
                return 0.0
            x /= norm
        return float(np.abs(x @ A @ x))

    def is_lift_connected_fast(signs):
        """Fast connectivity check for the 2-lift using BFS with arrays."""
        n = n_vertices
        total_nodes = 2 * n
        # Precompute neighbor lists as arrays for speed
        adj_u = np.zeros((n_edges * 4, 2), dtype=np.int32)
        for idx in range(n_edges):
            s = signs[idx]
            u = edge_u[idx]
            v = edge_v[idx]
            base = idx * 4
            if s == 1:
                adj_u[base, 0] = u * 2
                adj_u[base, 1] = v * 2
                adj_u[base + 1, 0] = v * 2
                adj_u[base + 1, 1] = u * 2
                adj_u[base + 2, 0] = u * 2 + 1
                adj_u[base + 2, 1] = v * 2 + 1
                adj_u[base + 3, 0] = v * 2 + 1
                adj_u[base + 3, 1] = u * 2 + 1
            else:
                adj_u[base, 0] = u * 2
                adj_u[base, 1] = v * 2 + 1
                adj_u[base + 1, 0] = v * 2 + 1
                adj_u[base + 1, 1] = u * 2
                adj_u[base + 2, 0] = u * 2 + 1
                adj_u[base + 2, 1] = v * 2
                adj_u[base + 3, 0] = v * 2
                adj_u[base + 3, 1] = u * 2 + 1

        # BFS from node 0
        visited = np.zeros(total_nodes, dtype=np.bool_)
        queue = deque([0])
        visited[0] = True
        count = 1
        while queue:
            node = queue.popleft()
            for j in range(n_edges * 4):
                if adj_u[j, 0] == node:
                    neighbor = adj_u[j, 1]
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        count += 1
                        queue.append(neighbor)
                elif adj_u[j, 1] == node:
                    neighbor = adj_u[j, 0]
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        count += 1
                        queue.append(neighbor)

        return count == total_nodes

    def evaluate(signs, use_power=True):
        """Check validity and compute radius. Returns (valid, radius)."""
        if not is_lift_connected_fast(signs):
            return False, float('inf')
        if use_power:
            radius = spectral_radius_power(signs)
        else:
            radius = spectral_radius_exact(signs)
        return True, radius

    # Initial random signs
    rng = np.random.default_rng(42)

    best_signs = None
    best_radius = float('inf')
    best_valid = False

    # Phase 1: Broad random search with power iteration screening
    n_random = 200
    for i in range(n_random):
        if time.time() - start_time > DEADLINE:
            break
        signs = rng.choice([-1, 1], size=n_edges).astype(np.float64)
        valid, radius = evaluate(signs, use_power=True)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            best_valid = True

    # Phase 2: Local search from best found
    if best_valid and best_signs is not None:
        signs = best_signs.copy()
        for flip_iter in range(200):
            if time.time() - start_time > DEADLINE:
                break

            improved = False
            for i in range(n_edges):
                if time.time() - start_time > DEADLINE:
                    break

                signs[i] = -signs[i]
                valid, new_radius = evaluate(signs, use_power=True)

                if valid and new_radius < best_radius - 1e-10:
                    best_radius = new_radius
                    best_signs = signs.copy()
                    best_valid = True
                    improved = True
                else:
                    signs[i] = -signs[i]

            if not improved:
                break

    # Phase 3: Try more random starts and local search
    for init_idx in range(30):
        if time.time() - start_time > DEADLINE:
            break

        signs = rng.choice([-1, 1], size=n_edges).astype(np.float64)
        valid, radius = evaluate(signs, use_power=True)
        if valid:
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
                best_valid = True

        # Quick local search from this start
        for flip_iter in range(50):
            if time.time() - start_time > DEADLINE:
                break

            improved = False
            for i in range(n_edges):
                if time.time() - start_time > DEADLINE:
                    break

                signs[i] = -signs[i]
                valid, new_radius = evaluate(signs, use_power=True)

                if valid and new_radius < best_radius - 1e-10:
                    best_radius = new_radius
                    best_signs = signs.copy()
                    best_valid = True
                    improved = True
                else:
                    signs[i] = -signs[i]

            if not improved:
                break

    # Phase 4: Final polish with exact eigenvalues
    if best_valid and best_signs is not None:
        signs = best_signs.copy()
        # Get exact radius of current best
        best_radius_exact = spectral_radius_exact(signs)
        best_radius = best_radius_exact

        for polish_iter in range(30):
            if time.time() - start_time > DEADLINE:
                break

            improved = False
            for i in range(n_edges):
                if time.time() - start_time > DEADLINE:
                    break

                signs[i] = -signs[i]
                valid, _ = evaluate(signs, use_power=False)
                if valid:
                    new_radius_exact = spectral_radius_exact(signs)
                    if new_radius_exact < best_radius - 1e-12:
                        best_radius = new_radius_exact
                        best_signs = signs.copy()
                        improved = True
                    else:
                        signs[i] = -signs[i]
                else:
                    signs[i] = -signs[i]

            if not improved:
                break

    # Return as int array
    if best_signs is None:
        # Fallback: return all +1
        return np.ones(n_edges, dtype=int)

    return best_signs.astype(int)
# EVOLVE-BLOCK-END
