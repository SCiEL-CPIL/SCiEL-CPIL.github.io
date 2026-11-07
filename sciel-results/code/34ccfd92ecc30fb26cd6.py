"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def sign_edges(graph, budget):
    start_time = time.time()
    DEADLINE = 85  # Safety margin before 90s kill

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

    def spectral_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        A = build_signed_adj(signs)
        eigvals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(eigvals)))

    def is_lift_connected_fast(signs):
        """Fast connectivity check for the 2-lift using BFS."""
        n = n_vertices
        # Build adjacency of the lift graph (2n vertices)
        adj = [[] for _ in range(2 * n)]
        for idx in range(n_edges):
            s = signs[idx]
            u = edge_u[idx]
            v = edge_v[idx]
            if s == 1:
                adj[u * 2].append(v * 2)
                adj[v * 2].append(u * 2)
                adj[u * 2 + 1].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2 + 1)
            else:
                adj[u * 2].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2)
                adj[u * 2 + 1].append(v * 2)
                adj[v * 2].append(u * 2 + 1)

        # BFS from node 0
        visited = [False] * (2 * n)
        queue = deque([0])
        visited[0] = True
        count = 1
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    count += 1
                    queue.append(neighbor)

        return count == 2 * n

    def evaluate(signs):
        """Check validity and compute radius. Returns (valid, radius)."""
        if not is_lift_connected_fast(signs):
            return False, float('inf')
        radius = spectral_radius(signs)
        return True, radius

    # Initial random signs
    rng = np.random.default_rng(42)

    best_signs = None
    best_radius = float('inf')
    best_valid = False

    def try_signs(signs):
        """Try a sign assignment and update best if valid and better."""
        nonlocal best_signs, best_radius, best_valid
        valid, radius = evaluate(signs)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            best_valid = True
            return True
        return False

    # Structured initializations exploiting prism graph structure
    # Prism graph: two m-cycles + matching
    # Edge types: cycle1 edges (m), cycle2 edges (m), matching edges (m)
    
    # Generate structured patterns
    def generate_structured_patterns():
        patterns = []
        
        # Pattern 1: All +1 (trivially valid, radius = 3)
        patterns.append(np.ones(n_edges, dtype=np.float64))
        
        # Pattern 2: Alternating on cycles, +1 on matching
        for phase in [0, 1]:
            signs = np.ones(n_edges, dtype=np.float64)
            # Cycle 1: edges 0..m-1
            for i in range(m):
                signs[i] = 1 if (i % 2 == phase) else -1
            # Cycle 2: edges m..2m-1
            for i in range(m):
                signs[m + i] = 1 if (i % 2 == phase) else -1
            # Matching: edges 2m..3m-1
            # signs[2*m:3*m] stays +1
            patterns.append(signs.copy())
        
        # Pattern 3: One cycle all +1, other alternating
        for phase in [0, 1]:
            signs = np.ones(n_edges, dtype=np.float64)
            # Cycle 1: all +1
            # Cycle 2: alternating
            for i in range(m):
                signs[m + i] = 1 if (i % 2 == phase) else -1
            patterns.append(signs.copy())
        
        # Pattern 4: Matching has mixed signs
        for match_pattern in range(4):
            signs = np.ones(n_edges, dtype=np.float64)
            for i in range(m):
                signs[2*m + i] = 1 if (i % 2 == match_pattern % 2) else -1
            patterns.append(signs.copy())
        
        # Pattern 5: Random but with structure - cycles consistent
        for _ in range(20):
            signs = np.ones(n_edges, dtype=np.float64)
            # Randomly flip some cycle edges
            for i in range(m):
                if rng.random() < 0.3:
                    signs[i] = -1
            for i in range(m):
                if rng.random() < 0.3:
                    signs[m + i] = -1
            # Matching: mostly +1
            for i in range(m):
                if rng.random() < 0.2:
                    signs[2*m + i] = -1
            patterns.append(signs.copy())
        
        # Pattern 6: Fully random
        for _ in range(30):
            signs = rng.choice([-1, 1], size=n_edges).astype(np.float64)
            patterns.append(signs)
        
        return patterns

    # Try all structured patterns
    for signs in generate_structured_patterns():
        if time.time() - start_time > DEADLINE:
            break
        try_signs(signs)

    # Local search from best found
    if best_valid and best_signs is not None:
        signs = best_signs.copy()
        
        for flip_iter in range(200):
            if time.time() - start_time > DEADLINE:
                break

            improved = False
            for i in range(n_edges):
                if time.time() - start_time > DEADLINE:
                    break

                # Flip edge i
                signs[i] = -signs[i]

                # Check validity and compute radius
                valid, new_radius = evaluate(signs)

                if valid and new_radius < best_radius - 1e-12:
                    best_radius = new_radius
                    best_signs = signs.copy()
                    best_valid = True
                    improved = True
                else:
                    # Revert
                    signs[i] = -signs[i]

            if not improved:
                break

    # If no valid lift found yet, try more random
    if not best_valid:
        for pattern_idx in range(100):
            if time.time() - start_time > DEADLINE:
                break

            signs = rng.choice([-1, 1], size=n_edges).astype(np.float64)
            try_signs(signs)

    # Final polish: more iterations with tighter tolerance
    if best_valid and best_signs is not None:
        signs = best_signs.copy()
        for polish_iter in range(100):
            if time.time() - start_time > DEADLINE:
                break

            improved = False
            for i in range(n_edges):
                if time.time() - start_time > DEADLINE:
                    break

                signs[i] = -signs[i]
                valid, new_radius = evaluate(signs)

                if valid and new_radius < best_radius - 1e-14:
                    best_radius = new_radius
                    best_signs = signs.copy()
                    improved = True
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
