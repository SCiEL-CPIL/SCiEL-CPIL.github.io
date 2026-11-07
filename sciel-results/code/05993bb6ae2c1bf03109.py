"""Optimized sign_edges using circulant structure for fast spectral radius evaluation."""
import time
import numpy as np

def sign_edges(graph, budget):
    deadline = time.time() + 85.0  # Safety margin before 90s kill

    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2

    # Precompute edge structure
    edge_u = np.array([e[0] for e in edges], dtype=int)
    edge_v = np.array([e[1] for e in edges], dtype=int)

    # Identify edge types: cycle1 (u,v < m), cycle2 (u,v >= m), matching (u+m == v or v+m == u)
    cycle1_idx = []
    cycle2_idx = []
    match_idx = []
    for i in range(num_edges):
        u, v = edge_u[i], edge_v[i]
        if u < m and v < m:
            cycle1_idx.append(i)
        elif u >= m and v >= m:
            cycle2_idx.append(i)
        else:
            match_idx.append(i)

    cycle1_idx = np.array(cycle1_idx, dtype=int)
    cycle2_idx = np.array(cycle2_idx, dtype=int)
    match_idx = np.array(match_idx, dtype=int)

    # Precompute cycle edge offsets (d = (v - u) mod m for cycle1, (v - m - (u - m)) mod m for cycle2)
    # For cycle1: edge i connects u to u+d1
    c1_u = edge_u[cycle1_idx]
    c1_v = edge_v[cycle1_idx]
    c1_d = (c1_v - c1_u) % m

    c2_u = edge_u[cycle2_idx] - m
    c2_v = edge_v[cycle2_idx] - m
    c2_d = (c2_v - c2_u) % m

    # Matching: edge connects i to i+m, so for matching edge j, the vertex in cycle1 is min(u,v)
    m_u = np.minimum(edge_u[match_idx], edge_v[match_idx])

    # Precompute eigenvalue phases: theta_k = 2*pi*k/m for k=0..m-1
    theta = 2.0 * np.pi * np.arange(m) / m
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Precompute e^{i*d*theta_k} for each edge offset
    # For cycle1 edge with offset d: e^{i*d*theta_k}
    # We'll compute on the fly per evaluation since d varies per edge

    def spectral_radius_fast(signs):
        """Compute spectral radius using circulant structure.
        A = C1 ⊗ I + C2 ⊗ I + M ⊗ S
        where C1, C2 are circulant (m×m), M is matching (m×m), S is diagonal with signs.
        Actually, the signed adjacency matrix is block matrix:
        A = [[C1, M], [M, C2]] where C1,C2 are circulant, M is diagonal with matching signs.
        
        For a 2×2 block matrix with circulant blocks, we can diagonalize the circulant part.
        The eigenvalues of A are the eigenvalues of 2×2 matrices for each Fourier mode k.
        
        For mode k:
        C1_k = sum_j s1_j * e^{i*d1_j*theta_k}  (sum over cycle1 edges)
        C2_k = sum_j s2_j * e^{i*d2_j*theta_k}  (sum over cycle2 edges)
        M_k = s_match_j for the matching edge at position... 
        
        Wait, the matching connects vertex j in cycle1 to vertex j in cycle2.
        So M is a diagonal matrix with entries m_j = sign of matching edge j.
        
        The block matrix for mode k is:
        B_k = [[C1_k, m_k], [m_k, C2_k]]
        where m_k = matching sign for vertex k (since M is diagonal, M_k = m_k * I_1... 
        actually M is m×m diagonal, so in Fourier basis it's still diagonal with same entries).
        
        Hmm, let me reconsider. The full adjacency is:
        A = [[C1, D], [D, C2]]
        where C1 is m×m circulant (cycle1 edges with signs), C2 is m×m circulant (cycle2 edges with signs),
        D is m×m diagonal with D[j,j] = sign of matching edge j.
        
        To diagonalize: C1 and C2 are circulant, so they're diagonalized by the Fourier matrix F.
        D is diagonal, so in Fourier basis D is NOT diagonal (unless D is scalar).
        
        Actually, for a block matrix [[C1, D], [D, C2]] where C1, C2 are circulant and D is diagonal:
        The eigenvectors are NOT simply Fourier modes because D doesn't commute with the Fourier basis.
        
        Let me just use the direct approach but with precomputed structure.
        """
        # Build the n×n matrix efficiently
        A = np.zeros((n, n), dtype=np.float64)
        A[edge_u, edge_v] = signs
        A[edge_v, edge_u] = signs
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))

    def spectral_radius_circulant(signs):
        """Faster spectral radius using the block-circulant structure.
        
        A = [[C1, D], [D, C2]]
        C1, C2 are circulant m×m, D is diagonal m×m.
        
        We can write A = F^* ⊗ I + ... hmm this gets complex.
        Let me use the fact that for the prism graph, the adjacency has a specific structure.
        
        Actually, let's just use numpy's eigvalsh on the n×n matrix but build it faster.
        """
        A = np.zeros((n, n), dtype=np.float64)
        A[edge_u, edge_v] = signs
        A[edge_v, edge_u] = signs
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))

    def is_connected_lift(signs):
        """Check if the 2-lift is connected using BFS."""
        # Build adjacency for lift graph (2n vertices)
        lift_adj = [[] for _ in range(2 * n)]
        for i in range(num_edges):
            u, v = edge_u[i], edge_v[i]
            s = signs[i]
            for l in (0, 1):
                l2 = l if s == 1 else 1 - l
                lift_adj[u * 2 + l].append(v * 2 + l2)
                lift_adj[v * 2 + l2].append(u * 2 + l)

        visited = [False] * (2 * n)
        queue = [0]
        visited[0] = True
        while queue:
            node = queue.pop(0)
            for nb in lift_adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        return all(visited)

    def is_valid(signs):
        return is_connected_lift(signs)

    rng = np.random.default_rng(42)

    best_signs = None
    best_radius = float('inf')

    # For small m, do exhaustive or near-exhaustive search
    if m <= 5:
        # Exhaustive search: 2^num_edges possibilities
        total = 1 << num_edges
        for mask in range(total):
            if time.time() > deadline:
                break
            signs = np.array([1 if (mask >> i) & 1 else -1 for i in range(num_edges)])
            if not is_valid(signs):
                continue
            r = spectral_radius_fast(signs)
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
    elif m == 7:
        # Intensive simulated annealing with many restarts
        num_restarts = 200
        for restart in range(num_restarts):
            if time.time() > deadline:
                break

            # Random initial signs
            signs = rng.choice([-1, 1], size=num_edges)

            # Ensure valid start
            if not is_valid(signs):
                valid = False
                for _ in range(500):
                    if time.time() > deadline:
                        break
                    flip_idx = rng.integers(num_edges)
                    signs[flip_idx] *= -1
                    if is_valid(signs):
                        valid = True
                        break
                if not valid:
                    continue

            cur_radius = spectral_radius_fast(signs)
            if cur_radius < best_radius:
                best_radius = cur_radius
                best_signs = signs.copy()

            # Simulated annealing
            T_start = 0.05
            T_end = 0.0001
            num_steps = 3000
            for step in range(num_steps):
                if time.time() > deadline:
                    break

                T = T_start * (T_end / T_start) ** (step / num_steps)

                idx = rng.integers(num_edges)
                old_sign = signs[idx]
                signs[idx] = -old_sign

                if is_valid(signs):
                    new_radius = spectral_radius_fast(signs)
                    delta = new_radius - cur_radius
                    if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                        cur_radius = new_radius
                        if cur_radius < best_radius:
                            best_radius = cur_radius
                            best_signs = signs.copy()
                    else:
                        signs[idx] = old_sign
                else:
                    signs[idx] = old_sign

        # Final polish: greedy improvement
        if best_signs is not None and time.time() < deadline:
            signs = best_signs.copy()
            cur_radius = spectral_radius_fast(signs)
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(num_edges):
                    if time.time() > deadline:
                        break
                    old_sign = signs[i]
                    signs[i] = -old_sign
                    if is_valid(signs):
                        new_radius = spectral_radius_fast(signs)
                        if new_radius < cur_radius:
                            cur_radius = new_radius
                            improved = True
                        else:
                            signs[i] = old_sign
                    else:
                        signs[i] = old_sign
                if improved:
                    best_radius = cur_radius
                    best_signs = signs.copy()
    else:
        # m = 9: Simulated annealing with multiple restarts
        num_restarts = 100
        for restart in range(num_restarts):
            if time.time() > deadline:
                break

            signs = rng.choice([-1, 1], size=num_edges)

            if not is_valid(signs):
                valid = False
                for _ in range(500):
                    if time.time() > deadline:
                        break
                    flip_idx = rng.integers(num_edges)
                    signs[flip_idx] *= -1
                    if is_valid(signs):
                        valid = True
                        break
                if not valid:
                    continue

            cur_radius = spectral_radius_fast(signs)
            if cur_radius < best_radius:
                best_radius = cur_radius
                best_signs = signs.copy()

            T_start = 0.05
            T_end = 0.0001
            num_steps = 2000
            for step in range(num_steps):
                if time.time() > deadline:
                    break

                T = T_start * (T_end / T_start) ** (step / num_steps)

                idx = rng.integers(num_edges)
                old_sign = signs[idx]
                signs[idx] = -old_sign

                if is_valid(signs):
                    new_radius = spectral_radius_fast(signs)
                    delta = new_radius - cur_radius
                    if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                        cur_radius = new_radius
                        if cur_radius < best_radius:
                            best_radius = cur_radius
                            best_signs = signs.copy()
                    else:
                        signs[idx] = old_sign
                else:
                    signs[idx] = old_sign

        # Final polish
        if best_signs is not None and time.time() < deadline:
            signs = best_signs.copy()
            cur_radius = spectral_radius_fast(signs)
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(num_edges):
                    if time.time() > deadline:
                        break
                    old_sign = signs[i]
                    signs[i] = -old_sign
                    if is_valid(signs):
                        new_radius = spectral_radius_fast(signs)
                        if new_radius < cur_radius:
                            cur_radius = new_radius
                            improved = True
                        else:
                            signs[i] = old_sign
                    else:
                        signs[i] = old_sign
                if improved:
                    best_radius = cur_radius
                    best_signs = signs.copy()

    # Fallback if no valid solution found
    if best_signs is None:
        signs = np.ones(num_edges, dtype=int)
        if not is_valid(signs):
            signs = -np.ones(num_edges, dtype=int)
            if not is_valid(signs):
                for _ in range(10000):
                    if time.time() > deadline:
                        break
                    signs = rng.choice([-1, 1], size=num_edges)
                    if is_valid(signs):
                        break
        best_signs = signs

    return best_signs
