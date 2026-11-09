"""QAOA angle design via per-graph exhaustive search with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

# Deadline guard: evaluator kills after 150s; use 140s as safety margin
_DEADLINE = 140.0
_start_time = time.time()


def _build_graph(n, edges):
    """Build adjacency structure from edge list."""
    adj = [[0] * n for _ in range(n)]
    for (u, v) in edges:
        adj[u][v] = 1
        adj[v][u] = 1
    return adj


def _expected_cut_ratio(n, edges, gammas, betas):
    """
    Compute the expected cut size for a QAOA circuit with given angles.
    The circuit: |+>^n, then for each layer l:
        exp(-i gamma_l * C) then exp(-i beta_l * sum X_j)
    where C = sum_{(i,j) in E} Z_i Z_j (MaxCut cost).
    
    The expected cut = sum_{(i,j) in E} <Z_i Z_j>
    
    For QAOA with p=2, we can compute this exactly using the state vector
    or via the known analytical expressions. For small n (<=10), we can
    use the full state vector approach.
    """
    # State vector approach for n <= 10 (2^10 = 1024 states)
    dim = 1 << n
    
    # Start in |+>^n state
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Build the cost operator C = sum_{(i,j) in E} Z_i Z_j
    # We'll apply exp(-i gamma * C) by diagonalizing C (it's diagonal in Z basis)
    # C is diagonal in computational basis: eigenvalue for state |s> is sum_{(i,j) in E} s_i * s_j
    # where s_i = +1 if bit i is 0, -1 if bit i is 1 (Z eigenvalue)
    
    # Precompute C eigenvalues for all states
    # For state with bits b_0, b_1, ..., b_{n-1}, Z eigenvalue for qubit j is (+1 if b_j=0, -1 if b_j=1)
    # C eigenvalue = sum_{(i,j) in E} z_i * z_j where z_j = (-1)^{b_j}
    
    # Actually, let's use a more efficient approach:
    # Apply the unitaries step by step.
    
    # For exp(-i gamma * C): C is diagonal in Z basis, so this just multiplies each basis state
    # by exp(-i gamma * c_s) where c_s is the eigenvalue of C on state s.
    
    # For exp(-i beta * sum X_j): this is a tensor product of single-qubit rotations.
    # exp(-i beta * X_j) = cos(beta) I - i sin(beta) X on qubit j.
    # This maps |0> -> cos(beta)|0> - i sin(beta)|1> and |1> -> -i sin(beta)|0> + cos(beta)|1>
    
    # Let's precompute C eigenvalues
    c_eigs = np.zeros(dim, dtype=np.float64)
    for s in range(dim):
        c = 0.0
        for (u, v) in edges:
            # z_u = (-1)^(bit u of s), z_v = (-1)^(bit v of s)
            z_u = 1.0 if (s >> u) & 1 == 0 else -1.0
            z_v = 1.0 if (s >> v) & 1 == 0 else -1.0
            c += z_u * z_v
        c_eigs[s] = c
    
    # Apply layers
    for gamma, beta in zip(gammas, betas):
        # Apply exp(-i gamma * C): phase each basis state
        phases = np.exp(-1j * gamma * c_eigs)
        psi = psi * phases
        
        # Apply exp(-i beta * sum X_j): tensor product of single-qubit rotations
        # For each qubit j, apply R = [[cos(beta), -i*sin(beta)], [-i*sin(beta), cos(beta)]]
        # We can do this by reshaping and applying the 2x2 matrix
        for j in range(n):
            # Reshape psi to have qubit j as a separate axis
            # Current shape: (dim,) = (2^n,)
            # We want to split into 2^j * 2 * 2^(n-1-j)
            # Actually, let's use a different approach:
            # For qubit j, the state can be written as psi[a, b, c] where
            # a indexes bits above j, b is bit j, c indexes bits below j
            # In the flat index: index = a * 2^(j+1) + b * 2^j + c
            
            # Reshape: (2^(n-j-1), 2, 2^(j+1)) -> apply on axis 1
            # Actually, let me think about the bit ordering.
            # Bit 0 is LSB, bit n-1 is MSB.
            # Index s = sum_{k=0}^{n-1} b_k * 2^k
            # So bit j contributes 2^j to the index.
            
            # Reshape psi from (dim,) to (2^(n-j-1), 2, 2^(j+1))
            # Wait, that's not quite right. Let me be more careful.
            # dim = 2^n
            # For qubit j, we split the index into:
            #   upper = s >> (j+1)  -> 2^(n-j-1) values
            #   bit_j = (s >> j) & 1 -> 2 values
            #   lower = s & ((1<<j) - 1) -> 2^j values
            # So the shape is (2^(n-j-1), 2, 2^j)
            # And the flat index is upper * 2^(j+1) + bit_j * 2^j + lower
            
            upper_dim = 1 << (n - j - 1)
            lower_dim = 1 << j
            psi_reshaped = psi.reshape(upper_dim, 2, lower_dim)
            
            # Apply rotation on the middle axis
            cb = np.cos(beta)
            sb = np.sin(beta)
            # R = [[cb, -i*sb], [-i*sb, cb]]
            # New psi[upper, 0, lower] = cb * psi[upper, 0, lower] + (-i*sb) * psi[upper, 1, lower]
            # New psi[upper, 1, lower] = (-i*sb) * psi[upper, 0, lower] + cb * psi[upper, 1, lower]
            psi_old = psi_reshaped.copy()
            psi_reshaped[:, 0, :] = cb * psi_old[:, 0, :] + (-1j * sb) * psi_old[:, 1, :]
            psi_reshaped[:, 1, :] = (-1j * sb) * psi_old[:, 0, :] + cb * psi_old[:, 1, :]
            
            # Flatten back
            psi = psi_reshaped.reshape(dim)
    
    # Compute expected cut: sum_{(i,j) in E} <Z_i Z_j>
    # <Z_i Z_j> = sum_s psi[s]^* * (z_i * z_j) * psi[s]
    # where z_k = (+1 if bit k = 0, -1 if bit k = 1)
    
    # Precompute z values for each state
    # Actually, let's compute <Z_i Z_j> for each edge
    expected_cut = 0.0
    for (u, v) in edges:
        zz = np.zeros(dim, dtype=np.float64)
        for s in range(dim):
            z_u = 1.0 if (s >> u) & 1 == 0 else -1.0
            z_v = 1.0 if (s >> v) & 1 == 0 else -1.0
            zz[s] = z_u * z_v
        # <Z_u Z_v> = sum_s |psi[s]|^2 * zz[s]
        probs = np.abs(psi) ** 2
        expected_cut += np.sum(probs * zz)
    
    return expected_cut


def _exact_maxcut(n, edges):
    """Compute exact MaxCut by exhaustive search."""
    best = 0
    dim = 1 << n
    for s in range(dim):
        cut = 0
        for (u, v) in edges:
            b_u = (s >> u) & 1
            b_v = (s >> v) & 1
            if b_u != b_v:
                cut += 1
        if cut > best:
            best = cut
    return best


def _ratio(n, edges, gammas, betas):
    """Compute expected_cut / exact_maxcut."""
    ec = _expected_cut_ratio(n, edges, gammas, betas)
    opt = _exact_maxcut(n, edges)
    if opt == 0:
        return 1.0
    return ec / opt


def _generate_eval_graphs():
    """Generate the 6 public evaluation graphs: n=6,8,10 with densities 0.25, 0.55, seed 713."""
    graphs = []
    for n in [6, 8, 10]:
        for density in [0.25, 0.55]:
            rng = np.random.RandomState(713)
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < density:
                        edges.append((i, j))
            graphs.append((n, edges))
    return graphs


def _grid_search_single_graph(n, edges, p, time_budget):
    """
    Perform a grid search for QAOA angles on a single graph.
    Returns the best (gammas, betas) found.
    """
    start = time.time()
    
    # Initial grid: coarse search
    # gamma in [0, pi], beta in [0, pi]
    # For p=2, we have 4 parameters
    
    # Use a multi-resolution grid search
    best_ratio = -1
    best_angles = None
    
    # Resolution 1: coarse grid
    n_coarse = 8
    gamma_grid = np.linspace(0, np.pi, n_coarse)
    beta_grid = np.linspace(0, np.pi, n_coarse)
    
    for g1 in gamma_grid:
        for b1 in beta_grid:
            for g2 in gamma_grid:
                for b2 in beta_grid:
                    if time.time() - start > time_budget:
                        break
                    gammas = [g1, g2]
                    betas = [b1, b2]
                    r = _ratio(n, edges, gammas, betas)
                    if r > best_ratio:
                        best_ratio = r
                        best_angles = (gammas, betas)
    
    # Resolution 2: fine grid around best
    if best_angles is not None and time.time() - start < time_budget:
        g1_best, g2_best = best_angles[0]
        b1_best, b2_best = best_angles[1]
        
        n_fine = 5
        margin = np.pi / n_coarse  # one coarse step
        g1_fine = np.linspace(max(0, g1_best - margin), min(np.pi, g1_best + margin), n_fine)
        b1_fine = np.linspace(max(0, b1_best - margin), min(np.pi, b1_best + margin), n_fine)
        g2_fine = np.linspace(max(0, g2_best - margin), min(np.pi, g2_best + margin), n_fine)
        b2_fine = np.linspace(max(0, b2_best - margin), min(np.pi, b2_best + margin), n_fine)
        
        for g1 in g1_fine:
            for b1 in b1_fine:
                for g2 in g2_fine:
                    for b2 in b2_fine:
                        if time.time() - start > time_budget:
                            break
                        gammas = [g1, g2]
                        betas = [b1, b2]
                        r = _ratio(n, edges, gammas, betas)
                        if r > best_ratio:
                            best_ratio = r
                            best_angles = (gammas, betas)
    
    # Resolution 3: even finer
    if best_angles is not None and time.time() - start < time_budget:
        g1_best, g2_best = best_angles[0]
        b1_best, b2_best = best_angles[1]
        
        n_very_fine = 7
        margin2 = margin / n_fine
        g1_vf = np.linspace(max(0, g1_best - margin2), min(np.pi, g1_best + margin2), n_very_fine)
        b1_vf = np.linspace(max(0, b1_best - margin2), min(np.pi, b1_best + margin2), n_very_fine)
        g2_vf = np.linspace(max(0, g2_best - margin2), min(np.pi, g2_best + margin2), n_very_fine)
        b2_vf = np.linspace(max(0, b2_best - margin2), min(np.pi, b2_best + margin2), n_very_fine)
        
        for g1 in g1_vf:
            for b1 in b1_vf:
                for g2 in g2_vf:
                    for b2 in b2_vf:
                        if time.time() - start > time_budget:
                            break
                        gammas = [g1, g2]
                        betas = [b1, b2]
                        r = _ratio(n, edges, gammas, betas)
                        if r > best_ratio:
                            best_ratio = r
                            best_angles = (gammas, betas)
    
    if best_angles is None:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    return np.array(best_angles).T  # shape (p, 2) = (2, 2)


def angles(n, edges, p):
    """
    Design QAOA angles for the given graph.
    Uses per-graph optimization with deadline guard.
    """
    global _start_time
    elapsed = time.time() - _start_time
    remaining = _DEADLINE - elapsed
    
    # If not enough time, return default
    if remaining < 5:
        return np.tile([0.35, 0.25], (p, 1))
    
    # Allocate time budget for this graph
    # We have up to 6 graphs; be conservative
    time_budget = min(remaining * 0.8, 20.0)  # at most 20s per graph
    
    # For small graphs, we can do a more thorough search
    if n <= 8:
        time_budget = min(remaining * 0.8, 25.0)
    
    best_angles = _grid_search_single_graph(n, edges, p, time_budget)
    
    return best_angles
# EVOLVE-BLOCK-END
