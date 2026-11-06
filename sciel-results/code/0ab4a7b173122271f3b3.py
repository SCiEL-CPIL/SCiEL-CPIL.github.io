import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    start_time = time.time()
    deadline = 115.0
    
    # Symmetry: C4v
    # Fundamental domain: top-left triangle of the top-left quadrant
    # Quadrant size: 16x16
    # Triangle: i in [0, 15], j in [0, i]
    
    half = size // 2
    tri_size = half
    indices = []
    for i in range(tri_size):
        for j in range(i + 1):
            indices.append((i, j))
    n_params = len(indices)
    
    # Precompute DFT phases
    # G vectors: -2 to 2
    g_indices = np.arange(-2, 3)
    G_vectors = []
    for gx in g_indices:
        for gy in g_indices:
            G_vectors.append((gx, gy))
    n_g = len(G_vectors)
    
    x_coords = (np.arange(size) + 0.5) / size
    y_coords = (np.arange(size) + 0.5) / size
    XX, YY = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    phases = []
    for gx, gy in G_vectors:
        Gx = 2 * np.pi * gx
        Gy = 2 * np.pi * gy
        phase = np.exp(-1j * (Gx * XX + Gy * YY))
        phases.append(phase)
    phases = np.array(phases) # shape (n_g, size, size)
    
    # PWE k-points
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5)
    ]
    
    # Precompute (k+G) dot (k+G') for each k point
    # H_GG' = (k+G) dot (k+G') delta + 1/eps(G-G')
    # We need eps(G-G') where G, G' are in G_vectors.
    # G-G' ranges from -4 to 4.
    
    # Map (dx, dy) to index in coeffs
    def get_eps_index(dx, dy):
        # We need to find coeff for G_diff = (dx, dy)
        # Our coeffs are for G in [-2, 2]
        # But G-G' can be up to 4.
        # We need to compute coeffs for G in [-4, 4]
        # Let's extend G_vectors for coeffs calculation?
        # No, we can just compute the specific coeffs needed.
        # Actually, it's easier to compute coeffs for all G in [-4, 4]
        # That's 9x9=81 coeffs.
        # My G_vectors for PWE matrix are 5x5=25.
        # The differences G_i - G_j will be in [-4, 4].
        
        pass

    # Let's compute coeffs for G in [-4, 4]
    g_indices_full = np.arange(-4, 5)
    G_vectors_full = []
    for gx in g_indices_full:
        for gy in g_indices_full:
            G_vectors_full.append((gx, gy))
    n_g_full = len(G_vectors_full)
    
    phases_full = []
    for gx, gy in G_vectors_full:
        Gx = 2 * np.pi * gx
        Gy = 2 * np.pi * gy
        phase = np.exp(-1j * (Gx * XX + Gy * YY))
        phases_full.append(phase)
    phases_full = np.array(phases_full)
    
    def compute_coeffs(eps_grid):
        # eps_grid is size x size
        # coeffs = sum(eps * phase) / (size*size)
        # phases_full shape (81, 32, 32)
        # eps_grid shape (32, 32)
        # Use einsum or broadcasting
        # coeffs = np.sum(eps_grid[None, :, :] * phases_full, axis=(1, 2)) / (size*size)
        coeffs = np.tensordot(phases_full, eps_grid, axes=([1, 2], [0, 1])) / (size * size)
        return coeffs

    def get_eps_val(coeffs, dx, dy):
        # Map (dx, dy) to index in coeffs
        # dx in [-4, 4], dy in [-4, 4]
        # Index = (dx + 4) * 9 + (dy + 4)
        idx = (dx + 4) * 9 + (dy + 4)
        return coeffs[idx]

    def solve_pwe(kx, ky, coeffs):
        N = len(G_vectors)
        H = np.zeros((N, N), dtype=complex)
        
        for i in range(N):
            Gi = G_vectors[i]
            ki = np.array([kx, ky]) + Gi
            for j in range(N):
                Gj = G_vectors[j]
                kj = np.array([kx, ky]) + Gj
                dot = np.dot(ki, kj)
                if i == j:
                    H[i, j] += dot
                dGx = Gi[0] - Gj[0]
                dGy = Gi[1] - Gj[1]
                eps_val = get_eps_val(coeffs, dGx, dGy)
                if eps_val != 0:
                    H[i, j] += 1.0 / eps_val
                else:
                    H[i, j] += 1.0
        
        vals = np.linalg.eigvalsh(H)
        vals = np.sort(vals)
        omegas = np.sqrt(vals)
        return omegas

    def evaluate_cell(triangle_vals):
        # Reconstruct full cell
        cell_q = np.zeros((half, half), dtype=int)
        for idx, (i, j) in enumerate(indices):
            val = triangle_vals[idx]
            cell_q[i][j] = val
            # Mirror within quadrant for C4v?
            # C4v implies M_ij = M_ji (in quadrant)
            # So we just set cell_q[j][i] = val as well?
            # Yes, because the quadrant is symmetric.
            if i != j:
                cell_q[j][i] = val
        
        # Mirror quadrant to full cell
        full_cell = np.zeros((size, size), dtype=int)
        full_cell[:half, :half] = cell_q
        full_cell[half:, half:] = np.flip(np.flip(cell_q, 0), 1)
        full_cell[:half, half:] = np.flip(cell_q, 1)
        full_cell[half:, :half] = np.flip(cell_q, 0)
        
        # Compute epsilon grid
        eps_grid = np.where(full_cell == 1, eps_high, eps_low)
        
        coeffs = compute_coeffs(eps_grid)
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in k_points:
            omegas = solve_pwe(kx, ky, coeffs)
            if len(omegas) >= 5:
                w4 = omegas[3]
                w5 = omegas[4]
                max_omega_4 = max(max_omega_4, w4)
                min_omega_5 = min(min_omega_5, w5)
        
        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return 0.0
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap

    # Initial guess: Ring
    triangle_vals = np.zeros(n_params, dtype=int)
    for idx, (i, j) in enumerate(indices):
        x = (i + 0.5) / size - 0.5
        y = (j + 0.5) / size - 0.5
        r2 = x*x + y*y
        if 0.2**2 < r2 < 0.4**2:
            triangle_vals[idx] = 1
            
    best_score = evaluate_cell(triangle_vals)
    best_vals = triangle_vals.copy()
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.99
    
    while time.time() < deadline:
        # Flip random bit
        idx = np.random.randint(0, n_params)
        triangle_vals[idx] = 1 - triangle_vals[idx]
        
        new_score = evaluate_cell(triangle_vals)
        
        if new_score > best_score:
            best_score = new_score
            best_vals = triangle_vals.copy()
        else:
            # Accept with probability
            delta = new_score - best_score
            if delta < 0 and np.random.rand() < np.exp(delta / T):
                pass # Keep new state
            else:
                # Revert
                triangle_vals[idx] = 1 - triangle_vals[idx]
        
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Return best full cell
    cell_q = np.zeros((half, half), dtype=int)
    for idx, (i, j) in enumerate(indices):
        val = best_vals[idx]
        cell_q[i][j] = val
        if i != j:
            cell_q[j][i] = val
            
    full_cell = np.zeros((size, size), dtype=int)
    full_cell[:half, :half] = cell_q
    full_cell[half:, half:] = np.flip(np.flip(cell_q, 0), 1)
    full_cell[:half, half:] = np.flip(cell_q, 1)
    full_cell[half:, :half] = np.flip(cell_q, 0)
    
    return full_cell.tolist()
