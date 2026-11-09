import numpy as np
import time

def design(problem):
    """
    Designs a 32x32 unit cell with C4v symmetry to maximize the TM band gap 
    between bands 4 and 5.
    
    Strategy:
    1. Coarse search using parametric superellipse shapes.
    2. Fine-tuning using local pixel hill-climbing (bit-flipping) on the 
       symmetry-reduced domain.
    """
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # --- Fast Band Solver ---
    
    # Precompute G vectors for the plane wave expansion
    # The evaluator uses 289 PWs. We match this with a square grid of G vectors.
    G_range = 8
    G_vals = np.arange(-G_range, G_range + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors) # 17*17 = 289
    
    # K points: Representative set covering the irreducible BZ
    k_points = []
    for t in np.linspace(0, 0.5, 6):
        k_points.append((t, 0.0))
    for t in np.linspace(0, 0.5, 5):
        k_points.append((0.5, t))
    for t in np.linspace(0, 0.5, 4):
        k_points.append((t, t))
    k_points.append((0.25, 0.25))
    k_points.append((0.4, 0.1))
    k_points.append((0.1, 0.4))
    
    unique_k = []
    seen = set()
    for k in k_points:
        key = (round(k[0], 10), round(k[1], 10))
        if key not in seen:
            seen.add(key)
            unique_k.append(k)

    def compute_coeff_matrix(cell):
        """
        Computes the matrix C[i,j] = <G_i | 1/eps | G_j>
        """
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        fft_2d = np.fft.fft2(inv_eps) / (size * size)
        
        Gx_i = G_vectors[:, 0:1]
        Gy_i = G_vectors[:, 1:2]
        Gx_j = G_vectors[0:1, 0]
        Gy_j = G_vectors[0:1, 1]
        
        dx = Gx_i - Gx_j
        dy = Gy_i - Gy_j
        
        idx_x = dx % size
        idx_y = dy % size
        
        C = fft_2d[idx_x, idx_y]
        return C

    def solve_bands(cell):
        """
        Solves the eigenvalue problem for the given cell.
        Returns the gap-midgap ratio.
        """
        C = compute_coeff_matrix(cell)
        
        max_omega4 = -1e10
        min_omega5 = 1e10
        
        for kx, ky in unique_k:
            k = np.array([kx, ky])
            k_minus_G = k[None, :] - G_vectors
            omega_sq = np.sum(k_minus_G ** 2, axis=1)
            
            M = C * omega_sq[None, :]
            H = (M + M.conj().T) / 2.0
            
            try:
                eigenvalues = np.linalg.eigvalsh(H)
                eigenvalues = np.sort(np.real(eigenvalues))
                if len(eigenvalues) >= 5:
                    omega4 = eigenvalues[3]
                    omega5 = eigenvalues[4]
                    if omega4 > max_omega4:
                        max_omega4 = omega4
                    if omega5 < min_omega5:
                        min_omega5 = omega5
            except:
                pass
        
        if max_omega4 <= 0 or min_omega5 <= 0 or min_omega5 <= max_omega4:
            return -1.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        
        if midgap <= 0:
            return -1.0
        
        return gap / midgap

    # --- Phase 1: Coarse Superellipse Search ---
    
    def make_cell_superellipse(radius, n_order, offset, size=32):
        cell = np.zeros((size, size), dtype=int)
        centers = [
            (size * 0.25 + offset, size * 0.25 + offset),
            (size * 0.75 - offset, size * 0.25 + offset),
            (size * 0.25 + offset, size * 0.75 - offset),
            (size * 0.75 - offset, size * 0.75 - offset),
        ]
        for i in range(size):
            for j in range(size):
                y_pix = i + 0.5
                x_pix = j + 0.5
                for cx, cy in centers:
                    dx = x_pix - cx
                    dy = y_pix - cy
                    if radius > 0:
                        val = abs(dx / radius) ** n_order + abs(dy / radius) ** n_order
                        if val < 1.0:
                            cell[i, j] = 1
                            break
        return cell

    best_score = -1e10
    best_cell = None
    
    radii = [4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
    orders = [2.0, 3.0, 4.0, 6.0]
    offsets = [-1.0, 0.0, 1.0]
    
    for r in radii:
        for n_ord in orders:
            for off in offsets:
                if time.time() > deadline - 60:
                    break
                cell = make_cell_superellipse(r, n_ord, off)
                score = solve_bands(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell.copy()
        if time.time() > deadline - 60:
            break

    if best_cell is None:
        best_cell = np.zeros((size, size), dtype=int)
        best_score = 0.0

    # --- Phase 2: Pixel Hill Climbing ---
    
    orbits = []
    for i in range(16):
        for j in range(i, 16):
            indices = set()
            positions = [
                (i, j),
                (i, size - 1 - j),
                (size - 1 - i, j),
                (size - 1 - i, size - 1 - j),
                (j, i),
                (j, size - 1 - i),
                (size - 1 - j, i),
                (size - 1 - j, size - 1 - i)
            ]
            for p in positions:
                indices.add(p)
            orbits.append(list(indices))
    
    np.random.shuffle(orbits)
    
    current_cell = best_cell.copy()
    current_score = best_score
    
    stall_count = 0
    
    while time.time() < deadline - 5:
        idx = np.random.randint(len(orbits))
        orbit = orbits[idx]
        
        for (r, c) in orbit:
            current_cell[r, c] = 1 - current_cell[r, c]
        
        new_score = solve_bands(current_cell)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_cell = current_cell.copy()
            stall_count = 0
        else:
            for (r, c) in orbit:
                current_cell[r, c] = 1 - current_cell[r, c]
            stall_count += 1
        
        if stall_count > 100:
            current_cell = best_cell.copy()
            current_score = best_score
            stall_count = 0

    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(best_cell[i, j]))
        result.append(row)
    
    return result
