import math
import time
import numpy as np

def design(problem):
    """
    Design a 32x32 unit cell with C4v symmetry using a parameterized
    trigonometric field. Optimizes coefficients to maximize the TM band gap
    between bands 4 and 5.
    """
    start_time = time.time()
    deadline = start_time + 210.0  # Safety margin
    
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12.0)
    
    # --- Pre-computation for the solver ---
    M = 8
    G_vals = np.arange(-M, M + 1)
    G_vecs = np.array([(gx, gy) for gx in G_vals for gy in G_vals])
    N_pw = len(G_vecs)
    
    # K-points: 30 points in the irreducible BZ
    # 11 points on Gamma-M, 11 on Gamma-X, 11 on M-X -> 30 unique
    num_per_line = 11
    pts_line1 = [(i/(num_per_line-1)*0.5, 0.0) for i in range(num_per_line)]
    pts_line2 = [(i/(num_per_line-1)*0.5, i/(num_per_line-1)*0.5) for i in range(num_per_line)]
    pts_line3 = [(0.5, i/(num_per_line-1)*0.5) for i in range(num_per_line)]
    all_pts = pts_line1 + pts_line2 + pts_line3
    k_points = list(set(all_pts))
    
    # Pixel coordinates
    x_coords = (np.arange(size) + 0.5) / size - 0.5
    y_coords = (np.arange(size) + 0.5) / size - 0.5
    XX, YY = np.meshgrid(x_coords, y_coords, indexing='xy')
    xx = XX.flatten()
    yy = YY.flatten()
    
    # Pre-compute phase factors for Fourier coefficients
    phases = np.zeros((N_pw, size*size), dtype=complex)
    for idx, (gx, gy) in enumerate(G_vecs):
        phases[idx] = np.exp(-1j * 2 * np.pi * (gx * xx + gy * yy))
        
    # Pre-compute G-G' indices for the matrix construction
    # We need to map (dgx, dgy) to an index in inv_eps_coeffs
    # inv_eps_coeffs is indexed by G_vecs order.
    # G_vecs order: gx from -M to M, gy from -M to M.
    # Index = (gx + M) * (2*M + 1) + (gy + M)
    
    def compute_fourier_coeffs(inv_eps_flat):
        return np.dot(inv_eps_flat, phases.T) / (size * size)

    def compute_gap_ratio(cell):
        cell_arr = np.array(cell, dtype=float)
        inv_eps = np.where(cell_arr == 1, 1.0/eps_high, 1.0)
        inv_eps_flat = inv_eps.flatten()
        
        inv_eps_coeffs = compute_fourier_coeffs(inv_eps_flat)
        
        min_omega_5 = float('inf')
        max_omega_4 = -float('inf')
        
        # Pre-compute dot products structure? No, k changes.
        
        for kx, ky in k_points:
            kg_x = kx + G_vecs[:, 0]
            kg_y = ky + G_vecs[:, 1]
            
            dot_products = np.outer(kg_x, kg_x) + np.outer(kg_y, kg_y)
            
            A = np.zeros((N_pw, N_pw), dtype=complex)
            
            # Efficient row construction
            for i in range(N_pw):
                gx_i, gy_i = G_vecs[i]
                dgx = gx_i - G_vecs[:, 0]
                dgy = gy_i - G_vecs[:, 1]
                
                # Check validity: dgx, dgy in [-M, M]
                valid = (dgx >= -M) & (dgx <= M) & (dgy >= -M) & (dgy <= M)
                
                if np.any(valid):
                    idx_dgx = dgx[valid] + M
                    idx_dgy = dgy[valid] + M
                    lin_idx = idx_dgx * (2*M+1) + idx_dgy
                    row_coeffs = np.zeros(N_pw, dtype=complex)
                    row_coeffs[valid] = inv_eps_coeffs[lin_idx]
                    A[i, :] = dot_products[i, :] * row_coeffs
            
            try:
                vals = np.linalg.eigvalsh(A)
                omegas = np.sort(np.sqrt(np.real(vals)))
                
                if len(omegas) >= 5:
                    w4 = omegas[3]
                    w5 = omegas[4]
                    max_omega_4 = max(max_omega_4, w4)
                    min_omega_5 = min(min_omega_5, w5)
            except np.linalg.LinAlgError:
                continue
        
        if min_omega_5 == float('inf') or max_omega_4 == -float('inf'):
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return -1.0
            
        ratio = 2 * gap / (min_omega_5 + max_omega_4)
        return ratio

    def generate_cell(c1, c2, c3, c4, t):
        """
        Generate C4v symmetric cell from:
        V = c1*cos(2pi x)*cos(2pi y) + c2*cos(4pi x)*cos(4pi y) + 
            c3*cos(2pi(x+y)) + c4*cos(2pi(x-y))
        """
        # Vectorized generation
        term1 = np.cos(2 * np.pi * XX) * np.cos(2 * np.pi * YY)
        term2 = np.cos(4 * np.pi * XX) * np.cos(4 * np.pi * YY)
        term3 = np.cos(2 * np.pi * (XX + YY))
        term4 = np.cos(2 * np.pi * (XX - YY))
        
        V = c1 * term1 + c2 * term2 + c3 * term3 + c4 * term4
        
        cell = (V > t).astype(int)
        
        # Ensure exact symmetry (should be true by construction, but float errors might bite)
        # Enforce C4v:
        # 1. cell == cell[::-1] (180 rot)
        # 2. cell == cell.T (transpose)
        # 3. cell == cell[:, ::-1] (reflect x) -> implied by 1 and 2?
        # Let's just enforce the required ones explicitly to be safe.
        cell = np.where(cell == 1, 1, 0)
        
        # Force symmetry
        # Transpose symmetry
        cell = (cell + cell.T) // 2
        # 180 rotation symmetry
        cell = (cell + cell[::-1, ::-1]) // 2
        
        return cell.tolist()

    # --- Optimization ---
    import random
    random.seed(42)
    
    best_score = -1000.0
    best_params = None
    
    # Initial candidates
    candidates = [
        (1.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0, 0.0),
        (1.0, 0.5, 0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5, 0.0, 0.0),
    ]
    
    for p in candidates:
        if time.time() > deadline:
            break
        cell = generate_cell(*p)
        score = compute_gap_ratio(cell)
        if score > best_score:
            best_score = score
            best_params = p

    # Random search
    max_evals = 150
    num_evals = 0
    
    while time.time() < deadline and num_evals < max_evals:
        # Random parameters
        c1 = random.uniform(-1.0, 1.0)
        c2 = random.uniform(-1.0, 1.0)
        c3 = random.uniform(-1.0, 1.0)
        c4 = random.uniform(-1.0, 1.0)
        t = random.uniform(-1.0, 1.0)
        
        cell = generate_cell(c1, c2, c3, c4, t)
        score = compute_gap_ratio(cell)
        num_evals += 1
        
        if score > best_score:
            best_score = score
            best_params = (c1, c2, c3, c4, t)
            
            # If we find a good one, do a local refinement
            if best_score > 0.05 and time.time() < deadline - 30.0:
                # Local search around best_params
                for _ in range(10):
                    if time.time() > deadline - 30.0:
                        break
                    nc1 = best_params[0] + random.gauss(0, 0.1)
                    nc2 = best_params[1] + random.gauss(0, 0.1)
                    nc3 = best_params[2] + random.gauss(0, 0.1)
                    nc4 = best_params[3] + random.gauss(0, 0.1)
                    nt = best_params[4] + random.gauss(0, 0.1)
                    
                    ncell = generate_cell(nc1, nc2, nc3, nc4, nt)
                    nscore = compute_gap_ratio(ncell)
                    if nscore > best_score:
                        best_score = nscore
                        best_params = (nc1, nc2, nc3, nc4, nt)

    if best_params is None:
        # Fallback
        best_params = (1.0, 0.0, 0.0, 0.0, 0.0)
        
    final_cell = generate_cell(*best_params)
    return final_cell
