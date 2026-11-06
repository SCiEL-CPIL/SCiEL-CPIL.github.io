"""Design a 32x32 photonic crystal unit cell to maximize TM band 4-5 gap."""
import numpy as np
import time
import math

def design(problem):
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]   # 1
    eps_high = problem["eps_high"] # 12
    
    t_start = time.time()
    DEADLINE = 225.0  # safety margin before 230s kill
    
    def time_left():
        return DEADLINE - (time.time() - t_start)
    
    # Precompute pixel coordinates
    coords = np.array([(i + 0.5) / size - 0.5 for i in range(size)])
    XX, YY = np.meshgrid(coords, coords, indexing='ij')
    
    def enforce_symmetry(cell):
        """Enforce D4 symmetry by averaging over the symmetry group."""
        sym_cells = [
            cell,
            cell[::-1, :],
            cell[:, ::-1],
            cell[::-1, ::-1],
            cell.T,
            cell.T[::-1, :],
            cell.T[:, ::-1],
            cell.T[::-1, ::-1]
        ]
        avg = np.mean(sym_cells, axis=0)
        return (avg >= 0.5).astype(int)
    
    def calculate_gap(cell, n_pw=7):
        """
        Calculate the band 4-5 gap ratio using the plane-wave method.
        Uses a subset of G-vectors for speed.
        """
        eps = eps_low + (eps_high - eps_low) * cell.astype(float)
        inv_eps = 1.0 / eps
        
        N = size
        half = n_pw // 2
        
        # G-vectors
        G_list = []
        for gx in range(-half, half + 1):
            for gy in range(-half, half + 1):
                G_list.append((gx, gy))
        
        nG = len(G_list)
        
        # Precompute inv_eps Fourier coefficients using FFT
        inv_eps_fft = np.fft.fft2(inv_eps)
        
        inv_eps_F = {}
        for gx, gy in G_list:
            fft_gx = gx % N
            fft_gy = gy % N
            inv_eps_F[(gx, gy)] = inv_eps_fft[fft_gx, fft_gy] / (N * N)
        
        # Key k-points in the irreducible BZ (in units where BZ is [0, 0.5]x[0, 0.5])
        # Irreducible BZ: 0 <= ky <= kx <= 0.5
        k_points_list = [
            (0.0, 0.0),      # Gamma
            (0.1, 0.0),
            (0.2, 0.0),
            (0.3, 0.0),
            (0.4, 0.0),
            (0.5, 0.0),      # X
            (0.5, 0.1),
            (0.5, 0.2),
            (0.5, 0.3),
            (0.5, 0.4),
            (0.5, 0.5),      # M
            (0.3, 0.1),
            (0.4, 0.2),
            (0.3, 0.2),
            (0.2, 0.1),
            (0.4, 0.3),
        ]
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points_list:
            # Build eigenvalue matrix
            # M_{G,G'} = (G-k)·(G'-k) * inv_eps_F[G-G']
            M = np.zeros((nG, nG), dtype=complex)
            
            for ig, (gx, gy) in enumerate(G_list):
                Gkx = gx - kx
                Gky = gy - ky
                for igp, (gxp, gyp) in enumerate(G_list):
                    Gpkx = gxp - kx
                    Gpky = gyp - ky
                    dGx = gx - gxp
                    dGy = gy - gyp
                    if (dGx, dGy) in inv_eps_F:
                        M[ig, igp] = (Gkx * Gpkx + Gky * Gpky) * inv_eps_F[(dGx, dGy)]
            
            # Solve eigenvalue problem
            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.sort(eigvals.real)
            
            if len(eigvals) >= 5:
                omega_4 = np.sqrt(max(0, eigvals[3]))
                omega_5 = np.sqrt(max(0, eigvals[4]))
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == float('inf'):
            return -1.0
        
        gap_ratio = 2.0 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
        return gap_ratio
    
    def make_superellipse_cell(a, b, n_exp):
        """
        Create a cell from a superellipse |x/a|^n + |y/b|^n <= 1.
        a, b: semi-axes (in pixel units, max ~16 for 32x32)
        n_exp: shape exponent (2=circle, 4=rounded square, large=square)
        """
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                # Convert to pixel units
                px = x * size
                py = y * size
                if abs(px) <= a and abs(py) <= b:
                    val = (abs(px) / a) ** n_exp + (abs(py) / b) ** n_exp
                    if val <= 1.0:
                        cell[i, j] = 1
        return enforce_symmetry(cell)
    
    def make_four_rod_cell(r1, r2, r3, r4, pos1, pos2, pos3, pos4):
        """
        Create a cell with four circular rods at specified positions.
        r1-r4: radii in pixels
        pos1-pos4: (x, y) positions in [-0.5, 0.5] range
        """
        cell = np.zeros((size, size), dtype=int)
        rods = [(r1, pos1), (r2, pos2), (r3, pos3), (r4, pos4)]
        for r, (px, py) in rods:
            for i in range(size):
                for j in range(size):
                    x = (i + 0.5) / size - 0.5
                    y = (j + 0.5) / size - 0.5
                    if (x - px) ** 2 + (y - py) ** 2 <= r ** 2:
                        cell[i, j] = 1
        return enforce_symmetry(cell)
    
    best_cell = None
    best_score = -float('inf')
    
    # Strategy 1: Optimize superellipse parameters
    if time_left() > 30:
        # Grid search over superellipse parameters
        a_values = np.linspace(0.1, 0.45, 8)
        b_values = np.linspace(0.1, 0.45, 8)
        n_values = [2, 3, 4, 5, 6, 8, 12]
        
        for a in a_values:
            if time_left() < 15:
                break
            for b in b_values:
                if time_left() < 10:
                    break
                for n_exp in n_values:
                    if time_left() < 5:
                        break
                    cell = make_superellipse_cell(a * size, b * size, n_exp)
                    score = calculate_gap(cell)
                    if score > best_score:
                        best_score = score
                        best_cell = cell.copy()
    
    # Strategy 2: Four-rod configurations (inspired by high-scoring solutions)
    if time_left() > 20:
        # Try various four-rod configurations
        rod_configs = [
            # (r, pos) tuples for four rods
            # Symmetric about center
            [(0.08, (-0.25, -0.25)), (0.08, (0.25, -0.25)), (0.08, (-0.25, 0.25)), (0.08, (0.25, 0.25))],
            [(0.10, (-0.25, -0.25)), (0.10, (0.25, -0.25)), (0.10, (-0.25, 0.25)), (0.10, (0.25, 0.25))],
            [(0.12, (-0.25, -0.25)), (0.12, (0.25, -0.25)), (0.12, (-0.25, 0.25)), (0.12, (0.25, 0.25))],
            [(0.08, (-0.3, -0.3)), (0.08, (0.3, -0.3)), (0.08, (-0.3, 0.3)), (0.08, (0.3, 0.3))],
            [(0.10, (-0.3, -0.3)), (0.10, (0.3, -0.3)), (0.10, (-0.3, 0.3)), (0.10, (0.3, 0.3))],
            [(0.12, (-0.3, -0.3)), (0.12, (0.3, -0.3)), (0.12, (-0.3, 0.3)), (0.12, (0.3, 0.3))],
            [(0.06, (-0.2, -0.2)), (0.06, (0.2, -0.2)), (0.06, (-0.2, 0.2)), (0.06, (0.2, 0.2))],
            [(0.14, (-0.25, -0.25)), (0.14, (0.25, -0.25)), (0.14, (-0.25, 0.25)), (0.14, (0.25, 0.25))],
            [(0.16, (-0.25, -0.25)), (0.16, (0.25, -0.25)), (0.16, (-0.25, 0.25)), (0.16, (0.25, 0.25))],
            # Non-square arrangements
            [(0.08, (-0.3, 0.0)), (0.08, (0.3, 0.0)), (0.08, (0.0, -0.3)), (0.08, (0.0, 0.3))],
            [(0.10, (-0.3, 0.0)), (0.10, (0.3, 0.0)), (0.10, (0.0, -0.3)), (0.10, (0.0, 0.3))],
            [(0.12, (-0.3, 0.0)), (0.12, (0.3, 0.0)), (0.12, (0.0, -0.3)), (0.12, (0.0, 0.3))],
            [(0.08, (-0.2, 0.0)), (0.08, (0.2, 0.0)), (0.08, (0.0, -0.2)), (0.08, (0.0, 0.2))],
            [(0.10, (-0.2, 0.0)), (0.10, (0.2, 0.0)), (0.10, (0.0, -0.2)), (0.10, (0.0, 0.2))],
        ]
        
        for config in rod_configs:
            if time_left() < 5:
                break
            cell = make_four_rod_cell(
                config[0][0], config[1][0], config[2][0], config[3][0],
                config[0][1], config[1][1], config[2][1], config[3][1]
            )
            score = calculate_gap(cell)
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
    
    # Strategy 3: Local optimization of best superellipse
    if best_cell is not None and best_score > 0 and time_left() > 20:
        # Try to refine by perturbing the best cell
        np.random.seed(42)
        for _ in range(10):
            if time_left() < 10:
                break
            
            # Random perturbation of the best cell
            perturbed = best_cell.copy()
            n_perturb = np.random.randint(1, 5)
            for _ in range(n_perturb):
                ri = np.random.randint(0, size)
                rj = np.random.randint(0, size)
                perturbed[ri, rj] = 1 - perturbed[ri, rj]
            
            # Enforce symmetry
            perturbed = enforce_symmetry(perturbed)
            score = calculate_gap(perturbed)
            if score > best_score:
                best_score = score
                best_cell = perturbed.copy()
    
    # Strategy 4: Try inverted patterns (more air than dielectric)
    if time_left() > 15:
        inverted_configs = [
            [(0.08, (-0.25, -0.25)), (0.08, (0.25, -0.25)), (0.08, (-0.25, 0.25)), (0.08, (0.25, 0.25))],
            [(0.10, (-0.25, -0.25)), (0.10, (0.25, -0.25)), (0.10, (-0.25, 0.25)), (0.10, (0.25, 0.25))],
        ]
        
        for config in inverted_configs:
            if time_left() < 5:
                break
            cell = make_four_rod_cell(
                config[0][0], config[1][0], config[2][0], config[3][0],
                config[0][1], config[1][1], config[2][1], config[3][1]
            )
            # Invert: 1 becomes 0 and vice versa
            cell_inv = 1 - cell
            cell_inv = enforce_symmetry(cell_inv)
            score = calculate_gap(cell_inv)
            if score > best_score:
                best_score = score
                best_cell = cell_inv.copy()
    
    # Final fallback
    if best_cell is None or best_score < 0:
        # Simple four-rod pattern
        cell = make_four_rod_cell(
            0.1, 0.1, 0.1, 0.1,
            (-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)
        )
        best_cell = cell
    
    return best_cell.tolist()
