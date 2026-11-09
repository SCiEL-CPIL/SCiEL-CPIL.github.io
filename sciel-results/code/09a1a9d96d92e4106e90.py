import numpy as np
import time

def design(problem):
    """
    Maximizes the TM band gap between bands 4 and 5 for a 32x32 square-lattice
    photonic crystal unit cell (eps=12 dielectric in air) with C4v symmetry.
    Uses a discrete C4v-symmetric binary cell (8x8 fundamental block) optimized
    via greedy local search + stochastic restarts.
    """
    start_time = time.time()
    deadline = start_time + 210.0  # Safety margin before 230s kill

    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)

    # Precompute plane-wave setup matching evaluator: 289 PWs (17x17), 30 k-points
    # Irreducible BZ: 0 <= ky <= kx, kx+ky <= 1 (for square lattice, BZ: -pi..pi, but normalized)
    # Actually for 2D square lattice photonic crystal, the BZ in normalized coordinates:
    # kx, ky in [-0.5, 0.5], irreducible: 0 <= ky <= kx <= 0.5
    # But the task says "30 wave vectors covering the boundary and interior"
    # Let me use a standard set: Gamma(0,0), M(0.5,0), X(0.5,0.5) plus interior points
    
    # Standard irreducible BZ points for square lattice (normalized: kx*period, ky*period)
    # The BZ extends from -pi to pi, so normalized to period: -0.5 to 0.5
    # Irreducible wedge: 0 <= ky <= kx <= 0.5
    
    # 30 points: Let me create a reasonable set
    k_points = []
    # Gamma
    k_points.append((0.0, 0.0))
    # M point (0.5, 0)
    k_points.append((0.5, 0.0))
    # X point (0.5, 0.5)
    k_points.append((0.5, 0.5))
    # Along Gamma-M (kx from 0 to 0.5, ky=0)
    for t in np.linspace(0, 0.5, 7)[1:-1]:
        k_points.append((t, 0.0))
    # Along Gamma-X (kx=ky, from 0 to 0.5)
    for t in np.linspace(0, 0.5, 7)[1:-1]:
        k_points.append((t, t))
    # Along M-X (kx=0.5, ky from 0 to 0.5)
    for t in np.linspace(0, 0.5, 7)[1:-1]:
        k_points.append((0.5, t))
    # Interior points
    for kx in [0.25, 0.35, 0.45]:
        for ky in [0.05, 0.15, 0.25, 0.35]:
            if ky <= kx:
                k_points.append((kx, ky))
    
    # Deduplicate and ensure we have ~30
    seen = set()
    unique_k = []
    for k in k_points:
        key = (round(k[0], 6), round(k[1], 6))
        if key not in seen:
            seen.add(key)
            unique_k.append(k)
    
    # Take up to 30
    unique_k = unique_k[:30]
    num_k = len(unique_k)

    # Plane wave vectors: 17x17 grid, G = (mx, my) with mx, my in [-8, 8]
    G_vals = np.arange(-8, 9)  # 17 values
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))  # (289, 2)
    num_pw = len(G_vectors)  # 289

    # Precompute G_diff: G_diff[i,j] = G_vectors[i] - G_vectors[j]
    # This is (289, 289, 2)
    G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]

    def build_coeff_matrix(eps_grid):
        """
        Build the coefficient matrix C[i,j] = <eps^{-1}>_{G_i - G_j}
        where <eps^{-1}> is the Fourier coefficient of 1/eps.
        """
        inv_eps = 1.0 / eps_grid
        # FFT of inv_eps to get Fourier coefficients
        # The pixel grid represents eps(x,y) on [0,1]x[0,1]
        # Fourier coefficient: c_n = integral f(x) exp(-2*pi*i*n*x) dx
        # For a pixel grid, approximate with: c_n ~ (1/N^2) * sum_{ij} f_{ij} exp(-2*pi*i*(n_x*j/N + n_y*i/N))
        # Which is essentially FFT / N^2
        fft_inv_eps = np.fft.fft2(inv_eps)
        fft_inv_eps /= (size * size)
        
        # Now C[i,j] = c_{G_i - G_j}
        # G_i - G_j has components in [-16, 16]
        # fft_inv_eps has indices 0..31, so we need to wrap: index = (G_diff) % 32
        coeff_matrix = np.zeros((num_pw, num_pw), dtype=complex)
        for i in range(num_pw):
            for j in range(num_pw):
                dx = int(G_diff[i, j, 0])
                dy = int(G_diff[i, j, 1])
                ix = dx % size
                iy = dy % size
                coeff_matrix[i, j] = fft_inv_eps[ix, iy]
        
        return coeff_matrix

    def compute_gap(eps_grid):
        """
        Compute the gap-midgap ratio between bands 4 and 5.
        Returns the ratio, or negative if bands overlap.
        """
        coeff_matrix = build_coeff_matrix(eps_grid)
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in unique_k:
            k = np.array([kx, ky])
            k_plus_G = k + G_vectors  # (289, 2)
            omega_sq = np.sum(k_plus_G ** 2, axis=1)  # (289,)
            
            # M[i,j] = coeff_matrix[i,j] * omega_sq[i]
            # This is a generalized eigenvalue problem, but we can solve the standard one
            # The plane-wave method gives: sum_j C_{ij} (k+G_j)^2 a_j = omega^2 a_i
            # Wait, let me reconsider. The standard PW expansion for TM:
            # The equation is: sum_j <1/eps>_{G_i-G_j} (k+G_j)^2 a_j = omega^2 a_i
            # So M[i,j] = C[i,j] * (k+G_j)^2, and we solve M @ a = omega^2 @ a
            
            M = coeff_matrix * omega_sq[np.newaxis, :]
            
            eigvals = np.linalg.eigvalsh(M)  # Hermitian, use eigvalsh
            real_eigvals = np.sort(eigvals.real)
            
            if len(real_eigvals) >= 5:
                omega_4 = np.sqrt(real_eigvals[3])
                omega_5 = np.sqrt(real_eigvals[4])
                
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0
        
        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap < 1e-10:
            return -1.0
        ratio = 2.0 * gap / midgap
        return ratio

    # C4v symmetric cell: 32x32, invariant under x->1-x, y->1-y, and transpose
    # Fundamental block: top-left 8x8 quadrant
    # Full cell is constructed by:
    # C[i,j] for i,j in 0..31
    # Quadrants: TL(0:8,0:8), TR(0:8,24:32), BL(24:32,0:8), BR(24:32,24:32)
    # TL = Q (8x8)
    # TR = Q[:, ::-1] (reflect horizontally)
    # BL = Q[::-1, :] (reflect vertically)
    # BR = Q[::-1, ::-1] (reflect both)
    
    def expand_to_32(Q):
        """Expand 8x8 fundamental block to 32x32 C4v symmetric cell."""
        C = np.zeros((32, 32), dtype=int)
        C[0:8, 0:8] = Q
        C[0:8, 24:32] = Q[:, ::-1]
        C[24:32, 0:8] = Q[::-1, :]
        C[24:32, 24:32] = Q[::-1, ::-1]
        return C

    def eps_from_cell(cell):
        """Convert binary cell to epsilon grid."""
        return np.where(cell == 1, eps_high, eps_low)

    def score_cell(cell):
        """Score a binary cell."""
        eps_grid = eps_from_cell(cell)
        return compute_gap(eps_grid)

    # Optimization: Greedy local search with random restarts
    best_score = -np.inf
    best_Q = None
    
    # Initial guess: a central dielectric region
    # Try a few initial patterns
    initial_patterns = [
        # Central cross
        np.array([[0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0],
                  [0,0,0,1,1,0,0,0],
                  [0,0,0,1,1,0,0,0],
                  [0,0,0,1,1,0,0,0],
                  [0,0,0,1,1,0,0,0],
                  [0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0]]),
        # Central square
        np.array([[0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0],
                  [0,0,0,1,1,0,0,0],
                  [0,0,0,1,1,0,0,0],
                  [0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0]]),
        # Ring-like
        np.array([[0,0,0,0,0,0,0,0],
                  [0,1,1,1,1,1,1,0],
                  [0,1,0,0,0,0,1,0],
                  [0,1,0,0,0,0,1,0],
                  [0,1,0,0,0,0,1,0],
                  [0,1,0,0,0,0,1,0],
                  [0,1,1,1,1,1,1,0],
                  [0,0,0,0,0,0,0,0]]),
    ]
    
    for pat in initial_patterns:
        Q = pat.copy()
        cell = expand_to_32(Q)
        s = score_cell(cell)
        if s > best_score:
            best_score = s
            best_Q = Q.copy()
    
    # If no good initial pattern, try random
    if best_Q is None:
        Q = np.random.randint(0, 2, (8, 8))
        cell = expand_to_32(Q)
        best_score = score_cell(cell)
        best_Q = Q.copy()

    # Local search: flip single pixels in the 8x8 fundamental block
    current_Q = best_Q.copy()
    current_score = best_score
    
    # Greedy hill climbing with occasional random restarts
    num_restarts = 0
    max_restarts = 15
    stall_count = 0
    max_stall = 200
    
    while time.time() < deadline - 15:
        improved = False
        stall_count = 0
        
        # Greedy: flip each pixel, keep if improves
        for i in range(8):
            if time.time() > deadline - 10:
                break
            for j in range(8):
                if time.time() > deadline - 10:
                    break
                # Flip pixel (i,j)
                old_val = current_Q[i, j]
                current_Q[i, j] = 1 - old_val
                cell = expand_to_32(current_Q)
                new_score = score_cell(cell)
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    # Keep the flip
                else:
                    # Revert
                    current_Q[i, j] = old_val
                
                stall_count += 1
        
        if improved:
            if current_score > best_score:
                best_score = current_score
                best_Q = current_Q.copy()
        else:
            # No improvement, try a random perturbation
            stall_count += 1
            if stall_count > max_stall:
                num_restarts += 1
                if num_restarts >= max_restarts:
                    break
                # Random restart
                current_Q = np.random.randint(0, 2, (8, 8))
                cell = expand_to_32(current_Q)
                current_score = score_cell(cell)
                stall_count = 0

    # Final: ensure we have the best
    if best_Q is None:
        best_Q = np.zeros((8, 8), dtype=int)
    
    final_cell = expand_to_32(best_Q)
    
    # Convert to nested list of 0/1
    result = []
    for i in range(32):
        row = []
        for j in range(32):
            row.append(int(final_cell[i, j]))
        result.append(row)
    
    return result
