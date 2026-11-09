"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def design(problem):
    """
    Design a symmetric 32x32 unit cell for a TM band gap.
    Strategy: 
    1. Use a fast local search (hill climbing) with vectorized evaluation.
    2. Start from a good seed (4-rod configuration).
    3. Enforce D4 symmetry by operating on the 8x8 fundamental domain.
    4. Use a time budget to ensure completion.
    """
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # 1. Setup Plane Wave Method Pre-computation
    # We need to solve the eigenvalue problem for TM modes.
    # The matrix is H = (1/sqrt((k+G)^2)) * (1/eps)_{G-G'} * (1/sqrt((k+G')^2))
    # where (1/eps)_{G-G'} are Fourier coefficients of 1/epsilon.
    
    # Number of plane waves. 289 = 17x17 implies G from -8 to 8.
    N_G = 8
    G_vals = np.arange(-N_G, N_G + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    Gx_flat = Gx.ravel()
    Gy_flat = Gy.ravel()
    N_pw = len(Gx_flat)
    
    # Precompute differences for Fourier coefficients
    # We need mu_{G-G'} where mu = 1/eps
    # The Fourier coefficient for difference (dx, dy) is:
    # mu_{dx,dy} = (1/N^2) * sum_{i,j} mu(i,j) * exp(-i 2pi (dx*i + dy*j)/N)
    # This is basically the DFT of the mu matrix, shifted.
    # However, we can compute the matrix M_{G, G'} = mu_{G-G'} directly.
    # M_{G, G'} = mu_{Gx-Gx', Gy-Gy'}
    
    # Precompute k-points
    # Irreducible BZ of square lattice: 0 <= kx <= ky, kx+ky <= pi (if a=1, b=2pi)
    # Actually, standard square BZ is -pi..pi. Irreducible is 0 <= kx <= ky, kx <= pi/2?
    # No, for square lattice, the irreducible wedge is 0 <= kx <= ky, kx+ky <= pi (in units of 2pi/a? No).
    # Let's use standard units: a=1. Reciprocal lattice vectors are 2pi.
    # BZ is [-pi, pi] x [-pi, pi].
    # Irreducible wedge: 0 <= kx <= ky, kx + ky <= pi.
    # Wait, the boundary is M-X-M.
    # Gamma (0,0), X (pi, 0), M (pi, pi).
    # The irreducible triangle is Gamma-X-M-Gamma? No, it's the triangle with vertices (0,0), (pi,0), (pi,pi).
    # The condition is 0 <= kx <= ky and kx <= pi, ky <= pi?
    # Actually, the symmetry group D4 maps (kx, ky) to (ky, kx), (-kx, ky), etc.
    # The fundamental domain is 0 <= kx <= ky, kx + ky <= pi? 
    # Let's check the corners.
    # (0,0) -> Gamma.
    # (pi, 0) -> X.
    # (pi, pi) -> M.
    # The line kx=ky is the boundary. The line kx+ky=pi? No.
    # The BZ boundary is kx=pi or ky=pi.
    # The irreducible wedge is 0 <= kx <= ky, kx <= pi, ky <= pi.
    # But we also need to consider the line connecting X(pi,0) and M(pi,pi)?
    # Actually, the standard irreducible BZ for square lattice is the triangle with vertices (0,0), (pi,0), (pi,pi).
    # Let's generate points in this triangle.
    
    num_k_points = 30
    k_points = []
    # We can just sample points in the triangle.
    # To be safe and cover the BZ well, let's use a grid and filter.
    # Or generate points along the high symmetry lines and interior.
    # Lines: Gamma-X (kx in [0, pi], ky=0)
    # X-M (kx=pi, ky in [0, pi])
    # M-Gamma (kx=ky in [0, pi])
    # Interior points.
    
    # Let's generate 10 points on each boundary and 10 interior.
    # Boundary 1: Gamma-X
    for i in range(10):
        kx = (i + 0.5) * (np.pi / 10)
        k_points.append((kx, 0))
    # Boundary 2: X-M
    for i in range(10):
        ky = (i + 0.5) * (np.pi / 10)
        k_points.append((np.pi, ky))
    # Boundary 3: M-Gamma
    for i in range(10):
        t = (i + 0.5) * (np.pi / 10)
        k_points.append((t, t))
    
    k_points = np.array(k_points)
    
    # Precompute (k+G)^2 for all k and G
    # (k + 2pi * G)^2
    # k is in radians (since a=1, b=2pi).
    KG_sq = np.zeros((len(k_points), N_pw))
    for i, k in enumerate(k_points):
        kx, ky = k
        # G vectors are integers, so G_vec = 2pi * (Gx, Gy)
        # k + G_vec = (kx + 2pi*Gx, ky + 2pi*Gy)
        kgx = kx + 2 * np.pi * Gx_flat
        kgy = ky + 2 * np.pi * Gy_flat
        KG_sq[i] = kgx**2 + kgy**2
    
    # Invert sqrt for the diagonal part of the Hamiltonian
    # H_{GG'} = mu_{G-G'} / sqrt((k+G)^2 (k+G')^2)
    # Let D_k = 1/sqrt(KG_sq[k])
    # H_k = D_k[:, None] * M * D_k[None, :]
    # where M is the matrix of mu_{G-G'}
    
    # 2. Define the evaluation function
    def evaluate_cell(cell):
        """
        cell: 32x32 numpy array of 0/1.
        Returns gap-midgap ratio.
        """
        # Compute epsilon
        eps = np.where(cell == 1, eps_high, eps_low)
        mu = 1.0 / eps
        
        # Compute Fourier coefficients of mu
        # mu_{dx, dy} = (1/N^2) sum mu(i,j) exp(-i 2pi (dx i + dy j)/N)
        # This is the DFT of mu.
        # numpy.fft.fft2(mu) gives sum mu(i,j) exp(-i 2pi (u i + v j)/N)
        # So mu_fft[u, v] = sum ...
        # We need mu_{G-G'}.
        # Let G = (Gx, Gy). G-G' = (Gx-Gx', Gy-Gy').
        # The index in the FFT array for difference (dx, dy) is (dx mod N, dy mod N).
        # Since G ranges from -8 to 8, dx ranges from -16 to 16.
        # N=32.
        # We can compute the full FFT and then index it.
        
        mu_fft = np.fft.fft2(mu)
        
        # Construct M matrix (N_pw x N_pw)
        # M[a, b] = mu_{G_a - G_b}
        # G_a = (Gx_flat[a], Gy_flat[a])
        # diff_x = Gx_flat[a] - Gx_flat[b]
        # diff_y = Gy_flat[a] - Gy_flat[b]
        # idx_x = diff_x % N
        # idx_y = diff_y % N
        # M[a, b] = mu_fft[idx_x, idx_y] / N^2
        
        # We can vectorize this.
        # diff_x_matrix[a, b] = Gx_flat[a] - Gx_flat[b]
        diff_x = Gx_flat[:, None] - Gx_flat[None, :]
        diff_y = Gy_flat[:, None] - Gy_flat[None, :]
        
        idx_x = diff_x % size
        idx_y = diff_y % size
        
        M = mu_fft[idx_x, idx_y] / (size * size)
        
        # Now compute eigenvalues for each k
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for i in range(len(k_points)):
            D = 1.0 / np.sqrt(KG_sq[i])
            # H = D[:, None] * M * D[None, :]
            # We only need the smallest 5 eigenvalues.
            # H is Hermitian.
            # We can use np.linalg.eigvalsh.
            # To speed up, we can use scipy.sparse or just numpy.
            # 289x289 is small.
            H = D[:, None] * M * D[None, :]
            evals = np.linalg.eigvalsh(H)
            
            # evals are omega^2.
            # We want bands 4 and 5 (0-indexed 3 and 4).
            # Note: The first few eigenvalues might be very small (near zero) due to the k->0 limit?
            # No, (k+G)^2 is non-zero.
            # At Gamma (0,0), G=(0,0) gives (2pi*0)^2 = 0.
            # Wait, if k=0 and G=0, (k+G)^2 = 0.
            # The term 1/sqrt(0) is infinite.
            # This is a singularity.
            # In the plane wave method, the G=0 component at k=0 is handled carefully.
            # Usually, the matrix element for G=G'=0 at k=0 is 0?
            # Let's check the limit.
            # The eigenvalue equation is sum_G' (1/eps)_{G-G'} (k+G')^2 E_{G'} = omega^2 (k+G)^2 E_G.
            # Or (1/sqrt((k+G)^2)) sum (1/eps)_{G-G'} (1/sqrt((k+G')^2)) E_{G'} = omega E_G.
            # If k=0, G=0, the diagonal term is mu_0 * 0 / 0?
            # Actually, the standard formulation avoids the singularity by noting that the G=0 mode at k=0 is a constant field, which has omega=0 for TM?
            # For TM, the field is H_z.
            # At k=0, the lowest mode is a constant H_z, which corresponds to omega=0?
            # Let's check the physics.
            # In a photonic crystal, the TM gap is usually in the higher bands.
            # The first few bands are typically TE or low-frequency TM.
            # The problem asks for bands 4 and 5.
            # If there are zero modes, they might be counted.
            # However, the evaluator uses 289 plane waves.
            # Let's assume the standard implementation handles the k=0, G=0 case by setting the corresponding matrix element to 0 or a large number?
            # Actually, if (k+G)^2 = 0, the term is 0/0.
            # But physically, the constant mode is decoupled.
            # Let's look at the inspiration code.
            # It uses `B_inv_sqrt = 1.0 / np.sqrt(kg_sq)`.
            # If kg_sq is 0, this is inf.
            # Then `S = mu_diff * B_inv_sqrt[:, None] * B_inv_sqrt[None, :]`.
            # If one is inf and the other is finite, it's inf.
            # If both are inf, it's inf*inf?
            # This suggests the inspiration code might have issues at Gamma point if not handled.
            # However, the inspiration code scored 2.43, so it must work.
            # Perhaps the k-points generated do not include exactly (0,0)?
            # My k-points:
            # Gamma-X: kx in [0.05pi, ...], ky=0. So k is not (0,0).
            # X-M: kx=pi, ky in [...]. Not (0,0).
            # M-Gamma: kx=ky in [...]. Not (0,0).
            # So I avoid the exact Gamma point. This is a good practice to avoid singularity.
            
            w4 = np.sqrt(evals[3])
            w5 = np.sqrt(evals[4])
            
            if w4 > max_omega_4:
                max_omega_4 = w4
            if w5 < min_omega_5:
                min_omega_5 = w5
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap

    def symmetrize_cell(quadrant):
        """
        quadrant: 8x8 numpy array.
        Returns 32x32 cell.
        """
        cell = np.zeros((size, size), dtype=int)
        h = size // 2
        for i in range(h):
            for j in range(h):
                val = quadrant[i, j]
                cell[i, j] = val
                cell[i, size-1-j] = val
                cell[size-1-i, j] = val
                cell[size-1-i, size-1-j] = val
        return cell

    def get_quadrant(cell):
        return cell[:size//2, :size//2]

    # 3. Initialization
    # Start with the 4-rod design from inspiration 1, which scored 5.78.
    # Centers at (+/-0.25, +/-0.25) relative to cell center?
    # Cell coordinates: -0.5 to 0.5.
    # Centers: (0.25, 0.25), (-0.25, 0.25), (0.25, -0.25), (-0.25, -0.25).
    # Radius 0.15.
    
    def generate_seed_cell():
        cell = np.zeros((size, size), dtype=int)
        r = 0.15
        centers = [
            (0.25, 0.25),
            (-0.25, 0.25),
            (0.25, -0.25),
            (-0.25, -0.25)
        ]
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                for cx, cy in centers:
                    dx = x - cx
                    dy = y - cy
                    if dx*dx + dy*dy < r*r:
                        cell[i, j] = 1
                        break
        # Ensure symmetry
        # The seed is already symmetric.
        return cell

    current_cell = generate_seed_cell()
    current_score = evaluate_cell(current_cell)
    best_cell = current_cell.copy()
    best_score = current_score
    
    # 4. Local Search
    # We will flip pixels in the 8x8 quadrant.
    # To speed up evaluation, we can precompute the effect of a flip?
    # No, the FFT is global.
    # However, we can use a faster FFT?
    # Or just rely on the fact that 32x32 FFT is fast.
    
    # Let's estimate time.
    # FFT2 of 32x32 is ~10-20 microseconds?
    # Eigenvalue of 289x289 is ~1-5 ms?
    # 30 k-points * 5 ms = 150 ms per evaluation.
    # 120s budget / 0.15s = 800 evaluations.
    # This is reasonable.
    
    start_time = time.time()
    deadline = start_time + 100.0  # 20s safety margin
    
    step = 0
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        step += 1
        # Pick a random pixel in the 8x8 quadrant
        i = np.random.randint(0, size//2)
        j = np.random.randint(0, size//2)
        
        # Flip it
        old_val = current_cell[i, j]
        current_cell[i, j] = 1 - old_val
        
        # Symmetrize
        # We only changed the quadrant, so we need to update the full cell.
        # Instead of rebuilding the whole cell, we can just flip the symmetric counterparts.
        # But `current_cell` is the full cell.
        # Let's maintain the full cell.
        # Flip counterparts:
        current_cell[i, size-1-j] = 1 - current_cell[i, size-1-j]
        current_cell[size-1-i, j] = 1 - current_cell[size-1-i, j]
        current_cell[size-1-i, size-1-j] = 1 - current_cell[size-1-i, size-1-j]
        
        new_score = evaluate_cell(current_cell)
        
        if new_score > current_score:
            current_score = new_score
            no_improve_count = 0
            if new_score > best_score:
                best_score = new_score
                best_cell = current_cell.copy()
        else:
            # Revert
            current_cell[i, j] = old_val
            current_cell[i, size-1-j] = 1 - current_cell[i, size-1-j]
            current_cell[size-1-i, j] = 1 - current_cell[size-1-i, j]
            current_cell[size-1-i, size-1-j] = 1 - current_cell[size-1-i, size-1-j]
            no_improve_count += 1
            
            # If stuck, try a random restart or larger move?
            # Let's try a random restart from the best cell if stuck too long.
            if no_improve_count == max_no_improve:
                # Reset to best
                current_cell = best_cell.copy()
                current_score = best_score
                no_improve_count = 0
                # Maybe perturb?
                # Let's just break and rely on the best found.
                # Or do a few random flips to escape.
                for _ in range(10):
                    ri = np.random.randint(0, size//2)
                    rj = np.random.randint(0, size//2)
                    current_cell[ri, rj] = 1 - current_cell[ri, rj]
                    current_cell[ri, size-1-rj] = 1 - current_cell[ri, size-1-rj]
                    current_cell[size-1-ri, rj] = 1 - current_cell[size-1-ri, rj]
                    current_cell[size-1-ri, size-1-rj] = 1 - current_cell[size-1-ri, size-1-rj]
                
                # Re-evaluate
                current_score = evaluate_cell(current_cell)
                if current_score < best_score:
                    # If it got worse, go back to best
                    current_cell = best_cell.copy()
                    current_score = best_score
                no_improve_count = 0

    # Convert best_cell to list of lists of 0/1
    best_list = best_cell.astype(int).tolist()
    return best_list

# EVOLVE-BLOCK-END
