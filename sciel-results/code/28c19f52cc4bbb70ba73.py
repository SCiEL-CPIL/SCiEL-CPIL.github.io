import numpy as np
import time

def design(problem):
    start_time = time.time()
    deadline = start_time + 220.0 # Leave 10s margin for 230s limit
    
    size = problem["pixels"] # 32
    eps_low = problem["eps_low"] # 1
    eps_high = problem["eps_high"] # 12
    
    # Pre-define the plane wave grid
    # 289 plane waves -> 17x17 grid
    gw = np.arange(-8, 9)
    Gx, Gy = np.meshgrid(gw, gw, indexing='ij')
    Gx = Gx.flatten()
    Gy = Gy.flatten()
    # G vectors are 2*pi * G / a. Let a=1. So G = 2*pi * (Gx, Gy)
    # We can work in units of 2*pi/a. k is also in these units.
    # |k+G|^2 = (kx+Gx)^2 + (ky+Gy)^2
    
    # Define k-points in the irreducible Brillouin zone
    # Gamma (0,0), X (1,0), M (1,1) in units of 2*pi/a
    k_points = []
    # Gamma to X
    for t in np.linspace(0, 1, 10):
        k_points.append((t, 0))
    # X to M
    for t in np.linspace(0, 1, 10):
        k_points.append((1, t))
    # Gamma to M
    for t in np.linspace(0, 1, 10):
        k_points.append((t, t))
    # Remove duplicates if any (none expected with this setup)
    
    best_score = -1.0
    best_cell = None
    
    # Search parameters
    # d: distance of rod centers from cell center (15.5, 15.5)
    # r: radius of rods
    # mode: 0 for rods in air (1 in air), 1 for holes in dielectric (0 in dielectric)
    
    # We will search a grid
    d_values = np.linspace(2.0, 14.0, 15) # Distance from center
    r_values = np.linspace(1.0, 10.0, 15) # Radius
    
    # Note: Centers are at (15.5 +/- d, 15.5 +/- d)
    # If d is small, rods overlap. If d is large, they are far.
    # Max d is 15.5 (edge).
    
    for mode in [0, 1]:
        for d in d_values:
            for r in r_values:
                if time.time() > deadline:
                    break
                
                # Construct cell
                # Centers
                cx1, cy1 = 15.5 - d, 15.5 - d
                cx2, cy2 = 15.5 - d, 15.5 + d
                cx3, cy3 = 15.5 + d, 15.5 - d
                cx4, cy4 = 15.5 + d, 15.5 + d
                
                cell = np.zeros((size, size), dtype=int)
                
                # Vectorized distance calculation
                i_idx, j_idx = np.indices((size, size))
                # Pixel centers are at (i+0.5, j+0.5)
                xi = i_idx + 0.5
                yj = j_idx + 0.5
                
                dist1 = (xi - cx1)**2 + (yj - cy1)**2
                dist2 = (xi - cx2)**2 + (yj - cy2)**2
                dist3 = (xi - cx3)**2 + (yj - cy3)**2
                dist4 = (xi - cx4)**2 + (yj - cy4)**2
                
                mask = (dist1 < r**2) | (dist2 < r**2) | (dist3 < r**2) | (dist4 < r**2)
                
                if mode == 0:
                    # Rods in air: 1 is dielectric
                    cell[mask] = 1
                else:
                    # Holes in dielectric: 0 is air, 1 is dielectric
                    # Background is dielectric (1), holes are air (0)
                    cell[:] = 1
                    cell[mask] = 0
                
                # Evaluate this cell
                # 1. Compute 1/epsilon array
                eps = np.where(cell == 1, eps_high, eps_low)
                inv_eps = 1.0 / eps
                
                # 2. Compute FFT
                # We need coefficients c_G = (1/N^2) sum V_ij exp(-i 2 pi (Gx i + Gy j) / N)
                # numpy.fft.fft2 computes sum V_ij exp(-i 2 pi (n i + m j) / N)
                # So fft2(inv_eps) gives the unnormalized sum.
                fft_inv_eps = np.fft.fft2(inv_eps)
                c_G = fft_inv_eps / (size * size)
                
                # 3. Extract coefficients for our G grid
                # Gx, Gy are integers from -8 to 8.
                # In FFT output, index [i, j] corresponds to frequency i (if i < N/2) or i-N (if i >= N/2).
                # We need to map our Gx, Gy to FFT indices.
                # Gx in [-8, 8].
                # FFT index for Gx:
                # If Gx >= 0: idx = Gx
                # If Gx < 0: idx = Gx + 32
                # Same for Gy.
                
                c_coeffs = np.zeros(289, dtype=complex)
                for idx in range(289):
                    gx = Gx[idx]
                    gy = Gy[idx]
                    fx = gx % size
                    fy = gy % size
                    c_coeffs[idx] = c_G[fx, fy]
                
                # 4. Compute bands
                min_omega5 = np.inf
                max_omega4 = -np.inf
                
                for kx, ky in k_points:
                    # Build matrix M
                    # M_{GG'} = |k+G|^2 * c_{G-G'}
                    # k is (kx, ky) in units of 2pi/a.
                    # G is (Gx, Gy) in units of 2pi/a.
                    
                    # |k+G|^2 for each G
                    kgx = kx + Gx
                    kgy = ky + Gy
                    kg_sq = kgx**2 + kgy**2
                    
                    # We need c_{G-G'} for each pair (G, G')
                    # This is a convolution.
                    # M = kg_sq[:, None] * C
                    # Where C_{i, j} = c_{G_i - G_j}
                    
                    # We can compute C by taking the inverse FFT of the coefficients?
                    # Or just loop. 289x289 = 83k operations. Fast.
                    C_mat = np.zeros((289, 289), dtype=complex)
                    for i in range(289):
                        for j in range(289):
                            # G_i - G_j
                            dgx = Gx[i] - Gx[j]
                            dgy = Gy[i] - Gy[j]
                            # Find coefficient c_{dgx, dgy}
                            # We have c_coeffs for G in [-8, 8].
                            # If dgx is outside [-8, 8], c is 0 (truncation).
                            if -8 <= dgx <= 8 and -8 <= dgy <= 8:
                                # Find index of (dgx, dgy) in Gx, Gy
                                # Since Gx, Gy are unique integers, we can use a map
                                # Or just use the fact that they are on a grid
                                # idx_g = (dgx + 8) * 17 + (dgy + 8)
                                idx_c = (dgx + 8) * 17 + (dgy + 8)
                                C_mat[i, j] = c_coeffs[idx_c]
                            else:
                                C_mat[i, j] = 0.0
                    
                    M = kg_sq[:, None] * C_mat
                    
                    # Solve eigenvalues
                    # M is Hermitian?
                    # c_G = c_{-G}^* for real V.
                    # M_{ij} = kg_sq_i c_{G_i - G_j}
                    # M_{ji} = kg_sq_j c_{G_j - G_i}
                    # Since kg_sq_i != kg_sq_j generally, M is not Hermitian.
                    # Wait. The equation is:
                    # sum_G' |k+G|^2 c_{G-G'} H_{G'} = k^2 H_G
                    # This is a generalized eigenvalue problem?
                    # No, it's a standard eigenvalue problem if we rearrange?
                    # Usually, the matrix is formed as:
                    # A_{GG'} = |k+G|^2 c_{G-G'}
                    # det(A - k^2 I) = 0
                    # Is A Hermitian?
                    # A_{GG'} = |k+G|^2 c_{G-G'}
                    # A_{G'G} = |k+G'|^2 c_{G'-G}
                    # c_{G'-G} = c_{-(G-G')} = c_{G-G'}^*
                    # So A_{G'G} = |k+G'|^2 c_{G-G'}^*
                    # A_{GG'}^* = |k+G|^2 c_{G-G'}^*
                    # They are not equal unless |k+G| = |k+G'|.
                    # So the matrix is NOT Hermitian.
                    # We must use `numpy.linalg.eigvals` not `eigvalsh`.
                    
                    eigvals = np.linalg.eigvals(M)
                    # Filter real positive values
                    eigvals = np.real(eigvals)
                    eigvals = eigvals[eigvals > 0]
                    eigvals.sort()
                    
                    if len(eigvals) >= 5:
                        omega4 = np.sqrt(eigvals[3])
                        omega5 = np.sqrt(eigvals[4])
                        
                        if omega5 > 0:
                            max_omega4 = max(max_omega4, omega4)
                            min_omega5 = min(min_omega5, omega5)
                
                if min_omega5 < np.inf and max_omega4 > -np.inf:
                    gap = min_omega5 - max_omega4
                    midgap = (min_omega5 + max_omega4) / 2.0
                    if midgap > 0:
                        ratio = 2.0 * gap / (min_omega5 + max_omega4)
                        if ratio > best_score:
                            best_score = ratio
                            best_cell = cell.copy()
                            # print(f"New best: {ratio:.4f}, d={d:.2f}, r={r:.2f}, mode={mode}")

    if best_cell is None:
        # Fallback: single rod
        cell = np.zeros((size, size), dtype=int)
        i_idx, j_idx = np.indices((size, size))
        xi = i_idx + 0.5 - 15.5
        yj = j_idx + 0.5 - 15.5
        cell[xi**2 + yj**2 < 6.4**2] = 1
        return cell.tolist()
        
    return best_cell.tolist()
