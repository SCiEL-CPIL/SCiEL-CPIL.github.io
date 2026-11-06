import numpy as np
import time
import math

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 210.0  # Safety margin for 230s limit

    # --- Rigorous Plane-Wave Solver ---
    # The evaluator uses 289 plane waves (17x17 grid).
    # We must match this accuracy to get a reliable score.
    # Matrix A = Eps^{-1} D.
    # D is diagonal: D_{k,k} = |k+G_k|^2.
    # Eps is a circulant matrix of size N_G x N_G, where N_G = 289.
    # Eps_{k,l} = (1/N_pix) * sum_{x,y} eps(x,y) exp(-i (G_k - G_l) . r_xy)
    # Since eps is a sum of delta functions (pixels), Eps_{k,l} = (1/N_pix^2) * sum_{pixels} exp(-i (G_k - G_l) . r)
    
    half_pw = 8
    G_vecs = []
    for m in range(-half_pw, half_pw + 1):
        for n in range(-half_pw, half_pw + 1):
            G_vecs.append((m * 2.0 * np.pi, n * 2.0 * np.pi))
    G_vecs = np.array(G_vecs)
    nG = len(G_vecs)
    
    # Precompute pixel positions
    pixel_coords = np.array([(i, j) for i in range(size) for j in range(size)])
    pixel_coords = pixel_coords.astype(float) / size # Normalized to [0,1) for dimensionless k
    
    # k-points for evaluation
    # The evaluator uses 30 points. We use a dense set to be safe.
    k_points = []
    # Triangle (0,0) to (pi,0) to (pi,pi)
    n_k = 10
    for i in range(n_k + 1):
        for j in range(n_k + 1):
            if i + j <= n_k:
                k_points.append((i / n_k * np.pi, j / n_k * np.pi))
    k_points = np.array(k_points)
    
    def compute_score(cell):
        """
        Compute the 4-5 TM band gap score.
        cell: 32x32 array of 0/1.
        """
        # 1. Construct Eps matrix (289x289)
        # Eps_{k,l} = (1/N_pix^2) * sum_{(x,y) in pixels} eps(x,y) exp(-i (G_k - G_l) . r_xy)
        # Let Delta_G = G_k - G_l.
        # We can compute this efficiently.
        
        # Actually, for TM polarization, the eigenvalue problem is:
        # (1/eps) * |k+G|^2 * E = omega^2 * E
        # This is a generalized eigenvalue problem or a standard one if we transform.
        # Standard form: A E = omega^2 E, where A = Eps^{-1} D.
        # We need to invert Eps. Eps is 289x289. Inversion is O(N^3) ~ 24 million ops. Very fast.
        
        # Let's build Eps.
        # Eps is circulant? No, the lattice is discrete pixels, but the periodicity is the cell.
        # The operator is periodic with the cell.
        # Eps_{k,l} depends on G_k - G_l.
        
        # Efficient construction:
        # For each unique difference vector d = G_k - G_l, compute the sum.
        # But simpler: 
        # Eps = F^H * diag(eps_F) * F ? No.
        
        # Let's use the explicit sum for accuracy and simplicity, optimized with numpy.
        # N_pix = 1024. N_G = 289.
        # Sum over 1024 pixels for each of 289*289 = 83k pairs? 83k * 1024 = 85M exp ops.
        # Might be slow in pure python loop. 
        # Vectorized:
        # G_diff = G_vecs[:, None, :] - G_vecs[None, :, :] # 289 x 289 x 2
        # phase = -1j * (G_diff[..., 0] * pixel_coords[None, None, :, 0] + G_diff[..., 1] * pixel_coords[None, None, :, 1])
        # Sum over pixel axis.
        
        # Memory: 289*289*1024*2 floats ~ 400MB. Too much?
        # Let's do it in chunks or use the FFT property if possible.
        # Actually, eps(x,y) is a grid. The Fourier coefficients of the *continuous* periodic extension
        # of the pixelated dielectric are what we need.
        
        # Let's stick to the definition:
        # Eps_{k,l} = (1/N^2) sum_{x,y} eps(x,y) exp(-i (G_k - G_l) . r_{xy})
        
        # We can precompute the "shape function" S(d) = (1/N^2) sum_{x,y} eps(x,y) exp(-i d . r_{xy})
        # where d runs over all possible G_k - G_l.
        # The possible differences are (2m, 2n) * pi for m,n in [-16, 16].
        # This is a 33x33 grid of differences.
        
        # Let's compute S for all 33x33 differences.
        # diff_m in -16..16, diff_n in -16..16.
        # d_x = 2 * diff_m * pi, d_y = 2 * diff_n * pi.
        
        # S[dm, dn] = (1/N^2) sum_{x,y} eps(x,y) exp(-i (2*dm*pi*x/N + 2*dn*pi*y/N))
        # This is just the 2D FFT of the eps grid!
        # S is the DFT of the eps grid.
        
        eps_grid = eps_low + (eps_high - eps_low) * np.array(cell, dtype=float)
        
        # DFT of eps grid.
        # Note: standard FFT uses exp(-2pi i (k x / N)).
        # Our G vectors are 2 pi m / L. If L=N (in units of pixel size), then G = 2 pi m.
        # r_xy = (x/N, y/N) * L ? No, let's keep dimensionless.
        # Let lattice constant a=1. Pixel size a/N.
        # G = 2 pi (m, n).
        # r = (x/N, y/N).
        # G . r = 2 pi (m x/N + n y/N).
        # This matches the FFT definition exactly.
        
        eps_F = np.fft.fft2(eps_grid) / (size * size)
        # eps_F[m, n] corresponds to G = 2 pi (m, n).
        # Indices: m, n in -N/2 .. N/2-1.
        # Our G_vecs are m, n in -8..8.
        
        # We need Eps_{k,l} = eps_F[ (G_k - G_l) / (2pi) ]
        # Let's map G_vecs indices to FFT indices.
        # G_vecs[i] = (2 pi m_i, 2 pi n_i).
        # Diff = (2 pi (m_i - m_j), 2 pi (n_i - n_j)).
        # FFT index for this diff: (m_i - m_j, n_i - n_j).
        # We need to wrap these into the FFT range [-16, 15].
        
        # Construct Eps matrix
        Eps = np.zeros((nG, nG), dtype=complex)
        for i in range(nG):
            for j in range(nG):
                dm = G_vecs[i, 0] / (2.0 * np.pi) - G_vecs[j, 0] / (2.0 * np.pi)
                dn = G_vecs[i, 1] / (2.0 * np.pi) - G_vecs[j, 1] / (2.0 * np.pi)
                # Wrap to [-16, 15]
                idx_m = int(round(dm)) % size
                idx_n = int(round(dn)) % size
                Eps[i, j] = eps_F[idx_m, idx_n]
                
        # Invert Eps
        try:
            inv_Eps = np.linalg.inv(Eps)
        except np.linalg.LinAlgError:
            return -1.0
            
        # Compute score
        min_omega5 = np.inf
        max_omega4 = 0.0
        
        for kx, ky in k_points:
            kG = np.stack([kx + G_vecs[:, 0], ky + G_vecs[:, 1]], axis=1)
            kG2 = kG[:, 0]**2 + kG[:, 1]**2
            
            # A = inv_Eps * D (element-wise multiplication of columns? No, D is diagonal)
            # A_{i,j} = inv_Eps_{i,j} * kG2_j
            A = inv_Eps * kG2[None, :]
            
            # Solve eigenvalue problem A v = lambda v
            # lambda = omega^2
            try:
                vals = np.linalg.eigvalsh(A) # Hermitian? 
                # Is A Hermitian?
                # Eps is Hermitian (real eps -> Eps_{i,j} = conj(Eps_{j,i})).
                # inv_Eps is Hermitian.
                # D is real diagonal.
                # A = inv_Eps * D. 
                # (A^H)_{i,j} = (A_{j,i})^* = (inv_Eps_{j,i} * kG2_i)^* = inv_Eps_{i,j}^* * kG2_i = inv_Eps_{i,j} * kG2_i.
                # A_{i,j} = inv_Eps_{i,j} * kG2_j.
                # So A is NOT Hermitian unless kG2 is constant.
                # We need general eigensolver.
                vals = np.linalg.eigvals(A)
                vals = np.sort(vals.real)
            except Exception:
                return -1.0
                
            if len(vals) < 5:
                return -1.0
                
            omega4 = np.sqrt(max(0, vals[3]))
            omega5 = np.sqrt(max(0, vals[4]))
            
            if omega5 < min_omega5:
                min_omega5 = omega5
            if omega4 > max_omega4:
                max_omega4 = omega4
                
        if min_omega5 == np.inf or max_omega4 == 0.0:
            return -1.0
            
        if min_omega5 <= max_omega4:
            return -1.0
            
        return 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)

    # --- Design Generation: Superelliptical Rods ---
    # We will sweep parameters for a 4-rod design with superelliptical cross-sections.
    # Rods at (8,8), (23,8), (8,23), (23,23).
    # Superellipse: |dx/a|^n + |dy/b|^n <= 1
    
    def generate_cell(radius, exponent, center_offset=0.0):
        cell = np.zeros((size, size), dtype=int)
        centers = [(8, 8), (23, 8), (8, 23), (23, 23)]
        
        for i in range(size):
            for j in range(size):
                for cx, cy in centers:
                    dx = (j - cx) / radius
                    dy = (i - cy) / radius
                    # Superellipse equation
                    if abs(dx)**exponent + abs(dy)**exponent <= 1.0:
                        cell[i, j] = 1
                        break
        return cell

    # Parameter Sweep
    best_score = -1.0
    best_cell = None
    
    radii = np.linspace(3.0, 6.0, 10)
    exponents = np.linspace(1.0, 4.0, 10)
    
    # Coarse sweep
    for r in radii:
        for e in exponents:
            if time.time() > deadline - 30:
                break
            cell = generate_cell(r, e)
            score = compute_score(cell)
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
                
    # Fine sweep around best
    if best_cell is not None:
        # Find which r and e gave the best score
        # We didn't store them, so let's just do a local refinement
        # by checking neighbors of the best parameters found in the coarse grid?
        # Actually, let's just run a second pass with finer steps over the full range if time permits.
        
        remaining_time = deadline - time.time() - 10
        if remaining_time > 20:
            fine_radii = np.linspace(3.0, 6.0, 20)
            fine_exponents = np.linspace(1.0, 4.0, 20)
            
            for r in fine_radii:
                for e in fine_exponents:
                    if time.time() > deadline - 10:
                        break
                    cell = generate_cell(r, e)
                    score = compute_score(cell)
                    if score > best_score:
                        best_score = score
                        best_cell = cell.copy()

    # Final pixel-level refinement (optional, if time)
    if best_cell is not None and time.time() < deadline - 5:
        # Simple local search on pixels
        # But we must maintain symmetry.
        # Given time constraints and the robustness of the superellipse sweep,
        # we might skip this to avoid breaking symmetry or running out of time.
        # The superellipse design is likely near-optimal for this class.
        
        pass

    if best_cell is None:
        # Fallback
        best_cell = generate_cell(4.5, 2.0)

    return best_cell.tolist()
