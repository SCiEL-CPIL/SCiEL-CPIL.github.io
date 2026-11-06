import numpy as np
import time
import math

def design(problem):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Initial guess: Centered rod
    # Try to initialize with a structure that likely has a gap or is close
    # The seed uses radius 0.2. Let's start there or slightly different.
    cell = np.zeros((size, size), dtype=np.float64)
    # Create a rod
    cx, cy = size // 2, size // 2
    # Use a radius that fills a reasonable fraction
    # Seed used 0.2 lattice constants. In pixels, that's 0.2 * 32 = 6.4 pixels radius?
    # The seed code: x = (i+0.5)/size - 0.5. x in [-0.5, 0.5].
    # x*x + y*y < 0.2**2 = 0.04.
    # This is a small rod.
    
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.2**2:
                cell[i, j] = 1.0
            else:
                cell[i, j] = 0.0
                
    # Symmetrize helper
    def symmetrize(c):
        # c is size x size
        # Invariant under square symmetry group about cell centre
        # cell == cell[::-1] row-wise and column-wise and cell == its transpose
        # This implies 4-fold rotational symmetry and reflection symmetry.
        # Average over the 8 symmetries to ensure strict invariance?
        # Or just project onto the symmetric subspace.
        c1 = c
        c2 = c[:, ::-1]
        c3 = c[::-1, :]
        c4 = c[::-1, ::-1]
        c5 = c.T
        c6 = c.T[:, ::-1]
        c7 = c.T[::-1, :]
        c8 = c.T[::-1, ::-1]
        avg = (c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8) / 8.0
        # Binarize
        return (avg > 0.5).astype(np.float64)

    cell = symmetrize(cell)
    
    # PWE Solver
    # Number of G vectors. 289 is used by evaluator. We can use fewer for speed.
    # Let's use a grid of G vectors.
    # G = (m, n) * 2pi.
    # We need enough to converge bands 4 and 5.
    # Let's try N_G = 15 (i.e., -7 to 7) -> 15x15 = 225 vectors.
    # Or N_G = 11 (-5 to 5) -> 121 vectors.
    # Let's start with 15x15 for accuracy, but might be slow.
    # 225x225 matrix eigenvalue solve.
    
    N_G = 15
    G_vecs = []
    for m in range(-N_G//2, N_G//2 + 1):
        for n in range(-N_G//2, N_G//2 + 1):
            G_vecs.append((2*math.pi*m, 2*math.pi*n))
    G_vecs = np.array(G_vecs)
    N_pws = len(G_vecs)
    
    # Precompute G-G' differences for epsilon coupling
    # We need epsilon_{G-G'}
    # Compute Fourier coefficients of epsilon from the pixel map
    def get_epsilon_fourier(cell_map):
        # cell_map is size x size, values 0 or 1 (dielectric fraction)
        # epsilon(x) = eps_low + (eps_high - eps_low) * cell_map(x)
        # We need Fourier coeffs of epsilon(x)
        # eps_G = 1/A * integral epsilon(x) e^{-i G.x} dA
        # Discrete:
        # eps_G = (1/N^2) sum_{i,j} epsilon(i,j) e^{-i G . r_{i,j}}
        # r_{i,j} = (i+0.5)/size - 0.5, (j+0.5)/size - 0.5
        
        eps_coeffs = np.zeros(N_pws, dtype=complex)
        # Precompute exponentials for all G and all pixels?
        # That's N_pws * size * size complex numbers.
        # 225 * 32 * 32 ~ 230k. Manageable.
        
        # Actually, we can compute this faster using FFT if we pad?
        # Or just direct sum.
        
        # Let's direct sum for simplicity and correctness.
        # r_i, r_j
        r = (np.arange(size) + 0.5) / size - 0.5
        R = np.meshgrid(r, r, indexing='ij')
        Rx = R[0]
        Ry = R[1]
        
        for k_idx, G in enumerate(G_vecs):
            Gx, Gy = G
            # exp(-i (Gx*Rx + Gy*Ry))
            phase = np.exp(-1j * (Gx * Rx + Gy * Ry))
            # epsilon(i,j)
            eps_val = eps_low + (eps_high - eps_low) * cell_map
            # Sum
            eps_coeffs[k_idx] = np.sum(eps_val * phase) / (size * size)
            
        return eps_coeffs

    eps_coeffs = get_epsilon_fourier(cell)
    
    # Build the B matrix (epsilon coupling)
    # B_{kk'} = eps_{G_k - G_k'}
    # We need to map difference vectors to index
    G_dict = {tuple(g): i for i, g in enumerate(G_vecs)}
    
    B = np.zeros((N_pws, N_pws), dtype=complex)
    for i, Gi in enumerate(G_vecs):
        for j, Gj in enumerate(G_vecs):
            diff = tuple(Gi - Gj)
            if diff in G_dict:
                idx = G_dict[diff]
                B[i, j] = eps_coeffs[idx]
            else:
                # If diff is outside our G set, we ignore it (truncation)
                # This is an approximation.
                pass
                
    # B should be Hermitian.
    B = (B + B.conj().T) / 2.0
    
    # K-points in irreducible BZ
    # Square lattice BZ is [-pi, pi] x [-pi, pi] (in units of 2pi/L, L=1)
    # Actually, G = 2pi * integer. So BZ is [-pi, pi].
    # Irreducible wedge: 0 <= kx <= ky <= pi? Or kx, ky >= 0, kx <= ky?
    # Standard: Gamma (0,0), X (pi, 0), M (pi, pi).
    # Path: Gamma -> X -> M -> Gamma.
    # We only need to find the global max omega_4 and min omega_5.
    # It's likely at M or X.
    # Let's sample a few points: Gamma, X, M, and mid-points.
    
    k_points = [
        (0.0, 0.0),      # Gamma
        (math.pi, 0.0),  # X
        (math.pi, math.pi), # M
        (math.pi/2, math.pi/2), # Center of X-M
        (math.pi/2, 0.0), # Center of Gamma-X
        (0.0, math.pi),  # Y (equivalent to X)
    ]
    
    def compute_bands(kx, ky, B_mat):
        # A is diagonal with (k+G)^2
        A_diag = np.zeros(N_pws)
        for i, G in enumerate(G_vecs):
            kx_g = kx + G[0]
            ky_g = ky + G[1]
            A_diag[i] = kx_g**2 + ky_g**2
            
        # Solve A v = w^2 B v
        # scipy.linalg.eigh(a, b)
        try:
            w2, v = np.linalg.eigh(B_mat, A_diag)
            # w2 are eigenvalues w^2
            # Sort them
            idx_sort = np.argsort(w2)
            w2 = w2[idx_sort]
            v = v[:, idx_sort]
            # Frequencies
            w = np.sqrt(np.maximum(w2, 0))
            return w, v
        except:
            return None, None

    def evaluate_gap(cell_map):
        # Compute eps coeffs
        eps_c = get_epsilon_fourier(cell_map)
        # Build B
        B_mat = np.zeros((N_pws, N_pws), dtype=complex)
        for i, Gi in enumerate(G_vecs):
            for j, Gj in enumerate(G_vecs):
                diff = tuple(Gi - Gj)
                if diff in G_dict:
                    idx = G_dict[diff]
                    B_mat[i, j] = eps_c[idx]
        B_mat = (B_mat + B_mat.conj().T) / 2.0
        
        max_w4 = -1
        min_w5 = 1e9
        best_k = None
        best_v4 = None
        best_v5 = None
        best_kx = 0
        best_ky = 0
        
        for kx, ky in k_points:
            w, v = compute_bands(kx, ky, B_mat)
            if w is None:
                continue
            # We need bands 4 and 5.
            # Index 3 and 4 (0-indexed)
            if len(w) >= 5:
                w4 = w[3]
                w5 = w[4]
                if w4 > max_w4:
                    max_w4 = w4
                    best_kx = kx
                    best_ky = ky
                    best_v4 = v[:, 3]
                if w5 < min_w5:
                    min_w5 = w5
                    # Note: v4 and v5 might be from different k-points if we track max/min separately.
                    # For gradient, we ideally want them at the same k-point if they are close.
                    # But for a first pass, let's just track the global max/min.
                    # If we want to be more accurate, we should find the k that maximizes the gap.
                    # Let's re-evaluate: find k that maximizes (w5 - w4).
        
        # Let's refine: find the k-point that gives the maximum gap
        max_gap = -1
        best_gap_k = None
        best_gap_v4 = None
        best_gap_v5 = None
        best_gap_w4 = 0
        best_gap_w5 = 0
        
        for kx, ky in k_points:
            w, v = compute_bands(kx, ky, B_mat)
            if w is None:
                continue
            if len(w) >= 5:
                w4 = w[3]
                w5 = w[4]
                gap = w5 - w4
                if gap > max_gap:
                    max_gap = gap
                    best_gap_k = (kx, ky)
                    best_gap_v4 = v[:, 3]
                    best_gap_v5 = v[:, 4]
                    best_gap_w4 = w4
                    best_gap_w5 = w5
        
        if max_gap <= 0:
            return 0.0, None, None, None, None # No gap
            
        ratio = 2 * max_gap / (best_gap_w5 + best_gap_w4)
        return ratio, best_gap_k, best_gap_v4, best_gap_v5, B_mat

    # Optimization Loop
    best_score = 0.0
    best_cell = cell.copy()
    
    # Compute initial score
    score, k_opt, v4, v5, B_mat = evaluate_gap(cell)
    best_score = score
    
    # Gradient update
    # We need field intensities |E|^2
    # E(r) = sum v_G exp(i (k+G) . r)
    # For TM, the relevant field is H_z. The energy density is proportional to |H_z|^2.
    
    def get_field_intensities(v_coeff, kx, ky):
        # v_coeff is vector of size N_pws
        # r grid
        r = (np.arange(size) + 0.5) / size - 0.5
        R = np.meshgrid(r, r, indexing='ij')
        Rx = R[0]
        Ry = R[1]
        
        field = np.zeros((size, size), dtype=complex)
        for i, G in enumerate(G_vecs):
            kx_g = kx + G[0]
            ky_g = ky + G[1]
            field += v_coeff[i] * np.exp(1j * (kx_g * Rx + ky_g * Ry))
            
        intensity = np.abs(field)**2
        # Normalize?
        # The magnitude depends on the normalization of v.
        # eigh returns normalized eigenvectors (v.T conj v = 1).
        # So intensity is just |field|^2.
        return intensity

    if k_opt is not None and v4 is not None and v5 is not None:
        kx, ky = k_opt
        int4 = get_field_intensities(v4, kx, ky)
        int5 = get_field_intensities(v5, kx, ky)
        
        # Update rule
        # We want to increase gap.
        # Increase epsilon where int5 is high (lowers w5)
        # Decrease epsilon where int4 is high (raises w4)
        # Update: delta_eps = alpha * (int5 - int4)
        # Then threshold.
        
        # Normalize intensities to [0, 1] to control step size
        max_int = max(np.max(int4), np.max(int5))
        if max_int > 0:
            int4 = int4 / max_int
            int5 = int5 / max_int
            
        # Several iterations
        alpha = 0.5
        for it in range(20):
            if time.time() > deadline:
                break
                
            # Gradient
            grad = int5 - int4
            
            # Update continuous epsilon
            cell_float = cell.copy()
            cell_float += alpha * grad
            
            # Project to binary and symmetrize
            # Simple threshold
            cell_new = (cell_float > 0.5).astype(np.float64)
            cell_new = symmetrize(cell_new)
            
            # Evaluate
            new_score, k_new, v4_new, v5_new, B_new = evaluate_gap(cell_new)
            
            if new_score > best_score:
                best_score = new_score
                best_cell = cell_new.copy()
                # Update intensities for next step if we want to follow the new gradient
                if k_new is not None and v4_new is not None and v5_new is not None:
                    kx, ky = k_new
                    int4 = get_field_intensities(v4_new, kx, ky)
                    int5 = get_field_intensities(v5_new, kx, ky)
                    max_int = max(np.max(int4), np.max(int5))
                    if max_int > 0:
                        int4 = int4 / max_int
                        int5 = int5 / max_int
            else:
                # If score didn't improve, try smaller step or different direction?
                # For now, just keep best.
                pass
            
            # If we have a good score, we can stop early?
            # The seed is 1.0. If we get > 2.0, that's good.
            if best_score > 3.0:
                break

    # Return best cell as nested list of 0/1
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(best_cell[i, j]))
        result.append(row)
        
    return result
