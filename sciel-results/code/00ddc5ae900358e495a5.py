import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)

    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 230s kill

    # --- Fast TM band solver using plane-wave expansion ---
    # Use a circular cutoff for plane waves for better accuracy
    N_PW = 12  # cutoff radius in G-space
    
    def make_g_pairs(n_pw):
        """Generate G vectors within a circular cutoff."""
        pairs = []
        for m in range(-n_pw, n_pw + 1):
            for n in range(-n_pw, n_pw + 1):
                if m * m + n * n <= n_pw * n_pw:
                    pairs.append((m, n))
        return pairs

    g_pairs = make_g_pairs(N_PW)
    n_pw_actual = len(g_pairs)
    
    # Precompute G vector array
    G_arr = np.array(g_pairs, dtype=float)
    Gx_arr = G_arr[:, 0]
    Gy_arr = G_arr[:, 1]
    
    # Precompute difference indices for FFT lookup
    # diff_m = Gx[i] - Gx[j], diff_n = Gy[i] - Gy[j]
    diff_m_mat = Gx_arr[:, None] - Gx_arr[None, :]
    diff_n_mat = Gy_arr[:, None] - Gy_arr[None, :]
    
    # Indices into the FFT array (size x size)
    # FFT index for (m, n) is (m % size, n % size)
    idx_m = (diff_m_mat + size) % size
    idx_n = (diff_n_mat + size) % size

    # K points in irreducible BZ
    # The irreducible BZ of a square lattice is the triangle (0,0)-(0.5,0)-(0.5,0.5)
    # We sample points along the edges and interior.
    k_points = []
    # Gamma to X (0,0) to (0.5, 0)
    for i in range(10):
        t = i / 9.0
        k_points.append((t * 0.5, 0.0))
    # X to M (0.5, 0) to (0.5, 0.5)
    for i in range(1, 10):
        t = i / 9.0
        k_points.append((0.5, t * 0.5))
    # M to Gamma (0.5, 0.5) to (0, 0)
    for i in range(1, 10):
        t = i / 9.0
        k_points.append((0.5 * (1 - t), 0.5 * (1 - t)))
    # Some interior points for better coverage
    k_points.append((0.25, 0.1))
    k_points.append((0.35, 0.2))
    k_points.append((0.2, 0.2))
    k_points.append((0.4, 0.3))
    
    k_points = np.array(k_points)
    n_k = len(k_points)

    def compute_fft_inv_eps(cell):
        """Compute 2D FFT of 1/epsilon."""
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        fft_2d = np.fft.fft2(inv_eps) / (size * size)
        return fft_2d

    def evaluate(cell):
        """
        Evaluate the gap-midgap ratio for bands 4 and 5.
        """
        fft_2d = compute_fft_inv_eps(cell)
        
        # Precompute the matrix of eps_inv coefficients for all G diffs
        # shape: (n_pw_actual, n_pw_actual)
        eps_inv_coeffs = fft_2d[idx_m, idx_n]
        
        max_omega4 = -1e10
        min_omega5 = 1e10
        
        for kx, ky in k_points:
            # |k - G|^2 for each G
            kx_g = kx - Gx_arr
            ky_g = ky - Gy_arr
            k_sq = kx_g * kx_g + ky_g * ky_g
            
            # H[i, j] = eps_inv[G_i - G_j] * |k - G_j|^2
            # Note: In the standard plane wave method for TM, the eigenvalue equation is:
            # sum_G' eps_inv[G - G'] |k - G'|^2 E(G') = omega^2 E(G)
            # So the matrix is M[i, j] = eps_inv[G_i - G_j] * |k - G_j|^2
            # This matrix is not Hermitian in general, but its eigenvalues are real and positive.
            # To use eigvalsh, we need a Hermitian matrix.
            # The matrix M is similar to a Hermitian matrix.
            # Alternatively, we can just use eigvals and take real parts, or symmetrize.
            # Symmetrization: H = (M + M^H) / 2 might not be correct if M is not normal.
            # Let's check the structure.
            # M[i,j] = c_{i-j} * w_j
            # M^H[i,j] = conj(c_{j-i}) * w_i
            # Since c is real and symmetric (c_m = c_-m), c_{i-j} = c_{j-i}.
            # So M^H[i,j] = c_{i-j} * w_i
            # The geometric mean is sqrt(w_i w_j) c_{i-j}
            # Let D be a diagonal matrix with D[i,i] = sqrt(w_i)
            # Then D^{-1} M D = [ sqrt(w_i)^{-1} * c_{i-j} * w_j * sqrt(w_j) ] = c_{i-j} sqrt(w_i w_j)
            # This matrix is Hermitian (and real symmetric).
            # So we can compute H = D^{-1} M D and use eigvalsh.
            # But we need the eigenvalues of M, which are the same as H.
            
            # Let's construct the Hermitian matrix H directly.
            # H[i, j] = c_{i-j} * sqrt(w_i * w_j)
            sqrt_w = np.sqrt(k_sq)
            # Outer product of sqrt_w
            sqrt_w_outer = sqrt_w[:, None] * sqrt_w[None, :]
            
            H = eps_inv_coeffs * sqrt_w_outer
            
            # Ensure symmetry (should be real and symmetric)
            H = (H + H.T) / 2.0
            
            try:
                eigenvalues = np.linalg.eigvalsh(H)
                eigenvalues = np.sort(eigenvalues)
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

    # --- Initial Cell Generation ---
    # Start with a 4-rod structure, which is a common candidate for large band gaps.
    # We will optimize the shape and position.
    
    def make_cell_4rod(radius, n_order, offset, size=32):
        """Create a symmetric 4-rod cell using superellipse boundaries."""
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
                    # Superellipse: |dx/r|^n + |dy/r|^n < 1
                    if radius > 0:
                        val = abs(dx / radius) ** n_order + abs(dy / radius) ** n_order
                        if val < 1.0:
                            cell[i, j] = 1
                            break
        return cell

    # --- Search Strategy ---
    # 1. Coarse search over superellipse parameters
    # 2. Local search with pixel flips
    
    best_score = -1e10
    best_cell = None
    
    # Coarse search
    radii = [3.0, 4.0, 5.0, 6.0, 7.0]
    orders = [2.0, 3.0, 4.0]
    offsets = [-1.0, 0.0, 1.0]
    
    for r in radii:
        for n_ord in orders:
            for off in offsets:
                if time.time() > deadline - 60:
                    break
                cell = make_cell_4rod(r, n_ord, off)
                score = evaluate(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell.copy()
        if time.time() > deadline - 60:
            break
    
    if best_cell is None:
        # Fallback
        best_cell = make_cell_4rod(5.0, 2.0, 0.0)
        best_score = evaluate(best_cell)

    # Local search
    def get_orbit(i, j):
        N = size
        points = [
            (i, j),
            (j, N-1-i),
            (N-1-i, N-1-j),
            (N-1-j, i),
            (i, N-1-j),
            (N-1-i, j),
            (j, i),
            (N-1-j, N-1-i)
        ]
        return list(set(points))

    current_cell = best_cell.copy()
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline - 10:
        # Pick random pixel
        i = np.random.randint(size)
        j = np.random.randint(size)
        
        orbit = get_orbit(i, j)
        
        # Flip
        for (r, c) in orbit:
            current_cell[r, c] = 1 - current_cell[r, c]
            
        score = evaluate(current_cell)
        
        if score > best_score:
            best_score = score
            best_cell = current_cell.copy()
            no_improve_count = 0
        else:
            # Revert
            for (r, c) in orbit:
                current_cell[r, c] = 1 - current_cell[r, c]
            no_improve_count += 1
            
        # Restart if stuck
        if no_improve_count > max_no_improve:
            current_cell = best_cell.copy()
            # Perturb
            for _ in range(5):
                ri = np.random.randint(size)
                rj = np.random.randint(size)
                orbit = get_orbit(ri, rj)
                for (r, c) in orbit:
                    current_cell[r, c] = 1 - current_cell[r, c]
            no_improve_count = 0

    # Return best_cell as nested list
    return best_cell.tolist()
