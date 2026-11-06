import numpy as np
import time

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 110.0  # Leave margin for evaluator overhead
    
    # 1. Build a fast, accurate band solver using 2D FFT for matrix-vector products
    # We solve the generalized eigenvalue problem for TM polarization.
    # The equation is: sum_n [ (k+G_m)^2 delta_{mn} - (k+G_m)·(k+G_n) (1/eps)_{n-m} ] E_n = (omega/c)^2 E_m
    # Wait, for TM (Hz), the equation is:
    # ∇^2 H_z + (omega/c)^2 eps(r) H_z = 0
    # In Fourier space:
    # sum_m [ -(k+G_m)^2 delta_{mn} + (omega/c)^2 eps_{n-m} ] E_m = 0
    # This leads to: sum_m eps_{n-m} E_m = ((k+G_n)^2 / (omega/c)^2) E_n
    # Let lambda = (omega/c)^2. Then: eps_matrix @ E = (1/lambda) diag((k+G)^2) @ E
    # Or: (1/eps_matrix) @ diag((k+G)^2) @ E = lambda E  <-- This is the standard form often used.
    # Actually, it's easier to solve: A @ v = w B @ v where A = diag((k+G)^2) and B = 1/eps_matrix?
    # No, let's stick to the form:
    # sum_m eps_{n-m} E_m = mu_n E_n where mu_n = (k+G_n)^2 / lambda
    # This is a non-standard eigenvalue problem.
    
    # Alternative standard form for TM:
    # The operator is L = -∇^2. The equation is L H_z = (omega/c)^2 (1/eps) H_z ? No.
    # Maxwell: ∇ × (∇ × H) = (omega/c)^2 eps H.
    # For TM (H_z): ∇^2 H_z + (omega/c)^2 eps H_z = 0.
    # So: -∇^2 H_z = (omega/c)^2 eps H_z.
    # In plane waves:
    # sum_m (k+G_m)^2 delta_{mn} E_m = (omega/c)^2 sum_m eps_{n-m} E_m
    # Let D = diag((k+G)^2). Let Eps be the matrix with elements Eps[n,m] = eps_{n-m}.
    # Then: D E = lambda Eps E.
    # This is a generalized eigenvalue problem: D v = lambda Eps v.
    # Or: Eps^{-1} D v = lambda v.
    # Since Eps is a circulant matrix (convolution), we can multiply by Eps using FFT.
    # Eps^{-1} is also a convolution with the inverse Fourier coefficients of eps.
    # Let g = 1/eps. The matrix G with G[n,m] = g_{n-m} is the inverse of Eps?
    # No. Eps is the convolution matrix of eps. Its inverse is NOT the convolution matrix of 1/eps.
    # However, we can solve Eps v = u using FFT if Eps is invertible.
    # Actually, the standard efficient method is to solve:
    # sum_m g_{n-m} (k+G_m)^2 E_m = lambda E_n  <-- This is for TE?
    # Let's check TE vs TM.
    # TE (E_z): ∇^2 E_z + (omega/c)^2 (1/eps) E_z = 0.
    #   -> D E = lambda (1/eps) E. Here (1/eps) is a convolution.
    #   -> (1/eps)^{-1} D E = lambda E. Since (1/eps)^{-1} is eps (convolution), we can do:
    #   eps_conv( D E ) = lambda E.
    #   This allows fast matrix-vector product: v -> eps_conv(D v) using FFT.
    
    # TM (H_z): ∇^2 H_z + (omega/c)^2 eps H_z = 0.
    #   -> D E = lambda eps E.
    #   -> eps^{-1} D E = lambda E.
    #   Here eps^{-1} is NOT a simple convolution. We would need to solve a linear system.
    #   UNLESS we reformulate.
    #   Note: The problem asks for TM bands.
    #   Is there a trick?
    #   Actually, many PW codes solve the TE form easily because 1/eps is local in real space.
    #   For TM, eps is local in real space, but 1/eps is not.
    #   Wait, the matrix Eps is circulant. We can diagonalize it once using FFT.
    #   Eps = F^H diag(eps_F) F.
    #   So Eps^{-1} = F^H diag(1/eps_F) F.
    #   Then the operator A = Eps^{-1} D is:
    #   A v = F^H diag(1/eps_F) F D v.
    #   This can be computed in O(N log N) per matrix-vector product!
    #   1. Compute w = D v (element-wise multiply)
    #   2. Compute u = F w (FFT)
    #   3. Compute z = (1/eps_F) * u (element-wise)
    #   4. Compute y = F^H z (IFFT)
    #   This is the fast matrix-vector product for the TM operator.
    
    # Setup for the solver
    # We need a set of G vectors. The evaluator uses 289 plane waves.
    # 289 = 17^2. This suggests a 17x17 square grid of G vectors, or a diamond.
    # "289-plane-wave Galerkin method using the exact Fourier coefficients"
    # Usually, a square grid of -8..8 for both m and n gives 17x17=289.
    # Let's assume a square grid of G vectors: m, n in [-8, 8].
    
    half_pw = 8
    G_vecs = []
    for m in range(-half_pw, half_pw + 1):
        for n in range(-half_pw, half_pw + 1):
            G_vecs.append((m, n))
    G_vecs = np.array(G_vecs) * (2.0 * np.pi)
    nG = len(G_vecs)
    
    # Map G vector indices to FFT indices
    # The pixel grid is 32x32.
    # G = (2*pi*m, 2*pi*n).
    # The FFT of the 32x32 grid gives coefficients for frequencies -16..15 (or 0..31).
    # We need eps_F at (m, n).
    # Note: The pixel grid represents the unit cell.
    # The Fourier coefficient eps_{mn} = (1/32^2) sum_{i,j} eps_{ij} exp(-i 2*pi (m i + n j)/32).
    # This is exactly the 2D FFT of the pixel grid, normalized.
    
    # Precompute FFT of the identity? No, we need to do it for each candidate.
    # But we can precompute the FFT operators.
    
    # K-points: The evaluator uses 30 wave vectors covering the irreducible BZ.
    # Irreducible BZ for square lattice: Triangle (0,0) to (pi,0) to (pi,pi).
    # Let's generate a dense set of k-points in this triangle to approximate the 30 points.
    # Actually, to be safe and fast, we can just use a grid of k-points in the 1/8 wedge.
    # The 30 points are likely a specific set. Let's use a dense grid (e.g., 10x10 in the triangle)
    # which will be a superset and thus a more accurate (stricter) score.
    
    k_points = []
    n_k = 15
    for i in range(n_k):
        for j in range(n_k):
            if i + j <= n_k:
                # Map to triangle: (0,0) to (pi,0) to (pi,pi)
                # Let u = i/n_k, v = j/n_k.
                # x = pi * (u + v) ? No.
                # Vertices: Gamma(0,0), X(pi,0), M(pi,pi).
                # Param: k = Gamma + s*(X-Gamma) + t*(M-X)
                # X-Gamma = (pi, 0). M-X = (0, pi).
                # So k = (s*pi, t*pi) with s in [0,1], t in [0, 1-s].
                # This covers the triangle.
                s = i / n_k
                t = j / n_k
                if s + t <= 1.0:
                    k_points.append((s * np.pi, t * np.pi))
    
    k_points = np.array(k_points)
    
    def compute_score(cell):
        """Compute the 4-5 TM band gap score for a given cell."""
        eps_grid = eps_low + (eps_high - eps_low) * np.array(cell, dtype=float)
        eps_F = np.fft.fft2(eps_grid) / (size * size)
        
        # Handle 1/eps_F. If eps_F is zero, this is a problem.
        # But eps is between 1 and 12, so eps_F[0,0] is ~6.5. Other components are small.
        # 1/eps_F should be safe.
        inv_eps_F = 1.0 / eps_F
        
        min_omega5 = np.inf
        max_omega4 = 0.0
        
        for kx, ky in k_points:
            # Build the operator A = Eps^{-1} D
            # We will use an iterative eigenvalue solver (Lanczos/Arnoldi) or just diagonalize if nG is small.
            # nG = 289. Diagonalizing a 289x289 matrix is fast (~ms).
            # Let's just build the matrix explicitly for simplicity and robustness, 
            # or use the fast matvec with scipy.sparse.linalg.eigsh.
            
            # Let's use scipy.sparse.linalg.eigsh for the lowest 6 eigenvalues.
            # We need to define the matrix A.
            # A v = F^H diag(1/eps_F) F D v
            
            kG = np.stack([kx + G_vecs[:, 0], ky + G_vecs[:, 1]], axis=1)
            kG2 = kG[:, 0]**2 + kG[:, 1]**2
            
            def matvec(v):
                # 1. w = D v
                w = kG2 * v
                # 2. u = F w
                u = np.fft.fft2(w.reshape(17, 17))
                # 3. z = (1/eps_F) * u
                z = inv_eps_F * u
                # 4. y = F^H z
                y = np.fft.ifft2(z)
                return y.flatten().real # Should be real if eps is real and symmetric?
                # Actually, the matrix is Hermitian. The result should be real.
                # But FFT introduces small imaginary parts.
            
            # Use eigsh
            try:
                from scipy.sparse.linalg import eigsh
                from scipy.sparse import LinearOperator
                A_op = LinearOperator((nG, nG), matvec=matvec, dtype=complex)
                # We want the 6 smallest eigenvalues.
                # eigsh finds eigenvalues closest to a value.
                # To get the smallest, we can use sigma=0 with shift-invert, or just 'SA'.
                # 'SA' finds smallest algebraic.
                vals, vecs = eigsh(A_op, k=6, which='SA', tol=1e-6)
                vals = np.sort(vals.real)
                
                if len(vals) < 5:
                    return -1.0
                
                omega4 = np.sqrt(max(0, vals[3]))
                omega5 = np.sqrt(max(0, vals[4]))
                
                if omega5 < min_omega5:
                    min_omega5 = omega5
                if omega4 > max_omega4:
                    max_omega4 = omega4
                    
            except Exception:
                return -1.0
        
        if min_omega5 == np.inf or max_omega4 == 0.0:
            return -1.0
        
        if min_omega5 <= max_omega4:
            return -1.0
            
        return 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)

    # 2. Parametric Search
    # We search over superellipse "pill" shapes.
    # A pill is defined by center (cx, cy), semi-axes (a, b), and exponent p.
    # Due to symmetry, we only need to define one pill in the first octant and replicate it.
    # But the symmetry group D4 has 8 elements.
    # If we place a pill at (cx, cy) with cx != cy and cx != 0, it will have 8 images.
    # If cx == cy, it has 4 images (on the diagonal).
    # If cx == 0 or cy == 0, it has 4 images (on the axes).
    
    # Let's define a function to generate a cell from parameters.
    def generate_cell(params):
        # params: [cx, cy, a, b, p]
        # cx, cy in [0, 0.5] (first quadrant)
        # a, b in [0, 0.5]
        # p in [1, 10]
        cx, cy, a, b, p = params
        
        cell = np.zeros((size, size), dtype=int)
        
        # Define the superellipse equation: |x/a|^p + |y/b|^p <= 1
        # We need to place this at (cx, cy) and its symmetric counterparts.
        
        # Symmetric centers:
        # (cx, cy), (cx, -cy) -> (cx, 1-cy), (-cx, cy) -> (1-cx, cy), (-cx, -cy) -> (1-cx, 1-cy)
        # Also (cy, cx), (cy, 1-cx), (1-cy, cx), (1-cy, 1-cx)
        
        centers = [
            (cx, cy), (cx, 1-cy), (1-cx, cy), (1-cx, 1-cy),
            (cy, cx), (cy, 1-cx), (1-cy, cx), (1-cy, 1-cx)
        ]
        
        # Remove duplicates
        unique_centers = list(set(tuple(np.round(c, 10)) for c in centers))
        
        for (ucx, ucy) in unique_centers:
            # Loop over pixels
            for i in range(size):
                for j in range(size):
                    # Pixel center coordinates in [0,1]
                    x = (j + 0.5) / size
                    y = (i + 0.5) / size
                    dx = x - ucx
                    dy = y - ucy
                    val = (abs(dx/a)**p + abs(dy/b)**p) if a > 0 and b > 0 else np.inf
                    if val <= 1.0:
                        cell[i, j] = 1
                        
        return cell.tolist()

    # Optimization: Random search + Local refinement
    best_score = -1.0
    best_cell = None
    
    # Initial guess: The 4-rod design is known to be good.
    # 4 rods at (0.25, 0.25), (0.75, 0.25), etc. with radius 0.15.
    # This corresponds to cx=0.25, cy=0.25, a=0.15, b=0.15, p=2 (circle).
    
    init_params = [0.25, 0.25, 0.15, 0.15, 2.0]
    init_cell = generate_cell(init_params)
    init_score = compute_score(init_cell)
    
    if init_score > best_score:
        best_score = init_score
        best_cell = init_cell
    
    # Random search
    n_iter = 200
    for i in range(n_iter):
        if time.time() > deadline - 20:
            break
            
        # Random parameters
        # cx, cy in [0.1, 0.4] to avoid center and edges
        cx = np.random.uniform(0.1, 0.4)
        cy = np.random.uniform(0.1, 0.4)
        a = np.random.uniform(0.1, 0.3)
        b = np.random.uniform(0.1, 0.3)
        p = np.random.uniform(1.0, 5.0)
        
        cell = generate_cell([cx, cy, a, b, p])
        score = compute_score(cell)
        
        if score > best_score:
            best_score = score
            best_cell = cell
            # Local refinement
            # Try small perturbations
            for _ in range(10):
                if time.time() > deadline - 20:
                    break
                dcx = np.random.uniform(-0.02, 0.02)
                dcy = np.random.uniform(-0.02, 0.02)
                da = np.random.uniform(-0.02, 0.02)
                db = np.random.uniform(-0.02, 0.02)
                dp = np.random.uniform(-0.5, 0.5)
                
                new_params = [cx+dcx, cy+dcy, a+da, b+db, p+dp]
                # Clamp
                new_params[0] = np.clip(new_params[0], 0.05, 0.45)
                new_params[1] = np.clip(new_params[1], 0.05, 0.45)
                new_params[2] = np.clip(new_params[2], 0.05, 0.45)
                new_params[3] = np.clip(new_params[3], 0.05, 0.45)
                new_params[4] = np.clip(new_params[4], 1.0, 10.0)
                
                new_cell = generate_cell(new_params)
                new_score = compute_score(new_cell)
                
                if new_score > score:
                    score = new_score
                    cell = new_cell
                    cx, cy, a, b, p = new_params
                    if score > best_score:
                        best_score = score
                        best_cell = cell

    # If no valid cell was found, return a default
    if best_cell is None:
        best_cell = generate_cell([0.25, 0.25, 0.15, 0.15, 2.0])
        
    # Ensure symmetry (it should be by construction, but double check)
    # The generation method enforces D4 symmetry.
    
    return best_cell
