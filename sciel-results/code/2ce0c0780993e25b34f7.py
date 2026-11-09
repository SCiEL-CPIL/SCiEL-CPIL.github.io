import numpy as np
import time

def design(problem):
    """
    Optimizes a symmetric 32x32 photonic crystal unit cell to maximize the
    TM band gap between bands 4 and 5.
    
    Uses a fast plane-wave solver with pre-computed structures and a
    multi-start local search on a continuous dielectric field that is
    binarized at the end.
    """
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 230s kill
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # --- Precompute Plane Wave Structures ---
    # The evaluator uses 289 plane waves. 289 = 17^2.
    # This implies a square grid of G vectors with indices -8 to 8.
    N_pw = 8
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)
    
    # Precompute indices for FFT extraction of mu = 1/epsilon
    # We need mu_{G-G'} for G, G' in the set.
    # The difference G-G' ranges from -16 to 16.
    # FFT of a 32x32 grid gives frequencies 0..31 (interpreted as 0..15, -16..-1).
    # We map integer difference d to FFT index d % 32.
    Diff_x = G_vectors[:, 0][:, None] - G_vectors[:, 0][None, :]
    Diff_y = G_vectors[:, 1][:, None] - G_vectors[:, 1][None, :]
    idx_x = (Diff_x + 16) % 32
    idx_y = (Diff_y + 16) % 32
    
    # --- K-points ---
    # The evaluator uses 30 wave vectors.
    # Irreducible BZ of square lattice is the triangle with vertices (0,0), (0.5, 0), (0.5, 0.5).
    # We will sample these 30 points.
    # To match the evaluator's "boundary and interior", we'll use a mix.
    # A common choice is 10 points along each edge of the irreducible triangle.
    # Edge 1: Gamma (0,0) to M (0.5, 0) -> k=(t, 0)
    # Edge 2: M (0.5, 0) to X (0.5, 0.5) -> k=(0.5, t)
    # Edge 3: X (0.5, 0.5) to Gamma (0,0) -> k=(t, t)
    # Note: The problem statement says "square symmetry group", which usually implies C4v.
    # The irreducible BZ for C4v is 1/8 of the full BZ.
    # Vertices: Gamma (0,0), M (pi, 0) [or (0.5, 0) in dimensionless], X (pi, pi) [or (0.5, 0.5)].
    # Wait, standard square BZ is [-pi, pi]^2.
    # Irreducible wedge is 0 <= ky <= kx <= pi.
    # Vertices: (0,0), (pi, 0), (pi, pi).
    # In dimensionless units (a=1, k in units of pi/a?), let's stick to k in [0, 0.5] range for simplicity if we define G as integers.
    # Actually, let's use standard physics units.
    # Lattice constant a=1.
    # Reciprocal lattice vectors b1=(2pi, 0), b2=(0, 2pi).
    # G = n1 b1 + n2 b2.
    # k-points in irreducible BZ: 0 <= ky <= kx <= pi.
    
    k_points = []
    # 10 points along Gamma-M: (t, 0) for t in [0, pi]
    # 10 points along M-X: (pi, t) for t in [0, pi]
    # 10 points along X-Gamma: (t, t) for t in [0, pi]
    
    # To avoid duplicates at corners, we can just generate 10 unique points per segment.
    # Let's use np.linspace(0, np.pi, 10) for each.
    
    # However, the "exact" k-points of the evaluator are unknown.
    # We assume a uniform sampling.
    
    # Let's generate 30 points.
    # Path 1: (0,0) to (pi, 0)
    for i in range(10):
        kx = i * (np.pi / 9.0)
        ky = 0.0
        k_points.append((kx, ky))
        
    # Path 2: (pi, 0) to (pi, pi)
    for i in range(10):
        kx = np.pi
        ky = i * (np.pi / 9.0)
        k_points.append((kx, ky))
        
    # Path 3: (pi, pi) to (0,0)
    for i in range(10):
        t = i * (np.pi / 9.0)
        k_points.append((t, t))
        
    k_points = np.array(k_points)
    
    # Precompute (k + G)^2 for each k-point and each G-vector
    # H_{G,G'} = mu_{G-G'} * (k + G')^2  (Note: convention varies, usually (k+G)^2 on the diagonal or similar)
    # Standard TM equation:
    # (k + G)^2 E_G = sum_{G'} mu_{G-G'} (k + G')^2 E_{G'} ? No.
    # The operator is (1/epsilon) * (-nabla^2).
    # In PW basis:
    # (k+G)^2 E_G = sum_{G'} mu_{G-G'} E_{G'}
    # This is a generalized eigenvalue problem? No, it's a standard eigenvalue problem if we rearrange:
    # sum_{G'} [ mu_{G-G'} - delta_{G,G'} (k+G)^2 / (k+G')^2 ? ] No.
    
    # Let's use the standard formulation:
    # H_{G,G'} = mu_{G-G'} (k + G')^2
    # Then H v = lambda v gives lambda = omega^2.
    # Wait, if H_{G,G'} = mu_{G-G'} (k+G)^2, then:
    # sum_{G'} mu_{G-G'} (k+G)^2 E_{G'} = omega^2 E_G
    # (k+G)^2 sum_{G'} mu_{G-G'} E_{G'} = omega^2 E_G
    # This matches the differential equation.
    # So the matrix is M_{G,G'} = mu_{G-G'} (k+G)^2.
    
    # Let's precompute (k+G)^2 for all k-points and G-vectors.
    # k is a vector, G is a vector.
    # (kx + 2*pi*Gx)^2 + (ky + 2*pi*Gy)^2
    
    KG_sq = np.zeros((len(k_points), num_pw))
    for i, k in enumerate(k_points):
        kx, ky = k
        # G_vectors are integers n1, n2.
        # Actual G vector is (2*pi*n1, 2*pi*n2).
        Gx_actual = 2 * np.pi * G_vectors[:, 0]
        Gy_actual = 2 * np.pi * G_vectors[:, 1]
        KG_sq[i, :] = (kx + Gx_actual)**2 + (ky + Gy_actual)**2
        
    # --- Evaluation Function ---
    
    def evaluate_cell_binary(cell):
        """
        Evaluate the gap for a binary 32x32 cell.
        cell: list of lists of 0/1.
        """
        # Convert to epsilon grid
        eps_grid = np.array([[eps_high if c == 1 else eps_low for c in row] for row in cell])
        mu_grid = 1.0 / eps_grid
        
        # Compute Fourier coefficients of mu
        mu_fft = np.fft.fft2(mu_grid)
        mu_fft /= (size * size)
        
        # Extract mu_{G-G'} matrix
        # mu_diff[i, j] = mu_fft[idx_x[i, j], idx_y[i, j]]
        mu_diff = mu_fft[idx_x, idx_y]
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for i in range(len(k_points)):
            # Construct H matrix
            # H_{G,G'} = mu_{G-G'} * (k+G)^2
            # Note: (k+G)^2 is associated with row G (the left side of the eq).
            # So H[i, j] = mu_diff[i, j] * KG_sq[i, i]  ??
            # Let's check the equation:
            # (k+G)^2 E_G = sum_{G'} mu_{G-G'} E_{G'}
            # This is NOT a matrix eigenvalue problem of the form H v = lambda v directly
            # because (k+G)^2 depends on the row index G and is on the LHS.
            # We can write:
            # D * E = mu_diff * E
            # where D is diagonal with D_{G,G} = (k+G)^2.
            # This is a generalized eigenvalue problem: mu_diff * E = D^{-1} * D * E ?
            # Actually, D E = mu_diff E  =>  D^{-1} mu_diff E = E
            # So we solve for eigenvalues of M = D^{-1} mu_diff.
            # The eigenvalues are 1/omega^2?
            # Let's check dimensions.
            # (k+G)^2 has units of 1/length^2.
            # mu is dimensionless.
            # So M is dimensionless.
            # Eigenvalues lambda of M satisfy M v = lambda v.
            # D^{-1} mu_diff v = lambda v
            # mu_diff v = lambda D v
            # sum mu_{G-G'} v_{G'} = lambda (k+G)^2 v_G
            # This matches the equation if lambda = 1/omega^2.
            # So omega = 1/sqrt(lambda).
            
            # Alternatively, we can form the matrix H = mu_diff * D and solve H v = omega^2 v?
            # No, that would be sum mu_{G-G'} (k+G')^2 v_{G'} = omega^2 v_G.
            # This is a different equation.
            
            # Let's stick to the standard:
            # H_{G,G'} = mu_{G-G'} (k+G)^2
            # This matrix is NOT Hermitian.
            # However, the eigenvalues are real and positive.
            # We can solve the generalized eigenvalue problem:
            # mu_diff v = omega^2 D^{-1} v  ? No.
            
            # Let's use the formulation:
            # sum_{G'} mu_{G-G'} (k+G')^2 E_{G'} = (k+G)^2 E_G  (This is also a common form, depending on where the (k+G)^2 is placed).
            # Actually, the most common PW expansion for TM is:
            # (k+G)^2 E_G = sum_{G'} (1/epsilon)_{G-G'} E_{G'}
            # This leads to the matrix M_{G,G'} = (1/epsilon)_{G-G'} / (k+G)^2 ? No.
            
            # Let's use the matrix:
            # H_{G,G'} = (1/epsilon)_{G-G'} (k+G')^2
            # Then H is Hermitian if (1/epsilon) is real and symmetric (which it is).
            # H_{G,G'} = mu_{G-G'} (k+G')^2
            # H_{G',G} = mu_{G'-G} (k+G)^2 = mu_{G-G'}^* (k+G)^2
            # If k=0, H is Hermitian. For k!=0, H is not Hermitian.
            
            # However, we can transform it to a Hermitian problem.
            # Or just use np.linalg.eigvals.
            
            # Let's use the Hermitian formulation if possible.
            # Define F_G = E_G / sqrt((k+G)^2) ?
            # Then (k+G)^2 E_G = sum mu_{G-G'} E_{G'}
            # (k+G)^2 F_G sqrt((k+G)^2) = sum mu_{G-G'} F_{G'} sqrt((k+G')^2)
            # (k+G)^2 F_G = sum [ mu_{G-G'} sqrt((k+G')^2) / sqrt((k+G)^2) ] F_{G'} ? No.
            
            # Let's just use the non-Hermitian matrix M_{G,G'} = mu_{G-G'} (k+G')^2 and take real parts of eigenvalues.
            # Or better, M_{G,G'} = mu_{G-G'} (k+G)^2.
            # Let's check which one is standard.
            # Joannopoulos:
            # (k+G)^2 E_G = sum_{G'} (1/epsilon)_{G-G'} E_{G'}
            # This is A E = B E where A is diagonal with (k+G)^2 and B is the mu matrix.
            # We want to solve det(A - omega^2 B) = 0? No, omega^2 is the eigenvalue of B^{-1} A.
            # So we solve B^{-1} A v = omega^2 v.
            # Since B is positive definite, we can solve the generalized eigenvalue problem.
            # Or, we can form the Hermitian matrix H = B^{-1/2} A B^{-1/2}.
            # But B is 289x289. Inverting it 30 times is feasible but slow.
            
            # Alternative: Use the matrix H_{G,G'} = mu_{G-G'} (k+G)^2.
            # This matrix satisfies H v = omega^2 v ?
            # sum_{G'} mu_{G-G'} (k+G)^2 v_{G'} = omega^2 v_G
            # (k+G)^2 sum_{G'} mu_{G-G'} v_{G'} = omega^2 v_G
            # This is exactly the equation if we identify the sum as the RHS.
            # So the eigenvalues of M = mu_diff * D (where D is diag((k+G)^2)) are omega^2.
            # M is not Hermitian, but eigenvalues are real.
            
            D = np.diag(KG_sq[i, :])
            M = mu_diff @ D  # mu_diff is (G, G'), D is (G', G) ?
            # mu_diff[i, j] is mu_{G_i - G_j}.
            # M[i, j] = mu_{G_i - G_j} * (k+G_j)^2.
            # This matches M_{G,G'} = mu_{G-G'} (k+G')^2.
            
            eigvals = np.linalg.eigvals(M)
            real_eigvals = np.sort(np.real(eigvals))
            
            if len(real_eigvals) >= 5:
                # Filter out negative or zero eigenvalues due to numerical noise
                valid = real_eigvals[real_eigvals > 1e-6]
                if len(valid) >= 5:
                    omega_4 = np.sqrt(valid[3])
                    omega_5 = np.sqrt(valid[4])
                    
                    max_omega_4 = max(max_omega_4, omega_4)
                    min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap == 0:
            return -1.0
        ratio = 2 * gap / midgap
        return ratio

    # --- Continuous Field Parameterization ---
    # We will use a Fourier series with C4v symmetry.
    # eps(x,y) = eps_avg + sum A_{mn} cos(2*pi*m*(x-0.5)) cos(2*pi*n*(y-0.5))
    # Symmetry A_{mn} = A_{nm}.
    # We will optimize the coefficients A.
    # Then we binarize.
    
    max_harmonic = 3
    params_idx = []
    for m in range(max_harmonic + 1):
        for n in range(m, max_harmonic + 1):
            params_idx.append((m, n))
    num_params = len(params_idx)
    
    # Precompute basis functions on the 32x32 grid
    x = np.linspace(0.5/size, 1 - 0.5/size, size)
    X, Y = np.meshgrid(x, x, indexing='ij')
    Xc = X - 0.5
    Yc = Y - 0.5
    
    basis_funcs = []
    for m, n in params_idx:
        f = np.cos(2 * np.pi * m * Xc) * np.cos(2 * np.pi * n * Yc)
        basis_funcs.append(f)
    basis_funcs = np.array(basis_funcs) # Shape: (num_params, 32, 32)
    
    def get_continuous_eps(params):
        # params: (num_params,)
        # field = sum params[i] * basis_funcs[i]
        field = np.tensordot(params, basis_funcs, axes=(0, 0))
        # Map to epsilon
        eps = 0.5 * (eps_low + eps_high) + 0.5 * (eps_high - eps_low) * np.tanh(field)
        return eps

    def get_binary_cell(params, threshold=0.5):
        eps = get_continuous_eps(params)
        # Threshold at the average
        avg_eps = 0.5 * (eps_low + eps_high)
        cell = [[1 if eps[i, j] > avg_eps else 0 for j in range(size)] for i in range(size)]
        return cell

    # --- Optimization ---
    
    best_score = -np.inf
    best_params = None
    
    # Initial guesses
    # 1. Rod: A(1,1) large
    init1 = [0.0] * num_params
    idx_11 = params_idx.index((1, 1))
    init1[idx_11] = 2.0
    
    # 2. Hole: A(1,1) negative
    init2 = [0.0] * num_params
    init2[idx_11] = -2.0
    
    # 3. Random
    inits = [init1, init2]
    for _ in range(5):
        inits.append([np.random.uniform(-2, 2) for _ in range(num_params)])
    
    # Evaluate inits
    for p in inits:
        cell = get_binary_cell(p)
        score = evaluate_cell_binary(cell)
        if score > best_score:
            best_score = score
            best_params = p[:]
            
    # Local search (hill climbing)
    # We will perform a simple local search on the continuous parameters.
    # Step size is adaptive.
    
    step_size = 0.5
    max_iterations = 500
    
    for _ in range(max_iterations):
        if time.time() > deadline - 20:
            break
            
        improved = False
        for i in range(num_params):
            # Try increasing
            p_new = best_params[:]
            p_new[i] += step_size
            cell = get_binary_cell(p_new)
            score = evaluate_cell_binary(cell)
            if score > best_score:
                best_score = score
                best_params = p_new
                improved = True
                continue # Try more in same direction? Or break? Let's break to keep it simple.
                
            # Try decreasing
            p_new = best_params[:]
            p_new[i] -= step_size
            cell = get_binary_cell(p_new)
            score = evaluate_cell_binary(cell)
            if score > best_score:
                best_score = score
                best_params = p_new
                improved = True
        
        if not improved:
            # Reduce step size
            step_size *= 0.9
            if step_size < 0.01:
                # Try a random restart
                p_new = [np.random.uniform(-2, 2) for _ in range(num_params)]
                cell = get_binary_cell(p_new)
                score = evaluate_cell_binary(cell)
                if score > best_score:
                    best_score = score
                    best_params = p_new
                    step_size = 0.5
                else:
                    break # Stuck
    
    # Final binarization and symmetry enforcement
    if best_params is None:
        # Fallback to a simple rod
        cell = [[0]*size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                if i < size//2 and j < size//2:
                    # Top-left quadrant
                    dist_sq = ((i + 0.5)/size - 0.5)**2 + ((j + 0.5)/size - 0.5)**2
                    if dist_sq < 0.2**2:
                        cell[i][j] = 1
        # Enforce symmetry
        # Actually, let's just return the best found or a known good one.
        # If best_params is None, it means we didn't even evaluate the inits?
        # Should not happen.
        return cell
        
    final_cell = get_binary_cell(best_params)
    
    # Enforce strict symmetry
    # cell[i][j] == cell[size-1-i][size-1-j]
    # cell[i][j] == cell[j][i]
    
    # We can average the symmetric counterparts and re-threshold?
    # Or just pick the value from the "master" quadrant (top-left) and mirror it.
    # But the continuous field was symmetric, so the binary cell should be symmetric
    # except for numerical noise in the tanh/threshold.
    # Let's enforce it by taking the top-left 16x16 and mirroring?
    # No, the symmetry is about the center.
    # Top-left 16x16 is not the fundamental domain.
    # The fundamental domain is 16x16? No.
    # Symmetry:
    # (i, j) <-> (i, 31-j)
    # (i, j) <-> (31-i, j)
    # (i, j) <-> (j, i)
    
    # Let's just ensure the output satisfies the conditions by construction.
    # We will iterate over the unique pixels and set their symmetric counterparts.
    
    final_cell_sym = [[0]*size for _ in range(size)]
    visited = set()
    
    for i in range(size):
        for j in range(size):
            if (i, j) in visited:
                continue
            
            # Determine the value
            # Use the value from the original cell
            val = final_cell[i][j]
            
            # Identify all symmetric positions
            positions = [
                (i, j),
                (i, size-1-j),
                (size-1-i, j),
                (size-1-i, size-1-j),
                (j, i),
                (j, size-1-i),
                (size-1-j, i),
                (size-1-j, size-1-i)
            ]
            
            # Count 1s and 0s to decide? Or just take the first?
            # Since the field is symmetric, they should all be the same.
            # If there's a tie or mismatch due to noise, we can vote.
            votes = sum(final_cell[r][c] for r, c in positions)
            # If majority is 1, set to 1.
            if votes > len(positions) // 2:
                final_val = 1
            else:
                final_val = 0
                
            for r, c in positions:
                final_cell_sym[r][c] = final_val
                visited.add((r, c))
                
    return final_cell_sym

# Helper to ensure the function signature is correct
def design(problem):
    # Call the inner logic
    pass

# Re-define design to use the inner logic properly
def design(problem):
    start_time = time.time()
    deadline = start_time + 200.0
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # --- Precompute Plane Wave Structures ---
    N_pw = 8
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)
    
    Diff_x = G_vectors[:, 0][:, None] - G_vectors[:, 0][None, :]
    Diff_y = G_vectors[:, 1][:, None] - G_vectors[:, 1][None, :]
    idx_x = (Diff_x + 16) % 32
    idx_y = (Diff_y + 16) % 32
    
    # --- K-points ---
    k_points = []
    for i in range(10):
        kx = i * (np.pi / 9.0)
        ky = 0.0
        k_points.append((kx, ky))
    for i in range(10):
        kx = np.pi
        ky = i * (np.pi / 9.0)
        k_points.append((kx, ky))
    for i in range(10):
        t = i * (np.pi / 9.0)
        k_points.append((t, t))
    k_points = np.array(k_points)
    
    KG_sq = np.zeros((len(k_points), num_pw))
    for i, k in enumerate(k_points):
        kx, ky = k
        Gx_actual = 2 * np.pi * G_vectors[:, 0]
        Gy_actual = 2 * np.pi * G_vectors[:, 1]
        KG_sq[i, :] = (kx + Gx_actual)**2 + (ky + Gy_actual)**2
        
    def evaluate_cell_binary(cell):
        eps_grid = np.array([[eps_high if c == 1 else eps_low for c in row] for row in cell])
        mu_grid = 1.0 / eps_grid
        mu_fft = np.fft.fft2(mu_grid)
        mu_fft /= (size * size)
        mu_diff = mu_fft[idx_x, idx_y]
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for i in range(len(k_points)):
            D = np.diag(KG_sq[i, :])
            M = mu_diff @ D
            eigvals = np.linalg.eigvals(M)
            real_eigvals = np.sort(np.real(eigvals))
            valid = real_eigvals[real_eigvals > 1e-6]
            if len(valid) >= 5:
                omega_4 = np.sqrt(valid[3])
                omega_5 = np.sqrt(valid[4])
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0
        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap == 0:
            return -1.0
        return 2 * gap / midgap

    # --- Continuous Field Parameterization ---
    max_harmonic = 3
    params_idx = []
    for m in range(max_harmonic + 1):
        for n in range(m, max_harmonic + 1):
            params_idx.append((m, n))
    num_params = len(params_idx)
    
    x = np.linspace(0.5/size, 1 - 0.5/size, size)
    X, Y = np.meshgrid(x, x, indexing='ij')
    Xc = X - 0.5
    Yc = Y - 0.5
    
    basis_funcs = []
    for m, n in params_idx:
        f = np.cos(2 * np.pi * m * Xc) * np.cos(2 * np.pi * n * Yc)
        basis_funcs.append(f)
    basis_funcs = np.array(basis_funcs)
    
    def get_binary_cell(params, threshold=0.5):
        field = np.tensordot(params, basis_funcs, axes=(0, 0))
        eps = 0.5 * (eps_low + eps_high) + 0.5 * (eps_high - eps_low) * np.tanh(field)
        avg_eps = 0.5 * (eps_low + eps_high)
        cell = [[1 if eps[i, j] > avg_eps else 0 for j in range(size)] for i in range(size)]
        return cell

    best_score = -np.inf
    best_params = None
    
    init1 = [0.0] * num_params
    idx_11 = params_idx.index((1, 1))
    init1[idx_11] = 2.0
    
    init2 = [0.0] * num_params
    init2[idx_11] = -2.0
    
    inits = [init1, init2]
    for _ in range(5):
        inits.append([np.random.uniform(-2, 2) for _ in range(num_params)])
    
    for p in inits:
        cell = get_binary_cell(p)
        score = evaluate_cell_binary(cell)
        if score > best_score:
            best_score = score
            best_params = p[:]
            
    step_size = 0.5
    max_iterations = 500
    
    for _ in range(max_iterations):
        if time.time() > deadline - 20:
            break
        improved = False
        for i in range(num_params):
            p_new = best_params[:]
            p_new[i] += step_size
            cell = get_binary_cell(p_new)
            score = evaluate_cell_binary(cell)
            if score > best_score:
                best_score = score
                best_params = p_new
                improved = True
                continue
            p_new = best_params[:]
            p_new[i] -= step_size
            cell = get_binary_cell(p_new)
            score = evaluate_cell_binary(cell)
            if score > best_score:
                best_score = score
                best_params = p_new
                improved = True
        if not improved:
            step_size *= 0.9
            if step_size < 0.01:
                p_new = [np.random.uniform(-2, 2) for _ in range(num_params)]
                cell = get_binary_cell(p_new)
                score = evaluate_cell_binary(cell)
                if score > best_score:
                    best_score = score
                    best_params = p_new
                    step_size = 0.5
                else:
                    break
    
    if best_params is None:
        cell = [[0]*size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                if i < size//2 and j < size//2:
                    dist_sq = ((i + 0.5)/size - 0.5)**2 + ((j + 0.5)/size - 0.5)**2
                    if dist_sq < 0.2**2:
                        cell[i][j] = 1
        return cell
        
    final_cell = get_binary_cell(best_params)
    
    final_cell_sym = [[0]*size for _ in range(size)]
    visited = set()
    for i in range(size):
        for j in range(size):
            if (i, j) in visited:
                continue
            val = final_cell[i][j]
            positions = [
                (i, j), (i, size-1-j), (size-1-i, j), (size-1-i, size-1-j),
                (j, i), (j, size-1-i), (size-1-j, i), (size-1-j, size-1-i)
            ]
            votes = sum(final_cell[r][c] for r, c in positions)
            final_val = 1 if votes > len(positions) // 2 else 0
            for r, c in positions:
                final_cell_sym[r][c] = final_val
                visited.add((r, c))
                
    return final_cell_sym

