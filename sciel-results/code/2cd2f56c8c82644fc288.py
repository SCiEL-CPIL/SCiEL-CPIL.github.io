import numpy as np
import time
import scipy.optimize as opt

def design(problem):
    """
    Optimizes a continuous dielectric field defined by a small set of
    C4-symmetric Fourier coefficients to maximize the TM band gap between
    bands 4 and 5.
    """
    start_time = time.time()
    deadline = start_time + 210.0  # Safety margin before 230s kill
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Parameterization: C4v symmetric Fourier series centered at (0.5, 0.5)
    # f(x,y) = sum A_{mn} cos(2*pi*m*(x-0.5)) cos(2*pi*n*(y-0.5))
    # Symmetry about x=0.5, y=0.5 is automatic.
    # Symmetry about y=x requires A_{mn} = A_{nm}.
    
    max_harmonic = 3
    params_idx = []
    for m in range(max_harmonic + 1):
        for n in range(m, max_harmonic + 1):
            params_idx.append((m, n))
    num_params = len(params_idx)
    
    # Precompute grid coordinates
    x = np.linspace(0.5/size, 1 - 0.5/size, size)
    X, Y = np.meshgrid(x, x, indexing='ij')
    Xc = X - 0.5
    Yc = Y - 0.5
    
    # Precompute basis functions
    basis_funcs = []
    for (m, n) in params_idx:
        if m == 0 and n == 0:
            basis_funcs.append(np.ones((size, size)))
        else:
            cm = np.cos(2 * np.pi * m * Xc)
            cn = np.cos(2 * np.pi * n * Yc)
            basis_funcs.append(cm * cn)
    basis_funcs = np.array(basis_funcs)  # shape (num_params, size, size)
    
    def get_field(params):
        """Returns the 32x32 epsilon grid from parameters."""
        field = np.tensordot(params, basis_funcs, axes=1)
        eps_grid = 0.5 * (eps_low + eps_high) + 0.5 * (eps_high - eps_low) * np.tanh(field)
        return eps_grid

    # Precompute G vectors for 289 PWs (N=16, -16 to 16)
    N_pw = 16
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)
    
    # K points: Denser sampling of the irreducible Brillouin zone
    # Irreducible BZ is the triangle with vertices (0,0), (0.5,0), (0.5,0.5)
    # We sample a grid in the (kx, ky) plane and keep points where 0 <= ky <= kx <= 0.5
    # Let's use a 20x20 grid
    kx_vals = np.linspace(0, 0.5, 20)
    ky_vals = np.linspace(0, 0.5, 20)
    k_points = []
    for kx in kx_vals:
        for ky in ky_vals:
            if 0 <= ky <= kx <= 0.5:
                k_points.append((kx, ky))
    # Add a few more points near the boundaries if needed, but 20x20 triangle gives ~210 points.
    # The evaluator uses 30 points. 210 is overkill but ensures we capture the max/min.
    # To save time, we can subsample. Let's use a 10x10 grid first.
    kx_vals = np.linspace(0, 0.5, 10)
    ky_vals = np.linspace(0, 0.5, 10)
    k_points = []
    for kx in kx_vals:
        for ky in ky_vals:
            if 0 <= ky <= kx <= 0.5:
                k_points.append((kx, ky))
    # ~55 points. Good balance.
    
    # Precompute G-G' differences for each pair (i,j)
    G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]  # (num_pw, num_pw, 2)
    
    # Precompute indices for fft
    fft_indices = np.zeros((num_pw, num_pw, 2), dtype=int)
    for i in range(num_pw):
        for j in range(num_pw):
            fft_indices[i, j, 0] = int(G_diff[i, j, 0]) % size
            fft_indices[i, j, 1] = int(G_diff[i, j, 1]) % size
            
    def evaluate_cell(eps_grid):
        """
        Computes the band gap using the plane-wave method.
        """
        inv_eps_grid = 1.0 / eps_grid
        fft_inv_eps = np.fft.fft2(inv_eps_grid)
        fft_inv_eps /= (size * size)
        
        # Extract coefficients for all G-G' differences
        # coeff_matrix[i, j] = fft_inv_eps[fft_indices[i, j, 0], fft_indices[i, j, 1]]
        coeff_matrix = fft_inv_eps[fft_indices[:,:,0], fft_indices[:,:,1]]
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in k_points:
            k = np.array([kx, ky])
            
            # (k+G)^2 for each G
            k_plus_G = k + G_vectors
            omega_sq = np.sum(k_plus_G ** 2, axis=1)  # (num_pw,)
            
            # M[i,j] = coeff_matrix[i,j] * omega_sq[i]
            M = coeff_matrix * omega_sq[:, np.newaxis]
            
            eigvals = np.linalg.eigvalsh(M) # M is symmetric/Hermitian
            real_eigvals = np.sort(eigvals)
            
            if len(real_eigvals) >= 5:
                omega_4 = np.sqrt(real_eigvals[3])
                omega_5 = np.sqrt(real_eigvals[4])
                
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

    # Optimization: L-BFGS-B
    best_params = None
    best_score = -np.inf
    
    # Initial guess
    init_params = np.zeros(num_params)
    idx_11 = params_idx.index((1, 1))
    init_params[idx_11] = 2.0
    init_params[0] = -0.5
    
    def objective(params):
        if time.time() > deadline - 5:
            return -best_score # Return current best to stop
        field = get_field(params)
        score = evaluate_cell(field)
        return -score # Minimize negative score
    
    # Try a few random starts
    for attempt in range(3):
        if time.time() > deadline - 30:
            break
            
        if attempt == 0:
            p0 = init_params.copy()
        else:
            p0 = np.random.uniform(-2, 2, num_params)
            p0[0] = -0.5 # Keep average low
            
        result = opt.minimize(objective, p0, method='L-BFGS-B', 
                              options={'maxiter': 50, 'maxfun': 50})
        
        if result.fun < -best_score:
            best_score = -result.fun
            best_params = result.x.copy()
            
    if best_params is None:
        best_params = init_params.copy()
        best_score = evaluate_cell(get_field(best_params))
        
    final_grid = get_field(best_params)
    
    # Binarize
    threshold = 0.5 * (eps_low + eps_high)
    binary_cell = []
    for i in range(size):
        row = []
        for j in range(size):
            if final_grid[i, j] > threshold:
                row.append(1)
            else:
                row.append(0)
        binary_cell.append(row)
    
    # Enforce strict symmetry
    sym_cell = np.zeros((size, size), dtype=int)
    counts = np.zeros((size, size), dtype=int)
    
    for i in range(size):
        for j in range(size):
            val = binary_cell[i][j]
            positions = [
                (i, j),
                (size-1-i, j),
                (i, size-1-j),
                (size-1-i, size-1-j),
                (j, i),
                (size-1-j, i),
                (j, size-1-i),
                (size-1-j, size-1-i)
            ]
            for pi, pj in positions:
                sym_cell[pi, pj] += val
                counts[pi, pj] += 1
    
    final_binary_cell = []
    for i in range(size):
        row = []
        for j in range(size):
            if counts[i, j] > 0:
                row.append(1 if sym_cell[i, j] > counts[i, j] / 2.0 else 0)
            else:
                row.append(0)
        final_binary_cell.append(row)
        
    return final_binary_cell
