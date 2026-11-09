"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def design(problem):
    """
    Designs a 32x32 unit cell using a C4v-symmetric Fourier synthesis representation,
    optimized via a coarse-to-fine random search with local refinement.
    The continuous dielectric field is binarized at the end.
    """
    start_time = time.time()
    deadline = start_time + 210.0  # Safety margin before 230s kill

    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)

    # --- Representation: C4v symmetric Fourier series ---
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

    def binarize(eps_grid):
        """Binarize the continuous field to 0/1 pixels, enforcing symmetry."""
        threshold = 0.5 * (eps_low + eps_high)
        # cell = 1 where eps > threshold (dielectric)
        cell = (eps_grid > threshold).astype(int)
        
        # Enforce strict symmetry:
        # 1. 180 rotation: cell[i][j] == cell[size-1-i][size-1-j]
        # 2. Transpose: cell[i][j] == cell[j][i]
        # 3. Reflection: cell[i][j] == cell[size-1-j][i]
        
        # Average over the 8 symmetry operations to get a consensus, then threshold
        # Symmetries:
        # S1: identity
        # S2: 90 deg rot
        # S3: 180 deg rot
        # S4: 270 deg rot
        # S5: reflect x
        # S6: reflect y
        # S7: reflect x=y
        # S8: reflect x=-y (or similar)
        
        sym_sum = np.zeros_like(cell, dtype=float)
        sym_sum += cell
        sym_sum += np.rot90(cell, 1)
        sym_sum += np.rot90(cell, 2)
        sym_sum += np.rot90(cell, 3)
        sym_sum += cell[:, ::-1]  # reflect y
        sym_sum += cell[::-1, :]  # reflect x
        sym_sum += cell.T  # transpose (reflect x=y)
        sym_sum += np.rot90(cell, 1).T  # reflect x=-y? Let's check.
        # Actually, for C4v, the 8 operations are:
        # 1, rot90, rot180, rot270, refl_x, refl_y, refl_xy, refl_yx
        
        # Simpler: just average the 8 unique transforms
        transforms = [
            cell,
            np.rot90(cell, 1),
            np.rot90(cell, 2),
            np.rot90(cell, 3),
            cell[:, ::-1],
            cell[::-1, :],
            cell.T,
            np.rot90(cell.T, 1) # This is reflection across y=x line? No.
        ]
        # Let's use the standard 8 symmetries of the square
        # 1. I
        # 2. C4
        # 3. C2
        # 4. C4^3
        # 5. S4 (improper rot, but in 2D it's a reflection + rot) -> Reflection across y=x
        # 6. Reflection across x=0 (y-axis)
        # 7. Reflection across y=0 (x-axis)
        # 8. Reflection across y=-x
        
        # For a grid centered at 0.5,0.5:
        # Refl y-axis: cell[:, ::-1]
        # Refl x-axis: cell[::-1, :]
        # Refl y=x: cell.T
        # Refl y=-x: np.rot90(cell.T, 1) is actually reflection across y=x followed by rot90?
        # Let's just use the 4 main ones and check if they are enough.
        # The task requires: cell == cell[::-1] row-wise and col-wise and cell == its transpose.
        # This implies:
        # 1. cell[i][j] == cell[size-1-i][size-1-j] (180 rot)
        # 2. cell[i][j] == cell[j][i] (transpose)
        # 3. cell[i][j] == cell[size-1-j][i] (reflection across y=x axis? No, that's not right)
        
        # Let's strictly enforce:
        # A = cell
        # B = cell[::-1, ::-1] (180 rot)
        # C = cell.T (transpose)
        # D = cell[::-1, ::-1].T (which is cell[::-1, ::-1] transposed)
        
        # Actually, if cell is symmetric under 180 rot and transpose, it is symmetric under all 8.
        # Let's average the 8 symmetries to be safe.
        sym_sum = 0
        sym_sum += cell
        sym_sum += np.rot90(cell, 1)
        sym_sum += np.rot90(cell, 2)
        sym_sum += np.rot90(cell, 3)
        sym_sum += cell[:, ::-1]
        sym_sum += cell[::-1, :]
        sym_sum += cell.T
        sym_sum += np.rot90(cell.T, 1) # This is reflection across y=-x?
        
        # Let's just use the 4 that are explicitly required or implied.
        # The prompt says: "invariant under the square symmetry group about the cell centre"
        # and "cell == cell[::-1] row-wise and column-wise and cell == its transpose"
        
        # So:
        # 1. cell[i][j] == cell[size-1-i][j] (flip vertical) -> cell[::-1, :]
        # 2. cell[i][j] == cell[i][size-1-j] (flip horizontal) -> cell[:, ::-1]
        # 3. cell[i][j] == cell[j][i] (transpose) -> cell.T
        
        # If these 3 hold, then 180 rot and the other reflections hold automatically.
        # Let's average these 4 (including identity) and threshold at 0.5.
        
        sym_sum = 0
        sym_sum += cell
        sym_sum += cell[::-1, :]
        sym_sum += cell[:, ::-1]
        sym_sum += cell.T
        
        # Also add 180 rot to be safe? It's implied.
        # sym_sum += cell[::-1, ::-1]
        
        # And the other reflections?
        # sym_sum += np.rot90(cell.T, 1) # Reflection across y=-x
        
        # Let's just use the 8 standard ones to be robust.
        sym_sum = 0
        sym_sum += cell
        sym_sum += np.rot90(cell, 1)
        sym_sum += np.rot90(cell, 2)
        sym_sum += np.rot90(cell, 3)
        sym_sum += cell[:, ::-1]
        sym_sum += cell[::-1, :]
        sym_sum += cell.T
        sym_sum += np.rot90(cell.T, 1)
        
        final_cell = (sym_sum / 8.0 > 0.5).astype(int)
        return final_cell

    # --- Fast Band Solver ---
    N_pw = 8
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)
    
    # K points: 15 points in irreducible BZ
    k_points = []
    for t in np.linspace(0, 0.5, 5):
        k_points.append((t, 0))
        k_points.append((0.5, t))
        k_points.append((t, t))
    # Remove duplicates
    seen = set()
    unique_k = []
    for k in k_points:
        key = (round(k[0], 10), round(k[1], 10))
        if key not in seen:
            seen.add(key)
            unique_k.append(k)
    
    # Precompute G-G' differences for each pair (i,j)
    G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]  # (num_pw, num_pw, 2)
    
    def evaluate_cell(eps_grid):
        """
        Computes the band gap using a simplified plane-wave method.
        """
        inv_eps_grid = 1.0 / eps_grid
        fft_inv_eps = np.fft.fft2(inv_eps_grid)
        fft_inv_eps /= (size * size)
        
        # Precompute coefficient matrix for all G-G' differences
        coeff_matrix = np.zeros((num_pw, num_pw), dtype=complex)
        for i in range(num_pw):
            for j in range(num_pw):
                dx = int(G_diff[i, j, 0])
                dy = int(G_diff[i, j, 1])
                ix = dx % size
                iy = dy % size
                coeff_matrix[i, j] = fft_inv_eps[ix, iy]
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in unique_k:
            k = np.array([kx, ky])
            
            # (k+G)^2 for each G
            k_plus_G = k + G_vectors
            omega_sq = np.sum(k_plus_G ** 2, axis=1)  # (num_pw,)
            
            # M[i,j] = coeff_matrix[i,j] * omega_sq[i]
            M = coeff_matrix * omega_sq[:, np.newaxis]
            
            # Symmetrize to Hermitian
            H = (M + M.conj().T) / 2.0
            
            try:
                eigvals = np.linalg.eigvalsh(H)
                real_eigvals = np.sort(np.real(eigvals))
                
                if len(real_eigvals) >= 5:
                    omega_4 = np.sqrt(real_eigvals[3])
                    omega_5 = np.sqrt(real_eigvals[4])
                    
                    max_omega_4 = max(max_omega_4, omega_4)
                    min_omega_5 = min(min_omega_5, omega_5)
            except:
                pass
        
        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap == 0:
            return -1.0
        ratio = 2 * gap / midgap
        return ratio

    # --- Optimization ---
    best_params = None
    best_score = -np.inf
    best_cell = None
    
    # Initial guess: a dielectric rod-like shape
    init_params = np.zeros(num_params)
    # A(1,1) creates a bump in the center
    idx_11 = params_idx.index((1, 1))
    init_params[idx_11] = 2.0
    # A(0,0) controls the average; keep it low for a rod
    init_params[0] = -0.5
    
    best_params = init_params.copy()
    best_score = evaluate_cell(get_field(best_params))
    best_cell = binarize(get_field(best_params))
    
    # Random search with restarts
    num_attempts = 150
    for attempt in range(num_attempts):
        if time.time() > deadline - 10:
            break
        
        # Generate candidate
        if attempt < 50:
            # Local perturbation of best
            p = best_params + np.random.normal(0, 0.3, num_params)
        else:
            # Random restart
            p = np.random.uniform(-2, 2, num_params)
            # Bias towards rod-like: A(1,1) positive
            if (1, 1) in params_idx:
                idx = params_idx.index((1, 1))
                p[idx] = abs(p[idx]) + 1.0
        
        score = evaluate_cell(get_field(p))
        
        if score > best_score:
            best_score = score
            best_params = p.copy()
            best_cell = binarize(get_field(best_params))
            
            # Local refinement around new best
            for _ in range(5):
                if time.time() > deadline - 10:
                    break
                p_local = best_params + np.random.normal(0, 0.1, num_params)
                score_local = evaluate_cell(get_field(p_local))
                if score_local > best_score:
                    best_score = score_local
                    best_params = p_local.copy()
                    best_cell = binarize(get_field(p_local))
    
    if best_cell is None:
        # Fallback to seed
        cell = []
        for i in range(size):
            x_c = (i + 0.5) / size - 0.5
            row = []
            for j in range(size):
                y_c = (j + 0.5) / size - 0.5
                if x_c * x_c + y_c * y_c < 0.2 ** 2:
                    row.append(1)
                else:
                    row.append(0)
            cell.append(row)
        return cell
        
    # Convert to nested list
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(best_cell[i, j]))
        result.append(row)

    return result
# EVOLVE-BLOCK-END
