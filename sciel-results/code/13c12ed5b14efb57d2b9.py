import numpy as np
import time

def design(problem):
    """
    Designs a 32x32 unit cell by optimizing C4v-symmetric Fourier coefficients
    to maximize the TM band gap between bands 4 and 5.
    """
    start_time = time.time()
    deadline = start_time + 205.0  # Safety margin before 230s kill

    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)

    # Parameterization: C4v symmetric Fourier series centered at (0.5, 0.5)
    # f(x,y) = sum A_{mn} cos(2*pi*m*(x-0.5)) cos(2*pi*n*(y-0.5))
    # Symmetry about x=0.5, y=0.5 is automatic.
    # Symmetry about y=x requires A_{mn} = A_{nm}.

    max_harmonic = 4
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

    # Precompute G vectors and k points for the evaluator approximation
    N_pw = 10
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)

    # K points: 20 points in irreducible BZ
    k_points = []
    # Along Gamma-X: (t, 0) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 5):
        k_points.append((t, 0))
    # Along M-X: (0.5, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 5):
        k_points.append((0.5, t))
    # Along Gamma-M: (t, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 4):
        k_points.append((t, t))
    # Interior points
    k_points.append((0.25, 0.125))
    k_points.append((0.375, 0.125))
    k_points.append((0.25, 0.375))
    k_points.append((0.375, 0.375))

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

            try:
                eigvals = np.linalg.eigvals(M)
                real_eigvals = np.sort(np.real(eigvals))

                if len(real_eigvals) >= 5:
                    omega_4 = np.sqrt(max(0, real_eigvals[3]))
                    omega_5 = np.sqrt(max(0, real_eigvals[4]))

                    if omega_4 > 0 and omega_5 > 0:
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

    # Optimization: Random search with local refinement
    best_params = None
    best_score = -np.inf

    # Initial guess: a dielectric rod-like shape
    init_params = np.zeros(num_params)
    # A(1,1) creates a bump in the center
    idx_11 = params_idx.index((1, 1))
    init_params[idx_11] = 2.0
    # A(0,0) controls the average; keep it low for a rod
    init_params[0] = -0.5

    best_params = init_params.copy()
    best_score = evaluate_cell(get_field(best_params))

    # Random search with restarts
    num_attempts = 80
    for attempt in range(num_attempts):
        if time.time() > deadline - 15:
            break

        # Generate candidate
        if attempt < 30:
            # Local perturbation of best
            p = best_params + np.random.normal(0, 0.4, num_params)
        elif attempt < 60:
            # Moderate exploration
            p = best_params * 0.5 + np.random.uniform(-1.5, 1.5, num_params) * 0.5
        else:
            # Random restart
            p = np.random.uniform(-2, 2, num_params)
            # Bias towards rod-like: A(1,1) positive
            if (1, 1) in params_idx:
                idx = params_idx.index((1, 1))
                p[idx] = abs(p[idx]) + 0.5

        score = evaluate_cell(get_field(p))

        if score > best_score:
            best_score = score
            best_params = p.copy()

            # Local refinement around new best
            for _ in range(3):
                if time.time() > deadline - 15:
                    break
                p_local = best_params + np.random.normal(0, 0.15, num_params)
                score_local = evaluate_cell(get_field(p_local))
                if score_local > best_score:
                    best_score = score_local
                    best_params = p_local.copy()

    if best_params is None:
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

    final_grid = get_field(best_params)

    # Binarize
    threshold = 0.5 * (eps_low + eps_high)
    binary_cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if final_grid[i, j] > threshold:
                binary_cell[i, j] = 1
            else:
                binary_cell[i, j] = 0

    # Enforce strict symmetry
    # cell[i][j] == cell[size-1-i][size-1-j] (180 rotation)
    # cell[i][j] == cell[j][i] (transpose)
    # cell[i][j] == cell[size-1-j][i] (reflection)

    # Average with symmetric counterparts to enforce symmetry
    sym_cell = np.zeros((size, size), dtype=float)
    for i in range(size):
        for j in range(size):
            # All symmetric equivalents
            equivs = [
                (i, j),
                (size - 1 - i, size - 1 - j),  # 180 rotation
                (j, i),  # transpose
                (size - 1 - j, size - 1 - i),  # transpose + 180
                (size - 1 - i, j),  # horizontal flip
                (i, size - 1 - j),  # vertical flip
                (size - 1 - j, i),  # main diagonal reflection
                (j, size - 1 - i),  # anti-diagonal reflection
            ]
            val = np.mean([binary_cell[e[0], e[1]] for e in equivs])
            sym_cell[i, j] = val

    # Threshold the averaged symmetric cell
    final_binary = (sym_cell > 0.5).astype(int)

    # Convert to nested list
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(final_binary[i, j]))
        result.append(row)

    return result
