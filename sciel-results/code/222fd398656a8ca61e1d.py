import numpy as np
import time

def design(problem):
    start_time = time.time()
    deadline = start_time + 110

    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    eps_low = problem.get("eps_low", 1)

    n_pw = 17
    half = n_pw // 2
    G_indices = np.arange(-half, half + 1)
    G_vecs = np.array([(m, n) for m in G_indices for n in G_indices])

    k_points_opt = [
        (0.0, 0.0),
        (np.pi / 2, 0.0),
        (np.pi, 0.0),
        (np.pi, np.pi / 2),
        (np.pi, np.pi),
    ]

    kG_sq = np.zeros((len(k_points_opt), n_pw))
    for ik, (kx, ky) in enumerate(k_points_opt):
        for i, (mi, ni) in enumerate(G_vecs):
            kx_tot = kx + 2.0 * np.pi * mi
            ky_tot = ky + 2.0 * np.pi * ni
            kG_sq[ik, i] = kx_tot * kx_tot + ky_tot * ky_tot

    S = size
    x_coords = (np.arange(S) + 0.5) / S
    X, Y = np.meshgrid(x_coords, x_coords, indexing='ij')
    cos_2px = np.cos(2.0 * np.pi * X)
    cos_2py = np.cos(2.0 * np.pi * Y)
    cos_2pxy = np.cos(2.0 * np.pi * (X + Y))
    cos_2pxmy = np.cos(2.0 * np.pi * (X - Y))
    cos_4px = np.cos(4.0 * np.pi * X)
    cos_4py = np.cos(4.0 * np.pi * Y)

    def make_cell(params):
        A, B, C, D, E, F = params
        val = 0.5 + A * cos_2px + B * cos_2py + C * cos_4px + D * cos_4py + E * cos_2pxy + F * cos_2pxmy
        cell = (val > 0.5).astype(np.int32)
        cell = np.round((cell + cell[::-1, :]) / 2.0).astype(np.int32)
        cell = np.round((cell + cell[:, ::-1]) / 2.0).astype(np.int32)
        cell = np.round((cell + cell.T) / 2.0).astype(np.int32)
        return cell

    def compute_gap(cell):
        cell_arr = cell.astype(float)
        eps = eps_low + (eps_high - eps_low) * cell_arr
        eps_inv = 1.0 / eps
        fft_eps_inv = np.fft.fft2(eps_inv) / (S * S)

        best_omega_5_min = np.inf
        best_omega_4_max = 0.0

        for ik in range(len(k_points_opt)):
            H = np.zeros((n_pw, n_pw), dtype=complex)
            for i in range(n_pw):
                mi, ni = G_vecs[i]
                for j in range(n_pw):
                    mj, nj = G_vecs[j]
                    G_diff_m = mi - mj
                    G_diff_n = ni - nj
                    if -half <= G_diff_m <= half and -half <= G_diff_n <= half:
                        eps_coeff = fft_eps_inv[G_diff_n % S, G_diff_m % S]
                    else:
                        eps_coeff = 0.0
                    H[i, j] = eps_coeff * kG_sq[ik, j]

            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sqrt(np.abs(eigvals)) / (2.0 * np.pi)
            if eigvals[4] < best_omega_5_min:
                best_omega_5_min = eigvals[4]
            if eigvals[3] > best_omega_4_max:
                best_omega_4_max = eigvals[3]

        midgap = (best_omega_5_min + best_omega_4_max) / 2.0
        if midgap < 1e-10:
            return -1.0
        return 2.0 * (best_omega_5_min - best_omega_4_max) / midgap

    def make_seed_cell():
        cell = np.zeros((S, S), dtype=np.int32)
        for i in range(S):
            x = (i + 0.5) / S - 0.5
            for j in range(S):
                y = (j + 0.5) / S - 0.5
                if x * x + y * y < 0.04:
                    cell[i, j] = 1
        return cell

    seed_cell = make_seed_cell()
    seed_score = compute_gap(seed_cell)

    best_score = seed_score
    best_cell = seed_cell.copy()

    from scipy.optimize import minimize

    def objective(params):
        if time.time() > deadline:
            return 1e10
        cell = make_cell(params)
        return -compute_gap(cell)

    np.random.seed(42)
    for trial in range(15):
        if time.time() > deadline - 10:
            break
        if trial == 0:
            x0 = np.array([0.3, 0.3, 0.1, 0.1, 0.1, 0.1])
        else:
            x0 = np.random.uniform(-0.5, 0.5, 6)

        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-3, 'fatol': 1e-4, 'adaptive': True})

        if time.time() > deadline - 5:
            break

        cell = make_cell(result.x)
        score = compute_gap(cell)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()

    return best_cell.tolist()
