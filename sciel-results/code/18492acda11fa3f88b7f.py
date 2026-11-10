import numpy as np
import time

def design(problem):
    """
    Designs a 32x32 unit cell using Simulated Annealing on pixels,
    enforcing C4v symmetry, to maximize the TM band gap between bands 4 and 5.
    """
    start_time = time.time()
    deadline = start_time + 220.0  # Safety margin before 230s kill
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Precompute symmetry orbits for all pixels
    orbits = {}
    for i in range(size):
        for j in range(size):
            if (i, j) not in orbits:
                orbit = set()
                orbit.add((i, j))
                orbit.add((j, size - 1 - i))
                orbit.add((size - 1 - i, size - 1 - j))
                orbit.add((size - 1 - j, i))
                orbit.add((i, size - 1 - j))
                orbit.add((size - 1 - i, j))
                orbit.add((j, i))
                orbit.add((size - 1 - j, size - 1 - i))
                orbit = list(orbit)
                for pos in orbit:
                    orbits[pos] = orbit

    # List of unique orbits for random selection
    unique_orbits = []
    seen_keys = set()
    for i in range(size):
        for j in range(size):
            orbit = orbits[(i, j)]
            key = tuple(sorted(orbit))
            if key not in seen_keys:
                seen_keys.add(key)
                unique_orbits.append(orbit)

    def get_fourier_coefficients(cell):
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        fft_2d = np.fft.fft2(inv_eps) / (size * size)
        return fft_2d

    def solve_bands_fast(fft_coeffs, n_pw_radius=4, k_points=None):
        """
        Solves bands using a circular cutoff of plane waves.
        Returns the gap ratio.
        """
        if k_points is None:
            k_points = [
                (0.0, 0.0),
                (0.5, 0.0),
                (0.5, 0.5),
                (0.25, 0.0),
                (0.25, 0.25),
                (0.5, 0.25),
            ]
        
        g_vals = []
        for m in range(-n_pw_radius, n_pw_radius + 1):
            for n in range(-n_pw_radius, n_pw_radius + 1):
                if m * m + n * n <= n_pw_radius * n_pw_radius:
                    g_vals.append((m, n))
        
        n_pw = len(g_vals)
        
        max_omega4 = -1e10
        min_omega5 = 1e10
        
        for kx, ky in k_points:
            # Precompute (k-G)^2 for each G
            k_sq = np.zeros(n_pw)
            for j, (mj, nj) in enumerate(g_vals):
                kx_g = kx - mj
                ky_g = ky - nj
                k_sq[j] = kx_g * kx_g + ky_g * ky_g
            
            # Construct symmetric matrix S = D^{-1/2} E D^{-1/2}
            # S_{ij} = E_{ij} / sqrt(k_sq[i] * k_sq[j])
            # E_{ij} = eps_inv[G_i - G_j]
            
            S = np.zeros((n_pw, n_pw), dtype=float)
            
            for i, (mi, ni) in enumerate(g_vals):
                for j, (mj, nj) in enumerate(g_vals):
                    diff_m = mi - mj
                    diff_n = ni - nj
                    m_idx = diff_m % size
                    n_idx = diff_n % size
                    eps_val = fft_coeffs[m_idx, n_idx].real # Should be real
                    
                    # Avoid division by zero if k-G is 0 (only at Gamma for G=0)
                    # If k=0 and G=0, k_sq=0.
                    # In that case, the term is 0/0.
                    # However, the eigenvalue equation at Gamma for G=0 is 0 = omega^2 c_0.
                    # So omega=0 is an eigenvalue.
                    # We can handle this by setting the row/col to 0 except diagonal?
                    # Actually, if k_sq[i] == 0, the i-th basis vector corresponds to the zero mode.
                    # We can just skip or handle carefully.
                    # For simplicity, if k_sq[i]*k_sq[j] < 1e-10, set S[i,j] = 0.
                    denom = np.sqrt(k_sq[i] * k_sq[j])
                    if denom > 1e-10:
                        S[i, j] = eps_val / denom
                    else:
                        S[i, j] = 0.0
                        
            try:
                eigenvalues = np.linalg.eigvalsh(S)
                eigenvalues = np.sort(eigenvalues)
                
                if len(eigenvalues) >= 5:
                    omega4 = eigenvalues[3]
                    omega5 = eigenvalues[4]
                    
                    if omega4 > max_omega4:
                        max_omega4 = omega4
                    if omega5 < min_omega5:
                        min_omega5 = omega5
            except:
                return -1.0
        
        if max_omega4 <= 0 or min_omega5 <= 0:
            return -1.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        
        if midgap <= 0:
            return -1.0
        
        return gap / midgap

    # Initialize cell
    cell = np.zeros((size, size), dtype=int)
    for orbit in unique_orbits:
        if np.random.rand() < 0.5:
            for pos in orbit:
                cell[pos[0], pos[1]] = 1
    
    best_cell = cell.copy()
    best_score = solve_bands_fast(get_fourier_coefficients(cell), n_pw_radius=4)
    
    T = 0.1
    T_min = 1e-4
    alpha = 0.999
    
    iter_count = 0
    while time.time() < deadline - 10.0:
        iter_count += 1
        
        idx = np.random.randint(len(unique_orbits))
        orbit = unique_orbits[idx]
        
        for pos in orbit:
            cell[pos[0], pos[1]] = 1 - cell[pos[0], pos[1]]
        
        score = solve_bands_fast(get_fourier_coefficients(cell), n_pw_radius=4)
        
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
        else:
            delta = score - best_score
            if np.random.rand() < np.exp(delta / T):
                pass
            else:
                for pos in orbit:
                    cell[pos[0], pos[1]] = 1 - cell[pos[0], pos[1]]
        
        T *= alpha
        if T < T_min:
            T = T_min

    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(best_cell[i, j]))
        result.append(row)
    
    return result
