"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _enforce_square_symmetry(arr):
    """Enforce full D4 square symmetry about center for a 2D array."""
    n = len(arr)
    arr = np.array(arr, dtype=np.float64)
    
    def rot90(a):
        return a.T[::-1, :]
    
    def reflect_v(a):
        return a[:, ::-1]
    
    def reflect_h(a):
        return a[::-1, :]
    
    transforms = [
        arr,
        rot90(arr),
        rot90(rot90(arr)),
        rot90(rot90(rot90(arr))),
        reflect_v(arr),
        reflect_h(arr),
        reflect_v(rot90(arr)),
        reflect_h(rot90(arr)),
    ]
    
    sym_arr = np.mean(transforms, axis=0)
    sym_arr = (sym_arr > 0.5).astype(int)
    return sym_arr


def _pwe_tm_gap(cell, num_pw=6, num_k=6, bands=6):
    """
    Fast PWE solver for TM bands, returns gap-midgap ratio for bands 4-5.
    """
    n = len(cell)
    eps_low = 1.0
    eps_high = 12.0
    
    eps = np.array(cell, dtype=np.float64)
    eps = eps_low + (eps_high - eps_low) * eps
    
    # Compute epsilon Fourier coefficients using FFT
    n_fft = 64
    eps_padded = np.zeros((n_fft, n_fft), dtype=np.float64)
    eps_padded[:n, :n] = eps
    
    eps_fourier = np.fft.fft2(eps_padded) / (n_fft * n_fft)
    
    # Generate G vectors within cutoff
    g_max = num_pw
    G_list = []
    for m in range(-g_max, g_max + 1):
        for l in range(-g_max, g_max + 1):
            if m * m + l * l <= g_max * g_max:
                G_list.append((m, l))
    
    num_G = len(G_list)
    G_array = np.array(G_list, dtype=np.float64)
    
    def get_eps_fourier(m, l):
        m_idx = m % n_fft
        l_idx = l % n_fft
        return eps_fourier[m_idx, l_idx]
    
    # K-points along high-symmetry path
    k_points = []
    for t in np.linspace(0, 1, num_k // 2 + 1):
        k_points.append((t * np.pi, 0.0))
    for t in np.linspace(0, 1, num_k // 2 + 1):
        k_points.append((np.pi, t * np.pi))
    
    band4_max = 0.0
    band5_min = float('inf')
    
    for kx, ky in k_points:
        kx_arr = kx + 2 * np.pi * G_array[:, 0]
        ky_arr = ky + 2 * np.pi * G_array[:, 1]
        
        # Build matrix vectorized
        A = np.zeros((num_G, num_G), dtype=np.complex128)
        
        # Compute all differences
        m_diffs = G_array[:, 0][:, None] - G_array[:, 0][None, :]
        l_diffs = G_array[:, 1][:, None] - G_array[:, 1][None, :]
        
        # Get epsilon values
        eps_vals = np.zeros((num_G, num_G), dtype=np.complex128)
        for i in range(num_G):
            for j in range(num_G):
                eps_vals[i, j] = get_eps_fourier(int(m_diffs[i, j]), int(l_diffs[i, j]))
        
        # Avoid division by zero
        eps_vals[abs(eps_vals) < 1e-10] = 1.0
        
        # Cross product: (kx_i)(ky_j) - (ky_i)(kx_j)
        cross = np.outer(kx_arr, ky_arr) - np.outer(ky_arr, kx_arr)
        
        A = cross / eps_vals
        
        try:
            eigenvalues = np.linalg.eigvalsh(A)
            positive_eigs = eigenvalues[eigenvalues > 0]
            if len(positive_eigs) >= bands:
                if positive_eigs[3] > band4_max:
                    band4_max = positive_eigs[3]
                if positive_eigs[4] < band5_min:
                    band5_min = positive_eigs[4]
        except:
            pass
    
    if band5_min == float('inf') or band4_max == 0:
        return 0.0
    
    gap = band5_min - band4_max
    midgap = (band5_min + band4_max) / 2.0
    if midgap == 0:
        return 0.0
    
    return gap / midgap


def design(problem):
    """Design a 32x32 photonic crystal unit cell maximizing TM band gap."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 170
    
    # Initial design: dielectric ring
    cell = np.zeros((size, size), dtype=int)
    center = (size - 1) / 2.0
    for i in range(size):
        for j in range(size):
            dx = i - center
            dy = j - center
            r = np.sqrt(dx * dx + dy * dy)
            if 7 <= r <= 13:
                cell[i, j] = 1
    
    cell = _enforce_square_symmetry(cell)
    best_score = _pwe_tm_gap(cell.tolist())
    best_cell = cell.copy()
    
    n_indep = size // 2
    indep_positions = [(i, j) for i in range(n_indep) for j in range(n_indep)]
    
    current_cell = best_cell.copy()
    current_score = best_score
    
    for iteration in range(3000):
        if time.time() > deadline:
            break
        
        idx = np.random.randint(len(indep_positions))
        i, j = indep_positions[idx]
        
        current_cell[i, j] = 1 - current_cell[i, j]
        new_cell = _enforce_square_symmetry(current_cell)
        
        new_score = _pwe_tm_gap(new_cell.tolist())
        
        if new_score > current_score:
            current_score = new_score
            current_cell = new_cell
            if current_score > best_score:
                best_score = current_score
                best_cell = new_cell.copy()
        else:
            current_cell[i, j] = 1 - current_cell[i, j]
    
    return best_cell.tolist()
# EVOLVE-BLOCK-END
