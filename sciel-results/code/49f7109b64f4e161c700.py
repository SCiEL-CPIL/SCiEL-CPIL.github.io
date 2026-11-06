import numpy as np
import time
from scipy.linalg import eigvalsh

def design(problem):
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s CPU / 200s wall
    
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    # 1. Setup Fourier Grid
    m_max = 8
    m_vals = np.arange(-m_max, m_max + 1)
    G_vectors = np.array(np.meshgrid(m_vals, m_vals)).T.reshape(-1, 2) * (2 * np.pi)
    
    # 2. Define K-points
    k_points = []
    for t in np.linspace(0, 0.5, 10):
        k_points.append([t, 0])
    for t in np.linspace(0, 0.5, 10):
        k_points.append([0.5, t])
    for t in np.linspace(0, 0.5, 10):
        k_points.append([0.5 - t, 0.5 - t])
    k_points = np.array(k_points)
    
    def compute_score(cell):
        eps = eps_low + (eps_high - eps_low) * cell
        fft_eps = np.fft.fft2(eps)
        fft_eps_centered = np.fft.fftshift(fft_eps)
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        m_G = G_vectors[:, 0]
        n_G = G_vectors[:, 1]
        
        for k in k_points:
            K_plus_G = k[None, :] + G_vectors
            KG_sq = np.sum(K_plus_G ** 2, axis=1)
            KG_dot = K_plus_G @ K_plus_G.T
            A = np.diag(KG_sq) - KG_dot
            
            dm = m_G[:, None] - m_G[None, :]
            dn = n_G[:, None] - n_G[None, :]
            
            idx_dm = (dm + 16) % 32
            idx_dn = (dn + 16) % 32
            
            B = fft_eps_centered[idx_dm, idx_dn] / (size * size)
            
            try:
                eigvals = eigvalsh(A, B)
                omegas = np.sqrt(np.maximum(eigvals, 0))
                
                if len(omegas) >= 5:
                    omega_4 = omegas[3]
                    omega_5 = omegas[4]
                    
                    if omega_5 < min_omega_5:
                        min_omega_5 = omega_5
                    if omega_4 > max_omega_4:
                        max_omega_4 = omega_4
            except Exception:
                return -1.0
        
        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        return 2 * gap / (min_omega_5 + max_omega_4)

    # Initial Design: 4 Rods
    cell = np.zeros((size, size), dtype=int)
    centers = [(8, 8), (8, 24), (24, 8), (24, 24)]
    radius = 4
    for cy, cx in centers:
        for i in range(size):
            for j in range(size):
                if (i - cy)**2 + (j - cx)**2 < radius**2:
                    cell[i, j] = 1
                    
    best_score = compute_score(cell)
    best_cell = cell.copy()
    
    def apply_flip(c, i, j):
        val = c[i, j]
        new_val = 1 - val
        positions = [
            (i, j), (i, 31-j), (31-i, j), (31-i, 31-j),
            (j, i), (j, 31-i), (31-j, i), (31-j, 31-i)
        ]
        for r, col in set(positions):
            c[r, col] = new_val

    while time.time() - start_time < time_limit:
        i = np.random.randint(0, 16)
        j = np.random.randint(0, 16)
        
        cell_new = best_cell.copy()
        apply_flip(cell_new, i, j)
        
        score_new = compute_score(cell_new)
        
        if score_new > best_score:
            best_score = score_new
            best_cell = cell_new
            
    return best_cell.tolist()
