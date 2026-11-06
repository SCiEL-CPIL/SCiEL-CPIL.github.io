"""Photonic crystal unit cell design for TM band gap optimization."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def _make_cell_from_quadrant(quadrant, size=32):
    """Expand a 16x16 quadrant to a full 32x32 4-fold symmetric cell."""
    half = size // 2
    cell = np.zeros((size, size), dtype=np.int8)
    cell[:half, :half] = quadrant
    cell[:half, half:] = quadrant[:, ::-1]
    cell[half:, :half] = quadrant[::-1, :]
    cell[half:, half:] = quadrant[::-1, ::-1]
    return cell.tolist()

def _precompute_pw_basis(n_pw):
    """Precompute plane wave vectors."""
    kx_list = []
    ky_list = []
    for i in range(-n_pw, n_pw+1):
        for j in range(-n_pw, n_pw+1):
            if i*i + j*j <= n_pw*n_pw:
                kx_list.append(i)
                ky_list.append(j)
    return np.array(kx_list, dtype=float), np.array(ky_list, dtype=float)

def _fast_tm_gap(cell_list, n_pw=4, n_k=6):
    """Fast approximate TM band gap calculation using reduced plane waves."""
    size = len(cell_list)
    eps = np.array(cell_list, dtype=float) * 11.0 + 1.0
    inv_eps = 1.0 / eps
    fft_inv_eps = np.fft.fftshift(np.fft.fft2(inv_eps))
    
    kx_arr, ky_arr = _precompute_pw_basis(n_pw)
    n_pw_actual = len(kx_arr)
    Gvec = np.stack([kx_arr, ky_arr], axis=1)
    
    # Precompute Gm-Gn indices
    gn_diffs = []
    for m in range(n_pw_actual):
        for n_ in range(n_pw_actual):
            gi = int(Gvec[m, 0] - Gvec[n_, 0])
            gj = int(Gvec[m, 1] - Gvec[n_, 1])
            idx_i = gi + size // 2
            idx_j = gj + size // 2
            if 0 <= idx_i < size and 0 <= idx_j < size:
                gn_diffs.append((m, n_, idx_i, idx_j))
            else:
                gn_diffs.append((m, n_, -1, -1))
    
    def compute_bands(kx, ky):
        kvec = np.array([kx, ky])
        kg = Gvec + kvec
        kg_mag = np.sqrt(np.sum(kg**2, axis=1))
        kg_mag = np.where(kg_mag < 1e-10, 1.0, kg_mag)
        
        A = np.zeros((n_pw_actual, n_pw_actual))
        for m, n_, idx_i, idx_j in gn_diffs:
            if idx_i >= 0:
                eps_inv_val = fft_inv_eps[idx_i, idx_j].real
            else:
                eps_inv_val = 0.0
            dot = np.dot(kg[m], kg[n_])
            A[m, n_] = eps_inv_val * dot / (kg_mag[m] * kg_mag[n_])
        
        eigvals = np.linalg.eigvalsh(A)
        eigvals = np.clip(eigvals, 0, None)
        return np.sqrt(eigvals)
    
    # Key k-points in irreducible BZ
    k_points = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)]
    for t in np.linspace(0, 1, n_k):
        k_points.append((t/2, t/2))
        k_points.append((t/2, 0.0))
        k_points.append((0.5, t/2))
    
    all_bands = [compute_bands(kx, ky) for kx, ky in k_points]
    
    max_band4 = max(b[3] for b in all_bands if len(b) > 3)
    min_band5 = min(b[4] for b in all_bands if len(b) > 4)
    
    if min_band5 + max_band4 < 1e-10:
        return 0.0
    return 2.0 * (min_band5 - max_band4) / (min_band5 + max_band4)

def design(problem):
    size = problem["pixels"]
    half = size // 2
    
    start_time = time.time()
    deadline = start_time + 100.0
    
    random.seed(123)
    np.random.seed(123)
    
    best_score = -np.inf
    best_quadrant = None
    
    def score_quadrant(quadrant):
        cell = _make_cell_from_quadrant(quadrant, size)
        return _fast_tm_gap(cell, n_pw=4, n_k=5)
    
    # Initial diverse patterns
    patterns = []
    
    # All dielectric
    patterns.append(np.ones((half, half), dtype=np.int8))
    # All air
    patterns.append(np.zeros((half, half), dtype=np.int8))
    # Dielectric corner blob
    q = np.zeros((half, half), dtype=np.int8)
    q[0:5, 0:5] = 1
    patterns.append(q)
    # Dielectric ring
    q = np.zeros((half, half), dtype=np.int8)
    for i in range(half):
        for j in range(half):
            d = max(i, j)
            if 3 <= d <= 7:
                q[i, j] = 1
    patterns.append(q)
    # Center blob
    q = np.zeros((half, half), dtype=np.int8)
    q[6:10, 6:10] = 1
    patterns.append(q)
    # Cross
    q = np.zeros((half, half), dtype=np.int8)
    q[:, 7:9] = 1
    q[7:9, :] = 1
    patterns.append(q)
    # Stripes
    q = np.zeros((half, half), dtype=np.int8)
    q[:, 0:8] = 1
    patterns.append(q)
    # Diagonal
    q = np.zeros((half, half), dtype=np.int8)
    for i in range(half):
        q[i, i:i+3] = 1
    patterns.append(q)
    
    for q in patterns:
        s = score_quadrant(q)
        if s > best_score:
            best_score = s
            best_quadrant = q.copy()
    
    # Hill climbing
    current = best_quadrant.copy()
    current_score = best_score
    stall = 0
    max_stall = 200
    
    while time.time() < deadline and stall < max_stall:
        improved = False
        for _ in range(30):
            if time.time() >= deadline:
                break
            i = random.randint(0, half - 1)
            j = random.randint(0, half - 1)
            old = current[i, j]
            current[i, j] = 1 - old
            s = score_quadrant(current)
            if s > current_score + 1e-8:
                current_score = s
                if s > best_score:
                    best_score = s
                    best_quadrant = current.copy()
                improved = True
            else:
                current[i, j] = old
        stall = 0 if improved else stall + 1
    
    cell = _make_cell_from_quadrant(best_quadrant, size)
    
    # Verify symmetry
    arr = np.array(cell)
    assert np.array_equal(arr, arr[::-1, :])
    assert np.array_equal(arr, arr[:, ::-1])
    assert np.array_equal(arr, arr.T)
    
    return cell

# EVOLVE-BLOCK-END
