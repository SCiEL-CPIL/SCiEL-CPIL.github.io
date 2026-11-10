"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """Optimized photonic crystal design for maximum TM 4-5 band gap."""
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 100.0  # 100s budget, leaving margin
    
    def make_cell(shape, sz1, sz2=None):
        """Create a 32x32 cell with 4-fold symmetric dielectric pattern."""
        cell = [[0]*size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                dx = (i + 0.5) / size - 0.5
                dy = (j + 0.5) / size - 0.5
                r = math.sqrt(dx*dx + dy*dy)
                if shape == 'circle':
                    inside = r < sz1
                elif shape == 'square':
                    inside = abs(dx) < sz1 and abs(dy) < sz1
                elif shape == 'diamond':
                    inside = abs(dx) + abs(dy) < sz1
                elif shape == 'rounded_square':
                    exp = 4.0
                    inside = (abs(dx)**exp + abs(dy)**exp) ** (1.0/exp) < sz1
                elif shape == 'air_hole_circle':
                    # Dielectric background with circular air hole
                    inside = r >= sz1
                elif shape == 'air_hole_square':
                    inside = not (abs(dx) < sz1 and abs(dy) < sz1)
                elif shape == 'air_hole_diamond':
                    inside = not (abs(dx) + abs(dy) < sz1)
                else:
                    inside = r < sz1
                cell[i][j] = 1 if inside else 0
        return cell
    
    def compute_eps_matrix(cell, N):
        """Compute the A matrix for plane wave method."""
        eps = np.array(cell, dtype=float)
        eps = eps_low + (eps_high - eps_low) * eps
        inv_eps = 1.0 / eps
        
        fft_inv_eps = np.fft.fft2(inv_eps)
        fft_inv_eps /= (size * size)
        
        half_N = N // 2
        g_indices = list(range(-half_N, half_N + 1))
        g_vectors = []
        for gm in g_indices:
            for gn in g_indices:
                g_vectors.append((gm, gn))
        
        def g_to_fft_idx(gm, gn):
            idx_m = gm % size
            idx_n = gn % size
            return idx_m, idx_n
        
        n_g = len(g_vectors)
        A = np.zeros((n_g, n_g), dtype=complex)
        for ki in range(n_g):
            for li in range(n_g):
                gm = g_vectors[ki][0] - g_vectors[li][0]
                gn = g_vectors[ki][1] - g_vectors[li][1]
                if abs(gm) > half_N or abs(gn) > half_N:
                    A[ki, li] = 0.0
                else:
                    idx_m, idx_n = g_to_fft_idx(gm, gn)
                    A[ki, li] = fft_inv_eps[idx_m, idx_n]
        
        return A, g_vectors
    
    def compute_band_gaps(cell, N=13, n_kpoints=12):
        """Compute the TM band gap between bands 4 and 5."""
        A, g_vectors = compute_eps_matrix(cell, N)
        n_g = len(g_vectors)
        
        # Define k-points in the irreducible BZ
        k_points = []
        # Along Gamma-X: (k, 0) for k from 0 to 0.5
        for k in np.linspace(0, 0.5, 6):
            k_points.append((k, 0.0))
        # Along Gamma-M: (k, k) for k from 0 to 0.5/sqrt(2)
        for k in np.linspace(0, 0.5/math.sqrt(2), 6):
            k_points.append((k, k))
        # Along X-M: (k, 0.5-k) for k from 0 to 0.5
        for k in np.linspace(0, 0.5, 6):
            k_points.append((k, 0.5-k))
        
        max_omega4 = 0.0
        min_omega5 = float('inf')
        
        for kx, ky in k_points:
            H = np.zeros((n_g, n_g), dtype=complex)
            for i in range(n_g):
                kx_i = kx + g_vectors[i][0]
                ky_i = ky + g_vectors[i][1]
                for j in range(n_g):
                    kx_j = kx + g_vectors[j][0]
                    ky_j = ky + g_vectors[j][1]
                    dot = kx_i * kx_j + ky_i * ky_j
                    H[i, j] = dot * A[i, j]
            
            try:
                eigvals = np.linalg.eigvalsh(H)
                eigvals = np.real(eigvals)
                eigvals = np.sort(eigvals)
                
                if len(eigvals) >= 5:
                    omega4 = math.sqrt(eigvals[3])
                    omega5 = math.sqrt(eigvals[4])
                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
            except:
                continue
        
        if min_omega5 == float('inf') or max_omega4 == 0:
            return -1.0
        
        gap = min_omega5 - max_omega4
        midgap = 0.5 * (min_omega5 + max_omega4)
        if midgap == 0:
            return -1.0
        return gap / midgap
    
    # Search over different geometries
    best_score = -1.0
    best_cell = None
    best_params = None
    
    # Phase 1: Coarse search with N=9 for speed
    shapes = ['circle', 'square', 'diamond', 'rounded_square', 
              'air_hole_circle', 'air_hole_square', 'air_hole_diamond']
    sizes = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
    for shape in shapes:
        for sz in sizes:
            if time.time() > deadline - 60:
                break
            cell = make_cell(shape, sz)
            score = compute_band_gaps(cell, N=9)
            if score > best_score:
                best_score = score
                best_cell = cell
                best_params = (shape, sz)
        if time.time() > deadline - 60:
            break
    
    # Phase 2: Refine around best with N=13
    if best_params is not None and time.time() < deadline - 30:
        best_shape, best_sz = best_params
        # Refine size
        for sz in np.linspace(best_sz - 0.05, best_sz + 0.05, 11):
            if time.time() > deadline - 30:
                break
            cell = make_cell(best_shape, sz)
            score = compute_band_gaps(cell, N=13)
            if score > best_score:
                best_score = score
                best_cell = cell
                best_params = (best_shape, sz)
    
    # Phase 3: Final evaluation with N=17 for top candidates
    if best_params is not None and time.time() < deadline - 10:
        best_shape, best_sz = best_params
        # Evaluate a few nearby shapes at the best size
        candidate_shapes = [best_shape]
        if best_shape not in ['circle', 'square', 'diamond']:
            candidate_shapes.extend(['circle', 'square', 'diamond'])
        
        for shape in candidate_shapes:
            if time.time() > deadline - 10:
                break
            for sz in [best_sz - 0.02, best_sz, best_sz + 0.02]:
                if time.time() > deadline - 10:
                    break
                cell = make_cell(shape, sz)
                score = compute_band_gaps(cell, N=17)
                if score > best_score:
                    best_score = score
                    best_cell = cell
                    best_params = (shape, sz)
    
    # Return the best cell found
    if best_cell is not None:
        return best_cell
    
    # Fallback: return a reasonable default
    cell = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            dx = (i + 0.5) / size - 0.5
            dy = (j + 0.5) / size - 0.5
            r = math.sqrt(dx*dx + dy*dy)
            cell[i][j] = 1 if r < 0.3 else 0
    return cell
# EVOLVE-BLOCK-END

