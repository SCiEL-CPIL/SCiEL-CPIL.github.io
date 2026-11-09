"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _enforce_symmetry(cell, size):
    """Enforce 4-fold rotational and mirror symmetry about center."""
    sym = np.zeros((size, size), dtype=int)
    for i in range(size // 2 + 1):
        for j in range(size // 2 + 1):
            val = cell[i][j]
            sym[i][j] = val
            sym[size - 1 - i][j] = val
            sym[i][size - 1 - j] = val
            sym[size - 1 - i][size - 1 - j] = val
    return sym

def _build_eps_fourier(cell, size, Npw):
    """Build the epsilon^{-1} Fourier matrix for a given PW basis size."""
    half = (Npw - 1) // 2
    eps = np.array(cell, dtype=float)
    eps_inv = np.zeros_like(eps)
    eps_inv[eps > 0.5] = 1.0 / 12.0
    eps_inv[eps <= 0.5] = 1.0
    
    G_vectors = []
    for gx in range(-half, half + 1):
        for gy in range(-half, half + 1):
            G_vectors.append((gx, gy))
    N = len(G_vectors)
    
    # Compute eps^{-1} Fourier coefficients using FFT for speed
    eps_F = np.zeros((N, N), dtype=complex)
    for idx_g, (gx, gy) in enumerate(G_vectors):
        for idx_h, (hx, hy) in enumerate(G_vectors):
            dgx = hx - gx
            dgy = hy - gy
            if dgx == 0 and dgy == 0:
                eps_F[idx_g][idx_h] = np.mean(eps_inv)
            else:
                # Use FFT-based computation
                # e^{2pi i (dgx * x + dgy * y) / size}
                x = np.arange(size)
                phase_x = np.exp(2j * np.pi * dgx * x / size)
                phase_y = np.exp(2j * np.pi * dgy * x / size)
                s = np.sum(eps_inv * np.outer(phase_y, phase_x))
                eps_F[idx_g][idx_h] = s / (size * size)
    return eps_F, G_vectors, N

def _score_cell(cell, size, eps_high=12.0, Npw=15, n_kpoints=30):
    """Score a cell using plane-wave method with specified basis size."""
    half = (Npw - 1) // 2
    eps = np.array(cell, dtype=float)
    eps_inv = np.zeros_like(eps)
    eps_inv[eps > 0.5] = 1.0 / eps_high
    eps_inv[eps <= 0.5] = 1.0
    
    G_vectors = []
    for gx in range(-half, half + 1):
        for gy in range(-half, half + 1):
            G_vectors.append((gx, gy))
    N = len(G_vectors)
    
    # Compute epsilon^{-1} Fourier coefficients
    eps_F = np.zeros((N, N), dtype=complex)
    for idx_g, (gx, gy) in enumerate(G_vectors):
        for idx_h, (hx, hy) in enumerate(G_vectors):
            dgx = hx - gx
            dgy = hy - gy
            if dgx == 0 and dgy == 0:
                eps_F[idx_g][idx_h] = np.mean(eps_inv)
            else:
                x = np.arange(size)
                phase_x = np.exp(2j * np.pi * dgx * x / size)
                phase_y = np.exp(2j * np.pi * dgy * x / size)
                s = np.sum(eps_inv * np.outer(phase_y, phase_x))
                eps_F[idx_g][idx_h] = s / (size * size)
    
    # Define wave vectors in irreducible BZ
    # Square lattice: irreducible wedge is 0 <= ky <= kx, kx in [0, 0.5]
    k_points = []
    # Generate points along boundaries and interior
    for i in range(n_kpoints):
        t = i / (n_kpoints - 1)
        # Mix of boundary and interior points
        if i % 3 == 0:
            # Gamma-X line
            k_points.append((0.5 * t, 0.0))
        elif i % 3 == 1:
            # Gamma-M line  
            k_points.append((0.5 * t, 0.5 * t))
        else:
            # X-M line
            k_points.append((0.5, 0.5 * t))
    
    min_omega5 = float('inf')
    max_omega4 = 0.0
    
    for (kx, ky) in k_points:
        Kx = 2 * np.pi * kx
        Ky = 2 * np.pi * ky
        
        # Build TM eigenvalue matrix
        # For TM: A[G,G'] = (G+K)^2 * eps^{-1}_{G-G'}
        A = np.zeros((N, N), dtype=complex)
        for idx_g, (gx, gy) in enumerate(G_vectors):
            kgx = gx + kx
            kgy = gy + ky
            kg2 = kgx * kgx + kgy * kgy
            for idx_h, (hx, hy) in enumerate(G_vectors):
                A[idx_g][idx_h] = kg2 * eps_F[idx_h][idx_g]
        
        try:
            eigvals = np.linalg.eigvalsh(A)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                omega4 = np.sqrt(max(eigvals[3], 0))
                omega5 = np.sqrt(max(eigvals[4], 0))
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        except:
            pass
    
    if max_omega4 > 0 and min_omega5 > 0:
        gap = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
        return gap
    return -1.0

def _gaussian_field_to_cell(params, size, n_gaussians=4, threshold=0.5):
    """Convert Gaussian mixture parameters to a binary cell with C4v symmetry."""
    # params: [amplitude_0, width_0, cx_0, cy_0, amplitude_1, width_1, cx_1, cy_1, ...]
    field = np.zeros((size, size), dtype=float)
    
    for g in range(n_gaussians):
        amp = params[g * 4]
        width = params[g * 4 + 1]
        cx = params[g * 4 + 2]
        cy = params[g * 4 + 3]
        
        # Ensure positive width
        width = max(abs(width), 0.01)
        
        # Create Gaussian in continuous coordinates
        x = (np.arange(size) + 0.5) / size - 0.5
        y = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # C4v symmetry: 8 symmetric copies
        for dx, dy in [(cx, cy), (-cx, cy), (cx, -cy), (-cx, -cy),
                       (cy, cx), (-cy, cx), (cy, -cx), (-cy, -cx)]:
            field += amp * np.exp(-((X - dx)**2 + (Y - dy)**2) / (2 * width**2))
    
    # Threshold to binary
    cell = (field > threshold).astype(int)
    
    # Enforce exact symmetry
    cell = _enforce_symmetry(cell, size)
    return cell

def _gaussian_field_continuous(params, size, n_gaussians=4):
    """Compute continuous Gaussian field (for gradient computation)."""
    field = np.zeros((size, size), dtype=float)
    
    for g in range(n_gaussians):
        amp = params[g * 4]
        width = params[g * 4 + 1]
        cx = params[g * 4 + 2]
        cy = params[g * 4 + 3]
        
        width = max(abs(width), 0.01)
        
        x = (np.arange(size) + 0.5) / size - 0.5
        y = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        for dx, dy in [(cx, cy), (-cx, cy), (cx, -cy), (-cx, -cy),
                       (cy, cx), (-cy, cx), (cy, -cx), (-cy, -cx)]:
            field += amp * np.exp(-((X - dx)**2 + (Y - dy)**2) / (2 * width**2))
    
    return field

def design(problem):
    """Optimize photonic crystal using Gaussian mixture field representation."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 110.0  # Leave margin before 120s limit
    
    random.seed(42)
    np.random.seed(42)
    
    n_gaussians = 3
    n_params = n_gaussians * 4  # amp, width, cx, cy for each
    
    best_cell = None
    best_score = -1.0
    
    # Initial parameters: start with a few well-separated Gaussians
    def init_params():
        params = np.zeros(n_params)
        for g in range(n_gaussians):
            params[g * 4] = random.uniform(0.5, 2.0)       # amplitude
            params[g * 4 + 1] = random.uniform(0.05, 0.15)  # width
            params[g * 4 + 2] = random.uniform(-0.3, 0.3)   # cx
            params[g * 4 + 3] = random.uniform(-0.3, 0.3)   # cy
        return params
    
    # Try multiple initializations
    n_inits = 5
    for init_i in range(n_inits):
        if time.time() > deadline - 20:
            break
        
        params = init_params()
        cell = _gaussian_field_to_cell(params, size, n_gaussians)
        score = _score_cell(cell, size, Npw=13, n_kpoints=20)
        
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
        
        # Gradient-free local search in parameter space
        # Use Nelder-Mead style simplex
        simplex = [params.copy()]
        for _ in range(n_params):
            p = params.copy()
            idx = random.randint(0, n_params - 1)
            p[idx] += random.uniform(-0.05, 0.05)
            simplex.append(p)
        
        max_iter = 200
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Evaluate all simplex points
            scores = []
            for p in simplex:
                c = _gaussian_field_to_cell(p, size, n_gaussians)
                s = _score_cell(c, size, Npw=11, n_kpoints=15)
                scores.append(s)
            
            # Find best and worst
            best_idx = np.argmax(scores)
            worst_idx = np.argmin(scores)
            
            if scores[best_idx] > best_score:
                best_score = scores[best_idx]
                best_cell = _gaussian_field_to_cell(simplex[best_idx], size, n_gaussians).copy()
            
            # Reflection
            centroid = np.mean([simplex[i] for i in range(len(simplex)) if i != worst_idx], axis=0)
            reflected = centroid + (centroid - simplex[worst_idx])
            c_ref = _gaussian_field_to_cell(reflected, size, n_gaussians)
            s_ref = _score_cell(c_ref, size, Npw=11, n_kpoints=15)
            
            if s_ref >= scores[best_idx]:
                # Expansion
                expanded = centroid + 2 * (reflected - centroid)
                c_exp = _gaussian_field_to_cell(expanded, size, n_gaussians)
                s_exp = _score_cell(c_exp, size, Npw=11, n_kpoints=15)
                if s_exp > s_ref:
                    simplex[worst_idx] = expanded
                    if s_exp > best_score:
                        best_score = s_exp
                        best_cell = c_exp.copy()
                else:
                    simplex[worst_idx] = reflected
                    if s_ref > best_score:
                        best_score = s_ref
                        best_cell = c_ref.copy()
            elif s_ref >= scores[worst_idx]:
                simplex[worst_idx] = reflected
                if s_ref > best_score:
                    best_score = s_ref
                    best_cell = c_ref.copy()
            else:
                # Contraction
                contracted = centroid + 0.5 * (simplex[worst_idx] - centroid)
                c_con = _gaussian_field_to_cell(contracted, size, n_gaussians)
                s_con = _score_cell(c_con, size, Npw=11, n_kpoints=15)
                if s_con > scores[worst_idx]:
                    simplex[worst_idx] = contracted
                    if s_con > best_score:
                        best_score = s_con
                        best_cell = c_con.copy()
                else:
                    # Shrink
                    for i in range(len(simplex)):
                        if i != best_idx:
                            simplex[i] = simplex[best_idx] + 0.5 * (simplex[i] - simplex[best_idx])
        
        # After Nelder-Mead, do fine-tuning with smaller steps
        current_params = simplex[np.argmax([_score_cell(_gaussian_field_to_cell(p, size, n_gaussians), size, Npw=11, n_kpoints=15) for p in simplex])]
        
        for ft_iter in range(100):
            if time.time() > deadline - 5:
                break
            
            # Try small perturbations
            improved = False
            for _ in range(10):
                p_new = current_params.copy()
                idx = random.randint(0, n_params - 1)
                p_new[idx] += random.gauss(0, 0.02)
                c_new = _gaussian_field_to_cell(p_new, size, n_gaussians)
                s_new = _score_cell(c_new, size, Npw=11, n_kpoints=15)
                
                if s_new > best_score:
                    best_score = s_new
                    best_cell = c_new.copy()
                    current_params = p_new
                    improved = True
                    break
            
            if not improved:
                # Try larger step
                p_new = current_params.copy()
                idx = random.randint(0, n_params - 1)
                p_new[idx] += random.gauss(0, 0.1)
                c_new = _gaussian_field_to_cell(p_new, size, n_gaussians)
                s_new = _score_cell(c_new, size, Npw=11, n_kpoints=15)
                if s_new > best_score:
                    best_score = s_new
                    best_cell = c_new.copy()
                    current_params = p_new
    
    # Final refinement: local pixel search around best cell
    if best_cell is not None and time.time() < deadline - 10:
        cell = best_cell.copy()
        unique_positions = []
        for i in range(size // 2 + 1):
            for j in range(size // 2 + 1):
                unique_positions.append((i, j))
        
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            idx = random.randint(0, len(unique_positions) - 1)
            i, j = unique_positions[idx]
            old_val = cell[i][j]
            cell[i][j] = 1 - old_val
            cell = _enforce_symmetry(cell, size)
            score = _score_cell(cell, size, Npw=13, n_kpoints=20)
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
            else:
                cell[i][j] = old_val
                cell = _enforce_symmetry(cell, size)
    
    # Convert to nested list
    if best_cell is None:
        # Fallback: simple circle
        best_cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < 0.2 ** 2:
                    best_cell[i][j] = 1
    
    result = [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]
    return result
# EVOLVE-BLOCK-END
