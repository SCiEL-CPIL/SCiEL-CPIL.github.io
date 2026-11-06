import numpy as np
import time
import math
import random
import scipy.optimize as opt
from scipy.sparse.linalg import eigsh
from scipy.sparse import LinearOperator

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin for 200s limit

    # 1. Setup Fast Band Solver (Surrogate)
    # Use a smaller plane-wave basis for speed during optimization
    half_pw = 6  # 13x13 = 169 plane waves
    G_vecs = []
    for m in range(-half_pw, half_pw + 1):
        for n in range(-half_pw, half_pw + 1):
            G_vecs.append((m, n))
    G_vecs = np.array(G_vecs)
    nG = len(G_vecs)
    
    # Precompute k-points (Subset of the 30 evaluator points for speed)
    # The evaluator uses 30 points. We use a representative subset.
    k_points = []
    n_k = 8
    # Gamma-X
    for i in range(n_k + 1):
        s = i / n_k
        k_points.append((s * np.pi, 0))
    # X-M
    for i in range(1, n_k + 1):
        s = i / n_k
        k_points.append((np.pi, s * np.pi))
    # Gamma-M
    for i in range(1, n_k + 1):
        s = i / n_k
        k_points.append((s * np.pi/2, s * np.pi/2))
    k_points = np.array(k_points)
    
    # Precompute kG for all k-points
    kG_all = np.zeros((len(k_points), nG, 2))
    for idx, (kx, ky) in enumerate(k_points):
        kG_all[idx, :, 0] = kx + G_vecs[:, 0] * (2.0 * np.pi)
        kG_all[idx, :, 1] = ky + G_vecs[:, 1] * (2.0 * np.pi)
    kG2_all = np.sum(kG_all**2, axis=2) # (n_k, nG)

    def compute_surrogate_score(cell):
        """
        Compute the 4-5 TM band gap score for a given cell using the fast surrogate.
        """
        eps_grid = eps_low + (eps_high - eps_low) * np.array(cell, dtype=float)
        
        # FFT of epsilon and 1/epsilon
        # Note: For binary dielectric, 1/eps is also binary (1/eps_low or 1/eps_high)
        # But we use the FFT of the actual 1/eps field for accuracy in the Galerkin method
        inv_eps_grid = 1.0 / eps_grid
        inv_eps_F = np.fft.fft2(inv_eps_grid) / (size * size)
        
        min_omega5 = np.inf
        max_omega4 = 0.0
        
        # We need to solve the eigenvalue problem for each k-point
        # Matrix M_{mn} = (k+G_m)^2 * inv_eps_F_{G_m - G_n} * (k+G_n)^2
        # This is a circulant matrix in G-space if we ignore the (k+G)^2 factors? No.
        # It's a generalized eigenvalue problem if we don't normalize, but here it's standard.
        
        # Efficient multiplication using FFT
        # v is a vector in G-space.
        # w = kG2 * v
        # u = FFT2(w)
        # z = inv_eps_F * u
        # y = IFFT2(z)
        
        # Precompute the FFT of the identity? No, just use the structure.
        
        for idx in range(len(k_points)):
            kG2 = kG2_all[idx]
            
            def matvec(v):
                w = kG2 * v
                u = np.fft.fft2(w.reshape(13, 13))
                z = inv_eps_F * u
                y = np.fft.ifft2(z)
                return y.flatten().real
            
            try:
                A_op = LinearOperator((nG, nG), matvec=matvec, dtype=complex)
                # We need bands 4 and 5 (indices 3 and 4)
                # eigsh finds eigenvalues near 'which'
                # 'SA' is smallest algebraic. Since we want the 4th and 5th smallest,
                # and the matrix is positive definite, this should work.
                # However, eigsh can be tricky for the 4th/5th.
                # Let's use k=6 and take the 4th and 5th.
                vals, vecs = eigsh(A_op, k=6, which='SA', tol=1e-3)
                vals = np.sort(vals.real)
                
                if len(vals) < 5:
                    return -1.0
                
                # Filter out negative eigenvalues (numerical noise)
                vals = np.maximum(vals, 0)
                omega4 = np.sqrt(vals[3])
                omega5 = np.sqrt(vals[4])
                
                if omega5 < min_omega5:
                    min_omega5 = omega5
                if omega4 > max_omega4:
                    max_omega4 = omega4
                    
            except Exception:
                return -1.0
        
        if min_omega5 == np.inf or max_omega4 == 0.0:
            return -1.0
            
        if min_omega5 <= max_omega4:
            return -1.0
            
        return 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)

    def enforce_full_symmetry(c):
        """
        Enforce C4v symmetry on 32x32 array.
        """
        c_sym = np.zeros_like(c)
        for i in range(16):
            for j in range(i, 16):
                val = c[i, j]
                c_sym[i, j] = val
                c_sym[j, i] = val
                c_sym[31-i, 31-j] = val
                c_sym[31-j, 31-i] = val
                c_sym[i, 31-j] = val
                c_sym[31-j, i] = val
                c_sym[31-i, j] = val
                c_sym[j, 31-i] = val
        return c_sym

    def generate_cell_from_fourier_modes(modes, threshold):
        """
        Generate a cell by summing specific Fourier modes and thresholding.
        modes: list of (mx, my, amplitude, phase)
        """
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        x = x / size
        y = y / size
        
        field = np.zeros((size, size))
        for mx, my, amp, phase in modes:
            field += amp * np.cos(2 * np.pi * (mx * x + my * y) + phase)
        
        cell = (field > threshold).astype(int)
        return enforce_full_symmetry(cell)

    # 2. Optimization Strategy
    # We optimize the amplitudes and phases of a set of Fourier modes.
    # Target modes for 4-5 TM gap in square lattice:
    # Typically, the gap is opened by the 2nd harmonics.
    # Modes: (2,0), (0,2), (2,2), (-2,2), etc.
    # Due to symmetry, we can restrict to a fundamental set.
    
    # Let's define a parameter vector:
    # [amp_20, amp_02, amp_22, amp_30, amp_03, amp_33, threshold]
    # Phases are fixed to 0 or pi/2 to maintain symmetry?
    # Actually, cos(kx+ky) is symmetric.
    
    # Let's use a more flexible representation:
    # Sum of A_i * cos(2*pi*(m_i*x + n_i*y) + phi_i)
    
    # Key modes for 4-5 gap:
    # (2,0) and (0,2) are the primary drivers.
    # (2,2) and (-2,2) help shape the gap.
    
    def objective(params):
        """
        Minus the score, to be minimized by scipy.
        """
        # Unpack params
        # We use 5 modes: (2,0), (0,2), (2,2), (3,0), (0,3)
        # Plus threshold
        # Params: [A20, A02, A22, A30, A03, threshold]
        # Phases are fixed to 0 for simplicity, or optimized?
        # Let's fix phases to 0 to reduce dimensionality and ensure symmetry.
        
        A20, A02, A22, A30, A03, threshold = params
        
        modes = [
            (2, 0, A20, 0),
            (0, 2, A02, 0),
            (2, 2, A22, 0),
            (3, 0, A30, 0),
            (0, 3, A03, 0)
        ]
        
        cell = generate_cell_from_fourier_modes(modes, threshold)
        score = compute_surrogate_score(cell)
        
        if score < 0:
            return 100.0  # Penalty for no gap
        
        return -score

    # Initial guess
    # Typical amplitudes around 1.0, threshold around 0.0
    x0 = np.array([1.0, 1.0, 0.5, 0.2, 0.2, 0.0])
    
    # Bounds
    # Amplitudes: 0 to 2.0
    # Threshold: -2.0 to 2.0
    bounds = [
        (0.0, 2.0),
        (0.0, 2.0),
        (0.0, 2.0),
        (0.0, 2.0),
        (0.0, 2.0),
        (-2.0, 2.0)
    ]
    
    best_cell = None
    best_score = -np.inf
    
    # Try multiple starting points to avoid local minima
    start_points = [
        np.array([1.0, 1.0, 0.5, 0.2, 0.2, 0.0]),
        np.array([1.5, 1.5, 1.0, 0.5, 0.5, 0.5]),
        np.array([0.5, 0.5, 0.2, 0.1, 0.1, -0.5]),
        np.array([2.0, 2.0, 1.5, 1.0, 1.0, 1.0]),
        np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
    ]
    
    for x0 in start_points:
        if time.time() > deadline - 10.0:
            break
            
        try:
            # Nelder-Mead is robust for small dimensions
            result = opt.minimize(objective, x0, method='Nelder-Mead', bounds=bounds, 
                                  options={'maxiter': 50, 'xatol': 0.01, 'fatol': 0.01})
            
            if result.fun < 100.0:  # Valid score
                score = -result.fun
                if score > best_score:
                    best_score = score
                    # Reconstruct cell
                    A20, A02, A22, A30, A03, threshold = result.x
                    modes = [
                        (2, 0, A20, 0),
                        (0, 2, A02, 0),
                        (2, 2, A22, 0),
                        (3, 0, A30, 0),
                        (0, 3, A03, 0)
                    ]
                    best_cell = generate_cell_from_fourier_modes(modes, threshold)
        except Exception:
            pass

    # 3. Local Refinement (Pixel Flipping)
    # If we found a good cell, try to refine it by flipping pixels.
    if best_cell is not None and best_score > 0:
        cell = best_cell.copy()
        current_score = best_score
        T = 0.05
        T_min = 0.001
        cooling = 0.99
        
        steps = 0
        max_steps = 2000
        
        while time.time() < deadline - 5.0 and steps < max_steps:
            steps += 1
            
            # Pick a random pixel
            i = random.randint(0, size - 1)
            j = random.randint(0, size - 1)
            
            # Flip it
            new_cell = cell.copy()
            new_cell[i, j] = 1 - new_cell[i, j]
            
            # Enforce symmetry
            new_cell = enforce_full_symmetry(new_cell)
            
            # If no change, skip
            if np.array_equal(new_cell, cell):
                continue
                
            # Calculate score
            new_score = compute_surrogate_score(new_cell)
            
            # Acceptance criterion
            if new_score > best_score:
                best_score = new_score
                best_cell = new_cell.copy()
                cell = new_cell
                current_score = new_score
            else:
                delta = new_score - current_score
                if delta < 0:
                    p = math.exp(delta / T)
                    if random.random() < p:
                        cell = new_cell
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_cell = cell.copy()
            
            T = max(T_min, T * cooling)
            
            # Check time frequently
            if steps % 10 == 0 and time.time() > deadline - 10.0:
                break
    else:
        # Fallback: Simple 4-rod structure
        cell = np.zeros((size, size), dtype=int)
        centers = [(8, 8), (23, 8), (8, 23), (23, 23)]
        for cx, cy in centers:
            cell[cx-2:cx+3, cy-2:cy+3] = 1
        best_cell = enforce_full_symmetry(cell)
        best_score = compute_surrogate_score(best_cell)

    # 4. Final Check and Return
    # Ensure the returned cell is a list of lists of ints
    if best_cell is None:
        # Absolute fallback
        best_cell = np.zeros((size, size), dtype=int)
        
    return best_cell.tolist()
