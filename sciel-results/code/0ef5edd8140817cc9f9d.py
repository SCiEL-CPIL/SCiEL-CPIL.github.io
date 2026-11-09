import numpy as np
import time
import math
import random

# EVOLVE-BLOCK-START

def _build_pw_solver(size, eps_low, eps_high, n_pw, k_points):
    """
    Build a fast plane-wave band solver for a 2D square lattice.
    Returns a function that given a permittivity field (size x size array of 0/1)
    computes the gap-midgap ratio for bands 4 and 5.
    """
    half = n_pw // 2
    idx = np.arange(-half, half + 1)
    Gx, Gy = np.meshgrid(idx, idx, indexing='ij')
    Gx_flat = Gx.ravel()
    Gy_flat = Gy.ravel()
    nG = len(Gx_flat)
    
    # Precompute G-G' differences for the circulant structure?
    # Actually, we need the Fourier coefficients of 1/eps.
    # We will compute these on the fly for each cell evaluation.
    
    # K-points in irreducible BZ (fractional coordinates)
    # Square lattice, irreducible BZ is 1/8 wedge: 0 <= ky <= kx, 0 <= kx <= 0.5
    if k_points is None:
        k_points = []
        # Along X: (k, 0)
        for i in range(5):
            k = i / 4.0 * 0.5
            k_points.append((k, 0.0))
        # Along M-X: (k, k)
        for i in range(1, 5):
            k = i / 4.0 * 0.5
            k_points.append((k, k))
        # Along M-G: (0.5, k)
        for i in range(1, 5):
            k = 0.5 - i / 4.0 * 0.5
            k_points.append((0.5, k))
        # Interior points
        k_points.append((0.25, 0.125))
        k_points.append((0.375, 0.25))
        k_points.append((0.125, 0.0625))
        k_points.append((0.4375, 0.21875))
        
    def compute_bands(eps_field):
        """
        Compute TM band structure. eps_field is size x size array of 0/1.
        Returns gap-midgap ratio.
        """
        # eps(x,y) = eps_low + (eps_high - eps_low) * eps_field
        eps_full = eps_low + (eps_high - eps_low) * eps_field
        inv_eps = 1.0 / eps_full
        
        # Compute Fourier coefficients of 1/eps
        # We need coefficients for G in [-half, half]
        # Use FFT with zero padding
        N_fft = 2 * size
        inv_eps_padded = np.zeros((N_fft, N_fft), dtype=np.float64)
        inv_eps_padded[:size, :size] = inv_eps
        inv_eps_fft = np.fft.fft2(inv_eps_padded) / (N_fft * N_fft)
        
        # Extract coefficients for our G indices
        # FFT index mapping: k_fft = (k_g + N_fft/2) % N_fft
        eps_fourier = np.zeros(nG, dtype=np.float64)
        for gi in range(nG):
            gx = Gx_flat[gi]
            gy = Gy_flat[gi]
            fft_ix = (gx + N_fft // 2) % N_fft
            fft_iy = (gy + N_fft // 2) % N_fft
            eps_fourier[gi] = inv_eps_fft[fft_ix, fft_iy].real
            
        # Precompute G^2
        G2_flat = Gx_flat**2 + Gy_flat**2
        
        max_omega_4 = 0.0
        min_omega_5 = np.inf
        
        for kx_frac, ky_frac in k_points:
            kx = 2.0 * math.pi * kx_frac
            ky = 2.0 * math.pi * ky_frac
            
            # (G+k)^2
            kvec2 = (Gx_flat + kx)**2 + (Gy_flat + ky)**2
            
            # Build matrix M
            # M[i,j] = eps_fourier[idx(G_i - G_j)] * kvec2[j]
            # We need a fast way to get eps_fourier for G_i - G_j.
            # Let's create a lookup table for the Fourier coefficients indexed by (dx, dy)
            # dx, dy range from -2*half to 2*half.
            # But we only have coefficients for -half..half.
            # If |dx| > half or |dy| > half, coeff is 0.
            
            # Efficient matrix construction:
            # M = C * diag(kvec2)
            # C is circulant. C[i,j] = eps_fourier[index(G_i - G_j)]
            # We can build C using the fact that it's circulant.
            # But building C explicitly is O(N^2). N=625, N^2=390k. This is fine.
            
            M = np.zeros((nG, nG), dtype=np.float64)
            
            # Optimization: Precompute the indices for differences?
            # Or just loop.
            # Let's try to vectorize the construction if possible, but simple loop might be okay for 625x625.
            # 625*625 = 390625 operations. In Python, this is slow (~0.1s).
            # We need to speed this up.
            
            # Alternative: Use the property that M = C * D.
            # We can compute the action of C on a vector quickly? No, we need eigenvalues.
            
            # Let's optimize the matrix fill.
            # Gx_flat and Gy_flat are 1D arrays of length nG.
            # We can create a 2D array of differences?
            # Gx_diff = Gx_flat[:, None] - Gx_flat[None, :]
            # Gy_diff = Gy_flat[:, None] - Gy_flat[None, :]
            # This creates 625x625 arrays.
            
            Gx_diff = Gx_flat[:, None] - Gx_flat[None, :]
            Gy_diff = Gy_flat[:, None] - Gy_flat[None, :]
            
            # We need to map (dx, dy) to the index in eps_fourier.
            # eps_fourier is indexed by (gx, gy) where gx,gy in [-half, half].
            # We can create a 2D lookup array of size (2*half+1, 2*half+1)
            # Lookup[dx+half, dy+half] = eps_fourier[index(dx, dy)] if valid, else 0.
            
            lookup = np.zeros((2*half+1, 2*half+1), dtype=np.float64)
            for gi in range(nG):
                gx = Gx_flat[gi]
                gy = Gy_flat[gi]
                lookup[gx + half, gy + half] = eps_fourier[gi]
                
            # Now, extract coefficients for all pairs
            # dx_idx = Gx_diff + half
            # dy_idx = Gy_diff + half
            # Coeffs = lookup[dx_idx, dy_idx]
            
            dx_idx = Gx_diff + half
            dy_idx = Gy_diff + half
            
            # Mask out of bounds (though Gx_diff is in [-2*half, 2*half])
            # Our lookup is only defined for [-half, half].
            # If dx is outside, coeff is 0.
            # We can just clip indices? No, we need 0.
            # Let's create a padded lookup or use np.take with mode='clip' and then zero out?
            # Simpler: The differences Gx_i - Gx_j are integers.
            # The range of Gx is [-half, half]. Diff is [-2*half, 2*half].
            # We only have coeffs for [-half, half].
            # So if |dx| > half or |dy| > half, coeff is 0.
            
            # Create a mask for valid indices
            valid_mask = (np.abs(Gx_diff) <= half) & (np.abs(Gy_diff) <= half)
            
            # We need to index lookup correctly.
            # lookup is (2*half+1, 2*half+1).
            # Indices must be in [0, 2*half].
            # dx_idx is in [-2*half, 2*half].
            # We can't directly index if out of bounds.
            # Let's use a larger lookup or careful indexing.
            
            # Let's just use the flat eps_fourier and a map.
            # But the vectorized lookup is faster if we can handle bounds.
            
            # Let's try a different approach:
            # C[i,j] = eps_fourier[ idx(G_i - G_j) ]
            # We can precompute the index for each pair (i,j) once!
            # Since Gx_flat, Gy_flat are fixed.
            
            # Precompute index_map[i,j] = index in eps_fourier for G_i - G_j, or -1 if out of range.
            # This is done once outside the k-point loop.
            
            # Wait, I am inside compute_bands which is called for each cell.
            # But Gx_flat is fixed. So index_map is fixed.
            # Let's move index_map calculation to the outer scope of compute_bands.
            
            # Re-structuring:
            # The function _build_pw_solver returns compute_bands.
            # We should precompute index_map in _build_pw_solver.
            
            # Let's assume index_map is available.
            # For now, I will implement a slower but correct version inside, 
            # and then optimize by moving precomputation.
            
            # Actually, I can't easily close over a precomputed map in this nested function 
            # unless I define it in the outer scope.
            
            # Let's refactor _build_pw_solver to precompute index_map.
            
            pass # Placeholder, see refactored version below

    return compute_bands

# Refactored Solver
def _build_pw_solver_v2(size, eps_low, eps_high, n_pw, k_points):
    half = n_pw // 2
    idx = np.arange(-half, half + 1)
    Gx, Gy = np.meshgrid(idx, idx, indexing='ij')
    Gx_flat = Gx.ravel()
    Gy_flat = Gy.ravel()
    nG = len(Gx_flat)
    
    # Precompute index map for differences
    # index_map[i, j] = index in eps_fourier array for G_i - G_j
    # If G_i - G_j is outside [-half, half], index is -1
    index_map = np.full((nG, nG), -1, dtype=np.int32)
    # Map (gx, gy) to index
    g_to_idx = {}
    for gi in range(nG):
        g_to_idx[(Gx_flat[gi], Gy_flat[gi])] = gi
        
    for i in range(nG):
        gx_i = Gx_flat[i]
        gy_i = Gy_flat[i]
        for j in range(nG):
            gx_j = Gx_flat[j]
            gy_j = Gy_flat[j]
            dx = gx_i - gx_j
            dy = gy_i - gy_j
            if -half <= dx <= half and -half <= dy <= half:
                index_map[i, j] = g_to_idx[(dx, dy)]
            else:
                index_map[i, j] = -1

    # Precompute G2
    G2_flat = Gx_flat**2 + Gy_flat**2

    if k_points is None:
        k_points = []
        for i in range(5):
            k = i / 4.0 * 0.5
            k_points.append((k, 0.0))
        for i in range(1, 5):
            k = i / 4.0 * 0.5
            k_points.append((k, k))
        for i in range(1, 5):
            k = 0.5 - i / 4.0 * 0.5
            k_points.append((0.5, k))
        k_points.append((0.25, 0.125))
        k_points.append((0.375, 0.25))

    def compute_bands(eps_field):
        eps_full = eps_low + (eps_high - eps_low) * eps_field
        inv_eps = 1.0 / eps_full
        
        N_fft = 2 * size
        inv_eps_padded = np.zeros((N_fft, N_fft), dtype=np.float64)
        inv_eps_padded[:size, :size] = inv_eps
        inv_eps_fft = np.fft.fft2(inv_eps_padded) / (N_fft * N_fft)
        
        eps_fourier = np.zeros(nG, dtype=np.float64)
        for gi in range(nG):
            gx = Gx_flat[gi]
            gy = Gy_flat[gi]
            fft_ix = (gx + N_fft // 2) % N_fft
            fft_iy = (gy + N_fft // 2) % N_fft
            eps_fourier[gi] = inv_eps_fft[fft_ix, fft_iy].real
            
        max_omega_4 = 0.0
        min_omega_5 = np.inf
        
        for kx_frac, ky_frac in k_points:
            kx = 2.0 * math.pi * kx_frac
            ky = 2.0 * math.pi * ky_frac
            kvec2 = (Gx_flat + kx)**2 + (Gy_flat + ky)**2
            
            # Build M
            # M[i,j] = eps_fourier[index_map[i,j]] * kvec2[j] if index_map[i,j] != -1 else 0
            # We can use np.take with a mask
            # Create a masked index array
            # indices = index_map.copy()
            # valid = indices != -1
            # coeffs = np.zeros(nG*nG)
            # coeffs[valid] = eps_fourier[indices[valid]]
            # M = coeffs.reshape(nG, nG) * kvec2[None, :]
            
            # Faster:
            # eps_fourier_ext = np.zeros(nG*nG)
            # But we can just do:
            # M = np.zeros((nG, nG))
            # valid_indices = index_map != -1
            # M[valid_indices] = eps_fourier[index_map[valid_indices]] * kvec2[None, :][valid_indices]
            
            # Even faster:
            # M = np.zeros((nG, nG))
            # idx_valid = index_map != -1
            # M[idx_valid] = eps_fourier[index_map[idx_valid]]
            # M *= kvec2[None, :]
            
            M = np.zeros((nG, nG), dtype=np.float64)
            idx_valid = index_map != -1
            M[idx_valid] = eps_fourier[index_map[idx_valid]]
            M *= kvec2[None, :]
            
            # Symmetrize
            M = (M + M.T) / 2.0
            
            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.sort(eigvals)
            
            if len(eigvals) >= 5:
                omega_4 = math.sqrt(eigvals[3])
                omega_5 = math.sqrt(eigvals[4])
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        gap = min_omega_5 - max_omega_4
        midgap = min_omega_5 + max_omega_4
        if midgap <= 0:
            return 0.0
        return 2.0 * gap / midgap

    return compute_bands

def _get_independent_indices(size):
    """
    Get indices of independent pixels for D4 symmetry.
    Returns a list of (i, j) tuples and a mapping function to reconstruct the full grid.
    """
    half_size = size // 2
    indep = []
    for i in range(size):
        for j in range(size):
            # Map to centered coords
            u = i - (half_size - 0.5)
            v = j - (half_size - 0.5)
            # Fundamental domain: u >= 0, v >= 0, u >= v
            # Note: with size 32, indices 0..31. Center is 15.5.
            # u ranges -15.5 .. 15.5.
            # Condition: u >= 0 and v >= 0 and u >= v
            if u >= 0 and v >= 0 and u >= v:
                indep.append((i, j))
    return indep

def _reconstruct_cell(indep_indices, values, size):
    """
    Reconstruct the full size x size cell from independent pixel values.
    """
    cell = np.zeros((size, size), dtype=np.int8)
    half_size = size // 2
    
    for (i, j), val in zip(indep_indices, values):
        # Apply all 8 symmetries
        u = i - (half_size - 0.5)
        v = j - (half_size - 0.5)
        
        # Generate 8 points
        pts = [
            (u, v), (v, u),
            (-u, v), (v, -u),
            (u, -v), (-v, u),
            (-u, -v), (-v, -u)
        ]
        
        for pu, pv in pts:
            pi = int(round(pu + half_size - 0.5))
            pj = int(round(pv + half_size - 0.5))
            if 0 <= pi < size and 0 <= pj < size:
                cell[pi, pj] = val
                
    return cell

def design(problem):
    t_start = time.time()
    DEADLINE = 220.0  # Safety margin
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    indep_indices = _get_independent_indices(size)
    n_indep = len(indep_indices)
    
    # Initialize with a dielectric background and a large air hole
    # Dielectric = 1, Air = 0
    # Let's start with all 1s, and a central hole of radius ~8
    values = np.ones(n_indep, dtype=np.int8)
    for idx, (i, j) in enumerate(indep_indices):
        u = i - (size // 2 - 0.5)
        v = j - (size // 2 - 0.5)
        dist = math.sqrt(u*u + v*v)
        if dist < 8.0:
            values[idx] = 0
            
    cell = _reconstruct_cell(indep_indices, values, size)
    
    # Setup solver
    n_pw = 25
    compute_bands = _build_pw_solver_v2(size, eps_low, eps_high, n_pw, None)
    
    best_score = compute_bands(cell.astype(np.float64))
    best_values = values.copy()
    
    # Hill Climbing
    temperature = 0.01
    cooling_rate = 0.999
    
    for iteration in range(2000):
        if time.time() - t_start > DEADLINE:
            break
            
        # Pick a random independent pixel
        idx = random.randint(0, n_indep - 1)
        
        # Flip it
        old_val = values[idx]
        values[idx] = 1 - old_val
        
        cell_new = _reconstruct_cell(indep_indices, values, size)
        score_new = compute_bands(cell_new.astype(np.float64))
        
        if score_new > best_score:
            best_score = score_new
            best_values = values.copy()
        else:
            # Accept with probability
            delta = best_score - score_new
            if delta < 0:
                pass # Better than current best? No, best_score is global best.
                # We should compare with current score? 
                # Hill climbing usually keeps current if better, or reverts.
                # Let's stick to simple hill climbing: if not better, revert.
                # But simulated annealing allows worse moves to escape local minima.
                # Let's use SA logic: accept if score_new > score_current or rand < exp((score_new - score_current)/T)
                # We need to track score_current.
                pass
            
        # Simple Hill Climbing: revert if not better than current best?
        # No, that's greedy and gets stuck.
        # Let's use: if score_new > score_current, accept. Else, revert.
        # We need score_current.
        
        # Let's refine the loop logic below.
        pass
        
    return best_values

# EVOLVE-BLOCK-END

# The above draft has issues with the SA logic and efficiency.
# Let's write the final clean version.

def design(problem):
    t_start = time.time()
    DEADLINE = 220.0
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    indep_indices = _get_independent_indices(size)
    n_indep = len(indep_indices)
    
    # Initialize: Dielectric background (1), central air hole (0)
    values = np.ones(n_indep, dtype=np.int8)
    for idx, (i, j) in enumerate(indep_indices):
        u = i - (size // 2 - 0.5)
        v = j - (size // 2 - 0.5)
        dist = math.sqrt(u*u + v*v)
        if dist < 8.0:
            values[idx] = 0
            
    cell = _reconstruct_cell(indep_indices, values, size)
    
    n_pw = 25
    compute_bands = _build_pw_solver_v2(size, eps_low, eps_high, n_pw, None)
    
    score_current = compute_bands(cell.astype(np.float64))
    best_score = score_current
    best_values = values.copy()
    
    temperature = 0.005
    cooling_rate = 0.9995
    
    for iteration in range(3000):
        if time.time() - t_start > DEADLINE:
            break
            
        idx = random.randint(0, n_indep - 1)
        old_val = values[idx]
        values[idx] = 1 - old_val
        
        cell_new = _reconstruct_cell(indep_indices, values, size)
        score_new = compute_bands(cell_new.astype(np.float64))
        
        delta = score_new - score_current
        
        if delta > 0 or random.random() < math.exp(delta / temperature):
            # Accept
            score_current = score_new
            if score_new > best_score:
                best_score = score_new
                best_values = values.copy()
        else:
            # Reject
            values[idx] = old_val
            
        temperature *= cooling_rate
        
    # Final check
    cell_best = _reconstruct_cell(indep_indices, best_values, size)
    return cell_best.tolist()

# EVOLVE-BLOCK-END

