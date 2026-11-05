"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell to maximize the TM band gap between bands 4 and 5.
    
    Strategy:
    1. Implement a fast internal PWE solver using vectorized matrix construction.
    2. Generate candidate cells using four-quadrant rod configurations with
       optimized radii and superellipse exponents.
    3. Score each candidate with the internal solver and return the best.
    4. Use eigenvector-guided local search to refine the best candidate.
    5. Maintain D4 symmetry and stay within time budget.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    
    # Time budget: 230s wall-clock limit, keep 15s margin for safety
    start_time = time.time()
    deadline = start_time + 215.0
    
    def check_time():
        return time.time() > deadline - 10.0
    
    # Fast internal PWE solver for TM modes using vectorized operations
    def compute_tm_bands(cell, n_pw=13, n_kpts=15, return_eigvecs=False):
        """
        Compute TM band structure using plane-wave expansion (vectorized).
        
        Parameters:
        -----------
        cell : 2D array of 0/1 (0=air, 1=dielectric)
        n_pw : number of plane waves in each direction (total = n_pw^2)
        n_kpts : number of k-points along the irreducible BZ path
        return_eigvecs : if True, also return eigenvectors at best gap k-point
        
        Returns:
        --------
        gap_ratio : the gap-midgap ratio between bands 4 and 5
        If return_eigvecs: (gap_ratio, best_k, best_vec4, best_vec5)
        """
        cell = np.array(cell, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell  # epsilon map
        inv_eps = 1.0 / eps
        
        N = size
        # Compute 2D FFT of inv_eps
        fft_coeffs = np.fft.fft2(inv_eps) / (N * N)
        
        # Define G vectors: use symmetric range around 0
        half = n_pw // 2
        if n_pw % 2 == 1:
            m_vals = np.arange(-half, half + 1)
        else:
            m_vals = np.arange(-half, half)
        
        n_total = len(m_vals) ** 2
        
        # Precompute G vector pairs
        g_indices = [(m, n) for m in m_vals for n in m_vals]
        
        # Build S matrix: S[i,j] = inv_eps(G_i - G_j)
        # Vectorize using broadcasting
        g_m = np.array([g[0] for g in g_indices])
        g_n = np.array([g[1] for g in g_indices])
        
        # Compute all differences
        dm = g_m[:, None] - g_m[None, :]
        dn = g_n[:, None] - g_n[None, :]
        
        # Map to FFT indices
        idx_m = dm % N
        idx_n = dn % N
        
        S = fft_coeffs[idx_m, idx_n]  # shape (n_total, n_total)
        
        # Define k-points in the irreducible Brillouin zone
        # For square lattice: Gamma(0,0), X(0.5,0), M(0.5,0.5)
        # Irreducible wedge: 0 <= kx <= ky, kx + ky <= 0.5
        k_points = []
        n_per_seg = max(1, n_kpts // 3)
        # Along Gamma to X: k = (t, 0) for t in [0, 0.5]
        for t in np.linspace(0, 0.5, n_per_seg):
            k_points.append((t, 0.0))
        # Along X to M: k = (0.5, t) for t in [0, 0.5]
        for t in np.linspace(0, 0.5, n_per_seg):
            k_points.append((0.5, t))
        # Along M to Gamma: k = (t, t) for t in [0.5, 0]
        for t in np.linspace(0.5, 0, n_per_seg):
            k_points.append((t, t))
        
        # Remove duplicates
        k_points_unique = []
        for kp in k_points:
            if not any(np.allclose(kp, kp2, atol=1e-10) for kp2 in k_points_unique):
                k_points_unique.append(kp)
        
        # Track max of band 4 and min of band 5
        max_omega4 = -np.inf
        min_omega5 = np.inf
        best_gap = -np.inf
        best_k = None
        best_vec4 = None
        best_vec5 = None
        
        for kx, ky in k_points_unique:
            if check_time():
                break
            
            # Build the eigenvalue matrix for TM modes
            # The equation is: sum_{G'} S(G_i - G') |G'+k|^2 c_{G'} = lambda * sum_{G'} S(G_i - G') c_{G'}
            # This is a generalized eigenvalue problem: A c = lambda B c
            # where A[i,j] = S[i,j] * |G_j + k|^2 and B[i,j] = S[i,j]
            
            # Compute |G_j + k|^2 for each plane wave
            # G_j = 2*pi*(m_j, n_j), k = 2*pi*(kx, ky) in units where a=1
            # |G_j + k|^2 = (2*pi)^2 * ((m_j + kx)^2 + (n_j + ky)^2)
            # We can drop the (2*pi)^2 factor since it cancels in the ratio
            
            gk_m = g_m + kx
            gk_n = g_n + ky
            gk_sq = gk_m**2 + gk_n**2  # shape (n_total,)
            
            # A = S * gk_sq[None, :] (broadcast over rows)
            A = S * gk_sq[None, :]
            B = S
            
            # Solve generalized eigenvalue problem
            # Use scipy.linalg.eig for non-symmetric case
            try:
                from scipy.linalg import eig
                vals, vecs = eig(A, B)
                # Take real parts and sort
                vals_real = np.real(vals)
                vals_real = vals_real[np.isfinite(vals_real)]
                vals_sorted = np.sort(vals_real)
                
                if len(vals_sorted) >= 5:
                    omega4 = vals_sorted[3]  # 4th band (0-indexed)
                    omega5 = vals_sorted[4]  # 5th band (0-indexed)
                    if omega5 > 0 and omega4 > 0:
                        max_omega4 = max(max_omega4, omega4)
                        min_omega5 = min(min_omega5, omega5)
                        
                        gap = omega5 - omega4
                        if gap > best_gap:
                            best_gap = gap
                            best_k = (kx, ky)
                            # Get corresponding eigenvectors
                            # Re-sort to get the right columns
                            order = np.argsort(vals_real)
                            vecs_sorted = vecs[:, order]
                            # Filter to keep only finite values
                            valid_mask = np.isfinite(vals_real)
                            valid_order = np.where(valid_mask)[0]
                            # Find indices of 4th and 5th bands in the sorted valid values
                            valid_vals = vals_real[valid_mask]
                            valid_order_sorted = np.argsort(valid_vals)
                            idx4 = valid_order_sorted[3]
                            idx5 = valid_order_sorted[4]
                            best_vec4 = vecs_sorted[:, valid_order[idx4]]
                            best_vec5 = vecs_sorted[:, valid_order[idx5]]
            except Exception:
                pass
        
        if max_omega4 > -np.inf and min_omega5 > 0:
            gap = min_omega5 - max_omega4
            midgap = (min_omega5 + max_omega4) / 2.0
            if midgap > 0:
                score = 2.0 * gap / midgap
                if return_eigvecs:
                    return score, best_k, best_vec4, best_vec5
                return score
            else:
                if return_eigvecs:
                    return -np.inf, None, None, None
                return -np.inf
        else:
            if return_eigvecs:
                return -np.inf, None, None, None
            return -np.inf
    
    # Generate four-quadrant rod cells with different parameters
    def make_four_quadrant_cell(rod_radius, exponent=2.0):
        """
        Create a cell with four rods centered in each quadrant.
        Maintains D4 symmetry by construction.
        """
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        
        # Rod centers in normalized coordinates [-0.5, 0.5]
        centers = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
        
        # Vectorized computation
        x_grid, y_grid = np.meshgrid(coords, coords, indexing='ij')
        
        for (cx, cy) in centers:
            dx = x_grid - cx
            dy = y_grid - cy
            if rod_radius > 0:
                val = np.abs(dx/rod_radius)**exponent + np.abs(dy/rod_radius)**exponent
                cell[val <= 1.0 + 1e-9] = 1
        
        # Ensure D4 symmetry: cell[i,j] == cell[j,i] and cell[i,j] == cell[size-1-i, size-1-j]
        # By construction with symmetric centers, this should hold, but let's verify
        # and enforce if needed
        cell = ((cell + cell.T) // 2).astype(int)
        cell = ((cell + cell[::-1, ::-1]) // 2).astype(int)
        
        return cell.tolist()
    
    def make_ring_cell(inner_r, outer_r, exponent=2.0):
        """Create a cell with a ring (annulus) centered at the cell center."""
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        x_grid, y_grid = np.meshgrid(coords, coords, indexing='ij')
        
        r_inner = np.abs(x_grid/inner_r)**exponent + np.abs(y_grid/inner_r)**exponent
        r_outer = np.abs(x_grid/outer_r)**exponent + np.abs(y_grid/outer_r)**exponent
        
        cell[(r_outer <= 1.0 + 1e-9) & (r_inner > 1.0 + 1e-9)] = 1
        
        cell = ((cell + cell.T) // 2).astype(int)
        cell = ((cell + cell[::-1, ::-1]) // 2).astype(int)
        return cell.tolist()
    
    def make_cross_cell(arm_width, exponent=2.0):
        """Create a cross-shaped cell."""
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        x_grid, y_grid = np.meshgrid(coords, coords, indexing='ij')
        
        # Horizontal arm
        h_val = np.abs(x_grid / 0.45)**exponent + np.abs(y_grid / arm_width)**exponent
        cell[h_val <= 1.0 + 1e-9] = 1
        # Vertical arm
        v_val = np.abs(x_grid / arm_width)**exponent + np.abs(y_grid / 0.45)**exponent
        cell[v_val <= 1.0 + 1e-9] = 1
        
        cell = ((cell + cell.T) // 2).astype(int)
        cell = ((cell + cell[::-1, ::-1]) // 2).astype(int)
        return cell.tolist()
    
    def make_corner_rods_cell(rod_radius, exponent=2.0):
        """Create rods at the four corners of the cell."""
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        centers = [(-0.4, -0.4), (0.4, -0.4), (-0.4, 0.4), (0.4, 0.4)]
        
        x_grid, y_grid = np.meshgrid(coords, coords, indexing='ij')
        
        for (cx, cy) in centers:
            dx = x_grid - cx
            dy = y_grid - cy
            if rod_radius > 0:
                val = np.abs(dx/rod_radius)**exponent + np.abs(dy/rod_radius)**exponent
                cell[val <= 1.0 + 1e-9] = 1
        
        cell = ((cell + cell.T) // 2).astype(int)
        cell = ((cell + cell[::-1, ::-1]) // 2).astype(int)
        return cell.tolist()
    
    def field_intensity_from_vec(vec, g_indices, kx, ky, size):
        """Compute |E|^2 on the pixel grid from a plane-wave expansion vector."""
        m_vals = np.array([g[0] for g in g_indices])
        n_vals = np.array([g[1] for g in g_indices])
        
        # Pixel centers in normalized coordinates
        coords = (np.arange(size) + 0.5) / size - 0.5
        xx, yy = np.meshgrid(coords, coords, indexing='ij')
        
        # Compute phase for each plane wave at each pixel
        # phase[i, px, py] = (m_i + kx) * xx[px,py] + (n_i + ky) * yy[px,py]
        # This is memory intensive, so do it in chunks
        intensity = np.zeros((size, size), dtype=float)
        chunk_size = max(1, 1000000 // (size * size))
        
        for start in range(0, len(g_indices), chunk_size):
            end = min(start + chunk_size, len(g_indices))
            m_chunk = m_vals[start:end] + kx
            n_chunk = n_vals[start:end] + ky
            c_chunk = vec[start:end]
            
            # phase has shape (chunk, size, size)
            phase = np.outer(m_chunk, xx.ravel()) + np.outer(n_chunk, yy.ravel())
            phase = phase.reshape(end - start, size, size)
            
            # E contribution
            e_contrib = np.sum(c_chunk[:, None, None] * np.exp(1j * phase), axis=0)
            intensity += np.abs(e_contrib)**2
        
        return intensity
    
    # Generate candidate cells with various parameters
    candidates = []
    
    # Four-quadrant circles with different radii
    for r in [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]:
        candidates.append(make_four_quadrant_cell(r, 2.0))
    
    # Four-quadrant squircles with different exponents
    for r in [0.08, 0.10, 0.12, 0.14]:
        for n in [3.0, 4.0, 6.0]:
            candidates.append(make_four_quadrant_cell(r, n))
    
    # Ring cells
    for inner_r in [0.08, 0.10, 0.12]:
        for outer_r in [0.20, 0.25, 0.30]:
            if outer_r > inner_r:
                candidates.append(make_ring_cell(inner_r, outer_r, 2.0))
    
    # Cross cells
    for w in [0.04, 0.06, 0.08]:
        candidates.append(make_cross_cell(w, 2.0))
    
    # Corner rods
    for r in [0.06, 0.08, 0.10]:
        candidates.append(make_corner_rods_cell(r, 2.0))
    
    # Also include the seed (single central circle) as a fallback
    def make_central_cell(r, n=2.0):
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        x_grid, y_grid = np.meshgrid(coords, coords, indexing='ij')
        val = np.abs(x_grid/r)**n + np.abs(y_grid/r)**n
        cell[val <= 1.0 + 1e-9] = 1
        cell = ((cell + cell.T) // 2).astype(int)
        cell = ((cell + cell[::-1, ::-1]) // 2).astype(int)
        return cell.tolist()
    
    candidates.append(make_central_cell(0.20, 2.0))  # Seed
    
    # Score all candidates
    best_score = -np.inf
    best_candidate = candidates[0]  # Default to first candidate
    
    for cand in candidates:
        if check_time():
            break
        
        score = compute_tm_bands(cand)
        if score > best_score:
            best_score = score
            best_candidate = cand
    
    # If we have time remaining, try eigenvector-guided local search
    if not check_time() and best_score > -np.inf:
        # Get eigenvectors from the best candidate
        score, best_k, vec4, vec5 = compute_tm_bands(best_candidate, return_eigvecs=True)
        
        if best_k is not None and vec4 is not None and vec5 is not None:
            # Compute field intensities
            n_pw = 13
            half = n_pw // 2
            m_vals = np.arange(-half, half + 1)
            g_indices = [(m, n) for m in m_vals for n in m_vals]
            
            int4 = field_intensity_from_vec(vec4, g_indices, best_k[0], best_k[1], size)
            int5 = field_intensity_from_vec(vec5, g_indices, best_k[0], best_k[1], size)
            
            # Normalize intensities
            if int4.max() > 0:
                int4 = int4 / int4.max()
            if int5.max() > 0:
                int5 = int5 / int5.max()
            
            # Compute a "guidance" score for each pixel
            # Pixels with high intensity in band 4 should become dielectric (to push band 4 up)
            # Pixels with high intensity in band 5 should become air (to push band 5 down)
            guidance = int4 - int5  # Positive means add dielectric, negative means remove
            
            best_arr = np.array(best_candidate)
            improved = False
            
            # Try guided flips
            for attempt in range(30):
                if check_time():
                    break
                
                # Find pixels where guidance is most extreme
                # Only consider pixels that are currently 0 (if guidance > 0) or 1 (if guidance < 0)
                mask_positive = (guidance > 0) & (best_arr == 0)
                mask_negative = (guidance < 0) & (best_arr == 1)
                
                if not mask_positive.any() and not mask_negative.any():
                    break
                
                # Pick the most extreme pixel
                if mask_positive.any() and mask_negative.any():
                    max_pos = np.max(guidance[mask_positive])
                    min_neg = np.min(guidance[mask_negative])
                    if max_pos > abs(min_neg):
                        # Flip a pixel with high positive guidance
                        candidates_pos = np.where(mask_positive)
                        idx = np.argmax(guidance[candidates_pos])
                        i, j = candidates_pos[0][idx], candidates_pos[1][idx]
                    else:
                        # Flip a pixel with high negative guidance
                        candidates_neg = np.where(mask_negative)
                        idx = np.argmin(guidance[candidates_neg])
                        i, j = candidates_neg[0][idx], candidates_neg[1][idx]
                elif mask_positive.any():
                    candidates_pos = np.where(mask_positive)
                    idx = np.argmax(guidance[candidates_pos])
                    i, j = candidates_pos[0][idx], candidates_pos[1][idx]
                else:
                    candidates_neg = np.where(mask_negative)
                    idx = np.argmin(guidance[candidates_neg])
                    i, j = candidates_neg[0][idx], candidates_neg[1][idx]
                
                # Get the symmetric orbit
                orbit = [(i, j), (j, i), (size-1-i, size-1-j), (size-1-j, size-1-i)]
                
                # Check if all pixels in orbit have the same value
                vals = set(best_arr[p, q] for p, q in orbit)
                if len(vals) != 1:
                    continue
                
                # Try flipping
                new_val = 1 - best_arr[i, j]
                for p, q in orbit:
                    best_arr[p, q] = new_val
                
                new_cell = best_arr.tolist()
                new_score = compute_tm_bands(new_cell)
                
                if new_score > best_score:
                    best_score = new_score
                    best_candidate = new_cell
                    improved = True
                else:
                    # Revert
                    for p, q in orbit:
                        best_arr[p, q] = 1 - new_val
        
        # If no improvement from guided search, try random perturbations
        if not improved:
            best_arr = np.array(best_candidate)
            
            for attempt in range(50):
                if check_time():
                    break
                
                # Randomly pick a pixel
                i = np.random.randint(0, size // 2)
                j = np.random.randint(0, size)
                
                # Get the symmetric orbit
                orbit = [(i, j), (j, i), (size-1-i, size-1-j), (size-1-j, size-1-i)]
                
                # Check if all pixels in orbit have the same value
                vals = set(best_arr[p, q] for p, q in orbit)
                if len(vals) != 1:
                    continue
                
                # Try flipping
                new_val = 1 - best_arr[i, j]
                for p, q in orbit:
                    best_arr[p, q] = new_val
                
                new_cell = best_arr.tolist()
                new_score = compute_tm_bands(new_cell)
                
                if new_score > best_score:
                    best_score = new_score
                    best_candidate = new_cell
                else:
                    # Revert
                    for p, q in orbit:
                        best_arr[p, q] = 1 - new_val
    
    return best_candidate

# EVOLVE-BLOCK-END
