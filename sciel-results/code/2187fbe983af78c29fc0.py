"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def design(problem):
    """
    Design a 32x32 photonic crystal unit cell with D4 symmetry to maximize
    the gap between TM bands 4 and 5.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 230s limit

    def get_cell_from_params(params):
        """
        Generate a D4 symmetric cell from parameters.
        params: list of (cx, cy, radius, shape_type)
        shape_type: 0 for square, 1 for circle, 2 for plus/cross
        """
        cell = np.zeros((size, size), dtype=float)
        for p in params:
            if len(p) < 4: continue
            cx, cy, radius, stype = p[0], p[1], p[2], p[3]
            if radius < 0: continue
            
            # Generate in the fundamental domain or just generate full and symmetrize?
            # Easier: generate full image and enforce symmetry by averaging/masking.
            # But for binary, we can just define it in one quadrant and reflect.
            # Let's generate the full grid and then enforce D4 symmetry by taking the majority vote or just defining it symmetrically.
            # To be safe and fast, we define the pattern in the top-left 1/8 (triangle) and reflect.
            # However, for simple shapes like circles/squares centered at specific points,
            # it's easier to just draw them and then enforce symmetry.
            
            # Let's just draw the shape on the full grid.
            if stype == 0: # Square
                x1 = int(cx - radius)
                x2 = int(cx + radius + 1)
                y1 = int(cy - radius)
                y2 = int(cy + radius + 1)
                x1 = max(0, min(size-1, x1))
                x2 = max(0, min(size-1, x2))
                y1 = max(0, min(size-1, y1))
                y2 = max(0, min(size-1, y2))
                cell[x1:x2+1, y1:y2+1] = 1.0
            elif stype == 1: # Circle
                # Vectorized circle drawing
                ys, xs = np.ogrid[:size, :size]
                mask = (xs - cx)**2 + (ys - cy)**2 <= radius**2
                cell[mask] = 1.0
            elif stype == 2: # Plus sign (cross)
                # Horizontal bar
                cell[int(cy-radius):int(cy+radius+1), int(cx-radius):int(cx+radius+1)] = 1.0 # This is a square, let's do thin cross
                # Let's stick to squares and circles for simplicity and robustness of the surrogate.
                # A "plus" is hard to parameterize simply without more params.
                pass
        
        # Enforce D4 symmetry
        # The cell must be invariant under:
        # 1. Transpose: cell[i,j] == cell[j,i]
        # 2. 180 Rotation: cell[i,j] == cell[N-1-i, N-1-j]
        # 3. 90 Rotation: cell[i,j] == cell[N-1-j, i]
        # 4. Reflection: cell[i,j] == cell[i, N-1-j] and cell[N-1-i, j]
        
        # We can enforce this by averaging the 8 symmetric copies and thresholding.
        # Copies:
        # 1. Identity
        # 2. Transpose
        # 3. 180 Rot
        # 4. 90 Rot (Counter-clockwise)
        # 5. 270 Rot
        # 6. Horizontal Flip
        # 7. Vertical Flip
        # 8. Diagonal Flip (Anti-transpose)
        
        c0 = cell
        c1 = cell.T
        c2 = cell[::-1, ::-1]
        c3 = cell.T[::-1, :] # 90 deg CW? Let's check.
        # Rotation 90 CW: (i,j) -> (j, N-1-i). So new[i,j] = old[j, N-1-i].
        # np.rot90(m, k=1) rotates CCW. np.rot90(m, k=-1) rotates CW.
        c3 = np.rot90(cell, k=-1)
        c4 = np.rot90(cell, k=-2) # Same as c2
        c5 = np.rot90(cell, k=-3)
        c6 = cell[:, ::-1] # Horizontal flip (flip y)
        c7 = cell[::-1, :] # Vertical flip (flip x)
        
        # Average all 8
        avg_cell = (c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7) / 8.0
        
        # Threshold at 0.5 to get binary
        sym_cell = (avg_cell > 0.5).astype(int)
        
        return sym_cell

    def approximate_gap_ratio(cell, max_g=4, n_kpoints=8):
        """
        Fast surrogate model to estimate the gap ratio.
        Uses plane-wave method with max_g order G-vectors.
        """
        # 1/eps coefficients
        inv_eps = np.where(cell == 1, 1.0/eps_high, 1.0/eps_low)
        inv_fft = np.fft.fft2(inv_eps) / (size * size)
        
        # G vectors
        g_vectors = []
        for m in range(-max_g, max_g + 1):
            for n in range(-max_g, max_g + 1):
                g_vectors.append((m, n))
        n_g = len(g_vectors)
        
        # Precompute inv_eps coefficients for differences in G
        # We need inv_eps_{G-G'}
        # Map (dm, dn) to coeff
        coeff_lookup = {}
        for m in range(-max_g, max_g + 1):
            for n in range(-max_g, max_g + 1):
                m_idx = m % size
                n_idx = n % size
                coeff_lookup[(m, n)] = inv_fft[m_idx, n_idx]
        
        # K-points in irreducible BZ
        # Gamma (0,0), X (pi/2, 0), M (pi/2, pi/2)
        # Plus some interior points
        k_points = [
            (0.0, 0.0),
            (np.pi/2, 0.0),
            (np.pi/2, np.pi/2),
            (np.pi/4, 0.0),
            (np.pi/4, np.pi/4),
            (0.0, np.pi/4),
            (np.pi/3, 0.0),
            (np.pi/3, np.pi/6),
        ]
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            H = np.zeros((n_g, n_g), dtype=complex)
            
            for i, (m1, n1) in enumerate(g_vectors):
                kx_g = kx + 2 * np.pi * m1
                ky_g = ky + 2 * np.pi * n1
                k_sq = kx_g**2 + ky_g**2
                
                for j, (m2, n2) in enumerate(g_vectors):
                    dm = m1 - m2
                    dn = n1 - n2
                    if (dm, dn) in coeff_lookup:
                        H[i, j] = k_sq * coeff_lookup[(dm, dn)]
                    else:
                        H[i, j] = 0.0
            
            # Solve eigenvalue problem
            # H is Hermitian
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(np.real(eigvals))
            
            # Filter out negative eigenvalues (numerical noise)
            eigvals = np.maximum(eigvals, 0.0)
            
            if len(eigvals) >= 5:
                omega_4 = math.sqrt(eigvals[3])
                omega_5 = math.sqrt(eigvals[4])
                
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 <= 0 or max_omega_4 <= 0:
            return -1.0
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap <= 0:
            return -1.0
        
        return gap / midgap

    # Main Optimization
    best_score = -1.0
    best_params = None
    
    # Parameter space:
    # We will search for a single central feature or a symmetric set of features.
    # Given D4 symmetry, a feature at (cx, cy) implies features at (cy, cx), (N-1-cx, N-1-cy), etc.
    # To avoid redundancy, we define features in the fundamental domain.
    # However, the `get_cell_from_params` function draws on the full grid and symmetrizes.
    # So we can just define one "primary" feature and let the symmetrization create the rest.
    # But wait, if I define a feature at (8,8), symmetrization creates it at (8,8), (24,8), (8,24), (24,24) etc.
    # So I only need to specify the primary feature's center and radius.
    
    # Let's try a grid search over:
    # cx, cy in the fundamental domain (0..15, 0..15)
    # radius in [1..10]
    # shape_type: 0 (square), 1 (circle)
    
    # To speed up, we'll do a coarse grid first.
    # Fundamental domain for D4 on 32x32:
    # i from 0 to 15, j from i to 31-i? No, the fundamental domain is a triangle.
    # (i,j) such that i <= j and i+j <= 31?
    # Actually, since we symmetrize, we can just pick centers in the top-left 16x16 quadrant.
    # But we must ensure the feature doesn't overlap with its symmetric counterparts in a way that breaks the intended shape?
    # Overlapping is fine, it just adds up (max operation effectively, or sum then threshold).
    # Since we use `avg > 0.5`, overlapping features reinforce each other.
    
    # Let's search for a single shape type and center/radius.
    # We'll also try a "checkerboard" like pattern which is a 4x4 grid of small squares.
    
    candidates = []
    
    # 1. Single central square/circle (at 16,16)
    # But 16,16 is on the symmetry axis.
    # Let's try centers in the 1/8 sector: i in [0, 16), j in [i, 32-i] ?
    # Simpler: Just iterate i,j in [0, 16) and radii.
    
    for i in range(0, 16, 2):
        for j in range(i, 16, 2):
            for r in [2.0, 3.0, 4.0, 5.0, 6.0]:
                # Square
                candidates.append([(float(i), float(j), r, 0)])
                # Circle
                candidates.append([(float(i), float(j), r, 1)])
                
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    if time.time() > deadline:
        break

    # 2. "Cross" or "Plus" shape centered at 16,16?
    # A plus shape can be simulated by two rectangles.
    # But our parametric generator only supports single blobs.
    # Let's add a special case for "Plus" centered at (16,16) or (8,8).
    # Actually, a "Plus" at (8,8) with symmetrization creates a complex pattern.
    # Let's stick to the blob search first.
    
    # 3. High density checkerboard?
    # Let's try a few hardcoded high-performing topologies.
    # A common one is a 4x4 array of small squares.
    # Centers at (4,4), (4,12), (12,4), (12,12) etc?
    # Let's add specific "multi-center" candidates.
    
    multi_centers = [
        [(4,4), (4,12), (12,4), (12,12)],
        [(6,6), (6,10), (10,6), (10,10)],
        [(8,8), (8,10), (10,8), (10,10)], # Small cluster
    ]
    
    for centers in multi_centers:
        for r in [1.5, 2.0, 2.5, 3.0]:
            # Square
            params = [(float(cx), float(cy), r, 0) for cx, cy in centers]
            candidates.append(params)
            # Circle
            params = [(float(cx), float(cy), r, 1) for cx, cy in centers]
            candidates.append(params)
            
    # Evaluate candidates
    for params in candidates:
        if time.time() > deadline:
            break
            
        cell = get_cell_from_params(params)
        score = approximate_gap_ratio(cell)
        
        if score > best_score:
            best_score = score
            best_params = params
    
    # If we have time, do a local refinement on the best parameters
    if best_params is not None and time.time() < deadline - 10:
        # Refine the best single-blob or multi-blob
        # We'll perturb the radius and center slightly.
        current_params = best_params[:]
        current_score = best_score
        
        for step in range(20):
            if time.time() > deadline:
                break
            
            improved = False
            # Perturb each parameter
            for idx, p in enumerate(current_params):
                cx, cy, r, stype = p
                # Try small changes in r
                for dr in [-0.5, 0.5]:
                    new_r = r + dr
                    if new_r < 0.5: continue
                    new_p = (cx, cy, new_r, stype)
                    new_params = current_params[:]
                    new_params[idx] = new_p
                    
                    cell = get_cell_from_params(new_params)
                    score = approximate_gap_ratio(cell)
                    
                    if score > current_score:
                        current_score = score
                        current_params = new_params
                        improved = True
            
            if not improved:
                # Try center shifts
                for idx, p in enumerate(current_params):
                    cx, cy, r, stype = p
                    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                        new_cx = cx + dx
                        new_cy = cy + dy
                        # Keep in bounds
                        if new_cx < 0 or new_cx >= size or new_cy < 0 or new_cy >= size:
                            continue
                        new_p = (new_cx, new_cy, r, stype)
                        new_params = current_params[:]
                        new_params[idx] = new_p
                        
                        cell = get_cell_from_params(new_params)
                        score = approximate_gap_ratio(cell)
                        
                        if score > current_score:
                            current_score = score
                            current_params = new_params
                            improved = True
                            
            if not improved:
                break
        
        if current_score > best_score:
            best_score = current_score
            best_params = current_params

    # Generate final cell
    if best_params is not None:
        final_cell = get_cell_from_params(best_params)
    else:
        # Fallback: simple central square
        final_cell = get_cell_from_params([(15.5, 15.5, 4.0, 0)])
        
    return final_cell.tolist()

# EVOLVE-BLOCK-END
