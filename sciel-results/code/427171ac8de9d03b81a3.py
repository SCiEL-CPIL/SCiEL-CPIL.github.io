import numpy as np
import time
import math

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    # Evaluator kills at 230s. We need to return well before.
    # Let's aim for a 150s hard stop to be safe, leaving 80s margin.
    deadline = start_time + 150.0

    # --- 1. Setup Fast Band Solver ---
    # 289 plane waves (17x17 grid)
    half_pw = 8
    G_vecs = []
    for m in range(-half_pw, half_pw + 1):
        for n in range(-half_pw, half_pw + 1):
            G_vecs.append((m, n))
    G_vecs = np.array(G_vecs) * (2.0 * np.pi)
    nG = len(G_vecs)
    
    # K-points: Use a dense grid in the irreducible BZ (triangle (0,0)-(pi,0)-(pi,pi))
    # The evaluator uses 30 points. We use a denser set to ensure we capture the gap accurately.
    # 12x12 triangle grid gives 91 points. This is fast enough.
    k_points = []
    n_k = 12
    for i in range(n_k + 1):
        for j in range(n_k + 1):
            if i + j <= n_k:
                s = i / n_k
                t = j / n_k
                k_points.append((s * np.pi, t * np.pi))
    k_points = np.array(k_points)
    
    # Precompute kG2 for all k-points and G-vectors? 
    # kG depends on k. 
    # We can precompute the G vector components squared.
    Gx = G_vecs[:, 0]
    Gy = G_vecs[:, 1]

    def compute_score(cell):
        """
        Compute the 4-5 TM band gap score for a given cell (32x32 array of 0/1).
        """
        eps_grid = eps_low + (eps_high - eps_low) * np.array(cell, dtype=float)
        
        # FFT of epsilon
        eps_F = np.fft.fft2(eps_grid) / (size * size)
        inv_eps_F = 1.0 / eps_F
        
        min_omega5 = np.inf
        max_omega4 = 0.0
        
        for kx, ky in k_points:
            kGx = kx + Gx
            kGy = ky + Gy
            kG2 = kGx**2 + kGy**2
            
            def matvec(v):
                w = kG2 * v
                u = np.fft.fft2(w.reshape(17, 17))
                z = inv_eps_F * u
                y = np.fft.ifft2(z)
                return y.flatten().real
            
            try:
                from scipy.sparse.linalg import eigsh
                from scipy.sparse import LinearOperator
                A_op = LinearOperator((nG, nG), matvec=matvec, dtype=complex)
                # We need bands 4 and 5 (indices 3 and 4). 
                # eigsh with k=6 is safe.
                vals, vecs = eigsh(A_op, k=6, which='SA', tol=1e-5)
                vals = np.sort(vals.real)
                
                if len(vals) < 5:
                    return -1.0
                
                omega4 = np.sqrt(max(0, vals[3]))
                omega5 = np.sqrt(max(0, vals[4]))
                
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

    # --- 2. Parameterized Geometry Search ---
    # Strategy: Single centered rod.
    # The cell is 32x32. Center is at (15.5, 15.5) or effectively the 4 central pixels.
    # We define the rod by a radius R.
    # To maintain integer pixel symmetry, we generate the cell by checking distance from center.
    
    # Precompute distance map from center for a 32x32 grid
    # Center is between (15,15), (15,16), (16,15), (16,16).
    # Let's use (15.5, 15.5) as the geometric center.
    yy, xx = np.mgrid[0:size, 0:size]
    dist_sq = (xx - 15.5)**2 + (yy - 15.5)**2
    
    # We will sweep the radius.
    # The maximum distance to the corner is sqrt(15.5^2 + 15.5^2) ~ 21.9
    # The maximum distance to the edge midpoint is 15.5
    # A "rod" usually implies filling the center.
    # Let's sweep R from 0 to 16.
    # Step size 0.1 for initial coarse pass, then refine.
    
    best_score = -1.0
    best_cell = None
    best_R = None
    
    # Coarse sweep
    radii = np.arange(0.5, 16.0, 0.2)
    
    for R in radii:
        if time.time() > deadline - 20:
            break
            
        # Generate cell: 1 if dist_sq <= R^2
        cell = (dist_sq <= R*R).astype(int)
        
        # Ensure strict symmetry (it should be, but float errors might cause issues on boundaries)
        # The distance formula is symmetric, so the integer mask should be symmetric.
        
        score = compute_score(cell)
        
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
            best_R = R
            
    # Refine sweep around the best R found
    if best_R is not None and time.time() < deadline - 10:
        # Refine with step 0.05
        refine_radii = np.arange(max(0.5, best_R - 1.0), min(16.0, best_R + 1.0), 0.05)
        
        for R in refine_radii:
            if time.time() > deadline - 5:
                break
                
            cell = (dist_sq <= R*R).astype(int)
            score = compute_score(cell)
            
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
                best_R = R

    # If no valid cell was found (e.g., all scores -1), fallback to a default symmetric cell
    if best_cell is None:
        # Default: 4 rods as in seed, or just a central block
        best_cell = np.zeros((size, size), dtype=int)
        # Try a few static candidates if sweep failed
        candidates = []
        # Candidate 1: Central 16x16 block
        c1 = np.zeros((size, size), dtype=int)
        c1[8:24, 8:24] = 1
        candidates.append(c1)
        
        # Candidate 2: 4 rods
        c2 = np.zeros((size, size), dtype=int)
        centers = [(8, 8), (23, 8), (8, 23), (23, 23)]
        radius = 4.5
        for i in range(size):
            for j in range(size):
                for cx, cy in centers:
                    if (j-cx)**2 + (i-cy)**2 <= radius*radius:
                        c2[i, j] = 1
                        break
        candidates.append(c2)
        
        for c in candidates:
            s = compute_score(c)
            if s > best_score:
                best_score = s
                best_cell = c.copy()

    # Final safety check: Ensure symmetry
    def is_symmetric(cell):
        if not np.array_equal(cell, cell[::-1, :]): return False
        if not np.array_equal(cell, cell[:, ::-1]): return False
        if not np.array_equal(cell, cell.T): return False
        return True

    if not is_symmetric(best_cell):
        # Force symmetry by averaging with its symmetric counterparts? 
        # No, just revert to the most symmetric simple pattern if this happens.
        # But our generation method (dist_sq <= R^2) is inherently symmetric.
        # So this check should pass.
        pass

    return best_cell.tolist()
