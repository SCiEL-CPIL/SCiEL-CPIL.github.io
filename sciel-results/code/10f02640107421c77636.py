import time
import math
import numpy as np

def design(problem):
    """Design a 32x32 photonic crystal unit cell maximizing the 4-5 TM band gap."""
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin within 120s budget
    
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    # Plane-wave setup: 289 = 17x17
    N = 17
    nmax = (N - 1) // 2  # 8
    
    # Generate G-vectors as 2D array
    G = np.array([(mx, my) for mx in range(-nmax, nmax+1) for my in range(-nmax, nmax+1)], dtype=float)
    NG = len(G)  # 289
    
    # Precompute dG indices: dG[i,j] = G[i] - G[j], mapped to F indices
    # dG_idx[i,j] = (dGx + nmax, dGy + nmax) if in range, else (-1,-1)
    dGx_arr = G[:, 0:1] - G[1:, 0].T  # shape (NG, NG-1)
    dGy_arr = G[:, 1:2] - G[1:, 1].T  # shape (NG, NG-1)
    # Full dG matrix
    dGx_full = np.zeros((NG, NG), dtype=int)
    dGy_full = np.zeros((NG, NG), dtype=int)
    for i in range(NG):
        for j in range(NG):
            dGx_full[i, j] = int(G[i, 0] - G[j, 0])
            dGy_full[i, j] = int(G[i, 1] - G[j, 1])
    
    # Precompute F index mapping: for each (dGx, dGy), what index in F do we use?
    # F is indexed [0..N-1, 0..N-1] corresponding to gmx, gmy in [-nmax, nmax]
    # dGx, dGy range from -16 to 16, but we only have -8 to 8
    f_idx_x = np.zeros((NG, NG), dtype=int)
    f_idx_y = np.zeros((NG, NG), dtype=int)
    f_valid = np.zeros((NG, NG), dtype=bool)
    for i in range(NG):
        for j in range(NG):
            dx = int(G[i, 0] - G[j, 0])
            dy = int(G[i, 1] - G[j, 1])
            fi = dx + nmax
            fj = dy + nmax
            if 0 <= fi < N and 0 <= fj < N:
                f_idx_x[i, j] = fi
                f_idx_y[i, j] = fj
                f_valid[i, j] = True
    
    # Precompute phase vectors for Fourier computation
    i_idx = np.arange(size)
    j_idx = np.arange(size)
    phase_vectors = {}
    for g in range(-nmax, nmax+1):
        phase_vectors[g] = np.exp(-1j * 2 * np.pi * g * (i_idx + 0.5) / size)
    
    def compute_fourier_fast(inv_eps_2d):
        """Compute Fourier coefficients of 1/eps using precomputed phases.
        inv_eps_2d is a 32x32 numpy array."""
        F = np.zeros((N, N), dtype=complex)
        for mx in range(N):
            gmx = mx - nmax
            pi = phase_vectors[gmx]
            for my in range(N):
                gmy = my - nmax
                pj = phase_vectors[gmy]
                F[mx, my] = (pi @ inv_eps_2d @ pj) / (size * size)
        return F
    
    def solve_bands_fast(F):
        """Vectorized eigenvalue solve for TM bands. Returns gap-midgap ratio."""
        # k-points in irreducible BZ (30 points)
        k_points = []
        for t in np.linspace(0, 1, 7):
            k_points.append((t * np.pi, 0))
        for t in np.linspace(1, 0, 7):
            k_points.append((np.pi, t * np.pi))
        for t in np.linspace(1, 0, 7):
            k_points.append((t * np.pi, t * np.pi))
        k_points.append((np.pi/2, np.pi/4))
        k_points.append((np.pi/2, np.pi/2))
        k_points.append((3*np.pi/4, np.pi/4))
        
        max_omega4 = 0.0
        min_omega5 = np.inf
        
        # Precompute (G-k)^2 for all k-points: shape (n_k, NG)
        k_arr = np.array(k_points)  # (n_k, 2)
        # (G_i - k)^2 = (Gx_i - kx)^2 + (Gy_i - ky)^2
        # G[:, 0:1] - k_arr[:, 0:1].T  -> (NG, n_k)
        diff_x = G[:, 0:1].T - k_arr[:, 0:1]  # (n_k, NG)
        diff_y = G[:, 1:2].T - k_arr[:, 1:2]  # (n_k, NG)
        Gk2 = diff_x**2 + diff_y**2  # (n_k, NG)
        
        for ki in range(len(k_points)):
            # Build A matrix: A[i,j] = Gk2[ki, i] * F[f_idx_x[i,j], f_idx_y[i,j]] * f_valid[i,j]
            # F values for all (i,j) pairs
            F_vals = np.zeros((NG, NG), dtype=complex)
            F_vals[f_valid] = F[f_idx_x[f_valid], f_idx_y[f_valid]]
            
            A = Gk2[ki, :].astype(complex)[:, np.newaxis] * F_vals
            
            # A should be Hermitian; enforce numerically
            A = 0.5 * (A + A.conj().T)
            
            try:
                eigvals = np.linalg.eigvalsh(A)
                eigvals = np.sort(eigvals)
                # Filter positive
                pos_mask = eigvals > 1e-10
                eigvals_pos = eigvals[pos_mask]
                if len(eigvals_pos) >= 5:
                    omega4 = math.sqrt(eigvals_pos[3])
                    omega5 = math.sqrt(eigvals_pos[4])
                    if omega4 > max_omega4:
                        max_omega4 = omega4
                    if omega5 < min_omega5:
                        min_omega5 = omega5
            except:
                pass
        
        if min_omega5 == np.inf or max_omega4 <= 0:
            return 0.0
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2
        if midgap <= 0:
            return 0.0
        return gap / midgap
    
    # --- Symmetry utilities ---
    def fund_to_cell(fund):
        """Convert 8x8 fundamental domain to full 32x32 cell with D4 symmetry."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(8):
            for j in range(8):
                val = fund[i, j]
                cell[i, j] = val
                cell[i, size-1-j] = val
                cell[size-1-i, j] = val
                cell[size-1-i, size-1-j] = val
                cell[j, i] = val
                cell[j, size-1-i] = val
                cell[size-1-j, i] = val
                cell[size-1-j, size-1-i] = val
        return cell
    
    def cell_to_fund(cell):
        """Extract 8x8 fundamental domain from 32x32 cell."""
        return cell[:8, :8].copy()
    
    def evaluate_cell(cell_np):
        """Full evaluation: cell -> Fourier -> bands -> score."""
        eps = np.where(cell_np == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        F = compute_fourier_fast(inv_eps)
        return solve_bands_fast(F)
    
    # --- Generate diverse initial candidates ---
    def make_rod(fund_size=8, radius=0.3):
        """Square rod in fundamental domain."""
        fund = np.zeros((8, 8), dtype=int)
        cx, cy = 3.5, 3.5
        for i in range(8):
            for j in range(8):
                if abs(i - cx) < radius * 4 and abs(j - cy) < radius * 4:
                    fund[i, j] = 1
        return fund
    
    def make_hole(hole_radius=0.2):
        """Dielectric background with air hole."""
        fund = np.ones((8, 8), dtype=int)
        cx, cy = 3.5, 3.5
        for i in range(8):
            for j in range(8):
                if abs(i - cx) < hole_radius * 4 and abs(j - cy) < hole_radius * 4:
                    fund[i, j] = 0
        return fund
    
    def make_cross(arm_width=2):
        """Cross pattern."""
        fund = np.zeros((8, 8), dtype=int)
        half = arm_width // 2
        for i in range(8):
            for j in range(8):
                if abs(i - 3.5) < half + 1 or abs(j - 3.5) < half + 1:
                    fund[i, j] = 1
        return fund
    
    def make_ring(inner_r, outer_r):
        """Ring pattern."""
        fund = np.zeros((8, 8), dtype=int)
        cx, cy = 3.5, 3.5
        for i in range(8):
            for j in range(8):
                r = math.sqrt((i-cx)**2 + (j-cy)**2)
                if inner_r < r < outer_r:
                    fund[i, j] = 1
        return fund
    
    def make_checkerboard(block=2):
        """Checkerboard pattern."""
        fund = np.zeros((8, 8), dtype=int)
        for i in range(8):
            for j in range(8):
                if (i // block + j // block) % 2 == 0:
                    fund[i, j] = 1
        return fund
    
    # Generate candidates
    candidates = []
    for r in [0.25, 0.3, 0.35, 0.4, 0.45]:
        candidates.append(make_rod(radius=r))
    for hr in [0.15, 0.2, 0.25, 0.3]:
        candidates.append(make_hole(hole_radius=hr))
    for aw in [2, 3, 4]:
        candidates.append(make_cross(arm_width=aw))
    candidates.append(make_ring(1.5, 3.0))
    candidates.append(make_ring(2.0, 3.5))
    candidates.append(make_checkerboard(block=2))
    candidates.append(make_checkerboard(block=3))
    
    # Evaluate all candidates
    best_score = -1.0
    best_fund = None
    
    for fund in candidates:
        cell = fund_to_cell(fund)
        score = evaluate_cell(cell)
        if score > best_score:
            best_score = score
            best_fund = fund.copy()
    
    # --- Local search: multistart with random restarts ---
    # Use the fast evaluator to do many iterations
    rng = np.random.default_rng(42)
    
    current_fund = best_fund.copy()
    current_score = best_score
    
    # Phase 1: Steep hill climbing from best candidate
    no_improve = 0
    max_no_improve = 50
    while time.time() < deadline - 20 and no_improve < max_no_improve:
        # Pick random pixel in fundamental domain
        i = rng.integers(0, 8)
        j = rng.integers(0, 8)
        
        old_val = current_fund[i, j]
        current_fund[i, j] = 1 - old_val
        
        cell = fund_to_cell(current_fund)
        new_score = evaluate_cell(cell)
        
        if new_score > current_score:
            current_score = new_score
            no_improve = 0
            if new_score > best_score:
                best_score = new_score
                best_fund = current_fund.copy()
        else:
            current_fund[i, j] = old_val
            no_improve += 1
    
    # Phase 2: Random restarts with short hill climbs
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Estimate time per evaluation
        t0 = time.time()
        test_fund = rng.integers(0, 2, (8, 8))
        test_cell = fund_to_cell(test_fund)
        _ = evaluate_cell(test_cell)
        eval_time = time.time() - t0
        
        # How many full hill climbs can we do?
        # Each hill climb: ~30-50 evaluations
        n_restarts = max(1, int(remaining_time / (eval_time * 40)))
        n_restarts = min(n_restarts, 30)
        
        for restart in range(n_restarts):
            if time.time() >= deadline - 10:
                break
            
            # Start from random perturbation of best
            fund_r = best_fund.copy()
            # Flip 5-15 random pixels
            n_flips = rng.integers(5, 16)
            for _ in range(n_flips):
                fi = rng.integers(0, 8)
                fj = rng.integers(0, 8)
                fund_r[fi, fj] = 1 - fund_r[fi, fj]
            
            cell_r = fund_to_cell(fund_r)
            score_r = evaluate_cell(cell_r)
            
            # Short hill climb
            no_imp_r = 0
            while time.time() < deadline - 10 and no_imp_r < 30:
                i = rng.integers(0, 8)
                j = rng.integers(0, 8)
                old_val = fund_r[i, j]
                fund_r[i, j] = 1 - old_val
                
                cell_r = fund_to_cell(fund_r)
                new_score_r = evaluate_cell(cell_r)
                
                if new_score_r > score_r:
                    score_r = new_score_r
                    no_imp_r = 0
                    if new_score_r > best_score:
                        best_score = new_score_r
                        best_fund = fund_r.copy()
                else:
                    fund_r[i, j] = old_val
                    no_imp_r += 1
    
    # Phase 3: Final polish - try all single-pixel flips from best
    if time.time() < deadline - 5:
        for i in range(8):
            for j in range(8):
                if time.time() >= deadline - 5:
                    break
                old_val = best_fund[i, j]
                best_fund[i, j] = 1 - old_val
                cell = fund_to_cell(best_fund)
                score = evaluate_cell(cell)
                if score > best_score:
                    best_score = score
                else:
                    best_fund[i, j] = old_val
    
    # Convert back to nested list
    final_cell = fund_to_cell(best_fund)
    result = [[int(final_cell[i, j]) for j in range(size)] for i in range(size)]
    
    # Verify symmetry
    for i in range(size):
        for j in range(size):
            assert result[i][j] == result[size-1-i][size-1-j], f"Symmetry fail at ({i},{j})"
            assert result[i][j] == result[j][i], f"Transpose fail at ({i},{j})"
            assert result[i][j] == result[size-1-i][j], f"Row flip fail at ({i},{j})"
    
    return result
