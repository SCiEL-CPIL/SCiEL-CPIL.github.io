import time
import math
import numpy as np

def design(problem):
    """Design a 32x32 photonic crystal unit cell maximizing the 4-5 TM band gap."""
    start_time = time.time()
    deadline = start_time + 210  # safety margin within 230s budget

    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)

    # Plane-wave setup: 289 = 17x17
    N = 17
    nmax = (N - 1) // 2  # 8

    # Generate G-vectors
    G = np.array([(mx, my) for mx in range(-nmax, nmax + 1) for my in range(-nmax, nmax + 1)], dtype=float)
    NG = len(G)  # 289

    # Precompute matrix structure: A[i,j] = |Gi-k|^2 * F[Gi-Gj]
    # Store pairs (i, j, gi_k_sq_factor, dGx, dGy) where dG is in range
    # Actually, for each k, A[i,j] = (Gi-kx)^2 + (Gi-ky)^2) * F[Gi-Gj]
    # Precompute which (i,j) pairs have dG in range, and their dG indices
    matrix_pairs = []  # list of (i, j, fi, fj) where fi,dGx+nmax, fj=dGy+nmax
    for i in range(NG):
        for j in range(NG):
            dGx = int(G[i][0] - G[j][0])
            dGy = int(G[i][1] - G[j][1])
            fi = dGx + nmax
            fj = dGy + nmax
            if 0 <= fi < N and 0 <= fj < N:
                matrix_pairs.append((i, j, fi, fj))
    matrix_pairs = np.array(matrix_pairs, dtype=int)
    MP = len(matrix_pairs)

    # Precompute |Gi|^2 and Gi components for fast k-dependent factor
    Gi_sq = G[:, 0]**2 + G[:, 1]**2
    Gix = G[:, 0]
    Giy = G[:, 1]

    # k-points in irreducible BZ (30 points as specified)
    k_points = []
    # Along Gamma-X: (0,0) to (pi,0) - 8 points
    for t in np.linspace(0, 1, 8):
        k_points.append((t * np.pi, 0))
    # Along X-M: (pi,0) to (pi,pi) - 8 points
    for t in np.linspace(0, 1, 8):
        k_points.append((np.pi, t * np.pi))
    # Along M-Gamma: (pi,pi) to (0,0) - 8 points
    for t in np.linspace(0, 1, 8):
        k_points.append((t * np.pi, t * np.pi))
    # 6 interior points
    k_points.append((np.pi/2, np.pi/4))
    k_points.append((3*np.pi/4, np.pi/4))
    k_points.append((np.pi/2, np.pi/2))
    k_points.append((np.pi/4, np.pi/4))
    k_points.append((3*np.pi/4, 3*np.pi/4))
    k_points.append((np.pi/4, np.pi/2))
    k_points = np.array(k_points)
    NK = len(k_points)

    def compute_fourier(cell):
        """Compute Fourier coefficients of 1/eps for the cell using FFT."""
        eps = np.where(np.array(cell) == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        F = np.fft.fft2(inv_eps) / (size * size)
        # Extract the 17x17 block centered at zero frequency
        F_block = np.zeros((N, N), dtype=complex)
        for mx in range(N):
            gmx = mx - nmax
            fi = gmx % size
            for my in range(N):
                gmy = my - nmax
                fj = gmy % size
                F_block[mx, my] = F[fi, fj]
        return F_block

    def solve_bands_fast(F):
        """Solve the eigenvalue problem for TM bands at given k-points using precomputed structure."""
        max_omega4 = 0.0
        min_omega5 = np.inf

        for kx, ky in k_points:
            # Compute |Gi - k|^2 for all i
            Gi_k_sq = (Gix - kx)**2 + (Giy - ky)**2
            
            # Build matrix using precomputed pairs
            # A[i,j] = Gi_k_sq[i] * F[fi, fj]
            i_idx = matrix_pairs[:, 0]
            j_idx = matrix_pairs[:, 1]
            fi_idx = matrix_pairs[:, 2]
            fj_idx = matrix_pairs[:, 3]
            
            # F values for all pairs
            F_vals = F[fi_idx, fj_idx]
            
            # A[i,j] = Gi_k_sq[i] * F[fi,fj]
            A_vals = Gi_k_sq[i_idx] * F_vals
            
            # Build sparse-like matrix
            A = np.zeros((NG, NG), dtype=complex)
            A[i_idx, j_idx] = A_vals
            
            # Make Hermitian (should already be if F is correct, but ensure)
            A = (A + A.conj().T) / 2.0
            
            # Solve eigenvalue problem
            try:
                eigvals = np.linalg.eigvalsh(A)
                eigvals = np.sort(eigvals)
                eigvals = eigvals[eigvals > 1e-10]
                if len(eigvals) >= 5:
                    omega4 = np.sqrt(eigvals[3])
                    omega5 = np.sqrt(eigvals[4])
                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
            except:
                pass

        if min_omega5 == np.inf:
            return 0.0
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2
        if midgap <= 0:
            return 0.0
        return gap / midgap

    # Symmetric cell generation from 8x8 fundamental domain
    def fund_to_cell(fund):
        """Create full 32x32 cell from 8x8 fundamental domain with C4v symmetry."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(8):
            for j in range(8):
                val = fund[i][j]
                # The 8x8 fundamental domain maps to 4 quadrants via symmetry
                cell[i, j] = val
                cell[i, size - 1 - j] = val
                cell[size - 1 - i, j] = val
                cell[size - 1 - i, size - 1 - j] = val
                cell[j, i] = val
                cell[j, size - 1 - i] = val
                cell[size - 1 - j, i] = val
                cell[size - 1 - j, size - 1 - j] = val
        return cell

    # Generate diverse initial candidates
    def make_dielectric_bg_holes(hole_half_width):
        """Dielectric background with square air hole."""
        fund = np.ones((8, 8), dtype=int)
        for i in range(8):
            for j in range(8):
                if abs(i - 3.5) < hole_half_width and abs(j - 3.5) < hole_half_width:
                    fund[i][j] = 0
        return fund

    def make_dielectric_rod(rod_half_width):
        """Air background with dielectric rod."""
        fund = np.zeros((8, 8), dtype=int)
        for i in range(8):
            for j in range(8):
                if abs(i - 3.5) < rod_half_width and abs(j - 3.5) < rod_half_width:
                    fund[i][j] = 1
        return fund

    def make_cross_dielectric(arm_half_width):
        """Cross-shaped dielectric in air background."""
        fund = np.zeros((8, 8), dtype=int)
        for i in range(8):
            for j in range(8):
                if abs(i - 3.5) < arm_half_width or abs(j - 3.5) < arm_half_width:
                    fund[i][j] = 1
        return fund

    def make_air_cross_dielectric_bg(arm_half_width):
        """Air cross in dielectric background."""
        fund = np.ones((8, 8), dtype=int)
        for i in range(8):
            for j in range(8):
                if abs(i - 3.5) < arm_half_width or abs(j - 3.5) < arm_half_width:
                    fund[i][j] = 0
        return fund

    def make_ring(hole_inner, hole_outer):
        """Dielectric ring in air background."""
        fund = np.zeros((8, 8), dtype=int)
        for i in range(8):
            for j in range(8):
                dist = max(abs(i - 3.5), abs(j - 3.5))
                if hole_inner <= dist < hole_outer:
                    fund[i][j] = 1
        return fund

    def make_diagonal_pattern():
        """Diagonal stripe pattern."""
        fund = np.zeros((8, 8), dtype=int)
        for i in range(8):
            for j in range(8):
                if (i + j) % 3 == 0:
                    fund[i][j] = 1
        return fund

    # Generate candidate funds
    candidates_fund = []
    # Dielectric background with air holes
    for hw in [1.0, 1.5, 2.0, 2.5, 3.0]:
        candidates_fund.append(make_dielectric_bg_holes(hw))
    # Air background with dielectric rods
    for rw in [1.0, 1.5, 2.0, 2.5]:
        candidates_fund.append(make_dielectric_rod(rw))
    # Cross patterns
    for aw in [1.0, 1.5, 2.0]:
        candidates_fund.append(make_cross_dielectric(aw))
        candidates_fund.append(make_air_cross_dielectric_bg(aw))
    # Ring patterns
    candidates_fund.append(make_ring(1.5, 3.0))
    candidates_fund.append(make_ring(2.0, 3.5))
    candidates_fund.append(make_ring(1.0, 2.5))
    # Diagonal
    candidates_fund.append(make_diagonal_pattern())
    # Random candidates
    np.random.seed(42)
    for _ in range(20):
        fund = np.random.randint(0, 2, (8, 8))
        candidates_fund.append(fund)

    # Evaluate all candidates
    best_score = -1
    best_fund = None

    for fund in candidates_fund:
        cell = fund_to_cell(fund)
        F = compute_fourier(cell)
        score = solve_bands_fast(F)
        if score > best_score:
            best_score = score
            best_fund = fund.copy()

    # Fast local search: try all single-pixel flips, accept best
    current_fund = best_fund.copy()
    current_score = best_score
    best_ls_score = best_score
    best_ls_fund = best_fund.copy()

    # Precompute cell for current fund
    current_cell = fund_to_cell(current_fund)
    current_F = compute_fourier(current_cell)

    # Hill climbing with all single-flip moves
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        best_neighbor_score = current_score
        best_neighbor_fund = None

        for i in range(8):
            if time.time() > deadline - 30:
                break
            for j in range(8):
                if time.time() > deadline - 30:
                    break
                # Try flipping pixel (i,j)
                old_val = current_fund[i][j]
                current_fund[i][j] = 1 - old_val

                cell = fund_to_cell(current_fund)
                F = compute_fourier(cell)
                score = solve_bands_fast(F)

                if score > best_neighbor_score:
                    best_neighbor_score = score
                    best_neighbor_fund = current_fund.copy()
                else:
                    current_fund[i][j] = old_val

        if best_neighbor_fund is not None:
            current_fund = best_neighbor_fund
            current_score = best_neighbor_score
            improved = True
            if current_score > best_ls_score:
                best_ls_score = current_score
                best_ls_fund = current_fund.copy()

    # Simulated annealing for further improvement
    sa_fund = best_ls_fund.copy()
    sa_score = best_ls_score
    best_sa_score = best_ls_score
    best_sa_fund = best_ls_fund.copy()

    temp = 0.03
    cooling = 0.999
    min_temp = 0.002
    no_improve_count = 0
    max_no_improve = 200

    while time.time() < deadline - 5 and temp > min_temp:
        i = np.random.randint(0, 8)
        j = np.random.randint(0, 8)

        old_val = sa_fund[i][j]
        sa_fund[i][j] = 1 - old_val

        cell = fund_to_cell(sa_fund)
        F = compute_fourier(cell)
        new_score = solve_bands_fast(F)

        delta = new_score - sa_score
        if new_score > sa_score or np.random.random() < math.exp(delta / temp):
            sa_score = new_score
            if new_score > best_sa_score:
                best_sa_score = new_score
                best_sa_fund = sa_fund.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            sa_fund[i][j] = old_val
            no_improve_count += 1

        temp *= cooling

        if no_improve_count > max_no_improve:
            # Restart with larger perturbation
            for _ in range(5):
                ri = np.random.randint(0, 8)
                rj = np.random.randint(0, 8)
                sa_fund[ri][rj] = 1 - sa_fund[ri][rj]
            cell = fund_to_cell(sa_fund)
            F = compute_fourier(cell)
            sa_score = solve_bands_fast(F)
            no_improve_count = 0
            temp = max(temp * 1.5, 0.01)

    # Final local search from best SA result
    final_fund = best_sa_fund.copy()
    final_score = best_sa_score
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        best_neighbor_score = final_score
        best_neighbor_fund = None

        for i in range(8):
            if time.time() > deadline - 2:
                break
            for j in range(8):
                if time.time() > deadline - 2:
                    break
                old_val = final_fund[i][j]
                final_fund[i][j] = 1 - old_val

                cell = fund_to_cell(final_fund)
                F = compute_fourier(cell)
                score = solve_bands_fast(F)

                if score > best_neighbor_score:
                    best_neighbor_score = score
                    best_neighbor_fund = final_fund.copy()
                else:
                    final_fund[i][j] = old_val

        if best_neighbor_fund is not None:
            final_fund = best_neighbor_fund
            final_score = best_neighbor_score
            improved = True

    # Return the best cell found
    best_cell = fund_to_cell(final_fund)

    # Verify symmetry
    for i in range(size):
        for j in range(size):
            if best_cell[i][j] != best_cell[size-1-i][size-1-j]:
                # Fix symmetry violation
                best_cell[i][j] = best_cell[size-1-i][size-1-j]
                best_cell[j][i] = best_cell[size-1-i][size-1-j]
                best_cell[j][size-1-i] = best_cell[size-1-i][size-1-j]
                best_cell[size-1-j][i] = best_cell[size-1-i][size-1-j]
                best_cell[size-1-j][size-1-i] = best_cell[size-1-i][size-1-j]

    return best_cell.tolist()
