import time
import math
import numpy as np

def design(problem):
    """Design a 32x32 photonic crystal unit cell maximizing the 4-5 TM band gap."""
    start_time = time.time()
    deadline = start_time + 115  # safety margin within 120s budget

    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)

    # Plane-wave setup: 289 = 17x17
    N = 17
    nmax = (N - 1) // 2  # 8

    # Generate G-vectors
    G = np.array([(mx, my) for mx in range(-nmax, nmax + 1) for my in range(-nmax, nmax + 1)], dtype=float)
    NG = len(G)  # 289

    # Precompute G index lookup for fast matrix construction
    G_index = {}
    for i, (gx, gy) in enumerate(G):
        G_index[(int(gx), int(gy))] = i

    def compute_fourier(cell):
        """Compute Fourier coefficients of 1/eps for the cell using FFT."""
        eps = np.where(np.array(cell) == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        F = np.fft.fft2(inv_eps) / (size * size)
        # Extract the 17x17 block centered at zero frequency
        # FFT output has frequencies 0,1,...,31; we need -8,...,8
        F_block = np.zeros((N, N), dtype=complex)
        for mx in range(N):
            gmx = mx - nmax
            fi = gmx % size
            for my in range(N):
                gmy = my - nmax
                fj = gmy % size
                F_block[mx, my] = F[fi, fj]
        return F_block

    # Precompute k-points in irreducible BZ
    k_points = []
    # Along Gamma-X: (0,0) to (pi,0)
    for t in np.linspace(0, 1, 8):
        k_points.append((t * np.pi, 0))
    # Along X-M: (pi,0) to (pi,pi)
    for t in np.linspace(0, 1, 8):
        k_points.append((np.pi, t * np.pi))
    # Along M-Gamma: (pi,pi) to (0,0)
    for t in np.linspace(0, 1, 8):
        k_points.append((t * np.pi, t * np.pi))
    # Interior points
    k_points.append((np.pi/2, np.pi/4))
    k_points.append((3*np.pi/4, np.pi/4))
    k_points.append((np.pi/2, np.pi/2))
    k_points = np.array(k_points)
    NK = len(k_points)

    def solve_bands(F):
        """Solve the eigenvalue problem for TM bands at given k-points."""
        max_omega4 = 0.0
        min_omega5 = np.inf

        for kx, ky in k_points:
            # Build the matrix efficiently
            A = np.zeros((NG, NG), dtype=complex)
            for i in range(NG):
                Gi = G[i]
                Gi_k2 = (Gi[0] - kx)**2 + (Gi[1] - ky)**2
                if Gi_k2 < 1e-10:
                    continue
                for j in range(NG):
                    Gj = G[j]
                    dGx = int(Gi[0] - Gj[0])
                    dGy = int(Gi[1] - Gj[1])
                    fi = dGx + nmax
                    fj = dGy + nmax
                    if 0 <= fi < N and 0 <= fj < N:
                        Fval = F[fi, fj]
                        if abs(Fval) > 1e-15:
                            A[i, j] = Gi_k2 * Fval

            # Solve eigenvalue problem: A c = omega^2 c
            try:
                eigvals = np.linalg.eigvalsh(A)
                eigvals = np.sort(eigvals)
                eigvals = eigvals[eigvals > 1e-10]
                if len(eigvals) >= 5:
                    omega4 = np.sqrt(eigvals[3])  # 4th band (0-indexed: 3)
                    omega5 = np.sqrt(eigvals[4])  # 5th band
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
                # Top-left: (i, j)
                cell[i, j] = val
                # Top-right mirror: (i, 31-j)
                cell[i, size - 1 - j] = val
                # Bottom-left mirror: (31-i, j)
                cell[size - 1 - i, j] = val
                # Bottom-right: (31-i, 31-j)
                cell[size - 1 - i, size - 1 - j] = val
                # Transpose symmetry: (j, i)
                cell[j, i] = val
                cell[j, size - 1 - i] = val
                cell[size - 1 - j, i] = val
                cell[size - 1 - j, size - 1 - i] = val
        return cell

    def cell_to_fund(cell):
        """Extract 8x8 fundamental domain from full cell."""
        fund = np.zeros((8, 8), dtype=int)
        for i in range(8):
            for j in range(8):
                fund[i][j] = cell[i][j]
        return fund

    # Generate diverse initial candidates
    def make_dielectric_bg_holes(hole_half_widths):
        """Dielectric background with square air holes of various sizes."""
        fund = np.ones((8, 8), dtype=int)
        for hw in hole_half_widths:
            for i in range(8):
                for j in range(8):
                    # Hole centered in the 8x8 domain
                    if abs(i - 3.5) < hw and abs(j - 3.5) < hw:
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

    # Generate candidate funds
    candidates_fund = []
    # Dielectric background with air holes (recommended approach)
    candidates_fund.append(make_dielectric_bg_holes([1.0]))
    candidates_fund.append(make_dielectric_bg_holes([1.5]))
    candidates_fund.append(make_dielectric_bg_holes([2.0]))
    candidates_fund.append(make_dielectric_bg_holes([2.5]))
    # Air background with dielectric rods
    candidates_fund.append(make_dielectric_rod(1.0))
    candidates_fund.append(make_dielectric_rod(1.5))
    candidates_fund.append(make_dielectric_rod(2.0))
    # Cross patterns
    candidates_fund.append(make_cross_dielectric(1.0))
    candidates_fund.append(make_cross_dielectric(1.5))
    candidates_fund.append(make_air_cross_dielectric_bg(1.0))
    candidates_fund.append(make_air_cross_dielectric_bg(1.5))

    # Evaluate all candidates
    best_score = -1
    best_fund = None

    for fund in candidates_fund:
        cell = fund_to_cell(fund)
        F = compute_fourier(cell)
        score = solve_bands(F)
        if score > best_score:
            best_score = score
            best_fund = fund.copy()

    # Simulated annealing on the 8x8 fundamental domain
    current_fund = best_fund.copy()
    current_score = best_score
    best_sa_score = best_score
    best_sa_fund = best_fund.copy()

    temp = 0.05
    cooling = 0.998
    min_temp = 0.001
    no_improve_count = 0
    max_no_improve = 500

    while time.time() < deadline and temp > min_temp:
        # Pick a random pixel in the fundamental domain
        i = np.random.randint(0, 8)
        j = np.random.randint(0, 8)

        # Flip it
        old_val = current_fund[i][j]
        current_fund[i][j] = 1 - old_val

        cell = fund_to_cell(current_fund)
        F = compute_fourier(cell)
        new_score = solve_bands(F)

        delta = new_score - current_score
        if new_score > current_score or np.random.random() < math.exp(delta / temp):
            current_score = new_score
            if new_score > best_sa_score:
                best_sa_score = new_score
                best_sa_fund = current_fund.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            current_fund[i][j] = old_val
            no_improve_count += 1

        temp *= cooling

        # Restart if stuck
        if no_improve_count > max_no_improve:
            # Try a larger perturbation
            for _ in range(10):
                ri = np.random.randint(0, 8)
                rj = np.random.randint(0, 8)
                current_fund[ri][rj] = 1 - current_fund[ri][rj]
            cell = fund_to_cell(current_fund)
            F = compute_fourier(cell)
            current_score = solve_bands(F)
            no_improve_count = 0
            temp = max(temp * 2, 0.02)

    # Return the best cell found
    best_cell = fund_to_cell(best_sa_fund)

    # Verify symmetry
    for i in range(size):
        for j in range(size):
            assert best_cell[i][j] == best_cell[size-1-i][size-1-j], f"Symmetry fail at ({i},{j})"
            assert best_cell[i][j] == best_cell[j][i], f"Transpose fail at ({i},{j})"
            assert best_cell[i][j] == best_cell[size-1-i][j], f"Mirror fail at ({i},{j})"

    return best_cell.tolist()
