"""Photonic-crystal unit-cell design: maximize TM band-4/5 gap-midgap ratio."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def design(problem):
    """Design a 32x32 symmetric unit cell to maximize TM band-4/5 gap-midgap ratio."""
    start_time = time.time()
    deadline = 220.0  # safety margin before 230s kill

    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)

    # The cell must be symmetric: cell == cell[::-1] rows, cols, and transpose
    # This means the cell is determined by its top-left quadrant (16x16)
    # with the constraint that it's symmetric about the center.
    # For a 32x32 cell, the center is at (16,16) in 0-indexed pixel centers.
    # Actually, for even size, symmetry about cell center means:
    # cell[i][j] == cell[31-i][31-j] (180° rotation)
    # cell[i][j] == cell[31-i][j] (horizontal flip about vertical axis)
    # cell[i][j] == cell[i][31-j] (vertical flip about horizontal axis)
    # cell[i][j] == cell[j][i] (transpose)
    # Together these give D4 symmetry. The fundamental domain is 1/8 of the cell.
    # For 32x32, the fundamental domain is 16x8 (top-left quadrant, left half).
    # Actually let me think more carefully.
    # D4 for even-sized grid: pixels (i,j) for i in [0,15], j in [0,15] determine the rest.
    # But with transpose symmetry, (i,j) and (j,i) are the same.
    # So the fundamental domain is i in [0,15], j in [0,15], with (i,j)~(j,i).
    # That's 16*16/2 + 16 = 136 independent pixels. Still too many for exhaustive search.

    # Strategy: Use a parametric family and optimize.
    # Known good designs for TM band gaps in square photonic crystals:
    # 1. Dielectric rods (circle) - seed
    # 2. Air holes in dielectric (inverted)
    # 3. Cross/diamond shapes
    # 4. Square holes
    # 5. Combined shapes

    # Let me implement a plane-wave solver to evaluate candidate cells,
    # then do a local search.

    def make_cell_from_quadrant(quadrant):
        """Expand a 16x8 fundamental domain to full 32x32 D4-symmetric cell."""
        # The fundamental domain for D4 symmetry on 32x32:
        # For even N=32, the symmetry group D4 acts on the 32x32 grid.
        # cell[i][j] is determined by cell[i][j] for i in [0,15], j in [0,15]
        # with the constraint cell[i][j] = cell[j][i].
        # So we need 16*16 = 256 values with 136 independent (upper triangle of 16x16).
        # Let me just work with the full 16x16 top-left quadrant and enforce transpose.
        # Actually, let me just define the cell by its 16x16 top-left quadrant
        # and enforce all symmetries.
        N = 32
        cell = [[0]*N for _ in range(N)]
        # quadrant[i][j] for i in [0,15], j in [0,15]
        for i in range(16):
            for j in range(16):
                val = quadrant[i][j]
                cell[i][j] = val
                cell[i][N-1-j] = val
                cell[N-1-i][j] = val
                cell[N-1-i][N-1-j] = val
        # Now enforce transpose: cell[i][j] = cell[j][i]
        # This may conflict with the above. Let me just build it properly.
        # Better approach: define by 16x8 fundamental domain.
        # For D4 on 32x32:
        # (i,j) -> (j,i) [transpose]
        # (i,j) -> (N-1-i, j) [flip x]
        # (i,j) -> (i, N-1-j) [flip y]
        # (i,j) -> (N-1-i, N-1-j) [rotate 180]
        # The orbit of (i,j) under D4 has up to 8 elements.
        # For the fundamental domain, take i in [0,15], j in [0,7] (since j and N-1-j pair up,
        # and i and N-1-i pair up, and (i,j)~(j,i) means j < i for i>7, but j can be up to 15).
        # This is getting complex. Let me just use the 16x16 top-left and enforce all symmetries
        # by averaging or by construction.

        # Simpler: just return the 16x16 quadrant expanded with 4-fold + reflection symmetry
        # and then enforce transpose by symmetrizing.
        N = 32
        cell = [[0]*N for _ in range(N)]
        for i in range(16):
            for j in range(16):
                val = quadrant[i][j]
                cell[i][j] = val
                cell[N-1-i][j] = val
                cell[i][N-1-j] = val
                cell[N-1-i][N-1-j] = val
        # Enforce transpose: for i<j, set both to the same value
        # Use the value from min(i,j), max(i,j) position
        for i in range(N):
            for j in range(i+1, N):
                # Average or just take one
                # To maintain the other symmetries, take the value that's consistent
                # cell[i][j] should equal cell[j][i]
                # Set both to the average (rounded) or just pick one
                # Let's just set cell[i][j] = cell[j][i] = (cell[i][j] + cell[j][i]) // 2
                # But this might break the 4-fold symmetry.
                # Alternative: define the cell ONLY by the fundamental domain of D4.
                # For 32x32, D4 fundamental domain: i in [0,15], j in [0, min(15, i)]
                # i.e., upper triangle of the 16x16 top-left quadrant.
                # Number of independent pixels: sum_{i=0}^{15} (i+1) = 136
                pass

        # Let me redo this properly.
        N = 32
        cell = [[0]*N for _ in range(N)]
        # Define by upper triangle of 16x16 top-left quadrant
        # i in [0,15], j in [i, 15]
        for i in range(16):
            for j in range(i, 16):
                val = quadrant[i][j] if j < len(quadrant[i]) else 0
                # Set all 8 symmetric positions
                # (i,j), (j,i), (N-1-i,j), (N-1-j,i), (i,N-1-j), (j,N-1-i),
                # (N-1-i,N-1-j), (N-1-j,N-1-i)
                positions = set()
                positions.add((i, j))
                positions.add((j, i))
                positions.add((N-1-i, j))
                positions.add((N-1-j, i))
                positions.add((i, N-1-j))
                positions.add((j, N-1-i))
                positions.add((N-1-i, N-1-j))
                positions.add((N-1-j, N-1-i))
                for (r, c) in positions:
                    if 0 <= r < N and 0 <= c < N:
                        cell[r][c] = val
        return cell

    def pw_solver(cell, n_pw=17, n_bands=6, n_kpts=15):
        """Solve TM bands using plane-wave method. Returns gap-midgap ratio for bands 4-5."""
        N = len(cell)
        eps = np.array(cell, dtype=float)
        eps[eps == 1] = eps_high
        eps[eps == 0] = eps_low

        # Fourier coefficients of 1/eps for TM mode
        # 1/eps(x,y) = sum_{G} eps_inv_G exp(iG.x)
        # For a pixelated cell, compute via DFT
        # Use n_pw x n_pw plane waves
        n = n_pw  # odd number
        # G vectors: indices from -(n-1)//2 to (n-1)//2
        g_idx = list(range(-(n-1)//2, (n+1)//2))

        # Compute 1/eps
        inv_eps = 1.0 / eps

        # DFT of inv_eps
        # eps_inv_G = (1/N^2) sum_{i,j} inv_eps[i,j] exp(-i (Gx*i + Gy*j) 2pi/N)
        # where Gx = gx * 2pi/N, Gy = gy * 2pi/N in pixel units
        # Actually, for a unit cell of size N pixels, the reciprocal lattice vectors
        # are G = (2pi*m/N, 2pi*l/N) in pixel units, or (2pi*m, 2pi*l) in lattice units.
        # Let's work in lattice units where the cell is 1x1.
        # Pixel (i,j) covers [i/N, (i+1)/N] x [j/N, (j+1)/N]
        # inv_eps_G = N^2 * integral of inv_eps(x,y) exp(-i 2pi (Gx x + Gy y)) dx dy
        # Approximate by sum over pixels:
        # inv_eps_G = (1/N^2) sum_{i,j} inv_eps[i,j] * exp(-i 2pi (gx*(i+0.5)/N + gy*(j+0.5)/N)) * ...
        # More carefully: the Fourier coefficient is
        # c_G = (1/N^2) sum_{i,j} inv_eps[i,j] * [sum_{x in pixel i} exp(-i 2pi gx x/N) dx] * [sum_{y in pixel j} exp(-i 2pi gy y/N) dy]
        # The inner sum is a geometric series. For gx != 0:
        # sum_{x=i/N}^{(i+1)/N} exp(-i 2pi gx x/N) dx = exp(-i 2pi gx (i+0.5)/N) * N/(2pi gx) * sin(pi gx/N)
        # For gx = 0: sum = 1/N
        # Let me compute this properly.

        # Actually, for the plane-wave method with pixelated dielectric, the standard approach
        # is to compute the Fourier coefficients of 1/eps(x,y) where eps is piecewise constant.
        # c_G = (1/N^2) sum_{i,j} eps_inv[i,j] * F(i, gx) * F(j, gy)
        # where F(i, g) = integral_{i/N}^{(i+1)/N} exp(-i 2pi g x/N) dx
        # = exp(-i pi g (2i+1)/N) * N/(2pi g) * sin(pi g/N) for g != 0
        # = 1/N for g = 0

        # Precompute F(i, g) for all i in [0,N) and g in g_idx
        F = np.zeros((N, len(g_idx)), dtype=complex)
        for ig, g in enumerate(g_idx):
            if g == 0:
                F[:, ig] = 1.0/N
            else:
                for i in range(N):
                    F[i, ig] = np.exp(-1j * np.pi * g * (2*i+1)/N) * N/(2*np.pi*g) * np.sin(np.pi*g/N)

        # Compute c_G for G = (gx, gy)
        c = np.zeros((len(g_idx), len(g_idx)), dtype=complex)
        for igx, gx in enumerate(g_idx):
            for igy, gy in enumerate(g_idx):
                # c[gx,gy] = (1/N^2) sum_{i,j} inv_eps[i,j] * F[i,gx] * F[j,gy]
                # = (1/N^2) * (sum_i inv_eps_col_i * F[i,gx]) dot (sum_j inv_eps_row_j * F[j,gy])
                # Wait, no. c[gx,gy] = (1/N^2) sum_{i,j} inv_eps[i,j] F[i,gx] F[j,gy]
                # This is (1/N^2) * (F[:,gx].conj() @ inv_eps @ F[:,gy]) ... no
                # = (1/N^2) * sum_i sum_j inv_eps[i,j] F[i,gx] F[j,gy]
                # = (1/N^2) * (F[:,gx] @ (inv_eps @ F[:,gy]))
                # But F is not conjugated here. Let me just compute directly.
                c[igx, igy] = (1.0/N**2) * np.sum(inv_eps * np.outer(F[:, igx], F[:, igy]))

        # Now solve the eigenvalue problem for each k-point
        # The matrix is: A_{G,G'} = (k+G)·(k+G') * c_{G-G'} for TM
        # where c_{G-G'} is the Fourier coefficient of 1/eps
        # Eigenvalues give omega^2

        # k-points in the irreducible BZ: from (0,0) to (pi,0) to (pi,pi)
        # In units of 2pi/L where L=1 (lattice constant)
        # k = (kx, ky) with kx, ky in [0, pi]
        # Irreducible BZ: kx >= ky >= 0, kx+ky <= pi
        # Let me use 15 k-points along the path (0,0) -> (pi,0) -> (pi,pi)

        kpts = []
        # Along (0,0) to (pi,0): 8 points
        for t in np.linspace(0, 1, 8):
            kpts.append((np.pi*t, 0.0))
        # Along (pi,0) to (pi,pi): 7 points (skip duplicate at (pi,0))
        for t in np.linspace(0, 1, 7):
            kpts.append((np.pi, np.pi*t))

        max_omega4 = 0.0
        min_omega5 = float('inf')

        for kx, ky in kpts:
            k = np.array([kx, ky])
            # Build the matrix
            M = np.zeros((n, n), dtype=complex)
            for i in range(n):
                for j in range(n):
                    Gi = np.array([g_idx[i]*2*np.pi, 0.0])
                    Gj = np.array([g_idx[j]*2*np.pi, 0.0])
                    # Wait, G vectors are 2D: G = (2pi*gx, 2pi*gy)
                    # I need to be more careful.
                    pass

            # Let me redo the matrix construction properly.
            # G vectors: G_a = (2pi*g_idx[a][0], 2pi*g_idx[a][1])
            # For 2D: G = (2pi*mx, 2pi*my) where mx, my are integers
            # g_idx for each component: -(n-1)//2 to (n-1)//2
            # Total plane waves: n x n = n^2
            n2 = n*n
            M = np.zeros((n2, n2), dtype=complex)

            # Map (mx, my) to index
            idx_map = {}
            for a, mx in enumerate(g_idx):
                for b, my in enumerate(g_idx):
                    idx_map[(mx, my)] = a*n + b

            for a, mx in enumerate(g_idx):
                for b, my in enumerate(g_idx):
                    i = a*n + b
                    kG_i = np.array([kx + 2*np.pi*mx, ky + 2*np.pi*my])
                    for c, mx2 in enumerate(g_idx):
                        for d, my2 in enumerate(g_idx):
                            j = c*n + d
                            kG_j = np.array([kx + 2*np.pi*mx2, ky + 2*np.pi*my2])
                            # c_{G_i - G_j} = c_{(mx-mx2, my-my2)}
                            dgx = mx - mx2
                            dgy = my - my2
                            if -len(g_idx)//2 <= dgx < len(g_idx)//2 + 1 and -len(g_idx)//2 <= dgy < len(g_idx)//2 + 1:
                                # Find the index in c
                                igx = g_idx.index(dgx) if dgx in g_idx else None
                                igy = g_idx.index(dgy) if dgy in g_idx else None
                                if igx is not None and igy is not None:
                                    cval = c[igx, igy]
                                else:
                                    cval = 0.0
                            else:
                                cval = 0.0
                            M[i, j] = np.dot(kG_i, kG_j) * cval

            # Solve eigenvalue problem
            # M is Hermitian (should be), so use eigvalsh
            # But M might not be exactly Hermitian due to numerical issues
            M = (M + M.conj().T) / 2
            try:
                eigvals = np.linalg.eigvalsh(M)
                eigvals = np.sort(eigvals.real)
                # Take the first n_bands
                if len(eigvals) >= n_bands:
                    omega4 = np.sqrt(max(eigvals[3], 0))
                    omega5 = np.sqrt(max(eigvals[4], 0))
                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
            except:
                pass

        if min_omega5 == float('inf') or max_omega4 == 0:
            return -1.0

        gap_midgap = 2*(min_omega5 - max_omega4) / (min_omega5 + max_omega4)
        return gap_midgap

    # Parametric design families
    def circle_cell(radius, center_x=0.5, center_y=0.5):
        """Dielectric circle of given radius."""
        N = 32
        cell = [[0]*N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                x = (i+0.5)/N - center_x
                y = (j+0.5)/N - center_y
                if x*x + y*y < radius*radius:
                    cell[i][j] = 1
        return cell

    def square_cell(half_width, center_x=0.5, center_y=0.5):
        """Dielectric square of given half-width."""
        N = 32
        cell = [[0]*N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                x = (i+0.5)/N - center_x
                y = (j+0.5)/N - center_y
                if abs(x) < half_width and abs(y) < half_width:
                    cell[i][j] = 1
        return cell

    def diamond_cell(half_width, center_x=0.5, center_y=0.5):
        """Dielectric diamond (rotated square)."""
        N = 32
        cell = [[0]*N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                x = (i+0.5)/N - center_x
                y = (j+0.5)/N - center_y
                if abs(x) + abs(y) < half_width:
                    cell[i][j] = 1
        return cell

    def cross_cell(arm_width, arm_length, center_x=0.5, center_y=0.5):
        """Dielectric cross shape."""
        N = 32
        cell = [[0]*N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                x = (i+0.5)/N - center_x
                y = (j+0.5)/N - center_y
                if (abs(x) < arm_width/2 and abs(y) < arm_length/2) or \
                   (abs(y) < arm_width/2 and abs(x) < arm_length/2):
                    cell[i][j] = 1
        return cell

    def four_circles_cell(radius, offset):
        """Four dielectric circles at (±offset, ±offset)."""
        N = 32
        cell = [[0]*N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                x = (i+0.5)/N - 0.5
                y = (j+0.5)/N - 0.5
                for ox, oy in [(offset, offset), (offset, -offset), (-offset, offset), (-offset, -offset)]:
                    if (x-ox)**2 + (y-oy)**2 < radius*radius:
                        cell[i][j] = 1
                        break
        return cell

    def grid_holes_cell(hole_radius, n_holes=2):
        """Dielectric with air holes in a grid pattern."""
        N = 32
        cell = [[1]*N for _ in range(N)]
        # Place holes at a grid
        spacing = 1.0/n_holes
        for i in range(N):
            for j in range(N):
                x = (i+0.5)/N
                y = (j+0.5)/N
                # Check if (x,y) is close to any hole center
                for mx in range(n_holes):
                    for my in range(n_holes):
                        cx = (mx + 0.5)/n_holes
                        cy = (my + 0.5)/n_holes
                        if (x-cx)**2 + (y-cy)**2 < hole_radius**2:
                            cell[i][j] = 0
        return cell

    # Local search: perturb individual pixels and check if score improves
    def local_search(cell, max_iters=500, time_limit=10.0):
        """Hill-climbing local search on individual pixels."""
        N = 32
        best_score = pw_solver(cell, n_pw=11, n_bands=6)
        best_cell = [row[:] for row in cell]
        start = time.time()

        for it in range(max_iters):
            if time.time() - start > time_limit:
                break
            if time.time() > deadline - 30:
                break

            # Try flipping a random pixel
            i = np.random.randint(0, N)
            j = np.random.randint(0, N)

            # Apply D4 symmetry: flip all symmetric positions
            new_cell = [row[:] for row in cell]
            positions = [(i, j), (j, i), (N-1-i, j), (N-1-j, i),
                         (i, N-1-j), (j, N-1-i), (N-1-i, N-1-j), (N-1-j, N-1-i)]
            old_val = new_cell[i][j]
            new_val = 1 - old_val
            for (r, c) in set(positions):
                new_cell[r][c] = new_val

            score = pw_solver(new_cell, n_pw=11, n_bands=6)
            if score > best_score:
                cell = new_cell
                best_score = score
                best_cell = [row[:] for row in cell]

        return best_cell, best_score

    # Start with several initial designs and do local search on each
    np.random.seed(42)
    candidates = []

    # 1. Circle of various radii
    for r in [0.15, 0.2, 0.25, 0.3]:
        candidates.append(circle_cell(r))

    # 2. Square
    for hw in [0.15, 0.2, 0.25]:
        candidates.append(square_cell(hw))

    # 3. Diamond
    for hw in [0.2, 0.25, 0.3]:
        candidates.append(diamond_cell(hw))

    # 4. Cross
    for aw, al in [(0.1, 0.3), (0.15, 0.35), (0.2, 0.4)]:
        candidates.append(cross_cell(aw, al))

    # 5. Four circles
    for r, off in [(0.1, 0.2), (0.12, 0.25), (0.15, 0.3)]:
        candidates.append(four_circles_cell(r, off))

    # 6. Grid holes (dielectric background with air holes)
    for hr in [0.08, 0.1, 0.12]:
        candidates.append(grid_holes_cell(hr, n_holes=2))

    # Evaluate all candidates
    best_cell = None
    best_score = -float('inf')
    for cell in candidates:
        if time.time() > deadline - 60:
            break
        score = pw_solver(cell, n_pw=11, n_bands=6)
        if score > best_score:
            best_score = score
            best_cell = [row[:] for row in cell]

    # Local search from the best candidate
    if best_cell is not None and time.time() < deadline - 30:
        best_cell, best_score = local_search(best_cell, max_iters=300, time_limit=15.0)

    # Try local search from a few other good candidates
    if time.time() < deadline - 30:
        for cell in candidates[:5]:
            if time.time() > deadline - 30:
                break
            improved_cell, improved_score = local_search(cell, max_iters=100, time_limit=5.0)
            if improved_score > best_score:
                best_cell = improved_cell
                best_score = improved_score

    # Final refinement with more plane waves if time permits
    if best_cell is not None and time.time() < deadline - 10:
        final_score = pw_solver(best_cell, n_pw=17, n_bands=6)
        # The final score is what the evaluator will compute, but we return the cell

    if best_cell is None:
        # Fallback: original circle
        best_cell = circle_cell(0.2)

    return best_cell
# EVOLVE-BLOCK-END
