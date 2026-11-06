import numpy as np
import time
import random
from scipy.linalg import eigh

def design(problem):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # We will optimize the top-left 8x8 quadrant
    # The full cell is 32x32.
    # Symmetry:
    # 1. Reflection across vertical midline (x -> size-1-x)
    # 2. Reflection across horizontal midline (y -> size-1-y)
    # 3. Rotation 90 deg.
    # This implies the cell is determined by the top-left 8x8 block.
    # Let q[i][j] be the quadrant pixel for i,j in 0..7.
    # cell[i][j] = q[i][j]
    # cell[i][31-j] = q[i][j]
    # cell[31-i][j] = q[i][j]
    # cell[31-i][31-j] = q[i][j]
    # Also, due to 90 deg rotation, q[i][j] must equal q[j][i]?
    # Let's check.
    # Rot 90: (i,j) -> (j, size-1-i).
    # If we define the cell from the top-left quadrant, does 90 deg symmetry impose q[i][j] == q[j][i]?
    # cell[i][j] = q[i][j].
    # cell[j][31-i] = q[j][31-i]? No, 31-i is in the right half.
    # Let's just construct the full 32x32 from the 8x8 quadrant and then enforce full symmetry by averaging/rounding?
    # Or simpler: Only allow designs that are symmetric under 90 deg rotation.
    # If a design is symmetric under 90 deg rotation, it is symmetric under 180, 270, and reflections.
    # So we just need to enforce 90 deg rotational symmetry.
    # This means cell[i][j] == cell[j][31-i] == cell[31-i][31-j] == cell[31-j][i].
    # This reduces the independent pixels to 1/4 of the cell, but with a specific mapping.
    # The orbit of (i,j) under C4 has size 4 (unless on axis).
    # We can just pick the top-left 8x8 quadrant and ensure that the generated 32x32 is invariant under rotation.
    # Actually, if we define the top-left 8x8 quadrant arbitrarily, the resulting 32x32 (by reflecting) is symmetric under reflections, but NOT necessarily under 90 deg rotation.
    # To ensure 90 deg rotation, we must have q[i][j] == q[j][i] is NOT sufficient for the whole 32x32.
    # Let's verify:
    # If we reflect an 8x8 block to make a 32x32, the block at (0,0) is Q.
    # The block at (0,24) is Q reflected horizontally.
    # The block at (24,0) is Q reflected vertically.
    # The block at (24,24) is Q reflected both.
    # For 90 deg rotation, the block at (0,0) must map to (0,24) rotated?
    # (i,j) in Q maps to (j, 31-i).
    # If i,j < 8, then 31-i > 23. So (j, 31-i) is in the top-right block.
    # The top-right block is Q reflected horizontally.
    # So cell[j][31-i] must equal cell[i][j].
    # cell[j][31-i] corresponds to Q_reflected_h[j][31-i-24] = Q_reflected_h[j][7-i] = Q[j][7-(7-i)] = Q[j][i].
    # So we require cell[i][j] == Q[j][i].
    # Thus, the 8x8 quadrant Q must be symmetric! Q[i][j] == Q[j][i].
    # So we only need to define the upper triangle of the 8x8 quadrant.
    # Number of independent bits: 8*9/2 = 36.
    
    n_q = 8
    n_bits = n_q * (n_q + 1) // 2
    
    def quadrant_to_cell(q):
        # q is a list of lists 8x8, symmetric
        cell = [[0]*size for _ in range(size)]
        for i in range(n_q):
            for j in range(n_q):
                val = q[i][j]
                # Top-left
                cell[i][j] = val
                # Top-right (reflect i, j -> i, size-1-j)
                cell[i][size-1-j] = val
                # Bottom-left (reflect i, j -> size-1-i, j)
                cell[size-1-i][j] = val
                # Bottom-right
                cell[size-1-i][size-1-j] = val
        return cell

    def cell_to_quadrant(cell):
        q = [[0]*n_q for _ in range(n_q)]
        for i in range(n_q):
            for j in range(n_q):
                q[i][j] = cell[i][j]
        return q

    # Precompute plane wave vectors
    # Use N_pw = 17 (289 waves) as per evaluator? 
    # For speed in optimization, maybe use N_pw = 9 or 11?
    # Let's try N_pw = 11 (121 waves) for fast eval, and refine with 17 at the end.
    # The gap location might shift, but usually the topology is stable.
    
    n_pw_fast = 11
    n_pw_full = 17
    
    def get_gvectors(n_pw):
        # G vectors are (nx, ny) * (2*pi/a). a=1.
        # Range: -n_pw//2 to n_pw//2
        half = n_pw // 2
        gxs = np.arange(-half, half + 1)
        gys = np.arange(-half, half + 1)
        gx, gy = np.meshgrid(gxs, gys, indexing='ij')
        gv = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(float)
        return gv

    gv_fast = get_gvectors(n_pw_fast)
    gv_full = get_gvectors(n_pw_full)

    # High symmetry points in irreducible BZ
    # Gamma: (0,0)
    # X: (0.5, 0)
    # M: (0.5, 0.5)
    # K: (1/3, 1/3) ? No, square lattice.
    # Irreducible BZ for square: 0 <= ky <= kx, kx+ky <= 0.5?
    # Vertices: Gamma(0,0), X(0.5,0), M(0.5,0.5).
    # We check Gamma, X, M, and midpoints of Gamma-X, X-M, Gamma-M.
    
    k_points_fast = [
        (0.0, 0.0),
        (0.25, 0.0),
        (0.5, 0.0),
        (0.5, 0.25),
        (0.5, 0.5),
        (0.25, 0.25),
        (0.0, 0.0) # duplicate gamma? no, just list unique
    ]
    # Unique: Gamma, (0.25,0), X, (0.5,0.25), M, (0.25,0.25)
    
    def compute_eps_fourier(cell, gv):
        # cell is 32x32 binary
        # eps(r) = eps_low + (eps_high - eps_low) * cell(r)
        # Fourier coeff: eps_G = 1/A * sum eps(r) exp(-i G . r)
        # Since cell is discrete pixels, we approximate integral with sum.
        # r_x = (i + 0.5)/size, r_y = (j + 0.5)/size
        # eps_G = (1/size^2) * sum_{i,j} eps_{ij} exp(-i (Gx * rx + Gy * ry))
        
        eps_arr = np.where(np.array(cell) == 1, eps_high, eps_low)
        
        # Vectorized calculation
        # G shape (N, 2)
        # r shape (size, size, 2)
        # We want sum over i,j of eps[i,j] * exp(-1j * (Gx_i * rx_j + Gy_i * ry_j))
        
        rx = (np.arange(size) + 0.5) / size
        ry = (np.arange(size) + 0.5) / size
        rx_grid, ry_grid = np.meshgrid(rx, ry, indexing='ij')
        
        # Phase = Gx * rx + Gy * ry
        # Gx shape (N,), rx_grid shape (S,S)
        # We can do: phase = np.einsum('ni,ij->nj', gv[:,0], rx_grid) + np.einsum('ni,ij->nj', gv[:,1], ry_grid)
        # This creates array (N, S, S)
        
        # To save memory/time, we can loop over G if N is small, or use FFT?
        # FFT is much faster if we can pad to power of 2? 32 is power of 2.
        # eps_G = FFT(eps_arr) * (1/size^2) ?
        # Discrete Fourier Transform: F[k] = sum_{n} f[n] exp(-i 2 pi k n / N).
        # Our G vectors are integers (nx, ny).
        # exp(-i 2 pi (nx * i / size + ny * j / size)).
        # This matches the DFT definition if we index correctly.
        # np.fft.fft2(eps_arr) gives F[kx, ky] = sum_{i,j} eps[i,j] exp(-i 2 pi (kx i / S + ky j / S)).
        # Our G = (nx, ny). The term is exp(-i 2 pi (nx i / S + ny j / S)).
        # So eps_G = fft2(eps_arr)[nx, ny] / S^2 ?
        # Wait, the sum in DFT is over i=0..S-1.
        # Our r_i = (i+0.5)/S.
        # The DFT assumes r_i = i/S.
        # There is a phase shift factor: exp(-i 2 pi (nx (i+0.5)/S + ...)) = exp(-i 2 pi nx i / S) * exp(-i 2 pi nx / 2S).
        # So eps_G = fft2(eps_arr)[nx, ny] * exp(-i pi (nx/S + ny/S)) / S^2.
        
        fft_eps = np.fft.fft2(eps_arr)
        
        # Map G vectors to FFT indices
        # FFT indices 0..S-1 correspond to frequencies 0..S-1 and -S/2..-1.
        # G ranges -half..half.
        # We need to extract the correct coefficients.
        
        eps_coeffs = np.zeros(len(gv), dtype=complex)
        for idx, (gx, gy) in enumerate(gv):
            # Map integer gx, gy to FFT index
            # If gx >= 0, idx_x = gx
            # If gx < 0, idx_x = S + gx
            ix = int(gx) % size
            iy = int(gy) % size
            fft_val = fft_eps[ix, iy]
            phase_shift = np.exp(-1j * np.pi * (gx / size + gy / size))
            eps_coeffs[idx] = fft_val * phase_shift / (size * size)
            
        return eps_coeffs

    def solve_bands(cell, k_point, gv, eps_coeffs):
        kx, ky = k_point
        # Build matrix M
        # M_{G, G'} = eps_{G-G'} * (k+G) dot (k+G')  -- Wait, this was for TE?
        # For TM: |k+G|^2 A_G = omega^2 sum_{G'} eps_{G-G'} A_{G'}
        # So A = diag(|k+G|^2), B = eps_matrix.
        # Solve A x = omega^2 B x.
        
        n_g = len(gv)
        # k+G vectors
        k_plus_g = gv + np.array([kx, ky])
        k_sq = np.sum(k_plus_g**2, axis=1)
        
        A_mat = np.diag(k_sq)
        
        # B_mat (eps convolution)
        # B_{i,j} = eps_{G_i - G_j}
        # We have eps_coeffs for G in gv.
        # We need eps for all differences G_i - G_j.
        # Since gv is symmetric and complete enough, G_i - G_j might be outside gv range?
        # If we use a cutoff, the convolution is truncated.
        # We need to find the index of G_diff in gv.
        
        # Create a dictionary for fast lookup
        g_dict = {}
        for idx, (gx, gy) in enumerate(gv):
            g_dict[(int(gx), int(gy))] = idx
            
        B_mat = np.zeros((n_g, n_g), dtype=complex)
        for i in range(n_g):
            gi = gv[i]
            for j in range(n_g):
                gj = gv[j]
                gd = (int(gi[0] - gj[0]), int(gi[1] - gj[1]))
                if gd in g_dict:
                    idx_gd = g_dict[gd]
                    B_mat[i, j] = eps_coeffs[idx_gd]
                else:
                    # Truncation error, assume 0
                    B_mat[i, j] = 0.0
                    
        # Solve generalized eigenvalue problem
        # A is real symmetric, B is complex Hermitian (since eps is real and symmetric)
        # Use scipy.linalg.eigh for Hermitian
        try:
            eigvals, eigvecs = eigh(A_mat, B_mat)
            # eigvals are omega^2
            eigvals = np.real(eigvals)
            # Sort
            sort_idx = np.argsort(eigvals)
            eigvals = eigvals[sort_idx]
            # omega = sqrt(eigvals)
            omegas = np.sqrt(eigvals)
            return omegas
        except Exception:
            return np.array([])

    def evaluate_gap(cell, use_full=False):
        gv = gv_full if use_full else gv_fast
        eps_coeffs = compute_eps_fourier(cell, gv)
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for k in k_points_fast:
            omegas = solve_bands(cell, k, gv, eps_coeffs)
            if len(omegas) < 5:
                continue
            # Take the 4th and 5th bands (index 3 and 4)
            # Note: The problem says "between the fourth and fifth TM bands".
            # Usually bands are 1-indexed. So band 4 is index 3, band 5 is index 4.
            w4 = omegas[3]
            w5 = omegas[4]
            
            if w5 < min_omega_5:
                min_omega_5 = w5
            if w4 > max_omega_4:
                max_omega_4 = w4
                
        if min_omega_5 == np.inf:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return -1.0
        ratio = gap / midgap
        return ratio

    # Optimization: Genetic Algorithm
    
    def random_quadrant():
        q = [[0]*n_q for _ in range(n_q)]
        for i in range(n_q):
            for j in range(i, n_q):
                q[i][j] = random.randint(0, 1)
                q[j][i] = q[i][j]
        return q

    def mutate(q, rate=0.1):
        new_q = [row[:] for row in q]
        for i in range(n_q):
            for j in range(i, n_q):
                if random.random() < rate:
                    new_q[i][j] = 1 - new_q[i][j]
                    new_q[j][i] = new_q[i][j]
        return new_q

    def crossover(q1, q2):
        new_q = [[0]*n_q for _ in range(n_q)]
        for i in range(n_q):
            for j in range(i, n_q):
                if random.random() < 0.5:
                    new_q[i][j] = q1[i][j]
                    new_q[j][i] = q1[j][i]
                else:
                    new_q[i][j] = q2[i][j]
                    new_q[j][i] = q2[j][i]
        return new_q

    # Initialize population
    pop_size = 40
    population = []
    scores = []
    
    # Add some heuristic starts
    # 1. Square pillar
    q_pillar = [[0]*n_q for _ in range(n_q)]
    for i in range(2, 6):
        for j in range(2, 6):
            q_pillar[i][j] = 1
            q_pillar[j][i] = 1
    population.append(q_pillar)
    
    # 2. Square hole (inverted)
    q_hole = [[1]*n_q for _ in range(n_q)]
    for i in range(2, 6):
        for j in range(2, 6):
            q_hole[i][j] = 0
            q_hole[j][i] = 0
    population.append(q_hole)
    
    # 3. Random
    while len(population) < pop_size:
        population.append(random_quadrant())
        
    # Evaluate initial population
    for q in population:
        cell = quadrant_to_cell(q)
        score = evaluate_gap(cell)
        scores.append(score)
        
    best_q = population[np.argmax(scores)]
    best_score = max(scores)
    
    # GA Loop
    generation = 0
    while time.time() < deadline:
        generation += 1
        # Sort by score
        pairs = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        new_population = [p for s, p in pairs[:10]]  # Elitism
        
        while len(new_population) < pop_size:
            # Selection
            p1 = pairs[random.randint(0, 19)][1]
            p2 = pairs[random.randint(0, 19)][1]
            
            # Crossover
            child = crossover(p1, p2)
            
            # Mutation
            child = mutate(child, rate=0.1)
            
            new_population.append(child)
            
        population = new_population
        scores = []
        for q in population:
            cell = quadrant_to_cell(q)
            score = evaluate_gap(cell)
            scores.append(score)
            
        current_best = max(scores)
        if current_best > best_score:
            best_score = current_best
            best_q = population[scores.index(current_best)]
            
        # Check time
        if time.time() > deadline - 5:
            break
            
    # Final refinement with full plane wave set
    final_cell = quadrant_to_cell(best_q)
    final_score = evaluate_gap(final_cell, use_full=True)
    
    # Try a few more random restarts if time permits?
    # No, just return the best.
    
    return final_cell

