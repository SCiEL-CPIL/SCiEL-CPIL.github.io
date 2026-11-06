"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """Design a symmetric 32x32 unit cell maximizing the TM band-4/5 gap-midgap ratio."""
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12)
    
    # Deadline guard: keep ~15s margin before 230s wall-clock kill
    t0 = time.time()
    deadline = t0 + 210.0

    half = size // 2  # 16

    # --- Symmetric cell construction ---
    def build_cell_from_quadrant(quadrant):
        """Build full 32x32 cell from 16x16 upper-triangle quadrant."""
        N = size
        cell = [[0]*N for _ in range(N)]
        for i in range(half):
            for j in range(i, half):
                val = quadrant[i][j]
                cell[i][j] = val
                cell[i][N-1-j] = val
                cell[N-1-i][j] = val
                cell[N-1-i][N-1-j] = val
                cell[j][i] = val
                cell[j][N-1-i] = val
                cell[N-1-j][i] = val
                cell[N-1-j][N-1-i] = val
        return cell

    # --- Fast 5x5 PWE evaluator for screening ---
    def compute_fast_score(cell, max_g=2):
        """Fast TM band-4/5 gap using 5x5=25 plane waves."""
        N = size
        eps_inv = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                eps_inv[i, j] = 1.0 / eps_high if cell[i][j] == 1 else 1.0

        fft_coeff = np.fft.fft2(eps_inv) / (N * N)

        def get_coeff(mx, my):
            return fft_coeff[my % N, mx % N]

        g_indices = [(mx, my) for mx in range(-max_g, max_g+1) for my in range(-max_g, max_g+1)]
        num_g = len(g_indices)
        G_vecs = np.array([(2*math.pi*mx, 2*math.pi*my) for mx, my in g_indices])

        eps_inv_mat = np.zeros((num_g, num_g))
        for p in range(num_g):
            mx_p, my_p = g_indices[p]
            for q in range(num_g):
                mx_q, my_q = g_indices[q]
                eps_inv_mat[p, q] = get_coeff(mx_p - mx_q, my_p - my_q)

        # K-points in irreducible BZ
        k_points = []
        for t in np.linspace(0, math.pi, 5):
            k_points.append((0.0, t))
        for t in np.linspace(0, math.pi/2, 5):
            k_points.append((t, t))
        for t in np.linspace(0, math.pi/2, 5):
            k_points.append((math.pi/2 - t, math.pi/2 + t))
        k_points.append((math.pi/6, math.pi/3))
        k_points.append((math.pi/4, math.pi/2))
        k_points.append((math.pi/8, 3*math.pi/8))
        k_points.append((math.pi/5, 2*math.pi/5))
        k_points.append((math.pi/3, 2*math.pi/3))
        k_points = k_points[:30]

        max_omega4 = 0.0
        min_omega5 = float('inf')

        for kx, ky in k_points:
            kG_sq = np.zeros(num_g)
            for q in range(num_g):
                kG_sq[q] = (kx + G_vecs[q, 0])**2 + (ky + G_vecs[q, 1])**2
            M = eps_inv_mat * kG_sq[None, :]
            try:
                eigvals = np.sort(np.linalg.eigvalsh(M))
            except:
                continue
            if len(eigvals) < 5:
                continue
            omega4 = math.sqrt(max(eigvals[3], 0))
            omega5 = math.sqrt(max(eigvals[4], 0))
            max_omega4 = max(max_omega4, omega4)
            min_omega5 = min(min_omega5, omega5)

        if min_omega5 == float('inf') or max_omega4 == 0:
            return -1.0
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        if midgap <= 0:
            return -1.0
        return gap / midgap

    # --- Full 17x17 PWE evaluator for refinement ---
    def compute_full_score(cell, max_g=8):
        """Full TM band-4/5 gap using 17x17=289 plane waves."""
        N = size
        eps_inv = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                eps_inv[i, j] = 1.0 / eps_high if cell[i][j] == 1 else 1.0

        fft_coeff = np.fft.fft2(eps_inv) / (N * N)

        def get_coeff(mx, my):
            return fft_coeff[my % N, mx % N]

        g_indices = [(mx, my) for mx in range(-max_g, max_g+1) for my in range(-max_g, max_g+1)]
        num_g = len(g_indices)
        G_vecs = np.array([(2*math.pi*mx, 2*math.pi*my) for mx, my in g_indices])

        eps_inv_mat = np.zeros((num_g, num_g))
        for p in range(num_g):
            mx_p, my_p = g_indices[p]
            for q in range(num_g):
                mx_q, my_q = g_indices[q]
                eps_inv_mat[p, q] = get_coeff(mx_p - mx_q, my_p - my_q)

        k_points = []
        for t in np.linspace(0, math.pi, 5):
            k_points.append((0.0, t))
        for t in np.linspace(0, math.pi/2, 5):
            k_points.append((t, t))
        for t in np.linspace(0, math.pi/2, 5):
            k_points.append((math.pi/2 - t, math.pi/2 + t))
        k_points.append((math.pi/6, math.pi/3))
        k_points.append((math.pi/4, math.pi/2))
        k_points.append((math.pi/8, 3*math.pi/8))
        k_points.append((math.pi/5, 2*math.pi/5))
        k_points.append((math.pi/3, 2*math.pi/3))
        k_points = k_points[:30]

        max_omega4 = 0.0
        min_omega5 = float('inf')

        for kx, ky in k_points:
            kG_sq = np.zeros(num_g)
            for q in range(num_g):
                kG_sq[q] = (kx + G_vecs[q, 0])**2 + (ky + G_vecs[q, 1])**2
            M = eps_inv_mat * kG_sq[None, :]
            try:
                eigvals = np.sort(np.linalg.eigvalsh(M))
            except:
                continue
            if len(eigvals) < 5:
                continue
            omega4 = math.sqrt(max(eigvals[3], 0))
            omega5 = math.sqrt(max(eigvals[4], 0))
            max_omega4 = max(max_omega4, omega4)
            min_omega5 = min(min_omega5, omega5)

        if min_omega5 == float('inf') or max_omega4 == 0:
            return -1.0
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        if midgap <= 0:
            return -1.0
        return gap / midgap

    # --- Initial designs ---
    def make_ring_quadrant(r_inner, r_outer):
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                x = (j + 0.5) / size
                y = (i + 0.5) / size
                r = math.sqrt(x*x + y*y)
                if r_inner <= r <= r_outer:
                    q[i][j] = 1
        return q

    def make_disk_quadrant(radius):
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                x = (j + 0.5) / size
                y = (i + 0.5) / size
                r = math.sqrt(x*x + y*y)
                if r <= radius:
                    q[i][j] = 1
        return q

    def make_cross_quadrant(width=0.15):
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                x = (j + 0.5) / size
                y = (i + 0.5) / size
                if x < width or y < width:
                    q[i][j] = 1
        return q

    def make_random_quadrant(density=0.5):
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                q[i][j] = 1 if np.random.random() < density else 0
        return q

    initial_designs = [
        make_ring_quadrant(0.15, 0.35),
        make_ring_quadrant(0.2, 0.4),
        make_ring_quadrant(0.1, 0.3),
        make_ring_quadrant(0.25, 0.45),
        make_disk_quadrant(0.2),
        make_disk_quadrant(0.3),
        make_cross_quadrant(0.1),
        make_cross_quadrant(0.15),
        make_random_quadrant(0.3),
        make_random_quadrant(0.5),
        make_random_quadrant(0.7),
    ]

    # --- Stage 1: Fast screening with 5x5 PWE ---
    best_fast_score = -1.0
    best_quadrant = initial_designs[0]
    top_candidates = []

    for q in initial_designs:
        if time.time() > deadline - 60:
            break
        cell = build_cell_from_quadrant(q)
        score = compute_fast_score(cell)
        if score > best_fast_score:
            best_fast_score = score
            best_quadrant = [row[:] for row in q]
        top_candidates.append((score, [row[:] for row in q]))

    top_candidates.sort(key=lambda x: x[0], reverse=True)

    # --- Stage 2: Evolutionary search with fast evaluation ---
    positions = [(i, j) for i in range(half) for j in range(i, half)]
    current_quadrant = [row[:] for row in best_quadrant]
    current_score = best_fast_score
    population = [(current_score, [row[:] for row in current_quadrant])]

    max_evo_iterations = 200
    for iteration in range(max_evo_iterations):
        if time.time() > deadline - 80:
            break

        # Mutate: flip 1-3 random pixels
        num_flips = np.random.randint(1, 4)
        mutated = [row[:] for row in current_quadrant]
        for _ in range(num_flips):
            idx = np.random.randint(len(positions))
            i, j = positions[idx]
            mutated[i][j] = 1 - mutated[i][j]

        cell = build_cell_from_quadrant(mutated)
        score = compute_fast_score(cell)

        if score > current_score:
            current_score = score
            current_quadrant = [row[:] for row in mutated]
            population.append((score, [row[:] for row in mutated]))
        else:
            # Sometimes accept worse solutions for diversity
            if np.random.random() < 0.1:
                current_quadrant = [row[:] for row in mutated]
                current_score = score

        # Keep top candidates
        population.append((score, [row[:] for row in mutated]))
        population.sort(key=lambda x: x[0], reverse=True)
        population = population[:5]

    # --- Stage 3: Full 17x17 refinement of top candidates ---
    # Take top 3-5 candidates from fast search and refine with full PWE
    top_for_refinement = population[:min(5, len(population))]
    
    # Also include the best initial design
    if best_quadrant not in [c[1] for c in top_for_refinement]:
        top_for_refinement.append((best_fast_score, [row[:] for row in best_quadrant]))

    best_full_score = -1.0
    best_final_quadrant = top_for_refinement[0][1]

    for score, quad in top_for_refinement:
        if time.time() > deadline - 20:
            break
        cell = build_cell_from_quadrant(quad)
        full_score = compute_full_score(cell)
        if full_score > best_full_score:
            best_full_score = full_score
            best_final_quadrant = [row[:] for row in quad]

    # --- Final local refinement with full PWE ---
    current_quadrant = [row[:] for row in best_final_quadrant]
    current_score = best_full_score
    no_improve = 0

    for _ in range(50):
        if time.time() > deadline - 10:
            break
        if no_improve > 15:
            break

        improved = False
        for idx in np.random.permutation(len(positions)):
            if time.time() > deadline - 8:
                break
            i, j = positions[idx]
            old_val = current_quadrant[i][j]
            current_quadrant[i][j] = 1 - old_val

            cell = build_cell_from_quadrant(current_quadrant)
            score = compute_full_score(cell)

            if score > current_score:
                current_score = score
                best_full_score = score
                best_final_quadrant = [row[:] for row in current_quadrant]
                improved = True
                no_improve = 0
            else:
                current_quadrant[i][j] = old_val
                no_improve += 1

        if not improved:
            break

    # Build final cell
    final_cell = build_cell_from_quadrant(best_final_quadrant)

    # Verify symmetry
    for i in range(size):
        for j in range(size):
            assert final_cell[i][j] == final_cell[size-1-i][j], f"Vertical symmetry broken at ({i},{j})"
            assert final_cell[i][j] == final_cell[i][size-1-j], f"Horizontal symmetry broken at ({i},{j})"
            assert final_cell[i][j] == final_cell[j][i], f"Transpose symmetry broken at ({i},{j})"

    return final_cell
# EVOLVE-BLOCK-END
