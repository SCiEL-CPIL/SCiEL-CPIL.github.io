import time
import numpy as np
from numpy.linalg import eigh

def _compute_gap(cell, pw_radius=10):
    """Compute TM gap-midgap ratio between bands 4 and 5 using PWE."""
    N = len(cell)
    eps = 1.0 + 11.0 * np.array(cell, dtype=float)
    inv_eps = 1.0 / eps
    fft_inv = np.fft.fft2(inv_eps) / (N * N)
    
    # Build G-vector list (circular cutoff)
    G_list = []
    for Gx in range(-pw_radius, pw_radius + 1):
        for Gy in range(-pw_radius, pw_radius + 1):
            if Gx * Gx + Gy * Gy <= pw_radius * pw_radius:
                G_list.append((Gx, Gy))
    n_pw = len(G_list)
    G_arr = np.array(G_list, dtype=float)
    
    # Build the full inv_eps matrix
    size_d = 2 * pw_radius + 1
    offset_d = pw_radius
    
    # Compute all pairwise differences
    Gx_diff = G_arr[:, 0][:, None] - G_arr[:, 0][None, :]
    Gy_diff = G_arr[:, 1][:, None] - G_arr[:, 1][None, :]
    
    # Only keep entries within the valid range
    valid = (np.abs(Gx_diff) <= pw_radius) & (np.abs(Gy_diff) <= pw_radius)
    
    # For efficiency, we build the matrix by direct indexing
    # Actually, let's build it properly
    inv_eps_mat = np.zeros((n_pw, n_pw), dtype=complex)
    for i in range(n_pw):
        for j in range(n_pw):
            dqx = G_arr[i, 0] - G_arr[j, 0]
            dqy = G_arr[i, 1] - G_arr[j, 1]
            if abs(dqx) <= pw_radius and abs(dqy) <= pw_radius:
                # The Fourier coefficient is fft_inv[(dqx mod N), (dqy mod N)]
                idx_x = int(dqx) % N
                idx_y = int(dqy) % N
                inv_eps_mat[i, j] = fft_inv[idx_x, idx_y]
    
    # K-points in the irreducible Brillouin zone
    # The square BZ goes from -0.5 to 0.5 in both kx and ky (in units of 2pi/a)
    # Irreducible wedge: kx >= |ky|, kx >= 0
    k_points = []
    
    # Along Gamma-X: (t, 0) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 11):
        k_points.append((t, 0.0))
    
    # Along X-M: (0.5, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 11):
        k_points.append((0.5, t))
    
    # Along Gamma-M: (t, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 11):
        k_points.append((t, t))
    
    # Some interior points
    k_points.extend([
        (0.2, 0.1), (0.3, 0.1), (0.3, 0.2), (0.4, 0.1),
        (0.4, 0.2), (0.45, 0.1), (0.45, 0.2), (0.25, 0.15),
        (0.35, 0.25), (0.4, 0.3), (0.45, 0.35), (0.2, 0.05),
    ])
    
    # Deduplicate
    k_points = list(set(k_points))
    
    max_w4 = 0.0
    min_w5 = float('inf')
    
    for k in k_points:
        kvec = np.array(k, dtype=float)
        kG = kvec + G_arr
        dots = np.sum(kG * kG, axis=1)  # |k+G|^2
        
        # Build the matrix: M[i,j] = |k+G_i|^2 * inv_eps[G_i - G_j]
        # But for TM, the eigenvalue problem is:
        # sum_j |k+G_i|^2 * (1/eps)_{G_i-G_j} * A_j = omega^2 * A_i
        # So M = diag(dots) @ inv_eps_mat
        M = np.diag(dots) @ inv_eps_mat
        M = (M + M.conj().T) / 2  # Ensure Hermitian
        
        try:
            eigvals = eigh(M, eigvals_only=True)
            eigvals = np.sqrt(np.maximum(eigvals, 0))
            if len(eigvals) >= 5:
                max_w4 = max(max_w4, eigvals[3])
                min_w5 = min(min_w5, eigvals[4])
        except:
            pass
    
    if min_w5 == float('inf') or max_w4 <= 0:
        return -1.0
    return 2.0 * (min_w5 - max_w4) / (min_w5 + max_w4)


def _compute_gap_fast(cell, pw_radius=5):
    """Very fast surrogate for ranking."""
    N = len(cell)
    eps = 1.0 + 11.0 * np.array(cell, dtype=float)
    inv_eps = 1.0 / eps
    fft_inv = np.fft.fft2(inv_eps) / (N * N)
    
    G_list = []
    for Gx in range(-pw_radius, pw_radius + 1):
        for Gy in range(-pw_radius, pw_radius + 1):
            if Gx * Gx + Gy * Gy <= pw_radius * pw_radius:
                G_list.append((Gx, Gy))
    n_pw = len(G_list)
    G_arr = np.array(G_list, dtype=float)
    
    # Build inv_eps_mat quickly
    inv_eps_mat = np.zeros((n_pw, n_pw), dtype=complex)
    for i in range(n_pw):
        for j in range(n_pw):
            dqx = int(G_arr[i, 0] - G_arr[j, 0])
            dqy = int(G_arr[i, 1] - G_arr[j, 1])
            if abs(dqx) <= pw_radius and abs(dqy) <= pw_radius:
                inv_eps_mat[i, j] = fft_inv[dqx % N, dqy % N]
    
    # Fewer k-points
    k_points = [
        (0.0, 0.0), (0.25, 0.0), (0.5, 0.0),
        (0.25, 0.25), (0.5, 0.25), (0.5, 0.5),
        (0.3, 0.1), (0.4, 0.2),
    ]
    
    max_w4 = 0.0
    min_w5 = float('inf')
    
    for k in k_points:
        kvec = np.array(k, dtype=float)
        kG = kvec + G_arr
        dots = np.sum(kG * kG, axis=1)
        M = np.diag(dots) @ inv_eps_mat
        M = (M + M.conj().T) / 2
        try:
            eigvals = eigh(M, eigvals_only=True)
            eigvals = np.sqrt(np.maximum(eigvals, 0))
            if len(eigvals) >= 5:
                max_w4 = max(max_w4, eigvals[3])
                min_w5 = min(min_w5, eigvals[4])
        except:
            pass
    
    if min_w5 == float('inf') or max_w4 <= 0:
        return -1.0
    return 2.0 * (min_w5 - max_w4) / (min_w5 + max_w4)


def _ensure_symmetry(cell):
    """Ensure the cell has full square symmetry about its center."""
    c = np.array(cell, dtype=float)
    N = len(c)
    # Average over all 8 symmetry operations
    c_sym = (
        c +
        c.T +
        c[::-1, :] +
        c[:, ::-1] +
        c[::-1, ::-1] +
        c.T[::-1, :] +
        c.T[:, ::-1] +
        c[::-1, ::-1].T
    ) / 8.0
    return (c_sym > 0.5).astype(int).tolist()


def _ga_search(deadline, pop_size=15, generations=50):
    """Genetic algorithm over the 32x32 pixel space with symmetry constraints."""
    N = 32
    half = N // 2
    
    # The independent pixels are in the upper-right quadrant with the diagonal
    # For a 32x32 cell centered at (15.5, 15.5), the symmetry group D4 has 8 elements.
    # Independent pixels: those in the region where row <= col and row + col < N
    # Actually, let's just work with the full cell and enforce symmetry after each mutation.
    
    # Initial population: mix of different structures
    def _random_cell():
        c = np.zeros((N, N), dtype=int)
        # Start with a base pattern
        mode = np.random.choice([0, 1, 2, 3])
        if mode == 0:  # Central rod
            r = np.random.uniform(4, 10)
            for i in range(N):
                for j in range(N):
                    x = j - half + 0.5
                    y = i - half + 0.5
                    if x*x + y*y < r*r:
                        c[i, j] = 1
        elif mode == 1:  # Central hole
            r = np.random.uniform(4, 10)
            c[:] = 1
            for i in range(N):
                for j in range(N):
                    x = j - half + 0.5
                    y = i - half + 0.5
                    if x*x + y*y < r*r:
                        c[i, j] = 0
        elif mode == 2:  # Checkerboard-like
            density = np.random.uniform(0.3, 0.7)
            c = (np.random.random((N, N)) < density).astype(int)
        else:  # Ring
            r_in = np.random.uniform(3, 6)
            r_out = np.random.uniform(r_in + 2, 12)
            for i in range(N):
                for j in range(N):
                    x = j - half + 0.5
                    y = i - half + 0.5
                    d2 = x*x + y*y
                    if r_in*r_in <= d2 <= r_out*r_out:
                        c[i, j] = 1
        
        return _ensure_symmetry(c.tolist())
    
    population = [_random_cell() for _ in range(pop_size)]
    
    # Evaluate initial population
    scores = []
    for cell in population:
        if time.time() > deadline - 60:
            break
        try:
            scores.append(_compute_gap_fast(cell, pw_radius=5))
        except:
            scores.append(-1.0)
    
    while time.time() < deadline - 30:
        # Sort by score
        order = np.argsort(scores)
        population = [population[i] for i in order]
        scores = [scores[i] for i in order]
        
        # Elitism: keep top 2
        new_pop = population[:2]
        
        # Generate offspring
        while len(new_pop) < pop_size:
            if time.time() > deadline - 30:
                break
            
            # Crossover + mutation
            p1 = population[np.random.randint(0, max(1, pop_size//2))]
            p2 = population[np.random.randint(0, max(1, pop_size//2))]
            c1 = np.array(p1, dtype=int)
            c2 = np.array(p2, dtype=int)
            
            # Uniform crossover
            mask = np.random.random((N, N)) > 0.5
            child = np.where(mask, c1, c2)
            
            # Mutation
            mut_rate = 0.05 + 0.05 * (1 - len(new_pop)/pop_size)
            mut_mask = np.random.random((N, N)) < mut_rate
            child[mut_mask] = 1 - child[mut_mask]
            
            child = _ensure_symmetry(child.tolist())
            new_pop.append(child)
        
        # Evaluate new population
        new_scores = []
        for cell in new_pop:
            if time.time() > deadline - 30:
                break
            try:
                new_scores.append(_compute_gap_fast(cell, pw_radius=5))
            except:
                new_scores.append(-1.0)
        
        # Merge and select
        all_pop = new_pop
        all_scores = new_scores
        
        # Keep best overall
        if all_scores and max(all_scores) > max(scores):
            pass  # Progress made
        
        population = all_pop
        scores = all_scores
    
    # Return best
    if scores:
        best_idx = np.argmax(scores)
        return population[best_idx], scores[best_idx]
    return _random_cell(), -1.0


def _local_refine(cell, deadline):
    """Local search: try flipping individual pixels or small groups."""
    N = 32
    c = np.array(cell, dtype=int)
    best_score = _compute_gap_fast(c.tolist(), pw_radius=6)
    best_c = c.copy()
    
    # Get list of independent pixels (upper triangle including diagonal, in the first quadrant)
    independent = []
    for i in range(N):
        for j in range(N):
            # Only consider pixels in the fundamental domain
            # By symmetry, pixel (i,j) determines (N-1-i, j), (i, N-1-j), (N-1-i, N-1-j),
            # (j, i), (j, N-1-i), (N-1-i, j), (N-1-j, i)
            # Independent: i <= j and i <= N-1-j (roughly upper-right triangle)
            if i <= j and i <= N - 1 - j:
                independent.append((i, j))
    
    np.random.shuffle(independent)
    
    for i, j in independent:
        if time.time() > deadline - 15:
            break
        
        # Try flipping this pixel and its symmetric partners
        test_c = best_c.copy()
        # Flip all symmetric partners
        positions = [
            (i, j), (i, N-1-j), (N-1-i, j), (N-1-i, N-1-j),
            (j, i), (j, N-1-i), (N-1-i, j), (N-1-j, i)
        ]
        for pi, pj in set(positions):
            test_c[pi, pj] = 1 - test_c[pi, pj]
        
        test_score = _compute_gap_fast(test_c.tolist(), pw_radius=6)
        if test_score > best_score + 1e-6:
            best_c = test_c
            best_score = test_score
    
    return _ensure_symmetry(best_c.tolist()), best_score


def design(problem):
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin before 230s limit
    
    size = problem["pixels"]
    assert size == 32
    
    # Phase 1: GA search with fast surrogate
    best_cell, best_score = _ga_search(deadline - 30, pop_size=15, generations=50)
    
    # Phase 2: Local refinement
    if time.time() < deadline - 25:
        best_cell, best_score = _local_refine(best_cell, deadline - 25)
    
    # Phase 3: Accurate evaluation and final refinement
    if time.time() < deadline - 15:
        try:
            acc_score = _compute_gap(best_cell, pw_radius=10)
        except:
            acc_score = best_score
        
        # Try a few more local flips with accurate scoring
        c = np.array(best_cell, dtype=int)
        N = 32
        independent = []
        for i in range(N):
            for j in range(N):
                if i <= j and i <= N - 1 - j:
                    independent.append((i, j))
        
        np.random.shuffle(independent)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i, j in independent:
                if time.time() > deadline - 10:
                    break
                test_c = c.copy()
                positions = [
                    (i, j), (i, N-1-j), (N-1-i, j), (N-1-i, N-1-j),
                    (j, i), (j, N-1-i), (N-1-i, j), (N-1-j, i)
                ]
                for pi, pj in set(positions):
                    test_c[pi, pj] = 1 - test_c[pi, pj]
                
                try:
                    test_score = _compute_gap(test_c.tolist(), pw_radius=10)
                except:
                    continue
                
                if test_score > acc_score + 1e-6:
                    c = test_c
                    acc_score = test_score
                    improved = True
        
        best_cell = _ensure_symmetry(c.tolist())
    
    return best_cell
