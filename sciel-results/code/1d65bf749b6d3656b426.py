import numpy as np
import time
import math

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    # Safety margin: 230s wall clock limit, leave 20s for safety
    deadline = start_time + 210.0 

    # 1. Setup Fast Band Solver
    # Use 13x13 = 169 plane waves for internal search speed.
    # The evaluator uses 17x17=289, but 13x13 is often sufficient for ranking 
    # if the gap is robust. We use a slightly higher resolution 15x15=225 
    # to be safer, but the vectorized matvec makes it fast.
    half_pw = 7
    N_pw = 2 * half_pw + 1
    N_total = N_pw * N_pw
    
    # Generate G vectors
    G_m = np.arange(-half_pw, half_pw + 1)
    G_n = np.arange(-half_pw, half_pw + 1)
    G_vecs = np.array(np.meshgrid(G_m, G_n, indexing='ij')).reshape(2, -1).T
    G_vecs = G_vecs * (2.0 * np.pi)
    
    # Precompute k-points
    # The evaluator uses 30 points. We use a representative set.
    # Irreducible BZ for square lattice: 0 <= kx <= ky <= pi
    k_points = []
    # Gamma
    k_points.append((0.0, 0.0))
    # M point (pi, pi)
    k_points.append((np.pi, np.pi))
    # X points (0, pi) and (pi, 0) -> (0, pi) in wedge
    k_points.append((0.0, np.pi))
    # Line Gamma-M: (t*pi, t*pi)
    for t in np.linspace(0.1, 0.9, 8):
        k_points.append((t * np.pi, t * np.pi))
    # Line Gamma-X: (0, t*pi)
    for t in np.linspace(0.1, 0.9, 8):
        k_points.append((0.0, t * np.pi))
    # Line X-M: (t*pi, (1-t)*pi) -> in wedge 0<=kx<=ky<=pi
    # X is (0, pi), M is (pi, pi). Line is (t*pi, pi).
    for t in np.linspace(0.1, 0.9, 8):
        k_points.append((t * np.pi, np.pi))
        
    k_points = np.array(k_points)
    n_k = len(k_points)

    # Precompute k+G vectors for all k-points? 
    # kG[k, i, 0] = kx_k + Gx_i
    # kG[k, i, 1] = ky_k + Gy_i
    kG = np.zeros((n_k, N_total, 2))
    for i_k, (kx, ky) in enumerate(k_points):
        kG[i_k, :, 0] = kx + G_vecs[:, 0]
        kG[i_k, :, 1] = ky + G_vecs[:, 1]
        
    # Precompute dot products (k+G) . (k+G') for each k-point
    # D[k, i, j] = kG[k,i,0]*kG[k,j,0] + kG[k,i,1]*kG[k,j,1]
    # This is (N_k, N_total, N_total). 
    # 30 * 225 * 225 * 8 bytes ~ 12 MB. Acceptable.
    D = np.zeros((n_k, N_total, N_total))
    for i_k in range(n_k):
        # Outer product trick
        D[i_k] = np.outer(kG[i_k, :, 0], kG[i_k, :, 0]) + \
                 np.outer(kG[i_k, :, 1], kG[i_k, :, 1])

    def compute_score(cell):
        """
        Compute the 4-5 TM band gap score.
        cell: 32x32 array of 0/1.
        """
        # Convert to float and compute 1/eps
        # eps = 1 + 11*cell
        # 1/eps = 1/(1+11*cell)
        eps_grid = 1.0 + 11.0 * np.array(cell, dtype=float)
        inv_eps_grid = 1.0 / eps_grid
        
        # FFT of 1/eps to get Fourier coefficients
        # inv_eps_F[m, n] = coeff for G = (2*pi*m, 2*pi*n)
        # Note: np.fft.fft2 returns coefficients in order 0..N-1.
        # We need to map G_vecs indices to FFT indices.
        # G_vecs are in -half_pw..half_pw.
        # FFT index for m is m % N_pw.
        inv_eps_F = np.fft.fft2(inv_eps_grid) / (size * size)
        
        min_omega5 = np.inf
        max_omega4 = 0.0
        
        # We need to solve A v = lambda v for each k-point.
        # A is Hermitian. We can use np.linalg.eigvalsh.
        # But constructing A explicitly is O(N_total^3) per k-point if done naively?
        # No, constructing A is O(N_total^2). Eigvalsh is O(N_total^3).
        # 225^3 ~ 11M ops. 30 k-points * 11M ~ 330M ops per evaluation.
        # This might be too slow for thousands of evaluations.
        
        # Optimization: Use a smaller basis for the internal loop?
        # Or use a cheaper approximation?
        # Let's try to make the eigensolve faster.
        # Actually, 330M floating point ops is ~0.1s in optimized BLAS.
        # 1000 evals * 0.1s = 100s. This fits in the budget.
        
        # Let's precompute the index mapping for inv_eps_F
        # G_vecs[:, 0] are m values.
        # m_fft = (m + half_pw) % N_pw ? 
        # No, standard FFT: coeff[k] corresponds to frequency k if k < N/2, k-N if k >= N/2.
        # Here N_pw is odd.
        # m ranges -7..7.
        # FFT index 0 -> 0
        # FFT index 1 -> 1
        # ...
        # FFT index 7 -> 7
        # FFT index 8 -> -7 (since 8-15 = -7)
        # So index = m % N_pw works?
        # -7 % 15 = 8. Correct.
        
        # Precompute the index array for E matrix construction
        # E[i, j] = inv_eps_F[(Gx_i - Gx_j) % N_pw, (Gy_i - Gy_j) % N_pw]
        Gx = G_vecs[:, 0]
        Gy = G_vecs[:, 1]
        G_diff_x = (Gx[:, None] - Gx[None, :]) % N_pw
        G_diff_y = (Gy[:, None] - Gy[None, :]) % N_pw
        
        # E_indices is a 2D array of indices into inv_eps_F
        # We can't just index with 2D array easily in a vectorized way for all k-points at once
        # without creating a huge 3D array.
        # Let's loop over k-points.
        
        for i_k in range(n_k):
            # Construct E matrix for this k-point? 
            # E is the same for all k-points! It only depends on G_diff.
            # E[i, j] = inv_eps_{G_i - G_j}
            # So E is constant for a given cell.
            # Wait, D depends on k. A = D * E (element-wise).
            
            # We can precompute E once per cell.
            # E is N_total x N_total.
            
            # Let's do it inside the loop for now, but E is constant.
            # Better: compute E once before the k-loop.
            
            pass # Placeholder, logic moved below

        # Compute E once
        # inv_eps_F is N_pw x N_pw.
        # We need to gather values based on G_diff_x and G_diff_y.
        # G_diff_x and G_diff_y are N_total x N_total.
        # We can use advanced indexing.
        E = inv_eps_F[G_diff_x, G_diff_y]
        
        # Now loop over k-points
        for i_k in range(n_k):
            A = D[i_k] * E
            
            # Solve eigenvalues
            # A is real symmetric? 
            # inv_eps is real and even (if cell is symmetric? No, cell is symmetric, but 1/eps is not necessarily even in G? 
            # 1/eps is real. FFT of real signal is Hermitian. 
            # E[i,j] = c_{G_i-G_j}. c_{-G} = c_G^*. 
            # Since 1/eps is real, c_G is real? No, c_G is complex in general.
            # However, if the cell has inversion symmetry or is real, the structure is specific.
            # Our cell is symmetric under D4. 
            # Is A Hermitian? 
            # A_{ij} = (k+G_i).(k+G_j) c_{G_i-G_j}
            # A_{ji} = (k+G_j).(k+G_i) c_{G_j-G_i}
            # Dot product is real and symmetric.
            # c_{-G} = c_G^* (for real signal).
            # So A_{ji} = A_{ij}^*. Yes, Hermitian.
            
            try:
                vals = np.linalg.eigvalsh(A)
                vals = np.sort(vals.real)
                
                if len(vals) < 5:
                    return -1.0
                
                # Bands 4 and 5 (indices 3 and 4)
                omega4 = np.sqrt(max(0.0, vals[3]))
                omega5 = np.sqrt(max(0.0, vals[4]))
                
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

    def apply_symmetry(cell):
        """
        Enforce D4 symmetry (square symmetry group) on the cell.
        Averages the 8 symmetric copies and thresholds at 0.5.
        """
        # Cell is 32x32. Center is between pixels (15.5, 15.5).
        # Symmetries:
        # 1. Identity
        # 2. Rotate 90
        # 3. Rotate 180
        # 4. Rotate 270
        # 5. Reflect X
        # 6. Reflect Y
        # 7. Reflect Diagonal 1
        # 8. Reflect Diagonal 2
        
        # It's easier to just take the min or max? 
        # To ensure binary 0/1, we can sum the 8 versions and threshold.
        # But we must be careful with the center.
        # For 32x32, there is no single center pixel.
        
        c = np.array(cell, dtype=float)
        
        # Rotations
        r90 = np.rot90(c, 1)
        r180 = np.rot90(c, 2)
        r270 = np.rot90(c, 3)
        
        # Reflections
        # Reflect across vertical axis (x -> -x)
        # In array indices: flip left-right
        fx = c[:, ::-1]
        # Reflect across horizontal axis (y -> -y)
        fy = c[::-1, :]
        
        # Diagonals
        # Main diagonal (transpose)
        d1 = c.T
        # Anti-diagonal
        d2 = np.rot90(c.T, 1) # or c[::-1, ::-1].T
        
        # Sum all 8
        s = c + r90 + r180 + r270 + fx + fy + d1 + d2
        
        # Threshold
        # If sum >= 4 (half of 8), it's likely dielectric?
        # Or just > 0.5?
        # If we start with a random cell, the average might be noisy.
        # Better: take the majority vote?
        # If sum > 4, set to 1.
        new_cell = (s > 4.0).astype(int)
        
        return new_cell

    def generate_initial_cell():
        """
        Generate a good starting cell.
        4 large dielectric circles at the corners.
        """
        cell = np.zeros((size, size), dtype=int)
        i, j = np.mgrid[0:size, 0:size]
        
        # Centers at (4,4), (27,4), (4,27), (27,27)
        # Radius 6
        centers = [(4, 4), (27, 4), (4, 27), (27, 27)]
        R = 6.0
        
        for cx, cy in centers:
            dist = np.sqrt((i - cx)**2 + (j - cy)**2)
            cell[dist <= R] = 1
            
        return apply_symmetry(cell)

    # Search Loop
    best_cell = generate_initial_cell()
    best_score = compute_score(best_cell)
    
    # If initial is bad, try a few random starts
    if best_score < 0.05:
        for _ in range(5):
            # Random symmetric cell
            c = np.random.rand(size, size)
            c = (c > 0.5).astype(int)
            c = apply_symmetry(c)
            s = compute_score(c)
            if s > best_score:
                best_score = s
                best_cell = c

    current_cell = best_cell.copy()
    current_score = best_score
    
    # Simulated Annealing / Hill Climbing
    # We mutate a few pixels and check if score improves.
    # To speed up, we can batch evaluate? No, compute_score is sequential.
    
    # Estimate time per eval
    t0 = time.time()
    compute_score(current_cell)
    t1 = time.time()
    eval_time = t1 - t0
    
    # Adjust number of iterations if needed? 
    # Just run until deadline.
    
    temp = 0.1
    cool_rate = 0.99
    
    while time.time() < deadline - 2.0:
        # Generate a mutation
        # Flip 1-3 random pixels
        n_mutations = np.random.randint(1, 4)
        
        new_cell = current_cell.copy()
        for _ in range(n_mutations):
            r = np.random.randint(0, size)
            c = np.random.randint(0, size)
            new_cell[r, c] = 1 - new_cell[r, c]
            
        # Enforce symmetry
        new_cell = apply_symmetry(new_cell)
        
        # If no change, skip
        if np.array_equal(new_cell, current_cell):
            continue
            
        new_score = compute_score(new_cell)
        
        if new_score > current_score:
            current_cell = new_cell
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_cell = current_cell.copy()
                temp *= cool_rate # Cool down on success
        else:
            # Accept with probability exp((new - cur) / temp)
            delta = new_score - current_score
            if delta < 0:
                prob = math.exp(delta / temp)
                if np.random.random() < prob:
                    current_cell = new_cell
                    current_score = new_score
            temp *= cool_rate
            if temp < 0.01:
                temp = 0.01

    return best_cell.tolist()

