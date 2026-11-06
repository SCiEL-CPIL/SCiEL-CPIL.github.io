import math
import time
import numpy as np

def design(problem):
    """
    Design a symmetric 32x32-pixel unit cell to maximize the TM band gap
    between bands 4 and 5 using wavelet-coefficient parameterization
    with gradient-free optimization in coefficient space.
    """
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]   # 1
    eps_high = problem["eps_high"] # 12
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110s out of 120s budget (with margin)
    
    def quadrant_to_full(q):
        """Convert a 16x16 quadrant to a full 32x32 symmetric cell."""
        q = np.array(q, dtype=int)
        top = np.hstack([q, q[:, ::-1]])
        bottom = top[::-1, :]
        return np.vstack([top, bottom])
    
    def full_to_quadrant(cell):
        """Extract the top-left 16x16 quadrant from a full 32x32 cell."""
        return np.array(cell[:16, :16], dtype=int)
    
    def compute_gap(cell_2d):
        """
        Compute the gap-midgap ratio between TM bands 4 and 5.
        Uses a vectorized plane-wave expansion.
        """
        N = 32
        eps = np.where(np.array(cell_2d) == 1, eps_high, eps_low)
        
        npw = 8
        # Generate G vectors
        mx = np.arange(-npw, npw+1)
        my = np.arange(-npw, npw+1)
        Gx, Gy = np.meshgrid(mx, my, indexing='ij')
        Gx = Gx.ravel()
        Gy = Gy.ravel()
        G = np.column_stack((Gx, Gy))
        nG = len(G)
        
        inv_eps = 1.0 / eps
        
        # Compute Fourier coefficients of 1/epsilon
        # x, y are pixel centers
        x = (np.arange(N) + 0.5) / N
        y = (np.arange(N) + 0.5) / N
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Vectorized calculation of exp(-i (Gx*X + Gy*Y))
        # G shape (nG, 2), X shape (N, N)
        # Phase shape (nG, N, N)
        phase = -1j * (Gx[:, None, None] * X[None, :, :] + Gy[:, None, None] * Y[None, :, :])
        exp_term = np.exp(phase)
        
        # inv_eps_coeffs[i] = sum(inv_eps * exp_term[i]) / N^2
        inv_eps_coeffs = np.sum(inv_eps[None, :, :] * exp_term, axis=(1, 2)) / (N * N)
        
        # Precompute G difference indices mapping
        # We need A[i, j] = inv_eps_coeffs[index of G[i]-G[j]]
        # G values are 2*pi*m. Difference is 2*pi*(m_i - m_j).
        # We can map (mx_diff, my_diff) to index.
        # Since G is generated from -npw to npw, differences range -2*npw to 2*npw.
        # We can create a lookup table for differences.
        
        # Alternative: Directly construct A using convolution-like property or just loop over i, j if nG is small (289).
        # nG=289 is small enough for a simple loop, but let's try to vectorize A construction.
        # G_diff[i, j, :] = G[i] - G[j]
        # This is (289, 289, 2).
        
        G_diff = G[:, None, :] - G[None, :, :]
        # G_diff is (nG, nG, 2)
        # We need to find index in G for each G_diff[i, j].
        # Since G contains all combinations of -npw..npw, and differences are -2npw..2npw,
        # not all differences are in G. But for a square lattice, if G_i and G_j are in the set,
        # their difference is also in the set IF the set is closed under subtraction? No.
        # The set G is a finite truncation. The difference G_i - G_j might fall outside the -npw..npw range.
        # However, for the matrix element <i|1/eps|j>, we need the Fourier coefficient at G_i - G_j.
        # If G_i - G_j is outside the computed range, the coefficient is effectively 0 (or we assume truncation).
        # In the previous code, it checked `if key in G_dict`.
        
        # Let's build a dictionary for G vectors to indices
        G_dict = {}
        for idx in range(nG):
            G_dict[(G[idx, 0], G[idx, 1])] = idx
            
        A = np.zeros((nG, nG), dtype=complex)
        for i in range(nG):
            for j in range(nG):
                key = (G_diff[i, j, 0], G_diff[i, j, 1])
                if key in G_dict:
                    A[i, j] = inv_eps_coeffs[G_dict[key]]
        
        k_points = [
            (0.0, 0.0),
            (np.pi, 0.0),
            (np.pi/2, np.pi/2),
            (np.pi/2, 0.0),
            (np.pi/2, np.pi/4),
            (3*np.pi/8, np.pi/8),
            (np.pi/4, np.pi/4),
            (np.pi/3, np.pi/6),
            (2*np.pi/5, np.pi/5),
            (np.pi/2, np.pi/6),
        ]
        
        max_omega4 = 0.0
        min_omega5 = np.inf
        
        for kx, ky in k_points:
            kvec = np.array([kx, ky])
            k_plus_G = kvec[None, :] + G
            kG_sq = np.sum(k_plus_G**2, axis=1)
            
            H = A * kG_sq[None, :]
            H = (H + H.conj().T) / 2
            
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals.real)
            
            if len(eigvals) >= 6:
                omega4 = eigvals[4]
                omega5 = eigvals[5]
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        
        if min_omega5 == np.inf or max_omega4 == 0:
            return 0.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2
        
        if midgap <= 0:
            return 0.0
        
        return gap / midgap
    
    # Cellular Automaton (Game of Life) based design strategy
    # 1. Generate a few CA seeds
    # 2. Evolve them
    # 3. Symmetrize to 16x16 quadrant
    # 4. Local search (hill climbing) on the quadrant
    
    def generate_ca_seed(n=16, density=0.5):
        """Generate a random symmetric seed for Game of Life."""
        # Start with random noise
        seed = np.random.rand(n, n) > density
        # Enforce D4 symmetry on the seed to keep it simple? 
        # Or just let CA evolve and symmetrize later.
        # Let's just use random noise.
        return seed.astype(int)
    
    def step_life(grid):
        """One step of Conway's Game of Life on a toroidal grid."""
        n = grid.shape[0]
        # Convolve to count neighbors
        # Kernel: 1 1 1
        #          1 0 1
        #          1 1 1
        # But center is 0.
        # We can use np.roll
        neighbors = np.zeros_like(grid)
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                neighbors += np.roll(np.roll(grid, i, axis=0), j, axis=1)
        
        # Rules:
        # 1. Any live cell with 2 or 3 neighbors survives.
        # 2. Any dead cell with exactly 3 neighbors becomes a live cell.
        # 3. All other live cells die.
        
        new_grid = np.zeros_like(grid)
        # Survive
        new_grid[(grid == 1) & ((neighbors == 2) | (neighbors == 3))] = 1
        # Birth
        new_grid[(grid == 0) & (neighbors == 3)] = 1
        
        return new_grid
    
    def symmetrize_quadrant(q):
        """Ensure the 16x16 quadrant is consistent with D4 symmetry of the full cell.
        Actually, the quadrant_to_full function handles the expansion.
        We just need to make sure the quadrant itself doesn't have conflicting boundaries?
        No, quadrant_to_full mirrors it. So any 16x16 binary array is valid.
        """
        return q
    
    # Generate initial candidates
    candidates = []
    
    # Candidate 1: 2x2 Rods (Known good structure)
    q_rod = np.zeros((16, 16), dtype=int)
    # Place rods at centers of the 4 sub-quadrants of the 16x16?
    # The full cell is 32x32. The quadrant is 16x16.
    # The full cell has 4 "super-pixels" if we think of 16x16 blocks? No.
    # The 16x16 quadrant maps to the top-left 16x16 of the 32x32.
    # A 2x2 rod pattern in the 32x32 cell would have rods at (8,8), (8,24), (24,8), (24,24) roughly.
    # In the 16x16 quadrant, this corresponds to a rod at (8,8) of the quadrant?
    # Let's just place a blob at (8,8) in the quadrant.
    X, Y = np.meshgrid(np.arange(16), np.arange(16), indexing='ij')
    q_rod = ((X - 8)**2 + (Y - 8)**2 < 4**2).astype(int)
    candidates.append(q_rod)
    
    # Candidate 2-5: CA Evolved
    for i in range(4):
        seed = generate_ca_seed(16, density=0.3 + 0.1*i)
        for step in range(5 + i*2):
            seed = step_life(seed)
        candidates.append(seed.astype(int))
    
    best_score = -np.inf
    best_quadrant = candidates[0]
    
    # Evaluate all candidates and pick the best
    for q in candidates:
        full_cell = quadrant_to_full(q)
        score = compute_gap(full_cell)
        if score > best_score:
            best_score = score
            best_quadrant = q
    
    # Local Search (Hill Climbing)
    # Flip random pixels in the quadrant, check if score improves.
    # Use a time-based loop.
    
    current_q = best_quadrant.copy()
    current_score = best_score
    
    # Precompute indices for faster random access? Not really needed for 16x16.
    # Just loop.
    
    # To speed up, we can try flipping pixels that are on boundaries first?
    # Or just random.
    
    # We will run this until deadline - 5s
    while time.time() < deadline - 5:
        i = np.random.randint(0, 16)
        j = np.random.randint(0, 16)
        
        old_val = current_q[i, j]
        current_q[i, j] = 1 - old_val
        
        full_cell = quadrant_to_full(current_q)
        score = compute_gap(full_cell)
        
        if score > current_score:
            current_score = score
            if score > best_score:
                best_score = score
                best_quadrant = current_q.copy()
        else:
            current_q[i, j] = old_val
            
            # If we haven't improved in a while, maybe restart from best?
            # Simple strategy: just keep trying.
    
    # Final evaluation of best_quadrant
    best_full_cell = quadrant_to_full(best_quadrant)
    
    return best_full_cell.tolist()
