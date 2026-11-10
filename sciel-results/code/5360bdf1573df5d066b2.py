import time
import numpy as np
import random

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell to maximize the TM band gap 
    between the 4th and 5th bands.
    Uses Simulated Annealing with a fast surrogate model (13x13 plane waves).
    """
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]  # 1
    eps_high = problem["eps_high"]  # 12
    
    # Symmetry: The cell is determined by the top-left 16x16 quadrant.
    # cell[i][j] for i,j in 0..15
    # cell[i][31-j] = cell[i][j]
    # cell[31-i][j] = cell[i][j]
    # cell[31-i][31-j] = cell[i][j]
    
    half = size // 2  # 16
    
    # Initialize with a random binary pattern or a specific seed
    # A "hole" in a dielectric background is a good starting point for TM gaps
    # Let's start with a central air hole (0s) in a dielectric (1s) background
    # In the 16x16 quadrant, the center of the cell is at (15.5, 15.5) in full coords?
    # No, the cell is 0..31. Center is between 15 and 16.
    # The 16x16 quadrant covers 0..15.
    # A central hole would affect the pixels near (15,15) in the quadrant.
    
    # Let's initialize with a random pattern with 50% density
    # Or better, start with a structure known to have gaps, e.g., a square hole.
    # Let's try a square hole of size 8x8 in the center of the 32x32 cell.
    # In the 16x16 quadrant, this corresponds to rows 12-15, cols 12-15 being 0.
    
    grid = np.ones((half, half), dtype=np.int8)
    # Create a central square hole in the full cell
    # Full cell center is (15.5, 15.5).
    # Hole from x=12 to 19, y=12 to 19 in full coords.
    # In quadrant (0-15), this is x=12..15, y=12..15.
    grid[12:16, 12:16] = 0
    
    # Helper to expand 16x16 to 32x32
    def expand_grid(g):
        # g is 16x16
        # Top-left: g
        # Top-right: g[:, ::-1]
        # Bottom-left: g[::-1, :]
        # Bottom-right: g[::-1, ::-1]
        top = np.hstack((g, g[:, ::-1]))
        bottom = np.hstack((g[::-1, :], g[::-1, ::-1]))
        return np.vstack((top, bottom))
    
    # Surrogate Model Parameters
    N_pw = 13  # Number of plane waves in each direction (-6 to 6)
    # G vectors: k_x, k_y in [-6, 6]
    G_vectors = []
    for gx in range(-N_pw//2, N_pw//2 + 1):
        for gy in range(-N_pw//2, N_pw//2 + 1):
            G_vectors.append((gx, gy))
    N_vecs = len(G_vectors)
    
    # Precompute G^2
    G2 = np.array([gx*gx + gy*gy for gx, gy in G_vectors])
    
    # IBZ Points
    # Triangle: 0 <= kx <= 1, 0 <= ky <= 1-kx (in units of 2*pi/a? No, 1/a * 2pi?)
    # Standard BZ is [-pi/a, pi/a]. Let's use normalized k where BZ is [-0.5, 0.5].
    # IBZ: 0 <= kx <= 0.5, 0 <= ky <= 0.5 - kx.
    # We will sample a grid of points in this triangle.
    k_points = []
    num_k = 5 # 5x5 grid in the triangle
    for i in range(num_k):
        kx = i / (num_k - 1) * 0.5
        for j in range(num_k):
            ky = j / (num_k - 1) * (0.5 - kx)
            k_points.append((kx, ky))
    # Add center and high symmetry points
    k_points.append((0.5, 0)) # X point
    k_points.append((0, 0.5)) # Y point
    k_points.append((0.25, 0.25)) # Gamma
    k_points.append((0.5, 0.5)) # M point (corner)
    # Remove duplicates
    unique_k = list(set(k_points))
    
    def calculate_gap(grid_32):
        """
        Calculate the surrogate band gap using the 13x13 PW method.
        grid_32: 32x32 array of 0s and 1s.
        """
        # Compute Fourier coefficients eps(G)
        # eps(G) = 1/1024 * sum_{i,j} eps_{i,j} exp(-i G . r_{i,j})
        # r_{i,j} = (i/32, j/32)
        # G = (gx, gy)
        
        # Vectorized computation of eps(G) for all G vectors
        # Create meshgrid of pixel centers
        x = np.arange(size) / size
        y = np.arange(size) / size
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        eps_vals = grid_32.astype(float)
        # Map 0->eps_low, 1->eps_high
        eps_phys = np.where(eps_vals == 1, eps_high, eps_low)
        
        eps_G = np.zeros(N_vecs, dtype=complex)
        for idx, (gx, gy) in enumerate(G_vectors):
            # exp(-i 2pi (gx * X + gy * Y))
            phase = -2 * np.pi * (gx * X + gy * Y)
            # Sum over all pixels
            # Note: G is in units of reciprocal lattice vector.
            # The term is exp(i (k+G) r). For Fourier coeff of eps, it's exp(-i G r).
            eps_G[idx] = np.sum(eps_phys * np.exp(phase)) / (size * size)
        
        # Find the max omega_4 and min omega_5
        max_w4 = 0.0
        min_w5 = float('inf')
        
        for kx, ky in unique_k:
            # Construct Matrix A and B
            # A_{G,G'} = delta_{G,G'} |k+G|^2
            # B_{G,G'} = eps(G-G')
            
            # k is in units of 1/a? 
            # If BZ is [-0.5, 0.5] in units of 1/a, then k is just kx, ky.
            # G is integer.
            # |k+G|^2 = (kx+gx)^2 + (ky+gy)^2
            
            A = np.zeros((N_vecs, N_vecs), dtype=complex)
            B = np.zeros((N_vecs, N_vecs), dtype=complex)
            
            # Fill A (diagonal)
            for i, (gx, gy) in enumerate(G_vectors):
                val = (kx + gx)**2 + (ky + gy)**2
                A[i, i] = val
            
            # Fill B (Toeplitz)
            # B_{i,j} = eps(G_i - G_j)
            # We need a map from G vector to index
            G_index_map = {g: i for i, g in enumerate(G_vectors)}
            
            for i, g_i in enumerate(G_vectors):
                for j, g_j in enumerate(G_vectors):
                    diff = (g_i[0] - g_j[0], g_i[1] - g_j[1])
                    # Check if diff is in G_vectors
                    if diff in G_index_map:
                        B[i, j] = eps_G[G_index_map[diff]]
                    else:
                        # If diff is out of range, eps is 0 (assumed)
                        B[i, j] = 0.0
            
            # Solve generalized eigenvalue problem A C = w^2 B C
            # Use scipy.linalg.eigh for symmetric/Hermitian
            try:
                # B is Hermitian and positive definite (eps > 0)
                # A is Hermitian and positive semi-definite
                # We can convert to standard form: B^{-1/2} A B^{-1/2} c = w^2 c
                # Or just use eigh(A, B)
                from scipy.linalg import eigh
                w2, _ = eigh(A, B)
                # Sort eigenvalues
                w2 = np.sort(w2)
                # We want the 4th and 5th bands (indices 3 and 4)
                # Note: The bands are ordered by frequency.
                # w = sqrt(w2)
                w = np.sqrt(w2)
                
                # Extract 4th (index 3) and 5th (index 4)
                if len(w) >= 5:
                    w4 = w[3]
                    w5 = w[4]
                    max_w4 = max(max_w4, w4)
                    min_w5 = min(min_w5, w5)
            except Exception:
                # If solver fails, return a bad score
                return -1.0
        
        if min_w5 == float('inf') or max_w4 == 0.0:
            return -1.0
        
        gap = min_w5 - max_w4
        midgap = (min_w5 + max_w4) / 2.0
        if midgap == 0:
            return -1.0
        
        return 2.0 * gap / midgap

    # Initial evaluation
    current_grid_32 = expand_grid(grid)
    current_score = calculate_gap(current_grid_32)
    best_score = current_score
    best_grid = grid.copy()
    
    # Simulated Annealing Parameters
    T = 1.0
    cooling_rate = 0.99
    min_T = 0.01
    iterations = 0
    
    while T > min_T:
        # Check time
        if time.time() - start_time > time_limit:
            break
        
        # Propose a change
        # Flip a random pixel in the 16x16 quadrant
        r = random.randint(0, half - 1)
        c = random.randint(0, half - 1)
        
        # Store old value
        old_val = grid[r, c]
        grid[r, c] = 1 - old_val
        
        # Evaluate new state
        new_grid_32 = expand_grid(grid)
        new_score = calculate_gap(new_grid_32)
        
        # Acceptance
        delta = new_score - current_score
        if delta > 0 or random.random() < np.exp(delta / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_grid = grid.copy()
        else:
            # Revert
            grid[r, c] = old_val
        
        T *= cooling_rate
        iterations += 1
        
        # Optional: Restart if stuck? 
        # For now, just continue cooling.
        
    # Return the best grid found, expanded to 32x32
    final_grid = expand_grid(best_grid)
    # Convert to nested list of 0/1 ints
    result = [[int(final_grid[i][j]) for j in range(size)] for i in range(size)]
    return result

# EVOLVE-BLOCK-END
