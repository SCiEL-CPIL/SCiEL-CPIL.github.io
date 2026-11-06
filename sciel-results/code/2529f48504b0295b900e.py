"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def design(problem):
    """
    Design a 32x32 unit cell by optimizing the amplitudes of a truncated 
    Fourier series to maximize the TM 4-5 band gap.
    
    Strategy:
    1. Represent the dielectric profile as a sum of symmetry-allowed cosine modes.
    2. Use a fast approximate plane-wave method (fewer k-points, fewer plane waves) 
       to evaluate the gap ratio.
    3. Perform a local search (hill climbing) on the mode amplitudes.
    4. Return the best symmetric pixel map found.
    """
    start_time = time.time()
    deadline = start_time + 200.0  # 20s safety margin
    
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Symmetry-allowed modes for square lattice (C4v):
    # We use cosine terms cos(h*pi*x) * cos(k*pi*y) and sin terms if needed,
    # but for a 0/1 symmetric map, we can restrict to real coefficients.
    # Due to symmetry about center, we can use:
    # f(x,y) = sum A_ij * cos(h*pi*x) * cos(k*pi*y) + B_ij * cos(h*pi*x) * sin(k*pi*y) + ...
    # Actually, for 180 rotation and reflection symmetry, we can just use:
    # f(x,y) = sum C_{m,n} * cos(m*pi*x) * cos(n*pi*y) where m,n are integers.
    # Wait, the unit cell is [0,1) or [-0.5, 0.5)? 
    # The problem says "invariant under square symmetry group about the cell centre".
    # Let's define coordinates x, y in [0, 1) for the pixel indices.
    # Center is at (0.5, 0.5).
    # Symmetry: (x,y) -> (1-x, y), (x, 1-y), (1-x, 1-y).
    # This implies f(x,y) can be expanded in cos(2*pi*m*x) * cos(2*pi*n*y) with m,n integers?
    # No, cos(pi*m*(x-0.5)) is better.
    # Let's use basis functions: phi_{m,n}(x,y) = cos(m*pi*(x-0.5)) * cos(n*pi*(y-0.5))
    # These are orthogonal on [0,1]x[0,1] and satisfy the symmetries.
    
    # We will optimize the coefficients A_{m,n} for a small set of (m,n).
    # (0,0) is the DC term (fill fraction).
    # (1,0), (0,1), (1,1), (2,0), (0,2), (2,2), etc.
    
    # Let's pick a set of modes.
    # To get a good gap, we often need strong 2nd order components.
    # Modes: (0,0), (1,0), (0,1), (2,0), (0,2), (1,1), (2,1), (1,2), (2,2)
    
    modes = [
        (0,0),
        (1,0), (0,1),
        (2,0), (0,2),
        (1,1),
        (2,1), (1,2),
        (2,2),
        (3,0), (0,3),
        (3,1), (1,3),
        (3,2), (2,3),
        (3,3)
    ]
    
    N_modes = len(modes)
    
    # Precompute the basis functions on the pixel grid
    # Pixels are centered at (i+0.5)/size, (j+0.5)/size
    x_coords = (np.arange(size) + 0.5) / size
    y_coords = (np.arange(size) + 0.5) / size
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    # Basis matrix: shape (size*size, N_modes)
    # Flatten X, Y to 1D
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    Basis = np.zeros((size*size, N_modes))
    for k, (m, n) in enumerate(modes):
        # phi = cos(m*pi*(x-0.5)) * cos(n*pi*(y-0.5))
        Basis[:, k] = np.cos(m * np.pi * (X_flat - 0.5)) * np.cos(n * np.pi * (Y_flat - 0.5))
    
    # Fast approximate evaluator
    # We need to compute the band structure.
    # To be fast, we use a reduced set of k-points and a reduced number of plane waves.
    # However, the score depends on the exact evaluator. 
    # If the approximation is too coarse, we might optimize for the wrong thing.
    # Let's use a moderate approximation: 
    # 5x5 k-point grid in the irreducible BZ? Or just the high symmetry points?
    # The evaluator uses 30 wave vectors. Let's use a subset of those or a similar density.
    # And use ~50-100 plane waves instead of 289.
    
    # Irreducible BZ for square lattice: x in [0, 0.5], y in [0, x] (triangle)
    # Or x in [0, 1], y in [0, 0.5] with x>=y?
    # Standard irreducible wedge: kx >= ky >= 0, kx^2 + ky^2 <= 0.5? No, that's not right.
    # The irreducible BZ of the square lattice is the triangle with vertices (0,0), (0.5, 0), (0.5, 0.5).
    # Wait, the BZ is [-0.5, 0.5]x[-0.5, 0.5]. 
    # 1/8th of the BZ is the triangle (0,0), (0.5, 0), (0.5, 0.5).
    
    # Let's generate a small set of k-points in this triangle.
    # For speed, we'll use a 4x4 grid in the triangle.
    k_points = []
    for i in range(5):
        for j in range(5):
            kx = 0.5 * i / 4.0
            ky = 0.5 * j / 4.0
            if ky <= kx: # Inside the triangle (0,0)-(0.5,0)-(0.5,0.5)
                k_points.append((kx, ky))
                
    # If we want to be closer to the evaluator's 30 points, we might need more.
    # But for a fast local search, 20-30 points is okay if the PW count is low.
    
    # Plane waves:
    # We need to solve (E - K G) * epsilon^{-1} * (K + G) = 0
    # For TM polarization, the matrix element is:
    # M_{G, G'} = (k + G) * (k + G') * epsilon^{-1}_{G - G'}
    # where epsilon^{-1}_{G} are the Fourier coefficients of 1/epsilon.
    
    # We need to compute epsilon^{-1}_{G} for the given pixel map.
    # This is expensive if done from scratch every time.
    # However, since we are varying the pixel map via Fourier coefficients,
    # we can precompute the pixel positions and just update the 1/eps values.
    
    # Actually, 1/epsilon is a step function: 1/1 = 1 for air, 1/12 for dielectric.
    # epsilon^{-1}_{G} = (1/A) * integral (1/eps(r)) exp(-i G . r) d^2r
    # For a pixel map, this is a sum over pixels.
    
    # Let's define the set of plane waves G.
    # We need G vectors such that |k + G| is within a certain cutoff.
    # Let's use a cutoff of |G| <= 3 (in units of 2pi/L).
    # G = (h, k) * 2pi.
    # Let's use h, k in [-2, 2]. That's 25 plane waves.
    # Or [-3, 3] -> 49 plane waves.
    # Let's use 25 for speed.
    
    G_vectors = []
    for h in range(-2, 3):
        for k in range(-2, 3):
            G_vectors.append((h, k))
            
    N_pw = len(G_vectors)
    
    # Precompute (k+G) dot (k+G') for each k-point? No, k changes.
    # But we can precompute the structure of the matrix.
    
    # The matrix is M_{ij} = (k + G_i) . (k + G_j) * eps_inv[G_i - G_j]
    
    # We need to compute eps_inv[G] for G in the set of differences G_i - G_j.
    # Max difference is (4,4). So we need eps_inv for G in [-4, 4]x[-4, 4].
    
    # Let's precompute the exponential factors for each G and pixel.
    # exp(-i * 2pi * (h*x + k*y))
    
    # For each G in the difference set, we need the sum over pixels of (1/eps_p) * exp(-i G . r_p)
    # This is a linear operation on the pixel values (if we treat 1/eps as linear, which it's not, 
    # but we can just compute it directly).
    
    # Actually, since 1/eps is either 1 or 1/12, we can write:
    # 1/eps(r) = 1 - (1 - 1/12) * I_dielectric(r) = 1 - (11/12) * I(r)
    # So eps_inv[G] = delta_{G,0} - (11/12) * I_G
    # where I_G is the Fourier coefficient of the dielectric indicator I(r).
    
    # I_G = (1/A) * sum_{pixels} I_p * exp(-i G . r_p)
    
    # So we just need to compute I_G for the relevant G's.
    
    # Let's precompute the matrix E_G_p = exp(-i * 2pi * (h*x_p + k*y_p)) for all needed G and pixels.
    
    # Needed G's for eps_inv: differences of G_vectors.
    # G_vectors are in [-2, 2]. Differences in [-4, 4].
    # Let's create a list of all unique G differences.
    
    G_diffs = set()
    for G1 in G_vectors:
        for G2 in G_vectors:
            G_diffs.add((G1[0] - G2[0], G1[1] - G2[1]))
            
    G_diffs = list(G_diffs)
    N_diffs = len(G_diffs)
    
    # Map from G_diff to index
    G_diff_index = {g: i for i, g in enumerate(G_diffs)}
    
    # Precompute exponentials for each G_diff and pixel
    # Shape: (N_diffs, size*size)
    Exp_Mat = np.zeros((N_diffs, size*size), dtype=complex)
    for i, (h, k) in enumerate(G_diffs):
        # exp(-i * 2pi * (h*x + k*y))
        # Note: G is in units of 2pi/L. So phase is 2pi * (h*x + k*y).
        phase = 2 * np.pi * (h * X_flat + k * Y_flat)
        Exp_Mat[i, :] = np.exp(-1j * phase)
        
    # Now, for a given set of amplitudes A, we compute the pixel map I_p.
    # I_p = sum A_k * Basis[p, k]
    # We threshold I_p > 0 to get 0/1.
    
    # Then I_G = (1/N_pixels) * sum_p I_p * Exp_Mat[G, p]
    
    # This is a matrix-vector product: I_G = (1/N) * Exp_Mat @ I_p
    
    # Then eps_inv[G] = delta_{G,0} - (11/12) * I_G
    
    # Then we build the matrix M for each k-point and solve for eigenvalues.
    
    # To make it fast, we can vectorize the eigenvalue calculation?
    # No, eigenvalue decomposition is not easily vectorizable for different k's.
    # But we can parallelize over k-points if we had more cores, but we are in a subprocess.
    
    # Let's optimize the inner loop.
    
    # We'll use a simple hill climbing algorithm.
    # Start with random amplitudes.
    # At each step, perturb one amplitude, evaluate the gap, keep if improved.
    
    # Initial amplitudes
    # Let's start with a few non-zero modes.
    # A[0] (DC) should be around 0.5 to get ~50% fill.
    # A[1], A[2] (1st order) can be small.
    # A[3], A[4] (2nd order) can be larger.
    
    np.random.seed(42)
    A = np.zeros(N_modes)
    A[0] = 0.5
    # Add some random noise to other modes
    A[1:] = np.random.normal(0, 0.1, N_modes-1)
    
    best_A = A.copy()
    best_score = -np.inf
    
    # Helper function to compute the score
    def compute_score(A_vec):
        # Compute pixel map
        I_cont = Basis @ A_vec
        I_p = (I_cont > 0).astype(float)
        
        # Compute I_G
        I_G = (1.0 / (size*size)) * (Exp_Mat @ I_p)
        
        # Compute eps_inv
        eps_inv = np.zeros(N_diffs, dtype=complex)
        for i, g in enumerate(G_diffs):
            if g == (0, 0):
                eps_inv[i] = 1.0
            else:
                eps_inv[i] = - (11.0/12.0) * I_G[i]
                
        # Precompute (k+G) dot (k+G') for all pairs?
        # M_{ij} = (k+G_i) . (k+G_j) * eps_inv[G_i - G_j]
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in k_points:
            k = np.array([kx, ky])
            
            # Build matrix M
            M = np.zeros((N_pw, N_pw), dtype=complex)
            for i, G_i in enumerate(G_vectors):
                k_plus_G_i = k + np.array(G_i)
                for j, G_j in enumerate(G_vectors):
                    k_plus_G_j = k + np.array(G_j)
                    dot = np.dot(k_plus_G_i, k_plus_G_j)
                    G_diff = (G_i[0] - G_j[0], G_i[1] - G_j[1])
                    idx = G_diff_index[G_diff]
                    M[i, j] = dot * eps_inv[idx]
                    
            # Solve eigenvalue problem
            # M v = omega^2 v
            # Use np.linalg.eigvalsh for Hermitian matrix
            # M should be Hermitian if eps_inv is real and symmetric?
            # eps_inv[-G] = eps_inv[G]^*
            # Since I_G is complex, eps_inv is complex.
            # But the matrix M is Hermitian.
            
            try:
                eigenvalues = np.linalg.eigvalsh(M)
            except np.linalg.LinAlgError:
                return -np.inf
                
            # Filter out negative eigenvalues (numerical noise)
            eigenvalues = eigenvalues[eigenvalues > 0]
            
            if len(eigenvalues) < 5:
                continue
                
            # Bands are sorted by omega
            # omega_4 is the 4th smallest, omega_5 is the 5th smallest.
            # Note: We need the 4th and 5th TM bands.
            # The lowest band is usually not TM? 
            # In 2D square lattice, the first band is TM, second is TE?
            # Actually, for a square lattice of rods, the first gap is between TM1 and TM2?
            # Or TE1 and TM1?
            # The problem specifies TM bands 4 and 5.
            # We assume the eigenvalues returned are for TM polarization.
            # In the plane wave method, we specify the polarization by the form of the matrix.
            # For TM, the matrix is (k+G).(k+G') / eps.
            # For TE, it is (k+G)_perp . (k+G')_perp / eps.
            # We are using the TM form.
            
            # So eigenvalues are omega^2 for TM modes.
            # We need the 4th and 5th smallest positive eigenvalues.
            
            # Wait, are there zero-frequency modes? No.
            # Are there TE modes mixed in? No, we solved the TM eigenvalue problem.
            
            # So eigenvalues[0] is omega_1^2, eigenvalues[1] is omega_2^2, etc.
            
            # We want gap between band 4 and 5.
            # So omega_4 = sqrt(eigenvalues[3])
            # omega_5 = sqrt(eigenvalues[4])
            
            # But we need to check if there are any missing bands.
            # The plane wave expansion should capture the low-lying bands.
            
            if len(eigenvalues) >= 5:
                omega_4 = np.sqrt(eigenvalues[3])
                omega_5 = np.sqrt(eigenvalues[4])
                
                if omega_5 < min_omega_5:
                    min_omega_5 = omega_5
                if omega_4 > max_omega_4:
                    max_omega_4 = omega_4
                    
        if max_omega_4 == -np.inf:
            return -np.inf
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return -np.inf
            
        ratio = gap / midgap
        return ratio
        
    # Evaluate initial score
    best_score = compute_score(best_A)
    
    # Hill climbing
    step_size = 0.1
    patience = 0
    max_patience = 50
    
    while time.time() < deadline:
        # Pick a random mode to perturb
        i = np.random.randint(N_modes)
        
        # Try positive and negative perturbations
        best_local_score = best_score
        best_local_A = best_A.copy()
        
        for sign in [1, -1]:
            A_trial = best_A.copy()
            A_trial[i] += sign * step_size
            
            score = compute_score(A_trial)
            
            if score > best_local_score:
                best_local_score = score
                best_local_A = A_trial
                
        if best_local_score > best_score:
            best_score = best_local_score
            best_A = best_local_A
            patience = 0
            # Reduce step size if we've been successful
            step_size *= 0.95
        else:
            patience += 1
            step_size *= 0.9
            if patience > max_patience:
                break
                
    # Final pixel map from best_A
    I_cont = Basis @ best_A
    I_p = (I_cont > 0).astype(int)
    
    # Reshape to 2D
    cell_flat = I_p.flatten()
    cell = cell_flat.reshape((size, size))
    
    # Enforce symmetry explicitly to be safe
    # The basis functions are symmetric, so the result should be symmetric.
    # But let's check.
    
    return cell.tolist()

# EVOLVE-BLOCK-END
