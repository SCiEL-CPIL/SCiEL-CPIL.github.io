import numpy as np
import time
import itertools

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell to maximize the gap between TM bands 4 and 5.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # 1. Generate G vectors for 289 plane waves
    # 289 = 17^2. We use a circular cutoff or square cutoff?
    # "289-plane-wave Galerkin method" usually implies a specific set.
    # 17x17 grid of G vectors is 289.
    # G_x, G_y from -8 to 8.
    g_max = 8
    g_vals = np.arange(-g_max, g_max + 1)
    G_vecs = np.array(list(itertools.product(g_vals, g_vals)))
    num_pw = len(G_vecs) # 289
    
    # 2. Precompute epsilon_G for a given cell
    def get_epsilon_fourier(cell):
        # cell is 32x32 list of lists
        eps_grid = np.array(cell)
        # eps = eps_low + (eps_high - eps_low) * pixel
        eps_vals = eps_low + (eps_high - eps_low) * eps_grid
        # Fourier transform
        # epsilon_G = 1/N^2 sum eps_ij exp(-i 2 pi (Gx i + Gy j) / N)
        # We can use FFT.
        # Note: FFT gives exp(-i 2 pi k n / N).
        # G is in units of 2 pi / a.
        # The index in FFT corresponds to frequency.
        # We need to map G_vecs to FFT indices.
        # FFT indices: 0, 1, ..., N-1.
        # Frequencies: 0, 1, ..., N/2, -N/2+1, ..., -1.
        # Our G_vecs are -8..8.
        # N=32.
        # FFT bin k corresponds to freq k if k < N/2, k-N if k >= N/2.
        
        fft_eps = np.fft.fft2(eps_vals)
        eps_G = np.zeros(num_pw, dtype=complex)
        for idx, (gx, gy) in enumerate(G_vecs):
            # Map gx, gy to FFT index
            ix = gx % size
            iy = gy % size
            eps_G[idx] = fft_eps[ix, iy] / (size * size)
        return eps_G

    # 3. Solve bands for a given cell and k-points
    # k-points in irreducible BZ
    # Square BZ: kx, ky in [0, pi]
    # Irreducible: kx >= ky >= 0
    # 30 points covering boundary and interior.
    # Boundary: X-M (kx=pi, ky=0->pi/2?), M-G (kx=ky, pi/2->0), X-G (kx=pi, ky=0).
    # Let's generate a set of 30 points.
    
    # Points along M-X (kx=pi, ky from 0 to pi/2)
    # Points along G-M (kx=ky, from 0 to pi/2)
    # Points along G-X (kx=pi, ky=0) -- wait, G is (0,0), X is (pi, 0).
    
    k_points = []
    # G-M line: k = t * (pi/2, pi/2), t in [0, 1]
    for t in np.linspace(0, 1, 10):
        k_points.append((t * np.pi/2, t * np.pi/2))
    # M-X line: k = (pi/2 + t*(pi/2), pi/2 - t*(pi/2))? No.
    # M is (pi/2, pi/2). X is (pi, 0).
    # Vector M->X: (pi/2, -pi/2).
    for t in np.linspace(0, 1, 10):
        k_points.append((np.pi/2 + t*np.pi/2, np.pi/2 - t*np.pi/2))
    # X-G line: k = (pi - t*pi, 0)
    for t in np.linspace(0, 1, 10):
        k_points.append((pi - t*pi, 0))
        
    # 30 points total.
    
    def compute_bands(cell, eps_G):
        bands = []
        for kx, ky in k_points:
            k = np.array([kx, ky])
            # Construct matrix M
            # M_GG' = 1/|k+G|^2 * epsilon_{G-G'}
            # We need epsilon_{G-G'} for all pairs.
            # This is a convolution.
            # We can precompute the epsilon_G array and use it.
            # M is 289x289.
            
            # For efficiency, we can compute the matrix in a loop or vectorized.
            # k_plus_G = k + G_vecs (broadcast)
            kpG = k + G_vecs.T # shape (289, 2)
            kpG_sq = np.sum(kpG**2, axis=1) # shape (289,)
            
            # Matrix E: E[i,j] = epsilon_{G_i - G_j}
            # G_i - G_j is a vector.
            # We can compute this by iterating over G_i and finding the index of G_i - G_j.
            # Since G_vecs is a grid, we can map (gx, gy) to index.
            # idx = (gx + g_max) * (2*g_max+1) + (gy + g_max)
            
            # Precompute index map
            # Actually, just build the matrix directly.
            M = np.zeros((num_pw, num_pw), dtype=complex)
            for i in range(num_pw):
                gi = G_vecs[i]
                for j in range(num_pw):
                    gj = G_vecs[j]
                    diff = gi - gj
                    # Find index of diff in G_vecs
                    # Since diff is also in range -16..16, it might not be in G_vecs (-8..8)?
                    # Wait. G_vecs are -8..8.
                    # If G_i = 8, G_j = -8, diff = 16.
                    # 16 is NOT in G_vecs.
                    # The plane wave expansion requires epsilon_G for all G in the set.
                    # If the set is not closed under subtraction, we can't form the matrix directly?
                    # No, the Fourier series is infinite. We truncate.
                    # The term epsilon_{G-G'} is only non-zero if G-G' is within the support?
                    # No, epsilon_G is the Fourier coefficient.
                    # If we truncate the basis to 289 waves, we must also truncate the epsilon_G to the same range?
                    # Usually, yes. We assume epsilon_G = 0 for |G| > 8.
                    # So if diff is outside [-8, 8], epsilon_diff = 0.
                    
                    if abs(diff[0]) <= g_max and abs(diff[1]) <= g_max:
                         # Find index
                         # Map (dx, dy) to index
                         # dx = diff[0], dy = diff[1]
                         # index = (dx + g_max) * 17 + (dy + g_max)
                         idx_diff = (int(diff[0]) + g_max) * 17 + (int(diff[1]) + g_max)
                         eps_val = eps_G[idx_diff]
                         M[i, j] = eps_val / kpG_sq[i]
                    else:
                         M[i, j] = 0.0
                         
            # Solve eigenvalues
            # M is not Hermitian.
            eigvals = np.linalg.eigvals(M)
            # eigvals are mu = c^2 / omega^2
            # omega = c / sqrt(mu)
            # c = 1 (normalized)
            # Filter real positive
            eigvals = eigvals[eigvals.imag < 1e-9]
            eigvals = eigvals.real
            eigvals = eigvals[eigvals > 1e-9]
            omegas = 1.0 / np.sqrt(eigvals)
            omegas.sort()
            bands.append(omegas)
            
        return bands

    def score_cell(cell):
        eps_G = get_epsilon_fourier(cell)
        bands = compute_bands(cell, eps_G)
        
        # Find min omega_5 and max omega_4
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for b in bands:
            if len(b) >= 5:
                min_omega_5 = min(min_omega_5, b[4]) # 5th band is index 4
                max_omega_4 = max(max_omega_4, b[3]) # 4th band is index 3
        
        if min_omega_5 == float('inf') or max_omega_4 == 0.0:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0: return 0.0
        return gap / midgap

    # 4. Optimization
    # Initial guess: Inverted rod (Air hole in center)
    # 1 = Dielectric, 0 = Air
    # Seed was Dielectric rod.
    # Let's start with Air hole (mostly 1s, center 0s).
    
    def make_initial_cell():
        cell = [[1 for _ in range(size)] for _ in range(size)]
        # Radius 0.2
        r2 = 0.2 * 0.2
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x*x + y*y < r2:
                    cell[i][j] = 0
        return cell

    best_cell = make_initial_cell()
    best_score = score_cell(best_cell)
    
    # Symmetry helper
    def get_symmetric_indices(i, j):
        # D4 symmetry
        # (i, j)
        # (j, size-1-i)
        # (size-1-i, j)
        # (size-1-j, size-1-i)
        # (size-1-i, size-1-j)
        # (size-1-j, i)
        # (i, size-1-j)
        # (j, i)
        inds = set()
        inds.add((i, j))
        inds.add((j, size-1-i))
        inds.add((size-1-i, j))
        inds.add((size-1-j, size-1-i))
        inds.add((size-1-i, size-1-j))
        inds.add((size-1-j, i))
        inds.add((i, size-1-j))
        inds.add((j, i))
        return list(inds)

    # Fundamental domain: i < size/2, j < size/2, i <= j?
    # Just iterate over all i, j in 0..15, 0..15 and check if it's the "primary" representative.
    # Or just iterate over all and skip if not primary.
    
    current_cell = [row[:] for row in best_cell]
    current_score = best_score
    
    # Simple hill climbing
    # Randomly flip a symmetric set of pixels
    np.random.seed(42)
    
    while time.time() - start_time < deadline:
        # Pick random pixel in upper-left quadrant
        i = np.random.randint(0, size // 2)
        j = np.random.randint(0, size // 2)
        
        # Get symmetric indices
        inds = get_symmetric_indices(i, j)
        
        # Flip
        new_cell = [row[:] for row in current_cell]
        for ii, jj in inds:
            new_cell[ii][jj] = 1 - new_cell[ii][jj]
        
        # Score
        new_score = score_cell(new_cell)
        
        if new_score > current_score:
            current_cell = new_cell
            current_score = new_score
            if current_score > best_score:
                best_cell = current_cell
                best_score = current_score
        else:
            # Keep current
            pass
            
    return best_cell

