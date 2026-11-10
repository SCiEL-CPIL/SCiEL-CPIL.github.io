import numpy as np
import time
import sys

def design(problem):
    """
    Designs a 32x32 pixel unit cell for a square-lattice photonic crystal.
    Uses air holes in a dielectric background (epsilon=12) to maximize the 
    TM band gap between bands 4 and 5.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for overhead
    
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    eps_low = problem["eps_low"]
    
    # Plane-wave parameters
    # 289 plane waves -> 17x17 grid
    n_pw = 17
    # G-vectors: from -n//2 to n//2
    g_max = n_pw // 2
    g_vals = np.arange(-g_max, g_max + 1)
    G_vectors = np.array([(gx, gy) for gx in g_vals for gy in g_vals])
    n_G = len(G_vectors)
    
    # Precompute G-vector magnitudes squared
    G2 = np.sum(G_vectors**2, axis=1)
    
    # Irreducible Brillouin Zone points
    # Square lattice BZ is a square from -pi to pi in kx, ky (in units of 2pi/a)
    # Irreducible sector: kx >= ky >= 0
    # Key points:
    # Gamma: (0, 0)
    # X: (0.5, 0)  [midpoint of edge]
    # M: (0.5, 0.5) [corner]
    # We sample points along the Gamma-X, X-M, and Gamma-M paths.
    k_points = []
    # Gamma to X
    for t in np.linspace(0, 0.5, 10):
        k_points.append((t, 0))
    # X to M
    for t in np.linspace(0.5, 0.5, 10): # x is 0.5, y goes 0 to 0.5
        k_points.append((0.5, t/2)) 
        # Wait, X is (0.5, 0), M is (0.5, 0.5). So y goes 0 to 0.5.
    # Gamma to M
    for t in np.linspace(0, 0.5, 10):
        k_points.append((t, t))
        
    # Deduplicate and ensure symmetry
    k_points = list(set(k_points))
    # Filter to irreducible sector kx >= ky >= 0
    k_points = [k for k in k_points if k[0] >= k[1] - 1e-9 and k[1] >= -1e-9]
    
    def get_fourier_coeffs(cell):
        """
        Compute Fourier coefficients of the dielectric function.
        cell is a 2D numpy array of 0s and 1s.
        0 = air (eps_low), 1 = dielectric (eps_high).
        """
        eps = np.zeros((size, size))
        eps[cell == 1] = eps_high
        eps[cell == 0] = eps_low
        
        # Compute FFT. The pixel grid represents the unit cell.
        # The Fourier coefficient G is the integral over the unit cell.
        # Using FFT: eps_G = (1/N^2) * sum eps_ij exp(-i G . r_ij)
        # Numpy FFT computes sum eps_ij exp(-i 2pi (u i + v j) / N)
        # Our G vectors are in units of 2pi/a.
        # r_ij = (i/N, j/N) * a.
        # G . r = 2pi (Gx i/N + Gy j/N).
        # So we map Gx, Gy to FFT indices.
        # Gx = m * 2pi/a. In FFT, frequency index u corresponds to 2pi u / N.
        # So m = u * N / a? No, G is in units of 2pi/a.
        # The FFT basis is exp(-i 2pi (u i + v j) / N).
        # We want exp(-i 2pi (Gx i + Gy j)).
        # So Gx = u/N * N? No.
        # Let's use the direct sum for small N or careful mapping.
        # N=32. G ranges from -8 to 8.
        # FFT indices 0..31.
        # Frequency u=0 -> 0. u=1 -> 1/32. 
        # G=1 corresponds to 2pi/a. 
        # The period is 1 pixel unit? No, the cell is 1 unit.
        # Pixel i is at x_i = (i+0.5)/size.
        # This is a discrete approximation.
        # Let's just use the discrete sum for accuracy, it's fast enough for 32x32.
        # Or use FFT with proper scaling.
        # eps_G = (1/size^2) * sum_{i,j} eps_{i,j} exp(-i 2pi (Gx (i+0.5)/size + Gy (j+0.5)/size))
        # This is a DFT with a half-pixel shift.
        
        # For speed, we can precompute the phase factors.
        # But since we are optimizing, we need to do this many times.
        # Let's use FFT.
        # eps_FFT = fft2(eps)
        # The FFT gives sum eps_ij exp(-i 2pi (u i + v j) / N).
        # We need sum eps_ij exp(-i 2pi (Gx (i+0.5)/N + Gy (j+0.5)/N))
        # = exp(-i 2pi (Gx/N + Gy/N) * 0.5) * sum eps_ij exp(-i 2pi (Gx i + Gy j) / N)
        # So we need to map Gx, Gy to integer FFT indices u, v such that u/N = Gx?
        # No, Gx is an integer. The FFT frequencies are u/N.
        # We want the coefficient for integer G.
        # The FFT index u corresponds to frequency u/N.
        # If we set u = Gx, then frequency is Gx/N.
        # This is not Gx.
        # The FFT of a periodic function sampled at N points gives the Fourier coefficients
        # for frequencies k = 2pi m / N.
        # Here our function is defined on the unit cell [0,1].
        # The Fourier basis is exp(i 2pi (m x + n y)).
        # So the FFT of the N x N grid gives the coefficients for m, n directly?
        # Yes, if the grid is uniform over [0,1].
        # x_i = i/N.
        # Our grid is x_i = (i+0.5)/N.
        # So there is a phase shift.
        
        # Let's use the direct sum for the specific G vectors we need.
        # n_G is 289. Size is 32. 289 * 32 * 32 ~ 300k ops. Very fast.
        i_coords = np.arange(size)[:, None] + 0.5
        j_coords = np.arange(size)[None, :] + 0.5
        
        # eps is size x size
        # We need eps_G for each G in G_vectors
        # eps_G = (1/size^2) * sum_{i,j} eps[i,j] exp(-i (Gx * 2pi * i/size + Gy * 2pi * j/size))
        # Note: the 0.5 offset adds a phase factor exp(-i pi (Gx + Gy)).
        
        # Vectorized calculation
        # Gx: (n_G, 1), Gy: (n_G, 1)
        # i: (1, size), j: (1, size)
        # Phase = -i 2pi (Gx * i / size + Gy * j / size)
        
        phase_x = np.exp(-1j * 2 * np.pi * (G_vectors[:, 0:1] * i_coords.T / size)) # (n_G, size)
        phase_y = np.exp(-1j * 2 * np.pi * (G_vectors[:, 1:2] * j_coords.T / size)) # (n_G, size)
        
        # Sum over i, j
        # Sum_i (eps[i,:] * phase_x[:,i]) -> (n_G, size)
        # Then sum over j
        # Actually, we can do:
        # S = sum_i sum_j eps[i,j] phase_x[:,i] phase_y[:,j]
        # S = (phase_x @ eps @ phase_y.T)
        
        S = phase_x @ eps @ phase_y.T
        eps_G = S / (size * size)
        
        # Apply half-pixel shift phase
        # exp(-i pi (Gx + Gy))
        phase_shift = np.exp(-1j * np.pi * (G_vectors[:, 0] + G_vectors[:, 1]))
        eps_G = eps_G * phase_shift
        
        return eps_G

    def solve_bands(eps_G):
        """
        Solve the eigenvalue problem for a given set of k-points.
        Returns the band frequencies (omega).
        """
        bands = []
        for k in k_points:
            kx, ky = k
            # K vector
            K = np.array([kx, ky]) * 2 * np.pi
            
            # Construct the matrix A
            # (K+G)^2 A_{K+G} = omega^2 sum_{G'} eps_{G-G'} A_{K+G'}
            # This is a generalized eigenvalue problem, but we can transform it.
            # Let B_{K+G} = sqrt(eps_{00}) A_{K+G} ? No.
            # Standard form:
            # sum_{G'} [ (K+G)^2 delta_{G,G'} - omega^2 eps_{G-G'} ] A_{K+G'} = 0
            # This is not a standard Hermitian eigenvalue problem because of the (K+G)^2 term.
            # We can rewrite it as:
            # sum_{G'} eps_{G-G'}^{-1} (K+G)^2 delta_{G,G'} A_{K+G'} = omega^2 A_{K+G}
            # Wait, the dielectric function is periodic.
            # The equation is:
            # -1/eps(r) nabla x (nabla x H) = (omega/c)^2 H
            # In plane wave:
            # (K+G)^2 A_{K+G} = (omega/c)^2 sum_{G'} eps_{G-G'} A_{K+G'}
            # We need to invert the eps matrix?
            # Actually, for TM, the matrix is:
            # M_{G,G'} = (K+G)^2 delta_{G,G'} - omega^2 eps_{G-G'}
            # This is not linear in omega^2 in a simple way if we just set up the matrix.
            # The standard trick is to define B_{K+G} = A_{K+G} / sqrt(eps_{00})? No.
            # Let's use the formulation:
            # sum_{G'} [ (K+G)^2 delta_{G,G'} ] A_{K+G'} = omega^2 sum_{G'} eps_{G-G'} A_{K+G'}
            # This is a generalized eigenvalue problem: M A = omega^2 N A
            # Where M is diagonal with (K+G)^2, and N is the eps matrix.
            # N is Hermitian and positive definite.
            # We can solve this using scipy.linalg.eigh.
            
            M = np.diag((K[0] + G_vectors[:, 0])**2 + (K[1] + G_vectors[:, 1])**2)
            N = np.zeros((n_G, n_G), dtype=complex)
            for i in range(n_G):
                for j in range(n_G):
                    G_diff = G_vectors[i] - G_vectors[j]
                    # Find index of G_diff in G_vectors
                    # Since G_vectors is a grid, we can map it.
                    # idx = (Gx + g_max) * (n_pw) + (Gy + g_max)
                    idx_gx = int(G_diff[0]) + g_max
                    idx_gy = int(G_diff[1]) + g_max
                    if 0 <= idx_gx < n_pw and 0 <= idx_gy < n_pw:
                        idx = idx_gx * n_pw + idx_gy
                        N[i, j] = eps_G[idx]
            
            # Solve M A = omega^2 N A
            # This is equivalent to N^{-1} M A = omega^2 A
            # But N is dense and singular? No, eps is positive.
            # Using eigh for generalized problem:
            try:
                vals, vecs = np.linalg.eigh(M, N)
                # vals are omega^2
                # Sort them
                order = np.argsort(vals)
                vals = vals[order]
                bands.append(np.sqrt(vals))
            except np.linalg.LinAlgError:
                # Fallback to standard if N is ill-conditioned
                N_inv = np.linalg.inv(N)
                vals, vecs = np.linalg.eig(N_inv @ M)
                vals = np.real(vals)
                order = np.argsort(vals)
                vals = vals[order]
                bands.append(np.sqrt(vals))
        
        return bands

    def evaluate_cell(cell):
        """
        Compute the gap ratio for a given cell.
        """
        eps_G = get_fourier_coeffs(np.array(cell))
        bands = solve_bands(eps_G)
        
        # Find min of band 5 (index 4) and max of band 4 (index 3)
        # bands is a list of arrays, one for each k-point
        min_omega_5 = float('inf')
        max_omega_4 = -float('inf')
        
        for b in bands:
            if len(b) >= 5:
                min_omega_5 = min(min_omega_5, b[4])
                max_omega_4 = max(max_omega_4, b[3])
        
        if min_omega_5 == float('inf') or max_omega_4 == -float('inf'):
            return 0.0
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        ratio = gap / midgap
        return ratio

    def make_cell(hole_size, shape_type='square'):
        """
        Generate a symmetric cell.
        hole_size: fraction of the cell size (0 to 1).
        shape_type: 'square' or 'circle'
        """
        cell = np.ones((size, size), dtype=int)
        center = size / 2.0
        half_size = hole_size * size / 2.0
        
        # Define the hole in the top-left quadrant and mirror
        # Hole is centered at (center, center)
        # In the top-left quadrant (i < center, j < center), the hole is a square
        # from center - half_size to center + half_size?
        # No, the hole is centered.
        # So in the full cell, the hole is |i - center| < half_size and |j - center| < half_size.
        
        for i in range(size):
            for j in range(size):
                di = abs(i + 0.5 - center)
                dj = abs(j + 0.5 - center)
                if shape_type == 'square':
                    if di < half_size and dj < half_size:
                        cell[i, j] = 0
                elif shape_type == 'circle':
                    if di**2 + dj**2 < half_size**2:
                        cell[i, j] = 0
        
        return cell.tolist()

    # Initial guess
    best_cell = make_cell(0.6, 'square')
    best_score = evaluate_cell(best_cell)
    
    # Hill Climbing
    # Parameters: hole_size, shape_type
    # We will try to optimize hole_size.
    # We can also try 'circle' vs 'square'.
    
    current_size = 0.6
    current_shape = 'square'
    current_score = best_score
    
    # Coarse search
    step = 0.05
    for _ in range(50): # Max 50 iterations
        if time.time() > deadline:
            break
        
        # Try perturbing hole_size
        new_size = current_size + np.random.uniform(-step, step)
        if 0.3 < new_size < 0.8:
            new_cell = make_cell(new_size, current_shape)
            new_score = evaluate_cell(new_cell)
            if new_score > current_score:
                current_score = new_score
                current_size = new_size
                best_cell = new_cell
                best_score = new_score
                step *= 0.9 # Shrink step
            else:
                step = max(step * 1.1, 0.01)
        
        # Try changing shape
        if current_shape == 'square':
            new_shape = 'circle'
        else:
            new_shape = 'square'
        
        new_cell = make_cell(current_size, new_shape)
        new_score = evaluate_cell(new_cell)
        if new_score > current_score:
            current_score = new_score
            current_shape = new_shape
            best_cell = new_cell
            best_score = new_score
    
    return best_cell

# EVOLVE-BLOCK-START
# The design function is defined above
# EVOLVE-BLOCK-END
