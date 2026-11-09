import numpy as np
import time
import itertools

def design(problem):
    start_time = time.time()
    deadline = 110.0  # Safety margin from 120s limit
    
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    eps_low = problem["eps_low"]
    
    # 1. Define the fundamental domain
    # The cell is 32x32.
    # Symmetries:
    # 1. cell[i][j] == cell[i][31-j] (Vertical reflection)
    # 2. cell[i][j] == cell[31-i][j] (Horizontal reflection)
    # 3. cell[i][j] == cell[j][i] (Transpose)
    # Fundamental domain: i in [0, 15], j in [i, 15]
    
    fd_indices = []
    for i in range(16):
        for j in range(i, 16):
            fd_indices.append((i, j))
            
    n_fd = len(fd_indices)
    
    # Helper to build the full 32x32 cell from the fundamental domain values
    # fd_values is a dict or list corresponding to fd_indices
    def build_cell(fd_values):
        # fd_values is a list of 0/1 of length n_fd
        # Create a 32x32 array of zeros
        cell = np.zeros((size, size), dtype=np.int8)
        
        # Map fd index to value
        # We need a way to look up value for (i, j) in the fundamental domain
        # Since fd_indices is small, we can create a lookup dict
        # But for speed in the inner loop, we might want to just assign directly.
        
        # Let's pre-compute the mapping from full cell indices to fd indices
        # Or just iterate over fd_indices and mirror.
        
        for k, (i, j) in enumerate(fd_indices):
            val = fd_values[k]
            # Symmetry 3: Transpose -> (j, i)
            # Symmetry 1: Vertical Reflection of (i, j) is (i, 31-j)
            # Symmetry 2: Horizontal Reflection of (i, j) is (31-i, j)
            
            # The 8 symmetric positions for (i, j) in the top-left quadrant (i<=15, j>=i):
            # 1. (i, j)
            # 2. (j, i)  [Transpose]
            # 3. (i, 31-j) [V-Refl]
            # 4. (31-j, i) [Transpose of V-Refl]
            # 5. (31-i, j) [H-Refl]
            # 6. (j, 31-i) [Transpose of H-Refl]
            # 7. (31-i, 31-j) [C2 Rotation]
            # 8. (31-j, 31-i) [Transpose of C2]
            
            positions = [
                (i, j),
                (j, i),
                (i, 31-j),
                (31-j, i),
                (31-i, j),
                (j, 31-i),
                (31-i, 31-j),
                (31-j, 31-i)
            ]
            
            for (r, c) in positions:
                cell[r, c] = val
                
        return cell

    # Fast Objective Function
    # Plane Wave Method
    # We use a reduced basis for speed.
    # N_waves = 11 (121 modes)
    # K-points: A few representative points in the irreducible BZ.
    
    N_w = 11
    # Generate plane wave vectors
    # G vectors: (m, n) * 2pi/L
    # We need |G| < cutoff
    # Cutoff usually 1.5 * k_max or similar.
    # For square lattice, k_max is at (pi, pi).
    # Let's use a cutoff radius of 2.0 (in units of 2pi/L)
    # m, n range from -N_w//2 to N_w//2
    
    g_vectors = []
    max_g = 6 # 121 waves approx: sum of (2n+1) for n where (n+0.5)^2 + ... 
    # Let's just generate a square grid of G vectors
    g_range = np.arange(-N_w//2, N_w//2 + 1)
    for m in g_range:
        for n in g_range:
            if m == 0 and n == 0:
                continue
            # Include origin? Yes, usually.
            g_vectors.append((m, n))
            
    # Wait, standard PW method includes G=0.
    g_vectors = [(m, n) for m in g_range for n in g_range]
    num_g = len(g_vectors)
    
    # K-points in irreducible BZ
    # Irreducible BZ for square lattice is the triangle (0,0) to (pi,0) to (pi,pi).
    # We sample a few points.
    k_points = [
        (0.0, 0.0),      # Gamma
        (np.pi, 0.0),    # X
        (np.pi, np.pi),  # M
        (0.5*np.pi, 0.0),
        (0.5*np.pi, 0.5*np.pi),
        (0.8*np.pi, 0.0),
        (0.0, 0.5*np.pi),
        (0.2*np.pi, 0.2*np.pi)
    ]
    
    # Pre-compute epsilon_G for the current cell?
    # Epsilon_G depends on the cell.
    # Epsilon_G = (1/A) * integral epsilon(r) exp(-i G.r) dr
    # For a pixelated cell, we can compute this via summing pixels.
    # eps(r) is eps_high if pixel is 1, eps_low if 0.
    # eps_G = (eps_high - eps_low) * (1/A) * sum_{pixels=1} exp(-i G.r_p) + eps_low * delta_{G,0}
    # Actually, simpler:
    # eps_G = (1/N_pix) * sum_{i,j} eps_{ij} exp(-i (G_x i + G_y j))
    # where i,j are pixel centers.
    
    # To speed up, we pre-compute the phase factors exp(-i (G_x i + G_y j))
    # for all G and all pixels.
    # This is a (num_g, size*size) matrix.
    # num_g ~ 121, size*size = 1024. 121*1024 ~ 124k complex numbers. Feasible.
    
    pixel_coords = np.array([(i, j) for i in range(size) for j in range(size)])
    # G vectors in radians per pixel?
    # G = (m, n) * 2pi / size
    g_mags = np.array(g_vectors)
    # g_mags shape (num_g, 2)
    
    # Phase matrix: (num_g, size*size)
    # Phase = -2pi/size * (m*i + n*j)
    # We can compute this on the fly or precompute.
    # Let's precompute the real and imaginary parts for each G.
    # Actually, we can just compute the sum using the cell array.
    # cell is 0/1.
    # sum_eps = sum(cell * exp(-i G.r))
    
    def compute_eps_g(cell_flat):
        # cell_flat is 1D array of length size*size, values 0 or 1
        # We want eps_G for all G
        # eps_G = eps_low + (eps_high - eps_low) * (1/size^2) * sum_{p: cell[p]=1} exp(-i G.r_p)
        
        # Precomputed phases:
        # phases[g_idx, p_idx] = exp(-i 2pi/size (m_g i_p + n_g j_p))
        # This is expensive to store? 121 * 1024 * 16 bytes = ~2MB. OK.
        
        # Let's compute phases once outside? No, G depends on N_w which is fixed.
        # But we need to map pixel index to (i, j).
        
        # Optimization:
        # Instead of summing over all pixels, we can use FFT?
        # The sum is essentially a DFT of the pixel map.
        # eps_G is the DFT of the dielectric map.
        # We can use np.fft.fftn on the 2D cell array to get the Fourier coefficients.
        # The indices in FFT output correspond to G vectors.
        # FFT frequency bins: 0, 1, ..., size/2, -size/2+1, ...
        # We need to map our G vectors (m, n) to FFT bins.
        
        # Let's use FFT.
        # cell_2d is size x size.
        # fft_cell = np.fft.fftn(cell_2d)
        # The coefficient for G=(m, n) is at index (m % size, n % size).
        # Note: np.fft.fftn uses ordering 0, 1, ..., N-1.
        # Negative frequencies wrap around.
        
        # So for each G vector (m, n), the FFT index is (m % size, n % size).
        
        return None # Placeholder, will implement in loop

    # Let's implement the fast evaluation
    def evaluate_gap(cell_2d):
        # cell_2d is 32x32 array of 0/1
        # Compute Fourier coefficients
        # eps(r) = eps_low + (eps_high - eps_low) * cell(r)
        # eps_G = (1/N) sum eps(r) exp(-i G.r)
        # Using FFT:
        # Let C be the pixel map (0/1).
        # FFT(C) gives sum C(r) exp(-i 2pi k r / N).
        # This matches our G definition if G = 2pi k / N.
        
        fft_cell = np.fft.fftn(cell_2d)
        
        # Construct the Hamiltonian matrix H
        # H_{G, G'} = (G+K) dot (G'+K) * eps_{G-G'}
        # Where eps_{G-G'} is the Fourier coefficient of 1/eps?
        # Wait, for TM mode, the equation is:
        # (G+K)^2 / eps_{G-G'} * E_{G'} = omega^2 E_G
        # Or:
        # sum_{G'} [ (G+K) dot (G'+K) / eps_{G-G'} ] E_{G'} = omega^2 E_G
        # 
        # Note: The dielectric function in the denominator is 1/eps(r).
        # Let's call chi(r) = 1/eps(r).
        # chi_G = (1/N) sum chi(r) exp(-i G.r).
        # 
        # We need chi_G, not eps_G.
        # chi(r) = 1/eps_low if cell=0, 1/eps_high if cell=1.
        # chi(r) = chi_low + (chi_high - chi_low) * cell(r).
        
        chi_low = 1.0 / eps_low
        chi_high = 1.0 / eps_high
        
        # Create chi map
        # chi_map = np.where(cell_2d == 1, chi_high, chi_low)
        # fft_chi = np.fft.fftn(chi_map)
        
        # To avoid creating a new array every time, we can compute:
        # chi_map = chi_low * ones + (chi_high - chi_low) * cell_2d
        # fft_chi = chi_low * fft(ones) + (chi_high - chi_low) * fft(cell_2d)
        # fft(ones) is zero everywhere except DC (0,0) which is N^2.
        # So fft_chi[0,0] = N^2 * chi_low + (chi_high - chi_low) * sum(cell)
        # fft_chi[G!=0] = (chi_high - chi_low) * fft_cell[G]
        
        fft_chi = np.zeros_like(fft_cell, dtype=complex)
        delta_chi = chi_high - chi_low
        fft_chi[:] = delta_chi * fft_cell
        # Add DC term
        fft_chi[0, 0] += chi_low * size * size
        
        # Now we have chi_G for all G in the FFT grid.
        # We need to construct the matrix for each K point.
        # We use a subset of G vectors (our g_vectors list).
        # We need to map g_vectors to FFT indices.
        
        # Pre-compute mapping from g_vectors to fft indices
        # g_vectors are (m, n) in range [-N_w//2, N_w//2]
        # FFT indices are m % size, n % size
        
        g_fft_indices = [(m % size, n % size) for m, n in g_vectors]
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            # Build H matrix
            # H[i, j] = (G_i + K) dot (G_j + K) * chi_{G_i - G_j}
            
            # G_i + K vector
            # G_i = (m_i, n_i) * 2pi / size
            # K = (kx, ky)
            
            # For efficiency, we can pre-compute the (G+K) vectors
            # But K changes.
            
            H = np.zeros((num_g, num_g), dtype=complex)
            
            for i in range(num_g):
                mi, ni = g_vectors[i]
                Gx_i = 2.0 * np.pi * mi / size
                Gy_i = 2.0 * np.pi * ni / size
                Kx_i = Gx_i + kx
                Ky_i = Gy_i + ky
                
                for j in range(num_g):
                    mj, nj = g_vectors[j]
                    Gx_j = 2.0 * np.pi * mj / size
                    Gy_j = 2.0 * np.pi * nj / size
                    Kx_j = Gx_j + kx
                    Ky_j = Gy_j + ky
                    
                    # Chi index: G_i - G_j = (mi-mj, ni-nj)
                    # Map to FFT index
                    chi_idx_i = (mi - mj) % size
                    chi_idx_j = (ni - nj) % size
                    chi_val = fft_chi[chi_idx_i, chi_idx_j]
                    
                    dot_prod = Kx_i * Kx_j + Ky_i * Ky_j
                    H[i, j] = dot_prod * chi_val
            
            # Solve eigenvalue problem
            # H is Hermitian?
            # H[i, j] = (G_i+K) dot (G_j+K) chi_{G_i-G_j}
            # H[j, i] = (G_j+K) dot (G_i+K) chi_{G_j-G_i}
            # chi_{-G} = conj(chi_G)
            # dot is symmetric.
            # So H is Hermitian.
            
            try:
                eigvals = np.linalg.eigvalsh(H)
                # eigvals are omega^2
                # We want positive eigenvalues
                eigvals = eigvals[eigvals > 0]
                eigvals = np.sort(eigvals)
                
                if len(eigvals) >= 5:
                    omega_4 = np.sqrt(eigvals[3])
                    omega_5 = np.sqrt(eigvals[4])
                    
                    max_omega_4 = max(max_omega_4, omega_4)
                    min_omega_5 = min(min_omega_5, omega_5)
            except Exception:
                pass
                
        if min_omega_5 == float('inf'):
            return -1.0
            
        gap = 2 * (min_omega_5 - max_omega_4)
        midgap = min_omega_5 + max_omega_4
        ratio = gap / midgap if midgap > 0 else 0
        return ratio

    # Search Loop
    best_score = -1.0
    best_fd = None
    
    # Initial guesses
    # 1. Centered Rod (Seed)
    # 2. Square
    # 3. Cross
    # 4. Random
    
    def fd_from_cell(cell_2d):
        fd_vals = []
        for i, j in fd_indices:
            fd_vals.append(cell_2d[i, j])
        return fd_vals

    # Generate seed rod fd
    seed_fd = [0] * n_fd
    for k, (i, j) in enumerate(fd_indices):
        # Center is (15.5, 15.5)
        # Radius 0.2 * 32 = 6.4 pixels?
        # x = (i - 15.5), y = (j - 15.5)
        if (i - 15.5)**2 + (j - 15.5)**2 < (0.2 * size)**2:
            seed_fd[k] = 1
            
    initial_configs = [seed_fd]
    
    # Add a square
    sq_fd = [0] * n_fd
    for k, (i, j) in enumerate(fd_indices):
        if abs(i - 15.5) < 4 and abs(j - 15.5) < 4:
            sq_fd[k] = 1
    initial_configs.append(sq_fd)
    
    # Add a cross
    cr_fd = [0] * n_fd
    for k, (i, j) in enumerate(fd_indices):
        if abs(i - 15.5) < 2 or abs(j - 15.5) < 2:
            cr_fd[k] = 1
    initial_configs.append(cr_fd)
    
    # Random
    rd_fd = [1 if np.random.rand() > 0.5 else 0 for _ in range(n_fd)]
    initial_configs.append(rd_fd)
    
    for init_fd in initial_configs:
        current_fd = init_fd[:]
        current_score = evaluate_gap(build_cell(current_fd))
        
        if current_score > best_score:
            best_score = current_score
            best_fd = current_fd[:]
            
        # Hill Climbing
        for step in range(500): # Max steps per restart
            if time.time() > deadline:
                break
                
            # Pick a random index to flip
            idx = np.random.randint(n_fd)
            
            # Flip
            current_fd[idx] = 1 - current_fd[idx]
            
            new_cell = build_cell(current_fd)
            new_score = evaluate_gap(new_cell)
            
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_fd = current_fd[:]
            else:
                # Revert
                current_fd[idx] = 1 - current_fd[idx]
                
            if time.time() > deadline:
                break

    # Finalize
    if best_fd is None:
        # Fallback to seed
        return build_cell(seed_fd).tolist()
        
    return build_cell(best_fd).tolist()
