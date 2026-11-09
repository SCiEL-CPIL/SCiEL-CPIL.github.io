import numpy as np
import time
import math

def design(problem):
    start_time = time.time()
    deadline = start_time + 220.0 # Safety margin
    
    size = problem['pixels'] # 32
    eps_low = problem['eps_low'] # 1
    eps_high = problem['eps_high'] # 12
    
    # 1. Define Symmetry Extension
    # Fundamental domain: 0 <= i < size, 0 <= j <= i
    # We will store the mask for the fundamental domain and extend it.
    
    def get_full_cell(fund_mask):
        # fund_mask is a 2D array of size x size, but only upper triangle valid?
        # Easier: store the full 32x32 but enforce symmetry when modifying.
        # For evaluation, we just need the full 32x32.
        # Let's represent the state as the full 32x32 binary array, 
        # but only modify pixels in the fundamental domain and mirror them.
        pass

    # 2. Band Gap Calculator
    def calc_gap(cell, n_pw=15, k_points=None):
        if k_points is None:
            # Irreducible BZ points: Gamma, X, M, and midpoints
            k_points = [
                (0.0, 0.0), # Gamma
                (0.5, 0.0), # X
                (0.5, 0.5), # M
                (0.25, 0.0), # Mid Gamma-X
                (0.5, 0.25), # Mid X-M
                (0.25, 0.25) # Center of BZ square? No, M is (0.5,0.5). 
                             # (0.25, 0.25) is inside.
            ]
        
        # Fourier coefficients of epsilon^-1
        # cell is 32x32, 0 or 1.
        # eps_inv = 1/12 for 1, 1/1 for 0.
        eps_inv = np.where(cell == 1, 1.0/eps_high, 1.0/eps_low)
        
        # Compute Fourier coefficients up to n_pw
        # Use FFT for speed
        # Pad to 2*n_pw x 2*n_pw? Or just use np.fft.fftn on 32x32 and slice.
        # The grid is 32x32. The plane waves are integer frequencies.
        # We need coefficients for m, n in range(-n_pw, n_pw).
        # FFT of 32x32 gives frequencies 0..31.
        # Negative frequencies are at 32-k.
        
        coeffs = np.fft.fftn(eps_inv)
        # Normalize by 32*32
        coeffs /= (size * size)
        
        # We need to map indices 0..31 to -15..16 (for n_pw=15, we want -15..14 or similar)
        # Actually, for a centered grid, we want frequencies -N/2 to N/2-1.
        # np.fft.fftfreq(32) gives [0, 1, ..., 15, -16, ..., -1]
        # Let's define the list of k indices we want.
        k_indices = list(range(-n_pw, n_pw))
        
        # Build the matrix for each k_point
        min_omega_5 = float('inf')
        max_omega_4 = -float('inf')
        
        for kx, ky in k_points:
            # k is in units of 2*pi/a. 
            # The plane wave vector is G + k.
            # G = (2*pi*m/a, 2*pi*n/a).
            # The matrix element A_{mn} = delta_{mn} / ( (G_m + k) dot (G_n + k) ) * eps_inv_{m-n}
            # Wait, the standard PWE equation is:
            # sum_n ( (k + G_m) dot (k + G_n) ) * eps_inv_{m-n} * c_n = omega^2 c_m
            # Or: sum_n eps_inv_{m-n} * c_n / (k+G_m) dot (k+G_n) ... no.
            
            # Let's use the standard form:
            # H_{mn} = (k + G_m) dot (k + G_n) * eps_inv_{m-n}
            # We solve H c = omega^2 c.
            
            # Get the relevant Fourier coefficients of eps_inv
            # eps_inv_{m-n} is the coefficient of frequency (m-n).
            
            # Pre-compute the list of G vectors
            G_vectors = []
            for m in k_indices:
                for n in k_indices:
                    G_vectors.append((m, n))
                    
            num_pw = len(G_vectors)
            
            # Build H matrix
            H = np.zeros((num_pw, num_pw), dtype=complex)
            
            # Map (m,n) to index in G_vectors
            # Since k_indices is symmetric, we can just iterate.
            
            for i, (mi, ni) in enumerate(G_vectors):
                for j, (mj, nj) in enumerate(G_vectors):
                    # k + G_i
                    kxi = kx + mi
                    ky_i = ky + ni
                    kxj = kx + mj
                    ky_j = ky + nj
                    
                    dot_prod = kxi * kxj + ky_i * ky_j
                    
                    # Frequency difference
                    dm = mi - mj
                    dn = ni - nj
                    
                    # Get coeff
                    # We need coeff[dm, dn]
                    # From FFT:
                    # Index in fft array:
                    idx_m = dm % size
                    idx_n = dn % size
                    eps_coeff = coeffs[idx_m, idx_n]
                    
                    H[i, j] = dot_prod * eps_coeff
            
            # Solve eigenvalues
            # H is Hermitian
            vals = np.linalg.eigvalsh(H)
            
            # Sort eigenvalues
            # They are already sorted by eigvalsh
            # We need bands 4 and 5 (index 3 and 4)
            if len(vals) >= 5:
                w4 = np.sqrt(vals[3])
                w5 = np.sqrt(vals[4])
                
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
        
        if min_omega_5 < max_omega_4:
            return -1.0 # Overlap
        else:
            gap = min_omega_5 - max_omega_4
            mid = (min_omega_5 + max_omega_4) / 2.0
            if mid == 0:
                return 0.0
            return gap / mid

    # 3. Search Loop
    # State: 32x32 binary array
    # We maintain a list of "active" pixels in the fundamental domain to flip.
    
    def symmetrize(cell):
        # Ensure C4v symmetry
        # 1. Reverse rows
        # 2. Reverse cols
        # 3. Transpose
        # We take the average? No, it must be binary.
        # We define the cell by its fundamental domain (upper right triangle).
        # Let's just construct it from the fundamental domain.
        pass

    # Better State Representation:
    # Store a 32x32 array, but only allow modifications in the region 0<=i<32, 0<=j<=i.
    # When evaluating, we use the full array.
    # When flipping, we flip the pixel and its 7 symmetric counterparts.
    
    # Initialize with a few candidates
    candidates = []
    
    # Candidate 1: Rod
    cell_rod = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.2**2:
                cell_rod[i, j] = 1
    # Symmetrize (it's already symmetric)
    candidates.append(cell_rod.copy())
    
    # Candidate 2: Cross
    cell_cross = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) < 0.15 or abs(y) < 0.15:
                cell_cross[i, j] = 1
    candidates.append(cell_cross.copy())
    
    # Candidate 3: Random
    cell_rand = np.zeros((size, size), dtype=int)
    # Random on fundamental domain
    for i in range(size):
        for j in range(i, size):
            if np.random.rand() < 0.5:
                # Set symmetric points
                cell_rand[i, j] = 1
                cell_rand[size-1-i, j] = 1
                cell_rand[i, size-1-j] = 1
                cell_rand[size-1-i, size-1-j] = 1
                cell_rand[j, i] = 1
                cell_rand[j, size-1-i] = 1
                cell_rand[size-1-j, i] = 1
                cell_rand[size-1-j, size-1-i] = 1
    candidates.append(cell_rand.copy())
    
    best_score = -np.inf
    best_cell = None
    
    # Evaluate initial candidates
    for cell in candidates:
        score = calc_gap(cell)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
            
    # Local Search
    # Define fundamental domain pixels
    fund_pixels = []
    for i in range(size):
        for j in range(i, size):
            fund_pixels.append((i, j))
            
    # Shuffled list for random access
    np.random.shuffle(fund_pixels)
    
    # Simulated Annealing / Hill Climbing
    T = 0.1 # Temperature
    current_cell = best_cell.copy()
    current_score = best_score
    
    iteration = 0
    while time.time() < deadline:
        # Pick a random pixel in fundamental domain
        i, j = fund_pixels[iteration % len(fund_pixels)]
        
        # Flip it and its symmetric counterparts
        # Original value
        old_val = current_cell[i, j]
        new_val = 1 - old_val
        
        # Symmetric points:
        # (i, j)
        # (i, size-1-j)
        # (size-1-i, j)
        # (size-1-i, size-1-j)
        # (j, i)
        # (j, size-1-i)
        # (size-1-j, i)
        # (size-1-j, size-1-i)
        
        # Check if they are unique
        pts = [
            (i, j),
            (i, size-1-j),
            (size-1-i, j),
            (size-1-i, size-1-j),
            (j, i),
            (j, size-1-i),
            (size-1-j, i),
            (size-1-j, size-1-i)
        ]
        
        # Remove duplicates
        unique_pts = set(pts)
        
        # Apply flip
        for u_i, u_j in unique_pts:
            current_cell[u_i, u_j] = new_val
            
        # Evaluate
        new_score = calc_gap(current_cell)
        
        # Accept if better, or with probability exp((new-old)/T)
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_cell = current_cell.copy()
        else:
            # Revert
            for u_i, u_j in unique_pts:
                current_cell[u_i, u_j] = old_val
            # Accept worse with probability
            if np.random.rand() < np.exp((new_score - current_score) / T):
                # Apply flip again
                for u_i, u_j in unique_pts:
                    current_cell[u_i, u_j] = new_val
                current_score = new_score
            # Cooldown
            T *= 0.995
            if T < 0.01:
                T = 0.01
                
        iteration += 1
        
        # If we've gone through all pixels, reshuffle
        if iteration % len(fund_pixels) == 0:
            np.random.shuffle(fund_pixels)
            
    # Convert best_cell to list of lists
    result = best_cell.tolist()
    return result
