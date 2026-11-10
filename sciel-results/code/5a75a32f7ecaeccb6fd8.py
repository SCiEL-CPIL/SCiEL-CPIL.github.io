import numpy as np
import time

def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    Strategy:
    1. Represent the dielectric distribution using a Fourier series with C4v symmetry.
    2. Optimize the Fourier coefficients using a fast local search.
    3. Use a vectorized plane-wave method for rapid internal evaluation.
    """
    start_time = time.time()
    deadline = start_time + 220.0  # Safety margin before 230s kill
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # --- Internal Plane-Wave Evaluator Setup ---
    # Use a smaller basis for the internal search to maximize iterations.
    # The evaluator uses 289 (17x17). We use 11x11 (121) for speed.
    N_pw_internal = 5  # G from -5 to 5
    G_vals = np.arange(-N_pw_internal, N_pw_internal + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)
    
    # K points: 10 points in irreducible BZ for quick estimation
    k_points = []
    for t in np.linspace(0, 0.5, 4):
        k_points.append((t, 0))
        k_points.append((0.5, t))
        k_points.append((t, t))
    unique_k = []
    seen = set()
    for k in k_points:
        key = (round(k[0], 10), round(k[1], 10))
        if key not in seen:
            seen.add(key)
            unique_k.append(k)
    unique_k = np.array(unique_k)

    def evaluate_cell_fast(cell):
        """
        Computes the band gap using a vectorized plane-wave method.
        cell: 32x32 binary array
        """
        eps_grid = np.where(cell == 1, eps_high, eps_low)
        inv_eps_grid = 1.0 / eps_grid
        
        # FFT of 1/eps to get Fourier coefficients
        # Note: fft2 returns unnormalized. We divide by N^2 to get coefficients.
        fft_inv_eps = np.fft.fft2(inv_eps_grid) / (size * size)
        
        # Precompute G-G' differences indices
        # We need coeff_matrix[i, j] = fft_inv_eps[(G_i - G_j) % size]
        # G_vectors are integers.
        G_diff_x = G_vectors[:, 0][:, np.newaxis] - G_vectors[np.newaxis, :, 0]
        G_diff_y = G_vectors[:, 1][:, np.newaxis] - G_vectors[np.newaxis, :, 1]
        
        # Map to 0..size-1 for FFT indexing
        idx_x = np.mod(G_diff_x, size)
        idx_y = np.mod(G_diff_y, size)
        
        # Vectorized lookup of Fourier coefficients
        coeff_matrix = fft_inv_eps[idx_x, idx_y]
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        # Evaluate at all k points
        # k_plus_G shape: (num_k, num_pw, 2)
        k_plus_G = unique_k[:, np.newaxis, :] + G_vectors[np.newaxis, :, :]
        omega_sq = np.sum(k_plus_G ** 2, axis=2) # (num_k, num_pw)
        
        # Build eigenvalue problem for each k
        # M_k = omega_sq_k * coeff_matrix
        # We solve det(M - I) = 0 or eig(M) = 1
        # Actually, the PW method equation is:
        # sum_j (k+G_i)^2 * (1/eps)_{G_i-G_j} * c_j = omega^2 c_i
        # So we want eigenvalues of A = D_k * C where D_k is diag((k+G)^2)
        
        for i_k in range(len(unique_k)):
            D_k = omega_sq[i_k, :]
            M = coeff_matrix * D_k[:, np.newaxis]
            
            # Solve generalized eigenvalue problem or standard eigenvalue
            # Since M is symmetric (if coeff is real and symmetric, which it is for real eps),
            # we can use eigh for speed and stability.
            # coeff_matrix is real? fft of real signal is conjugate symmetric.
            # M is Hermitian.
            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.sort(eigvals)
            
            if len(eigvals) >= 5:
                omega_4 = np.sqrt(eigvals[3])
                omega_5 = np.sqrt(eigvals[4])
                
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap == 0:
            return -1.0
        ratio = 2 * gap / midgap
        return ratio

    def fourier_to_cell(coeffs):
        """
        Converts a set of Fourier coefficients to a 32x32 binary cell.
        coeffs: list of (Gx, Gy, amplitude, phase) for unique C4v orbits.
        """
        # Reconstruct the real field
        field = np.zeros((size, size))
        
        # Add DC component (average epsilon)
        # We fix the average fill fraction to 0.5 for simplicity, or optimize it?
        # Let's fix DC to 0.5 to ensure 50% dielectric.
        field += 0.5 * (eps_high + eps_low) # Actually, let's just build the modulation
        # Let's define field as the modulation around the mean.
        # field = mean + sum a_n cos(...)
        
        # To ensure C4v symmetry, we sum over orbits.
        # The input `coeffs` are for the fundamental domain.
        # We need to map them to all symmetric G vectors.
        
        # Simple approach: 
        # 1. Create a full complex FFT array of size 32x32 initialized to 0.
        # 2. Fill in the coefficients for the unique G vectors and their symmetric counterparts.
        # 3. Inverse FFT to get the spatial domain field.
        # 4. Threshold.
        
        fft_coeffs = np.zeros((size, size), dtype=complex)
        
        # DC term
        # We want the resulting binary mask to have ~50% ones.
        # The mean of the field should be (eps_high+eps_low)/2? No, we threshold to 0/1.
        # If we threshold at the mean, we get 50% ones.
        # Let's set DC = 0.5 (in normalized 0..1 space) and modulate around it.
        # Then threshold at 0.5.
        fft_coeffs[0, 0] = 0.5
        
        for gx, gy, amp, phase in coeffs:
            # Add this coefficient and all its C4v symmetric counterparts
            # The C4v group acts on G vectors:
            # (gx, gy) -> (gy, gx), (-gx, gy), etc.
            # We must be careful not to double count if G is on a symmetry axis.
            
            # Generate all 8 symmetric G vectors
            # (gx, gy), (gy, gx), (-gx, gy), (gy, -gx), (-gx, -gy), (-gy, -gx), (gx, -gy), (-gy, gx)
            # Note: FFT indexing wraps around.
            
            syms = [
                (gx, gy),
                (gy, gx),
                (-gx, gy),
                (gy, -gx),
                (-gx, -gy),
                (-gy, -gx),
                (gx, -gy),
                (-gy, gx)
            ]
            
            # Count how many times each unique G vector appears in the list
            # to handle axes correctly (e.g., gx=0 or gy=0 or gx=gy)
            unique_g = {}
            for sgx, sgy in syms:
                # Map to FFT indices (0..31)
                idx_x = sgx % size
                idx_y = sgy % size
                key = (idx_x, idx_y)
                if key in unique_g:
                    unique_g[key] += 1
                else:
                    unique_g[key] = 1
            
            # The coefficient in the FFT array for a real signal must satisfy H[-k] = H[k]*.
            # If we sum amplitudes * exp(i phase) for all symmetric Gs, we need to ensure consistency.
            # A simpler way: Define the real-space field as:
            # f(x,y) = DC + Sum_n amp_n * cos(2*pi*(gx_n*x + gy_n*y)/N + phase_n)
            # This is automatically real and symmetric if we pick the right G vectors.
            # Let's stick to the FFT construction but ensure Hermitian symmetry.
            
            # For a real field, F[k] + F[-k] = 2 * Re(F[k]) if we place them correctly?
            # No, F[-k] = F[k]*.
            # If we want a term A * cos(kx + phi) = A/2 * (e^{i(kx+phi)} + e^{-i(kx+phi)}).
            # So F[k] += A/2 * e^{i phi}
            # F[-k] += A/2 * e^{-i phi}
            
            # We iterate over the unique G vectors in the orbit.
            for (idx_x, idx_y), count in unique_g.items():
                # If count > 1, it's on a symmetry axis.
                # The amplitude contribution might need scaling?
                # Actually, if we just want to add the term, we should add the appropriate FFT component.
                
                # Let's define the contribution to the real field.
                # If we add c to F[idx] and c* to F[-idx], we get 2*Re(c*exp(i*k*r)).
                # We want the term to be amp * cos(...).
                # So c = amp/2 * exp(i*phase).
                
                c = (amp / 2.0) * np.exp(1j * phase)
                
                # Add to F[idx]
                fft_coeffs[idx_x, idx_y] += c
                
                # Add to F[-idx] (conjugate)
                idx_neg_x = (-gx) % size  # Wait, gx here is the original from the loop, not idx_x
                # We need the negative of the specific symmetric G vector (sgx, sgy)
                # But we are iterating over unique indices.
                # Let's just find the negative index for each idx.
                idx_neg_x = (-idx_x) % size
                idx_neg_y = (-idx_y) % size
                
                fft_coeffs[idx_neg_x, idx_neg_y] += np.conj(c)
        
        # Inverse FFT
        field = np.real(np.fft.ifft2(fft_coeffs) * (size * size))
        
        # Threshold
        # The DC term was 0.5. The modulation is around it.
        # Threshold at 0.5
        cell = (field >= 0.5).astype(int)
        
        return cell

    # --- Optimization ---
    
    # Define a set of G vectors to optimize.
    # We pick low-frequency G vectors that are important for the band structure.
    # G vectors are in units of 2*pi/a.
    # We pick a few unique orbits.
    
    # Orbit 1: (1, 0) and symmetries -> (1,0), (0,1), (-1,0), (0,-1)
    # Orbit 2: (1, 1) and symmetries -> (1,1), (1,-1), (-1,1), (-1,-1)
    # Orbit 3: (2, 0) ...
    # Orbit 4: (2, 1) ...
    
    # We will optimize amplitude and phase for a few of these.
    # Let's start with 4 orbits: (1,0), (1,1), (2,0), (2,1)
    
    # Initial guess: Random phases, small amplitudes.
    np.random.seed(42)
    
    # coeffs: list of [gx, gy, amp, phase]
    # We only store one representative from each orbit.
    initial_coeffs = []
    orbits = [
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (3, 2)
    ]
    
    for gx, gy in orbits:
        amp = np.random.uniform(0.1, 0.5)
        phase = np.random.uniform(0, 2 * np.pi)
        initial_coeffs.append([gx, gy, amp, phase])
    
    best_coeffs = [c[:] for c in initial_coeffs]
    best_cell = fourier_to_cell(best_coeffs)
    best_score = evaluate_cell_fast(best_cell)
    
    # Local Search
    # We optimize amp and phase for each coefficient.
    
    num_iterations = 5000
    step_size_amp = 0.05
    step_size_phase = 0.1
    
    for it in range(num_iterations):
        if time.time() > deadline - 5:
            break
            
        # Pick a random coefficient to perturb
        idx_c = np.random.randint(0, len(best_coeffs))
        
        # Perturb amplitude
        new_coeffs = [c[:] for c in best_coeffs]
        new_coeffs[idx_c][2] += np.random.uniform(-step_size_amp, step_size_amp)
        # Clamp amplitude
        new_coeffs[idx_c][2] = np.clip(new_coeffs[idx_c][2], 0.0, 1.0)
        
        # Perturb phase
        new_coeffs[idx_c][3] += np.random.uniform(-step_size_phase, step_size_phase)
        
        new_cell = fourier_to_cell(new_coeffs)
        new_score = evaluate_cell_fast(new_cell)
        
        if new_score > best_score:
            best_score = new_score
            best_coeffs = new_coeffs
            best_cell = new_cell
    
    # Final check and return
    # Ensure the returned cell is exactly C4v symmetric (the fourier_to_cell should guarantee it, 
    # but thresholding might break it slightly if we are on the boundary? No, field is symmetric).
    
    # Convert to nested list
    final_cell = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(best_cell[i, j]))
        final_cell.append(row)
    
    return final_cell
