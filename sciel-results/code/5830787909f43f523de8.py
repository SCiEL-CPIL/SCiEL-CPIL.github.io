import numpy as np
import time

def design(problem):
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    # Constants
    N = 32
    eps_low = 1.0
    eps_high = 12.0
    
    # Define G-vectors
    # m^2 + n^2 <= 8
    Gs = []
    for m in range(-3, 4):
        for n in range(-3, 4):
            if m*m + n*n <= 8:
                Gs.append((m, n))
    Gs = np.array(Gs)
    NG = len(Gs)
    
    # Define K-points in irreducible BZ
    # Triangle: kx in [0, 1], ky in [0, kx] (normalized to 2pi/a = 1?)
    # Let's use normalized k where G = (2pi/a)*(m,n).
    # k = (2pi/a) * (kx, ky).
    # Irreducible BZ: 0 <= ky <= kx <= 0.5 (in units of 2pi/a? No, BZ edge is 0.5).
    # Let's use units where BZ is [0, 1]x[0,1] and G are integers.
    # Then k is in [0, 0.5]x[0, 0.5] with ky <= kx.
    k_points = []
    for i in range(6):
        for j in range(i+1):
            kx = 0.5 * i / 5.0
            ky = 0.5 * j / 5.0
            k_points.append((kx, ky))
    # Add some interior points? The grid above is 21 points.
    # Evaluator uses 30. Let's use a denser grid.
    k_points = []
    for i in range(8):
        for j in range(i+1):
            kx = 0.5 * i / 7.0
            ky = 0.5 * j / 7.0
            k_points.append((kx, ky))
    # 36 points.

    # Fundamental Domain Indices
    fd_indices = []
    for i in range(16, N):
        for j in range(16, i+1):
            fd_indices.append((i, j))
    N_FD = len(fd_indices)
    
    # Precompute symmetric mappings for FD pixels
    sym_maps = []
    for i, j in fd_indices:
        pts = [
            (i, j),
            (N-1-i, j),
            (i, N-1-j),
            (N-1-i, N-1-j),
            (j, i),
            (N-1-j, i),
            (j, N-1-i),
            (N-1-j, N-1-i)
        ]
        # Remove duplicates
        pts = list(set(pts))
        sym_maps.append(pts)
        
    # Precompute exponential factors for FD pixels and Gs
    # exp(-i * G . r)
    # r is in units of pixels?
    # G = (m, n) * 2pi / N_pixels?
    # The lattice constant a corresponds to N pixels.
    # So G_vector = (2pi * m / N, 2pi * n / N).
    # Position r = (i, j) pixels.
    # Phase = -2pi * (m*i + n*j) / N.
    
    exp_factors = np.zeros((N_FD, NG, 2), dtype=np.float64) # cos, sin
    for idx, (i, j) in enumerate(fd_indices):
        for g_idx, (m, n) in enumerate(Gs):
            angle = -2.0 * np.pi * (m*i + n*j) / N
            exp_factors[idx, g_idx, 0] = np.cos(angle)
            exp_factors[idx, g_idx, 1] = np.sin(angle)

    def evaluate_cell(fd_state):
        # fd_state: list of 0/1 for FD
        # Compute eps_inv_fourier
        # eps_inv = 1 if air (0), 1/12 if dielectric (1)
        # Wait, pixel 1 is dielectric.
        # eps_inv[p] = 1.0 if p==0 else 1.0/12.0
        
        # Sum over all pixels
        # We can sum over FD and multiply by symmetry?
        # No, just expand to full array for simplicity, it's fast enough.
        # Or optimize:
        # Sum = sum_{p in FD} val(p) * (exp(G.p) + exp(G.p_sym1) + ...)
        
        eps_inv_F = np.zeros(NG, dtype=np.complex128)
        
        # To speed up, we can precompute the sum of exp factors for each symmetric point?
        # Actually, just iterating 136 FD pixels and adding 8 contributions is 1088 ops.
        # 1088 * 25 Gs = 27200 complex adds. Very fast.
        
        for idx, (i, j) in enumerate(fd_indices):
            val = 1.0 if fd_state[idx] == 0 else 1.0/12.0
            for g_idx in range(NG):
                c = exp_factors[idx, g_idx, 0]
                s = exp_factors[idx, g_idx, 1]
                # This is exp(-i G . r) for the FD pixel itself.
                # But we need to add contributions from all symmetric pixels.
                # This approach is flawed because the exp factor depends on the specific (i,j).
                
                pass # This block is inefficient. Let's just expand.

        # Efficient Expansion & Sum
        # Create a 32x32 array of eps_inv
        eps_inv = np.empty((N, N), dtype=np.float64)
        for idx, (i, j) in enumerate(fd_indices):
            v = 1.0 if fd_state[idx] == 0 else 1.0/12.0
            for pi, pj in sym_maps[idx]:
                eps_inv[pi, pj] = v
                
        # Compute Fourier
        # F[G] = sum_{i,j} eps_inv[i,j] exp(-2pi i (m i + n j)/N)
        # Use FFT?
        # We can use np.fft.fft2 on the 32x32 array.
        # The FFT gives coefficients for G = (0..31, 0..31).
        # We need to map our Gs to FFT indices.
        # G = (m, n). FFT index k corresponds to frequency k/N.
        # If m is negative, index is N+m.
        
        F_full = np.fft.fft2(eps_inv)
        
        eps_inv_F = np.zeros(NG, dtype=np.complex128)
        for g_idx, (m, n) in enumerate(Gs):
            im = m % N
            in_ = n % N
            eps_inv_F[g_idx] = F_full[im, in_]
            
        # Evaluate bands
        max_gap = 0.0
        for kx, ky in k_points:
            k_vec = np.array([kx, ky])
            # k is in units of 2pi/a. G is in units of 2pi/a * integer.
            # So k + G is just vector addition.
            
            # Matrix M
            # M_ij = (k+Gi) dot (k+Gj) * eps_inv_F[Gi-Gj]
            # We need eps_inv_F for differences Gi - Gj.
            # Precompute eps_inv_F_diff?
            # Or compute on fly.
            
            # Let's precompute a dictionary of eps_inv_F for all differences?
            # There are NG*NG differences.
            
            # Construct M
            M = np.zeros((NG, NG), dtype=np.complex128)
            for i in range(NG):
                ki = k_vec + Gs[i]
                for j in range(NG):
                    kj = k_vec + Gs[j]
                    diff = Gs[i] - Gs[j]
                    # Find index of diff in Gs?
                    # Or compute eps_inv_F(diff) directly?
                    # It's faster to have a lookup.
                    
                    # Since we computed F_full, we can get any diff.
                    # diff_m, diff_n
                    # F[diff] = F_full[diff_m % N, diff_n % N]
                    
                    dm, dn = diff
                    F_diff = F_full[dm % N, dn % N]
                    
                    dot = ki[0]*kj[0] + ki[1]*kj[1]
                    M[i, j] = dot * F_diff
                    
            # Eigenvalues
            # M is Hermitian?
            # M_ij = dot * F(Gi-Gj)
            # M_ji = dot * F(Gj-Gi) = dot * conj(F(Gi-Gj))
            # So M is Hermitian.
            try:
                vals = np.linalg.eigvalsh(M)
                # Sort
                vals = np.sort(vals)
                # Take 5 smallest
                if len(vals) >= 5:
                    w4 = vals[3]
                    w5 = vals[4]
                    gap = w5 - w4
                    if gap > max_gap:
                        max_gap = gap
            except:
                pass
                
        return max_gap

    # Optimization
    best_fd = None
    best_score = -1.0
    
    # Initial guess: Ring
    # Or random
    np.random.seed(42)
    current_fd = np.random.randint(0, 2, N_FD).tolist()
    
    # Evaluate initial
    score = evaluate_cell(current_fd)
    if score > best_score:
        best_score = score
        best_fd = current_fd.copy()
        
    while time.time() - start_time < time_limit:
        # Random flip
        idx = np.random.randint(N_FD)
        current_fd[idx] = 1 - current_fd[idx]
        
        new_score = evaluate_cell(current_fd)
        
        if new_score > best_score:
            best_score = new_score
            best_fd = current_fd.copy()
        else:
            # Revert
            current_fd[idx] = 1 - current_fd[idx]
            
        # Occasional restart?
        # If stuck, maybe random restart.
        # For now, simple hill climbing.
        
    # Construct final cell
    cell = [[0]*N for _ in range(N)]
    for idx, (i, j) in enumerate(fd_indices):
        v = best_fd[idx]
        for pi, pj in sym_maps[idx]:
            cell[pi][pj] = v
            
    return cell
