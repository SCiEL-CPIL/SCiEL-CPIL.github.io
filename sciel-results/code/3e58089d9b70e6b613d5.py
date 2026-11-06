import numpy as np
import time
import math
import random

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 185.0  # Safety margin for the 200s/230s limit

    # -------------------------------------------------------
    # 1. Fast Band Solver (Internal Optimization)
    # -------------------------------------------------------
    # We use a reduced set of plane waves (15x15=225) for speed during search.
    # The evaluator uses 289 (17x17), but 225 is usually sufficient for ranking.
    half_pw = 7 
    G_vecs = []
    for m in range(-half_pw, half_pw + 1):
        for n in range(-half_pw, half_pw + 1):
            G_vecs.append((m, n))
    G_vecs = np.array(G_vecs) * (2.0 * np.pi)
    nG = len(G_vecs)
    N_pw = 15 # 15x15 grid

    # Precompute k-points in the irreducible BZ
    # We need points that capture the band edges.
    # High symmetry: Gamma(0,0), X(pi,0), M(pi,pi), and along X-M and Gamma-M.
    k_points = []
    # Gamma
    k_points.append((0.0, 0.0))
    # X-M line: k = (pi - t, t) for t in [0, pi]
    for t in np.linspace(0, np.pi, 6):
        k_points.append((np.pi - t, t))
    # Gamma-M line: k = (t, t) for t in [0, pi]
    for t in np.linspace(0, np.pi, 6):
        k_points.append((t, t))
    # X-Gamma line: k = (pi - t, 0) for t in [0, pi]
    for t in np.linspace(0, np.pi, 6):
        k_points.append((pi - t, 0))
    # X-M interior points
    for i in range(1, 4):
        for j in range(i, 4):
             kx = (4 - i) * np.pi / 4.0
             ky = j * np.pi / 4.0
             k_points.append((kx, ky))
    
    k_points = np.array(k_points)

    def compute_score(cell):
        """
        Compute the 4-5 TM band gap score for a given cell (32x32 array of 0/1).
        """
        eps_grid = eps_low + (eps_high - eps_low) * np.array(cell, dtype=float)
        
        # FFT of epsilon for fast convolution
        # We need 1/eps in Fourier space.
        eps_F = np.fft.fft2(eps_grid) / (size * size)
        inv_eps_F = 1.0 / eps_F
        
        min_omega5 = np.inf
        max_omega4 = 0.0
        
        for kx, ky in k_points:
            kG = np.stack([kx + G_vecs[:, 0], ky + G_vecs[:, 1]], axis=1)
            kG2 = kG[:, 0]**2 + kG[:, 1]**2
            
            # Build the operator A such that A v = k^2 v - (k+G) * (1/eps * (k+G) * v)
            # For TM: (1/eps) * (k+G) x ( (k+G) x E ) = (k^2 delta - (k+G)(k+G) 1/eps) E
            # In Fourier basis: D_{GG'} = k^2 delta_{GG'} - sum_{G''} (k+G) (k+G'') (1/eps)_{G-G''}
            # This is a dense matrix operation.
            # Let's use the matrix form for small N_pw.
            
            # Precompute K matrix components
            # Kx, Ky are vectors of size nG
            Kx = kx + G_vecs[:, 0]
            Ky = ky + G_vecs[:, 1]
            K2 = Kx**2 + Ky**2
            
            # Construct the matrix M = K2 * I - outer(K, K) * inv_eps_conv
            # Actually, the term is sum_{G''} (Kx G)(Ky G) ... no, (K x K) for TM is scalar in 2D cross product magnitude squared?
            # For TM polarization (H field along z, E in xy):
            # The eigenvalue equation is:
            # sum_{G'} [ k^2 delta_{GG'} - (k+G) * (k+G') * (1/eps)_{G-G'} ] E_{G'} = 0
            # Wait, the standard PW method for TM:
            # (1/eps_GG') (k+G) (k+G') ... 
            # Let's use the explicit matrix construction which is O(N^3) for eig, but N=225 is small.
            
            # Matrix element: A_{GG'} = k^2 delta_{GG'} - (k+G) (k+G') conv_inv_eps_{G-G'}
            # Note: conv_inv_eps_{G-G'} is the inverse Fourier transform of 1/eps? No.
            # 1/eps is periodic. Its Fourier series is (1/eps)_Q.
            # The convolution in real space becomes multiplication in Fourier space.
            # The term is (k+G) * ( (1/eps) * (k+G') * E_{G'} )_{G}
            # In Fourier space, (A * B)_G = sum_{Q} A_Q B_{G-Q}.
            # So the term is sum_{G'} sum_{Q} (k+G)_{x,y} dot (k+G')_{x,y} (1/eps)_Q E_{G'-Q}?
            # Actually, the operator is:
            # A_{GG'} = k^2 delta_{GG'} - sum_{Q} (1/eps)_Q (k+G-Q) (k+G')
            # Wait, indices must match.
            # Let's stick to the matrix form:
            # A = k^2 I - K K^T * Conv(1/eps) -- this is not quite right for vectors.
            # For TM:
            # A_{GG'} = k^2 delta_{GG'} - (k+G) (k+G') * (1/eps)_{G-G'}
            # This is only true if 1/eps is constant? No.
            # The correct form is:
            # A_{GG'} = k^2 delta_{GG'} - sum_{Q} (1/eps)_Q (k+G-Q) (k+G')
            
            # Let's build the matrix.
            # (1/eps)_Q is the Fourier coefficients of 1/eps.
            # Q ranges over the same G_vecs grid for the convolution?
            # Yes, we assume 1/eps is band-limited to the same grid.
            
            inv_eps_coeffs = 1.0 / np.fft.fft2(eps_grid, s=(N_pw, N_pw)) / (N_pw*N_pw)
            # Note: fft2 of size N_pw? No, eps_grid is 32x32.
            # We need the Fourier coefficients of the periodic function 1/eps(x,y) on the unit cell.
            # (1/eps)_Q = (1/A) ∫ (1/eps(x,y)) e^{-i Q·x} dx
            # This is just the IDFT of the pixel grid (upsampled or zero-padded?).
            # Since we are solving in a PW basis with G up to half_pw=7, we need (1/eps)_Q for Q in that range.
            # We can compute the DFT of the 32x32 grid and take the central 15x15.
            
            eps_F_full = np.fft.fft2(eps_grid) / (size * size)
            inv_eps_grid = 1.0 / eps_grid
            inv_eps_F_full = np.fft.fft2(inv_eps_grid) / (size * size)
            
            # Extract the 15x15 block of Fourier coefficients
            # Center of the 32x32 FFT is (16,16). We want indices -7..7.
            # In numpy fft, indices 0..31. -7 corresponds to 25. 7 corresponds to 7.
            # Range: 32-7=25 to 7? No.
            # Indices: -7 -> 32-7=25, ..., -1 -> 31, 0 -> 0, 1 -> 1, ..., 7 -> 7.
            # So we need [25..31, 0..7].
            
            idx = list(range(32-7, 32)) + list(range(0, 8))
            inv_eps_coeffs = inv_eps_F_full[np.ix_(idx, idx)]
            
            # Build matrix A
            # A_{GG'} = k^2 delta_{GG'} - sum_Q (1/eps)_Q (k+G-Q) (k+G')
            # This is O(N^3) = 225^3 ~ 11 million ops. Fast enough in C/NumPy.
            
            # Let's vectorize.
            # Q_vecs are the G_vecs.
            # (k+G-Q) is vector.
            # Term = sum_Q inv_eps_coeffs[Q] * (k+G-Q) * (k+G')
            
            # K_Q = k + G - Q
            # We can loop over Q (225 times) and accumulate outer products.
            
            A_mat = np.zeros((nG, nG), dtype=complex)
            A_mat[np.diag_indices(nG)] = K2 # k^2 term
            
            # Precompute k+G
            kGx = kx + G_vecs[:, 0]
            kGy = ky + G_vecs[:, 1]
            
            for q_idx, (qx, qy) in enumerate(G_vecs):
                coeff = inv_eps_coeffs[q_idx, q_idx] # Wait, inv_eps_coeffs is 2D.
                # The coefficient for Q=(mx,my) is at index (mx+7, my+7) in the 15x15 array.
                # But G_vecs order is arbitrary. We need to map G_vecs indices to inv_eps_coeffs indices.
                # G_vecs[m,n] -> (m, n).
                # inv_eps_coeffs is indexed by [i, j] corresponding to (i-7, j-7).
                # So for G_vecs[q_idx] = (mx, my), coeff is inv_eps_coeffs[mx+7, my+7].
                
                mx, my = G_vecs[q_idx]
                coeff = inv_eps_coeffs[int(mx+7), int(my+7)]
                
                if abs(coeff) < 1e-12:
                    continue
                    
                # k + G - Q
                # G is the row index vector, Q is the current q.
                # kGx - qx, kGy - qy
                KQx = kGx - qx
                KQy = kGy - qy
                KQ2 = KQx**2 + KQy**2 # No, it's dot product with (k+G')
                
                # We need (k+G-Q) dot (k+G')
                # This is a matrix B where B_{row, col} = (KQx_row * kGx_col + KQy_row * kGy_col)
                # B = outer(KQx, kGx) + outer(KQy, kGy)
                
                B = np.outer(KQx, kGx) + np.outer(KQy, kGy)
                A_mat -= coeff * B
                
            try:
                vals = np.linalg.eigvalsh(A_mat)
                vals = np.sort(vals.real)
                
                if len(vals) < 5:
                    return -1.0
                
                omega4 = np.sqrt(max(0, vals[3]))
                omega5 = np.sqrt(max(0, vals[4]))
                
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

    # -------------------------------------------------------
    # 2. Symmetry Enforcement
    # -------------------------------------------------------
    def enforce_full_symmetry(c):
        c_sym = np.zeros_like(c)
        for i in range(16):
            for j in range(i, 16):
                val = c[i, j]
                c_sym[i, j] = val
                c_sym[j, i] = val
                c_sym[31-i, 31-j] = val
                c_sym[31-j, 31-i] = val
                c_sym[i, 31-j] = val
                c_sym[31-j, i] = val
                c_sym[31-i, j] = val
                c_sym[j, 31-i] = val
        return c_sym

    # -------------------------------------------------------
    # 3. Generation of Initial Candidates
    # -------------------------------------------------------
    # Strategy: Moire patterns from square waves.
    # f(x,y) = sign(sin(2*pi*f1*x)) * sign(sin(2*pi*f1*y)) + sign(sin(2*pi*f2*x)) * sign(sin(2*pi*f2*y))
    
    def generate_moire_cell(f1, f2, phase1=0, phase2=0, threshold=0):
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        x = x / size
        y = y / size
        
        # Square waves
        s1x = np.sign(np.sin(2 * np.pi * f1 * x + phase1))
        s1y = np.sign(np.sin(2 * np.pi * f1 * y + phase1))
        g1 = s1x * s1y
        
        s2x = np.sign(np.sin(2 * np.pi * f2 * x + phase2))
        s2y = np.sign(np.sin(2 * np.pi * f2 * y + phase2))
        g2 = s2x * s2y
        
        val = g1 + g2
        cell = (val > threshold).astype(int)
        return enforce_full_symmetry(cell)

    # Candidate 1: Frequency 2 (checkerboard of 16x16 blocks)
    cell1 = generate_moire_cell(2.0, 3.0)
    # Candidate 2: Frequency 3
    cell2 = generate_moire_cell(3.0, 4.0)
    # Candidate 3: Frequency 4
    cell3 = generate_moire_cell(4.0, 5.0)
    # Candidate 4: Superposition with phase shift
    cell4 = generate_moire_cell(2.0, 2.5, 0, np.pi/2)
    
    candidates = [cell1, cell2, cell3, cell4]
    
    # Also add a simple 4-rod square pattern as a baseline
    cell_rod = np.zeros((size, size), dtype=int)
    for cx in [8, 23]:
        for cy in [8, 23]:
            cell_rod[cx-3:cx+4, cy-3:cy+4] = 1
    cell_rod = enforce_full_symmetry(cell_rod)
    candidates.append(cell_rod)

    best_cell = None
    best_score = -np.inf
    
    for c in candidates:
        score = compute_score(c)
        if score > best_score:
            best_score = score
            best_cell = c.copy()
            
    if best_score < 0:
        # Fallback
        best_cell = cell_rod
        best_score = compute_score(best_cell)

    # -------------------------------------------------------
    # 4. Local Search (Simulated Annealing / Hill Climbing)
    # -------------------------------------------------------
    cell = best_cell.copy()
    
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    
    steps = 0
    max_steps = 20000 # Cap to avoid infinite loops if time check fails
    
    while time.time() < deadline - 5.0 and steps < max_steps:
        steps += 1
        
        # Pick a random pixel in the fundamental domain to reduce redundant flips?
        # Or just random pixel.
        i = random.randint(0, size - 1)
        j = random.randint(0, size - 1)
        
        new_cell = cell.copy()
        new_cell[i, j] = 1 - new_cell[i, j]
        
        new_cell = enforce_full_symmetry(new_cell)
        
        if np.array_equal(new_cell, cell):
            continue
            
        new_score = compute_score(new_cell)
        
        if new_score > best_score:
            best_score = new_score
            best_cell = new_cell.copy()
            cell = new_cell
            T = max(T_min, T * 0.9) # Cool faster on improvement? Or keep standard.
        else:
            delta = new_score - best_score
            if delta < 0:
                p = math.exp(delta / T)
                if random.random() < p:
                    cell = new_cell
            
        T = max(T_min, T * cooling)
        
        if steps % 50 == 0 and time.time() > deadline - 10.0:
            break

    # Ensure the returned cell is a list of lists of ints
    return best_cell.tolist()
