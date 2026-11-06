"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def design(problem):
    """
    Designs a 32x32 symmetric photonic crystal unit cell to maximize the 
    4-5 TM band gap using a surrogate-guided genetic search.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    size = 32
    eps_low = 1.0
    eps_high = 12.0
    eps_diff = eps_high - eps_low

    # Pre-compute reciprocal lattice vectors G
    # We use a grid of G vectors: Gx, Gy in {-3, -2, -1, 0, 1, 2, 3}
    # This corresponds to 7x7 = 49 plane waves
    G_range = range(-3, 4)
    G_vectors = [(gx, gy) for gx in G_range for gy in G_range]
    G_indices = {g: i for i, g in enumerate(G_vectors)}
    N_pw = len(G_vectors)
    
    # K-points for evaluation
    # Irreducible BZ: 0 <= ky <= kx
    # Points: Gamma(0,0), X(0.5, 0), M(0.5, 0.5), and some intermediate
    k_points = []
    # Gamma
    k_points.append((0.0, 0.0))
    # X
    k_points.append((0.5, 0.0))
    # M
    k_points.append((0.5, 0.5))
    # Intermediate along X-M
    for t in [0.25, 0.5, 0.75]:
        kx = 0.5 * t
        ky = 0.5 * (1 - t) + 0.5 * t # Wait, X is (0.5, 0), M is (0.5, 0.5)
        # Path X->M: kx=0.5, ky from 0 to 0.5
        k_points.append((0.5, 0.5 * t))
    # Intermediate along Gamma-X
    for t in [0.25, 0.5, 0.75]:
        k_points.append((0.5 * t, 0.0))
    # Intermediate along Gamma-M
    for t in [0.25, 0.5, 0.75]:
        k_points.append((0.5 * t, 0.5 * t))

    def expand_quadrant(q):
        """
        Expands an 8x8 symmetric quadrant to a 32x32 cell.
        q is assumed to be symmetric (q[i][j] == q[j][i]).
        """
        # Ensure symmetry
        q = (q + q.T) // 2
        # 8x8 -> 8x16 (horizontal reflection)
        h = np.hstack([q, q[:, ::-1]])
        # 8x16 -> 16x16 (vertical reflection)
        v = np.vstack([h, h[::-1, :]])
        # 16x16 -> 32x32 (tile 2x2)
        cell = np.tile(v, (2, 2))
        return cell

    def compute_surrogate_gap(cell):
        """
        Computes a surrogate for the 4-5 TM band gap ratio.
        """
        # Compute DFT of the cell
        # The cell is 32x32.
        # eps_shape(G) = 1/(32*32) * sum eps(r) exp(-i G r)
        # Using numpy fft2, the result is scaled by N^2.
        # We need the Fourier coefficients of the *shape* (1/0).
        # eps_total(G) = eps_low * delta_G0 + eps_diff * shape_G
        
        # fft2 gives sum over r of f(r) exp(-2pi i (n/N) k)
        # G is in units of 2pi/a.
        # The indices in fft2 correspond to G_x = 2pi * n / 32.
        # We need G in units of 2pi/a (where a=1).
        # So G_index in fft2 corresponds to G = 2pi * index / 32.
        # Our G_vectors are integers (multiples of 2pi/a).
        # So G = 2pi * g.
        # This corresponds to fft2 index = g * 32 / 2 = 16 * g.
        # Wait, the DFT definition:
        # F[k] = sum_{n=0}^{N-1} f[n] exp(-2pi i n k / N)
        # G = 2pi k / N.
        # If we want G = 2pi * g (where g is integer), then k = g * N / 2 = 16 * g.
        # So we need to extract coefficients at indices 16*g.
        
        F = np.fft.fft2(cell)
        # Normalize: The coefficient c_G = F[index] / N^2
        # But wait, the standard Fourier series is f(r) = sum c_G exp(i G r).
        # c_G = (1/A) int f(r) exp(-i G r) dr.
        # Discrete: c_G = (1/N^2) sum f[n] exp(-i G n).
        # So c_G = F[index] / N^2.
        
        # Construct eps_total(G)
        # eps_total(G) = eps_low if G=0, else eps_low + eps_diff * c_G
        # Actually, eps(r) = eps_low + eps_diff * shape(r).
        # So eps(G) = eps_low * delta_{G,0} + eps_diff * shape(G).
        
        eps_G = np.zeros(N_pw, dtype=complex)
        for idx, (gx, gy) in enumerate(G_vectors):
            if gx == 0 and gy == 0:
                eps_G[idx] = eps_low
            else:
                # Get fft2 index
                # fx index: (gx * 16) % 32
                # fy index: (gy * 16) % 32
                ix = (gx * 16) % 32
                iy = (gy * 16) % 32
                shape_G = F[iy, ix] / (size * size)
                eps_G[idx] = eps_low + eps_diff * shape_G
        
        # Invert eps_G?
        # The PWE equation for TM is:
        # sum_{G'} (k+G)^2 / eps(G-G') c_{G'} = w^2 c_G
        # We need 1/eps(G).
        inv_eps_G = np.zeros(N_pw, dtype=complex)
        for i in range(N_pw):
            # G_i
            gx1, gy1 = G_vectors[i]
            for j in range(N_pw):
                # G_j
                gx2, gy2 = G_vectors[j]
                dGx = gx1 - gx2
                dGy = gy1 - gy2
                # Find index of dG in G_vectors
                # dG must be in the range of G_vectors
                # G_vectors are -3..3. dG is -6..6.
                # This is a problem. The convolution requires eps(G_i - G_j).
                # If G_i - G_j is outside our calculated DFT range, we need to calculate it.
                # But we only calculated eps_G for G in -3..3.
                # We need eps_G for G in -6..6.
                # We can calculate the DFT for a larger range.
                # Or, we can note that shape(r) is real and symmetric, so shape(-G) = shape(G)*.
                # And we can compute shape(G) for any G by direct summation if needed, 
                # but it's slow.
                # Better: Compute DFT for G in -6..6.
                pass
        
        # Let's re-compute eps_G for a larger range to handle the convolution.
        # Max G difference: 3 - (-3) = 6.
        # So we need G in -6..6.
        G_range_large = range(-6, 7)
        G_vectors_large = [(gx, gy) for gx in G_range_large for gy in G_range_large]
        eps_G_large = {}
        for gx in G_range_large:
            for gy in G_range_large:
                if gx == 0 and gy == 0:
                    eps_G_large[(gx, gy)] = eps_low
                else:
                    ix = (gx * 16) % 32
                    iy = (gy * 16) % 32
                    shape_G = F[iy, ix] / (size * size)
                    eps_G_large[(gx, gy)] = eps_low + eps_diff * shape_G
        
        # Now build the matrix for each k-point
        min_w5 = float('inf')
        max_w4 = 0.0
        
        for kx, ky in k_points:
            # k is in units of 2pi/a.
            # k + G
            k_vec = np.array([kx, ky])
            
            # Matrix M
            M = np.zeros((N_pw, N_pw), dtype=complex)
            for i in range(N_pw):
                gx1, gy1 = G_vectors[i]
                k_plus_G1 = np.array([kx + gx1, ky + gy1])
                k_plus_G1_sq = np.dot(k_plus_G1, k_plus_G1)
                
                for j in range(N_pw):
                    gx2, gy2 = G_vectors[j]
                    dGx = gx1 - gx2
                    dGy = gy1 - gy2
                    eps_val = eps_G_large[(dGx, dGy)]
                    # M[i, j] = k_plus_G1_sq / eps_val
                    if np.abs(eps_val) > 1e-9:
                        M[i, j] = k_plus_G1_sq / eps_val
                    else:
                        M[i, j] = 0 # Should not happen for eps_low=1, eps_high=12
            
            # Solve eigenvalues
            # M is Hermitian? 
            # M[i, j] = |k+G_i|^2 / eps(G_i - G_j).
            # eps(G) is complex.
            # For real eps(r), eps(-G) = eps(G)*.
            # M[j, i] = |k+G_j|^2 / eps(G_j - G_i) = |k+G_j|^2 / eps(-(G_i - G_j)) = |k+G_j|^2 / eps(G_i - G_j)*.
            # This is not necessarily Hermitian.
            # Wait, the operator is self-adjoint.
            # The matrix should be Hermitian.
            # Let's check.
            # <i|H|j> = sum_k <i|V|k> <k|T|j> ?
            # The standard PWE matrix for TM is:
            # A_{mn} = delta_{mn} (k+G_m)^2 / eps(G_m - G_n) ? No.
            # The equation is: sum_{G'} (k+G')^2 / eps(G-G') c_{G'} = w^2 c_G.
            # This is not symmetric.
            # However, it can be transformed to a symmetric form by multiplying by sqrt(eps).
            # Or we can just use eigh on the Hermitian part if it's close.
            # Actually, for real dielectric, the matrix is symmetric if we define it correctly.
            # Let's use the symmetric formulation:
            # B_{mn} = (k+G_m) (k+G_n) / sqrt(eps(G_m-G_n) eps(G_n-G_m)) ?
            # This is getting complicated.
            # Simpler: Use the non-symmetric matrix and take real parts of eigenvalues?
            # Or use scipy.linalg.eig.
            
            # For speed and stability, let's assume the matrix is close to Hermitian 
            # and use np.linalg.eigvalsh on (M + M.T)/2 if it's real?
            # eps(G) is complex.
            # Let's just use np.linalg.eig.
            
            eigvals = np.linalg.eigvals(M)
            # Filter for real positive eigenvalues
            real_eigvals = np.sort([np.real(e) for e in eigvals if np.imag(e) < 1e-3 and np.real(e) > 0])
            
            if len(real_eigvals) >= 5:
                w4 = np.sqrt(real_eigvals[3]) # 4th band (index 3)
                w5 = np.sqrt(real_eigvals[4]) # 5th band (index 4)
                max_w4 = max(max_w4, w4)
                min_w5 = min(min_w5, w5)
        
        if min_w5 == float('inf'):
            return -1.0
            
        gap = min_w5 - max_w4
        midgap = (min_w5 + max_w4) / 2.0
        if midgap == 0:
            return -1.0
            
        return gap / midgap

    # Search
    # 1. Generate initial candidates
    def random_quadrant():
        q = np.random.randint(0, 2, (8, 8))
        q = (q + q.T) // 2
        return q

    candidates = []
    # Rod
    q_rod = np.zeros((8, 8), dtype=int)
    for i in range(8):
        for j in range(8):
            x = (j + 0.5) / 8 - 0.5
            y = (i + 0.5) / 8 - 0.5
            if x*x + y*y < 0.25**2: # Radius 0.25 in 8x8 scale
                q_rod[i, j] = 1
    candidates.append(q_rod)
    
    # Checkerboard
    q_check = np.zeros((8, 8), dtype=int)
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                q_check[i, j] = 1
    candidates.append(q_check)
    
    # Random
    for _ in range(10):
        candidates.append(random_quadrant())
        
    best_q = None
    best_score = -1.0
    
    # Evaluate initial candidates
    for q in candidates:
        cell = expand_quadrant(q)
        score = compute_surrogate_gap(cell)
        if score > best_score:
            best_score = score
            best_q = q.copy()
            
    # Hill Climbing
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        # Generate mutants
        mutants = []
        for _ in range(50):
            q_mut = best_q.copy()
            # Flip 1-3 bits
            n_flips = np.random.randint(1, 4)
            for _ in range(n_flips):
                i = np.random.randint(0, 8)
                j = np.random.randint(0, 8)
                q_mut[i, j] = 1 - q_mut[i, j]
                # Enforce symmetry
                q_mut[j, i] = q_mut[i, j]
            mutants.append(q_mut)
            
        best_mutant = None
        best_mutant_score = best_score
        
        for q_mut in mutants:
            cell = expand_quadrant(q_mut)
            score = compute_surrogate_gap(cell)
            if score > best_mutant_score:
                best_mutant_score = score
                best_mutant = q_mut
                
        if best_mutant is not None:
            best_q = best_mutant
            best_score = best_mutant_score
        else:
            # Stagnation, try random restart
            q_new = random_quadrant()
            cell = expand_quadrant(q_new)
            score = compute_surrogate_gap(cell)
            if score > best_score:
                best_q = q_new
                best_score = score
            else:
                # Break if no improvement for a while?
                # Just continue for now.
                pass
                
    # Final evaluation
    final_cell = expand_quadrant(best_q)
    return final_cell.tolist()

# EVOLVE-BLOCK-END
