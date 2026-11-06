"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def design(problem):
    """Design a 32x32 photonic crystal unit cell to maximize the TM 4-5 band gap."""
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    
    start_time = time.time()
    time_limit = 190.0  # Safety margin

    # Precompute pixel coordinates and meshgrid for fast Fourier coefficient calculation
    # Pixel centers are at (i + 0.5) / size
    px = (np.arange(size) + 0.5) / size
    py = (np.arange(size) + 0.5) / size
    PX, PY = np.meshgrid(px, py, indexing='xy')
    
    # We use a plane-wave basis of size N=9 (indices -4 to 4)
    N = 9
    indices = np.arange(-N//2, N//2 + 1)
    
    # Precompute G vectors and their squared magnitudes for the basis
    # G = 2*pi * (gx, gy)
    # We work in units where a=1, so G = 2*pi*(gx, gy)
    # The eigenvalue equation for TM is:
    # sum_G' eps_{G-G'} |G'+k|^2 c_{G'} = omega^2 |G+k|^2 c_G
    # This is a generalized eigenvalue problem: M c = omega^2 D c
    # M_{ij} = eps_{G_i - G_j} |G_j + k|^2
    # D_{ii} = |G_i + k|^2
    
    # Precompute all G vectors
    G_vecs = []
    for i in indices:
        for j in indices:
            G_vecs.append((2*np.pi*i, 2*np.pi*j))
    G_vecs = np.array(G_vecs)
    num_pw = len(G_vecs)
    
    # Precompute the phase factors for Fourier coefficients
    # eps_G = (1/A) sum_pixels eps(r) exp(-i G . r)
    # We can precompute exp(-i G . r) for all G and all pixels
    # This is a (num_pw, size, size) array, which is 81 * 32 * 32 ~ 83k complex numbers. Fine.
    phases = np.zeros((num_pw, size, size), dtype=complex)
    for idx, (gx, gy) in enumerate(G_vecs):
        phases[idx] = np.exp(-1j * (gx * PX + gy * PY))
        
    # Map G difference to index in the N=9 grid
    # G_diff = G_i - G_j = 2*pi*(gx_i - gx_j, gy_i - gy_j)
    # The index in the eps_f array (which is N x N) corresponds to (gx_i - gx_j, gy_i - gy_j)
    # Since indices are -4..4, the difference is -8..8.
    # We need to map this to 0..N-1 with wrap-around? 
    # Actually, the Fourier series is periodic. eps_{G} = eps_{G + G_recip}.
    # In the basis of -4..4, if we need eps_{(5,0)}, it is equivalent to eps_{(-4,0)} because 
    # the pixel map is periodic with period 1, and the basis covers the first Brillouin zone components.
    # Wait, the convolution in real space becomes multiplication in Fourier space.
    # eps_{G} is the Fourier coefficient of the periodic dielectric function.
    # If we compute eps_f for G in -4..4, then for a difference of 5, we should use eps_f[-4] 
    # because e^{i 2 pi 5 r} = e^{i 2 pi (-4) r} e^{i 2 pi 9 r} and since r is in [0,1), 
    # actually no. The Fourier coefficient of a periodic function is defined for all integer G.
    # eps_{G+9} != eps_G generally. 
    # However, our basis is limited. The error comes from truncating the sum.
    # For the matrix element M_{ij}, we need eps_{G_i - G_j}.
    # If G_i - G_j is outside the range of our computed eps_f, we must compute it or approximate it.
    # Given N=9, the max difference is 8. We computed -4..4.
    # We can extend the computation of eps_f to -8..8 to be safe? 
    # That would be 17x17 = 289 coefficients. 
    # Let's just compute eps_f on the fly for the specific differences needed, or precompute a larger set.
    # Precomputing a 17x17 set of phases is cheap.
    
    N_large = 17
    indices_large = np.arange(-N_large//2, N_large//2 + 1)
    G_vecs_large = []
    for i in indices_large:
        for j in indices_large:
            G_vecs_large.append((2*np.pi*i, 2*np.pi*j))
    G_vecs_large = np.array(G_vecs_large)
    num_pw_large = len(G_vecs_large)
    
    phases_large = np.zeros((num_pw_large, size, size), dtype=complex)
    for idx, (gx, gy) in enumerate(G_vecs_large):
        phases_large[idx] = np.exp(-1j * (gx * PX + gy * PY))
        
    # Index map for large grid
    # index = (i + 8) * 17 + (j + 8)
    
    def get_eps_coeff(gx, gy):
        """Get Fourier coefficient for G = 2*pi*(gx, gy)."""
        # Map to index in large grid
        ix = int(gx) + N_large//2
        iy = int(gy) + N_large//2
        if ix < 0 or ix >= N_large or iy < 0 or iy >= N_large:
            return 0.0
        return eps_f_large[ix, iy]

    def fast_score(cell_list):
        """
        Fast approximation of the band gap score.
        """
        cell = np.array(cell_list, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell
        
        # Compute Fourier coefficients of epsilon
        # eps_f_large is (17, 17)
        # eps_G = (1/A) sum eps(r) exp(-i G . r)
        # We can do this efficiently using matrix multiplication or loop
        # Let's use a loop for clarity, but it might be slow. 
        # Better: 
        # eps_f = (1/(size*size)) * sum_pixels eps(pixel) * phases_large
        # This is a weighted sum of complex arrays.
        # We can reshape and use einsum or just dot product if flattened.
        eps_flat = eps.flatten()
        phases_flat = phases_large.reshape(num_pw_large, -1)
        eps_f_large = np.sum(phases_flat * eps_flat[None, :], axis=1) / (size * size)
        eps_f_large = eps_f_large.reshape(N_large, N_large)
        
        max_omega4 = 0.0
        min_omega5 = np.inf
        
        # Key k-points in the irreducible Brillouin zone
        # Gamma (0,0), M (0.5, 0.5), X (0.5, 0)
        # In units of 2pi/a, k = (kx, ky)
        # G = 2*pi*(integer, integer)
        # So G+k = 2*pi*(integer + kx, integer + ky)
        # Let's define k in units of 2pi/a for simplicity in the formula |G+k|^2
        # |G+k|^2 = (2*pi)^2 * |g + k|^2 where g is integer vector.
        # The factor (2*pi)^2 cancels in the ratio for omega?
        # omega^2 = (c^2 / eps_avg) * lambda? No.
        # The equation is: sum eps_{G-G'} (G'+k)^2 c_{G'} = omega^2 (G+k)^2 c_G
        # If we factor out (2*pi)^2, it cancels on both sides.
        # So we can work in units where G = (gx, gy) and k = (kx, ky).
        
        k_points = [
            (0.0, 0.0),    # Gamma
            (0.5, 0.5),    # M
            (0.5, 0.0),    # X
            (0.25, 0.25),  # Midpoint Gamma-M
            (0.5, 0.25),   # Midpoint M-X
            (0.25, 0.0),   # Midpoint Gamma-X
            (0.1, 0.1),
            (0.3, 0.3),
            (0.4, 0.1),
        ]
        
        for kx, ky in k_points:
            # Construct M and D
            # M_{ij} = eps_{G_i - G_j} |G_j + k|^2
            # D_{ii} = |G_i + k|^2
            
            # G_vecs are in units of 2pi. So G_i = 2*pi*(gx_i, gy_i).
            # |G_i + k|^2 = (2*pi)^2 * | (gx_i, gy_i) + (kx, ky) |^2
            # Since (2*pi)^2 is common to all terms, we can ignore it for the eigenvalues 
            # if we are consistent? 
            # Wait, omega^2 has units of 1/length^2.
            # If we use G in 2pi units, then |G+k|^2 is in (2pi/length)^2.
            # The eigenvalues lambda will be in units of (2pi/length)^2.
            # So omega = 2*pi*sqrt(lambda) / c? 
            # Actually, the score is a ratio, so units cancel.
            # Let's just compute the eigenvalues of the dimensionless matrix.
            
            # Calculate |G_j + k|^2 for all j in the small basis (N=9)
            # G_small_vecs are the first num_pw vectors from G_vecs (which are 2pi * integer)
            # Let's use the integer coordinates for the small basis to avoid 2pi factors
            g_small = np.array([(i, j) for i in indices for j in indices])
            
            kg2_small = np.sum((g_small + np.array([kx, ky]))**2, axis=1)
            
            # Build M
            M = np.zeros((num_pw, num_pw), dtype=complex)
            for i in range(num_pw):
                for j in range(num_pw):
                    dgx = g_small[i, 0] - g_small[j, 0]
                    dgy = g_small[i, 1] - g_small[j, 1]
                    # Get eps coefficient
                    # Need to map (dgx, dgy) to eps_f_large
                    # eps_f_large index: (dgx + 8, dgy + 8)
                    ix = int(dgx) + N_large//2
                    iy = int(dgy) + N_large//2
                    if 0 <= ix < N_large and 0 <= iy < N_large:
                        eps_val = eps_f_large[ix, iy]
                    else:
                        eps_val = 0.0
                    M[i, j] = eps_val * kg2_small[j]
            
            D = np.diag(kg2_small)
            
            # Solve generalized eigenvalue problem M c = lambda D c
            # Transform to symmetric: A = D^{-1/2} M D^{-1/2}
            D_inv_sqrt = 1.0 / np.sqrt(kg2_small)
            A = M * D_inv_sqrt[:, None] * D_inv_sqrt[None, :]
            A = (A + A.conj().T) / 2.0 # Ensure Hermitian
            
            try:
                evals = np.linalg.eigvalsh(A)
                evals = np.real(evals)
                evals = np.sort(evals)
                # We need the 4th and 5th bands (indices 3 and 4)
                if len(evals) >= 5:
                    omega4 = np.sqrt(evals[3])
                    omega5 = np.sqrt(evals[4])
                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
            except:
                pass
        
        if min_omega5 == np.inf or max_omega4 == 0:
            return -1.0
        
        gap = 2 * (min_omega5 - max_omega4)
        midgap = min_omega5 + max_omega4
        if midgap == 0:
            return -1.0
        return gap / midgap

    def make_symmetric_cell(block):
        """
        Take a 16x16 block (top-left quadrant) and mirror it to create a 32x32 cell.
        The cell must be symmetric under:
        1. x -> -x (left-right mirror)
        2. y -> -y (top-bottom mirror)
        3. (x,y) -> (y,x) (transpose)
        
        If we define the block as the top-left 16x16, we need to be careful with the center lines.
        Size is 32. Center is between 15 and 16.
        So pixels 0..15 are the first half.
        Mirror: cell[x, y] = cell[31-x, y] = cell[x, 31-y] = cell[y, x].
        
        If we fill block[0..15, 0..15], then:
        cell[0..15, 0..15] = block
        cell[16..31, 0..15] = block[::-1, :]  (bottom-left)
        cell[0..15, 16..31] = block[:, ::-1]  (top-right)
        cell[16..31, 16..31] = block[::-1, ::-1] (bottom-right)
        
        Check transpose symmetry:
        cell[x, y] should equal cell[y, x].
        For x,y in 0..15: block[x, y] == block[y, x]. So block must be symmetric.
        For x in 0..15, y in 16..31: cell[x, y] = block[x, 31-y].
        cell[y, x] (where y>=16, x<=15): 
        y' = 31-y (in 0..15), x' = x.
        cell[y, x] is in bottom-left quadrant.
        cell[16..31, 0..15] = block[::-1, :].
        So cell[y, x] = block[31-y, x] = block[y', x].
        We need block[x, y'] == block[y', x]. Since block is symmetric, this holds.
        
        So we just need to enforce that the 16x16 block is symmetric (block == block.T).
        """
        block = np.array(block)
        # Enforce symmetry of the block itself
        block = (block + block.T) // 2
        
        cell = np.zeros((size, size), dtype=np.uint8)
        cell[:16, :16] = block
        cell[16:, :16] = block[::-1, :]
        cell[:16, 16:] = block[:, ::-1]
        cell[16:, 16:] = block[::-1, ::-1]
        
        return cell.tolist()

    # Search Strategy
    best_cell = None
    best_score = -np.inf
    
    # Initialize with a few random symmetric blocks
    # We will perform a local search on the 16x16 block.
    
    # To speed up scoring, we can cache the score for a specific block hash?
    # Or just run the search.
    
    # Let's start with a 4-rod configuration as a seed
    # Rods at (4,4), (4,12), (12,4), (12,12) in the 16x16 block?
    # In 32x32, rods at (8,8) etc.
    # In 16x16 block (0..15), center is 7.5.
    # Let's place a rod at (4,4) with radius 2.
    
    def generate_seed_block():
        block = np.zeros((16, 16), dtype=np.uint8)
        # Create a symmetric pattern
        # e.g. a cross
        block[7, :] = 1
        block[:, 7] = 1
        # Or a ring
        # Let's try a 4-rod pattern
        centers = [(4, 4), (4, 11), (11, 4), (11, 11)]
        for cx, cy in centers:
            for x in range(16):
                for y in range(16):
                    if (x-cx)**2 + (y-cy)**2 < 4: # radius ~2
                        block[x, y] = 1
        return (block + block.T) // 2

    current_block = generate_seed_block()
    current_cell = make_symmetric_cell(current_block)
    current_score = fast_score(current_cell)
    
    best_block = current_block.copy()
    best_score = current_score
    best_cell = current_cell
    
    # Local Search
    # We flip pixels in the 16x16 block.
    # To ensure symmetry, we flip a pair (x,y) and (y,x) if x!=y, or just (x,y) if x==y.
    # Actually, if we flip (x,y), we must also flip (y,x) to maintain block symmetry.
    # And then the full cell symmetry is maintained by the mirroring.
    
    num_attempts = 0
    max_attempts = 5000
    
    while time.time() - start_time < time_limit - 10:
        # Pick a random position in the 16x16 block
        x = random.randint(0, 15)
        y = random.randint(0, 15)
        
        # Create a new block
        new_block = current_block.copy()
        
        # Flip (x, y) and (y, x)
        new_block[x, y] = 1 - new_block[x, y]
        new_block[y, x] = 1 - new_block[y, x]
        
        # Check if it changed
        if np.array_equal(new_block, current_block):
            continue
            
        new_cell = make_symmetric_cell(new_block)
        new_score = fast_score(new_cell)
        
        if new_score > current_score:
            current_block = new_block
            current_cell = new_cell
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_block = current_block.copy()
                best_cell = current_cell
                
            # Reset counter if improved?
            # num_attempts = 0 
        else:
            # Occasional random restart or larger jump?
            # For now, just try another flip.
            pass
            
        num_attempts += 1
        if num_attempts > max_attempts:
            # Random restart
            current_block = generate_seed_block()
            current_cell = make_symmetric_cell(current_block)
            current_score = fast_score(current_cell)
            num_attempts = 0
            max_attempts += 2000 # Allow more attempts in future restarts?
            
    # Final check
    if best_cell is None:
        best_cell = make_symmetric_cell(generate_seed_block())
        
    return best_cell
# EVOLVE-BLOCK-END
