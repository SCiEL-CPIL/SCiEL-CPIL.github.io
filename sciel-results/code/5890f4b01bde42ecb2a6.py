"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell maximizing TM band-4/5 gap."""
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    
    start_time = time.time()
    deadline = 180.0  # seconds, with safety margin for 200s limit
    
    # We'll use a parametric approach with concentric structures
    # The cell must be symmetric: cell == cell[::-1] row-wise, column-wise, and transpose
    # This means we only need to define the top-left 16x16 quadrant
    
    # Strategy: Try several parametric families and pick the best
    # Family 1: Concentric squares (air core + dielectric shell)
    # Family 2: Cross-shaped dielectric bars
    # Family 3: Combination of rod + cross
    
    best_cell = None
    best_score = -1.0
    
    def build_cell_from_quadrant(q):
        """Build full 32x32 cell from 16x16 top-left quadrant, enforcing C4v symmetry."""
        cell = [[0]*size for _ in range(size)]
        for i in range(size // 2):
            for j in range(size // 2):
                val = q[i][j]
                # Mirror to all four quadrants
                cell[i][j] = val
                cell[size-1-i][j] = val
                cell[i][size-1-j] = val
                cell[size-1-i][size-1-j] = val
        return cell
    
    def compute_gap_ratio(cell, n_pw=15, n_k=10):
        """Compute the gap-midgap ratio using plane-wave method.
        
        For TM polarization in 2D: (1/eps) * (-k^2) * E = omega^2 * E
        where k is the wavevector.
        
        We solve the eigenvalue problem:
        sum_N' (1/eps)_{k-k'} * |k'|^2 * E_{k'} = omega^2 * E_k
        
        where (1/eps)_{k-k'} are the Fourier coefficients of 1/eps(x,y).
        """
        # Build 1/eps array
        eps_inv = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                eps_val = eps_high if cell[i][j] == 1 else eps_low
                eps_inv[i, j] = 1.0 / eps_val
        
        # Compute Fourier coefficients of 1/eps
        # Use FFT for efficiency
        eps_inv_fft = np.fft.fft2(eps_inv)
        eps_inv_fft = eps_inv_fft / (size * size)  # Normalize
        
        # Plane wave indices
        # n_pw is the half-width, so total is (2*n_pw+1)^2
        # Let's stick to a reasonable number. n_pw=7 -> 15x15=225 PWs.
        # The task mentions 289 PWs in evaluator, which is 17x17.
        # We can use n_pw=7 (15x15) for speed, or n_pw=8 (17x17) for accuracy.
        # Let's use n_pw=8 to match evaluator resolution roughly, but check time.
        half_n = 8 
        pw_indices = list(range(-half_n, half_n + 1))
        N = len(pw_indices)
        
        # Precompute G vectors
        G_vectors = [(m, n) for m in pw_indices for n in pw_indices]
        
        # Precompute |k+G|^2 for all G and k-points? 
        # k-points change, so we can't fully precompute, but we can structure the loop.
        
        # K-points in the irreducible BZ
        # Square lattice, irreducible BZ: 0 <= kx <= ky <= 0.5 (in units of 2*pi/a)
        
        k_points = []
        # Gamma point
        k_points.append((0.0, 0.0))
        # X point (0.5, 0) -> in our units where G = (m, n) for integers m, n
        # BZ: -0.5 <= kx, ky <= 0.5
        # Irreducible: 0 <= kx <= ky <= 0.5
        
        # Sample points along the boundary and interior
        n_sample = 4
        for i in range(n_sample + 1):
            for j in range(n_sample + 1):
                ky = 0.5 * j / n_sample
                kx = 0.5 * i / n_sample
                if kx <= ky:
                    k_points.append((kx, ky))
        
        # Remove duplicates and limit count
        k_points = list(set(k_points))
        if len(k_points) > n_k:
            k_points = k_points[:n_k]
        
        max_omega4 = 0.0
        min_omega5 = float('inf')
        
        for kx, ky in k_points:
            # Build the eigenvalue matrix
            # A_{mn} = (1/eps)_{G_m - G_n} * |k + G_n|^2
            A = np.zeros((N*N, N*N), dtype=complex)
            
            # Vectorize the construction of A if possible, but N^2 is small enough for loops
            # N=17, N^2=289. 289^2 = ~83k operations per k-point.
            # With 10 k-points, ~800k ops. Very fast.
            
            for i in range(N*N):
                Gi_x, Gi_y = G_vectors[i]
                for j in range(N*N):
                    Gj_x, Gj_y = G_vectors[j]
                    dm = Gi_x - Gj_x
                    dn = Gi_y - Gj_y
                    # Get Fourier coefficient
                    idx_m = dm % size
                    idx_n = dn % size
                    coeff = eps_inv_fft[idx_m, idx_n]
                    # |k + Gj|^2
                    kmg2 = (kx + Gj_x)**2 + (ky + Gj_y)**2
                    A[i, j] = coeff * kmg2
            
            # Solve eigenvalue problem
            # A is Hermitian
            A = (A + A.conj().T) / 2.0
            
            try:
                eigvals = np.linalg.eigvalsh(A)
                eigvals = np.real(eigvals)
                # omega^2 = eigvals, so omega = sqrt(eigvals)
                # We want bands 4 and 5 (1-indexed), so indices 3 and 4 (0-indexed)
                if len(eigvals) >= 5:
                    omega4 = math.sqrt(max(eigvals[3], 0))
                    omega5 = math.sqrt(max(eigvals[4], 0))
                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
            except Exception:
                continue
        
        if min_omega5 == float('inf') or max_omega4 == 0:
            return -1.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        if midgap <= 0:
            return -1.0
        
        return 2.0 * gap / midgap
    
    # Define parametric families
    
    def family_concentric_squares(r1, r2, r3, r4):
        """Concentric square rings: dielectric in [r1,r2] and [r3,r4] from center."""
        q = [[0]*16 for _ in range(16)]
        for i in range(16):
            for j in range(16):
                # Distance from center of the 32x32 cell
                # Cell center is at (15.5, 15.5) in pixel coords
                # In the quadrant (top-left), pixel (i,j) corresponds to cell pixel (i,j)
                # Distance from center: dx = 15.5 - j, dy = 15.5 - i (in pixel units)
                dx = 15.5 - j
                dy = 15.5 - i
                r = math.sqrt(dx*dx + dy*dy) / 16.0  # Normalize to [0, ~0.71]
                if (r1 <= r < r2) or (r3 <= r < r4):
                    q[i][j] = 1
                else:
                    q[i][j] = 0
        return q
    
    def family_cross(w1, w2, offset):
        """Cross-shaped dielectric: horizontal and vertical bars."""
        q = [[0]*16 for _ in range(16)]
        for i in range(16):
            for j in range(16):
                dx = 15.5 - j
                dy = 15.5 - i
                # Horizontal bar: |dy| < w1/2 and |dx| < 16
                # Vertical bar: |dx| < w2/2 and |dy| < 16
                if abs(dy) < w1/2.0 or abs(dx) < w2/2.0:
                    q[i][j] = 1
                else:
                    q[i][j] = 0
        return q
    
    def family_rod_cross(r_rod, w_cross):
        """Dielectric rod at center plus cross-shaped bars."""
        q = [[0]*16 for _ in range(16)]
        for i in range(16):
            for j in range(16):
                dx = 15.5 - j
                dy = 15.5 - i
                r = math.sqrt(dx*dx + dy*dy) / 16.0
                # Rod
                if r < r_rod:
                    q[i][j] = 1
                # Cross
                elif abs(dy) < w_cross/2.0 and abs(dx) < 12:
                    q[i][j] = 1
                elif abs(dx) < w_cross/2.0 and abs(dy) < 12:
                    q[i][j] = 1
                else:
                    q[i][j] = 0
        return q

    # Try a grid of parameters for each family
    # Reduced grid to fit in time budget
    families = []
    
    # Family 1: Concentric squares
    # Focus on high contrast structures
    for r1 in [0.0, 0.1, 0.2]:
        for r2 in [0.3, 0.4, 0.5]:
            for r3 in [0.5, 0.6, 0.7]:
                for r4 in [0.7, 0.8, 0.9]:
                    if r1 < r2 < r3 < r4:
                        families.append(('concentric', (r1, r2, r3, r4)))
    
    # Family 2: Cross
    for w1 in [2, 4, 6]:
        for w2 in [2, 4, 6]:
            families.append(('cross', (w1, w2, 0)))
            
    # Family 3: Rod + cross
    for r_rod in [0.1, 0.2, 0.3]:
        for w_cross in [2, 4, 6]:
            families.append(('rod_cross', (r_rod, w_cross)))
    
    # Evaluate each candidate
    for family_name, params in families:
        if time.time() - start_time > deadline - 20:
            break
        
        if family_name == 'concentric':
            r1, r2, r3, r4 = params
            q = family_concentric_squares(r1, r2, r3, r4)
        elif family_name == 'cross':
            w1, w2, offset = params
            q = family_cross(w1, w2, offset)
        elif family_name == 'rod_cross':
            r_rod, w_cross = params
            q = family_rod_cross(r_rod, w_cross)
        else:
            continue
        
        cell = build_cell_from_quadrant(q)
        score = compute_gap_ratio(cell, n_pw=15, n_k=10)
        
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # If no good cell found, fall back to a simple design
    if best_cell is None:
        # Default: dielectric rod
        q = [[0]*16 for _ in range(16)]
        for i in range(16):
            for j in range(16):
                dx = 15.5 - j
                dy = 15.5 - i
                r = math.sqrt(dx*dx + dy*dy) / 16.0
                q[i][j] = 1 if r < 0.3 else 0
        best_cell = build_cell_from_quadrant(q)
    
    return best_cell
# EVOLVE-BLOCK-END
