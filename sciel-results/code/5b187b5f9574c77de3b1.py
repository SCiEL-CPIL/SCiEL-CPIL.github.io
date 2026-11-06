import math
import time
import numpy as np

def design(problem):
    """
    Design a 32x32 unit cell with C4v symmetry to maximize the 4-5 TM band gap.
    Uses a local search on geometric parameters with a fast surrogate band solver.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12.0)
    eps_low = problem.get("eps_low", 1.0)
    
    # Pre-compute inverse epsilon values
    inv_eps_high = 1.0 / eps_high
    inv_eps_low = 1.0 / eps_low
    
    def create_cell_from_params(param_type, r, d=0.0):
        """
        Create a 32x32 cell.
        param_type: 'central_hole' (0 center, 1 bg), 'central_rod' (1 center, 0 bg), 
                    'four_holes' (0 at +-d, 1 bg), 'four_rods' (1 at +-d, 0 bg)
        r: radius of feature
        d: offset from center for 4-feature designs
        """
        cell = [[0 for _ in range(size)] for _ in range(size)]
        
        if param_type == 'central_hole':
            # Background is dielectric (1), hole is air (0)
            for i in range(size):
                y = (i + 0.5) / size - 0.5
                for j in range(size):
                    x = (j + 0.5) / size - 0.5
                    if x*x + y*y > r*r:
                        cell[i][j] = 1
                    else:
                        cell[i][j] = 0
        elif param_type == 'central_rod':
            # Background is air (0), rod is dielectric (1)
            for i in range(size):
                y = (i + 0.5) / size - 0.5
                for j in range(size):
                    x = (j + 0.5) / size - 0.5
                    if x*x + y*y < r*r:
                        cell[i][j] = 1
                    else:
                        cell[i][j] = 0
        elif param_type == 'four_holes':
            # Background is dielectric (1), holes are air (0)
            centers = [(-d, -d), (d, -d), (-d, d), (d, d)]
            for i in range(size):
                y = (i + 0.5) / size - 0.5
                for j in range(size):
                    x = (j + 0.5) / size - 0.5
                    inside = False
                    for cx, cy in centers:
                        if (x-cx)**2 + (y-cy)**2 < r*r:
                            inside = True
                            break
                    if inside:
                        cell[i][j] = 0
                    else:
                        cell[i][j] = 1
        elif param_type == 'four_rods':
            # Background is air (0), rods are dielectric (1)
            centers = [(-d, -d), (d, -d), (-d, d), (d, d)]
            for i in range(size):
                y = (i + 0.5) / size - 0.5
                for j in range(size):
                    x = (j + 0.5) / size - 0.5
                    inside = False
                    for cx, cy in centers:
                        if (x-cx)**2 + (y-cy)**2 < r*r:
                            inside = True
                            break
                    if inside:
                        cell[i][j] = 1
                    else:
                        cell[i][j] = 0
        return cell

    def compute_fourier_chips(cell):
        """
        Compute Fourier coefficients of 1/epsilon for the cell.
        Returns a dictionary mapping (gx, gy) to value.
        """
        # Convert cell to numpy array
        arr = np.array(cell, dtype=float)
        
        # 1/epsilon map
        # 1 -> inv_eps_high, 0 -> inv_eps_low
        chi = np.where(arr == 1, inv_eps_high, inv_eps_low)
        
        # FFT
        # The grid is 32x32.
        # Fourier coefficients c_G = (1/N^2) sum_{i,j} chi_{i,j} exp(-i G . r_{i,j})
        # G = (2*pi*m/size, 2*pi*n/size)
        # Using numpy FFT: F = fft2(chi)
        # c_G = F[G] / (size * size)
        
        F = np.fft.fft2(chi)
        # Normalize
        F = F / (size * size)
        
        # We need coefficients for G vectors.
        # The evaluator uses specific G vectors. We will compute them on the fly in the solver.
        # Return the full FFT array for easy access.
        return F

    def solve_bands_fast(F, kx, ky, N_pw=101):
        """
        Solve for TM bands using plane wave expansion.
        F: 2D array of Fourier coefficients of 1/eps.
        kx, ky: wave vector in BZ (radians).
        N_pw: number of plane waves in each direction (odd number for symmetry).
        """
        # Select G vectors
        # G_mn = (2*pi*m/size, 2*pi*n/size)
        # We need |G| < cutoff.
        # Let's select a circular cutoff.
        max_g = 8 # Max order of G
        g_vectors = []
        for m in range(-max_g, max_g+1):
            for n in range(-max_g, max_g+1):
                g = (2*math.pi*m/size, 2*math.pi*n/size)
                # Check if within cutoff
                # Cutoff radius approx N_pw * 2*pi/size ?
                # Actually, just use a square box of G vectors for simplicity and speed
                # The evaluator uses 289 PWs, which is 17x17.
                # Let's use 15x15 for speed, or 17x17 to match.
                # 17x17 = 289.
                # Let's use 15x15 = 225 for speed in search.
                g_vectors.append(g)
                
        # Sort by magnitude? Not strictly necessary for eigensolver, but good for conditioning.
        # We just need the matrix D.
        
        N = len(g_vectors)
        D = np.zeros((N, N), dtype=complex)
        
        for i, (gx, gy) in enumerate(g_vectors):
            kx_g = kx + gx
            ky_g = ky + gy
            k_sq = kx_g**2 + ky_g**2
            
            for j, (gx_j, gy_j) in enumerate(g_vectors):
                # G_diff = G_i - G_j
                gm = gx - gx_j
                gn = gy - gy_j
                
                # Get Fourier coefficient chi(G_diff)
                # F[m, n] corresponds to G = (2*pi*m/size, 2*pi*n/size)
                # m, n can be negative.
                # numpy fft2 output index:
                # F[m, n] for m in 0..N-1.
                # To get negative m: m_idx = m % N.
                m_idx = int(round(gm / (2*math.pi/size))) % size
                n_idx = int(round(gn / (2*math.pi/size))) % size
                
                chi_diff = F[m_idx, n_idx]
                
                D[i, j] = k_sq * chi_diff
        
        # Solve eigenvalue problem: D A = omega^2 A
        # omega^2 are eigenvalues of D
        # D is Hermitian? 
        # D_ij = (k+Gi)^2 chi(Gi-Gj)
        # chi is real and symmetric (chi(G) = chi(-G)^*).
        # (k+Gi)^2 is real.
        # So D should be Hermitian if chi is real.
        # 1/epsilon is real. So F is real and symmetric (F[m,n] = F[-m,-n]).
        # So D is Hermitian.
        
        try:
            eigvals = np.linalg.eigvalsh(D)
        except:
            return []
        
        # omega = sqrt(eigval)
        # Filter out negative eigenvalues (numerical noise)
        eigvals = np.maximum(eigvals, 0)
        omegas = np.sqrt(eigvals)
        
        # Sort omegas
        omegas = np.sort(omegas)
        
        # We need the 4th and 5th bands.
        # Index 3 and 4 (0-indexed).
        if len(omegas) >= 5:
            return omegas[3], omegas[4]
        return 0, 0

    def evaluate_gap(cell):
        """
        Evaluate the gap ratio for a given cell.
        """
        F = compute_fourier_chips(cell)
        
        # Sample wave vectors in the irreducible BZ
        # Irreducible BZ of square lattice: 0 <= ky <= kx, kx+ky <= pi
        # Points: Gamma(0,0), M(pi,0), X(pi/2, pi/2) ?
        # Let's sample a few points.
        # Gamma: (0,0)
        # M: (pi, 0)
        # X: (pi/2, pi/2)
        # Midpoint Gamma-M: (pi/2, 0)
        # Midpoint Gamma-X: (pi/4, pi/4)
        
        k_points = [
            (0.0, 0.0),
            (math.pi, 0.0),
            (math.pi/2, math.pi/2),
            (math.pi/2, 0.0),
            (math.pi/4, math.pi/4),
            (3*math.pi/4, math.pi/4),
            (math.pi, math.pi/4) # Just in case
        ]
        
        max_omega_4 = 0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            w4, w5 = solve_bands_fast(F, kx, ky, N_pw=101)
            if w4 > 0 and w5 > 0:
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
        
        if min_omega_5 == float('inf') or max_omega_4 == 0:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = gap / midgap
        # The score is 2 * ratio
        return 2.0 * ratio

    # Search Strategy
    # Try a few initial designs and hill climb.
    
    best_score = -10.0
    best_cell = None
    
    # Initial candidates
    candidates = [
        ('central_hole', 0.2, 0.0),
        ('central_hole', 0.3, 0.0),
        ('central_rod', 0.2, 0.0),
        ('central_rod', 0.3, 0.0),
        ('four_holes', 0.1, 0.25),
        ('four_holes', 0.15, 0.25),
        ('four_rods', 0.1, 0.25),
        ('four_rods', 0.15, 0.25),
    ]
    
    for ptype, r, d in candidates:
        if time.time() > deadline:
            break
        cell = create_cell_from_params(ptype, r, d)
        score = evaluate_gap(cell)
        if score > best_score:
            best_score = score
            best_cell = cell
            # Local search around this point
            # Step size
            step_r = 0.02
            step_d = 0.02
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try variations
                for dr in [-step_r, step_r]:
                    for dd in [-step_d, step_d]:
                        new_r = r + dr
                        new_d = d + dd
                        # Constraints
                        if new_r < 0.05 or new_r > 0.4: continue
                        if ptype.startswith('four'):
                            if new_d < 0.1 or new_d > 0.4: continue
                        cell_test = create_cell_from_params(ptype, new_r, new_d)
                        score_test = evaluate_gap(cell_test)
                        if score_test > best_score:
                            best_score = score_test
                            best_cell = cell_test
                            r = new_r
                            d = new_d
                            improved = True
                if improved:
                    step_r *= 0.8
                    step_d *= 0.8
    
    # If no good result, fallback to a standard design
    if best_cell is None:
        best_cell = create_cell_from_params('central_hole', 0.25, 0.0)
        
    return best_cell
