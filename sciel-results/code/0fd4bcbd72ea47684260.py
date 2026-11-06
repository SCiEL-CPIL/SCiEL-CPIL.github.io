"""Design a 32x32 photonic crystal unit cell for maximum TM 4-5 band gap."""
import time
import math
import numpy as np
from scipy.optimize import minimize

def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell maximizing TM 4-5 band gap."""
    start_time = time.time()
    deadline = start_time + 220.0  # 10s safety margin
    
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Precompute pixel coordinates (normalized to [-0.5, 0.5])
    coords = (np.arange(size) + 0.5) / size - 0.5
    x_grid, y_grid = np.meshgrid(coords, coords, indexing='ij')
    
    # Define the symmetry group for a 32x32 cell (must be invariant under:
    # 1. 180° rotation: cell == cell[::-1] row-wise and column-wise
    # 2. Transpose: cell == cell.T
    # This means the cell has D2 symmetry (rectangular symmetry)
    # Actually, re-reading: "invariant under the square symmetry group about the cell centre"
    # and "cell == cell[::-1] row-wise and column-wise and cell == its transpose"
    # So we need: cell[i][j] == cell[31-i][31-j] (180° rot) and cell[i][j] == cell[j][i] (transpose)
    
    # The fundamental domain for these symmetries:
    # 180° rotation: (i,j) -> (31-i, 31-j)
    # Transpose: (i,j) -> (j,i)
    # Combined: (i,j) -> (31-j, 31-i)
    # Fundamental domain: i <= j and i <= 31-i (i.e., i <= 15.5, so i <= 15)
    # and j <= 31-i (i.e., i+j <= 31)
    # Actually let me think more carefully.
    # Symmetries: S1: (i,j) -> (31-i, 31-j), S2: (i,j) -> (j,i)
    # Orbit of (i,j): {(i,j), (31-i,31-j), (j,i), (31-j,31-i)}
    # Fundamental domain: pick one from each orbit.
    # We can use: i < j and i < 31-i (i <= 15) and j < 31-i (i+j <= 30)
    # Or more simply: i <= j and i <= 15 and i+j <= 31
    
    # Let's enumerate the fundamental domain
    fundamental_pairs = []
    for i in range(size):
        for j in range(size):
            # Check if this is the canonical representative of its orbit
            orbit = [(i, j), (size-1-i, size-1-j), (j, i), (size-1-j, size-1-i)]
            # Canonical: smallest in lexicographic order
            canonical = min(orbit)
            if (i, j) == canonical:
                fundamental_pairs.append((i, j))
    
    n_params = len(fundamental_pairs)
    
    def params_to_cell(params):
        """Convert binary parameters to a symmetric 32x32 cell."""
        cell = np.zeros((size, size), dtype=np.int32)
        for idx, (i, j) in enumerate(fundamental_pairs):
            val = 1 if params[idx] > 0.5 else 0
            # Apply all symmetries
            orbit = [(i, j), (size-1-i, size-1-j), (j, i), (size-1-j, size-1-i)]
            for (oi, oj) in orbit:
                cell[oi, oj] = val
        return cell
    
    def compute_1over_eps_fourier(cell):
        """Compute Fourier coefficients of 1/epsilon for the pixel map."""
        # 1/eps: eps=12 where cell=1, eps=1 where cell=0
        inv_eps = np.where(cell == 1, 1.0/eps_high, 1.0/eps_low)
        
        # Compute Fourier coefficients for the plane-wave expansion
        # The lattice constant is 1, so reciprocal lattice vectors are (m, n) with m,n integers
        # f_{mn} = sum_{i,j} inv_eps[i,j] * exp(-2*pi*i*(m*i + n*j)/size) / size^2
        # But we need to be careful with the grid spacing
        
        # Using the discrete Fourier transform
        # The unit cell has `size` pixels, each of size 1/size
        # f_{mn} = (1/size^2) * sum_{i,j} inv_eps[i,j] * exp(-2*pi*(m*i + n*j)/size)
        
        # Use numpy FFT
        fft_coeff = np.fft.fft2(inv_eps) / (size * size)
        
        # fft_coeff[m, n] gives f_{m-n, ...} due to numpy's convention
        # Actually np.fft.fft2 gives F[k] = sum f[n] exp(-2*pi*i*k*n/N)
        # So fft_coeff[m, n] = (1/N^2) sum_{i,j} f[i,j] exp(-2*pi*i*(m*i + n*j)/N)
        # But with the negative sign in the exponent, this is the coefficient for exp(+2*pi*i*(m*i+n*j)/N)
        # Wait, let me be more careful.
        # f(x) = sum_{mn} f_{mn} exp(2*pi*i*(mx+ny))
        # f_{mn} = integral f(x) exp(-2*pi*i*(mx+ny)) dxdy
        # Discretized: f_{mn} ≈ (1/N^2) sum_{i,j} f[i,j] exp(-2*pi*i*(m*i/N + n*j/N))
        # numpy fft2: F[m,n] = sum_{i,j} f[i,j] exp(-2*pi*i*(m*i + n*j)/N)
        # So f_{mn} = F[m,n] / N^2
        # But numpy's fft2 uses the convention with indices 0..N-1, and for negative frequencies
        # we wrap around: f_{-m} = f_{N-m}
        
        # For the plane-wave method, we need f_{mn} for various (m,n)
        # The coefficient for wavevector (m,n) where m,n can be negative:
        # f_{mn} = fft_coeff[m % size, n % size]
        
        return fft_coeff
    
    def band_structure_tm(cell, n_pw=17, n_k=30):
        """Compute TM band structure using plane-wave method.
        
        For TM polarization, the eigenvalue equation is:
        sum_{G'} |G+G'|^2 * f_{G-G'} * c_{G'} = omega^2 * c_G
        
        where f_{G} are Fourier coefficients of 1/epsilon, and |G+G'|^2 is in units of (2*pi/a)^2.
        
        We use a set of reciprocal lattice vectors within a circle of radius n_pw * 2*pi.
        """
        # Generate reciprocal lattice vectors
        # G = (m, n) * 2*pi, for m, n in range(-n_pw, n_pw+1)
        # Filter to those with |G| <= some cutoff
        
        # Actually, let's use a cutoff radius in reciprocal space
        # The problem says 289 plane waves, which is 17^2, so n_pw = 17 (m,n from -8 to 8)
        # Wait, 289 = 17^2, so we have 17x17 = 289 plane waves
        # That means m, n from -8 to 8 (17 values each)
        
        n_pw = 17  # as specified: 289 plane waves
        m_vals = np.arange(-n_pw//2, n_pw//2 + 1)  # -8 to 8
        g_vectors = [(m, n) for m in m_vals for n in m_vals]
        n_g = len(g_vectors)
        
        # Precompute 1/epsilon Fourier coefficients
        fft_coeff = compute_1over_eps_fourier(cell)
        
        def get_f(m, n):
            """Get Fourier coefficient f_{mn}."""
            # Handle negative indices by wrapping
            mi = m % size
            ni = n % size
            return fft_coeff[mi, ni]
        
        # Build the eigenvalue matrix
        # For TM: A[G, G'] = |G - G'|^2 * f_{G-G'}
        # Wait, let me reconsider the TM eigenvalue equation.
        
        # For TM polarization (E perpendicular to the plane, i.e., E_z):
        # The wave equation gives:
        # sum_{G'} |k+G'|^2 * (1/eps)_{G-G'} * E_{G'} = omega^2/c^2 * E_G
        # In dimensionless units (a=1, c=1):
        # sum_{G'} |k+G'|^2 * f_{G-G'} * E_{G'} = omega^2 * E_G
        # where f_{G} = Fourier coefficient of 1/epsilon
        
        # So the matrix element is: M[G, G'] = |k+G'|^2 * f_{(G-G')}
        # where G and G' are reciprocal lattice vectors
        
        # Define k-points in the irreducible BZ of a square lattice
        # The BZ is [-pi, pi] x [-pi, pi] (with a=1, so 2*pi/a = 2*pi)
        # Wait, for a square lattice with constant a=1, the reciprocal lattice vectors are (2*pi*m, 2*pi*n)
        # The BZ is [-pi, pi] x [-pi, pi]
        # The irreducible BZ (1/8 of the full BZ by symmetry) is the triangle with vertices
        # (0,0), (pi/2, pi/2), (pi, 0) -- no wait, for a square lattice the high-symmetry points are:
        # Gamma = (0,0), M = (pi, 0) or (0, pi), X = (pi/2, pi/2)
        # Actually for a square lattice: Gamma=(0,0), X=(pi,0), M=(pi/2,pi/2)
        # The irreducible wedge is the triangle Gamma-X-M (1/8 of BZ)
        
        # Let's use k-points along the path Gamma -> X -> M -> Gamma
        # But the problem says 30 wave vectors covering boundary and interior
        
        # Let's generate k-points in the irreducible BZ
        # Irreducible BZ: 0 <= ky <= kx, kx <= pi, ky >= 0, and kx - ky <= pi (for the triangle)
        # Actually the standard irreducible BZ for square lattice is:
        # 0 <= kx <= pi, 0 <= ky <= kx, kx + ky <= pi... no
        # Let me just use: 0 <= ky <= kx, 0 <= kx <= pi, and kx <= pi/2 + ky... 
        # The irreducible BZ is the 1/8 sector: 0 <= ky <= kx <= pi, with the constraint that
        # we're inside the BZ. The BZ boundary is at kx=pi or ky=pi.
        # So: 0 <= ky <= kx <= pi, and kx <= pi (always true in this range)
        # But we also need kx + ky <= pi? No, that's not right for a square BZ.
        
        # The square BZ is [-pi,pi] x [-pi,pi]. By symmetry (rotations and reflections),
        # the irreducible BZ is the triangle with vertices (0,0), (pi,0), (pi/2, pi/2).
        # So: 0 <= ky <= kx, 0 <= kx <= pi, and kx + ky <= pi... 
        # Wait, the line from (pi,0) to (pi/2,pi/2) is kx + ky = pi.
        # So the irreducible BZ is: 0 <= ky <= kx, kx + ky <= pi.
        
        # Generate k-points
        k_points = []
        # Along Gamma-X: (t*pi, 0) for t in [0,1]
        for t in np.linspace(0, 1, 10):
            k_points.append((t * np.pi, 0.0))
        # Along X-M: (pi - t*pi/2, t*pi/2) for t in [0,1]
        for t in np.linspace(0, 1, 10):
            k_points.append((np.pi - t * np.pi / 2, t * np.pi / 2))
        # Along M-Gamma: (t*pi/2, t*pi/2) for t in [0,1]
        for t in np.linspace(0, 1, 10):
            k_points.append((t * np.pi / 2, t * np.pi / 2))
        
        # Remove duplicates
        k_points = list(set(k_points))
        
        # Compute bands at each k-point
        all_omegas = []
        for kx, ky in k_points:
            k = np.array([kx, ky])
            
            # Build the matrix M where M[i,j] = |k + G_j|^2 * f_{G_i - G_j}
            M = np.zeros((n_g, n_g), dtype=np.complex128)
            for i, (mi, ni) in enumerate(g_vectors):
                Gi = np.array([2*np.pi*mi, 2*np.pi*ni])
                for j, (mj, nj) in enumerate(g_vectors):
                    Gj = np.array([2*np.pi*mj, 2*np.pi*nj])
                    kgj = k + Gj
                    kgj_sq = kgj[0]**2 + kgj[1]**2
                    # f_{Gi - Gj}
                    dm = mi - mj
                    dn = ni - nj
                    f_val = get_f(dm, dn)
                    M[i, j] = kgj_sq * f_val
            
            # Solve eigenvalue problem
            # M is Hermitian if f is real and symmetric, but let's use general eigensolver
            try:
                eigvals = np.linalg.eigvalsh(M)
                eigvals = np.sort(eigvals)
                # omega^2 = eigvals, so omega = sqrt(eigvals)
                # Filter out negative eigenvalues (numerical artifacts)
                eigvals = np.maximum(eigvals, 0)
                omegas = np.sqrt(eigvals)
                all_omegas.append(omegas)
            except:
                continue
        
        if not all_omegas:
            return 0.0
        
        # Find the 4th and 5th bands
        # For each k-point, we have bands sorted by omega
        # omega_4(k) = 4th smallest, omega_5(k) = 5th smallest
        # max_omega_4 = max over k of omega_4(k)
        # min_omega_5 = min over k of omega_5(k)
        
        max_omega_4 = 0.0
        min_omega_5 = np.inf
        
        for omegas in all_omegas:
            if len(omegas) >= 5:
                omega_4 = omegas[3]  # 0-indexed, so 4th band
                omega_5 = omegas[4]  # 5th band
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == np.inf:
            return 0.0
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap <= 0:
            return 0.0
        
        return 2.0 * gap / midgap
    
    # Initial guess: four-rod pattern at (±a/4, ±a/4)
    def initial_params():
        """Create initial parameters for the four-rod pattern."""
        params = np.zeros(n_params)
        for idx, (i, j) in enumerate(fundamental_pairs):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            # Four rods at (±0.25, ±0.25) with radius 0.15
            for cx, cy in [(0.25, 0.25), (-0.25, 0.25), (0.25, -0.25), (-0.25, -0.25)]:
                if (x - cx)**2 + (y - cy)**2 < 0.15**2:
                    params[idx] = 1.0
                    break
        return params
    
    # Objective function (negate for minimization)
    def objective(params):
        if time.time() > deadline - 5:
            return 0.0
        cell = params_to_cell(params)
        return -band_structure_tm(cell)
    
    # Run optimization
    best_score = -np.inf
    best_params = None
    
    # Try initial four-rod pattern
    p0 = initial_params()
    cell0 = params_to_cell(p0)
    score0 = -objective(p0)
    best_score = score0
    best_params = p0.copy()
    
    # Use Nelder-Mead optimization on the continuous parameters
    # But our parameters are binary (0 or 1), so let's use a different approach
    
    # Actually, let's use a local search / hill climbing on the binary parameters
    # Try flipping individual pixels and see if it improves the score
    
    current_params = p0.copy()
    current_score = score0
    
    max_iterations = 500
    iteration = 0
    
    while iteration < max_iterations:
        if time.time() > deadline - 10:
            break
        
        improved = False
        # Try flipping each parameter
        for idx in range(n_params):
            if time.time() > deadline - 10:
                break
            
            old_val = current_params[idx]
            current_params[idx] = 1.0 - old_val
            
            new_score = -objective(current_params)
            
            if new_score > current_score + 1e-6:
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_params = current_params.copy()
            else:
                current_params[idx] = old_val
        
        iteration += 1
        
        if not improved:
            # Try random restarts
            if time.time() > deadline - 15:
                break
            
            # Try a few random initializations
            for restart in range(3):
                if time.time() > deadline - 15:
                    break
                
                # Random sparse pattern
                p_rand = np.random.random(n_params)
                p_rand[p_rand < 0.7] = 0  # Sparse
                p_rand[p_rand >= 0.7] = 1
                
                # Ensure symmetry is maintained (it is, by construction)
                cell_rand = params_to_cell(p_rand)
                score_rand = -objective(p_rand)
                
                if score_rand > current_score:
                    current_params = p_rand.copy()
                    current_score = score_rand
                    if current_score > best_score:
                        best_score = current_score
                        best_params = current_params.copy()
                    improved = True
        
        if not improved and time.time() > deadline - 20:
            break
    
    # Also try some structured patterns
    structured_patterns = [
        # Single large rod
        lambda i, j: (x_grid[i,j]**2 + y_grid[i,j]**2 < 0.25**2),
        # Four rods, larger
        lambda i, j: any((x_grid[i,j]-cx)**2 + (y_grid[i,j]-cy)**2 < 0.18**2 
                         for cx, cy in [(0.25,0.25),(-0.25,0.25),(0.25,-0.25),(-0.25,-0.25)]),
        # Four rods, smaller
        lambda i, j: any((x_grid[i,j]-cx)**2 + (y_grid[i,j]-cy)**2 < 0.12**2 
                         for cx, cy in [(0.25,0.25),(-0.25,0.25),(0.25,-0.25),(-0.25,-0.25)]),
        # Checkerboard-like
        lambda i, j: ((i // 4 + j // 4) % 2 == 0) and (x_grid[i,j]**2 + y_grid[i,j]**2 < 0.4**2),
        # Cross pattern
        lambda i, j: (abs(x_grid[i,j]) < 0.1 or abs(y_grid[i,j]) < 0.1) and (x_grid[i,j]**2 + y_grid[i,j]**2 < 0.35**2),
    ]
    
    for pattern_func in structured_patterns:
        if time.time() > deadline - 10:
            break
        
        # Apply pattern and enforce symmetry
        raw_cell = pattern_func(np.arange(size)[:, None], np.arange(size)[None, :])
        # Enforce symmetry
        sym_cell = np.zeros((size, size), dtype=np.int32)
        for i in range(size):
            for j in range(size):
                orbit = [(i, j), (size-1-i, size-1-j), (j, i), (size-1-j, size-1-i)]
                # Use the canonical representative
                val = 0
                for (oi, oj) in orbit:
                    if raw_cell[oi, oj]:
                        val = 1
                        break
                sym_cell[i, j] = val
        
        # Convert to params
        p_struct = np.zeros(n_params)
        for idx, (i, j) in enumerate(fundamental_pairs):
            p_struct[idx] = sym_cell[i, j]
        
        score_struct = -objective(p_struct)
        if score_struct > best_score:
            best_score = score_struct
            best_params = p_struct.copy()
    
    # Return the best cell
    best_cell = params_to_cell(best_params)
    
    # Convert to nested list
    cell_list = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(best_cell[i, j]))
        cell_list.append(row)
    
    return cell_list
