import numpy as np
import time
from itertools import product

def design(problem):
    """
    Optimizes a discrete D4-symmetric binary mask using local search (simulated annealing)
    guided by a fast surrogate band-structure evaluator.
    """
    start_time = time.time()
    deadline = start_time + 215.0  # Safety margin before 230s kill
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # 1. Define the D4 symmetry orbits for the 32x32 grid
    # A pixel (i, j) is equivalent to 8 positions under D4 symmetry.
    # We only need to store/modify the "representative" of each orbit.
    # Representative: smallest (i, j) in the orbit.
    
    orbits = []
    visited = np.zeros((size, size), dtype=bool)
    
    for i in range(size):
        for j in range(size):
            if visited[i, j]:
                continue
            
            # Generate all 8 symmetric positions
            positions = [
                (i, j),
                (size-1-i, j),
                (i, size-1-j),
                (size-1-i, size-1-j),
                (j, i),
                (size-1-j, i),
                (j, size-1-i),
                (size-1-j, size-1-i)
            ]
            
            # Remove duplicates (for pixels on symmetry axes)
            unique_positions = []
            seen_pos = set()
            for p in positions:
                if p not in seen_pos:
                    unique_positions.append(p)
                    seen_pos.add(p)
            
            orbits.append(unique_positions)
            
            for p in unique_positions:
                visited[p[0], p[1]] = True
    
    num_orbits = len(orbits)
    
    # 2. Initialize the binary mask
    # Start with a pattern known to be a good starting point: 
    # A central dielectric square/rod.
    # Let's start with a 12x12 central square of dielectric (1s).
    # But we must respect the orbit structure.
    # It's easier to initialize the full grid, then extract orbit values.
    
    init_grid = np.zeros((size, size), dtype=int)
    # Center is at (15.5, 15.5). 
    # Let's make a square from 10 to 21 (12x12 pixels).
    init_grid[10:22, 10:22] = 1
    
    # Ensure strict symmetry in init_grid (should be symmetric by construction, but let's enforce)
    # Actually, the orbit representation guarantees symmetry if we just fill based on orbit value.
    
    orbit_values = np.zeros(num_orbits, dtype=int)
    for idx, positions in enumerate(orbits):
        # Take the value from the first position
        orbit_values[idx] = init_grid[positions[0][0], positions[0][1]]
    
    def get_grid_from_orbits(vals):
        """Reconstruct the full 32x32 grid from orbit values."""
        grid = np.zeros((size, size), dtype=int)
        for idx, positions in enumerate(orbits):
            for p in positions:
                grid[p[0], p[1]] = vals[idx]
        return grid
    
    # 3. Surrogate Evaluator
    # We approximate the band gap using a small number of plane waves and k-points.
    # The evaluator uses 289 PWs. We can use fewer for speed, e.g., 17x17=289 is the target, 
    # but maybe 9x9=81 or 13x13=169 is faster.
    # The task says "exact-coefficient plane-wave method... 289-plane-wave".
    # To be safe and accurate, let's stick to a reasonably accurate but fast solver.
    # However, running 289 PW eigenvalue decomposition for every flip in a local search 
    # might be too slow if we do many flips.
    # 
    # Alternative: Precompute the "sensitivity" of the bands to pixel changes?
    # That's complex. 
    # 
    # Let's try a fast solver with fewer PWs (e.g., 11x11=121) and see if it's fast enough.
    # If 1 evaluation takes ~0.1s, we can do ~1000 evaluations in 100s.
    # 121x121 matrix eigvalsh is fast.
    
    N_pw = 11  # 11x11 = 121 plane waves
    G_vals = np.arange(-N_pw//2, N_pw//2 + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)
    
    # K points: Irreducible BZ.
    # Gamma (0,0), M (0.5, 0), X (0.5, 0.5) and lines.
    # Let's use 10 points.
    k_points = [
        (0.0, 0.0),
        (0.1, 0.0),
        (0.2, 0.0),
        (0.3, 0.0),
        (0.4, 0.0),
        (0.5, 0.0),
        (0.5, 0.1),
        (0.5, 0.2),
        (0.5, 0.3),
        (0.5, 0.4),
        (0.5, 0.5),
        (0.25, 0.25),
        (0.1, 0.1),
        (0.3, 0.3),
        (0.4, 0.4)
    ]
    
    # Precompute G_diff for the fast solver
    G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]  # (num_pw, num_pw, 2)
    
    def evaluate_gap_fast(grid):
        """
        Fast evaluation of the band gap between band 4 and 5.
        Returns the ratio 2*(min_w5 - max_w4) / (min_w5 + max_w4).
        """
        inv_eps = 1.0 / (eps_low + grid * (eps_high - eps_low))
        
        # FFT of 1/epsilon
        fft_inv_eps = np.fft.fft2(inv_eps)
        fft_inv_eps /= (size * size)
        
        # Build the coefficient matrix C_ij = <G|1/eps|G'>
        # C_ij = fft_inv_eps[(G_i - G_j) mod size]
        # Note: FFT indices are 0..size-1. G vectors are integers.
        # We need to map G_diff to FFT indices.
        
        coeff_matrix = np.zeros((num_pw, num_pw), dtype=complex)
        for i in range(num_pw):
            for j in range(num_pw):
                dx = int(G_diff[i, j, 0])
                dy = int(G_diff[i, j, 1])
                ix = dx % size
                iy = dy % size
                coeff_matrix[i, j] = fft_inv_eps[ix, iy]
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in k_points:
            k = np.array([kx, ky])
            k_plus_G = k + G_vectors
            omega_sq = np.sum(k_plus_G ** 2, axis=1)
            
            # Matrix M_ij = C_ij * omega_sq_i
            # For TM, the eigenvalue problem is:
            # sum_j <G|1/eps|G'> (k+G')^2 c_j = omega^2 c_i
            # So M_ij = C_ij * omega_sq_j ? 
            # Let's check the standard formulation.
            # (k+G)^2 c_G - omega^2 sum_{G'} <G|1/eps|G'> c_{G'} = 0
            # sum_{G'} [ delta_{GG'} (k+G)^2 - omega^2 <G|1/eps|G'> ] c_{G'} = 0
            # This is a generalized eigenvalue problem or can be rewritten.
            # Usually, it's solved as:
            # M c = omega^2 c
            # where M_ij = <G_i|1/eps|G_j> * (k+G_j)^2 ?? No.
            
            # Let's use the standard non-Hermitian form often used in simple PW codes:
            # A c = omega^2 c
            # A_ij = <G_i|1/eps|G_j> * (k+G_i)^2
            # Wait, the operator is (k+G)^2 * (1/eps) * (k+G)^2? No.
            # Maxwell: curl curl E = omega^2 eps E
            # In Fourier: (k+G)^2 E_G - (k+G) x ((k+G) x E) ...
            # For TM (E_z): 
            # (k+G)^2 E_G = omega^2 sum_{G'} <G|eps|G'> E_{G'}
            # This is a generalized eigenvalue problem: A E = omega^2 B E
            # Where A_ij = delta_ij (k+G_i)^2
            # B_ij = <G_i|eps|G_j>
            
            # To solve A E = omega^2 B E, we can transform to:
            # B^{-1} A E = omega^2 E
            # Or use eig(A, B).
            # NumPy doesn't have eig(A,B) directly in a stable way for large N, 
            # but we can use scipy.linalg.eig or solve B^{-1}A.
            # Since B is Hermitian positive definite, we can use Cholesky.
            # Or, simpler: use the "effective" matrix formulation if we approximate.
            
            # Let's use the common approximation:
            # M_ij = <G_i|1/eps|G_j> * (k+G_i)^2
            # This assumes (k+G)^2 is local in G-space, which is an approximation.
            # A more accurate one is to solve the generalized problem.
            # Given time constraints, let's try the generalized eigenvalue problem using scipy.
            
            A = np.diag(omega_sq)
            B = coeff_matrix  # <G|eps|G'> ? No, coeff_matrix is <G|1/eps|G'>.
            # We need <G|eps|G'>. 
            # The relation is: 1/eps is in the differential operator.
            # The standard PW method for TM uses 1/eps in the matrix.
            # Equation: sum_{G'} <G|1/eps|G'> (k+G')^2 c_{G'} = omega^2 c_G
            # So M_ij = <G_i|1/eps|G_j> * (k+G_j)^2
            # Let's use this.
            
            M = coeff_matrix * omega_sq[np.newaxis, :]  # Multiply columns by omega_sq_j
            
            # M is not Hermitian. We need to find eigenvalues.
            # For stability, we can symmetrize or use complex eig.
            eigvals = np.linalg.eigvals(M)
            real_eigvals = np.sort(np.real(eigvals))
            
            if len(real_eigvals) >= 5:
                # Take the 4th and 5th smallest positive eigenvalues
                # Note: Bands are indexed 1-based in physics, 0-based in array.
                # Band 4 is index 3, Band 5 is index 4.
                omega_4 = np.sqrt(real_eigvals[3])
                omega_5 = np.sqrt(real_eigvals[4])
                
                if omega_4 > 0 and omega_5 > 0:
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

    # 4. Optimization Loop
    # Start with the initial orbit values
    current_orbits = orbit_values.copy()
    current_grid = get_grid_from_orbits(current_orbits)
    current_score = evaluate_gap_fast(current_grid)
    
    best_orbits = current_orbits.copy()
    best_score = current_score
    
    # Simulated Annealing / Random Local Search
    # We will randomly pick an orbit and flip its value (0->1 or 1->0).
    # Accept if score improves, or with some probability if it doesn't (to escape local minima).
    
    temp = 1.0
    cooling_rate = 0.99
    min_temp = 0.01
    
    max_steps = 5000  # Max number of flips to try
    
    for step in range(max_steps):
        if time.time() > deadline - 10:
            break
        
        # Pick a random orbit
        idx = np.random.randint(num_orbits)
        
        # Flip the value
        new_orbits = current_orbits.copy()
        new_orbits[idx] = 1 - new_orbits[idx]
        
        new_grid = get_grid_from_orbits(new_orbits)
        new_score = evaluate_gap_fast(new_grid)
        
        # Acceptance criterion
        if new_score > current_score:
            current_orbits = new_orbits
            current_score = new_score
            current_grid = new_grid
            
            if current_score > best_score:
                best_score = current_score
                best_orbits = current_orbits.copy()
        else:
            # Metropolis criterion
            delta = current_score - new_score
            if np.random.rand() < np.exp(-delta / temp):
                current_orbits = new_orbits
                current_score = new_score
                current_grid = new_grid
        
        # Cool down
        temp *= cooling_rate
        if temp < min_temp:
            temp = min_temp
    
    # 5. Final Refinement
    # Do a few rounds of single-flip improvements to polish the result.
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for idx in range(num_orbits):
            if time.time() > deadline - 5:
                break
            
            new_orbits = best_orbits.copy()
            new_orbits[idx] = 1 - new_orbits[idx]
            new_grid = get_grid_from_orbits(new_orbits)
            new_score = evaluate_gap_fast(new_grid)
            
            if new_score > best_score:
                best_score = new_score
                best_orbits = new_orbits
                improved = True
    
    # 6. Construct Final Output
    final_grid = get_grid_from_orbits(best_orbits)
    
    # Convert to nested list of 0/1
    final_cell = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(final_grid[i, j]))
        final_cell.append(row)
        
    return final_cell
