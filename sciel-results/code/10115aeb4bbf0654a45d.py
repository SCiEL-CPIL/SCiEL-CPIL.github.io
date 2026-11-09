"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    Strategy:
    1. Represent the dielectric field as a binary grid (0/1) with C4v symmetry.
    2. Use a fast approximate plane-wave method (17x17 = 289 PWs) to evaluate the gap.
    3. Optimize using simulated annealing on the D4-symmetric fundamental domain.
    4. Explore different structural families: central rod, air holes, crosses.
    """
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Time management
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin for 200s limit

    # --- 1. Precompute reciprocal lattice vectors ---
    # Use 17x17 = 289 plane waves (h, k in [-8, 8])
    pw_range = 8
    G_vectors = []
    for h in range(-pw_range, pw_range + 1):
        for k in range(-pw_range, pw_range + 1):
            G_vectors.append((h, k))
    N_pw = len(G_vectors)
    
    # Precompute G vector indices for fast lookup
    G_index = {}
    for idx, (h, k) in enumerate(G_vectors):
        G_index[(h, k)] = idx
    
    # --- 2. Fast Evaluator ---
    # Precompute k-points in the irreducible Brillouin zone
    # Square lattice BZ: 0 <= kx <= ky, kx + ky <= 1 (in units of 2pi/a)
    # Actually, the irreducible BZ for square lattice is:
    # 0 <= kx <= ky, kx + ky <= 0.5 (if we use units where G = 2pi/a)
    # Let's use fractional coordinates where the reciprocal lattice vector is 1.
    # Then the BZ is [-0.5, 0.5] x [-0.5, 0.5], and the irreducible part is:
    # 0 <= kx <= ky, kx + ky <= 0.5
    
    k_points = []
    # Sample points along high-symmetry lines and interior
    # Gamma (0,0), M (0.5, 0), X (0.5, 0.5)
    # Lines: Gamma-M, M-X, Gamma-X
    
    # Gamma to M: (0, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 8):
        k_points.append((0.0, t))
    
    # M to X: (t, 0.5-t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 8):
        k_points.append((t, 0.5 - t))
    
    # Gamma to X: (t, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 8):
        k_points.append((t, t))
    
    # Add some interior points
    for i in range(3):
        for j in range(3):
            kx = 0.1 * i
            ky = 0.1 * j
            if kx <= ky and kx + ky <= 0.5:
                k_points.append((kx, ky))
    
    # Remove duplicates and sort
    k_points = list(set([(round(kx, 6), round(ky, 6)) for kx, ky in k_points]))
    
    def evaluate_gap(cell):
        """
        Computes the approximate band gap between bands 4 and 5.
        cell is a 32x32 binary array (0=air, 1=dielectric).
        Returns the gap-midgap ratio.
        """
        # Compute epsilon grid
        eps_grid = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_grid
        
        # Compute FFT of 1/eps
        fft_1eps = np.fft.fft2(inv_eps)
        
        # Normalize: FFT gives sum, but we need the integral (divide by N^2)
        # Actually, for the plane-wave method, we need the Fourier coefficients:
        # f_G = (1/N^2) * sum_{i,j} f(i,j) * exp(-i G . r_ij)
        # numpy.fft.fft2 gives sum without the 1/N^2 factor.
        # So we divide by N^2.
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            # Build the Hamiltonian matrix
            # H_{ij} = (G_i + k) dot (G_j + k) * <1/eps>_{G_i - G_j}
            
            H = np.zeros((N_pw, N_pw), dtype=complex)
            
            for i, (hi, ki) in enumerate(G_vectors):
                Kx_i = hi + kx
                Ky_i = ki + ky
                
                for j, (hj, kj) in enumerate(G_vectors):
                    Kx_j = hj + kx
                    Ky_j = kj + ky
                    
                    dot = Kx_i * Kx_j + Ky_i * Ky_j
                    
                    dh = hi - hj
                    dk = ki - kj
                    
                    # Get Fourier coefficient
                    idx_h = dh % size
                    idx_k = dk % size
                    coeff = fft_1eps[idx_h, idx_k] / (size * size)
                    
                    H[i, j] = dot * coeff
            
            # H should be Hermitian. For real symmetric eps, it's real symmetric.
            # Let's take the real part and solve.
            H_real = H.real
            
            # Solve eigenvalue problem
            eigenvalues = np.linalg.eigvalsh(H_real)
            
            if len(eigenvalues) >= 5:
                w4_sq = eigenvalues[3]
                w5_sq = eigenvalues[4]
                
                w4 = np.sqrt(max(0, w4_sq))
                w5 = np.sqrt(max(0, w5_sq))
                
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
        
        if min_omega_5 == float('inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = 2 * gap / midgap
        return ratio

    # --- 3. Symmetry Enforcement ---
    # The cell must be invariant under C4v symmetry.
    # We work on the fundamental domain and reflect.
    
    def enforce_symmetry(cell):
        """
        Enforces C4v symmetry on the 32x32 cell.
        The fundamental domain is the upper-right triangle (including diagonals).
        """
        sym_cell = np.zeros_like(cell)
        
        for i in range(size):
            for j in range(i, size):  # Only upper triangle
                val = cell[i, j]
                
                # Apply all 8 symmetry operations
                # (i, j)
                sym_cell[i, j] = val
                # (j, i) - transpose
                sym_cell[j, i] = val
                # (size-1-i, j) - reflect x
                sym_cell[size-1-i, j] = val
                # (j, size-1-i) - transpose of reflect x
                sym_cell[j, size-1-i] = val
                # (i, size-1-j) - reflect y
                sym_cell[i, size-1-j] = val
                # (size-1-j, i) - transpose of reflect y
                sym_cell[size-1-j, i] = val
                # (size-1-i, size-1-j) - rotate 180
                sym_cell[size-1-i, size-1-j] = val
                # (size-1-j, size-1-i) - transpose of rotate 180
                sym_cell[size-1-j, size-1-i] = val
        
        return sym_cell

    # --- 4. Initial Candidates ---
    # Create several initial candidates with different structures
    
    def make_initial_candidate(idx):
        """Creates an initial candidate based on idx."""
        cell = np.zeros((size, size), dtype=int)
        
        if idx == 0:
            # Central rod (circular)
            center = size / 2.0
            radius = 8
            for i in range(size):
                for j in range(size):
                    dist = np.sqrt((i - center + 0.5)**2 + (j - center + 0.5)**2)
                    if dist < radius:
                        cell[i, j] = 1
        
        elif idx == 1:
            # Square rod
            margin = 8
            for i in range(margin, size - margin):
                for j in range(margin, size - margin):
                    cell[i, j] = 1
        
        elif idx == 2:
            # Air holes (inverted: dielectric background with air holes)
            cell[:] = 1
            center = size / 2.0
            radius = 4
            for i in range(size):
                for j in range(size):
                    dist = np.sqrt((i - center + 0.5)**2 + (j - center + 0.5)**2)
                    if dist < radius:
                        cell[i, j] = 0
            # Also add holes at corners
            for cx in [4, size-5]:
                for cy in [4, size-5]:
                    for i in range(cx-2, cx+3):
                        for j in range(cy-2, cy+3):
                            if 0 <= i < size and 0 <= j < size:
                                dist = np.sqrt((i - cx + 0.5)**2 + (j - cy + 0.5)**2)
                                if dist < 2.5:
                                    cell[i, j] = 0
        
        elif idx == 3:
            # Cross shape
            center = size // 2
            arm_width = 4
            arm_length = 12
            for i in range(size):
                for j in range(size):
                    # Horizontal arm
                    if abs(i - center + 0.5) < arm_width/2 and abs(j - center + 0.5) < arm_length/2:
                        cell[i, j] = 1
                    # Vertical arm
                    elif abs(j - center + 0.5) < arm_width/2 and abs(i - center + 0.5) < arm_length/2:
                        cell[i, j] = 1
        
        elif idx == 4:
            # Four small rods
            positions = [(8, 8), (8, 23), (23, 8), (23, 23)]
            radius = 4
            for cx, cy in positions:
                for i in range(size):
                    for j in range(size):
                        dist = np.sqrt((i - cx + 0.5)**2 + (j - cy + 0.5)**2)
                        if dist < radius:
                            cell[i, j] = 1
        
        else:
            # Random
            cell = np.random.randint(0, 2, (size, size))
        
        # Enforce symmetry
        cell = enforce_symmetry(cell)
        
        return cell

    # --- 5. Optimization Loop (Simulated Annealing) ---
    
    # Start with the best initial candidate
    best_cell = None
    best_score = -float('inf')
    
    # Evaluate a few initial candidates
    for idx in range(5):
        if time.time() > deadline:
            break
        cell = make_initial_candidate(idx)
        score = evaluate_gap(cell)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
    
    # If no good initial candidate, use a random one
    if best_cell is None:
        best_cell = make_initial_candidate(5)
        best_score = evaluate_gap(best_cell)
    
    current_cell = best_cell.copy()
    current_score = best_score
    
    # Simulated annealing parameters
    T = 1.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.999
    
    num_iterations = 0
    max_iterations = 100000
    
    while time.time() < deadline and num_iterations < max_iterations:
        num_iterations += 1
        
        # Pick a random pixel in the fundamental domain to flip
        i = np.random.randint(0, size)
        j = np.random.randint(i, size)  # Upper triangle
        
        # Flip the pixel and its symmetric equivalents
        old_val = current_cell[i, j]
        new_val = 1 - old_val
        
        # Apply to all symmetric positions
        positions = [
            (i, j),
            (j, i),
            (size-1-i, j),
            (j, size-1-i),
            (i, size-1-j),
            (size-1-j, i),
            (size-1-i, size-1-j),
            (size-1-j, size-1-i)
        ]
        
        # Remove duplicates
        positions = list(set(positions))
        
        # Apply flip
        for pi, pj in positions:
            current_cell[pi, pj] = new_val
        
        # Evaluate new score
        new_score = evaluate_gap(current_cell)
        
        # Accept or reject
        delta = new_score - current_score
        if delta > 0 or np.random.random() < np.exp(delta / max(T, 1e-10)):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_cell = current_cell.copy()
        else:
            # Revert
            for pi, pj in positions:
                current_cell[pi, pj] = old_val
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # --- 6. Finalize Design ---
    final_cell = best_cell
    
    # Ensure exact symmetry
    final_cell = enforce_symmetry(final_cell)
    
    # Convert to list of lists
    return final_cell.tolist()

# EVOLVE-BLOCK-END
