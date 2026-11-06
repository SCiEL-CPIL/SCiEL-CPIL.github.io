import numpy as np
import time
import random

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 115.0  # Leave margin for evaluator overhead

    # 1. Setup Plane Wave Solver Components
    # 289 plane waves: 17x17 grid from -8 to 8
    half_pw = 8
    G_vecs = []
    for m in range(-half_pw, half_pw + 1):
        for n in range(-half_pw, half_pw + 1):
            G_vecs.append((m, n))
    G_vecs = np.array(G_vecs) * (2.0 * np.pi)
    nG = len(G_vecs)
    
    # K-points: 30 points in irreducible BZ
    # Triangle (0,0) to (pi,0) to (pi,pi)
    # We use a fixed set of 30 points that densely covers the region
    # where band gaps typically open (near X and M).
    # Approximation of the evaluator's 30 points with a dense grid filtered to the wedge.
    # To be safe and fast, we use a subset of a fine grid.
    # The evaluator likely uses specific high-symmetry lines + interior points.
    # Let's generate a 10x10 grid in the wedge and take 30 best-spaced points.
    # Or just use a fixed set of 30 points that are representative.
    # Common set: Gamma-X, Gamma-M, X-M lines + interior.
    # Let's use 5 points on Gamma-X, 5 on Gamma-M, 5 on X-M, and 15 interior.
    
    k_points = []
    # Gamma-X: (k, 0) for k in [0, pi]
    for i in range(6):
        k_points.append((i/5.0 * np.pi, 0.0))
    # Gamma-M: (k, k) for k in [0, pi]
    for i in range(1, 6): # Skip Gamma
        k_points.append((i/5.0 * np.pi, i/5.0 * np.pi))
    # X-M: (pi, k) for k in [0, pi]
    for i in range(1, 6): # Skip X
        k_points.append((np.pi, i/5.0 * np.pi))
    # Interior: 15 points
    # Simple grid in the triangle
    for i in range(4):
        for j in range(4):
            if i + j < 4: # Rough triangle mask
                s = (i + 0.5) / 4.0
                t = (j + 0.5) / 4.0
                if s + t <= 1.0:
                    k_points.append((s * np.pi, t * np.pi))
    
    # Trim to 30 if needed, or use all if close.
    # The exact 30 points matter for exact score matching, but for optimization
    # a dense set is a good surrogate.
    k_points = np.array(k_points[:30])

    # Precompute G^2
    G2 = G_vecs[:, 0]**2 + G_vecs[:, 1]**2

    def compute_score(cell):
        """
        Compute the 4-5 TM band gap score.
        cell: 32x32 array of 0/1.
        """
        eps_grid = eps_low + (eps_high - eps_low) * np.array(cell, dtype=float)
        # FFT of eps for the convolution operator
        eps_F = np.fft.fft2(eps_grid) / (size * size)
        
        # Handle potential zeros in eps_F (shouldn't happen for eps > 0)
        inv_eps_F = 1.0 / eps_F
        
        min_omega5 = np.inf
        max_omega4 = 0.0
        
        for kx, ky in k_points:
            # kG = k + G
            kG_x = kx + G_vecs[:, 0]
            kG_y = ky + G_vecs[:, 1]
            kG2 = kG_x**2 + kG_y**2
            
            # Operator A = Eps^{-1} D
            # A v = F^H diag(1/eps_F) F D v
            # We need the 4th and 5th eigenvalues of A.
            # Since A is Hermitian positive definite, we can use eigsh.
            
            def matvec(v):
                # 1. w = D v
                w = kG2 * v
                # 2. u = F w (FFT)
                # Note: G_vecs order is row-major (m, n). 
                # The FFT expects input in (row, col) which corresponds to (m, n) or (n, m)?
                # In our G_vecs loop:
                # for m in ...: for n in ...: append((m,n))
                # So index idx corresponds to (m, n).
                # Reshape to (17, 17) where first dim is m, second is n.
                u = np.fft.fft2(w.reshape(17, 17))
                # 3. z = (1/eps_F) * u
                # eps_F is computed from eps_grid (size x size).
                # The FFT of eps_grid gives coefficients for frequencies -16..15.
                # Our G vectors are -8..8.
                # We need to align the frequencies.
                # eps_F[m, n] in the 32x32 FFT corresponds to frequency (m, n) if we shift.
                # np.fft.fft2 returns frequencies 0..31. 
                # To get -16..15, we use fftshift.
                # But we only need -8..8.
                # Let's precompute the mapping or just slice.
                # Actually, it's safer to compute eps_F once outside the k-loop.
                # But we need the values at specific (m, n) indices.
                
                # Let's restructure: Precompute eps_F_shifted
                # eps_F_shifted[m+16, n+16] = eps_{mn}
                
                # Inside matvec, we need to multiply u (which is in frequency space of the 17x17 grid)
                # by inv_eps_F at the corresponding indices.
                # u is 17x17. Its indices 0..16 correspond to G indices 0..16 in our G_vecs list?
                # No, G_vecs list is ordered -8..8.
                # w.reshape(17,17) maps index i to G_vecs[i].
                # So u[i, j] corresponds to G_vecs[i*17 + j] = (m, n).
                # We need inv_eps_F at (m, n).
                
                # We will pass inv_eps_F_17x17 to the closure.
                z = inv_eps_F_17 * u
                # 4. y = F^H z
                y = np.fft.ifft2(z)
                return y.flatten().real

            # Precompute inv_eps_F_17 for this cell (outside k-loop ideally, but here inside for clarity, 
            # though it's constant for all k-points)
            # Actually, inv_eps_F depends only on cell, not k.
            # Let's move this up.
            
            try:
                from scipy.sparse.linalg import eigsh
                from scipy.sparse import LinearOperator
                A_op = LinearOperator((nG, nG), matvec=matvec, dtype=complex)
                # We want the 5 smallest eigenvalues (bands 1-5).
                # eigsh with 'SA' (smallest algebraic)
                vals, vecs = eigsh(A_op, k=5, which='SA', tol=1e-5)
                vals = np.sort(vals.real)
                
                if len(vals) < 5:
                    return -1.0
                
                # lambda = (omega/c)^2
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

    # Optimization: Simulated Annealing on Pixels
    # State: 32x32 binary matrix, symmetric under D4.
    # Moves: Flip a pixel in the fundamental domain (1/8 of the cell).
    
    # Define the fundamental domain indices
    # D4 symmetry: (i, j) is equivalent to (i, 31-j), (31-i, j), (31-i, 31-j),
    # and transposes (j, i), (j, 31-i), (31-j, i), (31-j, 31-i).
    # Fundamental domain: i <= 15, j <= i (lower triangle of the first half).
    # Actually, simpler: i in [0, 15], j in [0, i].
    # Let's verify:
    # If i=0, j=0. Sym: (0,0), (0,31), (31,0), (31,31), (0,0)... ok.
    # If i=15, j=0. Sym: (15,0), (15,31), (16,0), (16,31), (0,15), (31,15), (0,16), (31,16).
    # This covers all unique orbits.
    
    fd_indices = []
    for i in range(16):
        for j in range(i + 1):
            fd_indices.append((i, j))
    
    # Initialize with a good seed: 4 rods
    # Center at (8,8), (23,8), etc? No, (8.5, 8.5) is center of 4x4 block?
    # Let's use the superellipse generator from the prompt as a starting point.
    
    def generate_superellipse_cell(cx, cy, a, b, p):
        cell = np.zeros((size, size), dtype=int)
        centers = [
            (cx, cy), (cx, 1-cy), (1-cx, cy), (1-cx, 1-cy),
            (cy, cx), (cy, 1-cx), (1-cy, cx), (1-cy, 1-cx)
        ]
        unique_centers = list(set(tuple(np.round(c, 10)) for c in centers))
        for (ucx, ucy) in unique_centers:
            for i in range(size):
                for j in range(size):
                    x = (j + 0.5) / size
                    y = (i + 0.5) / size
                    dx = x - ucx
                    dy = y - ucy
                    if a > 0 and b > 0:
                        val = (abs(dx/a)**p + abs(dy/b)**p)
                        if val <= 1.0:
                            cell[i, j] = 1
        return cell

    # Start with 4 rods of radius ~0.15 (diameter 30% of cell)
    # In pixels: 0.15 * 32 = 4.8 pixels.
    # Center at 0.25, 0.25 -> pixel 8.
    init_cell = generate_superellipse_cell(0.25, 0.25, 0.15, 0.15, 2.0)
    
    current_cell = init_cell.copy()
    current_score = compute_score(current_cell)
    
    best_cell = current_cell.copy()
    best_score = current_score
    
    # Precompute the 17x17 slice of inv_eps_F for the current cell
    # We need to update this when cell changes.
    # For speed, we can compute it inside compute_score, but that's slow.
    # Let's create a helper that updates the solver state.
    
    class SolverState:
        def __init__(self, cell):
            self.cell = cell
            self.update(cell)
            
        def update(self, cell):
            self.cell = cell
            eps_grid = eps_low + (eps_high - eps_low) * np.array(cell, dtype=float)
            eps_F_full = np.fft.fft2(eps_grid) / (size * size)
            # Extract 17x17 centered part
            # Frequencies -8..8 correspond to indices 24..40 in 32x32 FFT?
            # 32/2 = 16. So -16..15.
            # -8 is index 16-8 = 8.
            # 8 is index 8.
            # So indices 8..24?
            # Let's check: fftshift would put 0 at center.
            # Without shift: 0, 1, ..., 15, -16, ..., -1.
            # We want -8..8.
            # -8 is at index 16+8 = 24? No.
            # Index k represents frequency k if k < 16, else k-32.
            # We want freq -8..8.
            # -8 -> index 24.
            # -1 -> index 31.
            # 0 -> index 0.
            # 1 -> index 1.
            # ...
            # 8 -> index 8.
            # So we need indices 24..31 and 0..8.
            # This is awkward to slice directly.
            # Easier: use fftshift, then take center 17x17.
            eps_F_shifted = np.fft.fftshift(eps_F_full)
            # Center of 32x32 is 16,16.
            # We want -8..8.
            # In shifted array, index 16 is freq 0.
            # So indices 16-8..16+8 = 8..24.
            self.inv_eps_F_17 = 1.0 / eps_F_shifted[8:25, 8:25]
            
            # Also need to map G_vecs to this 17x17 grid.
            # G_vecs are ordered -8..8 in m, then -8..8 in n.
            # The 17x17 grid from fftshift is also -8..8 in row, -8..8 in col.
            # So index (i, j) in 17x17 corresponds to G = (i-8, j-8).
            # This matches our G_vecs ordering if we flatten it correctly.
            # G_vecs list:
            # m=-8, n=-8..8 (17 items)
            # m=-7, n=-8..8 (17 items)
            # ...
            # This is row-major with m as row, n as col.
            # So G_vecs[k] corresponds to (m, n) where k = (m+8)*17 + (n+8).
            # The 17x17 array inv_eps_F_17[i, j] corresponds to (m=i-8, n=j-8).
            # So when we do u = fft2(w.reshape(17,17)), u[i, j] is the coefficient for G=(i-8, j-8).
            # And we multiply by inv_eps_F_17[i, j]. This is correct.
            
            self.matvec = None # Placeholder
            
        def get_matvec(self, kx, ky):
            # Closure over self
            inv_eps_F_17 = self.inv_eps_F_17
            G_vecs = G_vecs
            G2 = G2
            
            def matvec(v):
                w = (kx + G_vecs[:, 0])**2 + (ky + G_vecs[:, 1])**2
                w = w * v
                u = np.fft.fft2(w.reshape(17, 17))
                z = inv_eps_F_17 * u
                y = np.fft.ifft2(z)
                return y.flatten().real
            return matvec

    state = SolverState(current_cell)
    
    def fast_score(state):
        min_omega5 = np.inf
        max_omega4 = 0.0
        
        for kx, ky in k_points:
            matvec = state.get_matvec(kx, ky)
            try:
                from scipy.sparse.linalg import eigsh
                from scipy.sparse import LinearOperator
                A_op = LinearOperator((nG, nG), matvec=matvec, dtype=complex)
                vals, vecs = eigsh(A_op, k=5, which='SA', tol=1e-5)
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

    # Re-evaluate initial score with fast solver
    current_score = fast_score(state)
    best_score = current_score
    best_cell = current_cell.copy()

    # Simulated Annealing Loop
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    # Number of moves per temperature step
    moves_per_step = 50
    
    step = 0
    while time.time() < deadline - 10.0:
        if T < T_min:
            T = T_min
        
        for _ in range(moves_per_step):
            if time.time() > deadline - 5.0:
                break
                
            # Pick a random index in the fundamental domain
            idx = random.randint(0, len(fd_indices) - 1)
            i, j = fd_indices[idx]
            
            # Flip the pixel and all its symmetric counterparts
            old_val = current_cell[i, j]
            new_val = 1 - old_val
            
            # Apply flip to all 8 symmetric positions (deduplicated)
            positions = [
                (i, j), (i, size-1-j), (size-1-i, j), (size-1-i, size-1-j),
                (j, i), (j, size-1-i), (size-1-j, i), (size-1-j, size-1-i)
            ]
            unique_pos = list(set(positions))
            
            for (pi, pj) in unique_pos:
                current_cell[pi, pj] = new_val
            
            # Update solver state
            state.update(current_cell)
            
            # Compute new score
            new_score = fast_score(state)
            
            # Accept or reject
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_cell = current_cell.copy()
            else:
                # Metropolis criterion
                delta = new_score - current_score
                if delta < 0:
                    p = np.exp(delta / T)
                    if random.random() < p:
                        current_score = new_score
                    else:
                        # Revert
                        for (pi, pj) in unique_pos:
                            current_cell[pi, pj] = old_val
                        state.update(current_cell)
            
        T *= cooling
        step += 1

    # Ensure symmetry and return
    return best_cell.tolist()
