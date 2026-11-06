import numpy as np
import time
import math
import random
from scipy.sparse.linalg import eigsh
from scipy.sparse import LinearOperator

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    # Evaluator kills at 230s. We need to return well before.
    # Internal solver is fast, but we have many k-points.
    # Let's aim for 180s max to be safe.
    deadline = start_time + 180.0

    # 1. Setup Fast Band Solver
    # We use 15x15 = 225 plane waves for the internal search loop.
    # This is a good balance between accuracy and speed for local search.
    # The evaluator uses 289 (17x17), but 225 is sufficient for relative ranking
    # in the local search phase, provided the gap is not extremely small.
    half_pw = 7 
    G_vecs = []
    for m in range(-half_pw, half_pw + 1):
        for n in range(-half_pw, half_pw + 1):
            G_vecs.append((m, n))
    G_vecs = np.array(G_vecs) * (2.0 * np.pi)
    nG = len(G_vecs)
    
    # Precompute k-points.
    # The evaluator uses 30 wave vectors. We use a dense set of 13x13 = 169 points
    # in the irreducible BZ triangle to ensure we catch the min/max accurately.
    # The irreducible BZ for square lattice is 0 <= kx <= ky <= pi.
    # Wait, standard irreducible BZ for square is triangle (0,0)-(pi,0)-(pi/2,pi/2)?
    # No, for square lattice, high symmetry points are Gamma(0,0), X(pi,0), M(pi/2,pi/2).
    # The irreducible BZ is the triangle with vertices (0,0), (pi,0), (pi/2,pi/2).
    # Let's generate points in this triangle.
    
    k_points = []
    n_k = 12 # Resolution along the axes
    for i in range(n_k + 1):
        for j in range(n_k + 1):
            kx = (i / n_k) * (np.pi / 2)
            ky = (j / n_k) * (np.pi / 2)
            # Check if in triangle: kx >= 0, ky >= 0, kx + ky <= pi/2? 
            # No, the triangle vertices are (0,0), (pi,0), (pi/2, pi/2).
            # The boundaries are:
            # 1. kx=0 to pi/2 (left edge? No, Gamma to M is (0,0) to (pi/2, pi/2))
            # Let's use the standard high symmetry path: Gamma-X-M-Gamma.
            # But the evaluator says "30 wave vectors covering the boundary and interior".
            # Let's just use a dense grid over the whole BZ quadrant and rely on symmetry?
            # No, TM bands are periodic. We can just sample the first quadrant 0..pi, 0..pi.
            # Since the cell is symmetric, the bands are well-behaved.
            # Let's use a 10x10 grid in [0, pi]x[0, pi] plus high symmetry points.
            pass
            
    # Better k-point strategy: 
    # The gap is usually at X (pi, 0) or M (pi/2, pi/2) or Gamma (0,0).
    # For a 4-rod square lattice, the gap is at X.
    # For a honeycomb-like structure, it might be at Gamma or M.
    # To be safe, we sample a dense grid.
    
    kx_vals = np.linspace(0, np.pi, 15)
    ky_vals = np.linspace(0, np.pi, 15)
    for kx in kx_vals:
        for ky in ky_vals:
            k_points.append((kx, ky))
            
    # Add specific high symmetry points to be sure
    k_points.append((0, 0))
    k_points.append((np.pi, 0))
    k_points.append((np.pi/2, np.pi/2))
    k_points.append((np.pi, np.pi))
    
    k_points = np.array(k_points)
    n_kpts = len(k_points)
    
    def compute_score(cell):
        """
        Compute the 4-5 TM band gap score for a given cell (32x32 array of 0/1).
        """
        eps_grid = eps_low + (eps_high - eps_low) * np.array(cell, dtype=float)
        
        # FFT of epsilon
        eps_F = np.fft.fft2(eps_grid) / (size * size)
        
        # Handle division by zero in 1/eps_F
        # eps_F[0,0] is the mean, which is non-zero.
        # Other components might be zero.
        inv_eps_F = np.zeros_like(eps_F, dtype=complex)
        mask = np.abs(eps_F) > 1e-12
        inv_eps_F[mask] = 1.0 / eps_F[mask]
        # If eps_F is 0, 1/eps_F is infinity. But in the Galerkin method,
        # if eps_F[G] is 0, that Fourier component doesn't couple.
        # Actually, the operator is (k+G)^2 * delta - sum_G' (1/eps_{G-G'}) ...
        # Wait, the standard PW method for TM is:
        # sum_G' (1/eps_{G-G'}) (k+G) x (k+G') a_{G'} = (omega/c)^2 a_G
        # This is a generalized eigenvalue problem or can be transformed.
        # The current code uses:
        # A_op = matvec(v) = (kG2 * v) convolved with 1/eps_F?
        # Let's check the current code's matvec:
        # w = kG2 * v
        # u = fft2(w)
        # z = inv_eps_F * u
        # y = ifft2(z)
        # This computes: ifft2( fft2((k+G)^2 v) * (1/eps_F) )
        # This is equivalent to (1/eps) * ((k+G)^2 v) in real space?
        # No, multiplication in Fourier space is convolution in real space.
        # (1/eps_F) * U_G = sum_{G'} (1/eps_{G-G'}) U_{G'}
        # So y_G = sum_{G'} (1/eps_{G-G'}) (k+G')^2 v_{G'}
        # This matches the TM eigenvalue equation:
        # sum_{G'} (1/eps_{G-G'}) (k+G')^2 a_{G'} = omega^2 a_G
        # So the eigenvalues are omega^2.
        
        min_omega5 = np.inf
        max_omega4 = 0.0
        
        for kx, ky in k_points:
            kG = np.stack([kx + G_vecs[:, 0], ky + G_vecs[:, 1]], axis=1)
            kG2 = kG[:, 0]**2 + kG[:, 1]**2
            
            def matvec(v):
                w = kG2 * v
                u = np.fft.fft2(w.reshape(15, 15))
                z = inv_eps_F * u
                y = np.fft.ifft2(z)
                return y.flatten().real
            
            try:
                A_op = LinearOperator((nG, nG), matvec=matvec, dtype=complex)
                # We need the 5th smallest eigenvalue (index 4) and 4th (index 3).
                # eigsh finds the smallest by default if which='SA'.
                vals, vecs = eigsh(A_op, k=6, which='SA', tol=1e-4, maxiter=1000)
                vals = np.sort(vals.real)
                
                if len(vals) < 5:
                    return -1.0
                
                # Filter out negative eigenvalues due to numerical noise
                vals = vals[vals > 0]
                if len(vals) < 5:
                    return -1.0

                omega4 = np.sqrt(vals[3])
                omega5 = np.sqrt(vals[4])
                
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

    def enforce_full_symmetry(c):
        """
        Enforce C4v symmetry on 32x32 array.
        Fundamental domain: 0 <= i <= 15, 0 <= j <= 15, i <= j.
        """
        c_sym = np.zeros_like(c)
        for i in range(16):
            for j in range(i, 16):
                val = c[i, j]
                # 1. (i, j)
                c_sym[i, j] = val
                # 2. (j, i)
                c_sym[j, i] = val
                # 3. (31-i, 31-j)
                c_sym[31-i, 31-j] = val
                # 4. (31-j, 31-i)
                c_sym[31-j, 31-i] = val
                # 5. (i, 31-j)
                c_sym[i, 31-j] = val
                # 6. (31-j, i)
                c_sym[31-j, i] = val
                # 7. (31-i, j)
                c_sym[31-i, j] = val
                # 8. (j, 31-i)
                c_sym[j, 31-i] = val
        return c_sym

    # 2. Generate Initial Cell using Fourier Target Strategy
    # We want to maximize the band gap.
    # Strong (2,0) and (2,2) Fourier components are key.
    # A "checkerboard" of 16x16 blocks gives strong (2,0) and (0,2).
    # A pattern with period 16 in both x and y gives (2,0), (0,2), (2,2).
    
    # Let's create a candidate that is a "fat" checkerboard.
    # Divide 32x32 into 2x2 blocks of 16x16.
    # Block (0,0) and (1,1) are Dielectric.
    # Block (0,1) and (1,0) are Air.
    # This is the standard 4-rod square lattice.
    
    # To improve, we can "round" the corners or add "moire" interference.
    # Let's try a "super-ellipse" based shape in each quadrant, but shifted.
    
    # Actually, let's use the "Fourier Target" approach directly.
    # We define a target potential V(x,y) that has strong (2,0) and (2,2) components.
    # V(x,y) = A * cos(2*pi*2*x) * cos(2*pi*2*y) + B * cos(2*pi*2*x) * sin(2*pi*2*y) ...
    
    # Simpler: Generate a random field, project it onto the subspace spanned by
    # the desired Fourier modes, then threshold.
    
    def generate_fourier_target_cell():
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        x = x / size
        y = y / size
        
        # Modes: (2,0), (0,2), (2,2), (2,-2)
        # We want a real, symmetric pattern.
        # cos(4*pi*x) + cos(4*pi*y) + cos(4*pi*(x+y)) + cos(4*pi*(x-y))
        
        f1 = np.cos(4 * np.pi * x)
        f2 = np.cos(4 * np.pi * y)
        f3 = np.cos(4 * np.pi * (x + y))
        f4 = np.cos(4 * np.pi * (x - y))
        
        # Combine with random weights to explore the space
        w1 = random.uniform(0.5, 1.5)
        w2 = random.uniform(0.5, 1.5)
        w3 = random.uniform(0.5, 1.5)
        w4 = random.uniform(0.5, 1.5)
        
        val = w1 * f1 + w2 * f2 + w3 * f3 + w4 * f4
        
        # Threshold at the median to get ~50% fill
        threshold = np.median(val)
        cell = (val > threshold).astype(int)
        
        return enforce_full_symmetry(cell)

    # Candidate 1: Standard 4-rod (baseline)
    cell_4rod = np.zeros((size, size), dtype=int)
    cell_4rod[0:16, 0:16] = 1
    cell_4rod[0:16, 16:32] = 0
    cell_4rod[16:32, 0:16] = 0
    cell_4rod[16:32, 16:32] = 1
    cell_4rod = enforce_full_symmetry(cell_4rod)
    
    # Candidate 2: Fourier Target
    cell_ft = generate_fourier_target_cell()
    
    # Candidate 3: "Honeycomb" like (shifted 4-rod)
    # Try shifting the dielectric blocks slightly to break the square symmetry locally?
    # No, must maintain global symmetry.
    # Try a pattern with a hole in the center of each dielectric block.
    cell_hole = np.zeros((size, size), dtype=int)
    cell_hole[0:16, 0:16] = 1
    cell_hole[16:32, 16:32] = 1
    # Punch holes
    cell_hole[4:12, 4:12] = 0
    cell_hole[20:28, 20:28] = 0
    cell_hole = enforce_full_symmetry(cell_hole)
    
    # Evaluate initial candidates
    candidates = [cell_4rod, cell_ft, cell_hole]
    best_cell = None
    best_score = -np.inf
    
    for c in candidates:
        score = compute_score(c)
        if score > best_score:
            best_score = score
            best_cell = c.copy()
            
    # 3. Multi-Restart Stochastic Hill Climbing
    # The current program uses Simulated Annealing with a single start.
    # We will use multiple starts from different Fourier targets and
    # a "greedy" hill climb with occasional random jumps to escape local minima.
    
    n_restarts = 5
    current_cell = best_cell.copy()
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
            
        # If not the first restart, generate a new random Fourier target
        if restart > 0:
            current_cell = generate_fourier_target_cell()
            current_score = compute_score(current_cell)
            if current_score > best_score:
                best_score = current_score
                best_cell = current_cell.copy()
        else:
            current_score = best_score
            
        # Hill Climbing
        T = 1.0
        T_min = 0.05
        cooling = 0.99
        
        while time.time() < deadline - 10.0:
            # Pick a random pixel in the fundamental domain
            i = random.randint(0, 15)
            j = random.randint(i, 15)
            
            new_cell = current_cell.copy()
            new_cell[i, j] = 1 - new_cell[i, j]
            new_cell = enforce_full_symmetry(new_cell)
            
            if np.array_equal(new_cell, current_cell):
                continue
                
            new_score = compute_score(new_cell)
            
            if new_score > current_score:
                current_score = new_score
                current_cell = new_cell
                if current_score > best_score:
                    best_score = current_score
                    best_cell = current_cell.copy()
                    # Reset temperature on improvement?
                    # T = max(T_min, T * 1.1) # Reheat
            else:
                delta = new_score - current_score
                if delta < 0:
                    p = math.exp(delta / T)
                    if random.random() < p:
                        current_cell = new_cell
                        current_score = new_score
                        
            T = max(T_min, T * cooling)
            
            # If we've cooled too much, reheat slightly to escape
            if T < 0.1 and random.random() < 0.01:
                T = 0.5

    # Ensure the returned cell is a list of lists of ints
    return best_cell.tolist()
