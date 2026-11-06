import numpy as np
import time
import math
import random

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 190.0  # Leave 10s margin for safety against 200s limit

    # --- Fast Internal Band Solver ---
    # We use a reduced set of plane waves for the internal search loop to be faster,
    # but the evaluator uses 289. The physics is similar enough for ranking.
    half_pw = 7 
    G_vecs = []
    for m in range(-half_pw, half_pw + 1):
        for n in range(-half_pw, half_pw + 1):
            G_vecs.append((m, n))
    G_vecs = np.array(G_vecs) * (2.0 * np.pi)
    nG = len(G_vecs)
    
    # Precompute k-points.
    # The evaluator uses 30 points. We use a representative subset for speed.
    # Irreducible BZ for square lattice: 0 <= kx <= ky <= pi
    k_points = []
    n_k = 8
    for i in range(n_k + 1):
        for j in range(i, n_k + 1): # Ensure kx <= ky
            s = i / n_k
            t = j / n_k
            k_points.append((s * np.pi, t * np.pi))
    # Add high symmetry points specifically
    k_points.append((np.pi, 0))
    k_points.append((np.pi/2, np.pi/2))
    k_points.append((np.pi, np.pi))
    k_points = np.array(k_points)
    
    def compute_score(cell):
        """
        Compute the 4-5 TM band gap score for a given cell (32x32 array of 0/1).
        """
        eps_grid = eps_low + (eps_high - eps_low) * np.array(cell, dtype=float)
        
        # FFT of epsilon
        eps_F = np.fft.fft2(eps_grid) / (size * size)
        inv_eps_F = 1.0 / eps_F
        
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
                from scipy.sparse.linalg import eigsh
                from scipy.sparse import LinearOperator
                A_op = LinearOperator((nG, nG), matvec=matvec, dtype=complex)
                vals, vecs = eigsh(A_op, k=6, which='SA', tol=1e-4)
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

    def enforce_full_symmetry(c):
        """
        Enforce C4v symmetry on 32x32 array.
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

    # --- New Strategy: Fourier-Space Direct Synthesis ---
    
    # The goal is to maximize the gap between TM bands 4 and 5.
    # This gap is strongly influenced by the Fourier components of the permittivity (or inverse permittivity)
    # at specific reciprocal lattice vectors, particularly G=(2,0) and G=(2,2) for square lattices.
    # We construct a potential field U(x,y) in real space such that its Fourier transform has
    # large positive components at G=(2,0) and G=(2,2), and appropriate values elsewhere.
    
    # We will use a "Pulse-Phase" method.
    # We define the dielectric function epsilon(x,y) as a function of a continuous field phi(x,y).
    # epsilon = eps_low + (eps_high - eps_low) * sigmoid(phi(x,y) - threshold)
    
    # To maximize specific Fourier components, we can construct phi(x,y) as a sum of cosines
    # corresponding to the desired G vectors.
    # phi(x,y) = A1*cos(2*pi*(2x/32 + 0y/32) + p1) + A2*cos(2*pi*(2x/32 + 2y/32) + p2) + ...
    
    # However, simple cosines might not yield the best binary thresholding.
    # We will search over the amplitudes and phases of the first few G vectors
    # that contribute most to the 4-5 gap coupling.
    
    # Key G vectors for TM 4-5 gap in square lattice:
    # (2,0), (0,2), (2,2), (-2,2), etc. Due to symmetry, we focus on (2,0) and (2,2).
    # Also, the mean value (0,0) determines the fill fraction.
    
    # We parameterize the field as:
    # phi(x,y) = A0 + A1*cos(G1.x + p1) + A2*cos(G2.x + p2) + A3*cos(G3.x + p3)
    
    # G1 = (2,0) * 2*pi/32
    # G2 = (0,2) * 2*pi/32  (symmetric to G1)
    # G3 = (2,2) * 2*pi/32
    
    # Since we need C4v symmetry, the field must be symmetric.
    # cos(G.x + G.y) is symmetric.
    # Let's use:
    # phi(x,y) = A0 + A1*cos(2*pi*2*x/32) + A2*cos(2*pi*2*y/32) + A3*cos(2*pi*2*(x+y)/32)
    # This form is automatically symmetric under x<->y and reflections if phases are 0 or pi.
    
    # We will optimize A0, A1, A2, A3 and the threshold T.
    # Actually, A0 is just a shift in T. So we optimize A1, A2, A3, T.
    # We can also include higher order terms if time permits, but let's start with these.
    
    # To allow for more complex shapes, we can add a term for G=(4,0) or G=(2,-2).
    # Let's stick to (2,0), (0,2), (2,2) for the first pass.
    
    def generate_cell_from_fourier_params(A1, A2, A3, T):
        """
        Generate a 32x32 cell from Fourier parameters.
        """
        x = np.arange(size)
        y = np.arange(size)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Frequencies
        f1 = 2 * np.pi * 2 / size
        f2 = 2 * np.pi * 2 / size
        f3 = 2 * np.pi * 2 / size
        
        # Field
        phi = A1 * np.cos(f1 * X) + A2 * np.cos(f2 * Y) + A3 * np.cos(f3 * (X + Y))
        
        # Threshold
        cell = (phi > T).astype(int)
        
        # Enforce symmetry (should be mostly symmetric already, but floating point might cause issues)
        cell = enforce_full_symmetry(cell)
        
        return cell

    # Initial guesses
    # We want a high fill fraction of dielectric? Or low?
    # Usually, for a gap, we want a moderate fill fraction.
    # Let's try A1, A2, A3 in a range and T in a range.
    
    best_score = -1.0
    best_cell = None
    
    # Grid search over parameters
    # A1, A2, A3 in [-5, 5]
    # T in [-5, 5]
    
    # To be efficient, we will use a coarse grid first, then refine.
    
    # Coarse grid
    A_range = np.linspace(-4, 4, 9)
    T_range = np.linspace(-4, 4, 9)
    
    # We assume A1 == A2 due to symmetry (x and y are equivalent).
    # So we only search A1, A3, T.
    
    for A1 in A_range:
        for A3 in A_range:
            for T in T_range:
                if time.time() > deadline - 5.0:
                    break
                cell = generate_cell_from_fourier_params(A1, A1, A3, T)
                
                # Check if cell is valid (not empty or full)
                fill = np.sum(cell) / (size * size)
                if fill < 0.2 or fill > 0.8:
                    continue
                    
                score = compute_score(cell)
                
                if score > best_score:
                    best_score = score
                    best_cell = cell.copy()
            
            if time.time() > deadline - 5.0:
                break
        if time.time() > deadline - 5.0:
            break
    
    # If we found a good cell, refine locally
    if best_cell is not None and time.time() < deadline - 10.0:
        # Local search around the best parameters
        # We don't store the parameters, so we will do a local pixel search from the best cell.
        # This combines the constructive start with local refinement.
        
        current_cell = best_cell.copy()
        current_score = best_score
        
        T_temp = 0.5
        cooling = 0.999
        
        while time.time() < deadline - 5.0:
            # Pick a random pixel
            i = random.randint(0, size - 1)
            j = random.randint(0, size - 1)
            
            # Flip it
            new_cell = current_cell.copy()
            new_cell[i, j] = 1 - new_cell[i, j]
            
            # Enforce symmetry
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
            else:
                delta = new_score - current_score
                if delta < 0:
                    p = math.exp(delta / T_temp)
                    if random.random() < p:
                        current_cell = new_cell
                        current_score = new_score
            
            T_temp = max(0.01, T_temp * cooling)
            
            # Check time
            if time.time() > deadline - 5.0:
                break

    # Fallback if no cell was found or score is bad
    if best_cell is None or best_score < 0:
        # Default to a simple pattern
        best_cell = np.zeros((size, size), dtype=int)
        # 4 squares
        centers = [(8, 8), (23, 8), (8, 23), (23, 23)]
        for cx, cy in centers:
            best_cell[cx-2:cx+3, cy-2:cy+3] = 1
        best_cell = enforce_full_symmetry(best_cell)
        best_score = compute_score(best_cell)

    return best_cell.tolist()
