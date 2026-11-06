import numpy as np
import time
import math
import random

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 195.0  # Leave 5s margin for safety against 200s limit

    # 1. Setup Fast Band Solver
    # We use 15x15 = 225 plane waves for the internal search loop.
    half_pw = 7 
    G_vecs = []
    for m in range(-half_pw, half_pw + 1):
        for n in range(-half_pw, half_pw + 1):
            G_vecs.append((m, n))
    G_vecs = np.array(G_vecs)
    nG = len(G_vecs)
    
    # Precompute k-points.
    # The evaluator uses 30 points. We use a representative subset for speed, 
    # prioritizing the high-symmetry lines and the Gamma point.
    k_points = []
    n_k = 10
    # Line Gamma-X (0,0) to (pi, 0)
    for i in range(n_k + 1):
        s = i / n_k
        k_points.append((s * np.pi, 0))
    # Line X-M (pi,0) to (pi,pi)
    for i in range(1, n_k + 1):
        s = i / n_k
        k_points.append((np.pi, s * np.pi))
    # Line Gamma-M (0,0) to (pi/2, pi/2)
    for i in range(1, n_k + 1):
        s = i / n_k
        k_points.append((s * np.pi/2, s * np.pi/2))
    k_points = np.array(k_points)
    
    # Precompute kG for all k-points to avoid repeated addition
    # kG shape: (n_k, nG, 2)
    kG_all = np.zeros((len(k_points), nG, 2))
    for idx, (kx, ky) in enumerate(k_points):
        kG_all[idx, :, 0] = kx + G_vecs[:, 0] * (2.0 * np.pi)
        kG_all[idx, :, 1] = ky + G_vecs[:, 1] * (2.0 * np.pi)
    kG2_all = np.sum(kG_all**2, axis=2) # (n_k, nG)

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
        
        # Precompute matrix structure parts if possible, but matvec is dynamic per k
        # We can precompute the FFT of the identity? No, we need to apply inv_eps_F.
        
        # To speed up, we can precompute the convolution kernel in real space?
        # A v = (k+G)^2 * (1/eps) * (k+G)^2 v
        # In real space (pixel space), 1/eps is a mask.
        # But plane waves are in Fourier space.
        
        for idx in range(len(k_points)):
            kG2 = kG2_all[idx]
            
            def matvec(v):
                w = kG2 * v
                # w is in G-space. We need to multiply by 1/eps in G-space.
                # 1/eps in G-space is the inverse FFT of 1/eps in real space.
                # Wait, the operator is (k+G)^2 * [1/eps] * (k+G)^2.
                # [1/eps] acts as a convolution in real space, or element-wise in Fourier space.
                # In plane wave expansion, 1/eps is represented by its Fourier series coefficients.
                # inv_eps_F are these coefficients.
                # So the matrix element M_{mn} = (k+G_m)^2 * inv_eps_F_{G_m - G_n} * (k+G_n)^2.
                # This is a circulant-like structure but with (k+G)^2 factors.
                
                # Efficient multiplication:
                # 1. Multiply v by (k+G)^2 -> w
                # 2. Transform w to "real" space (which is G-space convolution)?
                # Actually, the standard trick is:
                # The operator is diagonal in the basis of plane waves shifted by k.
                # Multiplication by 1/eps is a convolution in G-space.
                # We can use FFT if the G-grid is regular. It is.
                # Reshape w to 15x15, FFT, multiply by inv_eps_F (reshaped), IFFT.
                
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

    def generate_fourier_target_cell(target_freqs, amplitudes, phases, threshold):
        """
        Construct a cell by summing specific Fourier modes and thresholding.
        target_freqs: list of (mx, my)
        amplitudes: list of floats
        phases: list of floats
        """
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        x = x / size
        y = y / size
        
        field = np.zeros((size, size))
        for m, a, p in zip(target_freqs, amplitudes, phases):
            mx, my = m
            # Add cos/sin components to ensure real field
            field += a * np.cos(2 * np.pi * (mx * x + my * y) + p)
            
        cell = (field > threshold).astype(int)
        return enforce_full_symmetry(cell)

    def generate_moire_cell(f1, f2, phase_shift, threshold):
        """
        Generate cell using interference of two counter-rotating square-wave gratings.
        """
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        x = x / size
        y = y / size
        
        # Square wave approximation using sign(sin)
        # Pattern 1
        f1_val = np.sign(np.sin(2 * np.pi * f1 * x + phase_shift)) * np.sign(np.sin(2 * np.pi * f1 * y - phase_shift))
        # Pattern 2
        f2_val = np.sign(np.sin(2 * np.pi * f2 * x - phase_shift)) * np.sign(np.sin(2 * np.pi * f2 * y + phase_shift))
        
        val = f1_val + f2_val
        cell = (val > threshold).astype(int)
        return enforce_full_symmetry(cell)

    def generate_tiled_motif_cell(motif_size, motif):
        """
        Tile a small motif to fill the cell.
        """
        cell = np.zeros((size, size), dtype=int)
        n_repeats = size // motif_size
        for i in range(n_repeats):
            for j in range(n_repeats):
                cell[i*motif_size:(i+1)*motif_size, j*motif_size:(j+1)*motif_size] = motif
        return enforce_full_symmetry(cell)

    # 2. Generate a diverse set of candidates
    
    candidates = []
    
    # A. Fourier Target Candidates
    # We want to enhance (2,0) and (2,2) for the 4-5 gap.
    # Let's try combinations of these frequencies.
    freq_20 = [(2, 0), (0, 2)]
    freq_22 = [(2, 2), (-2, 2)]
    
    # Candidate 1: Pure (2,0)
    c1 = generate_fourier_target_cell(freq_20, [1.0, 1.0], [0, 0], 0.0)
    candidates.append(c1)
    
    # Candidate 2: Pure (2,2)
    c2 = generate_fourier_target_cell(freq_22, [1.0, 1.0], [0, 0], 0.0)
    candidates.append(c2)
    
    # Candidate 3: Mix (2,0) and (2,2)
    c3 = generate_fourier_target_cell(freq_20 + freq_22, [1.0, 1.0, 1.0, 1.0], [0, 0, 0, 0], 0.5)
    candidates.append(c3)
    
    # Candidate 4: Higher order mix to create complex structure
    freq_30 = [(3, 0), (0, 3)]
    c4 = generate_fourier_target_cell(freq_20 + freq_30, [1.0, 1.0, 0.5, 0.5], [0, 0, np.pi/4, np.pi/4], 0.2)
    candidates.append(c4)
    
    # B. Moire Candidates
    # Vary frequencies around 2 and 3
    for f1 in [1.5, 2.0, 2.5]:
        for f2 in [2.5, 3.0, 3.5]:
            for ps in [0, np.pi/4, np.pi/2]:
                for th in [-0.5, 0, 0.5]:
                    c = generate_moire_cell(f1, f2, ps, th)
                    candidates.append(c)
                    
    # C. Tiled Motif Candidates
    # Try small motifs that create complex boundaries
    # 4x4 motif: Cross
    motif_cross = np.array([[0,1,0],
                            [1,1,1],
                            [0,1,0]])
    # Pad to 4x4? No, let's use 4x4 directly
    motif_4_cross = np.array([[0,1,1,0],
                              [1,1,1,1],
                              [1,1,1,1],
                              [0,1,1,0]])
    candidates.append(generate_tiled_motif_cell(4, motif_4_cross))
    
    # 8x8 motif: Ring
    motif_8_ring = np.zeros((8,8), dtype=int)
    motif_8_ring[1:7, 1:7] = 1
    motif_8_ring[2:6, 2:6] = 0
    candidates.append(generate_tiled_motif_cell(8, motif_8_ring))
    
    # 16x16 motif: Complex shape
    motif_16 = np.zeros((16,16), dtype=int)
    # Central square
    motif_16[4:12, 4:12] = 1
    # Arms
    motif_16[6:10, 0:4] = 1
    motif_16[6:10, 12:16] = 1
    motif_16[0:4, 6:10] = 1
    motif_16[12:16, 6:10] = 1
    candidates.append(generate_tiled_motif_cell(16, motif_16))
    
    # 3. Evaluate candidates and select best
    best_cell = None
    best_score = -np.inf
    
    # Deduplicate candidates by hash to save time
    seen_hashes = set()
    unique_candidates = []
    for c in candidates:
        h = hash(c.tobytes())
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_candidates.append(c)
            
    for c in unique_candidates:
        score = compute_score(c)
        if score > best_score:
            best_score = score
            best_cell = c.copy()
            
        if time.time() > deadline - 20.0:
            break

    # If no good candidate found, fallback to a simple known good structure
    if best_score < 0:
        cell = np.zeros((size, size), dtype=int)
        # 4 rods
        centers = [(8, 8), (23, 8), (8, 23), (23, 23)]
        for cx, cy in centers:
            cell[cx-2:cx+3, cy-2:cy+3] = 1
        best_cell = enforce_full_symmetry(cell)
        best_score = compute_score(best_cell)

    # 4. Local Search (Refinement)
    # Use a faster, more targeted local search.
    # Instead of random pixels, try flipping "boundary" pixels or small clusters.
    # But given the time, a simple random hill-climbing with symmetry enforcement is robust.
    
    cell = best_cell.copy()
    T = 0.1
    T_min = 0.001
    cooling = 0.9995
    
    steps = 0
    max_steps = 50000 # Safety cap
    
    # To speed up, we can pre-calculate the score of the current cell
    current_score = best_score
    
    while time.time() < deadline - 5.0 and steps < max_steps:
        steps += 1
        
        # Pick a random pixel
        i = random.randint(0, size - 1)
        j = random.randint(0, size - 1)
        
        # Flip it
        new_cell = cell.copy()
        new_cell[i, j] = 1 - new_cell[i, j]
        
        # Enforce symmetry
        new_cell = enforce_full_symmetry(new_cell)
        
        # If no change, skip
        if np.array_equal(new_cell, cell):
            continue
            
        # Calculate score
        new_score = compute_score(new_cell)
        
        # Acceptance criterion
        if new_score > best_score:
            best_score = new_score
            best_cell = new_cell.copy()
            cell = new_cell
            current_score = new_score
        else:
            delta = new_score - current_score
            if delta < 0:
                p = math.exp(delta / T)
                if random.random() < p:
                    cell = new_cell
                    current_score = new_score
            
        T = max(T_min, T * cooling)
        
        # Check time frequently
        if steps % 50 == 0 and time.time() > deadline - 10.0:
            break

    # Ensure the returned cell is a list of lists of ints
    return best_cell.tolist()
