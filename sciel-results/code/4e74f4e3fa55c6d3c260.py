import numpy as np
import time
import math
import random

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin before 200s limit

    # Fast solver setup
    half_pw = 7
    G_vecs = []
    for m in range(-half_pw, half_pw + 1):
        for n in range(-half_pw, half_pw + 1):
            G_vecs.append((m, n))
    G_vecs = np.array(G_vecs) * (2.0 * np.pi)
    nG = len(G_vecs)
    
    # k-points for fast evaluation
    k_points = []
    n_k = 8
    for i in range(n_k + 1):
        for j in range(i, n_k + 1):
            s = i / n_k
            t = j / n_k
            k_points.append((s * np.pi, t * np.pi))
    k_points.append((np.pi, 0))
    k_points.append((np.pi/2, np.pi/2))
    k_points.append((np.pi, np.pi))
    k_points = np.array(k_points)
    
    def compute_score(cell):
        eps_grid = eps_low + (eps_high - eps_low) * np.array(cell, dtype=float)
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
        c_sym = np.zeros_like(c)
        for i in range(16):
            for j in range(i, 16):
                val = c[i, j]
                c_sym[i, j] = val
                c_sym[j, i] = val
                c_sym[31-i, 31-j] = val
                c_sym[31-j, 31-i] = val
                c_sym[i, 31-j] = val
                c_sym[31-j, i] = val
                c_sym[31-i, j] = val
                c_sym[j, 31-i] = val
        return c_sym

    # Generate Perlin noise field for diverse candidate structures
    def generate_perlin_cell(freq1, freq2, persistence, lacunarity, threshold):
        """
        Generate a binary cell from thresholded multi-scale noise field.
        """
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        x = x / size
        y = y / size
        
        field = np.zeros((size, size))
        amp = 1.0
        freq = 1.0
        
        for octave in range(4):
            # Use deterministic pseudo-random phases based on seed
            phase1 = (freq * 1.3 + octave * 0.7) % (2 * np.pi)
            phase2 = (freq * 1.7 + octave * 1.1) % (2 * np.pi)
            
            # Add sine waves at different frequencies
            field += amp * np.sin(2 * np.pi * freq * x + phase1) * np.cos(2 * np.pi * freq * y + phase2)
            amp *= persistence
            freq *= lacunarity
        
        # Normalize
        field -= np.mean(field)
        field /= (np.std(field) + 1e-10)
        
        # Threshold
        cell = (field > threshold).astype(int)
        return enforce_full_symmetry(cell)

    # Generate diverse initial candidates
    candidates = []
    
    # Perlin noise candidates with different parameters
    for freq in [2, 3, 4]:
        for threshold in [-0.5, 0, 0.5]:
            cell = generate_perlin_cell(freq, freq, 0.5, 2.0, threshold)
            candidates.append(cell)
    
    # Traditional geometric candidates
    # Checkerboard 4x4
    motif_4 = np.array([[1,0,1,0],[0,1,0,1],[1,0,1,0],[0,1,0,1]])
    cell_4 = np.zeros((size, size), dtype=int)
    for i in range(size // 4):
        for j in range(size // 4):
            cell_4[i*4:(i+1)*4, j*4:(j+1)*4] = motif_4
    candidates.append(enforce_full_symmetry(cell_4))
    
    # 4 squares
    cell_sq = np.zeros((size, size), dtype=int)
    for cx, cy in [(8, 8), (23, 8), (8, 23), (23, 23)]:
        cell_sq[cx-3:cx+4, cy-3:cy+4] = 1
    candidates.append(enforce_full_symmetry(cell_sq))
    
    # Rings
    cell_ring = np.zeros((size, size), dtype=int)
    for cx, cy in [(8, 8), (23, 8), (8, 23), (23, 23)]:
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                r = math.sqrt(dx*dx + dy*dy)
                if 2.5 <= r <= 4.5:
                    cell_ring[cx+dx, cy+dy] = 1
    candidates.append(enforce_full_symmetry(cell_ring))
    
    # Evaluate all candidates
    best_cell = None
    best_score = -np.inf
    
    for c in candidates:
        if time.time() > deadline - 20:
            break
        score = compute_score(c)
        if score > best_score:
            best_score = score
            best_cell = c.copy()
    
    if best_score < 0:
        best_cell = candidates[0].copy()
        best_score = 0.0

    # Local search with simulated annealing
    cell = best_cell.copy()
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    steps = 0
    while time.time() < deadline - 10.0:
        steps += 1
        
        i = random.randint(0, size - 1)
        j = random.randint(0, size - 1)
        
        new_cell = cell.copy()
        new_cell[i, j] = 1 - new_cell[i, j]
        new_cell = enforce_full_symmetry(new_cell)
        
        if np.array_equal(new_cell, cell):
            continue
        
        new_score = compute_score(new_cell)
        
        if new_score > best_score:
            best_score = new_score
            best_cell = new_cell.copy()
            cell = new_cell
        else:
            delta = new_score - best_score
            if delta < 0:
                p = math.exp(delta / T)
                if random.random() < p:
                    cell = new_cell
            
        T = max(T_min, T * cooling)
        
        if steps % 200 == 0 and time.time() > deadline - 15.0:
            break

    return best_cell.tolist()
