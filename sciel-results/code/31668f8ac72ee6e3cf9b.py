import numpy as np
import time
import math

def design(problem):
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 220.0  # Safety margin before 230s kill
    
    # Precompute C4v orbits
    def get_orbit(i, j):
        s = size
        orbit = set()
        positions = [
            (i, j),
            (s-1-i, j),
            (i, s-1-j),
            (s-1-i, s-1-j),
            (j, i),
            (s-1-j, i),
            (j, s-1-i),
            (s-1-j, s-1-i)
        ]
        for pi, pj in positions:
            orbit.add((pi, pj))
        return list(orbit)
    
    # Fast evaluator using reduced plane waves
    def eval_fast(cell_arr):
        """
        Fast approximate band gap evaluation using 5x5 = 25 plane waves.
        cell_arr is a numpy array of shape (size, size) with values 0 or 1.
        """
        # Compute epsilon grid
        eps_grid = eps_low + (eps_high - eps_low) * cell_arr.astype(float)
        inv_eps = 1.0 / eps_grid
        
        # FFT of 1/epsilon
        N_pw = 5
        fft_inv_eps = np.fft.fft2(inv_eps) / (size * size)
        
        # G vectors
        G_vals = np.arange(-N_pw, N_pw + 1)
        Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
        G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
        num_pw = len(G_vectors)
        
        # K points in irreducible BZ (subset)
        k_points = [
            (0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.4, 0.0), (0.5, 0.0),
            (0.0, 0.1), (0.1, 0.1), (0.2, 0.2), (0.3, 0.3), (0.4, 0.4),
            (0.5, 0.1), (0.5, 0.2), (0.5, 0.3), (0.5, 0.4)
        ]
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in k_points:
            k = np.array([kx, ky])
            k_plus_G = k + G_vectors
            omega_sq = np.sum(k_plus_G ** 2, axis=1)
            
            # Build coefficient matrix
            coeff_mat = np.zeros((num_pw, num_pw), dtype=complex)
            for i in range(num_pw):
                for j in range(num_pw):
                    dx = int(G_vectors[i, 0] - G_vectors[j, 0])
                    dy = int(G_vectors[i, 1] - G_vectors[j, 1])
                    ix = dx % size
                    iy = dy % size
                    coeff_mat[i, j] = fft_inv_eps[ix, iy]
            
            M = coeff_mat * omega_sq[:, np.newaxis]
            eigvals = np.linalg.eigvals(M)
            real_eigvals = np.sort(np.real(eigvals))
            
            if len(real_eigvals) >= 5:
                omega_4 = np.sqrt(real_eigvals[3])
                omega_5 = np.sqrt(real_eigvals[4])
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0
        
        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap == 0:
            return -1.0
        return 2 * gap / midgap
    
    # Accurate evaluator using 17x17 = 289 plane waves (matches evaluator)
    def eval_acc(cell_arr):
        """
        Accurate band gap evaluation using 17x17 = 289 plane waves.
        """
        eps_grid = eps_low + (eps_high - eps_low) * cell_arr.astype(float)
        inv_eps = 1.0 / eps_grid
        fft_inv_eps = np.fft.fft2(inv_eps) / (size * size)
        
        N_pw = 17
        G_vals = np.arange(-N_pw, N_pw + 1)
        Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
        G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
        num_pw = len(G_vectors)
        
        # 30 K points covering irreducible BZ
        k_points = []
        for t in np.linspace(0, 0.5, 10):
            k_points.append((t, 0))
        for t in np.linspace(0, 0.5, 10):
            k_points.append((0.5, t))
        for t in np.linspace(0, 0.5, 10):
            k_points.append((t, t))
        # Remove duplicates
        seen = set()
        unique_k = []
        for k in k_points:
            key = (round(k[0], 10), round(k[1], 10))
            if key not in seen:
                seen.add(key)
                unique_k.append(k)
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in unique_k:
            k = np.array([kx, ky])
            k_plus_G = k + G_vectors
            omega_sq = np.sum(k_plus_G ** 2, axis=1)
            
            coeff_mat = np.zeros((num_pw, num_pw), dtype=complex)
            for i in range(num_pw):
                for j in range(num_pw):
                    dx = int(G_vectors[i, 0] - G_vectors[j, 0])
                    dy = int(G_vectors[i, 1] - G_vectors[j, 1])
                    ix = dx % size
                    iy = dy % size
                    coeff_mat[i, j] = fft_inv_eps[ix, iy]
            
            M = coeff_mat * omega_sq[:, np.newaxis]
            eigvals = np.linalg.eigvals(M)
            real_eigvals = np.sort(np.real(eigvals))
            
            if len(real_eigvals) >= 5:
                omega_4 = np.sqrt(real_eigvals[3])
                omega_5 = np.sqrt(real_eigvals[4])
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0
        
        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap == 0:
            return -1.0
        return 2 * gap / midgap
    
    def enforce_symmetry(cell_list):
        """
        Enforce C4v symmetry on a list of lists.
        """
        cell = np.array(cell_list, dtype=int)
        s = size
        sym = np.zeros((s, s), dtype=int)
        counts = np.zeros((s, s), dtype=int)
        
        for i in range(s):
            for j in range(s):
                val = cell[i, j]
                orbit = get_orbit(i, j)
                for pi, pj in orbit:
                    sym[pi, pj] += val
                    counts[pi, pj] += 1
        
        result = []
        for i in range(s):
            row = []
            for j in range(s):
                if counts[i, j] > 0:
                    row.append(1 if sym[i, j] * 2 > counts[i, j] else 0)
                else:
                    row.append(0)
            result.append(row)
        return result
    
    # --- Phase 1: Generate diverse initial candidates ---
    
    def make_superlattice(radius, num=2):
        """Create a symmetric superlattice with num^2 disks."""
        cell = np.zeros((size, size), dtype=int)
        for r in range(num):
            for c in range(num):
                cx = (2*r + 1) * size / (2*num)
                cy = (2*c + 1) * size / (2*num)
                for i in range(size):
                    for j in range(size):
                        dx = (i - cx) ** 2
                        dy = (j - cy) ** 2
                        if dx + dy < radius ** 2:
                            cell[i, j] = 1
        return cell
    
    def make_cross(thickness):
        """Create a cross-shaped dielectric pattern."""
        cell = np.zeros((size, size), dtype=int)
        mid = size // 2
        half_t = thickness // 2
        for i in range(mid - half_t, mid + half_t):
            cell[i, :] = 1
        for j in range(mid - half_t, mid + half_t):
            cell[:, j] = 1
        return cell
    
    def make_ring(inner_r, outer_r):
        """Create a ring-shaped dielectric pattern."""
        cell = np.zeros((size, size), dtype=int)
        cx = cy = size / 2.0
        for i in range(size):
            for j in range(size):
                dx = (i + 0.5) - cx
                dy = (j + 0.5) - cy
                r2 = dx * dx + dy * dy
                if inner_r ** 2 <= r2 <= outer_r ** 2:
                    cell[i, j] = 1
        return cell
    
    def make_checker(block_size, on_frac=0.5):
        """Create a checkerboard-like pattern."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                block_i = i // block_size
                block_j = j // block_size
                if (block_i + block_j) % 2 == 0:
                    # Randomly fill with probability on_frac
                    if np.random.random() < on_frac:
                        cell[i, j] = 1
        # Enforce symmetry
        cell_sym = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                orbit = get_orbit(i, j)
                vals = [cell[pi, pj] for pi, pj in orbit]
                if sum(vals) * 2 >= len(vals):
                    for pi, pj in orbit:
                        cell_sym[pi, pj] = 1
        return cell_sym
    
    # Generate initial candidates
    candidates = []
    
    # Superlattice variants
    for num in [1, 2, 3]:
        for radius in [3, 4, 5, 6]:
            if radius < size / (2 * num):
                candidates.append(make_superlattice(radius, num))
    
    # Cross variants
    for t in [4, 6, 8]:
        candidates.append(make_cross(t))
    
    # Ring variants
    for inner in [4, 6, 8]:
        for outer in [inner + 3, inner + 5, inner + 7]:
            if outer < size / 2:
                candidates.append(make_ring(inner, outer))
    
    # Checkerboard variants
    for bs in [4, 8]:
        for frac in [0.3, 0.5, 0.7]:
            candidates.append(make_checker(bs, frac))
    
    # Random symmetric patterns
    for _ in range(20):
        cell = np.zeros((size, size), dtype=int)
        # Fill unique orbits
        visited = set()
        for i in range(size):
            for j in range(size):
                if (i, j) in visited:
                    continue
                orbit = get_orbit(i, j)
                if np.random.random() < 0.5:
                    for pi, pj in orbit:
                        cell[pi, pj] = 1
                        visited.add((pi, pj))
                else:
                    for pi, pj in orbit:
                        visited.add((pi, pj))
        candidates.append(cell)
    
    # Evaluate all candidates with fast evaluator
    best_cell = None
    best_score = -np.inf
    for c in candidates:
        score = eval_fast(c)
        if score > best_score:
            best_score = score
            best_cell = c.copy()
    
    # --- Phase 2: Simulated annealing with fast evaluator ---
    
    temperature = 1.0
    cooling_rate = 0.999
    sa_iterations = 5000
    
    current_cell = best_cell.copy()
    current_score = best_score
    best_cell_sa = best_cell.copy()
    best_score_sa = best_score
    
    for it in range(sa_iterations):
        if time.time() > deadline - 10:
            break
        
        # Pick a random orbit to flip
        i = np.random.randint(size)
        j = np.random.randint(size)
        orbit = get_orbit(i, j)
        
        new_cell = current_cell.copy()
        for pi, pj in orbit:
            new_cell[pi, pj] = 1 - new_cell[pi, pj]
        
        new_score = eval_fast(new_cell)
        delta = new_score - current_score
        
        if delta > 0 or np.random.random() < math.exp(delta / max(temperature, 1e-8)):
            current_cell = new_cell
            current_score = new_score
            if current_score > best_score_sa:
                best_score_sa = current_score
                best_cell_sa = new_cell.copy()
        
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
    
    best_cell = best_cell_sa
    best_score = best_score_sa
    
    # --- Phase 3: Local search with fast evaluator for fine tuning ---
    
    ls_iterations = 2000
    for it in range(ls_iterations):
        if time.time() > deadline - 8:
            break
        
        i = np.random.randint(size)
        j = np.random.randint(size)
        orbit = get_orbit(i, j)
        
        new_cell = best_cell.copy()
        for pi, pj in orbit:
            new_cell[pi, pj] = 1 - new_cell[pi, pj]
        
        new_score = eval_fast(new_cell)
        if new_score > best_score:
            best_score = new_score
            best_cell = new_cell
    
    # --- Phase 4: Polish with accurate evaluator ---
    
    current_cell = best_cell.copy()
    current_score = eval_acc(current_cell)
    best_cell_final = current_cell.copy()
    best_score_final = current_score
    
    temperature = 0.5
    cooling_rate = 0.999
    polish_iterations = 1000
    
    for it in range(polish_iterations):
        if time.time() > deadline - 3:
            break
        
        i = np.random.randint(size)
        j = np.random.randint(size)
        orbit = get_orbit(i, j)
        
        new_cell = current_cell.copy()
        for pi, pj in orbit:
            new_cell[pi, pj] = 1 - new_cell[pi, pj]
        
        new_score = eval_acc(new_cell)
        delta = new_score - current_score
        
        if delta > 0 or np.random.random() < math.exp(delta / max(temperature, 1e-8)):
            current_cell = new_cell
            current_score = new_score
            if current_score > best_score_final:
                best_score_final = current_score
                best_cell_final = new_cell.copy()
        
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
    
    # --- Final: Enforce strict symmetry and return ---
    final_cell = enforce_symmetry(best_cell_final.tolist())
    
    # Ensure it's a proper nested list
    return final_cell
