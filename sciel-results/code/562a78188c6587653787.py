import time
import numpy as np
from numpy.linalg import eigh

def _compute_gap_fast(cell, n_k=14, pw_radius=12):
    """Compute a fast estimate of the gap-midgap ratio between TM bands 4 and 5."""
    N = len(cell)
    cell_arr = np.array(cell, dtype=float)
    inv_eps = 1.0 / (1.0 + 11.0 * cell_arr)
    fft_inv = np.fft.fft2(inv_eps) / (N * N)
    
    G_list = []
    for Gx in range(-pw_radius, pw_radius + 1):
        for Gy in range(-pw_radius, pw_radius + 1):
            if Gx*Gx + Gy*Gy <= pw_radius * pw_radius:
                G_list.append((Gx, Gy))
    n_pw = len(G_list)
    G_arr = np.array(G_list, dtype=float)
    
    size_d = 2 * pw_radius + 1
    offset_d = pw_radius
    inv_eps_full = np.zeros((size_d, size_d), dtype=complex)
    
    Gx_diff = G_arr[:, 0][:, None] - G_arr[:, 0][None, :]
    Gy_diff = G_arr[:, 1][:, None] - G_arr[:, 1][None, :]
    
    for i in range(n_pw):
        for j in range(n_pw):
            dqx = int(Gx_diff[i, j])
            dqy = int(Gy_diff[i, j])
            if -pw_radius <= dqx <= pw_radius and -pw_radius <= dqy <= pw_radius:
                inv_eps_full[dqx + offset_d, dqy + offset_d] = fft_inv[dqx % N, dqy % N]
    
    # k-points in the irreducible BZ (first quadrant, kx >= ky)
    k_points = [
        (0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.4, 0.0), (0.5, 0.0),
        (0.5, 0.1), (0.5, 0.2), (0.5, 0.3), (0.5, 0.4), (0.5, 0.5),
        (0.3, 0.3), (0.4, 0.2), (0.2, 0.2), (0.3, 0.1), (0.4, 0.4)
    ][:n_k]
    
    max_w4 = 0.0
    min_w5 = float('inf')
    
    for k in k_points:
        kvec = np.array(k, dtype=float)
        kG = kvec + G_arr
        dots = kG @ kG.T
        
        inv_eps_mat = inv_eps_full[
            Gx_diff.astype(int) + offset_d, 
            Gy_diff.astype(int) + offset_d
        ]
        
        M = dots * inv_eps_mat
        M = (M + M.conj().T) / 2
        
        try:
            eigvals = eigh(M, eigvals_only=True)
            eigvals = np.sqrt(np.maximum(eigvals, 0))
            if len(eigvals) >= 5:
                max_w4 = max(max_w4, eigvals[3])
                min_w5 = min(min_w5, eigvals[4])
        except:
            pass
    
    if min_w5 == float('inf') or max_w4 <= 0:
        return -1.0
    return 2 * (min_w5 - max_w4) / (min_w5 + max_w4)


def _compute_gap_detailed(cell, k_points, pw_radius=12):
    """Compute gap and also return eigenvectors for topological optimization."""
    N = len(cell)
    cell_arr = np.array(cell, dtype=float)
    inv_eps = 1.0 / (1.0 + 11.0 * cell_arr)
    fft_inv = np.fft.fft2(inv_eps) / (N * N)
    
    G_list = []
    for Gx in range(-pw_radius, pw_radius + 1):
        for Gy in range(-pw_radius, pw_radius + 1):
            if Gx*Gx + Gy*Gy <= pw_radius * pw_radius:
                G_list.append((Gx, Gy))
    n_pw = len(G_list)
    G_arr = np.array(G_list, dtype=float)
    
    size_d = 2 * pw_radius + 1
    offset_d = pw_radius
    inv_eps_full = np.zeros((size_d, size_d), dtype=complex)
    
    Gx_diff = G_arr[:, 0][:, None] - G_arr[:, 0][None, :]
    Gy_diff = G_arr[:, 1][:, None] - G_arr[:, 1][None, :]
    
    for i in range(n_pw):
        for j in range(n_pw):
            dqx = int(Gx_diff[i, j])
            dqy = int(Gy_diff[i, j])
            if -pw_radius <= dqx <= pw_radius and -pw_radius <= dqy <= pw_radius:
                inv_eps_full[dqx + offset_d, dqy + offset_d] = fft_inv[dqx % N, dqy % N]
    
    max_w4 = 0.0
    min_w5 = float('inf')
    # Store field intensities for topological optimization
    # For each pixel, accumulate |E4|^2 and |E5|^2
    field4 = np.zeros((N, N), dtype=float)
    field5 = np.zeros((N, N), dtype=float)
    
    for k in k_points:
        kvec = np.array(k, dtype=float)
        kG = kvec + G_arr
        dots = kG @ kG.T
        
        inv_eps_mat = inv_eps_full[
            Gx_diff.astype(int) + offset_d, 
            Gy_diff.astype(int) + offset_d
        ]
        
        M = dots * inv_eps_mat
        M = (M + M.conj().T) / 2
        
        try:
            eigvals, eigvecs = eigh(M)
            eigvals = np.sqrt(np.maximum(eigvals, 0))
            if len(eigvals) >= 5:
                w4 = eigvals[3]
                w5 = eigvals[4]
                max_w4 = max(max_w4, w4)
                min_w5 = min(min_w5, w5)
                
                # Get eigenvectors for bands 4 and 5
                vec4 = eigvecs[:, 3]
                vec5 = eigvecs[:, 4]
                
                # Compute field intensity in real space
                # The field is sum over G of c_G * exp(i(k+G)·r)
                # We approximate by projecting back to pixel grid
                for pixel_i in range(N):
                    for pixel_j in range(N):
                        x = (pixel_i + 0.5) / N - 0.5
                        y = (pixel_j + 0.5) / N - 0.5
                        e4 = 0.0
                        e5 = 0.0
                        for idx, (Gx, Gy) in enumerate(G_list):
                            phase4 = 2 * np.pi * ((kvec[0]+Gx)*x + (kvec[1]+Gy)*y)
                            phase5 = 2 * np.pi * ((kvec[0]+Gx)*x + (kvec[1]+Gy)*y)
                            e4 += vec4[idx] * np.exp(1j * phase4)
                            e5 += vec5[idx] * np.exp(1j * phase5)
                        field4[pixel_i, pixel_j] += abs(e4)**2
                        field5[pixel_i, pixel_j] += abs(e5)**2
        except:
            pass
    
    if min_w5 == float('inf') or max_w4 <= 0:
        return -1.0, field4, field5
    gap_ratio = 2 * (min_w5 - max_w4) / (min_w5 + max_w4)
    return gap_ratio, field4, field5


def _make_symmetric(cell):
    """Ensure cell is symmetric under transpose, row-reverse, col-reverse."""
    N = len(cell)
    c = np.array(cell, dtype=float)
    c_t = c.T
    c_r = c[::-1, :]
    c_c = c[:, ::-1]
    c_rt = c[::-1, ::-1]
    c_tr = c.T[::-1, :]
    c_tc = c.T[:, ::-1]
    c_rtc = c[::-1, ::-1].T
    c_averaged = (c + c_t + c_r + c_c + c_rt + c_tr + c_tc + c_rtc) / 8.0
    c_sym = (c_averaged > 0.5).astype(int)
    return c_sym.tolist()


def _gen_rod(r):
    """Circular dielectric rod of radius r (in units of cell size)."""
    size = 32
    cell = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < r*r:
                cell[i][j] = 1
    return cell


def _gen_hole(r):
    """Dielectric with circular air hole of radius r."""
    size = 32
    cell = [[1]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < r*r:
                cell[i][j] = 0
    return cell


def _gen_rhombus(r):
    """Rhombus/diamond shape."""
    size = 32
    cell = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) + abs(y) < r:
                cell[i][j] = 1
    return cell


def _gen_cross(arm_width, arm_length):
    """Cross/plus shape."""
    size = 32
    cell = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if (abs(x) < arm_width/2 and abs(y) < arm_length/2) or \
               (abs(y) < arm_width/2 and abs(x) < arm_length/2):
                cell[i][j] = 1
    return cell


def _gen_random_symmetric(density):
    """Random symmetric pixel pattern respecting full square symmetry."""
    size = 32
    cell = np.zeros((size, size), dtype=int)
    for i in range(size//2):
        for j in range(size//2):
            if j > i:
                continue
            if np.random.random() < density:
                cell[i, j] = 1
                cell[i, size-1-j] = 1
                cell[size-1-i, j] = 1
                cell[size-1-i, size-1-j] = 1
                cell[j, i] = 1
                cell[j, size-1-i] = 1
                cell[size-1-j, i] = 1
                cell[size-1-j, size-1-i] = 1
    return cell.tolist()


def _topological_update(cell, field4, field5, step_size=0.1):
    """
    Topological optimization: update pixels based on local field intensities.
    Pixels where band 4 has high intensity should become more dielectric (1)
    to push band 4 down. Pixels where band 5 has high intensity should become
    more air (0) to push band 5 up.
    """
    size = len(cell)
    c = np.array(cell, dtype=float)
    
    # Normalize fields
    f4_norm = field4 / (np.max(field4) + 1e-10)
    f5_norm = field5 / (np.max(field5) + 1e-10)
    
    # Gradient: we want to increase eps where f4 is high, decrease eps where f5 is high
    # d(score)/d(eps) ~ f4 - f5 (roughly)
    gradient = f4_norm - f5_norm
    
    # Apply gradient with step size
    c_new = c + step_size * gradient
    
    # Enforce symmetry by averaging
    c_new_t = c_new.T
    c_new_r = c_new[::-1, :]
    c_new_c = c_new[:, ::-1]
    c_new_rt = c_new[::-1, ::-1]
    c_new_tr = c_new_t[::-1, :]
    c_new_tc = c_new_t[:, ::-1]
    c_new_rtc = c_new_rt.T
    c_avg = (c_new + c_new_t + c_new_r + c_new_c + c_new_rt + c_new_tr + c_new_tc + c_new_rtc) / 8.0
    
    # Threshold to binary
    c_binary = (c_avg > 0.5).astype(int)
    return c_binary.tolist()


def _mutate(cell, mutation_rate=0.1):
    """Randomly flip some pixels while maintaining full square symmetry."""
    size = len(cell)
    c = np.array(cell, dtype=int)
    for i in range(size//2):
        for j in range(size//2):
            if j > i:
                continue
            if np.random.random() < mutation_rate:
                val = 1 - c[i, j]
                c[i, j] = val
                c[i, size-1-j] = val
                c[size-1-i, j] = val
                c[size-1-i, size-1-j] = val
                c[j, i] = val
                c[j, size-1-i] = val
                c[size-1-j, i] = val
                c[size-1-j, size-1-i] = val
    return c.tolist()


def design(problem):
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin
    
    size = problem["pixels"]
    assert size == 32
    
    # Phase 1: Quick scan of geometric shapes
    candidates = []
    for r in [0.08, 0.12, 0.15, 0.18, 0.22, 0.25, 0.28, 0.32, 0.35, 0.38]:
        candidates.append(_gen_rod(r))
    for r in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        candidates.append(_gen_rhombus(r))
    for w in [0.04, 0.08, 0.12, 0.16, 0.2]:
        for l in [0.25, 0.35, 0.42, 0.48, 0.5]:
            candidates.append(_gen_cross(w, l))
    for r in [0.35, 0.4, 0.45, 0.5, 0.55]:
        candidates.append(_gen_hole(r))
    for density in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        for _ in range(3):
            candidates.append(_gen_random_symmetric(density))
    
    best_cell = None
    best_score = -1.0
    
    for idx, cell in enumerate(candidates):
        if time.time() > deadline - 100:
            break
        try:
            score = _compute_gap_fast(cell, n_k=10, pw_radius=10)
            if score > best_score:
                best_score = score
                best_cell = cell
        except:
            pass
    
    # Phase 2: Topological optimization
    if best_cell is not None and time.time() < deadline - 60:
        current = best_cell
        current_score = best_score
        
        # Use a subset of k-points for faster detailed computation
        k_points_topo = [
            (0.0, 0.0), (0.2, 0.0), (0.4, 0.0), (0.5, 0.0),
            (0.5, 0.2), (0.5, 0.4), (0.5, 0.5),
            (0.3, 0.3), (0.4, 0.2)
        ]
        
        for iteration in range(50):
            if time.time() > deadline - 30:
                break
            
            try:
                gap, field4, field5 = _compute_gap_detailed(current, k_points_topo, pw_radius=10)
                if gap > current_score:
                    current_score = gap
                    best_score = gap
                    best_cell = current
                
                # Topological update
                updated = _topological_update(current, field4, field5, step_size=0.15)
                
                # Also try a mutated version
                mutated = _mutate(updated, 0.05)
                
                # Evaluate both
                score1 = _compute_gap_fast(updated, n_k=10, pw_radius=10)
                score2 = _compute_gap_fast(mutated, n_k=10, pw_radius=10)
                
                if score1 > current_score:
                    current_score = score1
                    current = updated
                    best_score = max(best_score, current_score)
                    best_cell = current
                elif score2 > current_score:
                    current_score = score2
                    current = mutated
                    best_score = max(best_score, current_score)
                    best_cell = current
                else:
                    # Keep current, maybe try a small mutation
                    if np.random.random() < 0.3:
                        small_mut = _mutate(current, 0.03)
                        score3 = _compute_gap_fast(small_mut, n_k=10, pw_radius=10)
                        if score3 > current_score:
                            current_score = score3
                            current = small_mut
                            best_score = max(best_score, current_score)
                            best_cell = current
            except:
                break
    
    # Phase 3: Local search refinement
    if best_cell is not None and time.time() < deadline - 20:
        current = best_cell
        current_score = best_score
        mutation_rate = 0.06
        
        for iteration in range(100):
            if time.time() > deadline - 10:
                break
            
            mutated = _mutate(current, mutation_rate)
            try:
                score = _compute_gap_fast(mutated, n_k=12, pw_radius=11)
                if score > current_score:
                    current_score = score
                    current = mutated
                    best_score = score
                    best_cell = current
                    mutation_rate = min(0.12, mutation_rate * 1.05)
                else:
                    mutation_rate = max(0.02, mutation_rate * 0.95)
            except:
                pass
    
    if best_cell is None:
        best_cell = _gen_rod(0.15)
    
    return best_cell
