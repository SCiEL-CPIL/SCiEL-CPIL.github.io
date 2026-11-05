import time
import numpy as np
from numpy.linalg import eigh

def _compute_gap_fast(cell, n_k=12, pw_radius=12):
    N = len(cell)
    inv_eps = 1.0 / (1.0 + 11.0 * np.array(cell, dtype=float))
    fft_inv = np.fft.fft2(inv_eps) / (N * N)
    
    G_list = []
    for Gx in range(-pw_radius, pw_radius + 1):
        for Gy in range(-pw_radius, pw_radius + 1):
            if Gx*Gx + Gy*Gy <= pw_radius * pw_radius:
                G_list.append((Gx, Gy))
    n_pw = len(G_list)
    G_arr = np.array(G_list, dtype=float)
    
    size_d = 4 * pw_radius + 1
    offset_d = 2 * pw_radius
    inv_eps_full = np.zeros((size_d, size_d), dtype=complex)
    
    Gx_diff = G_arr[:, 0][:, None] - G_arr[:, 0][None, :]
    Gy_diff = G_arr[:, 1][:, None] - G_arr[:, 1][None, :]
    
    for i in range(n_pw):
        for j in range(n_pw):
            dqx = int(Gx_diff[i, j])
            dqy = int(Gy_diff[i, j])
            inv_eps_full[dqx + offset_d, dqy + offset_d] = fft_inv[dqx % N, dqy % N]
    
    # More k-points covering the irreducible BZ better
    k_points = [
        (0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.4, 0.0), (0.5, 0.0),
        (0.5, 0.1), (0.5, 0.2), (0.5, 0.3), (0.5, 0.4), (0.5, 0.5),
        (0.3, 0.3), (0.4, 0.2), (0.2, 0.2)
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


def _make_symmetric(cell):
    """Ensure cell is symmetric under transpose, row-reverse, col-reverse."""
    N = len(cell)
    c = np.array(cell, dtype=float)
    # Average over all 8 symmetry operations of the square group
    c_t = c.T
    c_r = c[::-1, :]
    c_c = c[:, ::-1]
    c_rt = c[::-1, ::-1]
    c_tr = c.T[::-1, :]  # transpose then row-reverse
    c_tc = c.T[:, ::-1]  # transpose then col-reverse
    c_rtc = c[::-1, ::-1].T  # 180 then transpose
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


def _gen_square(r):
    """Square dielectric inclusion of half-size r."""
    size = 32
    cell = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) < r and abs(y) < r:
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


def _gen_rhombus(r):
    """Rhombus/diamond shape - good for TM band gaps."""
    size = 32
    cell = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) + abs(y) < r:
                cell[i][j] = 1
    return cell


def _gen_star(arm_width, n_arms=4):
    """Star shape with n_arms (must be even for square symmetry)."""
    size = 32
    cell = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            # Check if in any arm
            in_arm = False
            for a in range(n_arms):
                angle = 2 * np.pi * a / n_arms
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                # Rotate point to arm frame
                xr = x * cos_a + y * sin_a
                yr = -x * sin_a + y * cos_a
                if abs(yr) < arm_width/2 and abs(xr) < 0.45:
                    in_arm = True
                    break
            if in_arm:
                cell[i][j] = 1
    return cell


def _gen_random_symmetric(density):
    """Random symmetric pixel pattern respecting full square symmetry."""
    size = 32
    cell = np.zeros((size, size), dtype=int)
    # Fill the fundamental domain (top-left 1/8 of the cell)
    # For square symmetry, the fundamental domain is a 1/8 wedge
    for i in range(size//2):
        for j in range(size//2):
            if j > i:  # Only in the triangular fundamental domain
                continue
            if np.random.random() < density:
                # Mirror to all 8 symmetric positions
                cell[i, j] = 1
                cell[i, size-1-j] = 1
                cell[size-1-i, j] = 1
                cell[size-1-i, size-1-j] = 1
                cell[j, i] = 1
                cell[j, size-1-i] = 1
                cell[size-1-j, i] = 1
                cell[size-1-j, size-1-i] = 1
    return cell.tolist()


def _mutate(cell, mutation_rate=0.1):
    """Randomly flip some pixels while maintaining full square symmetry."""
    size = len(cell)
    c = np.array(cell, dtype=int)
    # Flip in the fundamental domain (1/8 of the cell)
    for i in range(size//2):
        for j in range(size//2):
            if j > i:
                continue
            if np.random.random() < mutation_rate:
                val = 1 - c[i, j]
                # Apply to all 8 symmetric positions
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
    deadline = start_time + 195.0  # Safety margin (230s limit, leave 35s buffer)
    
    size = problem["pixels"]
    assert size == 32
    
    # Initial candidates - focus on shapes known to open TM band gaps
    candidates = []
    
    # Rods - smaller rods can open higher-order gaps
    for r in [0.08, 0.12, 0.15, 0.18, 0.22, 0.25, 0.28, 0.32]:
        candidates.append(_gen_rod(r))
    
    # Rhombus shapes - often better for TM gaps
    for r in [0.15, 0.2, 0.25, 0.3, 0.35]:
        candidates.append(_gen_rhombus(r))
    
    # Square inclusions
    for r in [0.12, 0.15, 0.18, 0.22, 0.25]:
        candidates.append(_gen_square(r))
    
    # Cross shapes
    for w in [0.04, 0.08, 0.12, 0.16]:
        for l in [0.25, 0.35, 0.42, 0.48]:
            candidates.append(_gen_cross(w, l))
    
    # Star shapes
    for w in [0.06, 0.1, 0.14]:
        candidates.append(_gen_star(w))
    
    # Dielectric with hole (inverse)
    for r in [0.35, 0.4, 0.45, 0.5]:
        candidates.append(_gen_hole(r))
    
    # Random symmetric patterns at various densities
    for density in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        for _ in range(4):
            candidates.append(_gen_random_symmetric(density))
    
    # Evaluate all candidates with fast settings
    best_cell = None
    best_score = -1.0
    
    for idx, cell in enumerate(candidates):
        if time.time() > deadline - 80:
            break
        try:
            score = _compute_gap_fast(cell, n_k=10, pw_radius=10)
            if score > best_score:
                best_score = score
                best_cell = cell
        except:
            pass
    
    # Local search around the best with adaptive mutation rate
    if best_cell is not None and time.time() < deadline - 60:
        current = best_cell
        current_score = best_score
        mutation_rate = 0.08
        no_improve_count = 0
        
        for iteration in range(300):
            if time.time() > deadline - 40:
                break
            
            mutated = _mutate(current, mutation_rate)
            try:
                score = _compute_gap_fast(mutated, n_k=10, pw_radius=10)
                if score > current_score:
                    current_score = score
                    current = mutated
                    best_score = score
                    best_cell = current
                    no_improve_count = 0
                    # Slightly increase mutation rate on success to explore more
                    mutation_rate = min(0.15, mutation_rate * 1.1)
                else:
                    no_improve_count += 1
                    # Decrease mutation rate when stuck
                    mutation_rate = max(0.02, mutation_rate * 0.95)
            except:
                pass
    
    # Final polish: try a few more random candidates and re-evaluate best with higher precision
    if time.time() < deadline - 20:
        for _ in range(30):
            if time.time() > deadline - 15:
                break
            cell = _gen_random_symmetric(np.random.uniform(0.1, 0.45))
            try:
                score = _compute_gap_fast(cell, n_k=10, pw_radius=10)
                if score > best_score:
                    best_score = score
                    best_cell = cell
            except:
                pass
    
    # Re-evaluate the best with higher precision settings
    if best_cell is not None and time.time() < deadline - 10:
        try:
            final_score = _compute_gap_fast(best_cell, n_k=14, pw_radius=12)
            # Use the best cell regardless of score difference (precision vs speed tradeoff)
        except:
            pass
    
    if best_cell is None:
        # Fallback: a small rod which is known to open some gap
        best_cell = _gen_rod(0.15)
    
    return best_cell
