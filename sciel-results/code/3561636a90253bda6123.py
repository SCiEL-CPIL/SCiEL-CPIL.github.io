import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _enforce_symmetry(size, base):
    """
    Enforces full C4v symmetry (square symmetry group).
    Input: base (size x size array of 0/1)
    Output: cell (size x size list of lists of 0/1)
    """
    b = np.array(base, dtype=np.int8)
    N = size
    cell = np.zeros((N, N), dtype=np.int8)
    processed = np.zeros((N, N), dtype=bool)
    
    for i in range(N):
        for j in range(N):
            if processed[i, j]:
                continue
            
            # Generate the 8 symmetric positions for (i, j)
            orbit = [
                (i, j),
                (N-1-i, j),
                (j, i),
                (N-1-i, N-1-j),
                (i, N-1-j),
                (j, N-1-i),
                (N-1-j, i),
                (N-1-j, N-1-i)
            ]
            
            # Remove duplicates
            unique_orbit = list(set(orbit))
            
            # Majority vote from input base
            vals = [b[r, c] for r, c in unique_orbit]
            if sum(vals) >= len(vals) / 2.0:
                val = 1
            else:
                val = 0
            
            # Assign value to all unique positions in the orbit
            for r, c in unique_orbit:
                cell[r, c] = val
                processed[r, c] = True
                
    return cell.tolist()

def _get_fundamental_domain(size):
    """Get the list of unique orbit representatives for C4v symmetry."""
    N = size
    processed = np.zeros((N, N), dtype=bool)
    representatives = []
    
    for i in range(N):
        for j in range(N):
            if processed[i, j]:
                continue
            
            orbit = [
                (i, j),
                (N-1-i, j),
                (j, i),
                (N-1-i, N-1-j),
                (i, N-1-j),
                (j, N-1-i),
                (N-1-j, i),
                (N-1-j, N-1-i)
            ]
            
            unique_orbit = list(set(orbit))
            representatives.append(unique_orbit)
            
            for r, c in unique_orbit:
                processed[r, c] = True
                
    return representatives

def _compute_surrogate_gap(cell, size, n_pw, n_kx, n_ky, eps_high=12.0, eps_low=1.0):
    """
    Fast plane-wave TM band calculation surrogate.
    """
    # Build reciprocal lattice vectors
    G_range = range(-n_pw, n_pw + 1)
    Gs = np.array([(mx, my) for mx in G_range for my in G_range], dtype=float)
    nG = len(Gs)
    
    # Compute inv_eps Fourier coefficients using FFT
    cell_arr = np.array(cell, dtype=float)
    eps_arr = np.where(cell_arr > 0.5, eps_high, eps_low)
    inv_eps_arr = 1.0 / eps_arr
    
    inv_eps_fft = np.fft.fft2(inv_eps_arr)
    
    # Build inv_eps matrix: M[i,j] = inv_eps(G_i - G_j)
    dmx = Gs[:, 0][:, None] - Gs[:, 0][None, :]
    dmy = Gs[:, 1][:, None] - Gs[:, 1][None, :]
    
    diff_bins_row = dmx % size
    diff_bins_col = dmy % size
    
    inv_eps_mat = inv_eps_fft[diff_bins_row, diff_bins_col]
    
    # K-points: Irreducible BZ of square lattice
    k_points = []
    k_points.append((0.0, 0.0))
    for t in np.linspace(0.1, 1.0, n_kx):
        k_points.append((t * np.pi, 0.0))
    for t in np.linspace(0.1, 1.0, n_ky):
        k_points.append((t * np.pi, t * np.pi))
    for t in np.linspace(0.1, 1.0, n_kx):
        k_points.append((np.pi, t * np.pi))
    
    max_band4 = 0.0
    min_band5 = np.inf
    
    for kx, ky in k_points:
        k = np.array([kx, ky])
        kGs = k + Gs
        
        kGs_x = kGs[:, 0]
        kGs_y = kGs[:, 1]
        dot = np.outer(kGs_x, kGs_x) + np.outer(kGs_y, kGs_y)
        
        A = dot * inv_eps_mat
        A = (A + A.conj().T) / 2.0
        
        eigvals = np.linalg.eigvalsh(A)
        eigvals = np.real(eigvals)
        eigvals = eigvals[eigvals > 1e-10]
        
        if len(eigvals) >= 5:
            b4 = eigvals[3]
            b5 = eigvals[4]
            max_band4 = max(max_band4, b4)
            min_band5 = min(min_band5, b5)
    
    if min_band5 == np.inf or max_band4 == 0:
        return -1.0
    
    gap = min_band5 - max_band4
    midgap = 0.5 * (min_band5 + max_band4)
    if midgap <= 0:
        return -1.0
    return 2.0 * gap / midgap

def _generate_initial_candidates(size):
    """Generate diverse parametric candidates."""
    candidates = []
    center = size // 2
    
    def make_cell_from_mask(mask_func, bg_val, shape_val):
        base = np.full((size, size), bg_val, dtype=np.int8)
        for i in range(size):
            for j in range(size):
                if mask_func(i, j):
                    base[i, j] = shape_val
        return _enforce_symmetry(size, base)
    
    # 1. Single central hole in dielectric background
    for r in [3.0, 4.0, 5.0, 6.0]:
        def mask(i, j, r=r):
            dx = i - center
            dy = j - center
            return dx*dx + dy*dy <= r*r
        candidates.append(make_cell_from_mask(mask, 1, 0))
        
    # 2. Four holes in dielectric background
    for d in [5.0, 6.0, 7.0]:
        for r in [1.5, 2.0, 2.5]:
            def mask(i, j, d=d, r=r):
                dx = i - center
                dy = j - center
                if dx*dx + (dy-d)*(dy-d) <= r*r: return True
                if dx*dx + (dy+d)*(dy+d) <= r*r: return True
                if (dx-d)*(dx-d) + dy*dy <= r*r: return True
                if (dx+d)*(dx+d) + dy*dy <= r*r: return True
                return False
            candidates.append(make_cell_from_mask(mask, 1, 0))
            
    # 3. Cross shape in air background
    for w in [2.0, 3.0, 4.0]:
        for l in [6.0, 8.0, 10.0]:
            def mask(i, j, w=w, l=l):
                dx = abs(i - center)
                dy = abs(j - center)
                if dx < l and dy < w: return True
                if dy < l and dx < w: return True
                return False
            candidates.append(make_cell_from_mask(mask, 0, 1))
    
    # 4. Square ring (dielectric square with air hole)
    for outer in [8.0, 10.0, 12.0]:
        for inner in [4.0, 5.0, 6.0]:
            def mask(i, j, outer=outer, inner=inner):
                dx = abs(i - center)
                dy = abs(j - center)
                in_outer = dx < outer and dy < outer
                in_inner = dx < inner and dy < inner
                return in_outer and not in_inner
            candidates.append(make_cell_from_mask(mask, 0, 1))
    
    # 5. Dielectric rods in air (4 rods)
    for d in [5.0, 6.0, 7.0]:
        for r in [1.5, 2.0, 2.5]:
            def mask(i, j, d=d, r=r):
                dx = i - center
                dy = j - center
                if (dx-d)**2 + dy*dy <= r*r: return True
                if (dx+d)**2 + dy*dy <= r*r: return True
                if dx*dx + (dy-d)**2 <= r*r: return True
                if dx*dx + (dy+d)**2 <= r*r: return True
                return False
            candidates.append(make_cell_from_mask(mask, 0, 1))
    
    return candidates

def _local_search(cell, size, reps, n_pw, n_kx, n_ky, eps_high, eps_low, deadline, rng, max_iter=50):
    """Local search on fundamental domain pixels."""
    current_cell = np.array(cell, dtype=np.int8)
    current_score = _compute_surrogate_gap(current_cell.tolist(), size, n_pw, n_kx, n_ky, eps_high, eps_low)
    best_cell = current_cell.copy()
    best_score = current_score
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick a random orbit representative to flip
        idx = rng.integers(0, len(reps))
        orbit = reps[idx]
        
        # Flip the representative pixel
        mut_cell = current_cell.copy()
        r, c = orbit[0]
        mut_cell[r, c] = 1 - mut_cell[r, c]
        
        # Enforce symmetry
        sym_cell = np.array(_enforce_symmetry(size, mut_cell), dtype=np.int8)
        
        # Evaluate
        try:
            score = _compute_surrogate_gap(sym_cell.tolist(), size, n_pw, n_kx, n_ky, eps_high, eps_low)
        except Exception:
            score = -1.0
        
        if score > current_score:
            current_cell = sym_cell
            current_score = score
            if score > best_score:
                best_score = score
                best_cell = sym_cell.copy()
        else:
            # Occasionally try a different mutation strategy
            if rng.random() < 0.3:
                # Flip 2 random orbits
                mut_cell = current_cell.copy()
                for _ in range(2):
                    idx2 = rng.integers(0, len(reps))
                    orbit2 = reps[idx2]
                    r2, c2 = orbit2[0]
                    mut_cell[r2, c2] = 1 - mut_cell[r2, c2]
                sym_cell = np.array(_enforce_symmetry(size, mut_cell), dtype=np.int8)
                try:
                    score = _compute_surrogate_gap(sym_cell.tolist(), size, n_pw, n_kx, n_ky, eps_high, eps_low)
                except Exception:
                    score = -1.0
                if score > current_score:
                    current_cell = sym_cell
                    current_score = score
                    if score > best_score:
                        best_score = score
                        best_cell = sym_cell.copy()
    
    return best_cell.tolist(), best_score

def design(problem):
    """Time-bounded local search for TM band gap maximization."""
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12.0)
    eps_low = problem.get("eps_low", 1.0)
    
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin (evaluator kills at 230s)
    
    # Get fundamental domain
    reps = _get_fundamental_domain(size)
    
    # Generate initial candidates
    candidates = _generate_initial_candidates(size)
    
    best_cell = None
    best_score = -1.0
    
    # Phase 1: Coarse screening with fast surrogate (n_pw=3, fewer k-points)
    for cell in candidates:
        if time.time() > deadline:
            break
        try:
            score = _compute_surrogate_gap(cell, size, n_pw=3, n_kx=4, n_ky=4, eps_high=eps_high, eps_low=eps_low)
        except Exception:
            score = -1.0
        if score > best_score:
            best_score = score
            best_cell = cell
    
    if best_cell is None:
        # Fallback
        base = np.zeros((size, size), dtype=np.int8)
        center = size // 2
        for i in range(size):
            for j in range(size):
                dx = i - center
                dy = j - center
                if dx*dx + dy*dy < 25:
                    base[i, j] = 1
        best_cell = _enforce_symmetry(size, base)
        best_score = -1.0
    
    # Phase 2: Local search with faster surrogate
    rng = np.random.default_rng(42)
    if time.time() < deadline - 10.0:
        best_cell, best_score = _local_search(
            best_cell, size, reps, n_pw=4, n_kx=5, n_ky=5,
            eps_high=eps_high, eps_low=eps_low,
            deadline=deadline - 10.0, rng=rng, max_iter=80
        )
    
    # Phase 3: Refine top candidate with full-resolution surrogate
    if time.time() < deadline:
        try:
            refined_score = _compute_surrogate_gap(best_cell, size, n_pw=8, n_kx=6, n_ky=6, eps_high=eps_high, eps_low=eps_low)
        except Exception:
            refined_score = -1.0
        
        if refined_score > best_score:
            best_score = refined_score
    
    return best_cell

# EVOLVE-BLOCK-END
