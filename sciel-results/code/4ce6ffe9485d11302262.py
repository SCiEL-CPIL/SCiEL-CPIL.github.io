import numpy as np
import time
import math

_START = time.time()
_DEADLINE = 225.0  # 5s safety margin

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _get_orbit(i, j, size):
    """Get the orbit of pixel (i, j) under D4 symmetry."""
    s = size
    pts = [
        (i, j),
        (j, i),
        (s-1-i, j),
        (s-1-j, i),
        (s-1-i, s-1-j),
        (s-1-j, s-1-i),
        (i, s-1-j),
        (j, s-1-i)
    ]
    # Remove duplicates
    return list(set(pts))

def _apply_flip(cell, orbit):
    """Flip pixels in the orbit. Returns new cell."""
    new_cell = np.array(cell, dtype=int)
    for i, j in orbit:
        new_cell[i, j] = 1 - new_cell[i, j]
    return new_cell

def _compute_gap(cell, eps_low, eps_high, N_pw=17, n_k=25):
    """
    Compute the 4-5 TM band gap ratio.
    cell: 32x32 array of 0/1.
    """
    size = len(cell)
    # 1. Compute epsilon inverse grid
    eps_grid = np.where(cell > 0.5, eps_high, eps_low)
    eps_inv = 1.0 / eps_grid
    
    # 2. Fourier coefficients of eps_inv
    # fft2 gives coefficients for frequencies 0..size-1
    # We need G indices in [-N//2, N//2]
    F = np.fft.fft2(eps_inv)
    # Normalize: coeff = F / size^2
    coeff = F / (size * size)
    
    half = N_pw // 2
    idx = np.arange(-half, half + 1)
    
    # Map idx to FFT indices
    # FFT index k corresponds to frequency k if k < size/2 else k - size
    # We want frequencies -8..8
    fft_indices = []
    for g in idx:
        if g >= 0:
            fft_indices.append(g)
        else:
            fft_indices.append(g + size)
    fft_indices = np.array(fft_indices)
    
    # Extract the NxN block of coefficients
    # E[a, b] = coeff[fft_indices[a], fft_indices[b]]
    E = coeff[np.ix_(fft_indices, fft_indices)]
    
    # G vectors in units of 2pi/a
    # Gx = idx * 2pi, Gy = idx * 2pi
    # But in the matrix formula, we use (k-G) dot (k-G).
    # If we work in units where k is in 2pi/a, then G is integer idx.
    # (k-G)^2 is in (2pi/a)^2.
    # The eigenvalue lambda = (omega a / 2pi c)^2.
    
    # Precompute G components
    Gx = idx
    Gy = idx
    
    # K-points in irreducible BZ (Triangle Gamma-X-M)
    # Gamma (0,0), X (1,0), M (1,1) in units of 2pi/a?
    # Square BZ: -pi/a to pi/a. Irreducible: 0 <= kx <= ky <= pi/a?
    # Or 0 <= ky <= kx <= pi/a?
    # Standard: Triangle with vertices (0,0), (pi/a, 0), (pi/a, pi/a).
    # Let's use kx in [0, 1], ky in [0, kx] where 1 = pi/a?
    # Wait, BZ boundary is pi/a.
    # Let's parameterize kx, ky in [0, 1] representing [0, pi/a].
    # Triangle: 0 <= ky <= kx <= 1.
    
    k_points = []
    for i in range(n_k):
        kx = (i + 0.5) / n_k
        for j in range(n_k):
            ky = (j + 0.5) / n_k
            if ky <= kx:
                k_points.append((kx, ky))
    
    # Also include high-symmetry points
    k_points.append((0.0, 0.0)) # Gamma
    k_points.append((1.0, 0.0)) # X
    k_points.append((1.0, 1.0)) # M
    k_points.append((0.5, 0.0))
    k_points.append((0.5, 0.5))
    
    max_o4 = 0.0
    min_o5 = float('inf')
    
    P = N_pw * N_pw
    
    for kx, ky in k_points:
        # Build Matrix H
        # H[p, q] = (k-G_p)^2 * E[p, q] ?
        # No, E is indexed by G_p - G_q.
        # E[a, b] corresponds to G difference (idx[a], idx[b]).
        # G_p = (Gx[a_p], Gy[b_p]). G_q = (Gx[a_q], Gy[b_q]).
        # Diff = (Gx[a_p]-Gx[a_q], Gy[b_p]-Gy[b_q]).
        # Index in E: a_diff = a_p - a_q, b_diff = b_p - b_q.
        # We stored E such that E[i, j] is coeff for diff (idx[i], idx[j]).
        # So we need E[a_p - a_q + half, b_p - b_q + half].
        
        # Let's precompute the difference index map
        # diff_idx[da, db] = index in E or -1 if out of range
        # da, db in -2*half..2*half
        # If out of range, coeff is 0 (approximation, valid for small N_pw relative to size)
        
        # Actually, simpler:
        # Loop over p, q
        # p = a_p * N_pw + b_p
        # q = a_q * N_pw + b_q
        
        # Vectorized construction of H
        # H is P x P
        # H[p, q] = (k-G_p)^2 * E[diff(p, q)]
        
        # (k-G_p)^2 depends only on p.
        # Let's create an array kg_sq of size P
        kg_sq = np.zeros(P)
        for a in range(N_pw):
            for b in range(N_pw):
                p = a * N_pw + b
                dx = kx - Gx[a]
                dy = ky - Gy[b]
                kg_sq[p] = dx*dx + dy*dy
        
        # E_diff[p, q]
        # We need to extract E[diff]
        # diff_a = a_p - a_q
        # diff_b = b_p - b_q
        
        # Create matrices A_p, B_p, A_q, B_q
        a_idx = np.arange(N_pw)
        b_idx = np.arange(N_pw)
        A_p, B_p = np.meshgrid(a_idx, b_idx, indexing='ij') # A_p[a, b] = a
        A_q, B_q = np.meshgrid(a_idx, b_idx, indexing='ij')
        
        # Flatten p and q
        # p = a_p * N_pw + b_p
        # We need H[p, q]
        
        # Let's compute the difference indices for all p, q
        # This is a P x P matrix
        # diff_a[p, q] = a_p - a_q
        # We can use broadcasting
        a_p_vec = np.repeat(A_p.flatten(), N_pw) # Wait, p = a*N+b.
        # Let's just use indices
        p_idx = np.arange(P)
        a_p_all = p_idx // N_pw
        b_p_all = p_idx % N_pw
        
        a_q_all = np.arange(P) // N_pw
        b_q_all = np.arange(P) % N_pw
        
        diff_a = a_p_all[:, None] - a_q_all[None, :]
        diff_b = b_p_all[:, None] - b_q_all[None, :]
        
        # Map to E indices
        # E index i corresponds to diff idx[i]
        # We need i such that idx[i] == diff_a
        # idx is -half..half
        # i = diff_a + half
        i_a = diff_a + half
        i_b = diff_b + half
        
        # Mask out of bounds
        mask = (i_a >= 0) & (i_a < N_pw) & (i_b >= 0) & (i_b < N_pw)
        
        # Initialize E_diff
        E_diff = np.zeros((P, P), dtype=complex)
        
        # Vectorized indexing
        # We need E[i_a, i_b] where mask is true
        # E is N_pw x N_pw
        # We can use np.take or advanced indexing
        # But i_a, i_b are P x P. E is N_pw x N_pw.
        # We can't directly index E with P x P arrays unless we flatten E?
        # No, E[i, j] is scalar.
        # We need to map (i_a, i_b) to E values.
        
        # This is effectively a gather operation.
        # Let's do it in a loop over the valid diffs? No, too slow.
        # Alternative: Construct E_diff using the fact that E is small.
        # For each valid (ia, ib) in E, find all p, q such that diff is (idx[ia], idx[ib]).
        
        # Simpler: Just loop over the N_pw^2 terms of E and add to H?
        # H = sum_{ia, ib} E[ia, ib] * (k-G_p)^2 * Delta(diff == (idx[ia], idx[ib]))
        # This is sparse.
        
        # Let's stick to the direct matrix construction but optimized.
        # Since N_pw=17, P=289. P^2 = 83521.
        # Creating diff_a, diff_b is fast.
        # Mapping to E values:
        # We can create a lookup table for E.
        # E_flat = E.flatten()
        # index_in_E = i_a * N_pw + i_b
        # E_diff = E_flat[index_in_E] where mask is true, else 0.
        
        idx_E = i_a * N_pw + i_b
        # Set invalid to 0
        idx_E[~mask] = 0
        E_diff = E.flatten()[idx_E]
        
        # H = kg_sq[:, None] * E_diff
        H = kg_sq[:, None] * E_diff
        
        # Symmetrize
        H = 0.5 * (H + H.conj().T)
        
        # Eigenvalues
        try:
            w = np.linalg.eigvalsh(H)
        except np.linalg.LinAlgError:
            continue
        
        w = np.sort(np.real(w))
        if len(w) < 5:
            continue
            
        o4 = math.sqrt(max(w[3], 0.0))
        o5 = math.sqrt(max(w[4], 0.0))
        
        if o4 > max_o4:
            max_o4 = o4
        if o5 < min_o5:
            min_o5 = o5
            
    if max_o4 <= 0 or min_o5 <= 0:
        return -1.0
        
    return 2.0 * (min_o5 - max_o4) / (min_o5 + max_o4)

def design(problem):
    global _START
    _START = time.time()
    
    size = problem['pixels']
    eps_low = problem['eps_low']
    eps_high = problem['eps_high']
    
    # Initial candidates
    # 1. Circle
    cell1 = np.zeros((size, size), dtype=int)
    y, x = np.mgrid[0:size, 0:size]
    cx, cy = size/2, size/2
    r = 10
    cell1[(x-cx)**2 + (y-cy)**2 < r*r] = 1
    
    # 2. Square
    cell2 = np.zeros((size, size), dtype=int)
    cell2[8:24, 8:24] = 1
    
    # 3. Cross
    cell3 = np.zeros((size, size), dtype=int)
    cell3[14:18, :] = 1
    cell3[:, 14:18] = 1
    
    # 4. Ring
    cell4 = np.zeros((size, size), dtype=int)
    dist = (x-cx)**2 + (y-cy)**2
    cell4[(dist > 6*6) & (dist < 12*12)] = 1
    
    candidates = [cell1, cell2, cell3, cell4]
    
    best_cell = None
    best_score = -1.0
    current_cell = None
    current_score = -1.0
    
    # Evaluate candidates
    for c in candidates:
        if _time_left() < 10:
            break
        try:
            s = _compute_gap(c, eps_low, eps_high, N_pw=17, n_k=20)
            if s > best_score:
                best_score = s
                best_cell = c
                current_cell = c
                current_score = s
        except Exception:
            pass
    
    if current_cell is None:
        current_cell = candidates[0]
        current_score = _compute_gap(current_cell, eps_low, eps_high, N_pw=17, n_k=20)
        best_cell = current_cell
        best_score = current_score

    # Simulated Annealing
    T = 0.05
    T_min = 0.001
    cooling = 0.99
    
    # Precompute orbits? No, just generate on fly.
    # Random seed for reproducibility? No, random is fine.
    
    while _time_left() > 15.0:
        # Pick random pixel
        i = np.random.randint(0, size)
        j = np.random.randint(0, size)
        orbit = _get_orbit(i, j, size)
        
        new_cell = _apply_flip(current_cell, orbit)
        
        try:
            new_score = _compute_gap(new_cell, eps_low, eps_high, N_pw=17, n_k=20)
        except Exception:
            new_score = -1.0
            
        delta = new_score - current_score
        
        if delta > 0:
            current_cell = new_cell
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_cell = new_cell
        else:
            # Accept with probability exp(delta / T)
            if delta / T > -50: # Avoid underflow
                p = math.exp(delta / T)
                if np.random.random() < p:
                    current_cell = new_cell
                    current_score = new_score
            
        T *= cooling
        if T < T_min:
            T = T_min
            
    # Final verification with higher precision if time allows
    if _time_left() > 30.0 and best_cell is not None:
        try:
            final_score = _compute_gap(best_cell, eps_low, eps_high, N_pw=21, n_k=30)
            # If final score is significantly different, we might want to do a bit more search?
            # For now, just return.
        except Exception:
            pass
            
    # Convert to nested list
    return best_cell.tolist()
