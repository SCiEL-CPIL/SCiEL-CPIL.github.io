import math
import time
import numpy as np

def design(problem):
    """
    Design a symmetric 32x32-pixel unit cell to maximize the TM band gap
    between bands 4 and 5.
    """
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]   # 1
    eps_high = problem["eps_high"] # 12
    
    # Deadline guard: 230s wall-clock limit, use 220s for safety
    start_time = time.time()
    deadline = start_time + 220
    
    N = 32
    
    def quadrant_to_full(q):
        """Convert a 16x16 quadrant to a full 32x32 symmetric cell."""
        q = np.array(q, dtype=int)
        top = np.hstack([q, q[:, ::-1]])
        bottom = top[::-1, :]
        return np.vstack([top, bottom])
    
    def full_to_quadrant(cell):
        """Extract the top-left 16x16 quadrant from a full 32x32 cell."""
        return np.array(cell[:16, :16], dtype=int)
    
    # Precompute Fourier coefficients for a given cell
    # We'll use a moderate plane-wave cutoff. The evaluator uses 289 (17x17), 
    # but we can use a slightly smaller one for speed if needed, or match it.
    # Let's use npw=8 (17x17 = 289 waves) to match the evaluator's resolution roughly.
    npw = 8
    mx = np.arange(-npw, npw+1)
    my = np.arange(-npw, npw+1)
    Gx, Gy = np.meshgrid(mx, my, indexing='ij')
    Gx = Gx.ravel()
    Gy = Gy.ravel()
    G = np.column_stack((Gx, Gy))
    nG = len(G)
    
    # Precompute the G difference mapping
    # G_diff[i, j] = G[i] - G[j]
    # We need the index in G for G_diff[i, j] if it exists.
    G_dict = {}
    for idx in range(nG):
        G_dict[(G[idx, 0], G[idx, 1])] = idx
    
    # Precompute the indices for the matrix A construction
    # A[i, j] = inv_eps_coeffs[index of G[i]-G[j]]
    # Let's precompute the indices or the G_diff array.
    G_diff = G[:, None, :] - G[None, :, :]
    # G_diff shape (nG, nG, 2)
    # We can flatten the last two dims to use as keys? No, just loop or vectorize with searchsorted?
    # Since nG=289, a loop is fine, but we can precompute the indices.
    A_indices = np.full((nG, nG), -1, dtype=int)
    for i in range(nG):
        for j in range(nG):
            key = (G_diff[i, j, 0], G_diff[i, j, 1])
            if key in G_dict:
                A_indices[i, j] = G_dict[key]
    
    # Precompute k-points
    # The evaluator uses 30 k-points. Let's use a dense grid in the irreducible BZ.
    # Irreducible BZ: 0 <= ky <= kx <= pi
    # We'll generate a set of points.
    k_points = []
    # High symmetry points
    k_points.append((0.0, 0.0))
    k_points.append((np.pi, 0.0))
    k_points.append((np.pi/2, np.pi/2))
    k_points.append((np.pi/2, 0.0))
    # Grid points
    for ix in range(1, 8):
        kx = np.pi * ix / 8.0
        for iy in range(0, ix + 1):
            ky = np.pi * iy / 8.0
            k_points.append((kx, ky))
    # Ensure unique points (remove duplicates if any)
    unique_k = []
    for k in k_points:
        if k not in unique_k:
            unique_k.append(k)
    k_points = unique_k
    
    # Precompute k+G and kG_sq for each k-point
    kG_sq_list = []
    for kx, ky in k_points:
        kvec = np.array([kx, ky])
        k_plus_G = kvec[None, :] + G
        kG_sq = np.sum(k_plus_G**2, axis=1)
        kG_sq_list.append(kG_sq)
    
    def compute_gap(cell_2d):
        """
        Compute the gap-midgap ratio between TM bands 4 and 5.
        """
        eps = np.where(np.array(cell_2d) == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        
        # Compute Fourier coefficients of 1/epsilon
        # x, y are pixel centers
        x = (np.arange(N) + 0.5) / N
        y = (np.arange(N) + 0.5) / N
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Phase shape (nG, N, N)
        phase = -1j * (Gx[:, None, None] * X[None, :, :] + Gy[:, None, None] * Y[None, :, :])
        exp_term = np.exp(phase)
        
        inv_eps_coeffs = np.sum(inv_eps[None, :, :] * exp_term, axis=(1, 2)) / (N * N)
        
        # Construct A matrix
        # A[i, j] = inv_eps_coeffs[A_indices[i, j]] if A_indices[i, j] != -1 else 0
        # Vectorized:
        A = np.zeros((nG, nG), dtype=complex)
        valid = A_indices >= 0
        A[valid] = inv_eps_coeffs[A_indices[valid]]
        
        max_omega4 = 0.0
        min_omega5 = np.inf
        
        for idx_k, kG_sq in enumerate(kG_sq_list):
            H = A * kG_sq[None, :]
            H = (H + H.conj().T) / 2
            
            # Solve eigenvalue problem
            # We only need the 4th and 5th bands (indices 3 and 4)
            # Using eigvalsh is fast enough for 289x289.
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals.real)
            
            if len(eigvals) >= 6:
                omega4 = eigvals[3]
                omega5 = eigvals[4]
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        
        if min_omega5 == np.inf or max_omega4 == 0:
            return 0.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2
        
        if midgap <= 0:
            return 0.0
        
        return gap / midgap
    
    # Initial candidates
    candidates = []
    
    # 1. Square Pillar (centered at 8,8 in quadrant)
    # The full cell has a pillar at (8,8), (8,24), (24,8), (24,24) if we mirror.
    # Wait, quadrant is top-left 16x16.
    # Mirror across x=16 and y=16.
    # So a blob at (8,8) in quadrant becomes blobs at (8,8), (8,24), (24,8), (24,24) in full cell.
    # Let's try a square pillar of size 4x4 centered at (8,8).
    q_square = np.zeros((16, 16), dtype=int)
    # Center at 8,8. Size 4x4 means indices 6,7,8,9? Or 7,8,9,10?
    # Let's try 4x4: rows 6-9, cols 6-9.
    q_square[6:10, 6:10] = 1
    candidates.append(q_square)
    
    # 2. Larger Square Pillar 6x6
    q_square2 = np.zeros((16, 16), dtype=int)
    q_square2[5:11, 5:11] = 1
    candidates.append(q_square2)
    
    # 3. Circular Rod
    X, Y = np.meshgrid(np.arange(16), np.arange(16), indexing='ij')
    q_circle = ((X - 8)**2 + (Y - 8)**2 < 4**2).astype(int)
    candidates.append(q_circle)
    
    # 4. Air Holes (Dielectric background)
    # Mostly 1s, with holes.
    q_hole = np.ones((16, 16), dtype=int)
    # Hole at center?
    q_hole[6:10, 6:10] = 0
    candidates.append(q_hole)
    
    # 5. Ring
    q_ring = np.zeros((16, 16), dtype=int)
    dist = np.sqrt((X - 8)**2 + (Y - 8)**2)
    q_ring[(dist > 2) & (dist < 5)] = 1
    candidates.append(q_ring)
    
    # 6. Random Seeds
    np.random.seed(42)
    for _ in range(10):
        q_rand = (np.random.rand(16, 16) > 0.5).astype(int)
        candidates.append(q_rand)
    
    # Evaluate candidates
    best_score = -np.inf
    best_quadrant = candidates[0]
    
    for q in candidates:
        full_cell = quadrant_to_full(q)
        score = compute_gap(full_cell)
        if score > best_score:
            best_score = score
            best_quadrant = q.copy()
    
    # Local Search (Hill Climbing)
    current_q = best_quadrant.copy()
    current_score = best_score
    
    # Precompute pixel indices for the quadrant
    # We will flip random pixels.
    # To speed up, we can precompute the change in inv_eps_coeffs for a single pixel flip.
    # inv_eps_coeffs[g] = sum(inv_eps[x,y] * exp(-i G . r)) / N^2
    # If we flip pixel (i,j) from 0 to 1 (air to dielectric):
    # inv_eps changes from 1 to 1/12. Delta inv_eps = 1/12 - 1 = -11/12.
    # Delta coeff[g] = Delta inv_eps * exp(-i G . r_ij) / N^2
    # This allows O(nG) update instead of O(N^2 * nG).
    
    # Let's implement this incremental update for speed.
    # inv_eps_coeffs is (nG,).
    # pixel_phase[g] = exp(-i G[g] . r_ij)
    
    # Precompute phases for all pixels in the quadrant?
    # Quadrant pixels map to full cell pixels.
    # If we flip a pixel in the quadrant, it affects 4 pixels in the full cell (due to symmetry).
    # Actually, the quadrant_to_full function mirrors.
    # If we flip (i,j) in quadrant, it flips (i,j), (i, 31-j), (31-i, j), (31-i, 31-j) in full cell.
    # So we need to update inv_eps_coeffs for these 4 positions.
    
    # Let's precompute the phases for all 32x32 pixels.
    x_full = (np.arange(N) + 0.5) / N
    y_full = (np.arange(N) + 0.5) / N
    Xf, Yf = np.meshgrid(x_full, y_full, indexing='ij')
    phase_full = -1j * (Gx[:, None, None] * Xf[None, :, :] + Gy[:, None, None] * Yf[None, :, :])
    exp_full = np.exp(phase_full) # shape (nG, 32, 32)
    
    # For a flip at (i,j) in quadrant:
    # Affected full cell coords:
    # (i, j), (i, 31-j), (31-i, j), (31-i, 31-j)
    # Note: if i == 31-i (i=15.5, impossible) or j == 31-j (j=15.5, impossible), no double counting.
    # Since N=32, indices 0..31. i, j in 0..15.
    # 31-i is 16..31. Distinct.
    
    # Let's store the current inv_eps_coeffs.
    # We need to recompute it for the initial best_quadrant.
    
    def get_inv_eps_coeffs(quadrant):
        full_cell = quadrant_to_full(quadrant)
        eps = np.where(full_cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        # Use precomputed exp_full?
        # inv_eps_coeffs = sum(inv_eps * exp_full) / N^2
        # exp_full is (nG, 32, 32). inv_eps is (32, 32).
        # Broadcasting: inv_eps[None, :, :] * exp_full
        return np.sum(inv_eps[None, :, :] * exp_full, axis=(1, 2)) / (N * N)
    
    current_inv_eps_coeffs = get_inv_eps_coeffs(current_q)
    
    def update_inv_eps_coeffs(coeffs, quadrant, i, j):
        """Update coeffs when flipping pixel (i,j) in quadrant."""
        # Get old value
        old_val = quadrant[i, j]
        new_val = 1 - old_val
        
        # Determine delta inv_eps
        if old_val == 0: # Air -> Dielectric
            delta_inv_eps = 1.0/eps_high - 1.0/eps_low
        else: # Dielectric -> Air
            delta_inv_eps = 1.0/eps_low - 1.0/eps_high
        
        # Affected pixels in full cell
        coords = [
            (i, j),
            (i, 31-j),
            (31-i, j),
            (31-i, 31-j)
        ]
        
        # Update coeffs
        # coeffs += delta_inv_eps * sum(exp_full[:, r, c] for (r,c) in coords) / N^2
        delta_coeffs = np.zeros(nG, dtype=complex)
        for r, c in coords:
            delta_coeffs += exp_full[:, r, c]
        
        coeffs += delta_inv_eps * delta_coeffs / (N * N)
        
        return coeffs
    
    # In the loop, we will:
    # 1. Flip pixel
    # 2. Update coeffs
    # 3. Compute gap
    # 4. If worse, revert pixel and coeffs
    
    # To compute gap, we need A, which depends on coeffs.
    # A[valid] = coeffs[A_indices[valid]]
    # Then for each k-point, H = A * kG_sq, eigvalsh.
    
    # We can optimize the gap computation slightly by precomputing A_indices and valid mask.
    valid_mask = A_indices >= 0
    
    def compute_gap_from_coeffs(coeffs):
        A = np.zeros((nG, nG), dtype=complex)
        A[valid_mask] = coeffs[A_indices[valid_mask]]
        
        max_omega4 = 0.0
        min_omega5 = np.inf
        
        for idx_k, kG_sq in enumerate(kG_sq_list):
            H = A * kG_sq[None, :]
            H = (H + H.conj().T) / 2
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals.real)
            if len(eigvals) >= 6:
                omega4 = eigvals[3]
                omega5 = eigvals[4]
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        
        if min_omega5 == np.inf or max_omega4 == 0:
            return 0.0
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2
        if midgap <= 0:
            return 0.0
        return gap / midgap
    
    # Main Loop
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline - 5:
        i = np.random.randint(0, 16)
        j = np.random.randint(0, 16)
        
        old_val = current_q[i, j]
        current_q[i, j] = 1 - old_val
        
        # Update coeffs
        new_coeffs = update_inv_eps_coeffs(current_inv_eps_coeffs, current_q, i, j)
        # Note: update_inv_eps_coeffs modifies coeffs in place if we pass a copy, 
        # but here I wrote it to return a new array? 
        # "coeffs += ..." modifies in place. 
        # Let's make a copy to be safe, or revert manually.
        # Better: make a copy of coeffs, update it, test, then revert if needed.
        
        # Actually, update_inv_eps_coeffs as written modifies the array passed to it.
        # So let's pass a copy.
        # This is expensive (copy of 289 complex numbers).
        # Alternative: store delta, apply, test, revert.
        
        # Let's refactor update to return delta.
        
        # Re-implementing the flip logic for efficiency:
        # 1. Calculate delta_coeffs
        # 2. Apply to current_inv_eps_coeffs
        # 3. Compute score
        # 4. If bad, revert delta_coeffs and pixel
        
        # Let's do that.
        
        # First, revert the pixel change in current_q? No, we changed it.
        # Let's calculate delta_coeffs for the flip.
        # We already changed current_q[i,j].
        # So we need the delta corresponding to old_val -> new_val.
        
        # Let's just stick to the copy method for simplicity, nG=289 is small.
        # But copying 289 complex numbers is ~2KB, very fast.
        
        # Wait, I already modified current_q.
        # I need to revert current_q[i,j] if score is bad.
        
        # Let's re-evaluate the current state (with the flip) using the updated coeffs.
        # But I need the updated coeffs.
        # Let's just recompute coeffs from scratch? No, too slow.
        
        # Let's use the delta method properly.
        
        # 1. Flip pixel in current_q (Done)
        # 2. Calculate delta_coeffs for this flip
        old_val_for_delta = 1 - current_q[i, j] # The value before flip
        new_val_for_delta = current_q[i, j]     # The value after flip
        
        if old_val_for_delta == 0:
            delta_inv_eps = 1.0/eps_high - 1.0/eps_low
        else:
            delta_inv_eps = 1.0/eps_low - 1.0/eps_high
            
        coords = [
            (i, j),
            (i, 31-j),
            (31-i, j),
            (31-i, 31-j)
        ]
        
        delta_coeffs = np.zeros(nG, dtype=complex)
        for r, c in coords:
            delta_coeffs += exp_full[:, r, c]
        delta_coeffs *= delta_inv_eps / (N * N)
        
        # Apply delta
        current_inv_eps_coeffs += delta_coeffs
        
        # Compute score
        score = compute_gap_from_coeffs(current_inv_eps_coeffs)
        
        if score > current_score:
            current_score = score
            no_improve_count = 0
            if score > best_score:
                best_score = score
                best_quadrant = current_q.copy()
        else:
            # Revert
            current_inv_eps_coeffs -= delta_coeffs
            current_q[i, j] = old_val
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Restart from best? Or random restart?
                # Let's try a few random restarts if stuck.
                # Or just keep going.
                pass
    
    best_full_cell = quadrant_to_full(best_quadrant)
    return best_full_cell.tolist()
