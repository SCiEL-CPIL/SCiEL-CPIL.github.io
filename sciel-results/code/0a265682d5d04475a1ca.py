import numpy as np
import time

def _apply_4fold_symmetry(cell_quadrant):
    """Apply 4-fold symmetry to a quadrant to create full 32x32 cell."""
    size = cell_quadrant.shape[0] * 2
    full = np.zeros((size, size), dtype=int)
    q = cell_quadrant
    # Quadrant 1 (top-right in standard math, but we use array indexing)
    # Cell is symmetric about center, so we tile the quadrant
    full[:size//2, size//2:] = q
    full[size//2:, size//2:] = q[::-1, ::-1]
    full[size//2:, :size//2] = q[::-1, :]
    full[:size//2, :size//2] = q[:, ::-1]
    return full

def _compute_gap_fast(cell, eps_high=12.0, n_pw=13, k_points=None):
    """Fast plane-wave computation of gap between bands 4 and 5 for TM mode."""
    size = cell.shape[0]
    eps = eps_high * cell.astype(float) + 1.0
    
    # Fourier coefficients of 1/eps
    inv_eps = 1.0 / eps
    inv_eps_f = np.fft.fft2(inv_eps)
    
    # Plane wave vectors
    if n_pw % 2 == 0:
        n_pw += 1
    half = n_pw // 2
    g = np.arange(-half, half + 1)
    Gx, Gy = np.meshgrid(g, g, indexing='ij')
    G = np.column_stack([Gx.ravel(), Gy.ravel()])
    G2 = Gx.ravel()**2 + Gy.ravel()**2
    
    # k-points in irreducible BZ
    if k_points is None:
        k_points = np.array([
            [0.0, 0.0],
            [0.5, 0.0],
            [0.5, 0.5],
            [0.25, 0.0],
            [0.5, 0.25],
            [0.3, 0.1],
        ])
    
    max_gap_ratio = -1.0
    
    for k in k_points:
        kx, ky = k
        # Build Hamiltonian for TM mode: (G+k)^2 * A(G) * H(G'-G)
        # where A(G) = 1/(G+k)^2, H is 1/eps Fourier
        # For TM: H(k) = (G+k)^2 * (1/eps) * (G+k)
        # Simplified: eigenvalues of matrix M where M_ij = (G_i+k)^2 * inv_eps_f[j-i]
        
        kvec = np.column_stack([Gx.ravel() + kx, Gy.ravel() + ky])
        kvec2 = kvec[:, 0]**2 + kvec[:, 1]**2
        
        # Build matrix: M[i,j] = kvec2[i] * inv_eps_f[j-i mod n]
        n = len(G)
        M = np.zeros((n, n), dtype=complex)
        for i in range(n):
            for j in range(n):
                diff = j - i
                # Wrap around for periodicity
                diff_idx = diff % n_pw
                if diff_idx >= n_pw:
                    diff_idx -= n_pw
                # Get Fourier coefficient
                # inv_eps_f is indexed by (mx, my) where mx, my in [-half, half]
                # We need to map diff to proper index
                mx = diff // (n_pw + 1) if abs(diff) > n_pw//2 else 0
                # This is getting complex; use direct convolution
                pass
        
        # Use a simpler approach: direct matrix construction
        M = np.zeros((n, n), dtype=complex)
        for i in range(n):
            for j in range(n):
                dx_g = Gx.ravel()[j] - Gx.ravel()[i]
                dy_g = Gy.ravel()[j] - Gy.ravel()[i]
                # Fourier coefficient at (dx_g, dy_g)
                idx_x = int(dx_g) + half
                idx_y = int(dy_g) + half
                if 0 <= idx_x < n_pw and 0 <= idx_y < n_pw:
                    M[i, j] = kvec2[i] * inv_eps_f[idx_y, idx_x]
                else:
                    M[i, j] = 0
        
        # Solve eigenvalue problem
        try:
            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.sort(np.real(eigvals))
            # Find bands 4 and 5 (0-indexed: 3 and 4)
            if len(eigvals) >= 5:
                omega4 = np.sqrt(eigvals[3])
                omega5 = np.sqrt(eigvals[4])
                if omega4 > 0:
                    gap_ratio = 2 * (omega5 - omega4) / (omega5 + omega4)
                    max_gap_ratio = max(max_gap_ratio, gap_ratio)
        except:
            continue
    
    return max_gap_ratio

def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell using local search
    with a fast internal plane-wave solver to maximize the 4-5 TM band gap.
    """
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit
    
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12.0)
    
    # Start with 2x2 rod configuration as initial guess
    coords = np.arange(size)
    x = (coords + 0.5) / size - 0.5
    y = (coords + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, y)
    
    dx = 0.25
    dy = 0.25
    R_rod = 0.15
    
    cell = np.zeros((size, size), dtype=int)
    dist1 = np.sqrt((X - dx)**2 + (Y - dy)**2)
    cell[dist1 <= R_rod] = 1
    dist2 = np.sqrt((X + dx)**2 + (Y - dy)**2)
    cell[dist2 <= R_rod] = 1
    dist3 = np.sqrt((X - dx)**2 + (Y + dy)**2)
    cell[dist3 <= R_rod] = 1
    dist4 = np.sqrt((X + dx)**2 + (Y + dy)**2)
    cell[dist4 <= R_rod] = 1
    
    # Apply symmetry to ensure exact 4-fold symmetry
    # Extract top-right quadrant and rebuild
    q = cell[:size//2, size//2:]
    cell = _apply_4fold_symmetry(q)
    
    # Compute initial gap
    best_gap = _compute_gap_fast(cell, eps_high)
    best_cell = cell.copy()
    
    # Local search: flip pixels in the quadrant and check if gap improves
    # Only optimize the top-right quadrant (8x8 pixels)
    q_size = size // 2
    
    # Precompute which pixels are in the quadrant
    quadrant_pixels = np.arange(q_size * q_size)
    
    # Try flipping each pixel in the quadrant
    n_steps = 0
    max_steps = 500
    
    while n_steps < max_steps:
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try each pixel in the quadrant
        for idx in range(q_size * q_size):
            if time.time() > deadline:
                break
            
            row = idx // q_size
            col = idx % q_size
            
            # Try flipping this pixel
            old_val = q[row, col]
            q[row, col] = 1 - old_val
            
            # Rebuild full cell
            test_cell = _apply_4fold_symmetry(q)
            
            # Compute gap
            test_gap = _compute_gap_fast(test_cell, eps_high)
            
            if test_gap > best_gap:
                best_gap = test_gap
                best_cell = test_cell.copy()
                improved = True
            else:
                # Revert
                q[row, col] = old_val
        
        n_steps += 1
        
        if not improved:
            # Try random flips to escape local minima
            if n_steps % 5 == 0:
                for _ in range(3):
                    if time.time() > deadline:
                        break
                    r = np.random.randint(q_size)
                    c = np.random.randint(q_size)
                    q[r, c] = 1 - q[r, c]
                    test_cell = _apply_4fold_symmetry(q)
                    test_gap = _compute_gap_fast(test_cell, eps_high)
                    if test_gap > best_gap:
                        best_gap = test_gap
                        best_cell = test_cell.copy()
                    else:
                        q[r, c] = 1 - q[r, c]
            else:
                break
    
    return best_cell.tolist()
