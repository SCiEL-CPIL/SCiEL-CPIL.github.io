# EVOLVE-BLOCK-START
import numpy as np


def _compute_union_volume_fast(positions, nx=48, ny=48, nz=24):
    """
    Compute the union volume using a voxel grid with optimized computation.
    positions: (64, 2) array of base positions (x_k, y_k)
    Returns: estimated union volume
    """
    N = 8
    num_tubes = 64
    
    i_flat = np.repeat(np.arange(N), N)
    j_flat = np.tile(np.arange(N), N)
    
    X0 = np.min(positions[:, 0])
    X1 = np.max(positions[:, 0] + (i_flat + 1) / N)
    Y0 = np.min(positions[:, 1])
    Y1 = np.max(positions[:, 1] + (j_flat + 1) / N)
    
    box_area = (X1 - X0) * (Y1 - Y0)
    if box_area <= 1e-9:
        return 0.0
    
    # Voxel grid
    xs = np.linspace(X0, X1, nx + 1)
    ys = np.linspace(Y0, Y1, ny + 1)
    zs = np.linspace(0, 1, nz + 1)
    
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    dz = zs[1] - zs[0]
    voxel_vol = dx * dy * dz
    
    # Voxel centers
    xc = 0.5 * (xs[:-1] + xs[1:])
    yc = 0.5 * (ys[:-1] + ys[1:])
    zc = 0.5 * (zs[:-1] + zs[1:])
    
    # Create 3D grid of voxel centers
    XX, YY, ZZ = np.meshgrid(xc, yc, zc, indexing='ij')
    
    # Process tubes in batches
    batch_size = 16
    occupied = np.zeros((nx, ny, nz), dtype=bool)
    
    for start in range(0, num_tubes, batch_size):
        end = min(start + batch_size, num_tubes)
        batch_pos = positions[start:end]
        batch_i = i_flat[start:end]
        batch_j = j_flat[start:end]
        
        xk = batch_pos[:, 0].reshape(1, 1, 1, -1)
        yk = batch_pos[:, 1].reshape(1, 1, 1, -1)
        ik = batch_i.reshape(1, 1, 1, -1)
        jk = batch_j.reshape(1, 1, 1, -1)
        
        ZZ_exp = ZZ[:, :, :, np.newaxis]
        XX_exp = XX[:, :, :, np.newaxis]
        YY_exp = YY[:, :, :, np.newaxis]
        
        x_center = xk + ZZ_exp * ik / N
        y_center = yk + ZZ_exp * jk / N
        
        cond_x = (XX_exp >= x_center) & (XX_exp <= x_center + 1.0 / N)
        cond_y = (YY_exp >= y_center) & (YY_exp <= y_center + 1.0 / N)
        
        occupied |= np.any(cond_x & cond_y, axis=-1)
    
    num_occupied = np.sum(occupied)
    return num_occupied * voxel_vol


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Uses grid-based morphological representation with iterative optimization.
    """
    N = 8
    num_tubes = 64
    
    i_flat = np.repeat(np.arange(N), N)
    j_flat = np.tile(np.arange(N), N)
    
    np.random.seed(42)
    
    # Initialize: place tubes so they all roughly pass through center at c=0.5
    positions = np.zeros((num_tubes, 2))
    for i in range(N):
        for j in range(N):
            k = i * N + j
            positions[k, 0] = 0.5 - i / (2.0 * N) + 0.01 * np.random.randn()
            positions[k, 1] = 0.5 - j / (2.0 * N) + 0.01 * np.random.randn()
    
    # Coarse grid for optimization
    nx, ny, nz = 24, 24, 12
    
    # Use a simple iterative improvement: try small random perturbations
    best_positions = positions.copy()
    best_vol = _compute_union_volume_fast(best_positions, nx, ny, nz)
    
    for iteration in range(500):
        # Try a random perturbation
        perturbation = np.random.normal(0, 0.05, (num_tubes, 2))
        new_positions = best_positions + perturbation
        new_vol = _compute_union_volume_fast(new_positions, nx, ny, nz)
        
        if new_vol < best_vol:
            best_positions = new_positions
            best_vol = new_vol
    
    # Verify with finer grid
    final_vol = _compute_union_volume_fast(best_positions, 48, 48, 24)
    if final_vol <= 0:
        best_positions = np.random.uniform(0, 1, (num_tubes, 2))
    
    return best_positions


# EVOLVE-BLOCK-END
