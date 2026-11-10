# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _compute_union_volume_fast(positions, nx=64, ny=64, nz=32):
    """
    Compute the union volume using a voxel grid with vectorized tube checks.
    positions: (64, 2) array of base positions (x_k, y_k)
    Returns: estimated union volume
    """
    N = 8
    num_tubes = 64
    
    # Compute bounding box
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
    
    # Process tubes in batches for memory efficiency
    occupied = np.zeros((nx, ny, nz), dtype=bool)
    
    batch_size = 16
    for start in range(0, num_tubes, batch_size):
        end = min(start + batch_size, num_tubes)
        
        xk_batch = positions[start:end, 0]
        yk_batch = positions[start:end, 1]
        ik_batch = i_flat[start:end]
        jk_batch = j_flat[start:end]
        
        # For each tube in batch, compute conditions
        # xk + c*ik/N <= XX <= xk + c*ik/N + 1/N
        # yk + c*jk/N <= YY <= yk + c*jk/N + 1/N
        
        # Reshape for broadcasting: (batch, 1, 1, 1), (1, nx, ny, nz)
        xk_r = xk_batch[:, np.newaxis, np.newaxis, np.newaxis]
        yk_r = yk_batch[:, np.newaxis, np.newaxis, np.newaxis]
        ik_r = ik_batch[:, np.newaxis, np.newaxis, np.newaxis]
        jk_r = jk_batch[:, np.newaxis, np.newaxis, np.newaxis]
        ZZ_r = ZZ[np.newaxis, :, :, :]
        
        # Compute left and right boundaries for x
        x_left = xk_r + ZZ_r * ik_r / N
        x_right = x_left + 1.0 / N
        y_left = yk_r + ZZ_r * jk_r / N
        y_right = y_left + 1.0 / N
        
        # Check if voxel center is inside any tube in this batch
        in_x = (XX[np.newaxis, :, :, :] >= x_left) & (XX[np.newaxis, :, :, :] <= x_right)
        in_y = (YY[np.newaxis, :, :, :] >= y_left) & (YY[np.newaxis, :, :, :] <= y_right)
        
        # Any tube in batch contains this voxel
        in_any = np.any(in_x & in_y, axis=0)
        
        occupied |= in_any
    
    num_occupied = np.sum(occupied)
    return num_occupied * voxel_vol


def _initial_positions_structured():
    """
    Generate structured initial positions that encourage overlap.
    """
    N = 8
    num_tubes = 64
    
    positions = np.zeros((num_tubes, 2))
    
    # Use a pattern where tubes with similar i,j values are close together
    # This encourages overlap between tubes with similar shear directions
    for i in range(N):
        for j in range(N):
            k = i * N + j
            # Place tubes in a grid pattern with some randomness
            # Tubes with same i are close in x, same j are close in y
            positions[k, 0] = 0.1 + 0.8 * (i / (N - 1)) + np.random.uniform(-0.05, 0.05)
            positions[k, 1] = 0.1 + 0.8 * (j / (N - 1)) + np.random.uniform(-0.05, 0.05)
    
    return positions


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Uses multi-stage voxel-based optimization with structured initialization.
    """
    N = 8
    num_tubes = 64
    
    # Stage 1: Coarse optimization with structured initialization
    np.random.seed(42)
    positions = _initial_positions_structured()
    
    # Coarse grid
    nx_coarse, ny_coarse, nz_coarse = 24, 24, 12
    
    def objective_coarse(pos_flat):
        pos = pos_flat.reshape(num_tubes, 2)
        vol = _compute_union_volume_fast(pos, nx_coarse, ny_coarse, nz_coarse)
        return vol
    
    result1 = minimize(
        objective_coarse,
        positions.flatten(),
        method='L-BFGS-B',
        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    opt_pos1 = result1.x.reshape(num_tubes, 2)
    
    # Stage 2: Medium resolution refinement
    nx_med, ny_med, nz_med = 40, 40, 20
    
    def objective_med(pos_flat):
        pos = pos_flat.reshape(num_tubes, 2)
        vol = _compute_union_volume_fast(pos, nx_med, ny_med, nz_med)
        return vol
    
    result2 = minimize(
        objective_med,
        opt_pos1.flatten(),
        method='L-BFGS-B',
        options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12}
    )
    
    opt_pos2 = result2.x.reshape(num_tubes, 2)
    
    # Stage 3: Fine resolution final optimization
    nx_fine, ny_fine, nz_fine = 56, 56, 28
    
    def objective_fine(pos_flat):
        pos = pos_flat.reshape(num_tubes, 2)
        vol = _compute_union_volume_fast(pos, nx_fine, ny_fine, nz_fine)
        return vol
    
    result3 = minimize(
        objective_fine,
        opt_pos2.flatten(),
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-14}
    )
    
    opt_pos3 = result3.x.reshape(num_tubes, 2)
    
    # Verify the result is valid
    final_vol = _compute_union_volume_fast(opt_pos3, 48, 48, 24)
    if final_vol <= 0 or not np.all(np.isfinite(opt_pos3)):
        # Fallback to medium resolution result
        opt_pos3 = opt_pos2
        final_vol = _compute_union_volume_fast(opt_pos3, 48, 48, 24)
        if final_vol <= 0 or not np.all(np.isfinite(opt_pos3)):
            # Fallback to coarse result
            opt_pos3 = opt_pos1
            final_vol = _compute_union_volume_fast(opt_pos3, 48, 48, 24)
            if final_vol <= 0 or not np.all(np.isfinite(opt_pos3)):
                # Last resort: random
                opt_pos3 = np.random.uniform(0, 1, (num_tubes, 2))
    
    return opt_pos3


# EVOLVE-BLOCK-END
