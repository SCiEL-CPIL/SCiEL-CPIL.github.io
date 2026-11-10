# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _compute_union_volume(positions, nx=64, ny=64, nz=32):
    """
    Compute the union volume using a voxel grid.
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
    
    # For each voxel, check if it's inside any tube
    # A voxel at (a, b, c) is inside tube k if:
    #   x_k + c*i_k/N <= a <= x_k + c*i_k/N + 1/N
    #   y_k + c*j_k/N <= b <= y_k + c*j_k/N + 1/N
    
    # Reshape for broadcasting: positions is (64, 2), i_flat/j_flat are (64,)
    # We need to check all 64 tubes for all voxels
    
    # Vectorized: for each tube, compute the region in (a,b,c) space
    # Tube k occupies: a in [x_k + c*i_k/N, x_k + c*i_k/N + 1/N]
    #                 b in [y_k + c*j_k/N, y_k + c*j_k/N + 1/N]
    
    # Let's do it tube by tube but vectorized over voxels
    occupied = np.zeros((nx, ny, nz), dtype=bool)
    
    for k in range(num_tubes):
        xk = positions[k, 0]
        yk = positions[k, 1]
        ik = i_flat[k]
        jk = j_flat[k]
        
        # At height c, tube k occupies:
        # a in [xk + c*ik/N, xk + c*ik/N + 1/N]
        # b in [yk + c*jk/N, yk + c*jk/N + 1/N]
        
        # For voxel at (a, b, c):
        # xk + c*ik/N <= a <= xk + c*ik/N + 1/N
        # => a - 1/N <= xk + c*ik/N <= a
        # => a - 1/N - c*ik/N <= xk <= a - c*ik/N
        
        # Similarly for y.
        
        # Check condition for all voxels
        cond_x = (XX >= xk + ZZ * ik / N) & (XX <= xk + ZZ * ik / N + 1.0 / N)
        cond_y = (YY >= yk + ZZ * jk / N) & (YY <= yk + ZZ * jk / N + 1.0 / N)
        
        occupied |= (cond_x & cond_y)
    
    num_occupied = np.sum(occupied)
    return num_occupied * voxel_vol


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Uses voxel-based morphological representation and gradient optimization.
    """
    N = 8
    num_tubes = 64
    
    i_flat = np.repeat(np.arange(N), N)
    j_flat = np.tile(np.arange(N), N)
    
    # Initial positions: use a spread-out configuration
    np.random.seed(42)
    positions = np.random.uniform(0, 1, (num_tubes, 2))
    
    # Coarse voxel grid for optimization
    nx, ny, nz = 32, 32, 16
    
    # Objective function: minimize union volume
    def objective(pos_flat):
        pos = pos_flat.reshape(num_tubes, 2)
        vol = _compute_union_volume(pos, nx, ny, nz)
        return vol
    
    # Use L-BFGS with numerical gradient
    result = minimize(
        objective,
        positions.flatten(),
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    opt_positions = result.x.reshape(num_tubes, 2)
    
    # Verify the result
    final_vol = _compute_union_volume(opt_positions, nx, ny, nz)
    if final_vol <= 0:
        # Fallback to a simple configuration
        opt_positions = np.random.uniform(0, 1, (num_tubes, 2))
    
    return opt_positions


# EVOLVE-BLOCK-END
