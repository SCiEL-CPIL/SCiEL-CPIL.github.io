import numpy as np
from scipy.optimize import differential_evolution

def calculate_1d_union_length(X, c, N):
    """
    Calculate the length of the union of intervals [X_i + c*i/N, X_i + c*i/N + 1/N] for i=0..N-1.
    X: array of shape (N,)
    c: scalar
    N: int
    """
    starts = X + (c * np.arange(N) / N)
    ends = starts + 1.0 / N
    
    # Sort by start
    order = np.argsort(starts)
    s = starts[order]
    e = ends[order]
    
    # Merge intervals
    # Since N is small (8), we can just iterate
    union_len = 0.0
    current_start = s[0]
    current_end = e[0]
    for i in range(1, N):
        if s[i] > current_end:
            union_len += current_end - current_start
            current_start = s[i]
            current_end = e[i]
        else:
            if e[i] > current_end:
                current_end = e[i]
    union_len += current_end - current_start
    return union_len

def objective_1d(X, N=8, num_c=100):
    """
    Objective function to minimize: integral of union length over c in [0,1].
    We approximate the integral with a Riemann sum.
    """
    X = np.asarray(X)
    # Symmetrize or center? The problem is translation invariant in x and y separately?
    # The box X0, X1 is defined by min/max of the tube positions.
    # Shifting all X by a constant shifts the box but not the relative geometry or the volume.
    # So we can fix one variable, e.g., X[0] = 0, to reduce dimensionality and help optimizer.
    # But differential_evolution handles unconstrained well.
    
    c_vals = np.linspace(0, 1, num_c, endpoint=False) + 0.5/num_c
    total_len = 0.0
    for c in c_vals:
        total_len += calculate_1d_union_length(X, c, N)
    return total_len / num_c

def optimize_1d_positions(N=8, seed=12345):
    # Initial bounds: The positions should be within a reasonable range.
    # The width of the union is at most 1 (if they are disjoint) and at least 1/N (if they overlap completely).
    # The shearing adds up to 1 unit of shift.
    # So X_i should be in [-1, 1] roughly.
    bounds = [(-1.0, 1.0) for _ in range(N)]
    
    # Use differential evolution for global search
    # popsize=15, maxiter=1000 might be too slow?
    # N=8 variables.
    # Let's try a smaller population and iterations to fit in time budget.
    # 600s budget.
    
    res = differential_evolution(
        objective_1d, 
        bounds, 
        seed=seed,
        maxiter=200,
        popsize=10,
        tol=1e-6,
        polish=True,
        args=(N,)
    )
    
    return res.x

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    
    # Optimize X
    X_opt = optimize_1d_positions(N, seed=42)
    # Optimize Y
    Y_opt = optimize_1d_positions(N, seed=43)
    
    # Construct the 64x2 array
    # k = i*N + j
    # x_k = X_opt[i]
    # y_k = Y_opt[j]
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = X_opt[i_flat]
    base_y = Y_opt[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    return positions
