"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist
import time


def construct():
    """
    Construct centres of 300 equal circles inside the unit circle.
    Uses scipy.optimize.least_squares to maximize the minimum radius.
    """
    n = N
    # Initial guess: hexagonal lattice points inside a slightly smaller disk
    # to give some room for optimization
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 0.95
    points = points[mask]
    # Take the first n points
    if len(points) < n:
        # Fallback: grid
        xs = np.linspace(-0.9, 0.9, 20)
        ys = np.linspace(-0.9, 0.9, 20)
        grid = np.array([[x, y] for x in xs for y in ys])
        points = grid[:n]
    else:
        points = points[:n]
    
    x0 = points.flatten()
    
    def residuals(x):
        """
        Residuals for least_squares. We want to maximize r such that:
        1. |c_i - c_j| >= 2r for all i < j
        2. |c_i| + r <= 1 for all i
        
        We reformulate as minimizing violations. Let r be part of the variable?
        No, least_squares minimizes sum of squares of residuals.
        
        Alternative approach: Fix a target radius r and check feasibility?
        Or, minimize the sum of squared deficits from a target radius.
        
        Better approach: Minimize a function that penalizes:
        - Pairwise distances < 2r
        - Distances to boundary < r
        
        But r is what we want to maximize.
        
        Standard trick: Introduce r as a variable. Minimize -r? No, least_squares
        minimizes ||F(x)||^2.
        
        Let's use the following formulation:
        Variables: x = [x_1, y_1, ..., x_n, y_n] (centres)
        We want to maximize r.
        
        We can minimize the sum of squared "violations" for a given r, but r is unknown.
        
        Alternative: Minimize the sum of squared (2r - d_ij) for d_ij < 2r,
        and (r - (1 - |c_i|)) for |c_i| + r > 1.
        
        But r is not in the variable vector.
        
        Let's use a different formulation. Minimize:
        sum_{i<j} max(0, 2r - d_ij)^2 + sum_i max(0, r - (1 - |c_i|))^2
        where r is the minimum of (min_{i<j} d_ij / 2, min_i (1 - |c_i|)).
        
        This is still implicit in r.
        
        Simpler approach: Use a penalty method.
        Minimize: sum_{i<j} (2r - d_ij)^2 * I(d_ij < 2r) + ...
        
        Actually, let's use the fact that we can define residuals as:
        For each pair (i,j): residual = 2r - d_ij (we want this <= 0)
        For each point i: residual = r - (1 - |c_i|) (we want this <= 0)
        
        But r is not a variable.
        
        Let's try a different approach: Minimize the sum of squared distances
        that are too small.
        
        Define residuals as:
        For each pair (i,j): if d_ij < 2r, residual = 2r - d_ij, else 0
        For each point i: if 1 - |c_i| < r, residual = r - (1 - |c_i|), else 0
        
        But r depends on the configuration.
        
        Alternative: Use r as a variable.
        Variables: x = [x_1, y_1, ..., x_n, y_n, r]
        Residuals:
        For each pair (i,j): 2r - d_ij (we want this <= 0, so we square it)
        For each point i: r - (1 - |c_i|) (we want this <= 0)
        
        But we want to maximize r, not minimize the violations.
        
        If we minimize the sum of squares of these residuals, we'll get a configuration
        where the violations are minimized, but r might be small.
        
        Let's think differently. We can use a two-phase approach:
        1. Start with a reasonable r.
        2. Minimize the violations for that r.
        3. Increase r slightly and repeat.
        
        Or, use the following:
        Minimize: sum_{i<j} (2r - d_ij)^2 + sum_i (r - (1 - |c_i|))^2
        subject to d_ij >= 2r and 1 - |c_i| >= r.
        
        But least_squares doesn't handle constraints directly.
        
        Let's use the following practical approach:
        - Use a fixed r and minimize the violations.
        - Start with a small r and increase it.
        
        Actually, let's use the following formulation that is known to work:
        Minimize the sum of squared "shortfalls" from the target radius.
        
        Let r be a parameter. Define residuals:
        For each pair (i,j): residual = max(0, 2r - d_ij)
        For each point i: residual = max(0, r - (1 - |c_i|))
        
        Minimize the sum of squares of these residuals.
        
        Then, increase r until the residuals become large, indicating infeasibility.
        
        Let's implement this.
        """
        x = x[:2*n]
        centres = x.reshape(n, 2)
        
        # Compute pairwise distances
        # Use pdist for efficiency
        dists = pdist(centres)
        # Convert to matrix if needed, but we only need the values
        # For residuals, we need to flatten
        
        # Boundary distances
        boundary_dists = 1.0 - np.linalg.norm(centres, axis=1)
        
        # We need to know r. Let's estimate r from the current configuration.
        min_pair_dist = np.min(dists) if len(dists) > 0 else 1.0
        min_boundary_dist = np.min(boundary_dists)
        r_est = min(min_pair_dist / 2.0, min_boundary_dist)
        
        # Residuals for pairs
        pair_residuals = np.maximum(0, 2 * r_est - dists)
        
        # Residuals for boundary
        boundary_residuals = np.maximum(0, r_est - boundary_dists)
        
        return np.concatenate([pair_residuals, boundary_residuals])
    
    # Run least_squares
    result = least_squares(residuals, x0, method='trf', max_nfev=10000, 
                          xtol=1e-8, ftol=1e-8, gtol=1e-8)
    
    centres = result.x[:2*n].reshape(n, 2)
    
    # Compute the actual radius
    dists = pdist(centres)
    min_pair_dist = np.min(dists)
    boundary_dists = 1.0 - np.linalg.norm(centres, axis=1)
    min_boundary_dist = np.min(boundary_dists)
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    
    # Verify that all circles are inside and non-overlapping
    if r <= 0 or not np.isfinite(r):
        # Fallback to hexagonal lattice
        spacing = 2.0 / (np.sqrt(n) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= n:
                return points[np.argsort(radius, kind="stable")[:n]]
            spacing *= 0.98
    
    return centres
# EVOLVE-BLOCK-END
