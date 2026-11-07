"""
Rewritten program: Multi-scale Radial Basis Function (RBF) basis for symbolic regression.

Strategy:
The previous approach used a low-dimensional Chebyshev polynomial basis (10 features).
This approach switches to a high-capacity RBF basis. We define a large number of
Gaussian basis functions centered at a grid of points in [-1, 1] x [-1, 1].
The 10 parameters are linear weights applied to these 100 basis functions.

Why this is different:
1. Capacity: 100 basis functions vs 10 polynomial terms.
2. Locality: RBFs are local, allowing the model to fit complex, non-smooth, or
   highly oscillatory functions that polynomials cannot easily represent without
   high degrees (which cause ill-conditioning).
3. Flexibility: The fixed centers cover the domain, and the weights determine
   the shape. BFGS can optimize these 10 weights to match the target function
   by selecting the appropriate linear combination of local bumps.

Implementation Details:
- 10x10 grid of centers in [-1, 1]^2.
- Gaussian width sigma = 0.25 (covers roughly 2 units, so neighbors interact).
- The equation computes the sum of params[i] * sum_of_basis_functions_in_group_i.
  To keep the output dimensionality 10, we group the 100 basis functions into
  10 groups (e.g., by row or column of the grid) and let each param weight a group.
  Alternatively, and more simply, we can just use 10 specific centers or a
  subset. However, with only 10 params, we can't weight 100 centers independently.
  
  Revised Strategy for 10 Params:
  We will use 10 distinct RBF centers. To maximize coverage, we place them
  strategically. A common choice is a set of points that covers the domain
  well. We can use a 3x3 grid plus the center, or a 4x4 grid subset.
  Let's use 10 centers:
  (-1, -1), (-0.5, -1), (0, -1), (0.5, -1), (1, -1)
  (-1, 0),  (0, 0),  (1, 0)
  (-1, 1),  (1, 1)
  
  Actually, a more robust approach with 10 params is to use a smaller set of
  well-distributed centers. Let's use a 3x3 grid (9 points) + 1 center point?
  No, let's use 10 points from a 5x5 grid?
  
  Let's stick to a simple, well-distributed set of 10 centers:
  c0: (-0.8, -0.8)
  c1: (0.0, -0.8)
  c2: (0.8, -0.8)
  c3: (-0.8, 0.0)
  c4: (0.0, 0.0)
  c5: (0.8, 0.0)
  c6: (-0.8, 0.8)
  c7: (0.0, 0.8)
  c8: (0.8, 0.8)
  c9: (0.0, 0.0) -- duplicate? No, let's use (0.0, 0.0) once.
  
  Let's use a 3x3 grid (9 points) and one additional point at (0,0) is already there.
  So 9 points from 3x3 grid:
  x_coords = [-0.8, 0.0, 0.8]
  y_coords = [-0.8, 0.0, 0.8]
  This gives 9 centers. We have 10 params. We can add one more center, e.g., (0.4, 0.4) or just use a 4x4 grid subset.
  
  Let's use a 4x4 grid of centers but select 10 of them?
  Or simply use 10 centers defined as:
  centers = [
      (-0.75, -0.75), (0.0, -0.75), (0.75, -0.75),
      (-0.75, 0.0),   (0.0, 0.0),    (0.75, 0.0),
      (-0.75, 0.75),  (0.0, 0.75),   (0.75, 0.75),
      (0.0, 0.0)      # This is a duplicate of center 4.
  ]
  Duplicates are bad for identifiability.
  
  Let's use 10 distinct centers:
  A 3x3 grid gives 9. Add one at (0.5, 0.5)?
  
  Actually, a better strategy with 10 params is to use 10 different scales or
  10 different centers.
  
  Let's define 10 centers:
  1. (-1, -1)
  2. (1, -1)
  3. (-1, 1)
  4. (1, 1)
  5. (-1, 0)
  6. (1, 0)
  7. (0, -1)
  8. (0, 1)
  9. (0, 0)
  10. (0.5, 0.5)
  
  Sigma = 0.5.
  
  The function f(x, y) = sum_{i=1}^{10} params[i] * exp(-((x-x_i)^2 + (y-y_i)^2) / (2*sigma^2))
  
  This is a linear model in the parameters. BFGS will optimize the 10 weights.
  The basis is fixed.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a Radial Basis Function (RBF) basis with 10 centers.
    
    The 10 parameters are weights for 10 Gaussian basis functions centered at
    specific points in the [-1, 1] x [-1, 1] domain.
    """
    # Define 10 centers
    centers = np.array([
        [-1.0, -1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [1.0, 1.0],
        [-1.0, 0.0],
        [1.0, 0.0],
        [0.0, -1.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.5, 0.5]
    ])
    
    # Gaussian width
    sigma = 0.5
    sigma_sq_2 = 2.0 * sigma * sigma
    
    # x and y are 1D arrays of length N
    # We need to compute the distance from each point (x[i], y[i]) to each center (cx, cy)
    # dist_sq[i, j] = (x[i] - cx[j])^2 + (y[i] - cy[j])^2
    
    # x shape: (N,)
    # y shape: (N,)
    # centers shape: (10, 2)
    
    # Expand dimensions for broadcasting
    x_col = x[:, np.newaxis]  # (N, 1)
    y_col = y[:, np.newaxis]  # (N, 1)
    
    cx = centers[:, 0]  # (10,)
    cy = centers[:, 1]  # (10,)
    
    # Compute squared distances
    # (N, 1) - (1, 10) -> (N, 10)
    dx = x_col - cx[np.newaxis, :]
    dy = y_col - cy[np.newaxis, :]
    
    dist_sq = dx**2 + dy**2  # (N, 10)
    
    # Compute Gaussian values
    # exp(-dist_sq / (2 * sigma^2))
    rbf_values = np.exp(-dist_sq / sigma_sq_2)  # (N, 10)
    
    # Combine with parameters
    # params shape: (10,)
    # rbf_values shape: (N, 10)
    # Result shape: (N,)
    f = rbf_values @ params
    
    return f
# EVOLVE-BLOCK-END
