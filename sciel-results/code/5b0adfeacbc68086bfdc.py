# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model using a global piecewise linear 'hat function' (B-spline) basis
    with fixed, evenly spaced knots across [-2, 2], plus a linear trend term.
    
    The basis is strictly linear in all 10 parameters, so BFGS converges to
    the global least-squares solution quickly and reliably.
    
    Design:
      - 10 hat functions with centers evenly spaced from -2 to 2.
      - Hat i has support [center_i - 0.4, center_i + 0.4] (width 0.8).
      - Adjacent hats overlap by 0.0 (centers are 0.444... apart, so support
        width 0.8 means overlap of ~0.355). Actually let me recalculate:
        centers: -2, -1.556, -1.111, -0.667, -0.222, 0.222, 0.667, 1.111, 1.556, 2
        spacing = 4/9 ≈ 0.444
        If hat width is 0.8, each hat spans ±0.4 from its center.
        Adjacent centers are 0.444 apart, so overlap is 0.8 - 0.444 = 0.356. Good.
      
      Additionally, to handle boundary effects better and provide a smooth
      global trend, we can include the hat functions which already cover
      the full domain.
      
      To make the basis more flexible, let's use a slightly wider support
      so that consecutive hats overlap more, ensuring smooth interpolation.
      
      Actually, for a proper linear basis that can represent any continuous
      piecewise linear function with knots at the centers, we should use
      "tent" functions that peak at each knot and go to zero at neighboring
      knots. With 10 knots evenly spaced, the tent for knot i is:
        - 0 for x < x_{i-1}
        - (x - x_{i-1}) / (x_i - x_{i-1}) for x_{i-1} <= x <= x_i
        - (x_{i+1} - x) / (x_{i+1} - x_i) for x_i <= x <= x_{i+1}
        - 0 for x > x_{i+1}
      
      But with only 10 parameters and 10 tents, the boundary tents are
      half-tents. This is a standard linear B-spline basis.
      
      Let me define knots at: -2, -1.556, -1.111, -0.667, -0.222, 0.222, 
      0.667, 1.111, 1.556, 2
      
      For interior knots, the tent goes from previous knot to next knot.
      For boundary knots, we extend the tent to the domain boundary.
      
      Actually, a simpler and more robust approach: use 10 Gaussian-like
      but piecewise linear "bump" functions that are:
        - Linear rise from left support edge to center
        - Linear fall from center to right support edge
      
      Let's use support width = 2 * spacing = 2 * (4/9) = 8/9 ≈ 0.889
      So each hat has support [center - 4/9, center + 4/9].
      
      For the first center at -2: support is [-2 - 4/9, -2 + 4/9] but we clip
      to [-2, 2]. So it's a half-hat on the left.
      Similarly for the last center at 2.
      
      This gives a partition of unity (approximately) and good approximation
      properties.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Define 10 evenly spaced centers in [-2, 2]
    centers = np.linspace(-2.0, 2.0, 10)
    spacing = centers[1] - centers[0]  # 4/9
    half_width = spacing  # support is [center - spacing, center + spacing]
    
    result = np.zeros_like(x, dtype=np.float64)
    
    for i in range(10):
        c = centers[i]
        left = c - half_width
        right = c + half_width
        
        # Compute the tent function value for each x
        # Tent is 0 outside [left, right]
        # Rises linearly from 0 at left to 1 at c
        # Falls linearly from 1 at c to 0 at right
        
        # For efficiency, compute in vectorized form
        # Left slope: 1 / half_width
        # Right slope: -1 / half_width
        
        # value = max(0, 1 - |x - c| / half_width)
        dist = np.abs(x - c)
        tent = np.maximum(0.0, 1.0 - dist / half_width)
        
        result += params[i] * tent
    
    # Ensure finite output
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
