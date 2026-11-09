"""
Strategy: Dense Multi-Scale Fixed RBF Basis.
Unlike the current sparse single-scale approach, this strategy uses a dense grid of 
Gaussian centers with multiple fixed scales (widths). The model is linear in the 
weights, ensuring BFGS from all-ones converges to the global minimum for this basis.
The multi-scale nature allows the model to capture both low-frequency trends (large 
scale Gaussians) and high-frequency details (small scale Gaussians) of the unknown 
target function.
"""
import numpy as np
import time

# Pre-compute the fixed basis centers and scales
# We define a grid of centers and assign scales to them.
# To keep the parameter count at 10, we must be careful. 
# Wait, the contract says params is an array of length 10.
# This is a CRITICAL CONSTRAINT.
# I cannot use 20-30 basis functions if I only have 10 parameters.
# 
# Let's re-read the contract: "The evaluator fits params, an array of length 10".
# This means I can ONLY have 10 degrees of freedom.
#
# The current program uses 5 weights for 5 fixed Gaussians.
# If I use more basis functions, I need more weights, which I don't have.
# Unless... I use a different parameterization.
#
# Can I use the 10 parameters to define the centers and weights in a coupled way?
# No, if I make centers depend on params, the problem becomes non-convex and BFGS 
# will likely fail from all-ones.
#
# So I am stuck with 10 parameters.
# The current program uses 5 weights and ignores the other 5.
# Can I do better with 10 parameters?
#
# Idea: Use 10 basis functions, but link their weights or centers?
# Or, use a different basis entirely that is richer with 10 coefficients.
#
# Let's look at the Inspiration 1: "Adaptive Multi-Scale RBF... parameterize weights as exp(w_i)".
# It scored 1.46. The current one scored 9.38.
# Why did the current one score so poorly?
# Maybe the fixed centers (0,0), (0.6,0.6)... are bad for the target function?
# Or maybe the single scale 0.6 is too wide or too narrow?
#
# Let's try to use all 10 parameters effectively.
# Option A: 10 fixed Gaussians with 10 weights.
# Option B: 5 fixed Gaussians with 5 weights + 5 "shape" parameters? 
#           If I allow sigmas to vary, it's non-convex.
#           But if I fix sigmas and use 10 centers/weights, it's linear.
#
# Let's try Option A: 10 fixed Gaussians.
# We can place them on a 3x3 grid (9 points) + 1 center, or a 4x4 grid subset.
# A dense grid of 10 points in [-1, 1]x[-1, 1] allows for more flexibility.
#
# Let's define 10 centers:
# Grid points:
# (-0.75, -0.75), (0, -0.75), (0.75, -0.75)
# (-0.75,  0),    (0, 0),     (0.75, 0)
# (-0.75, 0.75),  (0, 0.75),  (0.75, 0.75)
# (0, 0) is duplicated? No, let's use a 3x3 grid (9 points) and one extra at (0,0) with a different scale?
# But if scales are fixed, we just have 9 or 10 centers.
#
# Let's use a 3x3 grid of centers. That's 9 parameters.
# What is the 10th parameter?
# Maybe a global bias? No, the Gaussians at the center can handle bias.
# Maybe a 10th Gaussian with a very large scale to act as a global trend?
#
# Let's define:
# 9 Gaussians on a 3x3 grid: centers in {-0.7, 0, 0.7} x {-0.7, 0, 0.7}.
# Scale: sigma = 0.5 (moderate).
# 1 Gaussian at (0,0) with scale sigma = 1.0 (large, global trend).
#
# Total 10 weights.
# This is linear in params.
# BFGS will solve the linear least squares problem for these 10 weights.
# This should be much more robust and expressive than 5 Gaussians.
#
# Let's refine the centers and scales.
# The domain is [-1, 1].
# Centers:
# x_coords = [-0.75, 0.0, 0.75]
# y_coords = [-0.75, 0.0, 0.75]
#
# Basis functions:
# 1-9: exp(-((x-cx)^2 + (y-cy)^2) / (2 * 0.5^2))
# 10:  exp(-((x-0)^2 + (y-0)^2) / (2 * 1.2^2))
#
# This allows the model to fit local features (via the 9 small Gaussians) 
# and global trends (via the 1 large Gaussian).
#
# Why is this different from the current program?
# Current: 5 Gaussians, single scale, sparse centers.
# New: 10 Gaussians, multi-scale (effectively, by having one large one), dense centers.
#
# Is it materially different? Yes, it uses a denser basis and utilizes all 10 parameters 
# for the weights, rather than ignoring half of them. The increased capacity and 
# better spatial coverage should lead to a lower MSE.

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    
    # Ensure params is a flat array of length 10
    # The evaluator passes params, we assume it's np.ndarray
    w = params[:10]
    
    # If params is not length 10, pad or truncate? 
    # Contract says length 10. We assume it is.
    
    # Define centers
    # 3x3 grid
    xs = np.array([-0.75, 0.0, 0.75])
    ys = np.array([-0.75, 0.0, 0.75])
    
    # Pre-compute basis matrix? 
    # x and y are arrays of shape (N,). 
    # We can compute the 10 basis values for all N points at once.
    
    pred = np.zeros_like(x)
    
    # Scales
    sigma_small = 0.5
    sigma_large = 1.2
    
    # 9 small Gaussians
    for i in range(3):
        for j in range(3):
            cx = xs[i]
            cy = ys[j]
            dist_sq = (x - cx)**2 + (y - cy)**2
            exponent = -dist_sq / (2.0 * sigma_small**2)
            # Index for weight: i*3 + j
            pred += w[i*3 + j] * np.exp(exponent)
            
    # 1 large Gaussian
    cx = 0.0
    cy = 0.0
    dist_sq = (x - cx)**2 + (y - cy)**2
    exponent = -dist_sq / (2.0 * sigma_large**2)
    pred += w[9] * np.exp(exponent)
    
    # Safety check for time
    if time.time() - start_time > 10:
         # Fallback to simple linear if something goes wrong
         return params[0] * x + params[1] * y + params[2]
         
    return pred
