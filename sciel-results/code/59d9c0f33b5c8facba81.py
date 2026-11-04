"""
Strategy: Symmetric Geometric Reduction via Fourier Radial Parameters.

The problem is to place 14 points to maximize the minimum triangle area relative to the convex hull area.
The current approach uses a 28-dimensional search or a 1D angular gap search.
This new approach hypothesizes that the optimal configuration is a perturbation of a regular 14-gon.
We parameterize the points by their angles $\theta_k = 2\pi k / 14$ and radii $r_k$.
To enforce symmetry and reduce dimensionality, we model $r_k$ using a Fourier series:
$r_k = 1 + \sum_{m=1}^{M} A_m \cos(2\pi m k / 14) + B_m \sin(2\pi m k / 14)$.
Given the rotational symmetry of the regular 14-gon, the optimal perturbation might also respect some symmetries.
However, to be safe, we include both sine and cosine terms for the first few harmonics.
For $n=14$, the fundamental frequency is 1.
Let's use $M=3$. This gives $2 \times 3 = 6$ parameters.
We can also add a global scale factor, but the score is scale-invariant, so we fix the mean radius to 1.
Actually, the convex hull area depends on the radii.
Let's use a slightly richer parameterization:
$r_k = 1 + c_1 \cos(2\pi k / 14) + s_1 \sin(2\pi k / 14) + c_2 \cos(4\pi k / 14) + s_2 \sin(4\pi k / 14) + \dots$
Since the base is a regular 14-gon, the "1st harmonic" (frequency 1) corresponds to a translation (if we allowed it) or a shift in the center of mass relative to the origin. But the points are defined relative to the origin.
Wait, if we shift the center, the convex hull area changes.
Let's stick to radial perturbations around the unit circle.
The number of parameters is small (e.g., 6-10), making global optimization feasible.
We will use Differential Evolution (DE) for the global search and L-BFGS-B for local polishing.
"""

import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

N = 14

def _score(pts):
    """
    Compute the score: min triangle area / convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(pts) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    n = len(pts)
    min_area = float('inf')
    
    # Efficient vectorized computation
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            
            # Cross product of (pj - pi) and (pk - pi)
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            v2x = pk[:, 0] - pi[0]
            v2y = pk[:, 1] - pi[1]
            
            crosses = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * crosses
            
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                
    return min_area / hull_area

def _generate_points_from_fourier(params):
    """
    Generate 14 points from Fourier radial coefficients.
    params: [c1, s1, c2, s2, c3, s3, ...]
    """
    k = np.arange(N)
    angles = 2 * np.pi * k / N
    
    # Base radius 1
    r = np.ones(N)
    
    # Add harmonics
    # We assume the parameters are ordered: c1, s1, c2, s2, ...
    # Number of harmonics M = len(params) // 2
    M = len(params) // 2
    for m in range(1, M + 1):
        c_m = params[2 * (m - 1)]
        s_m = params[2 * (m - 1) + 1]
        r += c_m * np.cos(2 * np.pi * m * k / N)
        r += s_m * np.sin(2 * np.pi * m * k / N)
        
    # Ensure radii are positive to avoid degeneracy
    # If any radius is <= 0, we penalize heavily or clip
    # Clipping might distort the symmetry, so we use a large penalty in the objective
    if np.any(r <= 0):
        return None
        
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    
    return np.column_stack([x, y])

def _objective(params):
    """
    Objective function for optimization: negative of the score.
    """
    pts = _generate_points_from_fourier(params)
    if pts is None:
        return 1e6  # Penalty for invalid radii
        
    s = _score(pts)
    if s <= 0:
        return 1e6
    return -s

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Define the search space for Fourier coefficients
    # We'll try up to 3 harmonics (6 parameters)
    # Bounds: The perturbation should be small enough to keep points in general position
    # and not collapse. A reasonable bound is [-0.5, 0.5] for coefficients.
    # If coefficients are too large, points might cross or overlap.
    bounds = [(-0.5, 0.5)] * 6
    
    # Initial guess: Regular 14-gon (all coefficients 0)
    x0 = np.zeros(6)
    
    best_score = -float('inf')
    best_params = None
    
    # 1. Global Search: Differential Evolution
    # DE is robust for low-dimensional global optimization.
    try:
        res = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            tol=1e-5,
            popsize=15,
            polish=True,  # Use L-BFGS-B for final polish
            init='sobol'
        )
        
        if time.time() < deadline:
            best_params = res.x
            current_score = -res.fun
            
            # Verify and update best
            pts = _generate_points_from_fourier(best_params)
            if pts is not None:
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts
            else:
                # Fallback to regular polygon if optimization failed
                angles = np.linspace(0, 2*np.pi, N, endpoint=False)
                best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
                best_score = _score(best_pts)
                
    except Exception as e:
        # Fallback to regular polygon if DE fails
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = _score(best_pts)
        
    # 2. Local Refinement: L-BFGS-B
    # If we have time, run a few local refinements from slightly perturbed starts
    if time.time() < deadline - 10.0:
        rng = np.random.default_rng(42)
        
        # Try a few random restarts around the best found or around 0
        for i in range(5):
            if time.time() > deadline - 5.0:
                break
                
            # Start from best_params + small noise, or from 0 + small noise
            if best_params is not None and i % 2 == 0:
                x_start = best_params + rng.normal(0, 0.05, size=6)
            else:
                x_start = rng.normal(0, 0.1, size=6)
                
            try:
                res_local = minimize(
                    _objective, 
                    x_start, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-8}
                )
                
                if time.time() > deadline - 5.0:
                    break
                    
                if res_local.success:
                    s_local = -res_local.fun
                    if s_local > best_score:
                        best_score = s_local
                        best_params = res_local.x
                        pts = _generate_points_from_fourier(best_params)
                        if pts is not None:
                            # Re-verify score
                            s_verify = _score(pts)
                            if s_verify > best_score:
                                best_score = s_verify
                                best_pts = pts
            except Exception:
                pass
                
    # Ensure we return a valid point set
    if best_params is not None:
        pts_final = _generate_points_from_fourier(best_params)
        if pts_final is not None:
            # Final check
            s_final = _score(pts_final)
            if s_final > best_score:
                best_score = s_final
            return pts_final
            
    # Fallback: Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

# EVOLVE-BLOCK-END
