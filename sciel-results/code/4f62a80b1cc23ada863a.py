"""
Optimized sphere covering with 130 points via Differentiable Soft Covering Radius.

Strategy:
Instead of Lloyd's Algorithm (which moves points to Voronoi centroids), we treat the
covering radius minimization as a smooth optimization problem. 
We define a surrogate objective: the Log-Sum-Exp (LSE) of the squared angular distances 
from a fixed set of 'test points' (dense Fibonacci sphere) to the nearest of the N points.
Minimizing this LSE with a high temperature parameter T approximates minimizing the 
maximum distance (the covering radius) while providing smooth gradients.
We use scipy.optimize.minimize with L-BFGS-B for efficient convergence.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 130
DEADLINE = time.time() + 140  # 10s safety margin for the 150s limit

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius(pts):
    """
    Compute the exact covering radius in radians using ConvexHull.
    This is the ground truth metric for evaluation.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi / 2  # Fallback if hull fails
    
    max_r = 0.0
    for simplex in hull.simplices:
        p = pts[simplex]
        a, b, c = p[0], p[1], p[2]
        v1 = b - a
        v2 = c - a
        cross = np.cross(v1, v2)
        cnorm = np.linalg.norm(cross)
        if cnorm < 1e-12:
            continue
        x = cross / cnorm
        centroid = (a + b + c) / 3.0
        if np.dot(x, centroid) < 0:
            x = -x
        dot_val = np.dot(x, a)
        dot_val = np.clip(dot_val, -1.0, 1.0)
        r = np.arccos(dot_val)
        if r > max_r:
            max_r = r
    return max_r

def _generate_test_points(M=2000):
    """Generate a dense set of test points on the sphere using Fibonacci spiral."""
    j = np.arange(M) + 0.5
    z = 1.0 - 2.0 * j / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * j
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

TEST_POINTS = _generate_test_points(2000)

def _soft_covering_objective_and_grad(x, T=100.0):
    """
    Objective: LSE of squared angular distances from TEST_POINTS to nearest x_i.
    x: flat array of shape (3N,)
    Returns: (value, gradient)
    """
    pts = x.reshape(N, 3)
    pts = _normalize(pts)
    
    # Compute dot products between TEST_POINTS (M, 3) and pts (N, 3)
    # dots[m, i] = TEST_POINTS[m] . pts[i]
    dots = TEST_POINTS @ pts.T  # (M, N)
    
    # Angular distance squared approx: 2 * (1 - dot) for small angles?
    # Actually, we want to minimize max angular distance.
    # Let d_{m,i} = arccos(dots[m,i]).
    # To make it differentiable and robust, we can minimize the LSE of (1 - dots).
    # Minimizing (1 - dots) is equivalent to maximizing dots, which minimizes angular distance.
    # Let y_{m,i} = 1 - dots[m,i].
    # We want min_m max_i y_{m,i}.
    # Surrogate: (1/T) * log(sum_i exp(T * y_{m,i}))
    
    y = 1.0 - dots  # (M, N)
    
    # For numerical stability in LSE:
    # LSE(T*y) = T * max(y) + log(sum(exp(T*(y - max(y)))))
    # We compute row-wise.
    
    y_max = np.max(y, axis=1, keepdims=True)  # (M, 1)
    # Clip y_max to avoid issues if all points are far? No, y is in [0, 2].
    
    # exp_arg = T * (y - y_max)
    # This is stable.
    exp_arg = T * (y - y_max)
    exp_arg = np.clip(exp_arg, -100, 100) # Clip for extreme stability, though T*(y-ymax) <= 0
    
    sum_exp = np.sum(np.exp(exp_arg), axis=1)  # (M,)
    
    # LSE_val_m = T * y_max[m,0] + log(sum_exp[m])
    # Note: y_max is the max of (1 - dots), which is 1 - min(dots).
    # So T * y_max = T * (1 - min_dots).
    
    lse_vals = T * y_max[:, 0] + np.log(sum_exp)
    
    # The objective is the mean of these LSE values over all test points?
    # Or the max? 
    # If we take the mean, we minimize the average "soft max" distance.
    # If we take the max, we minimize the worst-case soft max distance.
    # Taking the max of LSEs is non-differentiable at the max.
    # Taking the mean is smooth. Since the test points are uniform, the mean
    # of the soft-max distances correlates strongly with the true covering radius.
    # However, to focus on the worst regions, we can use a higher T.
    
    obj_val = np.mean(lse_vals)
    
    # Gradient calculation:
    # d(obj)/dx = (1/M) * sum_m d(lse_m)/dx
    # lse_m = T * y_max_m + log(sum_i exp(T * (y_{m,i} - y_max_m)))
    # Note: y_max_m depends on x if the argmax changes, but in the LSE formulation,
    # we treat y_max as a constant shift for the log-sum-exp stability trick?
    # Actually, the derivative of LSE(z) = log(sum exp(z_i)) is softmax(z).
    # Here z_{m,i} = T * y_{m,i}.
    # d(LSE(z))/dz_i = exp(z_i) / sum(exp(z_j)) = softmax(z_i).
    # So d(lse_m)/d(y_{m,i}) = T * softmax(T * (y_{m,i} - y_max_m))_i
    # And d(y_{m,i})/d(x) = - d(dots_{m,i})/d(x) = - TEST_POINTS[m] / ||pts[i]||^2 ?
    # Since pts are normalized, d(pts[i])/dx is complex.
    # Let's use the chain rule with the normalized points.
    # Let P be the matrix of points. x is the flattened P.
    # d(dots[m,i])/d(P[i]) = TEST_POINTS[m]
    # But P[i] is constrained to unit sphere.
    # We can project the gradient to the tangent space.
    
    # Let G_{m,i} = T * softmax(T * (y_{m,i} - y_max_m))_i
    # This is the gradient of lse_m with respect to y_{m,i}.
    
    # Compute softmax weights
    # softmax_vals[m, i] = exp(T*(y_{m,i} - y_max_m)) / sum_j exp(T*(y_{m,j} - y_max_m))
    exp_vals = np.exp(exp_arg)
    sum_exp_vals = np.sum(exp_vals, axis=1, keepdims=True)
    softmax_weights = exp_vals / sum_exp_vals  # (M, N)
    
    # grad_y[m, i] = T * softmax_weights[m, i]
    grad_y = T * softmax_weights  # (M, N)
    
    # Now, grad_obj = (1/M) * sum_m sum_i grad_y[m, i] * d(y_{m,i})/dx
    # y_{m,i} = 1 - dots[m,i] = 1 - sum_k TEST_POINTS[m,k] * P[i,k]
    # d(y_{m,i})/d(P[i,k]) = - TEST_POINTS[m,k]
    # So, d(obj)/d(P[i]) = (1/M) * sum_m grad_y[m, i] * (- TEST_POINTS[m])
    #                    = -(1/M) * (grad_y[:, i] @ TEST_POINTS)
    
    # This gives the gradient with respect to the unnormalized coordinates P.
    # We need to project this to the tangent space of the sphere for each point i.
    # grad_tangent[i] = grad_P[i] - (grad_P[i] . P[i]) * P[i]
    
    # Compute grad_P: shape (N, 3)
    # grad_P[i] = -(1/M) * sum_m grad_y[m, i] * TEST_POINTS[m]
    # This is equivalent to: -(1/M) * (TEST_POINTS.T @ grad_y[:, i])
    # Let's vectorize: 
    # grad_P = -(1/M) * (TEST_POINTS.T @ grad_y) ? No.
    # TEST_POINTS is (M, 3). grad_y is (M, N).
    # TEST_POINTS.T @ grad_y is (3, M) @ (M, N) = (3, N).
    # We want (N, 3). So transpose: (grad_y.T @ TEST_POINTS) is (N, M) @ (M, 3) = (N, 3).
    
    grad_P = -(1.0 / len(TEST_POINTS)) * (grad_y.T @ TEST_POINTS)  # (N, 3)
    
    # Project to tangent space
    # P is (N, 3)
    dots_grad = np.sum(grad_P * pts, axis=1, keepdims=True)  # (N, 1)
    grad_tangent = grad_P - dots_grad * pts  # (N, 3)
    
    # Flatten gradient
    grad_flat = grad_tangent.flatten()
    
    return obj_val, grad_flat

def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    best_pts = None
    best_cr = np.inf
    
    # We will run multiple restarts with different initializations and L-BFGS optimizations.
    max_restarts = 15
    
    for i in range(max_restarts):
        if time.time() > DEADLINE - 10:
            break
        
        # Initialization
        if i % 3 == 0:
            # Fibonacci spiral
            j = np.arange(N) + 0.5
            z = 1.0 - 2.0 * j / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * j
            r = np.sqrt(np.maximum(0, 1.0 - z * z))
            pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            pts += rng.randn(N, 3) * 0.1
        elif i % 3 == 1:
            # Random
            pts = rng.randn(N, 3)
        else:
            # Perturbed Fibonacci
            j = np.arange(N) + 0.5
            z = 1.0 - 2.0 * j / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * j + rng.randn(N) * 0.5
            r = np.sqrt(np.maximum(0, 1.0 - z * z))
            pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            pts += rng.randn(N, 3) * 0.2
            
        pts = _normalize(pts)
        x0 = pts.flatten()
        
        # Optimization
        # Use L-BFGS-B. It handles the smooth objective well.
        # We monitor time.
        
        try:
            # Callback to check time and potentially stop
            # L-BFGS-B doesn't have a built-in time limit, so we rely on the maxiter
            # and the fact that each step is fast.
            # However, if it runs too long, we might timeout.
            # We can wrap the objective to check time? 
            # Better: just run a fixed number of iterations that we know is fast.
            # One LSE eval is O(M*N). M=2000, N=130. 260k ops. Very fast.
            # 100 iterations should be < 1s.
            
            res = minimize(
                _soft_covering_objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,  # We return gradient
                options={'maxiter': 100, 'gtol': 1e-6, 'maxfun': 200}
            )
            
            if res.success or res.status == 0:
                opt_pts = _normalize(res.x.reshape(N, 3))
                cr = _covering_radius(opt_pts)
                
                if cr < best_cr:
                    best_cr = cr
                    best_pts = opt_pts.copy()
                    
        except Exception as e:
            # If optimization fails, skip this restart
            pass
            
        # If time is running out, break
        if time.time() > DEADLINE - 5:
            break

    if best_pts is None:
        # Fallback: Return a random configuration or the last one
        # We should have at least one from the loop unless it failed immediately
        j = np.arange(N) + 0.5
        z = 1.0 - 2.0 * j / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * j
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    # Final polish: A few Lloyd steps on the best gradient result can sometimes help
    # because LSE might not perfectly capture the sharp corners of the Voronoi cells.
    if time.time() < DEADLINE - 2:
        current_best = best_pts.copy()
        for _ in range(10):
            if time.time() > DEADLINE - 1:
                break
            # Simple Lloyd step
            # Assign grid points to nearest
            M_grid = 512
            j = np.arange(M_grid**2) + 0.5
            z = 1.0 - 2.0 * j / (M_grid**2)
            phi = np.pi * (1.0 + np.sqrt(5.0)) * j
            r = np.sqrt(np.maximum(0, 1.0 - z * z))
            grid_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            
            dots = grid_pts @ current_best.T
            nearest_indices = np.argmax(dots, axis=1)
            
            sums = np.zeros((N, 3))
            counts = np.zeros(N)
            np.add.at(sums, nearest_indices, grid_pts)
            np.add.at(counts, nearest_indices, 1)
            
            new_pts = current_best.copy()
            mask = counts > 0
            new_pts[mask] = _normalize(sums[mask])
            
            # Check if it improved
            cr_new = _covering_radius(new_pts)
            if cr_new < best_cr:
                best_cr = cr_new
                best_pts = new_pts
            else:
                # If it didn't improve, stop Lloyd to avoid oscillation
                break
                
    return _normalize(best_pts)

if __name__ == "__main__":
    pts = construct()
    print("Shape:", pts.shape)
    print("Covering radius (deg):", np.degrees(_covering_radius(pts)))
