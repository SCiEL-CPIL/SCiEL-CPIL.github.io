"""
Optimized circle packing for N=50 in a unit square.
Uses repulsion-based optimization with soft-min surrogate for the minimum distance.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    """
    Returns an array of shape (N, 2) with the centers of N equal circles
    packed in the square [-0.5, 0.5]^2 to maximize the radius.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # 1. Initialization: Hexagonal lattice
    # Approximate number of rows for hex packing
    # N ~ k^2 * 2 / sqrt(3) ? 
    # For N=50, a 7x8 grid with hex offsets is good.
    # Let's just generate a grid and then hex-ify it or use a standard hex init.
    
    def init_hex():
        # Generate points in a hexagonal pattern
        # Side length of square is 1.
        # We want to fit N points.
        # Let's try a 7x8 grid with alternating shifts.
        rows = 7
        cols = 8
        # Spacing
        # Horizontal spacing dx, vertical spacing dy = dx * sqrt(3)/2
        # We want to fit in 1x1.
        # Let's just create a grid of size 10x10 and pick the best N? 
        # No, better to scale a hex grid.
        
        # Simple hex init:
        # x in [-0.5, 0.5], y in [-0.5, 0.5]
        # Let's use a 7x8 grid.
        # Width covered by 8 cols: 7 * dx
        # Height covered by 7 rows: 6 * dy + 2*margin?
        # Let's just use a standard grid and perturb, or a specific hex layout.
        
        # A robust init:
        # Create a grid of points, then apply hexagonal distortion.
        # Or simply:
        m = int(np.ceil(np.sqrt(N)))
        # 8x8 grid = 64 points, we take 50.
        # Let's use 7 rows, 8 cols roughly.
        
        # Let's try to generate a hexagonal lattice that fits in the square.
        # dx = 1 / (cols - 1) ? No, we need to scale.
        
        # Let's use a simple heuristic:
        # Start with a square grid, then optimize. The optimizer is strong enough
        # to rearrange the grid into a hex-like structure if initialized reasonably.
        # However, a hex init is better.
        
        # Hex init for N=50:
        # 5 rows of 10? No.
        # 7 rows: 8,7,8,7,8,7,5 = 50? 
        # 8+7+8+7+8+7+5 = 50.
        
        ys = np.linspace(-0.45, 0.45, 7)
        xs_list = []
        counts = [8, 7, 8, 7, 8, 7, 5]
        
        for i, y in enumerate(ys):
            c = counts[i]
            # Center the row
            # Spacing dx approx 1/8 = 0.125
            # Let's scale dx based on row count to keep density uniform
            # Actually, just use a fixed dx and center them.
            dx = 0.12
            # Shift odd rows by dx/2
            offset = 0 if i % 2 == 0 else dx / 2.0
            xs = np.linspace(-0.45, 0.45, c)
            # If c is even, centering is symmetric. If odd, also symmetric.
            # Let's just use np.linspace and it will be centered.
            xs_list.append(np.column_stack((xs, np.full(c, y))))
            
        pts = np.vstack(xs_list)
        return pts

    x0 = init_hex()
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    x0 = x0 + np.random.randn(N, 2) * 0.01
    
    # Ensure initial points are strictly inside
    x0[:, 0] = np.clip(x0[:, 0], -0.49, 0.49)
    x0[:, 1] = np.clip(x0[:, 1], -0.49, 0.49)

    # Precompute indices for pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    pairs = np.stack((i_idx, j_idx), axis=1)

    def objective(x):
        # x is shape (N, 2)
        # 1. Compute pairwise distances
        # Vectorized calculation
        # d_ij = ||x_i - x_j||
        # We want to minimize -min(d_ij/2, b_i)
        # Using soft-min:
        # S_beta(z) = -1/beta * log(sum(exp(-beta * z)))
        
        # Beta controls the sharpness. Start with lower beta, increase later?
        # Or just use a fixed reasonable beta.
        beta = 20.0
        
        # Pairwise distances
        # x[:, np.newaxis, :] - x[np.newaxis, :, :] is (N, N, 2)
        # This is memory heavy for N=50? 50*50*2 = 5000 floats. Fine.
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        # Take upper triangle
        dists = np.sqrt(dists_sq[i_idx, j_idx])
        
        # Boundary distances
        # min(0.5 - |x|, 0.5 - |y|)
        b = np.minimum(0.5 - np.abs(x[:, 0]), 0.5 - np.abs(x[:, 1]))
        
        # Combine distances (scaled by 0.5 for radius) and boundaries
        # The radius is min(d/2, b)
        # Let's work with radii directly.
        r_pairs = dists / 2.0
        r_bound = b
        
        all_r = np.concatenate((r_pairs, r_bound))
        
        # Soft min
        # max(all_r) is used for numerical stability
        max_r = np.max(all_r)
        log_sum_exp = np.log(np.sum(np.exp(-beta * (all_r - max_r))))
        soft_min_val = max_r - (1.0 / beta) * log_sum_exp
        
        # We want to MAXIMIZE soft_min_val.
        # So we minimize -soft_min_val.
        return -soft_min_val

    def gradient(x):
        # Derivative of soft-min
        # S = max_r - 1/beta * log(sum(exp(-beta(r_k - max_r))))
        # dS/dx = sum_k ( -exp(-beta(r_k - max_r)) / sum_j exp(-beta(r_j - max_r)) ) * (-beta * dr_k/dx)
        #       = sum_k p_k * dr_k/dx
        # where p_k = exp(-beta(r_k - max_r)) / sum_j ... (softmax probabilities)
        
        beta = 20.0
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(np.maximum(dists_sq, 1e-12)) # avoid div by 0
        r_pairs = dists[i_idx, j_idx] / 2.0
        
        b = np.minimum(0.5 - np.abs(x[:, 0]), 0.5 - np.abs(x[:, 1]))
        r_bound = b
        
        all_r = np.concatenate((r_pairs, r_bound))
        
        max_r = np.max(all_r)
        exp_terms = np.exp(-beta * (all_r - max_r))
        sum_exp = np.sum(exp_terms)
        probs = exp_terms / sum_exp
        
        # Gradient of r_pairs w.r.t x
        # r_ij = 0.5 * ||x_i - x_j||
        # dr_ij/dx_i = 0.5 * (x_i - x_j) / ||x_i - x_j||
        # dr_ij/dx_j = 0.5 * (x_j - x_i) / ||x_i - x_j||
        
        grad = np.zeros_like(x)
        
        # Pair contributions
        # p_pairs = probs[:N_pairs]
        # We need to map back to i, j
        p_pairs = probs[:len(r_pairs)]
        
        # diff[i, j] = x[i] - x[j]
        # dists[i, j]
        # We only have upper triangle indices.
        # Let's compute full diff and dists again for clarity, or use the stored ones.
        # Using stored:
        # diff is (N, N, 2).
        # We need diff[i_idx, j_idx]
        d_vecs = diff[i_idx, j_idx] # (M, 2)
        d_norms = dists[i_idx, j_idx] # (M,)
        
        # Avoid division by zero
        d_norms_safe = np.where(d_norms < 1e-8, 1e-8, d_norms)
        
        # grad_x_i += sum_j p_ij * 0.5 * (x_i - x_j) / d_ij
        # grad_x_j += sum_i p_ij * 0.5 * (x_j - x_i) / d_ij
        
        # Vectorized accumulation
        # For each pair (i, j), contribution to i is p * 0.5 * d_vec / d_norm
        # contribution to j is p * 0.5 * (-d_vec) / d_norm
        
        contribs = p_pairs[:, np.newaxis] * 0.5 * d_vecs / d_norms_safe[:, np.newaxis]
        
        # Scatter add
        # np.add.at(grad, i_idx, contribs)
        # np.add.at(grad, j_idx, -contribs)
        
        np.add.at(grad, i_idx, contribs)
        np.add.at(grad, j_idx, -contribs)
        
        # Boundary contributions
        # r_bound_i = min(0.5 - |x_i|, 0.5 - |y_i|)
        # This is non-smooth. We use a smooth approximation for gradient?
        # Or just use subgradient.
        # For soft-min, the gradient is a weighted sum of gradients of components.
        # dr_bound_i / dx_i:
        # If 0.5 - |x_i| < 0.5 - |y_i|, then r = 0.5 - |x_i|.
        # dr/dx = -sign(x_i)
        # Else r = 0.5 - |y_i|. dr/dy = -sign(y_i)
        
        p_bound = probs[len(r_pairs):]
        
        # Determine which bound is active
        bx = 0.5 - np.abs(x[:, 0])
        by = 0.5 - np.abs(x[:, 1])
        active_x = bx < by
        
        # Gradient w.r.t x
        # d(0.5 - |x|)/dx = -sign(x)
        grad_bx = -np.sign(x[:, 0])
        grad_by = -np.sign(x[:, 1])
        
        # Where active_x is True, use grad_bx for x-component, 0 for y?
        # Wait, r = min(bx, by).
        # If bx < by, r = bx. dr/dx = -sign(x), dr/dy = 0.
        # If by < bx, r = by. dr/dx = 0, dr/dy = -sign(y).
        # If equal, subgradient is average? For smoothness, we can blend or just pick one.
        # Since we are using soft-min, the "probability" p_bound is distributed.
        # But the gradient of the component itself is needed.
        
        # Let's compute gradients for each component separately and weight by p_bound.
        # Component i:
        # grad_x_i = p_bound[i] * ( -sign(x_i) if active_x[i] else 0 )
        # grad_y_i = p_bound[i] * ( 0 if active_x[i] else -sign(y_i) )
        
        # Handle ties (bx == by) by splitting gradient? Or just pick x.
        # It's a measure zero set.
        
        gx = np.where(active_x, -np.sign(x[:, 0]), 0.0)
        gy = np.where(active_x, 0.0, -np.sign(x[:, 1]))
        
        grad[:, 0] += p_bound * gx
        grad[:, 1] += p_bound * gy
        
        return -grad # We are minimizing -S, so return -dS/dx

    # Optimization loop
    best_x = x0
    best_val = objective(x0)
    
    # Run L-BFGS-B
    # Options
    options = {
        'maxiter': 500,
        'maxfun': 2000,
        'ftol': 1e-10,
        'gtol': 1e-8,
        'disp': False
    }
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', jac=gradient, options=options)
        if res.success or res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > deadline - 20:
        return best_x

    # Second pass with higher beta for sharper min
    # This helps refine the "bottleneck" constraints
    
    # We need to re-define objective with higher beta.
    # Since objective captures beta from closure, we can't easily change it without redefining.
    # Let's create a factory.
    
    def make_objective(beta):
        def obj(x):
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            dists = np.sqrt(np.maximum(dists_sq, 1e-12))
            r_pairs = dists[i_idx, j_idx] / 2.0
            
            b = np.minimum(0.5 - np.abs(x[:, 0]), 0.5 - np.abs(x[:, 1]))
            r_bound = b
            
            all_r = np.concatenate((r_pairs, r_bound))
            max_r = np.max(all_r)
            log_sum_exp = np.log(np.sum(np.exp(-beta * (all_r - max_r))))
            soft_min_val = max_r - (1.0 / beta) * log_sum_exp
            return -soft_min_val
        return obj

    def make_gradient(beta):
        def grad(x):
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            dists = np.sqrt(np.maximum(dists_sq, 1e-12))
            r_pairs = dists[i_idx, j_idx] / 2.0
            
            b = np.minimum(0.5 - np.abs(x[:, 0]), 0.5 - np.abs(x[:, 1]))
            r_bound = b
            
            all_r = np.concatenate((r_pairs, r_bound))
            max_r = np.max(all_r)
            exp_terms = np.exp(-beta * (all_r - max_r))
            sum_exp = np.sum(exp_terms)
            probs = exp_terms / sum_exp
            
            grad_x = np.zeros_like(x)
            
            d_vecs = diff[i_idx, j_idx]
            d_norms = dists[i_idx, j_idx]
            d_norms_safe = np.where(d_norms < 1e-8, 1e-8, d_norms)
            
            p_pairs = probs[:len(r_pairs)]
            contribs = p_pairs[:, np.newaxis] * 0.5 * d_vecs / d_norms_safe[:, np.newaxis]
            
            np.add.at(grad_x, i_idx, contribs)
            np.add.at(grad_x, j_idx, -contribs)
            
            p_bound = probs[len(r_pairs):]
            bx = 0.5 - np.abs(x[:, 0])
            by = 0.5 - np.abs(x[:, 1])
            active_x = bx < by
            
            gx = np.where(active_x, -np.sign(x[:, 0]), 0.0)
            gy = np.where(active_x, 0.0, -np.sign(x[:, 1]))
            
            grad_x[:, 0] += p_bound * gx
            grad_x[:, 1] += p_bound * gy
            
            return -grad_x
        return grad

    # Run with beta=50
    if time.time() - start_time < deadline - 10:
        obj2 = make_objective(50.0)
        grad2 = make_gradient(50.0)
        try:
            res2 = minimize(obj2, best_x, method='L-BFGS-B', jac=grad2, options=options)
            if res2.success or res2.fun < best_val:
                best_x = res2.x
                best_val = res2.fun
        except Exception:
            pass

    # Final polish: Local search on the hard minimum
    # Calculate actual radius
    def calc_radius(x):
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        b = np.minimum(0.5 - np.abs(x[:, 0]), 0.5 - np.abs(x[:, 1]))
        min_bound = np.min(b)
        
        return min(min_pair, min_bound)

    r_best = calc_radius(best_x)
    
    # Try a few random restarts if time permits?
    # Or just return best_x.
    
    # Ensure validity
    best_x[:, 0] = np.clip(best_x[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
    best_x[:, 1] = np.clip(best_x[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_x

# Keep the original structure required by the prompt?
# The prompt says: "Keep the function names and signatures of the seed program."
# Seed program:
# def construct():
#    ...
#    return grid[:N]

# My code above defines construct().
# I should wrap it properly.

# Wait, the prompt says "Rewrite the program... Output the COMPLETE program".
# I will output the single file content.

# One detail: The seed program has N=50 globally.
# I will keep that.

# Final check on the "EVOLVE-BLOCK" requirement.
# The prompt says "SCiEL evolves the EVOLVE-BLOCK".
# But the directive says "Rewrite the program from scratch... Output the COMPLETE program".
# And "Keep the program's entry-point function name(s)...".
# It does NOT explicitly say I must keep the EVOLVE-BLOCK markers, but it's good practice if the evaluator expects them.
# However, "Rewrite from scratch" usually implies I can structure it however I want, as long as `construct()` works.
# I will include the markers to be safe, or just the code.
# Given "Output the COMPLETE program in a single
