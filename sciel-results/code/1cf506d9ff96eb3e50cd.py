"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _hexagonal_lattice(n):
    """Generate a dense hexagonal lattice arrangement of n points in [-0.5, 0.5]^2."""
    # We want to generate a hexagonal grid that covers the square and pick n points.
    # A standard hexagonal packing has rows offset.
    # Let's generate a grid with a spacing that allows for at least n points.
    
    # Approximate number of points in a hexagonal packing of area 1 (unit square)
    # Area per point ~ sqrt(3)/2 * d^2. 
    # Total area 1. N ~ 1 / (sqrt(3)/2 * d^2) => d ~ sqrt(2 / (sqrt(3) * N))
    d_est = np.sqrt(2.0 / (np.sqrt(3) * n))
    
    # Generate points on a hexagonal grid with this spacing
    points = []
    # x range: -0.5 to 0.5
    # y range: -0.5 to 0.5
    
    # Hex grid:
    # Row j: y = j * d * sqrt(3) / 2
    # x = i * d + (j % 2) * d / 2
    
    # Number of rows needed
    row_spacing = d_est * np.sqrt(3) / 2
    max_j = int(1.0 / row_spacing) + 2
    max_i = int(1.0 / d_est) + 2
    
    for j in range(-max_j, max_j + 1):
        y = j * row_spacing
        if -0.5 <= y <= 0.5:
            for i in range(-max_i, max_i + 1):
                x = i * d_est + (j % 2) * d_est / 2
                if -0.5 <= x <= 0.5:
                    points.append([x, y])
    
    points = np.array(points)
    
    # If we have more than n points, we need to select n.
    # We want them to be well-distributed.
    # Strategy: Keep points that are farthest from the boundary? Or just take a random subset?
    # Better: If we have more, we can try to "thin" them out.
    # For simplicity and robustness, if len > n, we can just take the first n? 
    # That might bias towards one corner.
    # Let's sort by distance from center and take the ones closest to center? No, that clusters.
    # Let's try to select a subset that maximizes minimum distance using a greedy approach? Too slow.
    # Alternative: Generate a grid that is slightly sparser so we get exactly ~n points?
    # Or, if we have more, we can use K-Means to find n centers?
    
    # A simpler robust init: Use a square grid if hex fails or is too complex to subset.
    # But hex is better.
    # Let's just take a random sample of n points if we have more.
    # Or, even better, adjust the spacing d_est so we get close to n.
    
    # Let's refine d_est to get exactly n or close to it.
    # This is tricky. Let's just use the generated points and if len > n, 
    # we will use a simple "repulsion" step later to fix overlaps/clustering.
    # But for init, a random subset of a good lattice is okay.
    
    if len(points) == 0:
        # Fallback
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n]
        
    if len(points) > n:
        # Randomly select n points to avoid bias
        indices = np.random.choice(len(points), n, replace=False)
        points = points[indices]
    elif len(points) < n:
        # If we don't have enough, fill in with a square grid
        # This shouldn't happen with the estimated d, but just in case
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        # Combine and take first n? Or just return grid?
        # Let's just return the grid if hex fails
        return grid[:n]
        
    return points


def construct():
    # 1. Initialization
    points = _hexagonal_lattice(N).copy()
    
    # 2. Optimization using Gradient Descent on Soft-Min
    # Objective: Maximize min(distances)
    # Approximation: Maximize (1/beta) * log( sum exp(-beta * dist) )
    # Gradient of (1/beta) log sum exp(-beta d_ij) w.r.t points is:
    # - sum exp(-beta d_ij) * (c_i - c_j) / d_ij / sum exp(-beta d_ij)
    # This points in the direction that increases the smallest distances.
    
    # We will iterate with increasing beta (annealing)
    betas = [10, 50, 100, 200, 500, 1000]
    steps = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    for beta, step_size in zip(betas, steps):
        for _ in range(200): # Iterations per beta
            # Compute distances to all other points
            # To be efficient, we can just compute the gradient contribution
            # grad_i = sum_{j != i} weight_ij * (c_i - c_j) / d_ij
            # where weight_ij = exp(-beta * d_ij) / sum_k exp(-beta * d_ik) ... wait, the normalization is global for the sum.
            
            # Let's compute the vector field.
            # D[i, j] is distance between i and j.
            # We need sum_{j} exp(-beta D[i,j]) * (c_i - c_j) / D[i,j]
            
            # Efficient calculation:
            # diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            # dists = np.linalg.norm(diff, axis=-1)
            # np.fill_diagonal(dists, np.inf)
            
            # This is O(N^2) memory and time. N=200, so 40k pairs. This is fine.
            
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dists, np.inf)
            
            # Weights for pairwise distances
            # To avoid underflow/overflow, we can subtract the min distance from dists before exp?
            # Actually, we just want the relative weights.
            # w_ij = exp(-beta * dists[i,j])
            # If beta is large, only small distances matter.
            
            # Boundary constraints:
            # dist_to_boundary_i = min(0.5 - |x_i|, 0.5 - |y_i|)
            # We want to maximize this too.
            # Gradient of soft-min boundary:
            # Force towards center for points close to boundary.
            # Force_i_boundary = exp(-beta * b_i) * (center - c_i) / |center - c_i| ?
            # Actually, the gradient of -b_i w.r.t c_i is just the direction towards the boundary.
            # So to increase b_i, we move towards center.
            # Vector: (0 - x_i, 0 - y_i) normalized? No, just the component that increases distance to boundary.
            # If x_i > 0, move left (negative x). If x_i < 0, move right.
            # So force_x = -sign(x_i) * exp(-beta * b_i)
            # force_y = -sign(y_i) * exp(-beta * b_i)
            
            b_dists = np.minimum(0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1]))
            
            # Calculate pairwise forces
            # To handle the sum in the denominator of the soft-min gradient properly:
            # The gradient of (1/beta) log(sum_j exp(-beta d_ij)) is:
            # - (1/sum_j exp(-beta d_ij)) * sum_j exp(-beta d_ij) * grad d_ij
            # grad d_ij = (c_i - c_j) / d_ij
            # So force_i = sum_j exp(-beta d_ij) * (c_i - c_j) / d_ij / sum_j exp(-beta d_ij)
            
            # We can compute this without storing the full matrix of exps if we are careful, 
            # but N=200 is small enough.
            
            # Mask diagonal
            mask = np.eye(N, dtype=bool)
            
            # Compute exp terms
            # To prevent underflow for large beta, we can normalize dists by min_dist?
            # Or just clip dists.
            # Let's just compute exp(-beta * dists).
            # For beta=1000, exp(-1000*0.03) is tiny.
            # This effectively picks the nearest neighbor.
            
            exp_dists = np.exp(-beta * dists)
            np.fill_diagonal(exp_dists, 0)
            
            # Sum of weights for each i
            weight_sums = np.sum(exp_dists, axis=1)
            
            # Avoid division by zero
            weight_sums[weight_sums == 0] = 1.0
            
            # Force from neighbors: sum_j w_ij * (c_i - c_j) / d_ij
            # diff is c_i - c_j.
            # We need to divide diff by dists.
            # Handle dists=0 (shouldn't happen with inf diagonal).
            safe_dists = np.where(dists > 1e-10, dists, 1.0)
            
            # Force term: (c_i - c_j) / d_ij * w_ij
            # This is a (N, N, 2) tensor.
            # force_vec = sum_j (diff[i,j] / safe_dists[i,j]) * exp_dists[i,j]
            
            # diff is (N, N, 2)
            # safe_dists is (N, N)
            # exp_dists is (N, N)
            
            # Normalize diff by dists
            unit_vectors = diff / safe_dists[:, :, np.newaxis]
            
            # Weighted sum
            # Multiply by exp_dists (N, N, 1)
            weighted_vectors = unit_vectors * exp_dists[:, :, np.newaxis]
            
            # Sum over j (axis 1)
            neighbor_forces = np.sum(weighted_vectors, axis=1)
            
            # Normalize by weight_sums
            neighbor_forces /= weight_sums[:, np.newaxis]
            
            # Boundary forces
            # We want to move points away from boundaries.
            # Gradient of b_i w.r.t c_i:
            # b_i = min(0.5 - |x_i|, 0.5 - |y_i|)
            # If 0.5 - |x_i| < 0.5 - |y_i| (x is closer to boundary):
            #   db/dx = -sign(x), db/dy = 0
            # Else:
            #   db/dx = 0, db/dy = -sign(y)
            
            # Soft boundary force:
            # We use exp(-beta * b_i) as weight.
            # Direction: towards center.
            # If x > 0, move -x. If x < 0, move +x. So -sign(x).
            
            b_exp = np.exp(-beta * b_dists)
            
            # Determine which boundary is closer
            x_bound = 0.5 - np.abs(points[:, 0])
            y_bound = 0.5 - np.abs(points[:, 1])
            
            # Force x
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Points where x-bound is the minimum
            mask_x = x_bound <= y_bound
            fx[mask_x] = -np.sign(points[mask_x, 0]) * b_exp[mask_x]
            
            # Points where y-bound is the minimum
            mask_y = ~mask_x
            fy[mask_y] = -np.sign(points[mask_y, 1]) * b_exp[mask_y]
            
            # Combine forces
            # We want to maximize the soft-min.
            # So we move in the direction of the gradient of the soft-min.
            # The neighbor force calculated above is the gradient of the log-sum-exp of distances.
            # Wait, let's check signs.
            # f = (1/beta) log sum exp(-beta d).
            # df/dc_i = (1/beta) * (1/sum) * sum exp(-beta d) * (-beta) * (c_i - c_j)/d
            # df/dc_i = - sum exp(-beta d) * (c_i - c_j)/d / sum exp(-beta d)
            # This is exactly -neighbor_forces.
            # So to maximize f, we move in direction -df/dc_i = neighbor_forces.
            
            # So update: points += step_size * (neighbor_forces + boundary_forces)
            
            # Boundary force direction:
            # We want to increase b_i.
            # db/dc = -sign(coord) for the active boundary.
            # So gradient of b is -sign.
            # To increase b, move in direction of gradient? No, gradient points to steepest increase.
            # So move in direction of db/dc.
            # db/dc = -sign(x). So move in -sign(x) direction.
            # My fx calculation: fx = -sign(x) * weight. This is the direction to move.
            
            total_forces = neighbor_forces
            total_forces[:, 0] += fx
            total_forces[:, 1] += fy
            
            # Update points
            points += step_size * total_forces
            
            # Clip to boundaries
            points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
            points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
            
    # Final check and return
    return points

# EVOLVE-BLOCK-END
