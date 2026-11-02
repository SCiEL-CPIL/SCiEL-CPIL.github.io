"""Pack 50 equal spheres in a unit cube to maximize the common radius."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

N = 50


def _score(c):
    """Compute the radius for a set of centres c (N,3)."""
    c = np.asarray(c)
    # Wall distances
    wall = 0.5 - np.abs(c).max(axis=1)
    min_wall = wall.min()
    # Pairwise distances
    # Use broadcasting for pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = dist.min()
    return min(min_wall, min_dist / 2.0)


def _negative_score(c):
    return -_score(c)


def _project_into_cube(c, r):
    """Ensure centres are within [-(0.5-r), 0.5-r] in each coordinate."""
    lo = -(0.5 - r)
    hi = 0.5 - r
    return np.clip(c, lo, hi)


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit

    # Initial guess: use a body-centered cubic (BCC) inspired grid or random
    # Let's start with a simple cubic grid and then optimize
    # Better: use a random initialization with some structure
    
    # Use a grid-based initialization that's denser in the center
    # For 50 spheres, a 4x4x4 grid would give 64 positions, we take 50
    # But let's use a better initialization: FCC-like or random
    
    # Start with random points, then run optimization
    rng = np.random.default_rng(42)
    
    best_c = None
    best_score = -np.inf
    
    # Multiple restarts
    for restart in range(5):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Simple cubic grid
            m = 4
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            c0 = grid[:N].copy()
        elif restart == 1:
            # BCC-like: FCC lattice points
            # Generate FCC points in a cube
            pts = []
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        pts.append((i / 3.0 - 0.5, j / 3.0 - 0.5, k / 3.0 - 0.5))
                        # Face centers
                        pts.append((i / 3.0 - 0.5, (j + 0.5) / 3.0 - 0.5, (k + 0.5) / 3.0 - 0.5))
                        pts.append(((i + 0.5) / 3.0 - 0.5, j / 3.0 - 0.5, (k + 0.5) / 3.0 - 0.5))
                        pts.append(((i + 0.5) / 3.0 - 0.5, (j + 0.5) / 3.0 - 0.5, k / 3.0 - 0.5))
            pts = np.array(pts)
            # Shuffle and take first N
            idx = rng.permutation(len(pts))
            c0 = pts[idx[:N]].copy()
        else:
            # Random initialization
            c0 = rng.uniform(-0.45, 0.45, size=(N, 3))
        
        # Run gradient-free optimization using Nelder-Mead on the negative score
        # But N=50*3=150 dimensions is too high for Nelder-Mead
        
        # Instead, use a coordinate-wise or gradient-based approach
        # Let's use L-BFGS-B with a smooth approximation
        
        # Smooth approximation: use soft-min
        def soft_min(vals, beta=50):
            """Smooth approximation of min using log-sum-exp."""
            vals = np.asarray(vals)
            m = vals.min()
            return m - np.log(np.sum(np.exp(beta * (vals - m)))) / beta
        
        def negative_smooth_score(c):
            c = np.asarray(c).reshape(-1, 3)
            # Wall distances
            wall = 0.5 - np.abs(c).max(axis=1)
            # Pairwise distances
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dist = np.sqrt((diff ** 2).sum(axis=2))
            np.fill_diagonal(dist, np.inf)
            pair_dist = dist / 2.0
            all_vals = np.concatenate([wall, pair_dist])
            return -soft_min(all_vals, beta=100)
        
        def negative_smooth_score_grad(c):
            c = np.asarray(c).reshape(-1, 3)
            wall = 0.5 - np.abs(c).max(axis=1)
            
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dist = np.sqrt((diff ** 2).sum(axis=2))
            np.fill_diagonal(dist, np.inf)
            pair_dist = dist / 2.0
            
            all_vals = np.concatenate([wall, pair_dist])
            beta = 100
            m = all_vals.min()
            exps = np.exp(beta * (all_vals - m))
            sum_exps = np.sum(exps)
            
            # Gradient of soft_min: -sum(exp(beta*(v-m)) / sum(exp)) * beta * dv/dx
            weights = exps / sum_exps
            grad = np.zeros_like(c)
            
            # Wall gradient
            for i in range(N):
                # wall_i = 0.5 - max(|c_ix|, |c_iy|, |c_iz|)
                abs_c = np.abs(c[i])
                max_idx = np.argmax(abs_c)
                # d(wall_i)/d(c_ik) = -sign(c_ik) if k==max_idx else 0
                # But soft_min gives weighted average
                # Actually, for the soft_min, the gradient is:
                # d(soft_min)/d(v_j) = -beta * exp(beta*(v_j - m)) / sum(exp)
                # So d(soft_min)/d(c_ik) = sum_j weight_j * d(v_j)/d(c_ik)
                
                # Wall contribution
                d_wall_dc = np.zeros(3)
                d_wall_dc[max_idx] = -np.sign(c[i, max_idx]) if c[i, max_idx] != 0 else 1
                # But this is the hard min gradient. For soft_min, we need to be more careful.
                # Let's just use the hard min gradient as an approximation for the gradient step.
                
                # For now, use a simpler approach: just compute the gradient numerically
                pass
            
            # This is getting complicated. Let's use a simpler optimization approach.
            # Use Powell's method or a custom hill-climbing approach.
            return None  # Not using gradient
        
        # Let's use a different approach: iterative local optimization
        # Start from c0, and iteratively move points to improve the minimum distance
        
        c = c0.copy()
        
        # Initial radius
        r = _score(c)
        
        # Iterative improvement: for each point, try to move it to increase the minimum distance
        # Use a simple approach: for each point, compute the direction that maximizes the minimum distance to other points and walls
        
        for iteration in range(2000):
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                
                # Current point
                p = c[i].copy()
                
                # Compute the minimum distance to other points and walls
                # For point i, the constraints are:
                # 1. Distance to wall: 0.5 - max(|p_k|) for k=0,1,2
                # 2. Distance to other points j: |p - c_j| / 2
                
                # Try small perturbations in random directions
                best_p = p
                best_local_r = r
                
                # Compute current local radius for point i
                wall_dist = 0.5 - np.abs(p).max()
                other_dists = np.sqrt(((c - p) ** 2).sum(axis=1))
                other_dists[i] = np.inf
                min_other = other_dists.min() / 2.0
                local_r = min(wall_dist, min_other)
                
                # Try to improve by moving in the direction that increases the limiting constraint
                # Determine which constraint is limiting
                if wall_dist < min_other:
                    # Wall is limiting: move toward center
                    # The limiting wall coordinate
                    max_idx = np.argmax(np.abs(p))
                    # Move in the direction that reduces |p[max_idx]|
                    direction = -np.sign(p[max_idx]) * np.eye(3)[max_idx]
                    direction = direction / (np.linalg.norm(direction) + 1e-10)
                else:
                    # A neighbor is limiting
                    closest_j = np.argmin(other_dists)
                    if closest_j == i:
                        continue
                    # Move away from closest_j
                    direction = p - c[closest_j]
                    direction = direction / (np.linalg.norm(direction) + 1e-10)
                
                # Try several step sizes
                for step_size in [0.001, 0.005, 0.01, 0.02, 0.05]:
                    new_p = p + direction * step_size
                    # Check if new_p is better
                    new_wall_dist = 0.5 - np.abs(new_p).max()
                    new_other_dists = np.sqrt(((c - new_p) ** 2).sum(axis=1))
                    new_other_dists[i] = np.inf
                    new_min_other = new_other_dists.min() / 2.0
                    new_local_r = min(new_wall_dist, new_min_other)
                    
                    if new_local_r > best_local_r + 1e-10:
                        best_local_r = new_local_r
                        best_p = new_p
                
                if np.linalg.norm(best_p - p) > 1e-10:
                    c[i] = best_p
                    # Recompute global radius
                    new_r = _score(c)
                    if new_r > r + 1e-12:
                        r = new_r
                        improved = True
                    else:
                        # Revert if it didn't help globally
                        c[i] = p
            
            if not improved:
                break
        
        # Update best
        if r > best_score:
            best_score = r
            best_c = c.copy()
        
        # If we're close to the known best, try to polish
        if best_score > 0.135:
            break
    
    if best_c is None:
        # Fallback: simple grid
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]
    
    return best_c
