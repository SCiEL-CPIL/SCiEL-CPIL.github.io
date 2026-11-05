import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

def compute_score(points):
    """
    Compute the raw score Delta / n^n.
    Delta is the product of ordered pair distances after scaling to diameter 2.
    """
    points = np.array(points)
    n = len(points)
    if n < 2:
        return 0.0
    
    # Compute pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Diameter is the max distance
    diameter = np.max(dists)
    if diameter <= 0:
        return 0.0
        
    # Mask diagonal to avoid log(0)
    mask = ~np.eye(n, dtype=bool)
    valid_dists = dists[mask]
    
    if np.any(valid_dists <= 0):
        return 0.0
        
    # Scale to diameter 2
    # d_scaled = d * (2 / diameter)
    # Delta = prod_{i!=j} d_scaled_{ij}
    # log(Delta) = sum_{i!=j} log(d_{ij}) + n*(n-1)*log(2) - n*(n-1)*log(diameter)
    
    n_pairs = n * (n - 1)
    log_delta = np.sum(np.log(valid_dists)) + n_pairs * np.log(2) - n_pairs * np.log(diameter)
    
    # Score = Delta / n^n
    log_score = log_delta - n * np.log(n)
    
    return np.exp(log_score)

def mds_2d(dists):
    """
    Embed a distance matrix into 2D using Classical MDS.
    Returns points of shape (n, 2).
    """
    n = len(dists)
    # Centering matrix
    J = np.eye(n) - np.ones((n, n)) / n
    
    # Double centering: B = -0.5 * J * D^2 * J
    D2 = dists ** 2
    B = -0.5 * (J @ D2 @ J)
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    
    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take top 2 eigenvalues
    # If eigenvalues are negative, they might be small due to numerical errors
    # We take the square root of the positive ones
    vals = np.maximum(eigenvalues[:2], 0)
    vecs = eigenvectors[:, :2]
    
    points = vecs * np.sqrt(vals)
    
    return points

def log_distance_objective(log_dists_flat):
    """
    Objective function for optimizing log distances.
    We want to maximize the sum of log distances (which is the log of the product).
    Since the diameter is fixed to 1 in this parameterization (d_max = 1),
    the score is proportional to exp(sum(log_dists) - n*log(n) + n*(n-1)*log(2)).
    Maximizing sum(log_dists) is equivalent to maximizing the score.
    
    Constraints:
    1. Triangle inequalities: d_ij <= d_ik + d_kj for all i,j,k.
       In log space: exp(l_ij) <= exp(l_ik) + exp(l_kj)
       This is non-linear. We use a penalty or barrier.
    2. d_max = 1 => max(l_ij) = 0.
    3. Metric embedding must be valid (GPD matrix).
    
    To handle the "max = 0" constraint, we can fix one distance to 0 (log(1))
    or use a penalty. However, fixing one pair to be the diameter is a discrete choice.
    A better approach: Use a soft constraint or simply penalize if any log_dist > 0.
    And ensure the maximum is close to 0.
    
    Actually, a cleaner approach for the optimization:
    Let's optimize the log distances l_ij.
    Objective: minimize -sum(l_ij)
    Subject to:
    - l_ij <= 0 for all i != j (diameter <= 1)
    - Triangle inequalities: exp(l_ij) <= exp(l_ik) + exp(l_kj)
    
    We can use a barrier function for the triangle inequalities.
    B(t) = -sum_{i,j,k} log(exp(l_ik) + exp(l_kj) - exp(l_ij) + epsilon)
    This is valid if the argument is positive.
    
    Let's implement a penalty method for triangle inequalities to be safer.
    Penalty = sum_{i,j,k} max(0, exp(l_ij) - exp(l_ik) - exp(l_kj))^2
    
    And we enforce max(l_ij) <= 0 with a penalty or constraint.
    """
    n = 8
    l = log_dists_flat.reshape(n, n)
    
    # Symmetrize to ensure l_ij = l_ji
    l = (l + l.T) / 2.0
    
    # Objective: minimize -sum(l_ij) for i != j
    mask = ~np.eye(n, dtype=bool)
    obj = -np.sum(l[mask])
    
    # Penalty for triangle inequalities
    # d_ij <= d_ik + d_kj
    # exp(l_ij) - exp(l_ik) - exp(l_kj) <= 0
    penalty = 0.0
    # To save time, we can subsample or just do all if n=8 is small
    # n=8, 8*8*8 = 512 terms. Very fast.
    for k in range(n):
        for i in range(n):
            if i == k: continue
            for j in range(n):
                if j == k or i == j: continue
                # Triangle inequality: d_ij <= d_ik + d_kj
                val = np.exp(l[i, j]) - np.exp(l[i, k]) - np.exp(l[k, j])
                if val > 0:
                    penalty += val * val
                    
    # Penalty for diameter constraint: max(l_ij) should be <= 0
    # If any l_ij > 0, penalize heavily
    max_l = np.max(l)
    if max_l > 0:
        penalty += 100.0 * max_l * max_l
        
    return obj + 10.0 * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 8
    best_score = 0.0
    best_points = None
    
    # Regular octagon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = compute_score(reg_pts)
    best_points = reg_pts.copy()
    
    rng = np.random.default_rng(42)
    
    # List of initial distance matrices (log space)
    inits = []
    
    # 1. Regular octagon distances
    diff = reg_pts[:, np.newaxis, :] - reg_pts[np.newaxis, :, :]
    dists_reg = np.sqrt(np.sum(diff**2, axis=2))
    # Normalize to diameter 1
    diam_reg = np.max(dists_reg)
    dists_reg_norm = dists_reg / diam_reg
    # Avoid log(0)
    dists_reg_norm[dists_reg_norm == 0] = 1e-10
    l_reg = np.log(dists_reg_norm)
    inits.append(l_reg.flatten())
    
    # 2. Random valid metrics? Hard.
    # Instead, generate random points, compute distances, normalize.
    for _ in range(20):
        # Random points in a disk
        r = np.sqrt(rng.uniform(0, 1, n))
        theta = rng.uniform(0, 2 * np.pi, n)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        diam = np.max(dists)
        if diam > 0:
            dists_norm = dists / diam
            dists_norm[dists_norm == 0] = 1e-10
            l_init = np.log(dists_norm)
            inits.append(l_init.flatten())
            
    # 3. Perturbed octagon
    for _ in range(10):
        angles_p = angles + rng.normal(0, 0.2, n)
        radii_p = 1.0 + rng.normal(0, 0.2, n)
        radii_p = np.abs(radii_p) + 0.1
        pts_p = np.column_stack([radii_p * np.cos(angles_p), radii_p * np.sin(angles_p)])
        diff = pts_p[:, np.newaxis, :] - pts_p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        diam = np.max(dists)
        if diam > 0:
            dists_norm = dists / diam
            dists_norm[dists_norm == 0] = 1e-10
            l_init = np.log(dists_norm)
            inits.append(l_init.flatten())

    # Optimize in log-distance space
    for l0 in inits:
        if time.time() > deadline - 30:
            break
            
        try:
            # Use BFGS or L-BFGS-B on the log distances
            res = minimize(log_distance_objective, l0, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-10})
            
            if res.fun < 1e10:
                l_opt = res.x.reshape(n, n)
                l_opt = (l_opt + l_opt.T) / 2.0
                # Ensure symmetry and diagonal 0
                np.fill_diagonal(l_opt, 0)
                
                # Convert to distances
                d_opt = np.exp(l_opt)
                # Fix diagonal
                np.fill_diagonal(d_opt, 0)
                
                # Check if it's a valid metric (triangle inequalities approx)
                # and embed it
                pts_embedded = mds_2d(d_opt)
                
                # Evaluate score of embedded points
                score_embedded = compute_score(pts_embedded)
                
                if score_embedded > best_score:
                    best_score = score_embedded
                    best_points = pts_embedded
                    
        except Exception:
            pass

    # Polish the best points in coordinate space
    if time.time() < deadline - 10 and best_points is not None:
        
        def neg_score_coords(pts_flat):
            pts = pts_flat.reshape(n, 2)
            s = compute_score(pts)
            if s <= 0 or not np.isfinite(s):
                return 1e10
            return -np.log(s)
            
        x0 = best_points.flatten()
        
        # Nelder-Mead polish
        try:
            res_polish = minimize(neg_score_coords, x0, method='Nelder-Mead',
                                  options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res_polish.fun < 1e10:
                pts_polished = res_polish.x.reshape(n, 2)
                score_polished = compute_score(pts_polished)
                if score_polished > best_score:
                    best_score = score_polished
                    best_points = pts_polished
        except Exception:
            pass

    # Scale best_points to diameter 2 for final return
    if best_points is not None:
        diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        diameter = np.max(dists)
        if diameter > 0:
            best_points = best_points * (2.0 / diameter)
            
    return best_points.tolist()

if __name__ == "__main__":
    pts = construct()
    print(f"Best score: {compute_score(np.array(pts))}")
