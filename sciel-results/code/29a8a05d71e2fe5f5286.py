import numpy as np
import time
from scipy.linalg import eigh

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def _project_to_psd(G):
        """
        Project matrix G onto the set of Positive Semi-Definite matrices.
        This is done by eigendecomposition: keep positive eigenvalues, set negative to 0.
        """
        # Use eigh for symmetric matrices
        vals, vecs = eigh(G)
        # Clip negative eigenvalues to 0
        vals_clipped = np.clip(vals, 0, None)
        return vecs @ np.diag(vals_clipped) @ vecs.T

    def _covering_radius(points):
        """
        Compute the covering radius by sampling the sphere.
        """
        n_theta = 80
        n_phi = 40
        theta = np.linspace(0, np.pi, n_theta, endpoint=False)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        T, P = np.meshgrid(theta, phi)
        x = np.sin(T) * np.cos(P)
        y = np.sin(T) * np.sin(P)
        z = np.cos(T)
        samples = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
        
        dots = samples @ points.T  # (M, N)
        min_dots = np.min(dots, axis=1)
        min_dots = np.clip(min_dots, -1.0, 1.0)
        angles = np.arccos(min_dots)
        return np.max(angles)

    def _gram_to_points(G):
        """
        Convert a PSD Gram matrix to points on the sphere.
        G = X X^T where X is (3, N) or (N, 3).
        We assume G is (N, N).
        We need to extract 3D coordinates.
        Since the points are on the unit sphere, the rank of G must be <= 3.
        We perform eigendecomposition and take the top 3 eigenvectors.
        """
        vals, vecs = eigh(G)
        # Sort in descending order
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
        
        # Take top 3
        V3 = vecs[:, :3]
        # The points are the rows of V3 scaled by sqrt(eigenvalues)?
        # Actually, if G = V D V^T, then X = sqrt(D) V^T.
        # The columns of X are the points.
        # X = V3 * sqrt(vals[:3])
        # Let's construct X (3, N)
        X = (vals[:3] ** 0.5)[:, None] * V3.T
        
        # Normalize rows to unit sphere
        norms = np.linalg.norm(X, axis=0, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        points = X.T / norms
        
        return points

    def _points_to_gram(points):
        """
        Compute Gram matrix from points.
        """
        return points @ points.T

    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    G = _points_to_gram(x0)
    
    best_points = x0
    best_val = _covering_radius(x0)
    
    # Optimization in Gram space
    # We want to maximize the minimum off-diagonal entry of G.
    # This is equivalent to minimizing the maximum angular distance between pairs.
    # While not exactly the covering radius, it's a good proxy and often leads to better packings/coverings.
    # The objective in Gram space: f(G) = -min_{i != j} G[i, j]
    # We want to minimize this (i.e., make the min dot product as large as possible, i.e., close to 1, but constrained by geometry).
    # Wait, maximizing min dot product means making points closer together? No.
    # Max angle corresponds to min dot product.
    # We want to MINIMIZE the max angle.
    # Max angle = arccos(min_dot).
    # Minimizing max angle is equivalent to MAXIMIZING min_dot.
    # So we want to maximize the minimum off-diagonal element of G.
    
    # This is a max-min problem.
    # Let t be the minimum off-diagonal element.
    # Maximize t subject to G[i,j] >= t for all i!=j, G succeq 0, G[i,i]=1.
    # This is a convex problem in G if we fix t? No, t is a variable.
    # But we can use a penalty or just gradient ascent on the min element.
    
    # Gradient of min(G[i,j]) w.r.t G is 1 at the minimizing pair, 0 elsewhere.
    # So we can do:
    # G_new = G + step * (E_min - lambda * G) ?
    # Actually, let's just use a simple iterative projection method (like Dykstra or alternating projections).
    # Project G onto:
    # 1. Set of matrices with G[i,i] = 1
    # 2. Set of matrices with G[i,j] >= t (for some t, or just maximize min)
    # 3. Set of PSD matrices
    
    # Simpler approach: Gradient ascent on the min off-diagonal element.
    # Step 1: Compute G
    # Step 2: Find min off-diagonal element and its indices
    # Step 3: Gradient direction is 1 at those indices, 0 elsewhere.
    # Step 4: Update G += step * gradient
    # Step 5: Enforce G[i,i]=1
    # Step 6: Project to PSD
    # Step 7: Convert to points, evaluate covering radius, update best.
    
    rng = np.random.default_rng(42)
    
    # We will run a loop of these steps.
    # To escape local minima in this landscape, we can add noise.
    
    step_size = 0.05
    num_iterations = 5000
    
    for k in range(num_iterations):
        if time.time() > deadline - 5.0:
            break
            
        # Current G
        # Find min off-diagonal
        mask = ~np.eye(N, dtype=bool)
        min_val = np.min(G[mask])
        # Get indices of min
        # We need the specific pair (i, j) where G[i,j] is min.
        # If there are multiple, pick one or average? Picking one is standard subgradient.
        # Let's find all pairs with value close to min.
        close_mask = mask & (G < min_val + 1e-6)
        
        # Gradient direction
        grad = np.zeros_like(G)
        # Add 0.5 to symmetric pairs to maintain symmetry in the update
        # If we update G[i,j] by delta, we must update G[j,i] by delta.
        # The subgradient of min(x) is a unit vector in the face of the min.
        # Let's pick a random one from the close_mask pairs to add stochasticity.
        pairs = np.argwhere(close_mask)
        if len(pairs) > 0:
            # Pick a few random pairs to update
            num_pairs = min(10, len(pairs))
            selected = rng.choice(len(pairs), num_pairs, replace=False)
            for idx in selected:
                i, j = pairs[idx]
                grad[i, j] += 1.0
                grad[j, i] += 1.0
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad /= grad_norm
            
            # Update G
            # We want to increase the min element.
            # But we also need to keep diagonal at 1.
            # The gradient has 0 on diagonal, so G diagonal stays 0 in the update direction.
            # However, the PSD projection might change the diagonal.
            
            G = G + step_size * grad
            
            # Enforce unit diagonal
            np.fill_diagonal(G, 1.0)
            
            # Symmetrize (just in case of numerical drift)
            G = (G + G.T) / 2.0
            
            # Project to PSD
            G = _project_to_psd(G)
            
            # Enforce unit diagonal again (projection might change it)
            np.fill_diagonal(G, 1.0)
            
            # Convert to points
            points = _gram_to_points(G)
            
            # Check validity (non-zero, etc.)
            if not np.all(np.isfinite(points)):
                continue
                
            # Evaluate covering radius
            val = _covering_radius(points)
            
            if val < best_val:
                best_val = val
                best_points = points
                # If we found a significantly better point, maybe restart or adjust step?
                # For now, just continue.
                
            # Adaptive step size?
            # If we are making progress, keep step. If not, maybe reduce or increase noise.
            # Let's add a bit of noise to G to escape local minima.
            noise = rng.normal(0, 0.01, size=(N, N))
            noise = (noise + noise.T) / 2.0
            np.fill_diagonal(noise, 0)
            G = G + noise
            np.fill_diagonal(G, 1.0)
            G = _project_to_psd(G)
            np.fill_diagonal(G, 1.0)

    # Final check: ensure best_points is valid and normalized
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    best_points = best_points / norms
    
    return best_points
