# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Strategy: Sphere-constrained optimization in spherical coordinates.
    
    1. Constrain all 14 points to lie on the unit sphere (radius 1).
       This eliminates the radial scaling degree of freedom.
       For points on a sphere, d_max is at most 2 (diameter).
       Maximizing (d_min / d_max)^2 is equivalent to maximizing d_min / d_max.
       Since d_max is bounded and d_min is what we want to maximize relative to d_max,
       and d_max tends to be near 2 for well-distributed points, this is a good proxy.
       Actually, strictly speaking, we just maximize the ratio.
       
    2. Parameterize each point by (theta, phi) in [0, pi] x [0, 2*pi].
       Total dimensions: 28.
       
    3. Use multi-start Nelder-Mead to optimize these 28 parameters.
       The objective function is negative of the score (d_min/d_max)^2.
       
    4. To handle the non-smoothness of min/max, Nelder-Mead is a robust choice
       as it does not require gradients.
       
    5. We will use several random starts and also a few symmetric starts 
       (e.g., based on icosahedron or cube projections) to improve convergence.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    N = 14
    
    def sph_to_cart(theta, phi):
        """Convert spherical coords to Cartesian on unit sphere."""
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack([x, y, z])

    def get_points(params):
        """
        params: array of shape (28,) -> [theta_0, phi_0, theta_1, phi_1, ...]
        Returns: array of shape (14, 3)
        """
        thetas = params[0::2]
        phis = params[1::2]
        return sph_to_cart(thetas, phis)

    def objective(params):
        """
        Minimize negative score.
        Score = (d_min / d_max)^2
        """
        pts = get_points(params)
        # Check for validity (though sphere mapping ensures finite coords)
        if np.any(~np.isfinite(pts)):
            return 1e10
        
        dists = pdist(pts, metric='euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max == 0 or np.isnan(d_min) or np.isnan(d_max):
            return 1e10
            
        score = (d_min / d_max) ** 2
        return -score

    best_score = -np.inf
    best_points = None
    best_params = None

    # Generate initial guesses
    # 1. Random starts
    rng = np.random.default_rng(42)
    
    # 2. Symmetric-ish starts
    # Icosahedron has 12 vertices. We can add 2 more (e.g., north/south pole) and perturb.
    # Golden ratio
    phi_golden = (1 + np.sqrt(5)) / 2
    a = 1.0
    b = 1.0 / phi_golden
    
    ico_verts = np.array([
        [0, 1, phi_golden], [0, -1, phi_golden],
        [0, 1, -phi_golden], [0, -1, -phi_golden],
        [1, phi_golden, 0], [-1, phi_golden, 0],
        [1, -phi_golden, 0], [-1, -phi_golden, 0],
        [phi_golden, 0, 1], [-phi_golden, 0, 1],
        [phi_golden, 0, -1], [-phi_golden, 0, -1]
    ])
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # Add 2 points (poles)
    poles = np.array([[0, 0, 1], [0, 0, -1]])
    seed_points = np.vstack([ico_verts, poles])
    
    # Convert seed points to spherical coords
    def cart_to_sph(pts):
        thetas = np.arccos(pts[:, 2])
        phis = np.arctan2(pts[:, 1], pts[:, 0])
        phis = np.mod(phis, 2 * np.pi)
        return np.column_stack([thetas, phis]).flatten()

    initial_params_list = []
    
    # Add the icosahedron-based seed
    initial_params_list.append(cart_to_sph(seed_points))
    
    # Add a few random seeds
    for _ in range(20):
        thetas = rng.uniform(0, np.pi, N)
        phis = rng.uniform(0, 2 * np.pi, N)
        params = np.column_stack([thetas, phis]).flatten()
        initial_params_list.append(params)

    # Run optimization for each seed
    for i, init_params in enumerate(initial_params_list):
        if time.time() > deadline:
            break
            
        # Nelder-Mead options
        # maxiter needs to be high enough for 28D
        # xatol/fatol tight
        res = minimize(
            objective, 
            init_params, 
            method='Nelder-Mead',
            options={
                'xatol': 1e-8, 
                'fatol': 1e-8, 
                'maxiter': 2000,
                'adaptive': True
            }
        )
        
        if res.success:
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x
                best_points = get_points(res.x)

    # If no valid points found (should not happen), fallback
    if best_points is None:
        # Return a simple grid or random points
        thetas = np.linspace(0, np.pi, N)
        phis = np.linspace(0, 2*np.pi, N)
        best_points = get_points(np.column_stack([thetas, phis]).flatten())
        best_score = -objective(np.column_stack([thetas, phis]).flatten())

    # Final check and return
    # Ensure shape is (14, 3)
    best_points = np.asarray(best_points)
    if best_points.shape != (14, 3):
        # Fallback
        best_points = np.random.randn(14, 3)
        best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)

    return best_points
# EVOLVE-BLOCK-END
