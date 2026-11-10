import numpy as np
import itertools
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s kill
    
    n = 14
    
    def project_to_sphere(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return pts / norms

    def compute_true_score(pts):
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def smooth_objective(theta_phi):
        """
        Minimize the negative of the squared ratio using smooth approximations.
        theta_phi: array of shape (14, 2) or (28,). We use (28,) for scipy.
        """
        if time.time() > deadline:
            return 0.0
            
        tp = theta_phi.reshape(n, 2)
        theta = tp[:, 0]
        phi = tp[:, 1]
        
        # Convert to Cartesian
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        pts = np.column_stack((x, y, z))
        
        # Compute pairwise distances
        dists = pdist(pts)
        
        # Smooth approximations
        # Large p makes min approximation sharp
        p = 100.0
        q = 100.0
        
        # Avoid division by zero if points coincide
        dists_safe = np.maximum(dists, 1e-8)
        
        sum_inv_p = np.sum(dists_safe ** (-p))
        sum_pow_q = np.sum(dists_safe ** q)
        
        d_min_approx = (sum_inv_p ** (-1.0 / p))
        d_max_approx = (sum_pow_q ** (1.0 / q))
        
        if d_max_approx == 0:
            return 0.0
            
        ratio = d_min_approx / d_max_approx
        # We want to maximize ratio^2, so minimize -ratio^2
        return - (ratio ** 2)

    def smooth_jacobian(theta_phi):
        """
        Numerical gradient for L-BFGS-B if analytical is too complex.
        Actually, let's just use 'L-BFGS-B' which can do finite differences if jac is None,
        but it's slower. Let's provide a numerical gradient function for speed/accuracy balance.
        """
        eps = 1e-6
        grad = np.zeros_like(theta_phi)
        f0 = smooth_objective(theta_phi)
        for i in range(len(theta_phi)):
            theta_phi_plus = theta_phi.copy()
            theta_phi_plus[i] += eps
            f_plus = smooth_objective(theta_phi_plus)
            grad[i] = (f_plus - f0) / eps
            theta_phi_plus[i] -= 2 * eps
            f_minus = smooth_objective(theta_phi_plus)
            theta_phi_plus[i] += eps # restore
            # Central difference is more accurate
            grad[i] = (f_plus - f_minus) / (2 * eps)
        return grad

    def cartesian_objective(x_cart):
        """
        Objective for final polish in Cartesian coordinates.
        Projects to sphere internally.
        """
        pts = x_cart.reshape(n, 3)
        pts = project_to_sphere(pts)
        score = compute_true_score(pts)
        return -score

    # 1. Generate Initial Configurations
    initial_guesses = []
    
    # A. Dodecahedron based
    phi_gold = (1 + np.sqrt(5)) / 2.0
    verts = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                verts.append([x, y, z])
    for a in [-1/phi_gold, 1/phi_gold]:
        for b in [-phi_gold, phi_gold]:
            verts.append([0, a, b])
            verts.append([a, b, 0])
            verts.append([b, 0, a])
    verts = np.array(verts)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Find best subset of 14 from Dodecahedron (quick combinatorial check)
    best_dodeca_score = 0
    best_dodeca_pts = None
    # 20 choose 14 is 38760, very fast
    for combo in itertools.combinations(range(20), 14):
        pts = verts[list(combo)]
        score = compute_true_score(pts)
        if score > best_dodeca_score:
            best_dodeca_score = score
            best_dodeca_pts = pts.copy()
            
    if best_dodeca_pts is not None:
        initial_guesses.append(best_dodeca_pts)

    # B. Fibonacci Sphere
    def fib_sphere(n_pts):
        pts = []
        golden_angle = np.pi * (3. - np.sqrt(5.))
        for i in range(n_pts):
            y = 1 - (i / float(n_pts - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = golden_angle * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            pts.append([x, y, z])
        return np.array(pts)
    
    initial_guesses.append(fib_sphere(14))

    # C. Random configurations
    np.random.seed(42)
    for _ in range(10):
        pts = np.random.randn(14, 3)
        pts = project_to_sphere(pts)
        initial_guesses.append(pts)

    # 2. Optimize using Smooth Surrogate in Spherical Coordinates
    best_candidates = []
    
    for guess in initial_guesses:
        if time.time() > deadline - 60:
            break
            
        # Convert Cartesian to Spherical
        # x = r sin(theta) cos(phi), y = r sin(theta) sin(phi), z = r cos(theta)
        # On sphere r=1
        x, y, z = guess[:, 0], guess[:, 1], guess[:, 2]
        theta = np.arccos(np.clip(z, -1.0, 1.0))
        phi = np.arctan2(y, x)
        
        # Handle singularities at poles (theta=0 or pi)
        # phi is undefined, but arctan2 handles it.
        # We need to ensure theta is in [0, pi]
        theta_phi = np.column_stack((theta, phi)).flatten()
        
        try:
            res = minimize(
                smooth_objective, 
                theta_phi, 
                method='L-BFGS-B', 
                jac=smooth_jacobian,
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            # Convert result back to Cartesian
            tp = res.x.reshape(n, 2)
            theta = tp[:, 0]
            phi = tp[:, 1]
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            pts = np.column_stack((x, y, z))
            pts = project_to_sphere(pts)
            
            true_score = compute_true_score(pts)
            best_candidates.append((true_score, pts))
            
        except Exception:
            continue

    # 3. Final Polish with Nelder-Mead on True Objective
    # Sort candidates by score
    best_candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_final_points = best_candidates[0][1] if best_candidates else initial_guesses[0]
    best_final_score = best_candidates[0][0] if best_candidates else compute_true_score(best_final_points)
    
    # Take the top 3 candidates and polish them
    for score, pts in best_candidates[:3]:
        if time.time() > deadline - 30:
            break
            
        x0 = pts.flatten()
        try:
            res = minimize(
                cartesian_objective, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            opt_pts = project_to_sphere(res.x.reshape(n, 3))
            opt_score = compute_true_score(opt_pts)
            
            if opt_score > best_final_score:
                best_final_score = opt_score
                best_final_points = opt_pts
                
        except Exception:
            pass

    return best_final_points

if __name__ == "__main__":
    # Test run
    pts = min_max_dist_dim3_14()
    print(pts.shape)
    dists = pdist(pts)
    print(f"Score: {(np.min(dists)/np.max(dists))**2}")
