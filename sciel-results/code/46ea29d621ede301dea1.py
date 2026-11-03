import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_hermite
from numpy.polynomial.hermite import hermgauss

def construct():
    t_start = time.time()
    deadline = 205.0  # 5s margin from 210s

    # Degrees
    degrees = [0, 4, 8, 12]

    # Precompute Hermite values on a grid for the sup-norm
    # The function decays as e^{-pi x^2}, so we need a range where the Gaussian is non-negligible
    # e^{-pi * 8^2} is ~ 1e-111, so [-8, 8] is sufficient for high precision
    x_grid = np.linspace(-8, 8, 4000)
    H_grid = np.array([eval_hermite(d, x_grid) for d in degrees])
    gauss_factor = np.exp(-np.pi * x_grid**2)

    # Gauss-Hermite for norm calculation
    # n_points = 200 is usually sufficient for high accuracy with polynomials of degree <= 12
    n_points = 200
    y_points, y_weights = hermgauss(n_points)
    # Standard Hermite polynomials are orthogonal w.r.t e^{-x^2}.
    # The task uses e^{-pi x^2}. 
    # Let's check the norm definition carefully.
    # Usually ||f||_2^2 = int |f|^2 dx.
    # If g(x) = P(x) e^{-pi x^2}, then ||g||^2 = int |P(x)|^2 e^{-2 pi x^2} dx.
    # Standard Hermite H_n(x) are orthogonal w.r.t e^{-x^2}.
    # We need to be careful with the weights.
    # Let's just use numerical integration or adjust weights.
    # Actually, the seed program used:
    # x_points = y_points / np.sqrt(2 * np.pi)
    # H_points = eval_hermite(d, x_points)
    # norm_sq = sum((P_points**2) * y_weights) / np.sqrt(2 * np.pi)
    # This looks like a transformation to standard normal or something specific.
    # Let's stick to the seed's norm calculation to ensure consistency with the "score" definition,
    # unless we can verify it. The seed scored 0.1145.
    # Let's re-implement the norm calculation exactly as the seed to be safe, 
    # assuming the seed's interpretation of "upper bound on C4" is correct for the evaluator.
    
    x_points = y_points / np.sqrt(2 * np.pi)
    H_points = np.array([eval_hermite(d, x_points) for d in degrees])

    def get_c12(c0, c4, c8):
        # c0*1 + c4*(-3) + c8*(105) + c12*(10395) = 0
        # H0(0)=1, H4(0)=-3, H8(0)=105, H12(0)=10395
        return (-c0 + 3*c4 - 105*c8) / 10395.0

    def objective_from_coeffs(coeffs):
        """
        coeffs: array of shape (4,) [c0, c4, c8, c12]
        Returns the objective value.
        """
        # Calculate P on grid
        P_grid = H_grid.T @ coeffs
        sup_val = np.max(np.abs(P_grid * gauss_factor))
        
        # Calculate Norm
        P_points = H_points.T @ coeffs
        norm_sq = np.sum((P_points**2) * y_weights) / np.sqrt(2 * np.pi)
        
        if norm_sq < 1e-12:
            return 1e12
            
        return (sup_val**2) / norm_sq

    def objective_from_angles(params):
        """
        params: array of shape (2,) [theta, phi]
        Converts spherical coordinates to Cartesian (c0, c4, c8) on the unit sphere.
        The scale is arbitrary, so we fix it to 1.
        c0 = sin(theta) * cos(phi)
        c4 = sin(theta) * sin(phi)
        c8 = cos(theta)
        """
        theta, phi = params
        c0 = np.sin(theta) * np.cos(phi)
        c4 = np.sin(theta) * np.sin(phi)
        c8 = np.cos(theta)
        
        c12 = get_c12(c0, c4, c8)
        coeffs = np.array([c0, c4, c8, c12])
        
        return objective_from_coeffs(coeffs)

    # Initial best from seed
    seed_params = np.array([-113/100, 1/25, 1/3240])
    seed_c12 = get_c12(seed_params[0], seed_params[1], seed_params[2])
    seed_coeffs = np.array([seed_params[0], seed_params[1], seed_params[2], seed_c12])
    best_score = objective_from_coeffs(seed_coeffs)
    best_coeffs = seed_coeffs.copy()

    # Global Optimization using Differential Evolution
    # Bounds for theta: [0, pi], phi: [-pi, pi]
    bounds = [(0, np.pi), (-np.pi, np.pi)]
    
    # We need to be careful with the time budget.
    # DE can be slow. Let's use a reasonable number of iterations.
    # popsize=15, maxiter=200 might take ~10-20s depending on objective speed.
    # Objective is ~4000 points vectorized, very fast.
    
    try:
        if time.time() - t_start < deadline - 30:
            res = differential_evolution(
                objective_from_angles, 
                bounds, 
                maxiter=300, 
                popsize=20, 
                tol=1e-10,
                mutation=(0.5, 1.0),
                recombination=0.9,
                seed=42
            )
            if res.fun < best_score:
                best_score = res.fun
                theta, phi = res.x
                c0 = np.sin(theta) * np.cos(phi)
                c4 = np.sin(theta) * np.sin(phi)
                c8 = np.cos(theta)
                c12 = get_c12(c0, c4, c8)
                best_coeffs = np.array([c0, c4, c8, c12])
    except Exception as e:
        pass

    # Polish with Nelder-Mead in the original 3D space (or 2D angle space?)
    # Since the objective is scale invariant, the 3D space has a "valley" of constant depth.
    # Nelder-Mead in 3D might struggle to converge to a specific point but will find the minimum value.
    # However, polishing in the 2D angle space is safer because the manifold is smooth and bounded.
    
    if time.time() - t_start < deadline - 10:
        # Extract current angles from best_coeffs
        c0, c4, c8 = best_coeffs[0], best_coeffs[1], best_coeffs[2]
        norm_c = np.linalg.norm([c0, c4, c8])
        if norm_c > 1e-12:
            c0_n, c4_n, c8_n = c0/norm_c, c4/norm_c, c8/norm_c
            # Convert to spherical
            # c0 = sin(theta) cos(phi), c4 = sin(theta) sin(phi), c8 = cos(theta)
            cos_theta = c8_n
            sin_theta = np.sqrt(1 - cos_theta**2)
            if sin_theta > 1e-12:
                cos_phi = c0_n / sin_theta
                sin_phi = c4_n / sin_theta
                phi = np.arctan2(sin_phi, cos_phi)
            else:
                phi = 0
            theta = np.arccos(cos_theta)
            initial_angles = np.array([theta, phi])
            
            try:
                res2 = minimize(
                    objective_from_angles, 
                    initial_angles, 
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if res2.fun < best_score:
                    best_score = res2.fun
                    theta, phi = res2.x
                    c0 = np.sin(theta) * np.cos(phi)
                    c4 = np.sin(theta) * np.sin(phi)
                    c8 = np.cos(theta)
                    c12 = get_c12(c0, c4, c8)
                    best_coeffs = np.array([c0, c4, c8, c12])
            except Exception:
                pass

    # Return the first 3 coefficients as required by the interface
    # The interface expects "three coefficients of the Hermite combination (degrees 0, 4, 8)"
    # The degree-12 weight is fixed by the constraint g(0)=0.
    return best_coeffs[:3]
