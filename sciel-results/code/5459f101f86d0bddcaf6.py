import numpy as np
from scipy.optimize import minimize
import itertools
import cmath
import math

def construct():
    """
    Finds 8 points maximizing the product of pairwise distances normalized by diameter.
    Uses a parameterization based on the coefficients of a degree-8 polynomial with real coefficients.
    The roots of this polynomial are the 8 points.
    """
    
    def get_points_from_coeffs(coeffs):
        """
        coeffs: array of 8 floats [a7, a6, ..., a0] for polynomial x^8 + a7*x^7 + ... + a0
        Returns: array of 8 complex numbers (roots)
        """
        # Monic polynomial: [1, a7, a6, ..., a0]
        poly_coeffs = np.concatenate(([1.0], coeffs))
        roots = np.roots(poly_coeffs)
        return roots

    def log_objective(coeffs):
        """
        Computes -log(J(P)) where J(P) = prod(d_ij) / D^56
        We minimize this.
        """
        try:
            roots = get_points_from_coeffs(coeffs)
        except Exception:
            return 1e10

        # Check for distinct roots
        # Distance matrix
        diff = roots[:, np.newaxis] - roots[np.newaxis, :]
        dists = np.abs(diff)
        
        # Exclude self-distances
        mask = ~np.eye(8, dtype=bool)
        valid_dists = dists[mask]
        
        # If any distance is too small, penalty
        min_dist = np.min(valid_dists)
        if min_dist < 1e-9:
            return 1e10
            
        # Calculate log of product of distances
        log_prod = np.sum(np.log(valid_dists))
        
        # Diameter
        D = np.max(dists)
        
        if D < 1e-9:
            return 1e10
            
        # Normalized score: Delta / D^56
        # log(Score) = log_prod - 56 * log(D)
        log_J = log_prod - 56.0 * np.log(D)
        
        return -log_J

    # Initial guesses
    # 1. Regular Octagon
    # Vertices: exp(i * k * pi/4) for k=0..7
    # Polynomial with these roots: (x^2 - 2x + 1)(x^2 + 2x + 1)(x^2 - 2x + 1)(x^2 + 2x + 1)? No.
    # Roots are 1, e^{i pi/4}, i, e^{i 3pi/4}, -1, e^{i 5pi/4}, -i, e^{i 7pi/4}.
    # Polynomial: (x-1)(x+1)(x-i)(x+i)(x-e^{i pi/4})(x-e^{-i pi/4})(x-e^{i 3pi/4})(x-e^{-i 3pi/4})
    # = (x^2-1)(x^2+1)(x^2 - 2cos(pi/4)x + 1)(x^2 - 2cos(3pi/4)x + 1)
    # = (x^4-1)(x^2 - sqrt(2)x + 1)(x^2 + sqrt(2)x + 1)
    # = (x^4-1)(x^4 + 2x^2 + 1) = (x^4-1)(x^2+1)^2 = (x^4-1)(x^4+2x^2+1)
    # = x^8 + 2x^6 + x^4 - x^4 - 2x^2 - 1 = x^8 + 2x^6 - 2x^2 - 1
    # Coeffs: [0, 2, 0, 0, -2, 0, 0, -1]
    
    init_oct = np.array([0.0, 2.0, 0.0, 0.0, -2.0, 0.0, 0.0, -1.0])
    
    # 2. Two Squares (rotated)
    # Square 1: 1, -1, i, -i -> (x^2-1)(x^2+1) = x^4-1
    # Square 2: rotated by 45 deg? No, that's the octagon.
    # Let's try a square and another square scaled/shifted?
    # Maybe just random starts are enough.
    
    best_score = -1e10
    best_coeffs = init_oct
    
    rng = np.random.default_rng(42)
    
    starts = [init_oct]
    
    # Add random perturbations to the regular octagon
    for _ in range(20):
        pert = rng.normal(0, 0.1, size=8)
        starts.append(init_oct + pert)
        
    # Add some random starts
    for _ in range(10):
        starts.append(rng.normal(0, 1.0, size=8))
        
    # Add "two circles" guess?
    # Roots on two concentric circles?
    # x^8 - R1^4 x^4 - R2^4 = 0? No.
    
    for start in starts:
        try:
            # BFGS or Nelder-Mead?
            # Nelder-Mead is robust for non-smooth-ish landscapes but slower.
            # The objective is smooth if roots are distinct.
            # Let's use Nelder-Mead for robustness.
            res = minimize(log_objective, start, method='Nelder-Mead', 
                           options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 2000, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x
        except Exception:
            pass

    # Get the points
    pts = get_points_from_coeffs(best_coeffs)
    
    # Scale to diameter 2
    diff = pts[:, np.newaxis] - pts[np.newaxis, :]
    dists = np.abs(diff)
    D = np.max(dists)
    
    if D > 1e-12:
        scale_factor = 2.0 / D
        pts = pts * scale_factor
        
    # Convert to list of [x, y]
    result = [[float(p.real), float(p.imag)] for p in pts]
    
    return result

if __name__ == "__main__":
    construct()
