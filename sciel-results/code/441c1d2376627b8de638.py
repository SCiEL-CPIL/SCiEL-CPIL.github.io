import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def get_points_from_params(params):
        """
        Generate 8 points from a parameter vector.
        We use a general 16D parameterization for the final polish, 
        but for the initial search, we can use the symmetric 'two squares' family.
        However, to keep it simple and robust, we will just do a direct 16D search 
        initialized with the 'two squares' configuration.
        """
        # params is a flat array of 16 floats: x1,y1, ..., x8,y8
        pts = params.reshape(8, 2)
        return pts

    def normalize_and_score(pts):
        """
        Scale points to diameter 2 and return the score.
        Score = (Product of ordered pair distances) / 8^8
        Note: Product of ordered pairs = (Product of unordered pairs)^2
        """
        # Calculate pairwise distances
        # dist[i, j] = || p_i - p_j ||
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Diameter is the max distance
        diam = np.max(dists)
        if diam < 1e-10:
            return 0.0
        
        # Scale to diameter 2
        # If we scale points by 2/diam, distances scale by 2/diam.
        # Product of distances scales by (2/diam)^N_pairs
        # Number of ordered pairs = 8*7 = 56
        # Number of unordered pairs = 28
        
        # We want to maximize the product of distances of the *scaled* set.
        # Let P be the product of unordered pair distances for the original set.
        # Scaled product = P * (2/diam)^28
        # Score = (Scaled Product)^2 / 8^8  (if ordered)
        # Or Score = Scaled Product / 8^8? 
        # Prompt: "Delta is the product over ordered pairs ... score is Delta/n^n"
        # So Delta = (P_unordered)^2.
        # Score = (P_unordered * (2/diam)^28)^2 / 8^8
        
        # It is easier to compute log score for stability
        # log(Delta) = 2 * sum_{i<j} log(d_ij)
        # We exclude i==j where d=0
        mask = ~np.eye(8, dtype=bool)
        # Only use upper triangle to avoid double counting in log sum, then multiply by 2 for ordered
        iu = np.triu_indices(8, k=1)
        d_unordered = dists[iu]
        
        if np.any(d_unordered < 1e-10):
            return 0.0
            
        log_p_unordered = np.sum(np.log(d_unordered))
        # Product of unordered pairs for scaled set:
        # P_scaled = P_orig * (2/diam)^28
        # log(P_scaled) = log_p_unordered + 28 * log(2/diam)
        
        # Delta = P_scaled^2
        # log(Delta) = 2 * log(P_scaled)
        
        log_delta = 2 * (log_p_unordered + 28 * (np.log(2) - np.log(diam)))
        
        # Score = Delta / 8^8
        log_score = log_delta - 8 * np.log(8)
        
        # Convert back to linear space for the optimizer? 
        # Optimizers work better with the value itself or negative log.
        # We want to MAXIMIZE log_score.
        # So we MINIMIZE -log_score.
        return -log_score

    def objective(params):
        pts = get_points_from_params(params)
        return normalize_and_score(pts)

    # Initial Guess: Two Squares
    # Square 1: Radius R1, angles 0, 90, 180, 270
    # Square 2: Radius R2, angles 45, 135, 225, 315
    # We can optimize R1, R2 first.
    
    def two_squares_params(r1, r2):
        pts = np.zeros((8, 2))
        # Square 1
        pts[0] = [r1, 0]
        pts[1] = [0, r1]
        pts[2] = [-r1, 0]
        pts[3] = [0, -r1]
        
        # Square 2 (rotated 45 deg)
        c = r2 / np.sqrt(2)
        pts[4] = [c, c]
        pts[5] = [-c, c]
        pts[6] = [-c, -c]
        pts[7] = [c, -c]
        
        return pts.flatten()

    # Search for best R1, R2 in the two-squares family
    # We can just run a local optimization on the 16D space starting from a few guesses.
    # Guess 1: Regular Octagon equivalent in this param? 
    # For regular octagon, all radii are equal. R1 = R2.
    # Let's try R1=1, R2=1.
    
    initial_params_list = []
    
    # Regular Octagon (R1=R2=1)
    initial_params_list.append(two_squares_params(1.0, 1.0))
    
    # "Two Squares" known to be better. 
    # Often R1 != R2. 
    # Let's try a few variations.
    initial_params_list.append(two_squares_params(1.0, 0.8))
    initial_params_list.append(two_squares_params(1.0, 1.2))
    initial_params_list.append(two_squares_params(1.2, 1.0))
    initial_params_list.append(two_squares_params(0.8, 1.0))
    
    # Also a random symmetric guess
    initial_params_list.append(two_squares_params(1.1, 0.9))
    
    best_score = -np.inf
    best_params = None
    
    # Optimize each initial guess
    for i, x0 in enumerate(initial_params_list):
        if time.time() > deadline - 10.0:
            break
            
        try:
            # Use L-BFGS-B for fast convergence
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_params = res.x
        except Exception:
            pass

    # If we have a good candidate, do a final high-precision polish with Nelder-Mead
    # Nelder-Mead is derivative-free and robust for final polishing
    if best_params is not None and time.time() < deadline - 5.0:
        try:
            res_final = minimize(objective, best_params, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res_final.success:
                score_final = -res_final.fun
                if score_final > best_score:
                    best_score = score_final
                    best_params = res_final.x
        except Exception:
            pass

    # If no params found, fallback to regular octagon
    if best_params is None:
        best_params = two_squares_params(1.0, 1.0)

    # Extract final points
    final_pts = get_points_from_params(best_params)
    
    # Ensure distinct points (they should be, but just in case of numerical issues)
    # Check if any points are too close
    for i in range(8):
        for j in range(i+1, 8):
            if np.linalg.norm(final_pts[i] - final_pts[j]) < 1e-8:
                # Perturb slightly
                final_pts[j][0] += 1e-5

    # Convert to list of lists of floats
    return [[float(p[0]), float(p[1])] for p in final_pts]

if __name__ == "__main__":
    construct()
