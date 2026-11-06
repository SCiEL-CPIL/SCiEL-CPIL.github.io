"""
Strategy:
1. Fix the diameter constraint by construction: place two points at (-1, 0) and (1, 0).
   This ensures the diameter is at least 2. We then optimize the other 6 points.
   To ensure the diameter is exactly 2 (or close to it, as the evaluator scales), 
   we must ensure no pair of points has distance > 2. 
   Actually, the evaluator scales the set to diameter 2. So we just need to maximize 
   the product of distances for a set with diameter <= 2? 
   No, the score is Delta / n^n where Delta is product of distances AFTER scaling to diameter 2.
   This is equivalent to maximizing (Product of Distances) / (Diameter)^(n choose 2).
   Let P be the product of all pairwise distances. Let D be the diameter.
   Score = (P / D^28) / 8^8 ? 
   Wait, the contract says: "score is Delta/n^n, where Delta is the product over ordered pairs of distances after scaling the set to diameter 2".
   "Ordered pairs" usually means i != j. Number of ordered pairs is n(n-1) = 56.
   "Product over ordered pairs" = (Product over unordered pairs)^2.
   Let U be product of unordered distances. Delta = U^2.
   If we scale to diameter 2, distances become d_ij * (2/D).
   Delta_scaled = (U * (2/D)^28)^2 = U^2 * (2/D)^56.
   Score = Delta_scaled / 8^8.
   Since 8^8 is constant, we maximize U^2 * (2/D)^56.
   This is equivalent to maximizing U / D^28.
   
   So the objective is: Maximize log(U) - 28 * log(D).
   
   If we fix two points at distance 2, D >= 2. If we ensure all other points are within distance 2 of each other and the fixed points, D might be > 2.
   If D > 2, the penalty is high. 
   
   Better approach: 
   Parameterize 8 points freely. 
   Use a global optimizer (Differential Evolution) on the 16 coordinates.
   Objective: log(U) - 28 * log(D).
   This is smooth except where D is determined by different pairs, but DE is robust to non-smoothness.
   However, D is the max of distances. 
   
   Alternative:
   Fix 2 points at (-1,0) and (1,0). 
   Optimize the other 6 points (12 variables).
   Objective: log(U) - 28 * log(D).
   Here D = max(2, max_dist(other points), max_dist(fixed, other)).
   Since fixed points are at dist 2, D >= 2.
   If the other points stay within the unit disk centered at origin, D=2.
   If they go outside, D increases, penalizing the score.
   This effectively constrains the points to the intersection of two unit disks (the lens shape) if we want D=2.
   But the optimal configuration might have D > 2 in the unscaled version? 
   No, scaling is invariant. We can assume WLOG that the diameter is 2.
   So we can constrain the search space to points within distance 2 of each other.
   Specifically, if we fix p0=(-1,0) and p1=(1,0), then all other points p_i must satisfy:
   ||p_i - p0|| <= 2 and ||p_i - p1|| <= 2 and ||p_i - p_j|| <= 2 for all i,j.
   The region defined by ||p_i - p0|| <= 2 and ||p_i - p1|| <= 2 is the intersection of two disks of radius 2 centered at (-1,0) and (1,0).
   This region is a lens. The points must also be pairwise distance <= 2.
   
   Let's use DE to optimize the 12 coordinates of the 6 free points.
   Bounds: x in [-2, 2], y in [-2, 2] is a safe box. The objective will push them inside the valid region.
   
   Objective function for DE:
   def obj(x):
       pts = [[-1,0], [1,0]] + list of 6 points from x
       Calculate U = product of 28 distances.
       Calculate D = max distance.
       return -(log(U) - 28*log(D))  # Minimize negative score
       
   DE is slow for 12 variables? No, 12 is small. 
   We can run DE for a fixed number of generations.
   Then polish with BFGS on the best result.
"""

import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 8
    
    def get_score(pts):
        # pts is (8, 2)
        # Calculate product of unordered distances U
        log_U = 0.0
        max_d = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 1e-12:
                    return -1e9  # Invalid
                log_U += np.log(d)
                if d > max_d:
                    max_d = d
        
        if max_d < 1e-12:
            return -1e9
            
        # Score proportional to U / max_d^28
        # We want to maximize log_U - 28 * log(max_d)
        return log_U - 28.0 * np.log(max_d)

    def objective_de(x):
        # x is 12 values: 6 points
        # Fixed points: p0 = (-1, 0), p1 = (1, 0)
        pts = np.empty((8, 2))
        pts[0] = [-1.0, 0.0]
        pts[1] = [1.0, 0.0]
        for i in range(6):
            pts[i+2] = [x[2*i], x[2*i+1]]
        
        score = get_score(pts)
        # Minimize negative score
        if score < -1e8:
            return 1e9
        return -score

    # Bounds for the 6 free points
    # They should be in the lens intersection of disks radius 2 centered at (-1,0) and (1,0)
    # x range: -2 to 2
    # y range: -sqrt(4-1)= -sqrt(3) to sqrt(3) approx -1.732 to 1.732
    # But to be safe and allow DE to explore, use a slightly larger box.
    # The objective penalizes large distances, so it will pull them in.
    bounds = [(-2.5, 2.5), (-2.5, 2.5)] * 6
    
    # Run Differential Evolution
    # maxiter=100, popsize=20 might be enough for 12 dims
    # seed for reproducibility
    try:
        de_res = differential_evolution(
            objective_de, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=15,
            tol=1e-8,
            polish=False  # We will do our own polish
        )
    except Exception:
        # Fallback to regular octagon if DE fails
        return [[np.cos(2*np.pi*k/8), np.sin(2*np.pi*k/8)] for k in range(8)]
    
    if time.time() > deadline:
         # Return best found so far? DE doesn't return intermediate best easily in this call.
         # Let's just return the octagon if we are out of time.
         return [[np.cos(2*np.pi*k/8), np.sin(2*np.pi*k/8)] for k in range(8)]

    best_x = de_res.x
    best_pts = np.empty((8, 2))
    best_pts[0] = [-1.0, 0.0]
    best_pts[1] = [1.0, 0.0]
    for i in range(6):
        best_pts[i+2] = [best_x[2*i], best_x[2*i+1]]
    
    # Polish with BFGS
    # We need to optimize all 16 coordinates now? 
    # Or keep the fixed points? 
    # The optimal solution likely has the diameter pair at distance 2. 
    # But which pair? 
    # The DE run assumed p0, p1 are the diameter pair. 
    # To be robust, we should try to polish the full 8 points.
    
    def objective_full(x):
        pts = x.reshape(8, 2)
        score = get_score(pts)
        if score < -1e8:
            return 1e9
        return -score

    def grad_full(x):
        pts = x.reshape(8, 2)
        grad = np.zeros_like(pts)
        # log_U = sum log(d_ij)
        # d = ||p_i - p_j||
        # grad log_U wrt p_k = sum_{j!=k} (p_k - p_j) / d_kj^2
        
        # D = max d_ij. 
        # grad D is tricky. 
        # Approximation: assume the max pair is fixed locally.
        max_d = 0
        max_i, max_j = 0, 1
        for i in range(8):
            for j in range(i+1, 8):
                d = np.linalg.norm(pts[i] - pts[j])
                if d > max_d:
                    max_d = d
                    max_i, max_i_j = i, j
        
        # Calculate grad of log_U
        for i in range(8):
            for j in range(i+1, 8):
                diff = pts[i] - pts[j]
                d_sq = np.dot(diff, diff)
                if d_sq < 1e-12:
                    continue
                d = np.sqrt(d_sq)
                # grad log(d) wrt p_i = (p_i - p_j) / d^2
                grad[i] += diff / d_sq
                grad[j] -= diff / d_sq
        
        # Calculate grad of -28 log(D)
        # = -28 / D * grad D
        # grad D wrt p_k:
        # If D is determined by pair (a,b), grad D wrt p_a = (p_a - p_b)/D, wrt p_b = (p_b - p_a)/D
        # Other points have 0 gradient for D (locally)
        
        diff_max = pts[max_i] - pts[max_j]
        if max_d > 1e-9:
            grad[max_i] -= (28.0 / max_d) * (diff_max / max_d)
            grad[max_j] += (28.0 / max_d) * (diff_max / max_d)
            
        # Total gradient for minimization: -grad(log_U - 28 log D) = -grad_log_U + 28/D * grad_D
        # Wait, objective is - (log_U - 28 log D) = -log_U + 28 log D
        # grad obj = -grad_log_U + 28/D * grad_D
        
        # My grad variable above is grad_log_U.
        # So return -grad + 28/D * grad_D_part
        
        # Let's recompute cleanly
        grad_obj = -grad
        
        if max_d > 1e-9:
            grad_obj[max_i] += (28.0 / max_d) * (diff_max / max_d)
            grad_obj[max_j] += (28.0 / max_d) * (-diff_max / max_d) # Wait, grad D wrt p_b is (p_b-p_a)/D = -diff/D
            # So term is 28/D * (-diff/D) for p_j
            # My code: grad_obj[max_j] += (28.0 / max_d) * (-diff_max / max_d)
            # This is correct.
            
        return grad_obj.flatten()

    x0 = best_pts.flatten()
    
    try:
        res = minimize(
            objective_full, 
            x0, 
            method='L-BFGS-B', 
            jac=grad_full,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        polished_pts = res.x.reshape(8, 2)
        
        # Verify polish improved score
        if get_score(polished_pts) > get_score(best_pts):
            best_pts = polished_pts
            
    except Exception:
        pass

    # Final scaling to ensure diameter is exactly 2 for output (optional but good practice)
    d_final = 0
    for i in range(8):
        for j in range(i+1, 8):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d > d_final:
                d_final = d
    if d_final > 1e-6:
        best_pts = best_pts * (2.0 / d_final)
        
    return [[float(x), float(y)] for x, y in best_pts]

# EVOLVE-BLOCK-END
