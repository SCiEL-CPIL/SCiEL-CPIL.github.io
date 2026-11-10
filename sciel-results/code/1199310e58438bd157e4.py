import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin for 120s limit
    
    # The problem is equivalent to maximizing the product of distances 
    # for 8 points on a unit circle, because the diameter of any set of points 
    # on a unit circle is 2 (the diameter of the circle), and scaling to 
    # diameter 2 leaves the points on the unit circle if they are the 
    # maximizers.
    # Actually, the diameter of the set is the max distance. For points on 
    # a unit circle, the max distance is 2 (if antipodal) or less.
    # The score is Delta / 8^8 where Delta is the product of distances 
    # after scaling to diameter 2.
    # If we place points on the unit circle, the diameter is at most 2.
    # If the diameter is D, we scale by 2/D. The product of distances scales 
    # by (2/D)^28.
    # So Score = (Product of distances on unit circle) * (2/D)^28 / 8^8.
    # To maximize this, we want to maximize Product / D^28.
    
    def score_from_angles(angles):
        # Ensure angles are sorted? No, order doesn't matter for product.
        # But Nelder-Mead works better with continuous variables.
        # Angles are in [0, 2*pi).
        angles = np.asarray(angles)
        # Wrap to [0, 2*pi)
        angles = np.mod(angles, 2 * np.pi)
        
        # Sort angles to make the geometry well-defined and avoid 
        # redundant permutations in the optimization landscape.
        angles = np.sort(angles)
        
        x = np.cos(angles)
        y = np.sin(angles)
        points = np.column_stack((x, y))
        
        # Calculate pairwise distances
        # n=8, so 28 pairs
        dists = []
        for i in range(8):
            for j in range(i + 1, 8):
                d = np.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
                dists.append(d)
        
        dists = np.array(dists)
        product = np.prod(dists)
        
        # Diameter of the set
        D = np.max(dists)
        
        if D < 1e-10:
            return -1e10
            
        # Score = Product * (2/D)^28 / 8^8
        # Log score for stability?
        # log_score = sum(log(dists)) + 28*log(2) - 28*log(D) - 8*log(8)
        # Maximize log_score is equivalent to maximizing score.
        
        log_score = np.sum(np.log(dists)) + 28 * np.log(2) - 28 * np.log(D) - 8 * np.log(8)
        
        # We want to minimize negative log score
        return -log_score

    # Initial guess: Regular Octagon
    angles0 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    
    # Optimization
    best_score = score_from_angles(angles0)
    best_angles = angles0.copy()
    
    # Nelder-Mead is derivative-free and good for this smooth-ish function
    # (though D has kinks, the log form is relatively stable)
    
    # We can try multiple restarts if time permits
    max_restarts = 5
    restarts_done = 0
    
    while restarts_done < max_restarts and time.time() < deadline:
        # For restarts, perturb the initial guess
        if restarts_done > 0:
            # Small random perturbation
            angles0_trial = angles0 + np.random.uniform(-0.1, 0.1, 8)
        else:
            angles0_trial = angles0
            
        # Run minimization
        try:
            res = minimize(score_from_angles, angles0_trial, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'fatol': 1e-12, 'xatol': 1e-12, 'adaptive': True})
            
            if res.success or res.fun < best_score:
                best_score = res.fun
                best_angles = res.x
        except Exception:
            pass
        
        restarts_done += 1
        
        # Check time
        if time.time() > deadline - 10:
            break
    
    # Convert best angles to points
    angles = np.mod(best_angles, 2 * np.pi)
    angles = np.sort(angles)
    
    x = np.cos(angles)
    y = np.sin(angles)
    
    # Return as list of lists of floats
    points = [[float(x[i]), float(y[i])] for i in range(8)]
    
    return points
