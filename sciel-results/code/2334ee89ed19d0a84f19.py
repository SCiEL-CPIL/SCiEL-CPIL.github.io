import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    def evaluate(centers, sides):
        """
        Returns (is_valid, score).
        score is sum of sides.
        """
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False, 0.0
        if np.any(sides <= 0):
            return False, 0.0
        
        tol = 1e-7
        # Check boundaries
        for i in range(n):
            cx, cy, s = centers[i, 0], centers[i, 1], sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False, 0.0
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False, 0.0
        
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j])/2 - tol
                if dx < t and dy < t:
                    return False, 0.0
        
        return True, np.sum(sides)

    def penalty_objective(x):
        """
        Objective for scipy.optimize.minimize.
        x: [cx0, cy0, s0, cx1, cy1, s1, ...]
        """
        cx = x[0::3]
        cy = x[1::3]
        s = x[2::3]
        
        # Enforce positive sides softly
        s = np.maximum(s, 1e-6)
        
        penalty = 0.0
        tol = 1e-7
        
        # Boundary penalties (quadratic for smoothness)
        for i in range(n):
            # Left
            if cx[i] - s[i]/2 < 0:
                penalty += 1000 * (0 - (cx[i] - s[i]/2))**2
            # Right
            if cx[i] + s[i]/2 > 1:
                penalty += 1000 * ((cx[i] + s[i]/2) - 1)**2
            # Bottom
            if cy[i] - s[i]/2 < 0:
                penalty += 1000 * (0 - (cy[i] - s[i]/2))**2
            # Top
            if cy[i] + s[i]/2 > 1:
                penalty += 1000 * ((cy[i] + s[i]/2) - 1)**2
                
        # Overlap penalties
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                dist_needed = (s[i] + s[j]) / 2
                # If they overlap, push them apart.
                # Use a smooth penalty: if dx < dist_needed and dy < dist_needed
                # Penalty is proportional to the overlap area or penetration depth.
                # A common trick: max(0, dist_needed - dx)^2 + max(0, dist_needed - dy)^2
                # This encourages separation in either axis.
                pen_x = max(0, dist_needed - dx)
                pen_y = max(0, dist_needed - dy)
                # If both are positive, they overlap.
                # We want to minimize the product or sum of squares.
                # Using sum of squares of penetrations is smooth enough for NM.
                penalty += 100 * (pen_x**2 + pen_y**2)
                
        return -(np.sum(s)) + penalty

    def run_optimization(x0):
        try:
            res = minimize(penalty_objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            # Unpack
            cx = res.x[0::3]
            cy = res.x[1::3]
            s = res.x[2::3]
            centers = np.stack([cx, cy], axis=1)
            # Clamp sides to positive
            s = np.maximum(s, 1e-6)
            # Clamp centers to box to help validity check if penalty failed slightly
            for i in range(n):
                s_i = s[i]
                centers[i, 0] = np.clip(centers[i, 0], s_i/2, 1 - s_i/2)
                centers[i, 1] = np.clip(centers[i, 1], s_i/2, 1 - s_i/2)
            
            is_valid, score = evaluate(centers, s)
            if is_valid:
                return centers, s, score
            else:
                return None, None, 0.0
        except Exception:
            return None, None, 0.0

    # Generate initial guesses
    guesses = []
    
    # Guess 1: The 3.0 grid from original
    c1 = np.array([
        [0.25, 0.25], [0.75, 0.25],
        [0.125, 0.625], [0.375, 0.625], [0.625, 0.625], [0.875, 0.625],
        [0.125, 0.875], [0.375, 0.875], [0.625, 0.875], [0.875, 0.875]
    ])
    s1 = np.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
    x1 = np.zeros(30)
    for i in range(n):
        x1[3*i] = c1[i, 0]
        x1[3*i+1] = c1[i, 1]
        x1[3*i+2] = s1[i]
    guesses.append(x1)
    
    # Guess 2: One large, rest small
    # 1 square 0.5 at (0.25, 0.25)
    # Remaining space: Top strip (1x0.5), Right strip (0.5x0.5)
    # Fill Top strip with 4 squares of 0.25?
    # Fill Right strip (0.5x0.5) with 4 squares of 0.25?
    # Total: 1 (0.5) + 4 (0.25) + 4 (0.25) = 9 squares. Need 10.
    # Let's try:
    # 1 square 0.4 at (0.2, 0.2)
    # 1 square 0.4 at (0.6, 0.2)
    # 1 square 0.4 at (0.2, 0.6)
    # 1 square 0.4 at (0.6, 0.6)
    # Center gap 0.2x0.2? No, 0.4+0.4=0.8, gap is 0.2.
    # We have 4 squares of 0.4. Sum = 1.6.
    # Remaining 6 squares must fit in the gaps.
    # This is hard to construct manually.
    
    # Let's use a random valid generator for more guesses
    def generate_random_guess():
        # Greedy placement
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        # Place first square
        s0 = 0.4 + 0.1 * np.random.rand()
        centers[0, 0] = s0/2 + (1 - s0) * np.random.rand()
        centers[0, 1] = s0/2 + (1 - s0) * np.random.rand()
        sides[0] = s0
        
        # Place remaining squares in empty space
        # This is a simple heuristic, might fail to place all 10 if not careful
        # We'll just try random positions and sizes and let the optimizer fix it
        # This is risky for validity, but the penalty will push it.
        
        for i in range(1, n):
            s_i = 0.1 + 0.3 * np.random.rand()
            cx = s_i/2 + (1 - s_i) * np.random.rand()
            cy = s_i/2 + (1 - s_i) * np.random.rand()
            centers[i, 0] = cx
            centers[i, 1] = cy
            sides[i] = s_i
            
        x = np.zeros(30)
        for i in range(n):
            x[3*i] = centers[i, 0]
            x[3*i+1] = centers[i, 1]
            x[3*i+2] = sides[i]
        return x

    # Add 3 random guesses
    for _ in range(3):
        guesses.append(generate_random_guess())
        
    best_centers = None
    best_sides = None
    best_score = 0.0
    
    # Evaluate initial guesses
    for g in guesses:
        cx = g[0::3]
        cy = g[1::3]
        s = g[2::3]
        centers = np.stack([cx, cy], axis=1)
        is_valid, score = evaluate(centers, s)
        if is_valid and score > best_score:
            best_score = score
            best_centers = centers.copy()
            best_sides = s.copy()
            
    # Run optimization on each guess
    for g in guesses:
        if time.time() > deadline:
            break
        c, s, score = run_optimization(g)
        if c is not None and score > best_score:
            best_score = score
            best_centers = c
            best_sides = s
            
    # If we still have time, do more random restarts
    while time.time() < deadline - 10:
        g = generate_random_guess()
        c, s, score = run_optimization(g)
        if c is not None and score > best_score:
            best_score = score
            best_centers = c
            best_sides = s

    # Fallback if no valid solution found (should not happen with Guess 1)
    if best_centers is None:
        # Return the known valid 3.0 solution
        best_centers = c1
        best_sides = s1

    return best_centers, best_sides
