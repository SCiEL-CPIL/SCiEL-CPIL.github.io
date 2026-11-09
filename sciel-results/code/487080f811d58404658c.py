# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    """
    Maximize sum of side lengths of 10 axis-aligned squares in unit square.
    Uses Multi-Start Local Search with Nelder-Mead optimization.
    """
    n = 10
    tol = 1e-7
    
    def objective(params):
        # params: [s0, cx0, cy0, s1, cx1, cy1, ..., s9, cx9, cy9]
        s = params[0::3]
        cx = params[1::3]
        cy = params[2::3]
        
        # Ensure sides are positive
        if np.any(s <= 0):
            return 1e10
            
        penalty = 0.0
        
        # Boundary penalty: squares must be inside [-tol, 1+tol]^2
        # Actually, the constraint is [cx - s/2, cx + s/2] inside [-tol, 1+tol].
        # So cx - s/2 >= -tol and cx + s/2 <= 1 + tol.
        # cx >= s/2 - tol and cx <= 1 - s/2 + tol.
        
        for i in range(n):
            half = s[i] / 2.0
            # Lower bound violation
            if cx[i] - half < -tol:
                penalty += 1e5 * (abs(cx[i] - half + tol) + 1.0)
            # Upper bound violation
            if cx[i] + half > 1.0 + tol:
                penalty += 1e5 * (abs(cx[i] + half - 1.0 - tol) + 1.0)
                
            if cy[i] - half < -tol:
                penalty += 1e5 * (abs(cy[i] - half + tol) + 1.0)
            if cy[i] + half > 1.0 + tol:
                penalty += 1e5 * (abs(cy[i] + half - 1.0 - tol) + 1.0)
        
        # Overlap penalty
        # Squares i, j overlap if |cx_i - cx_j| < (s_i + s_j)/2 - tol AND |cy_i - cy_j| < (s_i + s_j)/2 - tol
        # We want to penalize any "intrusion" into the forbidden zone.
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j]) / 2.0 - tol
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                
                # If they are separated enough in x or y, no penalty.
                if dx >= t or dy >= t:
                    continue
                
                # Calculate how much they overlap in the "forbidden" sense
                # The condition for overlap is dx < t AND dy < t.
                # We can define a continuous penalty based on how far inside the box [0, t]x[0, t] the point (dx, dy) is.
                # A simple quadratic penalty for the overlap area or depth.
                # Let's use the sum of the penetrations.
                # If dx < t, penetration in x is t - dx.
                # If dy < t, penetration in y is t - dy.
                # But we only care if BOTH are violated.
                # A smooth approximation: penalty = max(0, t - dx) * max(0, t - dy) ? No, that's area.
                # Let's use: penalty = 1e5 * (max(0, t - dx) + max(0, t - dy)) if (dx < t and dy < t) else 0.
                # Actually, if dx < t and dy < t, they overlap.
                # The "depth" of overlap can be measured by how much we need to push them apart.
                # min(t - dx, t - dy) is the minimum separation needed to break the overlap in one axis.
                # Let's use a quadratic penalty on the "violation" magnitude.
                viol_x = max(0.0, t - dx)
                viol_y = max(0.0, t - dy)
                if viol_x > 0 and viol_y > 0:
                    penalty += 1e5 * (viol_x * viol_y + 0.1 * (viol_x + viol_y))
        
        # Objective: Minimize negative sum of sides + penalty
        return -np.sum(s) + penalty

    def get_initial_configs():
        configs = []
        
        # Config 1: 3x3 grid of 1/3 squares + 1 small in center? No, 10 squares.
        # Let's try a 4x3 grid? 12 squares. We need 10.
        # 4 columns, 3 rows. 12 slots. Remove 2.
        # Size 0.25. Sum = 10 * 0.25 = 2.5.
        # Let's place 10 squares of size 0.25 in a 4x3 grid area (1.0 x 0.75).
        # x coords: 0.125, 0.375, 0.625, 0.875
        # y coords: 0.125, 0.375, 0.625
        # We have 12 positions. Pick 10.
        # To maximize symmetry, maybe remove 2 from corners?
        # Or just fill row by row.
        
        s1 = np.full(n, 0.25)
        cx1 = np.zeros(n)
        cy1 = np.zeros(n)
        # Fill 4x3 grid, skip last 2
        idx = 0
        for r in range(3):
            for c in range(4):
                if idx < n:
                    cx1[idx] = 0.125 + c * 0.25
                    cy1[idx] = 0.125 + r * 0.25
                    idx += 1
        configs.append(np.concatenate([s1, cx1, cy1]))

        # Config 2: 2 large (0.4), 8 small (0.15)?
        # 2*0.4 = 0.8. 8*0.15 = 1.2. Sum = 2.0.
        # Place 2 large in bottom left and bottom right?
        # Large 1: [0, 0.4] x [0, 0.4] -> c=(0.2, 0.2)
        # Large 2: [0.6, 1.0] x [0, 0.4] -> c=(0.8, 0.2)
        # Remaining space: Top strip [0, 1] x [0.4, 1] (height 0.6).
        # And middle gap [0.4, 0.6] x [0, 0.4] (width 0.2, height 0.4).
        # 8 small squares of size 0.15.
        # In top strip (1.0 x 0.6):
        # Can fit 4 in a row (0.15*4 = 0.6 < 1.0).
        # Row 1 (y=0.475): x=0.075, 0.225, 0.375, 0.525
        # Row 2 (y=0.625): x=0.075, 0.225, 0.375, 0.525
        # That's 8 squares.
        # Check if they fit in y range [0.4, 1.0].
        # y=0.475: [0.4, 0.55]. OK.
        # y=0.625: [0.55, 0.7]. OK.
        # x range: 0.075-0.075=0. OK. 0.525+0.075=0.6. OK.
        # Wait, the large squares occupy y in [0, 0.4].
        # So top strip is free.
        
        s2 = np.zeros(n)
        cx2 = np.zeros(n)
        cy2 = np.zeros(n)
        
        s2[0] = 0.4; cx2[0] = 0.2; cy2[0] = 0.2
        s2[1] = 0.4; cx2[1] = 0.8; cy2[1] = 0.2
        
        for k in range(8):
            r = k // 4
            c = k % 4
            s2[2+k] = 0.15
            cx2[2+k] = 0.075 + c * 0.15
            cy2[2+k] = 0.475 + r * 0.15
            
        configs.append(np.concatenate([s2, cx2, cy2]))

        # Config 3: 1 large (0.5), 4 medium (0.25), 5 small (0.15) - similar to seed but different layout
        # Large: [0, 0.5] x [0, 0.5]
        # Meds: [0.5, 1.0] x [0.5, 1.0] -> 4 squares of 0.25 in 0.5x0.5 area.
        # Small: [0.5, 1.0] x [0, 0.5] -> 5 squares of 0.15.
        # This is basically the seed's greedy init. Let's include it as a start point.
        
        s3 = np.zeros(n)
        cx3 = np.zeros(n)
        cy3 = np.zeros(n)
        
        s3[0] = 0.5; cx3[0] = 0.25; cy3[0] = 0.25
        
        # 4 Meds in top right
        s3[1] = 0.25; cx3[1] = 0.625; cy3[1] = 0.625
        s3[2] = 0.25; cx3[2] = 0.875; cy3[2] = 0.625
        s3[3] = 0.25; cx3[3] = 0.625; cy3[3] = 0.875
        s3[4] = 0.25; cx3[4] = 0.875; cy3[4] = 0.875
        
        # 5 Smalls in bottom right
        s3[5] = 0.15; cx3[5] = 0.575; cy3[5] = 0.075
        s3[6] = 0.15; cx3[6] = 0.725; cy3[6] = 0.075
        s3[7] = 0.15; cx3[7] = 0.875; cy3[7] = 0.075
        s3[8] = 0.15; cx3[8] = 0.575; cy3[8] = 0.225
        s3[9] = 0.15; cx3[9] = 0.725; cy3[9] = 0.225
        
        configs.append(np.concatenate([s3, cx3, cy3]))

        # Config 4: Randomized valid-ish start
        # 10 squares of size 0.3? Area 0.9.
        # Try to pack 10 squares of size 0.3.
        # 3x3 grid of 0.3 takes 0.9x0.9.
        # 10th square?
        # Let's just use a random perturbation of Config 1.
        np.random.seed(123)
        base = configs[0].copy()
        # Perturb centers slightly
        for i in range(n):
            base[i*3+1] += np.random.uniform(-0.05, 0.05)
            base[i*3+2] += np.random.uniform(-0.05, 0.05)
            base[i*3] += np.random.uniform(-0.05, 0.05)
            # Clip to valid range roughly
            s_val = base[i*3]
            if s_val > 0:
                half = s_val/2
                base[i*3+1] = np.clip(base[i*3+1], half, 1-half)
                base[i*3+2] = np.clip(base[i*3+2], half, 1-half)
        configs.append(base)

        return configs

    best_result = None
    best_score = -np.inf

    configs = get_initial_configs()
    
    for i, x0 in enumerate(configs):
        # Ensure x0 is valid enough to start
        # Nelder-Mead options
        options = {
            'maxiter': 5000,
            'maxfev': 5000,
            'xatol': 1e-6,
            'fatol': 1e-6,
            'disp': False
        }
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options=options)
            score = -res.fun # res.fun is -sum(s) + penalty. If penalty is 0, score is sum(s).
            # We need to check if the result is valid (penalty ~ 0)
            # If penalty is high, the score is misleading.
            # Let's re-evaluate the objective to check penalty.
            # objective returns -sum(s) + penalty.
            # So penalty = res.fun + sum(s).
            # We want penalty to be small.
            
            s_res = res.x[0::3]
            cx_res = res.x[1::3]
            cy_res = res.x[2::3]
            
            # Check validity manually
            valid = True
            for k in range(n):
                if s_res[k] <= 0:
                    valid = False
                    break
                half = s_res[k] / 2.0
                if cx_res[k] - half < -tol or cx_res[k] + half > 1.0 + tol:
                    valid = False
                    break
                if cy_res[k] - half < -tol or cy_res[k] + half > 1.0 + tol:
                    valid = False
                    break
                    
            if valid:
                # Check overlaps
                for a in range(n):
                    for b in range(a+1, n):
                        t = (s_res[a] + s_res[b]) / 2.0 - tol
                        dx = abs(cx_res[a] - cx_res[b])
                        dy = abs(cy_res[a] - cy_res[b])
                        if dx < t and dy < t:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                sum_s = np.sum(s_res)
                if sum_s > best_score:
                    best_score = sum_s
                    best_result = res.x.copy()
                    
        except Exception as e:
            pass

    if best_result is None:
        # Fallback to a simple valid configuration
        # 10 squares of 0.25 in a grid
        s = np.full(n, 0.25)
        cx = np.zeros(n)
        cy = np.zeros(n)
        idx = 0
        for r in range(3):
            for c in range(4):
                if idx < n:
                    cx[idx] = 0.125 + c * 0.25
                    cy[idx] = 0.125 + r * 0.25
                    idx += 1
        best_result = np.concatenate([s, cx, cy])

    s = best_result[0::3]
    cx = best_result[1::3]
    cy = best_result[2::3]
    
    # Final cleanup: clip to ensure strict validity if floating point errors exist
    for i in range(n):
        s[i] = max(s[i], 1e-4)
        half = s[i] / 2.0
        # Strictly inside [0, 1] to be safe from tol issues, though tol allows [-tol, 1+tol]
        # Clipping to [half, 1-half] ensures it's inside [0,1]
        cx[i] = np.clip(cx[i], half, 1.0 - half)
        cy[i] = np.clip(cy[i], half, 1.0 - half)
        
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = s.astype(np.float64)
    
    return centers, sides

# EVOLVE-BLOCK-END
