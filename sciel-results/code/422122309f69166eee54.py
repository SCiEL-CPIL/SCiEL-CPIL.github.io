import numpy as np
from scipy.optimize import minimize

N = 21

def construct():
        # Time management
        start_time = np.datetime64('now')
        deadline = 110  # seconds, leaving 10s safety margin
        
        best_val = -np.inf
        best_x = None
        
        # Initial configuration: 5x5 grid minus 4 corners in 1x1 square
        # w=1, h=1
        w_init = 1.0
        h_init = 1.0
        
        # Grid points
        pts = []
        for i in range(5):
            for j in range(5):
                if (i == 0 and j == 0) or (i == 4 and j == 0) or (i == 0 and j == 4) or (i == 4 and j == 4):
                    continue
                pts.append([(i+0.5)/5.0, (j+0.5)/5.0, 0.1])
        
        x0 = np.array(pts).flatten()
        # x0 structure: [x1, y1, r1, x2, y2, r2, ..., w]
        x0 = np.append(x0, w_init)
        
        def objective(v):
            # v: [x1, y1, r1, ..., xN, yN, rN, w]
            xs = v[0:N*3:3]
            ys = v[1:N*3:3]
            rs = v[2:N*3:3]
            w = v[-1]
            h = 2.0 - w
            
            # Check validity
            if w <= 0 or h <= 0:
                return 1e6
            if np.any(rs <= 0):
                return 1e6
            if np.any(xs < rs) or np.any(xs > w - rs):
                return 1e6
            if np.any(ys < rs) or np.any(ys > h - rs):
                return 1e6
            
            # Overlaps
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = rs[i] + rs[j]
                    if dist < min_dist:
                        penalty += (min_dist - dist)**2
            
            if penalty > 1e-6:
                return -np.sum(rs) + 1000 * penalty
            
            return -np.sum(rs)
        
        # Run multiple restarts
        attempts = 0
        while True:
            elapsed = (np.datetime64('now') - start_time).astype('timedelta64[s]').astype(float)
            if elapsed > deadline:
                break
            
            attempts += 1
            
            # Perturb initial state
            if attempts == 1:
                x_curr = x0
            else:
                x_curr = x0 + np.random.normal(0, 0.01, size=x0.shape)
                # Ensure w is reasonable
                x_curr[-1] = 1.0 + np.random.uniform(-0.2, 0.2)
                if x_curr[-1] < 0.5 or x_curr[-1] > 1.5:
                    x_curr[-1] = 1.0
            
            res = minimize(objective, x_curr, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-6})
            
            if res.success or res.fun < best_val:
                # Check if valid
                xs = res.x[0:N*3:3]
                ys = res.x[1:N*3:3]
                rs = res.x[2:N*3:3]
                w = res.x[-1]
                h = 2.0 - w
                
                valid = True
                if w <= 0 or h <= 0 or np.any(rs <= 0):
                    valid = False
                else:
                    if np.any(xs < rs - 1e-9) or np.any(xs > w - rs + 1e-9):
                        valid = False
                    if np.any(ys < rs - 1e-9) or np.any(ys > h - rs + 1e-9):
                        valid = False
                    if valid:
                        for i in range(N):
                            for j in range(i+1, N):
                                dx = xs[i] - xs[j]
                                dy = ys[i] - ys[j]
                                dist = np.sqrt(dx*dx + dy*dy)
                                if dist < rs[i] + rs[j] - 1e-9:
                                    valid = False
                                    break
                            if not valid: break
                
                if valid:
                    val = np.sum(rs)
                    if val > best_val:
                        best_val = val
                        best_x = res.x
        
        if best_x is None:
            # Fallback to initial valid config (shrink radii to be safe)
            xs = x0[0:N*3:3]
            ys = x0[1:N*3:3]
            rs = np.full(N, 0.09) # 0.1 is touching, 0.09 is safe
            w = 1.0
            best_x = np.concatenate([xs, ys, rs, [w]])
            # Interleave: x1, y1, r1...
            # The objective function expects [x1, y1, r1, ...]
            # Let's construct the final array carefully
            
        # Construct final output
        # best_x is [x1, y1, r1, ..., xN, yN, rN, w]
        w = best_x[-1]
        h = 2.0 - w
        
        # Safety margin: shrink radii by 1e-5 and move centers inward if needed
        # Actually, just shrinking radii is safest.
        result = np.zeros((N, 3))
        for i in range(N):
            result[i, 0] = best_x[i*3]
            result[i, 1] = best_x[i*3+1]
            result[i, 2] = best_x[i*3+2] * 0.999 # Shrink slightly to ensure disjoint interiors
            
        # Final validity check
        # If invalid, return a known valid grid
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = result[i,0] - result[j,0]
                dy = result[i,1] - result[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist <= result[i,2] + result[j,2]:
                    # Overlap! Fallback
                    result = np.array([[(i+0.5)/5.0, (j+0.5)/5.0, 0.09] for i in range(5) for j in range(5) if not ((i==0 and j==0) or (i==4 and j==0) or (i==0 and j==4) or (i==4 and j==4))])
                    return result
        
        # Check bounds
        for i in range(N):
            if result[i,0] - result[i,2] < -1e-9 or result[i,0] + result[i,2] > w + 1e-9:
                 result = np.array([[(i+0.5)/5.0, (j+0.5)/5.0, 0.09] for i in range(5) for j in range(5) if not ((i==0 and j==0) or (i==4 and j==0) or (i==0 and j==4) or (i==4 and j==4))])
                 return result
            if result[i,1] - result[i,2] < -1e-9 or result[i,1] + result[i,2] > h + 1e-9:
                 result = np.array([[(i+0.5)/5.0, (j+0.5)/5.0, 0.09] for i in range(5) for j in range(5) if not ((i==0 and j==0) or (i==4 and j==0) or (i==0 and j==4) or (i==4 and j==4))])
                 return result

        return result
