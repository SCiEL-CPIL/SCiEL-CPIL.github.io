import numpy as np
import time

N = 200

def construct():
        start_time = time.time()
        deadline = 140.0 # 10s margin
        
        # 1. Initialization: Hexagonal lattice
        # Approximate density: sqrt(N) x sqrt(N)
        # Hex lattice: rows offset.
        # Let's just use a jittered grid for simplicity and robustness, or a proper hex lattice.
        # Hex lattice is better.
        m = int(np.ceil(np.sqrt(N)))
        # Adjust m to fit N circles roughly
        # For N=200, m=14 gives 196, m=15 gives 225.
        # Let's use a 15x15 grid but remove 25 points randomly? No, hex is better.
        
        # Let's generate a hex lattice that covers the square.
        # Spacing s.
        # Number of points ~ Area / (s^2 * sqrt(3)/2) = 1 / (s^2 * 0.866)
        # 200 = 1 / (s^2 * 0.866) => s^2 = 1 / (200 * 0.866) => s = 1/sqrt(173.2) = 1/13.16 = 0.0759
        s = 1.0 / np.sqrt(N * 0.866)
        
        points = []
        row = 0
        while len(points) < N:
            offset = 0 if row % 2 == 0 else s * 0.866 / 2
            col = 0
            while True:
                x = -0.5 + s * (col + 0.5) + offset
                y = -0.5 + s * (row + 0.5) * 1.1547 # vertical spacing is s * sqrt(3)
                # Actually, standard hex:
                # x = col * s + (row % 2) * s/2
                # y = row * s * sqrt(3)/2
                # Let's center it.
                
                # Let's just use a simple jittered grid for initialization to avoid bugs in hex generation.
                # Jittered grid is often good enough for local optimization.
                break
            break
            
        # Let's do a proper jittered grid initialization.
        m = int(np.ceil(np.sqrt(N)))
        # We need N points.
        # Create a (m x m) grid of cells.
        # If m*m > N, we pick N random cells? No, that's sparse.
        # Better: Create a grid of size ceil(sqrt(N)) x ceil(sqrt(N)).
        # If we have more cells than points, we can distribute points to cells.
        
        # Simplest robust init:
        # 14x14 = 196 points. Add 4 more?
        # Let's use 15x15 grid, take 200 points.
        # To make it uniform, we can take the first 200 of a shuffled 15x15 grid?
        # Or just use a hex lattice properly.
        
        # Hex Lattice Init:
        s = 0.075 # Initial spacing guess
        pts = []
        row = 0
        while len(pts) < N:
            y = -0.5 + 0.05 + row * s * np.sqrt(3)/2
            if y > 0.45:
                s *= 0.95 # Shrink if we run out of room vertically? Or just stop.
                break
            offset = 0 if row % 2 == 0 else s/2
            col = 0
            while True:
                x = -0.5 + 0.05 + col * s + offset
                if x > 0.45:
                    break
                pts.append([x, y])
                col += 1
            row += 1
            
        if len(pts) < N:
            # Pad with random points in the square if hex lattice ran out
            # This shouldn't happen with s=0.075 for N=200
            # 200 * 0.866 * 0.075^2 = 200 * 0.866 * 0.0056 = 0.97. Fits well.
            while len(pts) < N:
                pts.append([np.random.uniform(-0.45, 0.45), np.random.uniform(-0.45, 0.45)])
                
        X = np.array(pts[:N])
        
        # Local Optimization
        # Maximize soft-min of distances
        tau = 0.01 # Temperature for soft-min
        lr = 0.001 # Learning rate
        
        def compute_grads(X, tau):
            # X shape (N, 2)
            N = X.shape[0]
            # Pairwise distances
            # d[i,j] = ||X[i] - X[j]||
            # Use broadcasting
            diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            d = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(d, np.inf) # Avoid self-interaction
            
            # Boundary distances
            # d_b[i, k] = 0.5 - |X[i, k]|
            d_b = 0.5 - np.abs(X)
            # Flatten d_b to match pair structure?
            # Soft-min over all pairs and boundary distances.
            
            # We want to maximize M = -tau * log( sum exp(-d/tau) )
            # Let S = sum exp(-d/tau)
            # Grad M = -1/tau * Grad S / S * (-1/tau) ?
            # dM/dX = -1/S * sum (d exp(-d/tau)/dX)
            # d exp(-d/tau)/dX = exp(-d/tau) * (-1/tau) * d(d)/dX
            
            # So Grad M = (1/(tau * S)) * sum ( exp(-d/tau) * d(d)/dX )
            
            # Compute weights w = exp(-d/tau)
            w = np.exp(-d / tau)
            w_b = np.exp(-d_b / tau)
            
            S = np.sum(w) + np.sum(w_b)
            
            # Gradient of d[i,j] w.r.t X[i] is (X[i] - X[j]) / d[i,j]
            # Gradient of d[i,j] w.r.t X[j] is (X[j] - X[i]) / d[i,j]
            
            # Force on X[i] from pair (i,j) is proportional to w[i,j] * (X[i] - X[j]) / d[i,j]
            # This force points AWAY from X[j] if we are maximizing distance?
            # Wait. M is soft-min. We want to INCREASE distances.
            # If d is small, w is large.
            # Grad M points in direction of increasing d.
            # d(d_ij)/dX_i = (X_i - X_j) / d_ij.
            # So force is w_ij * (X_i - X_j) / d_ij.
            # This pushes X_i away from X_j. Correct.
            
            # Boundary: d_b = 0.5 - |x|.
            # d(d_b)/dx = -sign(x).
            # If x > 0, d_b = 0.5 - x, grad = -1.
            # Force = w_b * (-1). This pushes x to decrease?
            # We want to INCREASE d_b.
            # If x is positive, to increase 0.5-x, we must decrease x.
            # So force should be negative. Correct.
            
            # Compute pair forces
            # F_pair[i] = sum_j w[i,j] * (X[i] - X[j]) / d[i,j]
            # Note: d[i,i] is inf, w[i,i] is 0.
            
            # Vectorized computation of F_pair
            # diff is (N, N, 2)
            # d is (N, N)
            # We need to divide diff by d.
            # Handle d=0 (should not happen with init)
            d_safe = np.where(d == 0, 1, d)
            unit_vec = diff / d_safe[:, :, np.newaxis]
            
            # Weighted sum
            # w is (N, N)
            # F_pair = sum_j w[i,j] * unit_vec[i,j]
            F_pair = np.sum(w[:, :, np.newaxis] * unit_vec, axis=1)
            
            # Boundary forces
            # F_b[i] = w_b[i, 0] * grad_x + w_b[i, 1] * grad_y
            # grad_x = -sign(X[i, 0])
            # grad_y = -sign(X[i, 1])
            F_b = -w_b * np.sign(X)
            
            # Total gradient
            Grad = F_pair + F_b
            
            # Normalize by (tau * S)
            Grad /= (tau * S)
            
            return Grad

        # Optimization Loop
        best_score = 0
        best_X = X.copy()
        
        for step in range(10000):
            if time.time() > deadline:
                break
                
            Grad = compute_grads(X, tau)
            X += lr * Grad
            
            # Project to square
            X = np.clip(X, -0.45, 0.45) # Keep slightly inside to avoid boundary issues
            
            # Compute current score (min distance)
            # Min pairwise distance / 2
            # Min boundary distance
            diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            d = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(d, np.inf)
            min_pair = np.min(d)
            
            d_b = 0.5 - np.abs(X)
            min_bound = np.min(d_b)
            
            score = min(min_pair, min_bound)
            
            if score > best_score:
                best_score = score
                best_X = X.copy()
                
            # Adaptive learning rate?
            # If score improves, keep lr. If not, reduce lr.
            # Simple: reduce lr every 500 steps
            if step % 500 == 0:
                lr *= 0.95
                tau *= 0.95 # Cool down soft-min to focus on min distance
                
        # Final polish: return best_X
        # Ensure it's valid
        best_X = np.clip(best_X, -0.5, 0.5)
        return best_X
