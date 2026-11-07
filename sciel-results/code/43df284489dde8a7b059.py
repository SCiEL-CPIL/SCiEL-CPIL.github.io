"""DFTR optimizer with local quadratic surrogates for BBOB f18 (Schaffers F7)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-Free Trust-Region optimizer using local quadratic models."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        # Safety margin: 330s kill limit, leave 20s buffer
        deadline = t0 + 310.0

        # State
        x0 = np.zeros(dim)
        f0 = np.inf
        best_x = x0.copy()
        best_f = np.inf
        calls = 0

        def safe_call(x):
            nonlocal calls, best_x, best_f, f0
            if calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                fx = np.inf
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # 1. Initial Exploration: Find a decent starting point
        # Use a small random search to avoid starting at a bad spot
        # Schaffers F7 has a shifted minimum.
        # We will do 20 random evaluations to find a local basin.
        # But to be efficient, we'll just start at 0 and let DFTR handle it,
        # or do a quick spiral.
        # Given the budget is 500, and DFTR is fast per iteration,
        # we can afford a better start.
        
        # Quick random initialization
        for _ in range(10):
            x_rand = np.random.uniform(-5, 5, dim)
            f_rand = safe_call(x_rand)
            if f_rand < f0:
                f0 = f_rand
                x0 = x_rand.copy()

        # 2. DFTR Main Loop
        # We maintain a set of points (X) and function values (F) around the current center x0.
        # We fit a quadratic model: f(x) ~ c0 + c1^T (x-x0) + (x-x0)^T Q (x-x0)
        # We find the minimizer of the model within a trust region radius r.
        # If the reduction is sufficient, we accept and increase r. Else decrease.
        
        # Initialize points around x0
        # We need at least dim*(dim+3)/2 + 1 points for a full quadratic fit.
        # For dim=5: 5*8/2 + 1 = 21 points.
        # We will use a slightly larger set for robustness, or update incrementally.
        
        # Let's use a fixed set of directions for the local model.
        # Standard basis + diagonals + random?
        # To fit a full quadratic, we need points that span the space.
        # A common choice is the "coordinate star" plus some extra points.
        
        # However, Schaffers F7 is ill-conditioned.
        # We will use an adaptive trust region.
        
        # Initial trust region radius
        r = 0.5 
        # Initial step size for points in the star
        # We will generate points on the boundary of the trust region?
        # Or inside?
        # Standard DFTR: Generate points at distance r from center in specific directions.
        
        # Directions: e_i, -e_i, and maybe (e_i + e_j)/sqrt(2) etc?
        # For dim=5, we have 5 axes.
        # Let's use the 2*dim coordinate points first.
        # That's 10 points.
        # Plus the center. Total 11.
        # This only fits a linear model + diagonal quadratic.
        # To fit full Q, we need more.
        # Let's add diagonal points: (e_i + e_j)/sqrt(2) for i<j.
        # There are 5*4/2 = 10 such pairs.
        # Total points = 1 (center) + 10 (axes) + 10 (diagonals) = 21.
        # This is exactly enough for a full quadratic fit in 5D.
        
        # Generate directions
        dirs = []
        # Axes
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = 1.0
            dirs.append(e)
            dirs.append(-e)
        # Diagonals
        for i in range(dim):
            for j in range(i+1, dim):
                e = np.zeros(dim)
                e[i] = 1.0 / np.sqrt(2)
                e[j] = 1.0 / np.sqrt(2)
                dirs.append(e)
                dirs.append(-e)
        
        # dirs has 20 vectors.
        
        # Store history of points and values for the current model
        # We will re-evaluate the center if it changes.
        # To save budget, we keep a cache of evaluated points?
        # Since the function is deterministic, we can cache.
        # But memory is fine, and 500 calls is small.
        # We'll just re-evaluate if the center moves far enough.
        
        # Actually, a simpler robust DFTR:
        # 1. Evaluate center x0.
        # 2. Evaluate 2*dim points at x0 +/- r*e_i.
        # 3. Fit linear + diagonal quadratic model.
        # 4. Find minimizer of this model.
        # 5. If new point is better, move center to new point, reduce r if stuck, increase if success.
        # 6. If linear model is insufficient, add diagonal points to fit full Q.
        
        # Let's implement the full quadratic fit approach as it handles ill-conditioning better.
        
        # Current center
        x_c = x0.copy()
        f_c = f0
        
        # Evaluate the star points around x_c
        # We need to evaluate x_c + r*d for all d in dirs.
        # And x_c itself.
        
        # We will maintain a list of (x, f) for the current model.
        # When center moves, we discard old points and evaluate new ones.
        
        # To save calls, we can reuse points if the center hasn't moved much?
        # But r might change.
        # Let's keep it simple: Re-evaluate the star at each iteration.
        # Cost per iteration: 1 (center) + 20 (star) = 21 calls.
        # Budget 500 / 21 = ~23 iterations.
        # This is a bit low.
        # Can we do better?
        # Yes, we can use a smaller set of points if we assume diagonal Q initially?
        # Or we can update the model incrementally.
        
        # Alternative: Use only the 2*dim coordinate points (10 points) + center.
        # Fit: f(x) ~ f_c + g^T (x-x_c) + (x-x_c)^T D (x-x_c)
        # This is 1 + 5 + 5 = 11 parameters.
        # We have 11 points. Perfect fit.
        # This assumes Q is diagonal.
        # For ill-conditioned Schaffers, the Hessian might not be diagonal in the global frame.
        # But the rotation is fixed for an instance.
        # If we rotate our coordinate system to align with principal axes, diagonal assumption holds.
        # But we don't know the axes.
        
        # Let's stick to the full quadratic fit but optimize the number of calls.
        # Maybe we don't need to re-evaluate the center if it's the same?
        # Yes, if x_c is the same, f_c is known.
        
        # Let's try to be more efficient.
        # Iteration 1:
        # Eval center (1)
        # Eval 20 star points (20)
        # Fit Q, g, c.
        # Solve for minimizer x_new.
        # Eval x_new (1).
        # Total 22 calls.
        
        # If x_new is better:
        # x_c = x_new
        # f_c = f_new
        # Now we need to evaluate the star around the NEW center.
        # That's 20 more calls.
        # So each successful step costs 21 calls.
        # 500 / 21 = 23 steps.
        
        # This seems okay for 5D.
        # Let's implement this.
        
        # We will store the last evaluated star points to avoid re-evaluating if the center doesn't change?
        # But the center usually changes.
        
        # One optimization: If the model predicts a step in a direction d, and we already have a point in that direction from the star, we can use it.
        # But the star is fixed relative to the center.
        
        # Let's just run the loop.
        
        while calls < budget and time.time() < deadline:
            # 1. Evaluate Star
            # We need points: x_c, and x_c + r*d for d in dirs.
            # We already have x_c and f_c.
            
            X = [x_c]
            F = [f_c]
            
            for d in dirs:
                x_pt = x_c + r * d
                # Clip to box?
                # If x_pt is out of box, the function clips it.
                # But the model might be wrong near boundaries.
                # For now, let func handle clipping.
                f_pt = safe_call(x_pt)
                X.append(x_pt)
                F.append(f_pt)
                
                if calls >= budget or time.time() > deadline:
                    break
            
            if calls >= budget or time.time() > deadline:
                break
            
            # 2. Fit Quadratic Model
            # f(x) = c0 + c1^T (x-x_c) + (x-x_c)^T Q (x-x_c)
            # Let u = x - x_c
            # f(x_c + u) = f_c + g^T u + u^T Q u
            # We have 21 points.
            # Design matrix A:
            # Row i: [1, u_i^T, (u_i u_i^T)^T]
            # Target: F_i - f_c
            
            U = np.array([x - x_c for x in X])
            Y = np.array(F) - f_c
            
            # Build A
            # dim = 5
            # u has 5 components.
            # u^T Q u has 15 unique components (symmetric 5x5).
            # We can vectorize the symmetric part.
            # Indices for unique elements of 5x5 matrix:
            # (0,0), (0,1), (0,2), (0,3), (0,4)
            # (1,1), (1,2), (1,3), (1,4)
            # (2,2), (2,3), (2,4)
            # (3,3), (3,4)
            # (4,4)
            # Total 15.
            
            n_pts = len(X)
            A = np.zeros((n_pts, 1 + dim + dim*(dim+1)//2))
            
            for i in range(n_pts):
                u = U[i]
                A[i, 0] = 1.0
                A[i, 1:1+dim] = u
                # Fill quadratic terms
                idx = 1 + dim
                for r_idx in range(dim):
                    for c_idx in range(r_idx, dim):
                        A[i, idx] = u[r_idx] * u[c_idx]
                        idx += 1
            
            # Solve least squares
            # A * theta = Y
            try:
                theta, residuals, rank, s = np.linalg.lstsq(A, Y, rcond=None)
            except np.linalg.LinAlgError:
                # If singular, fallback to simple gradient step?
                # Or break.
                break
            
            # Extract parameters
            c0 = theta[0] # Should be ~0
            g = theta[1:1+dim]
            
            # Reconstruct Q
            Q_flat = theta[1+dim:]
            Q = np.zeros((dim, dim))
            idx = 0
            for r_idx in range(dim):
                for c_idx in range(r_idx, dim):
                    val = Q_flat[idx]
                    Q[r_idx, c_idx] = val
                    Q[c_idx, r_idx] = val
                    idx += 1
            
            # 3. Find Minimizer of Model within Trust Region
            # Model: f(u) = c0 + g^T u + u^T Q u
            # Gradient: g + 2 Q u = 0  => u* = -0.5 Q^-1 g
            # Hessian: 2 Q
            
            # If Q is singular or ill-conditioned, use pseudo-inverse or regularize.
            # Add small regularization to Q for stability
            Q_reg = Q + 1e-6 * np.eye(dim)
            
            try:
                u_opt = -0.5 * np.linalg.solve(Q_reg, g)
            except np.linalg.LinAlgError:
                u_opt = -g * 0.1 # Fallback: gradient descent step
            
            # Check if u_opt is within trust region
            norm_u = np.linalg.norm(u_opt)
            if norm_u > r:
                # Project to boundary
                u_opt = u_opt * (r / norm_u)
            
            x_new = x_c + u_opt
            
            # 4. Evaluate x_new
            f_new = safe_call(x_new)
            
            if calls >= budget or time.time() > deadline:
                break
            
            # 5. Accept/Reject
            # Predicted reduction: f_c - f_model(x_new)
            # Actual reduction: f_c - f_new
            
            f_pred = c0 + g @ u_opt + u_opt @ Q @ u_opt
            pred_red = f_c - f_pred
            act_red = f_c - f_new
            
            # If actual reduction is good relative to predicted, accept.
            # rho = act_red / pred_red
            if pred_red > 1e-8:
                rho = act_red / pred_red
            else:
                rho = 1.0 if act_red > 0 else 0.0
            
            if rho > 0.25: # Accept
                x_c = x_new.copy()
                f_c = f_new
                # Increase trust region
                r = min(r * 1.5, 5.0) # Cap r at 5 to stay in box roughly
            else:
                # Reject
                # Decrease trust region
                r = max(r * 0.5, 1e-4)
                
                # If r is very small, we are stuck.
                # Try a random jump to escape local minimum?
                if r < 1e-3:
                    # Random restart
                    x_jump = x_c + np.random.normal(0, 1.0, dim)
                    x_jump = np.clip(x_jump, -5.0, 5.0)
                    f_jump = safe_call(x_jump)
                    if f_jump < f_c:
                        x_c = x_jump
                        f_c = f_jump
                        r = 0.5 # Reset r
                    else:
                        # If still stuck, try a larger jump
                        x_jump2 = x_c + np.random.normal(0, 2.0, dim)
                        x_jump2 = np.clip(x_jump2, -5.0, 5.0)
                        f_jump2 = safe_call(x_jump2)
                        if f_jump2 < f_c:
                            x_c = x_jump2
                            f_c = f_jump2
                            r = 1.0
                        else:
                            # Give up on this basin?
                            # Just keep shrinking r or break?
                            # Let's break to avoid wasting budget.
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
