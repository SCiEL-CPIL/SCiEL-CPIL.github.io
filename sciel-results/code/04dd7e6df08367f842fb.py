import numpy as np
import time
import sys
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    # Uniform distribution as a safe fallback
    P_uniform = np.ones((n, n)) / (n * n)
    
    start_time = time.time()
    # Safety margin: evaluator kills at 630s. We stop at 600s to be safe.
    deadline = start_time + 590
    
    # Pre-compute projection indices for each slope
    xs = np.arange(n)
    ys = np.arange(n)
    X_grid, Y_grid = np.meshgrid(xs, ys, indexing='xy')
    
    proj_info = {}
    for r in [-1, 0, 1, 2]:
        vals = X_grid + r * Y_grid
        min_val = int(np.min(vals))
        max_val = int(np.max(vals))
        n_unique = max_val - min_val + 1
        indices = (vals - min_val).ravel()
        proj_info[r] = (indices, n_unique)
    
    def get_projected_entropies(P):
        """
        Compute Shannon entropies for slopes -1, 0, 1, 2.
        Returns a dictionary {slope: entropy_in_bits}
        """
        P_flat = P.ravel()
        entropies = {}
        for r in [-1, 0, 1, 2]:
            indices, n_unique = proj_info[r]
            # Pool probabilities
            counts = np.bincount(indices, weights=P_flat, minlength=n_unique)
            # Filter out zeros for log calculation
            probs = counts[counts > 1e-15]
            if len(probs) == 0:
                entropies[r] = 0.0
            else:
                probs = probs / probs.sum()
                entropies[r] = -np.sum(probs * np.log2(probs))
        return entropies

    def calculate_score(P):
        """
        Calculate the raw score D / M.
        """
        ents = get_projected_entropies(P)
        D = ents[-1]
        M = max(ents[0], ents[1], ents[2])
        if M < 1e-9:
            return 0.0
        return D / M

    # Initial best score
    best_score = calculate_score(P_uniform)
    best_P = P_uniform.copy()
    
    # Softmax mapping from unconstrained Z to probability P
    def softmax(Z_flat):
        Z = Z_flat.reshape(n, n)
        Z_max = np.max(Z)
        exp_Z = np.exp(Z - Z_max)
        P = exp_Z / np.sum(exp_Z)
        return P

    # Smooth max for M
    def smooth_max(H0, H1, H2, tau):
        arr = np.array([H0, H1, H2])
        m = np.max(arr)
        if m < 1e-15:
            return 1e-15
        scaled = arr / tau
        sm = np.max(scaled)
        return tau * (sm + np.log(np.sum(np.exp(scaled - sm))))

    def objective_smooth(Z_flat, tau):
        """
        Objective to minimize: - (ln(D) - ln(M_smooth))
        Equivalent to maximizing ln(D/M_smooth)
        """
        P = softmax(Z_flat)
        ents = get_projected_entropies(P)
        D = ents[-1]
        M_smooth = smooth_max(ents[0], ents[1], ents[2], tau)
        
        # Handle edge cases
        if D < 1e-15 or M_smooth < 1e-15:
            return 1e10  # Large penalty
        
        return -(np.log(D) - np.log(M_smooth))

    # Generate diverse initializations
    def get_initializations():
        inits = []
        
        # 1. Uniform
        inits.append(np.zeros((n, n)))
        
        # 2. Diagonal
        Z_diag = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                Z_diag[y, x] = -0.5 * abs(x - y)
        inits.append(Z_diag.flatten())
        
        # 3. Anti-diagonal
        Z_anti = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                Z_anti[y, x] = -0.5 * abs(x + y - (n - 1))
        inits.append(Z_anti.flatten())
        
        # 4. Centered Gaussian
        Z_center = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                Z_center[y, x] = -0.5 * ((x - 4.5)**2 + (y - 4.5)**2)
        inits.append(Z_center.flatten())
        
        # 5. Random sparse supports
        for _ in range(20):
            Z_rand = np.random.normal(0, 1, (n, n))
            # Make it sparser by thresholding
            Z_rand[Z_rand < np.percentile(Z_rand, 20)] = -10
            inits.append(Z_rand.flatten())
            
        # 6. Random dense
        for _ in range(10):
            inits.append(np.random.normal(0, 1, (n, n)).flatten())
            
        return inits

    inits = get_initializations()
    
    # Optimization parameters
    # We will run a few rounds with decreasing tau
    tau_schedule = [1.0, 0.5, 0.2, 0.1, 0.05]
    
    current_best_Z = None
    if best_score > 0:
        # Initialize best_Z from uniform or best found
        current_best_Z = np.zeros((n, n)).flatten()
        
    for tau in tau_schedule:
        if time.time() > deadline - 30:
            break
            
        # If we have a good current best, add it to inits for this round
        if current_best_Z is not None:
            inits.append(current_best_Z.copy())
            
        for Z_init in inits:
            if time.time() > deadline - 10:
                break
                
            try:
                # L-BFGS-B on unconstrained Z
                res = minimize(
                    lambda z: objective_smooth(z, tau),
                    Z_init,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                
                P_opt = softmax(res.x)
                score_opt = calculate_score(P_opt)
                
                if score_opt > best_score:
                    best_score = score_opt
                    best_P = P_opt.copy()
                    current_best_Z = res.x.copy()
                    
            except Exception:
                continue

    # Final Polish with very small tau from the best Z
    if time.time() < deadline - 5 and current_best_Z is not None:
        try:
            tau_final = 0.01
            res = minimize(
                lambda z: objective_smooth(z, tau_final),
                current_best_Z,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            P_opt = softmax(res.x)
            score_opt = calculate_score(P_opt)
            
            if score_opt > best_score:
                best_score = score_opt
                best_P = P_opt.copy()
        except Exception:
            pass

    # Discrete refinement: Try to zero out small probabilities and renormalize
    # This can sometimes jump to a better local optimum with simpler support
    if time.time() < deadline - 2:
        P_work = best_P.copy()
        flat = P_work.ravel()
        
        # Sort probabilities
        sorted_idx = np.argsort(flat)
        
        # Try removing the smallest 10%, 20%, etc.
        for fraction in [0.1, 0.2, 0.3]:
            num_zero = int(n * n * fraction)
            to_zero = sorted_idx[:num_zero]
            
            P_trial = P_work.copy()
            P_trial.ravel()[to_zero] = 0
            
            s = P_trial.sum()
            if s > 1e-9:
                P_trial /= s
                score_trial = calculate_score(P_trial)
                
                if score_trial > best_score:
                    best_score = score_trial
                    best_P = P_trial.copy()

    # Final validation and normalization
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s <= 1e-9:
        return P_uniform
    best_P /= s
    
    return best_P
