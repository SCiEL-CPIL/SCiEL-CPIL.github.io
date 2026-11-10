"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def construct():
    """
    Optimize 100 sphere centres in [-0.5, 0.5]^3 to maximize the minimum
    distance (which equals 2*r). We use a smooth-max surrogate for the
    objective and gradient-based optimization (L-BFGS-B) with multiple
    restarts and random perturbations to escape local minima.
    
    Objective: maximize f(X) = smooth_min_{i<j} ||x_i - x_j||, ||x_i - wall||
    We minimize -f(X) instead.
    
    For the smooth min, we use a soft-min:
    soft_min(a, b) = -logsumexp(-a, -b) approximately, but for vectorized
    computation we use: soft_min = - (1/tau) * log(sum(exp(-val/tau)))
    
    Actually, a simpler and more robust approach: 
    We want to maximize the minimum of many distances. 
    Let d_k be the list of all pair distances and wall distances.
    We maximize soft_min(d) = - (1/tau) * log(sum(exp(-d_k / tau)))
    As tau -> 0, this approaches the true min.
    
    We use a decreasing tau schedule or a fixed small tau.
    
    For efficiency, we compute all pair distances in O(N^2) vectorized form.
    """
    t_start = time.time()
    deadline = t_start + 110.0  # leave 10s safety margin before 120s kill
    
    def soft_min_obj(X_flat, tau=0.01):
        """
        Compute negative soft-min of all inter-sphere distances and wall distances.
        X_flat: shape (300,) flat array of 100 points in 3D.
        Returns: scalar (negative soft-min) and its gradient.
        """
        X = X_flat.reshape(N, 3)
        
        # Wall distances: min over 6 faces
        # Distance to face x=0.5: 0.5 - x
        # Distance to face x=-0.5: 0.5 + x
        # Similarly for y, z
        wall_dists = np.concatenate([
            0.5 - X[:, 0],
            0.5 + X[:, 0],
            0.5 - X[:, 1],
            0.5 + X[:, 1],
            0.5 - X[:, 2],
            0.5 + X[:, 2],
        ])  # shape (600,)
        
        # Pair distances: for i < j
        # Compute all pairwise differences
        # X[:, np.newaxis, :] - X[np.newaxis, :, :] gives (N, N, 3)
        # But we only need i < j
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, 3)
        dists = np.sqrt(np.sum(diff ** 2, axis=2))  # (N, N)
        
        # Get upper triangle (i < j)
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]  # shape (N*(N-1)/2,)
        
        # Combine all distances
        all_dists = np.concatenate([wall_dists, pair_dists])
        
        # Soft min: - (1/tau) * log(sum(exp(-d/tau)))
        # For numerical stability, use log-sum-exp trick
        # soft_min = - (1/tau) * (max_neg + log(sum(exp((d - max_neg)/tau))))
        # where max_neg = max(-d) = -min(d)
        neg_d = -all_dists
        max_neg = np.max(neg_d)
        log_sum = max_neg + np.log(np.sum(np.exp((neg_d - max_neg) / tau)))
        soft_min_val = -log_sum / tau
        
        # Gradient of -soft_min w.r.t. X
        # d(soft_min)/d(d_k) = exp(-d_k/tau) / sum(exp(-d_j/tau))
        # Note: soft_min = - (1/tau) * log(sum(exp(-d_j/tau)))
        # d(soft_min)/d(d_k) = - (1/tau) * (1/sum(exp(-d_j/tau))) * exp(-d_k/tau) * (-1)
        #                      = (1/tau) * exp(-d_k/tau) / sum(exp(-d_j/tau))
        #                      = exp(-d_k/tau) / sum(exp(-d_j/tau))  [the 1/tau cancels with the -1 from derivative of -d_k/tau]
        # Wait, let me redo:
        # f = - (1/tau) * log(S) where S = sum_j exp(-d_j/tau)
        # df/d(d_k) = - (1/tau) * (1/S) * dS/d(d_k)
        # dS/d(d_k) = exp(-d_k/tau) * (-1/tau)
        # df/d(d_k) = - (1/tau) * (1/S) * exp(-d_k/tau) * (-1/tau)
        #            = (1/tau^2) * exp(-d_k/tau) / S
        # Hmm, that doesn't look right dimensionally. Let me use a cleaner formulation.
        
        # Alternative: use the fact that for soft-min, the gradient w.r.t. each d_k is:
        # p_k = exp(-d_k/tau) / sum_j exp(-d_j/tau)
        # and d(soft_min)/d(d_k) = p_k  (this is a known result for the log-sum-exp based soft-min)
        # Let me verify: soft_min = - (1/tau) log(sum exp(-d_j/tau))
        # d(soft_min)/d(d_k) = - (1/tau) * [1/sum exp(-d_j/tau)] * exp(-d_k/tau) * (-1/tau)
        #                     = (1/tau^2) * exp(-d_k/tau) / sum exp(-d_j/tau)
        # That has 1/tau^2 which is problematic. Let me use a different parameterization.
        
        # Actually, the standard soft-min is: soft_min(x) = - (1/tau) log(sum exp(-x_i / tau))
        # Its derivative w.r.t. x_k is: exp(-x_k/tau) / sum_i exp(-x_i/tau)
        # Let me recompute:
        # f = - (1/tau) * log(S), S = sum_i exp(-x_i/tau)
        # df/dx_k = - (1/tau) * (1/S) * dS/dx_k
        # dS/dx_k = exp(-x_k/tau) * (-1/tau)
        # df/dx_k = - (1/tau) * (1/S) * exp(-x_k/tau) * (-1/tau) = (1/tau^2) * exp(-x_k/tau)/S
        # Hmm, so it IS 1/tau^2. That means the gradient magnitude scales as 1/tau^2.
        # This is numerically fine as long as we handle it correctly.
        
        # Let's just compute the weights:
        weights = np.exp(neg_d / tau)  # exp(-d_k/tau)
        weight_sum = np.sum(weights)
        p = weights / weight_sum  # shape (len(all_dists),)
        
        # Now, gradient of soft_min w.r.t. each d_k is p_k * (1/tau) ... wait no.
        # From above: df/d(d_k) = (1/tau^2) * exp(-d_k/tau)/S = (1/tau) * p_k
        # So d(soft_min)/d(d_k) = p_k / tau
        
        # But we want to MINIMIZE -soft_min, so the objective is g = -soft_min
        # dg/d(d_k) = -p_k / tau
        
        # Now we need dg/d(X). 
        # d_k are functions of X.
        
        # For wall distances:
        # d_0 = 0.5 - X[0,0], d_1 = 0.5 + X[0,0], ...
        # Let's build the gradient.
        
        grad_X = np.zeros((N, 3))
        
        # Wall distances: first 600 entries in all_dists
        # Indices 0..599 correspond to wall distances
        wall_p = p[:600]  # shape (600,)
        
        # d_0 = 0.5 - X[0,0] -> d(d_0)/d(X[0,0]) = -1
        # d_1 = 0.5 + X[0,0] -> d(d_1)/d(X[0,0]) = +1
        # d_2 = 0.5 - X[1,0] -> d(d_2)/d(X[1,0]) = -1
        # d_3 = 0.5 + X[1,0] -> d(d_3)/d(X[1,0]) = +1
        # ... pattern: for sphere i, face x=-: index 6*i, face x=+: index 6*i+1
        #             face y=-: index 6*i+2, face y=+: index 6*i+3
        #             face z=-: index 6*i+4, face z=+: index 6*i+5
        
        for i in range(N):
            base = 6 * i
            # x component
            grad_X[i, 0] += -(wall_p[base] / tau) * (-1) + (wall_p[base + 1] / tau) * (1)
            # Wait: dg/d(X[i,0]) = sum_k dg/d(d_k) * d(d_k)/d(X[i,0])
            # dg/d(d_k) = -p_k / tau
            # d(d_0)/d(X[i,0]) = -1, d(d_1)/d(X[i,0]) = +1
            # So contribution = (-p_0/tau)*(-1) + (-p_1/tau)*(1) = p_0/tau - p_1/tau
            grad_X[i, 0] = (wall_p[base] - wall_p[base + 1]) / tau
            
            # y component
            grad_X[i, 1] = (wall_p[base + 2] - wall_p[base + 3]) / tau
            
            # z component
            grad_X[i, 2] = (wall_p[base + 4] - wall_p[base + 5]) / tau
        
        # Pair distances: indices 600..600+M-1 where M = N*(N-1)/2
        pair_p = p[600:]  # shape (M,)
        
        # For each pair (i, j) with i < j, d = ||X_i - X_j||
        # d(d)/d(X_i) = (X_i - X_j) / d
        # d(d)/d(X_j) = (X_j - X_i) / d = -(X_i - X_j) / d
        # Contribution to grad_X[i] += (-p_k/tau) * (X_i - X_j)/d
        # Contribution to grad_X[j] += (-p_k/tau) * (X_j - X_i)/d
        
        # We need to iterate over pairs. For efficiency, we can vectorize.
        # But N=100, M=4950, so a loop is acceptable but let's try to be smart.
        
        # Actually, let's just loop over pairs for clarity and correctness.
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                d = dists[i, j]
                if d < 1e-12:
                    d = 1e-12
                diff_ij = X[i] - X[j]  # shape (3,)
                unit = diff_ij / d
                pk = pair_p[idx] / tau
                grad_X[i] += -pk * unit
                grad_X[j] += pk * unit
                idx += 1
        
        # Objective value: -soft_min
        obj_val = -soft_min_val
        
        return obj_val, grad_X.flatten()
    
    # Initial solution: simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    X0 = grid[:N].copy()
    
    best_X = X0.copy()
    best_score = -np.inf
    
    def compute_true_score(X):
        """Compute the actual minimum distance (2*r) for a configuration."""
        X = X.reshape(N, 3)
        # Clip to cube
        X_clipped = np.clip(X, -0.5, 0.5)
        
        # Wall distances
        wall_dists = np.concatenate([
            0.5 - X_clipped[:, 0],
            0.5 + X_clipped[:, 0],
            0.5 - X_clipped[:, 1],
            0.5 + X_clipped[:, 1],
            0.5 - X_clipped[:, 2],
            0.5 + X_clipped[:, 2],
        ])
        min_wall = np.min(wall_dists)
        
        # Pair distances
        diff = X_clipped[:, np.newaxis, :] - X_clipped[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        iu = np.triu_indices(N, k=1)
        min_pair = np.min(dists[iu])
        
        return min(min_wall, min_pair)
    
    # Evaluate initial
    best_score = compute_true_score(best_X)
    
    # Try multiple restarts with different initializations
    rng = np.random.default_rng(42)
    
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            X_init = X0.copy()
        else:
            # Perturb the best solution found so far
            X_init = best_X.copy()
            X_init += rng.normal(0, 0.02, size=(N, 3))
            X_init = np.clip(X_init, -0.49, 0.49)
        
        # Run L-BFGS-B
        try:
            result = minimize(
                soft_min_obj,
                X_init.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'gtol': 1e-6, 'maxls': 50},
                bounds=[(-0.5, 0.5)] * (3 * N),
            )
            
            if result.success or result.fun < best_score:
                X_new = result.x.reshape(N, 3)
                # Clip to ensure validity
                X_new = np.clip(X_new, -0.5, 0.5)
                score_new = compute_true_score(X_new)
                if score_new > best_score:
                    best_score = score_new
                    best_X = X_new.copy()
        except Exception:
            pass
    
    return best_X

# EVOLVE-BLOCK-END
