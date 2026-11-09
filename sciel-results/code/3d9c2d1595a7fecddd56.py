# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


def _shannon(p):
    p = p[p > 0]
    return float(np.sum(-p * np.log2(p)))


def _proj_entropy(P, r):
    # P[y, x]; value = x + r*y
    n = P.shape[0]
    # Precomputed values for speed? No, P changes. 
    # But the binning structure is fixed.
    # vals[y, x] = x + r*y
    # We can precompute the bin indices for each r.
    pass


# Precompute bin indices for slopes -1, 0, 1, 2
# n=10
# x, y in 0..9
# v = x + r*y
# For r=-1: x-y ranges -9..9 (19 bins)
# For r=0: x ranges 0..9 (10 bins)
# For r=1: x+y ranges 0..18 (19 bins)
# For r=2: x+2y ranges 0..27 (28 bins)

def _precompute_bins(n=10):
    bins = {}
    for r in [-1, 0, 1, 2]:
        # Create a 10x10 array of bin indices
        bin_idx = np.zeros((n, n), dtype=int)
        min_val = None
        max_val = None
        for y in range(n):
            for x in range(n):
                v = x + r * y
                if min_val is None or v < min_val:
                    min_val = v
                if max_val is None or v > max_val:
                    max_val = v
        # Map values to 0..K-1
        # Since values are integers, we can just subtract min_val
        # But some values might be missing. We need a mapping.
        # Actually, for entropy, we just need to group by value.
        # We can use np.unique on the flattened array of values.
        vals = np.array([x + r * y for y in range(n) for x in range(n)])
        unique_vals, inverse = np.unique(vals, return_inverse=True)
        bin_idx_flat = inverse.reshape(n, n)
        bins[r] = bin_idx_flat
    return bins

BINS = _precompute_bins(10)

def _entropy_from_weights(weights, bin_idx):
    # weights: 10x10 array
    # bin_idx: 10x10 array of bin indices
    # sum weights by bin
    # np.bincount requires 1D input
    w_flat = weights.ravel()
    b_flat = bin_idx.ravel()
    sums = np.bincount(b_flat, weights=w_flat)
    return _shannon(sums)

def _compute_entropies(P):
    H_neg1 = _entropy_from_weights(P, BINS[-1])
    H_0 = _entropy_from_weights(P, BINS[0])
    H_1 = _entropy_from_weights(P, BINS[1])
    H_2 = _entropy_from_weights(P, BINS[2])
    return H_neg1, H_0, H_1, H_2

def _soft_max(vals, beta):
    # vals is a list or array
    vals = np.array(vals)
    m = np.max(vals)
    if m <= 1e-9:
        return 1e-9
    exp_vals = np.exp(beta * (vals - m))
    return m + np.log(np.sum(exp_vals)) / beta

def _objective_and_grad(z, beta=10.0):
    # z: 100-dim array of log-weights
    # P = softmax(z)
    # To avoid overflow/underflow, subtract max
    z_max = np.max(z)
    exp_z = np.exp(z - z_max)
    P = exp_z / np.sum(exp_z)
    
    H_neg1, H_0, H_1, H_2 = _compute_entropies(P)
    
    denom = _soft_max([H_0, H_1, H_2], beta)
    
    if denom < 1e-9:
        return 0.0, np.zeros_like(z)
        
    val = H_neg1 / denom
    obj = -val
    
    # Gradient calculation
    # dH/dz_k = sum_i (dH/dP_i) (dP_i/dz_k)
    # dP_i/dz_k = P_i (delta_ik - P_k)
    # dH/dP_i = -log2(P_i) - 1/ln(2)  (for P_i > 0)
    
    # We need gradients for H_neg1, H_0, H_1, H_2
    # Let G_H be the gradient of H w.r.t z.
    # G_H[k] = sum_i ( -log2(P_i) - 1/ln(2) ) P_i (delta_ik - P_k)
    #        = ( -log2(P_k) - 1/ln(2) ) P_k - P_k sum_i ( -log2(P_i) - 1/ln(2) ) P_i
    #        = P_k ( -log2(P_k) - 1/ln(2) - H_scaled )
    # where H_scaled = sum_i P_i (-log2(P_i) - 1/ln(2)) = H - 1/ln(2) * sum(P_i) = H - 1/ln(2)
    # Wait, H is sum -P log2 P.
    # sum_i P_i (-log2 P_i - 1/ln 2) = H - 1/ln 2.
    
    ln2 = np.log(2.0)
    log2_P = np.log2(P + 1e-300) # avoid log(0)
    
    # Gradient of a single entropy H w.r.t z
    # G_H[k] = P_k * ( -log2(P_k) - 1/ln2 - (H - 1/ln2) )
    #        = P_k * ( -log2(P_k) - H )
    
    # Let's verify:
    # dH/dz_k = sum_i (dH/dP_i) P_i (delta_ik - P_k)
    # dH/dP_i = -log2(P_i) - 1/ln2
    # sum_i (dH/dP_i) P_i = sum_i (-log2(P_i) P_i - P_i/ln2) = H - 1/ln2
    # Term k: (dH/dP_k) P_k - P_k * (H - 1/ln2)
    # = (-log2(P_k) - 1/ln2) P_k - P_k H + P_k/ln2
    # = -P_k log2(P_k) - P_k/ln2 - P_k H + P_k/ln2
    # = -P_k log2(P_k) - P_k H
    # = -P_k (log2(P_k) + H)
    
    # So G_H[k] = -P_k (log2(P_k) + H)
    
    G_neg1 = -P * (log2_P + H_neg1)
    G_0 = -P * (log2_P + H_0)
    G_1 = -P * (log2_P + H_1)
    G_2 = -P * (log2_P + H_2)
    
    # Gradient of soft max
    # S = m + 1/beta log(sum(exp(beta(v_i - m))))
    # dS/dv_j = (1/S) * exp(beta(v_j - m)) * beta / beta = exp(beta(v_j - m)) / sum(exp(...))
    # Let w_j = exp(beta(v_j - m)) / sum(exp(...))
    # dS/dv_j = w_j
    
    vals = np.array([H_0, H_1, H_2])
    m = np.max(vals)
    exp_vals = np.exp(beta * (vals - m))
    sum_exp = np.sum(exp_vals)
    w_0 = exp_vals[0] / sum_exp
    w_1 = exp_vals[1] / sum_exp
    w_2 = exp_vals[2] / sum_exp
    
    # d/denom/dv_j = w_j
    # denom = S
    # d/denom/dH_0 = w_0, etc.
    
    # obj = - H_neg1 / denom
    # d obj / d z = - ( (dH_neg1/dz * denom) - H_neg1 * (d denom / dz) ) / denom^2
    #             = - ( G_neg1 * denom - H_neg1 * (w_0 G_0 + w_1 G_1 + w_2 G_2) ) / denom^2
    
    grad_denom = w_0 * G_0 + w_1 * G_1 + w_2 * G_2
    
    grad_obj = - ( G_neg1 * denom - H_neg1 * grad_denom ) / (denom * denom)
    
    return obj, grad_obj

def arithmetic_kakeya_distribution() -> np.ndarray:
    start = time.time()
    DEADLINE = start + 600.0
    n = 10
    best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    best_score = -np.inf

    # Initializations
    # 1. Uniform
    # 2. Anti-diagonal bias (maximize x-y spread)
    # 3. Random

    inits = []
    
    # Uniform
    z_uni = np.zeros(n*n)
    inits.append(z_uni)
    
    # Anti-diagonal: high weight on x-y extremes
    # x-y = 9: (0,9), x-y=-9: (9,0)
    # We want mass on corners to maximize H_neg1
    z_corner = np.zeros(n*n)
    # Boost corners
    for y in range(n):
        for x in range(n):
            diff = abs(x - y)
            z_corner[y*n + x] = diff * 0.5
    inits.append(z_corner)
    
    # Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        z_rand = rng.normal(0, 1, n*n)
        inits.append(z_rand)

    for z_init in inits:
        if time.time() > DEADLINE - 60:
            break
        
        # Run optimization
        # Use L-BFGS-B with numerical gradient check? No, we have analytical.
        # We need to ensure P is valid. Softmax ensures this.
        
        def fun(z):
            obj, grad = _objective_and_grad(z, beta=10.0)
            return obj, grad

        try:
            res = minimize(fun, z_init, method='L-BFGS-B', jac=True, options={'maxiter': 200, 'ftol': 1e-10})
            
            # Convert final z to P
            z_final = res.x
            z_max = np.max(z_final)
            exp_z = np.exp(z_final - z_max)
            P = exp_z / np.sum(exp_z)
            P = P.reshape(n, n)
            
            # Calculate true score with hard max
            H_neg1, H_0, H_1, H_2 = _compute_entropies(P)
            M = max(H_0, H_1, H_2)
            if M > 1e-9:
                score = H_neg1 / M
                if score > best_score:
                    best_score = score
                    best_P = P
                    
        except Exception:
            pass

    # Polish: use the best P found as a new initialization for a higher beta (sharper max)
    if best_score > -np.inf:
        # Re-optimize from best_P with higher beta
        # Convert P to z (log P)
        z_best = np.log(best_P.ravel() + 1e-300)
        
        for beta in [50.0, 100.0]:
            if time.time() > DEADLINE - 30:
                break
            
            def fun_polish(z):
                obj, grad = _objective_and_grad(z, beta=beta)
                return obj, grad

            try:
                res = minimize(fun_polish, z_best, method='L-BFGS-B', jac=True, options={'maxiter': 100, 'ftol': 1e-12})
                z_final = res.x
                z_max = np.max(z_final)
                exp_z = np.exp(z_final - z_max)
                P = exp_z / np.sum(exp_z)
                P = P.reshape(n, n)
                
                H_neg1, H_0, H_1, H_2 = _compute_entropies(P)
                M = max(H_0, H_1, H_2)
                if M > 1e-9:
                    score = H_neg1 / M
                    if score > best_score:
                        best_score = score
                        best_P = P
                        z_best = z_final
            except Exception:
                pass

    # Final safety
    best_P = np.clip(best_P, 0.0, None)
    tot = best_P.sum()
    if tot <= 1e-9:
        best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        best_P = best_P / tot

    return best_P


# EVOLVE-BLOCK-END
