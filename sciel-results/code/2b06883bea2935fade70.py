import numpy as np
import time
import scipy.optimize as opt

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    timeout = 600  # seconds
    
    def compute_P(support, weights):
        # support: 1D array
        # weights: 1D array
        n = len(support)
        # Compute P = sum w_k w_l (sum w_i w_j I(s_i+s_j < 2s_l - s_k))
        # Vectorized:
        # s_i + s_j < 2 s_l - s_k
        # s_i + s_j + s_k < 2 s_l
        
        # Create meshgrids for i, j, k, l
        # This is O(N^4). For N=50, 6.25M ops. OK.
        # For N=100, 100M ops. Might be slow in Python loop, but vectorized numpy is fast.
        
        s = support
        w = weights
        
        # s_i + s_j
        # We need sum over i,j of w_i w_j I(s_i + s_j < 2 s_l - s_k)
        # Let A_{kl} = sum_{i,j: s_i+s_j < 2s_l - s_k} w_i w_j
        # P = sum_{k,l} w_k w_l A_{kl}
        
        # Efficient computation:
        # For each l, compute CDF of S2 = X1+X2 at 2s_l - s_k for all k?
        # Or just do the full 4D sum if N is small.
        
        if n > 40:
             # Use a faster method or smaller N
             pass
            
        # 4D array might be too big for memory if N is large.
        # N=20 -> 160k floats. 1.3 MB. Fine.
        # N=50 -> 6.25M floats. 50 MB. Fine.
        
        # s_i + s_j + s_k < 2 s_l
        # s_i + s_j + s_k - 2 s_l < 0
        
        # Let's construct the matrix M where M[i,j,k] = s_i+s_j+s_k
        # And check against 2*s_l
        
        # Actually, simpler:
        # P = sum_{i,j,k,l} w_i w_j w_k w_l I(s_i+s_j+s_k < 2s_l)
        
        # We can compute this by iterating l and using the CDF of S3 = X1+X2+X3.
        # But S3 distribution depends on w.
        
        # For small N, direct sum is fine.
        # s_i + s_j + s_k < 2 s_l
        
        # s_i + s_j
        # s_ij = s[:, None] + s[None, :] # N x N
        # s_ijk = s_ij[:, :, None] + s[None, None, :] # N x N x N
        # s_l = s[None, None, None, :] # 1 x 1 x 1 x N
        # cond = s_ijk[..., None] < 2 * s_l # N x N x N x N
        # P = np.sum(w[:, None, None, None] * w[None, :, None, None] * w[None, None, :, None] * w[None, None, None, :] * cond)
        
        # This creates a N^4 boolean array.
        # N=50 -> 6.25M booleans. ~6 MB. OK.
        # N=100 -> 100M booleans. ~100 MB. OK.
        
        s = support
        w = weights
        
        # To save memory, we can loop over l.
        P_val = 0.0
        for l in range(n):
            threshold = 2 * s[l]
            # We need sum_{i,j,k} w_i w_j w_k I(s_i+s_j+s_k < threshold) * w_l
            # Let S3_dist be the distribution of X1+X2+X3.
            # But we can just sum over i,j,k.
            # s_i + s_j + s_k < threshold
            
            # s_i + s_j
            s_ij = s[:, None] + s[None, :]
            # s_ijk = s_ij + s[None, None, :]
            # cond = s_ijk < threshold
            
            # Vectorized inner sum
            # s_ijk shape N x N x N
            # w_ijk shape N x N x N
            # prod = w[:, None, None] * w[None, :, None] * w[None, None, :]
            
            # This is still N^3 per l. Total N^4.
            
            # Optimization:
            # Precompute s_ij.
            # For each l, compute sum_{i,j,k} w_i w_j w_k I(s_i+s_j+s_k < 2s_l)
            
            # We can precompute the sorted sums? No, w changes.
            
            # Let's just do the N^4 loop in numpy.
            # s_ijk = s[:, None, None] + s[None, :, None] + s[None, None, :]
            # This is N^3.
            # Then for each l, mask = s_ijk < 2*s[l]
            # sum_masked = np.sum(w[:, None, None] * w[None, :, None] * w[None, None, :] * mask)
            # P += w[l] * sum_masked
            
            # Precompute s_ijk and w_prod
            s_ijk = s[:, None, None] + s[None, :, None] + s[None, None, :]
            w_prod = w[:, None, None] * w[None, :, None] * w[None, None, :]
            
            # Loop over l is fast in Python if inner is vectorized
            for l in range(n):
                mask = s_ijk < 2 * s[l]
                P_val += w[l] * np.sum(w_prod * mask)
                
        return P_val

    def objective(w):
        return -compute_P(support, w)

    # Try a few configurations
    best_P = 0
    best_w = None
    best_support = None
    
    # Configuration 1: Uniform grid
    N = 20
    support = np.arange(N, dtype=np.float64)
    w0 = np.ones(N) / N
    
    res = opt.minimize(objective, w0, method='SLSQP', bounds=[(1e-6, None)]*N, constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}], options={'maxiter': 1000, 'ftol': 1e-9})
    if res.success:
        P = -res.fun
        if P > best_P:
            best_P = P
            best_w = res.x
            best_support = support

    # Configuration 2: Geometric-like support
    N = 20
    support = np.exp(np.linspace(0, 2, N))
    w0 = np.ones(N) / N
    # Normalize support to start at 0? Support must be >= 0.
    # If support is exp, min is 1.
    # Let's shift to start at 0.
    support = support - support[0]
    
    res = opt.minimize(objective, w0, method='SLSQP', bounds=[(1e-6, None)]*N, constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}], options={'maxiter': 1000, 'ftol': 1e-9})
    if res.success:
        P = -res.fun
        if P > best_P:
            best_P = P
            best_w = res.x
            best_support = support

    # Return best
    # Ensure valid output
    best_w = np.maximum(best_w, 0)
    best_w = best_w / np.sum(best_w)
    
    return best_support, best_w
