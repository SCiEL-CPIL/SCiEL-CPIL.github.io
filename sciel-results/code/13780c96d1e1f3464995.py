import numpy as np
from scipy.optimize import minimize
import time

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Optimizes the 64 base positions x_i to minimize the area of the union
    of the corresponding triangles using L-BFGS-B, starting from the
    Keich construction.
    """
    n = 64
    k = int(np.log2(n))
    
    # 1. Generate Keich positions as initial guess
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        # Binary representation of i, padded to length k
        # bin(i) is like '0b101', we take the bits
        bits = [int(b) for b in bin(i)[2:]]
        bits = [0] * (k - len(bits)) + bits
        eps = np.array(bits, dtype=float)
        # x_i = sum_{j=1}^k ((1 - j) / k) * eps_j * 2^{-j}
        # Note: j in formula corresponds to index in bits.
        # bits[0] is 2^{k-1}? No, standard binary expansion.
        # Let's stick to the exact formula from the seed code to ensure consistency
        eps_seed = np.array(list(str(bin(i))[2:]), dtype=int)
        eps_seed = np.pad(eps_seed, (k - len(eps_seed), 0), "constant")
        pos = np.sum((1 - j) * eps_seed * (2.0 ** -j) / k)
        positions.append(pos)
    x0 = np.array(positions, dtype=np.float64)
    
    # Pre-compute constants for the area calculation
    # y_m = (m + 0.5) / 20000
    M = 20000
    y = (np.arange(M) + 0.5) / M
    # a_i = (i + 1) / 64
    a = (np.arange(n) + 1) / n
    # w = (1 - y) / 64
    w = (1 - y) / n
    
    def compute_area(x):
        # x shape (n,)
        # a shape (n,)
        # y shape (M,)
        # We need L[m, i] = x[i] + a[i] * y[m]
        # L = x[None, :] + a[None, :] * y[:, None]
        L = x[None, :] + a[None, :] * y[:, None]
        
        # Sort L along axis 1 for each m
        L_sorted = np.sort(L, axis=1)
        
        # Compute differences
        diffs = np.diff(L_sorted, axis=1)
        
        # Compute min(diffs, w[m])
        # w shape (M,), need to broadcast to (M, n-1)
        mins = np.minimum(diffs, w[:, None])
        
        # Union length = w[m] + sum(mins[m, :])
        union_lengths = w + np.sum(mins, axis=1)
        
        # Area is the mean
        return np.mean(union_lengths)
    
    # Optimize
    # Use L-BFGS-B
    # Options: maxiter, ftol, gtol
    # Since the function is convex, we want to converge tightly.
    
    start_time = time.time()
    timeout = 600 # Safety margin
    
    try:
        res = minimize(
            compute_area, 
            x0, 
            method='L-BFGS-B',
            jac='2-point', # Central differences
            options={
                'maxiter': 1000,
                'ftol': 1e-12,
                'gtol': 1e-10,
                'maxls': 20
            }
        )
        if res.success:
            x_opt = res.x
        else:
            x_opt = res.x # Use best found even if not perfect convergence
            
        # Check time
        elapsed = time.time() - start_time
        if elapsed > timeout:
             # Should not happen, but return best so far
             pass
             
        return x_opt

    except Exception as e:
        # Fallback to Keich if optimization fails
        return x0

# EVOLVE-BLOCK-END
