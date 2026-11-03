"""
Strategy: Exploit the structural symmetry between X^2 and P^2 in the Hermite basis.

The key insight is that the matrices X^2 and P^2 have identical diagonal entries (n + 1/2) 
and off-diagonal entries that differ only by a sign for the (n, n±2) terms.
Specifically:
X^2_{n, n+2} = + sqrt((n+1)(n+2)/4)
P^2_{n, n+2} = - sqrt((n+1)(n+2)/4)

This structure allows us to analyze the objective function J(c) = sqrt(c^T X^2 c * c^T P^2 c) / ||c||^2 
more effectively. 

A different strategy from the current one (which uses Differential Evolution on the raw coefficients) 
is to use a **global grid search followed by high-precision local refinement**, but with a 
**parameterization that respects the scale invariance** of the problem.

Since J(λc) = J(c), we can fix ||c|| = 1. The problem reduces to minimizing over the unit sphere 
in 4D (with the constraint g(0)=0, which is a hyperplane, so effectively a 2D sphere).

However, a more robust and different approach is to use **Basin Hopping** with a high-precision 
local optimizer (L-BFGS-B) as the inner loop. Basin Hopping is designed for problems with many 
local minima and is more effective than simple Differential Evolution for smooth but non-convex 
landscapes when combined with a strong local optimizer.

Additionally, we will implement a **multi-start** strategy using deterministic and random initial 
points spread across the domain, leveraging the time budget to explore multiple basins.

We will also correct the matrix construction to ensure the signs are exactly as derived:
X^2 has + off-diagonals, P^2 has - off-diagonals.
"""

import time
import numpy as np
from scipy.optimize import minimize, basin_hopping

DEADLINE_MARGIN = 10.0
MAX_RUN_TIME = 200.0

def construct():
    t0 = time.time()
    
    # Basis indices: degrees 0, 4, 8, 12
    basis_idx = [0, 4, 8, 12]
    size = 13  # up to 12
    
    # Precompute full X2 and P2 matrices for n=0..12
    X2_full = np.zeros((size, size))
    P2_full = np.zeros((size, size))
    
    for n in range(size):
        diag = n + 0.5
        X2_full[n, n] = diag
        P2_full[n, n] = diag
        
        if n >= 2:
            coeff = np.sqrt(n * (n - 1) / 4.0)
            X2_full[n, n-2] = coeff
            X2_full[n-2, n] = coeff
            P2_full[n, n-2] = -coeff
            P2_full[n-2, n] = -coeff
            
        if n < size - 2:
            coeff = np.sqrt((n + 1) * (n + 2) / 4.0)
            X2_full[n, n+2] = coeff
            X2_full[n+2, n] = coeff
            P2_full[n, n+2] = -coeff
            P2_full[n+2, n] = -coeff

    # Extract submatrices
    X2 = np.zeros((4, 4))
    P2 = np.zeros((4, 4))
    for i, ni in enumerate(basis_idx):
        for j, nj in enumerate(basis_idx):
            X2[i, j] = X2_full[ni, nj]
            P2[i, j] = P2_full[ni, nj]
            
    # Compute psi_n(0) for n=0,4,8,12
    # psi_n(x) = (2^n n! sqrt(pi))^{-1/2} H_n(x) e^{-x^2/2}
    # psi_0(0) = pi^{-1/4}
    # psi_n(0) = 0 for odd n.
    # Recurrence: H_n(x) = 2x H_{n-1}(x) - 2(n-1) H_{n-2}(x)
    # At x=0: H_n(0) = -2(n-1) H_{n-2}(0)
    # psi_n(0) = (2^n n! sqrt(pi))^{-1/2} H_n(0) pi^{-1/2} e^0
    # Let's compute directly using the recurrence for psi_n(0).
    # psi_n(0) = -sqrt((n-1)/n) * psi_{n-2}(0) * (2 * n * (n-1) / (n-1)) ? 
    # Actually, let's use the known values or compute H_n(0) and normalize.
    
    psi_vals = np.zeros(size)
    psi_vals[0] = np.pi ** (-0.25)
    for n in range(2, size, 2):
        # H_n(0) = -2(n-1) H_{n-2}(0)
        # psi_n(0) = H_n(0) / sqrt(2^n n! sqrt(pi))
        # psi_{n-2}(0) = H_{n-2}(0) / sqrt(2^{n-2} (n-2)! sqrt(pi))
        # Ratio: psi_n(0)/psi_{n-2}(0) = [H_n(0)/H_{n-2}(0)] * [sqrt(2^{n-2}(n-2)!) / sqrt(2^n n!)]
        # = [-2(n-1)] * [1/sqrt(4 n (n-1))]
        # = -2(n-1) / (2 sqrt(n(n-1)))
        # = -sqrt(n-1)/sqrt(n)
        # So psi_n(0) = -sqrt((n-1)/n) * psi_{n-2}(0)
        psi_vals[n] = -np.sqrt((n-1)/n) * psi_vals[n-2]
        
    h0 = np.array([psi_vals[0], psi_vals[4], psi_vals[8], psi_vals[12]])
    
    def objective(c_free):
        c0, c4, c8 = c_free
        # c12 = -(c0*h0[0] + c4*h0[1] + c8*h0[2]) / h0[3]
        c12 = -(c0 * h0[0] + c4 * h0[1] + c8 * h0[2]) / h0[3]
        c = np.array([c0, c4, c8, c12])
        
        norm_sq = np.dot(c, c)
        if norm_sq < 1e-15:
            return 1e15
            
        x2_val = np.dot(c, X2.dot(c))
        p2_val = np.dot(c, P2.dot(c))
        
        if x2_val < 0 or p2_val < 0:
            return 1e15
            
        return np.sqrt(x2_val * p2_val) / norm_sq

    # Seed
    c_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_score = objective(c_seed)
    best_c = c_seed.copy()
    
    # Multi-start Basin Hopping
    # We will use a set of starting points.
    # The coefficients are likely small. Let's try a grid of starting points.
    starts = []
    # Seed
    starts.append(c_seed.copy())
    
    # Small perturbations around seed
    rng = np.random.default_rng(42)
    for _ in range(20):
        perturbation = rng.normal(0, 0.5, 3)
        starts.append(c_seed + perturbation)
        
    # Random points in a bounded box
    for _ in range(20):
        starts.append(rng.uniform(-2, 2, 3))
        
    # Even more spread
    for _ in range(10):
        starts.append(rng.uniform(-5, 5, 3))

    for start in starts:
        if time.time() - t0 > MAX_RUN_TIME - DEADLINE_MARGIN:
            break
            
        try:
            # Use Basin Hopping with L-BFGS-B as the local optimizer
            res = basin_hopping(objective, start, niter=50, seed=42, 
                                minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 100}})
            if res.fun < best_score:
                best_score = res.fun
                best_c = res.x.copy()
        except Exception:
            pass

    # Final refinement with Nelder-Mead for high precision
    try:
        res2 = minimize(objective, best_c, method='Nelder-Mead', 
                        options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
        if res2.fun < best_score:
            best_score = res2.fun
            best_c = res2.x.copy()
    except Exception:
        pass
        
    return np.array(best_c)

if __name__ == "__main__":
    print(construct())
