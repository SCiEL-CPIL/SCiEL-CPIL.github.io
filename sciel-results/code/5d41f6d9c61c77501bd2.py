"""
Strategy: Systematic Search for Rank-16 via Basis Transformation and Sparsity.

The current program achieves rank 17 via a block decomposition (Strassen 2x2x2 + direct terms).
The theoretical lower bound is 9, but known bounds for 2x3x3 might allow 16?
Actually, the rank of C_{2,3,3} is not known to be less than 17 in standard literature, 
but let's try to find a rank-16 decomposition using a more flexible search.

The core idea:
1. We fix the structure of the decomposition.
2. We allow the factor matrices U, V, W to have entries in {0, 0.5, 1, -0.5, -1}.
3. We use a randomized local search / simulated annealing approach to minimize the Frobenius norm of the reconstruction error, hoping to snap to zero.
4. Since the dimensions are small (6, 9, 6), we can afford a moderate search.

However, a more robust "different" strategy is to use **Linear Algebraic Constraints**.
We know that if a rank-R decomposition exists, then for any vector u, the matrix T_u (obtained by contracting T with u) must be in the span of {v_c outer w_c}.
Specifically, if we fix a random U (dim_U x R), we can solve for V and W? No, it's bilinear.

Let's try a different angle: **Combinatorial Search over Sparse Supports**.
The rank-17 solution has a very specific sparsity pattern.
Maybe a rank-16 solution exists with a different sparsity pattern?
We can search for a set of 16 supports (triples of index sets for U, V, W) and then solve for the coefficients.
If we fix the supports (which entries are non-zero in each of the R vectors), the problem becomes linear in the coefficients?
No, T = sum u_c v_c^T w_c. This is still multilinear.

But wait! If we fix the *supports* (the sets of indices that are non-zero), we still have the values.
However, if we restrict values to {0, 1, -1}, we can enumerate? No, too many.

Let's look at the "different design family" directive.
Current: Explicit construction (Block decomposition).
New: **Randomized Optimization with Discrete Snapping**.
We initialize R=16 random matrices U, V, W.
We minimize || sum u_c v_c w_c - T ||_F.
We use gradient descent.
After each step, we snap the values to the nearest {0, 0.5, 1, -0.5, -1}.
We check if the error is zero.

This is a heuristic, but it explores the continuous space before snapping, which might find a valid integer/half-integer solution that a pure discrete search would miss due to local minima.

Another option: **Use the known Rank 17 but try to merge terms?**
No, that's refining.

Let's go with the **Randomized Optimization with Snapping** for R=16.
If it fails, we return the Rank 17 construction (which we know works) to ensure a valid score, but we try to beat it.

Actually, there is a known result that the rank of C_{2,3,3} is **17**.
If the rank is indeed 17, then searching for 16 is futile and will waste time.
The lower bound is max(6,9,6)=9.
But let's check the literature.
Landsberg (1998) "Tensors: Geometry and Applications".
The rank of C_{2,3,3} is not explicitly given as 16 in standard quick references.
However, the current program's score is 1.0, which implies it found the seed value.
The seed value is likely 17 (the best known).
If the true rank is 17, we cannot do better.
If the true rank is 16, we can.

Let's assume the task allows for finding a better decomposition if one exists.
I will implement a search for R=16 using a hybrid of random initialization and local search (Nelder-Mead or similar) on the squared error, with periodic snapping to half-integers.

If no solution is found for R=16 within the time limit, I will fall back to the R=17 construction.

To make this "MATERIALLY DIFFERENT", I will not hardcode the R=17 construction in the main flow, but generate it via the same optimization if R=17 is the target? No, that's inefficient.
I will keep the R=17 construction as a verified fallback, but the *search* for R=16 is the new component.

Wait, the directive says "Keep the function names and signatures of the seed program."
It does not say I must use the seed's logic.

I will implement:
1. A function to construct the R=17 solution (reusing the logic from the prompt, but cleaned up).
2. A function `search_rank(R)` that tries to find a decomposition of rank R using randomized optimization.
3. `construct()` will first try `search_rank(16)`. If it succeeds, return it. Otherwise, return the R=17 solution.

This is a structural change: adding an active search phase for a lower rank before falling back to the known construction.
"""

import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

def _construct_rank_17():
    """
    Reconstructs the rank-17 solution from the current program.
    This serves as the guaranteed valid fallback.
    """
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n

    T = np.zeros((dim_U, dim_V, dim_W), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    def make_U_coeff(indices_ij, coeffs):
        vec = np.zeros(dim_U)
        for (i, j), c in zip(indices_ij, coeffs):
            vec[i*m + j] += c
        return vec

    def make_V_coeff(indices_jk, coeffs):
        vec = np.zeros(dim_V)
        for (j, k), c in zip(indices_jk, coeffs):
            vec[j*p + k] += c
        return vec

    def make_W_coeff(indices_ki, coeffs):
        vec = np.zeros(dim_W)
        for (k, i), c in zip(indices_ki, coeffs):
            vec[k*n + i] += c
        return vec

    # Block 1: Strassen 2x2x2
    U_b1 = [
        make_U_coeff([(0,0), (1,1)], [1, 1]),
        make_U_coeff([(1,0), (1,1)], [1, 1]),
        make_U_coeff([(0,0)], [1]),
        make_U_coeff([(1,1)], [1]),
        make_U_coeff([(0,0), (0,1)], [1, 1]),
        make_U_coeff([(1,0), (0,0)], [1, -1]),
        make_U_coeff([(0,1), (1,1)], [1, -1])
    ]
    V_b1 = [
        make_V_coeff([(0,0), (1,1)], [1, 1]),
        make_V_coeff([(0,0)], [1]),
        make_V_coeff([(0,1), (1,1)], [1, -1]),
        make_V_coeff([(1,0), (0,0)], [1, -1]),
        make_V_coeff([(1,1)], [1]),
        make_V_coeff([(0,0), (0,1)], [1, 1]),
        make_V_coeff([(1,0), (1,1)], [1, 1])
    ]
    W_b1 = [
        make_W_coeff([(0,0), (1,1)], [1, 1]),
        make_W_coeff([(0,1), (1,1)], [1, -1]),
        make_W_coeff([(1,0)], [1]),
        make_W_coeff([(0,1), (1,1)], [1, 1]),
        make_W_coeff([(0,0), (1,0)], [-1, 1]),
        make_W_coeff([(1,1)], [1]),
        make_W_coeff([(0,0), (1,1)], [1, 1])
    ]

    # Block 2: j=2
    U_b2 = []
    V_b2 = []
    W_b2 = []
    for i in range(2):
        for k in range(3):
            U_b2.append(make_U_coeff([(i, 2)], [1]))
            V_b2.append(make_V_coeff([(2, k)], [1]))
            W_b2.append(make_W_coeff([(k, i)], [1]))

    # Block 3: k=2, j<2
    U_b3 = []
    V_b3 = []
    W_b3 = []
    for i in range(2):
        for j in range(2):
            U_b3.append(make_U_coeff([(i, j)], [1]))
            V_b3.append(make_V_coeff([(j, 2)], [1]))
            W_b3.append(make_W_coeff([(2, i)], [1]))

    U_all = U_b1 + U_b2 + U_b3
    V_all = V_b1 + V_b2 + V_b3
    W_all = W_b1 + W_b2 + W_b3

    # Verify
    T_recon = np.zeros((dim_U, dim_V, dim_W))
    for c in range(len(U_all)):
        T_recon += np.einsum('i,j,k->ijk', U_all[c], V_all[c], W_all[c])
    
    if not np.allclose(T_recon, T):
        # Should not happen, but fallback to schoolbook just in case
        best_rank = n * m * p
        best_U = np.zeros((dim_U, best_rank))
        best_V = np.zeros((dim_V, best_rank))
        best_W = np.zeros((dim_W, best_rank))
        for c in range(best_rank):
            i = c // (m*p)
            j = (c // p) % m
            k = c % p
            best_U[i*m+j, c] = 1.0
            best_V[j*p+k, c] = 1.0
            best_W[k*n+i, c] = 1.0
        return best_U, best_V, best_W

    return np.array(U_all).T, np.array(V_all).T, np.array(W_all).T

def _search_rank(R, time_limit=100.0):
    """
    Attempts to find a rank-R decomposition using randomized optimization.
    Returns (U, V, W) if found, else None.
    """
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n

    T = np.zeros((dim_U, dim_V, dim_W), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    # Flatten T for error calculation
    T_flat = T.flatten()

    # Parameters: U (dim_U x R), V (dim_V x R), W (dim_W x R)
    # Total params: R * (dim_U + dim_V + dim_W)
    
    def objective(x):
        U = x[:R*dim_U].reshape(R, dim_U).T
        V = x[R*dim_U:R*dim_U + R*dim_V].reshape(R, dim_V).T
        W = x[R*dim_U + R*dim_V:].reshape(R, dim_W).T
        
        # Compute sum of outer products
        # Efficiently: 
        # T_recon[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
        # We can do this via einsum or loops. Loops are slow, einsum is faster.
        T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.sum((T_recon - T)**2)
        return err

    # Snap function to half-integers
    def snap(x):
        # x is a flat array of parameters
        # We want to snap values to {0, 0.5, 1, -0.5, -1}
        # This is a heuristic.
        U = x[:R*dim_U].reshape(R, dim_U)
        V = x[R*dim_U:R*dim_U + R*dim_V].reshape(R, dim_V)
        W = x[R*dim_U + R*dim_V:].reshape(R, dim_W)
        
        U_snapped = np.round(U * 2) / 2
        V_snapped = np.round(V * 2) / 2
        W_snapped = np.round(W * 2) / 2
        
        # Also try to zero out small values
        U_snapped[np.abs(U_snapped) < 0.1] = 0
        V_snapped[np.abs(V_snapped) < 0.1] = 0
        W_snapped[np.abs(W_snapped) < 0.1] = 0
        
        return np.concatenate([U_snapped.flatten(), V_snapped.flatten(), W_snapped.flatten()])

    start_time = time.time()
    best_err = float('inf')
    best_x = None

    # Try a few random initializations
    for attempt in range(5):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with small random values
        np.random.seed(42 + attempt)
        x0 = np.random.randn(R * (dim_U + dim_V + dim_W)) * 0.1
        
        # Run Nelder-Mead
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.fun < best_err:
                best_err = res.fun
                best_x = res.x
                
                # Check if we found an exact solution
                if best_err < 1e-6:
                    # Snap and verify
                    x_snapped = snap(best_x)
                    err_snapped = objective(x_snapped)
                    if err_snapped < 1e-6:
                        U = x_snapped[:R*dim_U].reshape(R, dim_U).T
                        V = x_snapped[R*dim_U:R*dim_U + R*dim_V].reshape(R, dim_V).T
                        W = x_snapped[R*dim_U + R*dim_V:].reshape(R, dim_W).T
                        return U, V, W
                        
        except Exception:
            continue

    return None

def construct():
    start_time = time.time()
    
    # First, try to find a rank-16 decomposition
    # This is the "new" part of the strategy.
    # If it succeeds, we return rank 16.
    # If it fails, we return the known rank 17.
    
    # We allocate 100 seconds for the search.
    # If the search fails or times out, we use the remaining time to construct rank 17.
    
    U16, V16, W16 = _search_rank(16, time_limit=100.0)
    
    if U16 is not None:
        # Verify rank 16 solution
        n, m, p = DIMENSIONS
        dim_U = n * m
        dim_V = m * p
        dim_W = p * n
        T = np.zeros((dim_U, dim_V, dim_W), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m + j, j*p + k, k*n + i] = 1.0
        
        T_recon = np.einsum('ic,jc,kc->ijk', U16, V16, W16)
        if np.allclose(T_recon, T):
            return {
                "U": U16.tolist(),
                "V": V16.tolist(),
                "W": W16.tolist()
            }
        else:
            # Fallback if verification fails (should not happen if objective was 0)
            pass

    # Fallback to Rank 17
    U17, V17, W17 = _construct_rank_17()
    
    return {
        "U": U17.tolist(),
        "V": V17.tolist(),
        "W": W17.tolist()
    }
