"""
Rank-minimal tensor decomposition for 2x4 * 4x6 matrix multiplication.

The matrix multiplication tensor T[i,j,k] for (A: n x m) * (B: m x p) = (C: n x p)
has entries T[i, j, k] = 1 if the product C[i,k] = sum_j A[i,j]*B[j,k] includes the term.
Actually, the tensor is defined as:
  T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
  where the decomposition is: C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
  
Wait, let me re-read the convention. The output factor W follows W[k*n+i, r].
So the tensor is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,k}... no.

Actually, the matrix multiplication tensor: we want to compute C = A * B where A is n x m, B is m x p.
C[i,k] = sum_j A[i,j] B[j,k]

The tensor decomposition expresses this as:
C[i,k] = sum_c (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k]) * W[k*n+i, c]

Wait no, the standard tensor decomposition for matrix multiplication:
The multiplication tensor has rank r if there exist U, V, W of size (n*m) x r, (m*p) x r, (p*n) x r
such that for all i, j, k:
  sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1

This is because the tensor T_{ijk} = 1 (for the standard matrix multiplication, every "slice" is identity-like).

Actually, the matrix multiplication tensor T is a 3-tensor of dimensions (n*m) x (m*p) x (p*n) where:
T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
T[a, b, c] = 0 otherwise

So T[(i*m+j), (j*p+k), (k*n+i)] = 1 for all i, j, k.

The rank is the minimum r such that T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c where u_c is in R^{n*m}, etc.

For 2x4 * 4x6: n=2, m=4, p=6.
Schoolbook rank = n*m*p = 48.

We want to find a rank-minimal decomposition. The known lower bound for n x m * m x p is max(n*p, m*p, n*m) = max(12, 24, 8) = 24.
But the actual minimal rank is likely higher.

For small cases, we can try to find decompositions with rank < 48 using linear algebra / optimization.

Approach: 
1. Start with schoolbook (rank 48).
2. Try to reduce rank by finding linear dependencies or using known algorithms.
3. Use a randomized approach: try to find a rank-r decomposition by solving the system of equations.

For a given rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r) such that:
  sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k

This is a system of n*m*p = 48 equations in (n*m + m*p + p*n) * r = (8 + 24 + 12) * r = 44r unknowns.

For r < 48, this is a nonlinear system. We can try to solve it using numerical methods.

Strategy:
- Try ranks from 24 up to 47.
- For each rank, use a randomized initialization and gradient descent or Newton's method.
- Alternatively, use the fact that this is a polynomial system and try to find a solution.

Given the time constraint (150s), I'll try a few ranks with randomized starts and a simple optimization.

Let me implement a basic approach:
1. For a target rank r, initialize U, V, W randomly.
2. Define the residual: for each (i,j,k), residual = sum_c U[i*m+j, c]*V[j*p+k, c]*W[k*n+i, c] - 1.
3. Minimize the sum of squares of residuals using gradient descent.
4. If the residual is close to zero, we have a valid decomposition.

But this is a non-convex problem. Let's try it with multiple random restarts.

Actually, let me think about this differently. The matrix multiplication tensor has a known structure. For the case 2x4 * 4x6, maybe there's a known lower rank decomposition.

Let me try a different approach: use the fact that we can express the multiplication using fewer terms if we exploit the structure.

For 2x4 * 4x6, one approach is to use the "border rank" or known decompositions.

Let me just implement the optimization approach and see if we can find a lower rank.

I'll use NumPy for the optimization.
"""

import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def to_list(mat):
        return mat.tolist()
    
    # Try to find lower rank decompositions
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    start_time = time.time()
    deadline = start_time + 120  # Leave 30s margin
    
    # Try ranks from a lower bound up to schoolbook - 1
    # Lower bound: max(n*p, m*p, n*m) = max(12, 24, 8) = 24
    lower_bound = max(n * p, m * p, n * m)
    
    # We'll try a few ranks with randomized optimization
    ranks_to_try = list(range(lower_bound, min(rank_school, lower_bound + 20)))
    
    # Shuffle to try various ranks
    np.random.seed(42)
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        # Try multiple random restarts for this rank
        num_restarts = 5
        success = False
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            # Random initialization
            U = np.random.randn(nm, r) * 0.1
            V = np.random.randn(mp, r) * 0.1
            W = np.random.randn(pn, r) * 0.1
            
            # Flatten the parameters
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(params):
                U_flat = params[:nm * r].reshape(nm, r)
                V_flat = params[nm * r:nm * r + mp * r].reshape(mp, r)
                W_flat = params[nm * r + mp * r:].reshape(pn, r)
                
                # Compute residuals for all (i, j, k)
                residuals = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            idx_w = k * n + i
                            val = np.dot(U_flat[idx_u, :], V_flat[idx_v, :]) * W_flat[idx_w, :]
                            # Wait, this is wrong. The decomposition is:
                            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            # Not (U dot V) * W
                            pass
                
                # Correct computation:
                residuals = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            idx_w = k * n + i
                            val = 0.0
                            for c in range(r):
                                val += U_flat[idx_u, c] * V_flat[idx_v, c] * W_flat[idx_w, c]
                            residuals.append(val - 1.0)
                
                return np.sum(np.array(residuals) ** 2)
            
            # Use L-BFGS-B for optimization
            try:
                result = minimize(objective, params, method='L-BFGS-B', 
                                  options={'maxiter': 100, 'ftol': 1e-10})
                
                if result.fun < 1e-6:
                    U_opt = result.x[:nm * r].reshape(nm, r)
                    V_opt = result.x[nm * r:nm * r + mp * r].reshape(mp, r)
                    W_opt = result.x[nm * r + mp * r:].reshape(pn, r)
                    
                    # Verify
                    max_err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx_u = i * m + j
                                idx_v = j * p + k
                                idx_w = k * n + i
                                val = 0.0
                                for c in range(r):
                                    val += U_opt[idx_u, c] * V_opt[idx_v, c] * W_opt[idx_w, c]
                                max_err = max(max_err, abs(val - 1.0))
                    
                    if max_err < 1e-4:
                        best_U, best_V, best_W = U_opt, V_opt, W_opt
                        best_rank = r
                        success = True
                        break
                        
            except Exception:
                continue
        
        if success:
            # Found a lower rank, break early
            break
    
    # Convert to lists with integer/half-integer entries if possible
    # Round to nearest 0.5 if close
    def round_half(mat):
        rounded = np.round(mat * 2) / 2
        # Check if rounding introduces error
        if np.allclose(mat, rounded, atol=1e-6):
            return rounded
        return mat
    
    U_out = round_half(best_U)
    V_out = round_half(best_V)
    W_out = round_half(best_W)
    
    return {"U": to_list(U_out), "V": to_list(V_out), "W": to_list(W_out)}
# EVOLVE-BLOCK-END
