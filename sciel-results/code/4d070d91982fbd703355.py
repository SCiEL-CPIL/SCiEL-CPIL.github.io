"""
Rank-minimal tensor decomposition for (2,3,6) matrix multiplication.

The multiplication tensor is T[a,b,c,d,i,k] = delta_{a,i} * delta_{b,c} * delta_{d,k}.
This is a Kronecker product of three identity tensors.

We use a constructive approach: fix U to a sparse pattern, then solve for V and W
via linear algebra. The key is that once U is fixed, the equations become:
  sum_c V[c*p+d] * (U[a*m+b] * W[k*n+i]) = delta_{a,i} * delta_{b,c} * delta_{d,k}

For fixed (b, d): the equation over (a, i, k) is:
  sum_c V[b*p+d, c] * U[a*m+b, c] * W[k*n+i, c] = delta_{a,i} * delta_{d,k}

We fix U and W to have specific structures, then solve for V.
"""

import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # The tensor: sum_c U[a*m+b, c] * V[c'*p+d, c] * W[k*n+i, c] = delta_{a,i} delta_{b,c'} delta_{d,k}
    # Note: V is indexed by [c'*p + d] where c' is the "row" of B (which must equal b for nonzero)
    
    # Strategy: We'll try to find a decomposition of rank r < 36.
    # Fix U to have a specific structure, then solve linear system for V and W.
    
    # Key insight: For fixed b, the sub-tensor over (a, c', d, i, k) with delta_{b,c'}
    # means only c'=b contributes. So for each b, we need:
    #   sum_c U[a*m+b, c] * V[b*p+d, c] * W[k*n+i, c] = delta_{a,i} delta_{d,k}
    
    # This is: for each b, the (n*p) x (n*p) matrix M_b where
    #   M_b[k*n+i, a] ... hmm, let me think differently.
    
    # For fixed b: sum_c U[a*m+b, c] * W[k*n+i, c] * V[b*p+d, c] = delta_{a,i} delta_{d,k}
    # Let A_b[a, c] = U[a*m+b, c], B_b[d, c] = V[b*p+d, c], C_b[k*n+i, c] = W[k*n+i, c]
    # Then: sum_c A_b[a,c] * B_b[d,c] * C_b[k*n+i, c] = delta_{a,i} delta_{d,k}
    
    # This means: for each c, the term A_b[a,c] * B_b[d,c] * C_b[k*n+i, c] must sum to delta.
    # If we set C_b[k*n+i, c] = delta_{k,d_c} * delta_{i,a_c} (a "pinned" W), then:
    #   sum_c A_b[a,c] * B_b[d,c] * delta_{k,d_c} * delta_{i,a_c} = delta_{a,i} delta_{d,k}
    # This requires: for each (a,i) with a!=i, sum must be 0; for a=i, delta_{d,k}.
    
    # Schoolbook: one term per (a, d) pair (i.e., per (i, k)), with A_b[a,c]=delta_{a,a_r},
    # B_b[d,c]=delta_{d,d_r}, C_b[k*n+i,c]=delta_{k,d_r}*delta_{i,a_r}.
    # That's n*p = 12 terms per b, total 3*12 = 36.
    
    # To reduce: use terms where C_b is NOT pinned to a single (k,i) but spreads out.
    # If C_b[k*n+i, c] = f_c(k) * g_c(i), then:
    #   sum_c A_b[a,c] * B_b[d,c] * f_c(k) * g_c(i) = delta_{a,i} delta_{d,k}
    
    # For the (a,i) part: sum_c A_b[a,c] * g_c(i) = delta_{a,i}  (this is an n x n matrix equation)
    # For the (d,k) part: sum_c B_b[d,c] * f_c(k) = delta_{d,k}  (this is a p x p matrix equation)
    
    # These are SEPARABLE! So we can use:
    #   A_b * G = I_n  where G[i,c] = g_c(i), so A_b = G^{-1} (if G is square and invertible)
    #   B_b * F = I_p  where F[k,c] = f_c(k), so B_b = F^{-1}
    
    # If we use n*p terms total (shared across b), with G being n x (n*p) and F being p x (n*p):
    #   We need G to have rank n (so A_b = G^+ or similar) and F to have rank p.
    #   But A_b must be n x R and B_b must be p x R where R is the number of terms.
    #   If R = n*p = 12, then G is 2x12, F is 6x12.
    #   A_b = G^{-1} doesn't work directly since G is not square.
    
    # Better: use R = n*p = 12 terms. Set G = I_n ⊗ 1_p (n x n*p), F = 1_n ⊗ I_p (p x n*p).
    # Then A_b[a,c] where c = a'*p + d': A_b[a, a'*p+d'] = delta_{a,a'}
    # B_b[d, a'*p+d'] = delta_{d,d'}
    # C_b[k*n+i, a'*p+d'] = delta_{k,d'} * delta_{i,a'}
    # This is exactly the schoolbook! 12 terms per b, total 36.
    
    # To reduce below 36, we need to SHARE terms across different b values.
    # If term c is used for multiple b values, then U[a*m+b, c] can be nonzero for multiple b.
    
    # Let's try: use R terms total. For each term c, let it be "active" for a set S_c of b values.
    # For b in S_c: U[a*m+b, c], V[b*p+d, c], W[k*n+i, c] are all set.
    # For b not in S_c: U[a*m+b, c] = 0 (so term doesn't contribute to that b).
    
    # The constraint for each b: sum_{c in S_c containing b} U[a*m+b,c] V[b*p+d,c] W[k*n+i,c] = delta_{a,i}delta_{d,k}
    
    # If all terms are active for all b (S_c = {0,1,2} for all c), then we need:
    #   For each b: the same R-term decomposition works, but with different U, V slices.
    #   This means we need R terms such that for EACH b, the (U_b, V_b, W) triple satisfies the identity.
    #   Since W is shared, this is a strong constraint.
    
    # Let me try a different approach: use the Kronecker product structure directly.
    # T = I_n ⊗ I_m ⊗ I_p (as a 6-index tensor, but grouped into 3 pairs).
    # The tensor rank of I_n ⊗ I_m ⊗ I_p is at most n*m*p = 36, and at least max(n*m, m*p, p*n) = 18.
    # Actually, the flattening rank gives a lower bound.
    
    # Let me just try to solve for rank 20 using a numerical approach.
    # Fix U to be a 6x20 matrix with a specific sparse pattern, then solve for V and W.
    
    # Approach: 
    # 1. Fix U to have 2 nonzeros per row (one for each "phase")
    # 2. For each column c of U, the constraint is:
    #    sum_c U[a*m+b,c] V[b*p+d,c] W[k*n+i,c] = delta_{a,i} delta_{b,c'} delta_{d,k}
    #    where c' is the row index of V (must equal b).
    
    # Let me try a constructive rank-20 solution.
    # Use 20 columns. For the 3 b-values, we'll use overlapping sets of columns.
    
    # Actually, let me try the simplest possible improvement: 
    # For the (a,i) part (2x2 identity), use 2 terms: (1,0,0,1) and (0,1,1,0) in (a0,a1,i0,i1).
    # For the (d,k) part (6x6 identity), use 6 terms.
    # For the (b,c') part (3x3 identity), use 3 terms.
    # Total: 2*6*3 = 36. Same as schoolbook.
    
    # To get below 36, I need to find a non-trivial decomposition.
    # Let me try rank 20 by solving a linear system.
    
    # Parameterization: 
    # U is 6xR, V is 18xR, W is 12xR.
    # Equations: for each (a,b,c',d,i,k) with c'=b: 
    #   sum_r U[a*3+b, r] * V[b*6+d, r] * W[k*2+i, r] = delta_{a,i} * delta_{d,k}
    # Number of equations: 2*3*6*2*6 = 432? No, it's 2*3*6*2*6 but with the delta constraints.
    # Actually: for each b (3 values), a (2), d (6), i (2), k (6): 3*2*6*2*6 = 432 equations.
    # But many are trivially satisfied (delta=0 cases).
    
    # Let me just try to find ANY valid decomposition with rank < 36 by random search + linear solve.
    
    best_rank = n * m * p  # 36, schoolbook
    best_result = None
    
    # Generate schoolbook as fallback
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try to find lower rank solutions
    for target_rank in range(20, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            rng = np.random.RandomState(attempt * 7 + target_rank)
            
            # Fix U to a random sparse matrix
            U = np.zeros((n*m, target_rank))
            # Each row of U gets 2-3 nonzeros
            for row in range(n*m):
                nnz = rng.randint(2, 4)
                cols = rng.choice(target_rank, size=nnz, replace=False)
                for c in cols:
                    U[row, c] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
            
            # For each column c, we need to determine V[:, c] and W[:, c]
            # such that: sum_c U[a*m+b, c] * V[b*p+d, c] * W[k*n+i, c] = delta_{a,i} delta_{b,c'} delta_{d,k}
            
            # This is nonlinear. Instead, fix W as well and solve for V.
            # Or: fix U and W, solve for V.
            
            # Fix W randomly
            W = np.zeros((p*n, target_rank))
            for row in range(p*n):
                nnz = rng.randint(2, 4)
                cols = rng.choice(target_rank, size=nnz, replace=False)
                for c in cols:
                    W[row, c] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
            
            # Now solve for V: for each (b, d), V[b*p+d, c] for c=0..R-1
            # Equation: for each (a, i, k): sum_c U[a*m+b, c] * V[b*p+d, c] * W[k*n+i, c] = delta_{a,i} delta_{d,k}
            
            # For fixed b: we have a system in V[b*p+0..p-1, :] (p * R unknowns)
            # with n*n*p = 2*2*6 = 24 equations (for each a, i, k)
            
            # Actually, for fixed b, the unknowns are V[b*p+d, c] for d=0..p-1, c=0..R-1: p*R unknowns.
            # Equations: for each (a, i, k): sum_c sum_d U[a*m+b, c] * V[b*p+d, c] * W[k*n+i, c] * delta_{d,k}
            # = sum_c U[a*m+b, c] * V[b*p+k, c] * W[k*n+i, c] = delta_{a,i}
            
            # So for fixed (b, k): sum_c U[a*m+b, c] * V[b*p+k, c] * W[k*n+i, c] = delta_{a,i} for all a, i.
            # This is: sum_c (U[a*m+b, c] * W[k*n+i, c]) * V[b*p+k, c] = delta_{a,i}
            
            # For fixed (b, k, i): sum_c U[a*m+b, c] * W[k*n+i, c] * V[b*p+k, c] = delta_{a,i} for all a.
            # This is a system in V[b*p+k, c] for c=0..R-1, with n=2 equations (one per a).
            
            # So for each (b, k): we have n*p = 12 equations in R unknowns (V[b*p+k, c] for c=0..R-1).
            # Wait, for each (b, k, i, a): one equation. That's n*n = 4 equations per (b,k) in R unknowns.
            # With R=20 and 4 equations, it's underdetermined. We can find a solution if the system is consistent.
            
            # Let me set up the linear system properly.
            # For fixed b, k: unknowns are x_c = V[b*p+k, c] for c=0..R-1.
            # Equations: for each (a, i): sum_c U[a*m+b, c] * W[k*n+i, c] * x_c = delta_{a,i}
            
            # This is a 4 x R system (for n=2: a in {0,1}, i in {0,1}).
            # We need to find x such that A @ x = rhs where A[a*n+i, c] = U[a*m+b, c] * W[k*n+i, c].
            
            valid = True
            V_sol = np.zeros((m*p, target_rank))
            
            for b in range(m):
                for k in range(p):
                    # Build system: (n*n) x R
                    A_sys = np.zeros((n*n, target_rank))
                    rhs = np.zeros(n*n)
                    for a in range(n):
                        for i in range(n):
                            A_sys[a*n+i, :] = U[a*m+b, :] * W[k*n+i, :]
                            rhs[a*n+i] = 1.0 if a == i else 0.0
                    
                    # Check if system is consistent and find a solution
                    try:
                        # Use least squares
                        x, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, rhs, rcond=None)
                        # Check if solution is valid
                        if np.allclose(A_sys @ x, rhs, atol=1e-10):
                            V_sol[b*p+k, :] = x
                        else:
                            valid = False
                            break
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Verify the full decomposition
                U_check = U
                V_check = V_sol
                W_check = W
                max_err = 0.0
                for a in range(n):
                    for b in range(m):
                        for d in range(p):
                            for i in range(n):
                                for k in range(p):
                                    expected = 1.0 if (a == i and b == b and d == k) else 0.0
                                    # Wait, the condition is delta_{a,i} * delta_{b,c'} * delta_{d,k}
                                    # where c' is the row index in V, which is b. So delta_{b,c'} = 1 always (since c'=b).
                                    # So expected = delta_{a,i} * delta_{d,k}
                                    expected = 1.0 if (a == i and d == k) else 0.0
                                    actual = 0.0
                                    for c in range(target_rank):
                                        actual += U_check[a*m+b, c] * V_check[b*p+d, c] * W_check[k*n+i, c]
                                    max_err = max(max_err, abs(actual - expected))
                
                if max_err < 1e-8:
                    # Found a valid decomposition!
                    best_rank = target_rank
                    best_result = {"U": U.tolist(), "V": V_sol.tolist(), "W": W.tolist()}
                    break  # No need to try more attempts for this rank
            
            if time.time() > deadline - 5:
                break
    
    return best_result
# EVOLVE-BLOCK-END
