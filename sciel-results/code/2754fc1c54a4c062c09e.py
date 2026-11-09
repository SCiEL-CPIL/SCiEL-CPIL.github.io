"""Improved program: search for lower-rank exact tensor decomposition of 2x5 * 5x6 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]

# Deadline guard
_START_TIME = time.time()
TIME_LIMIT = 140.0  # seconds, leaving 10s margin

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook decomposition as baseline
    rank_school = n * m * p
    U = np.zeros((nm, rank_school))
    V = np.zeros((mp, rank_school))
    W = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_school
    
    # Try to find lower rank decompositions using randomized search
    # The tensor T[i,j,k] = delta[i,k] for matrix mult (A[i,:] @ B[:,k])
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,k}
    
    # Strategy: use the fact that for fixed i,k, we need sum_j,c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 if i==k else 0
    # This is a system of equations. Let's try small ranks.
    
    # For rank r, we have r * (nm + mp + pn) variables and n*m*p equations (but many are zero)
    # Non-zero equations: n*p of them (one for each (i,k) pair where we need delta_{i,k})
    # Actually all n*m*p entries of the tensor are specified, but many are 0.
    
    # Let's try a structured approach: since n=2, p=6, we have 12 constraints that are non-trivial
    # (the ones where i != k need to sum to 0, and i==k need to sum to 1)
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # Actually the tensor rank is at least max(nm, mp, pn) = 30
    
    # Let's try to find rank 30, 31, ... decompositions
    
    def try_rank(r, num_trials=100):
        nonlocal best_U, best_V, best_W, best_rank
        if r >= best_rank:
            return False
        if _time_remaining() < 1.0:
            return False
        
        # Randomized approach: pick random U, V, then solve for W
        # For each trial:
        # 1. Generate random U (nm x r) and V (mp x r) with small integer/half-integer entries
        # 2. For each (i,k), we need sum_j,c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta_{i,k}
        # This gives us a linear system in W entries.
        
        for trial in range(num_trials):
            if _time_remaining() < 1.0:
                return False
            
            # Generate random sparse U and V with entries in {-1, 0, 1}
            # This helps with exact arithmetic
            U_trial = np.zeros((nm, r))
            V_trial = np.zeros((mp, r))
            
            # Each column of U and V should have some non-zero entries
            for c in range(r):
                # Pick a random subset of rows for U
                u_rows = np.random.choice(nm, size=min(r, nm), replace=False)
                for row in u_rows:
                    U_trial[row, c] = np.random.choice([-1, 1])
                
                v_rows = np.random.choice(mp, size=min(r, mp), replace=False)
                for row in v_rows:
                    V_trial[row, c] = np.random.choice([-1, 1])
            
            # Now solve for W
            # For each (i, k): sum_{j,c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,k}
            # This is: sum_c (sum_j U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = delta_{i,k}
            
            # Build linear system: for each (i,k), equation is sum_c A_{(i,k),c} * W[k*n+i, c] = delta_{i,k}
            # But W has pn entries, indexed by (k*n+i) for i in range(n), k in range(p)
            # Wait, W is indexed by [k*n + i] for i in 0..n-1, k in 0..p-1
            # So W has p*n = 12 entries.
            
            # For each (i,k), the equation involves W entries at positions k*n + i
            # But the sum is over c, and for each c, the coefficient is sum_j U[i*m+j, c] * V[j*p+k, c]
            
            # Let's build the system:
            # Variables: W[0], W[1], ..., W[pn-1]
            # For each (i,k): sum_c coeff[i,k,c] * W[k*n+i, c] ... 
            # Wait, that's not right. W is a matrix of shape (pn, r).
            # W[k*n+i, c] for c in range(r).
            
            # So for fixed (i,k), the equation is:
            # sum_c (sum_j U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = delta_{i,k}
            
            # This means: for fixed (i,k), we have r terms involving W[k*n+i, 0], ..., W[k*n+i, r-1]
            # But W[k*n+i, c] appears in multiple equations (for different (i',k') where k'*n+i' = k*n+i)
            # Actually k*n+i uniquely determines (k,i) since i < n and k < p.
            # So W[k*n+i, c] only appears in the equation for (i,k).
            
            # This means the system decouples! For each (i,k), we have r unknowns W[k*n+i, 0..r-1]
            # and one equation. That's underdetermined unless r=1.
            
            # Hmm, that's a problem. Let me reconsider.
            
            # Actually, I think I'm confusing myself. Let me re-derive.
            # The tensor decomposition says:
            # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # For matrix multiplication, T[i,j,k] = delta_{i,k} (independent of j)
            # Wait no, that's not right either. Let me think again.
            
            # Matrix multiplication: C = A @ B where A is n x m, B is m x p, C is n x p
            # C[i,k] = sum_j A[i,j] * B[j,k]
            
            # The tensor for this operation: we want to compute all C[i,k] from A and B.
            # The "inputs" are A[i,j] for i in 0..n-1, j in 0..m-1 and B[j,k] for j in 0..m-1, k in 0..p-1
            # The "outputs" are C[i,k] for i in 0..n-1, k in 0..p-1
            
            # The tensor T has indices (i, j, k) where:
            # - i indexes rows of A (and rows of C)
            # - j indexes columns of A / rows of B (the contraction index)
            # - k indexes columns of B (and columns of C)
            
            # T[i,j,k] = 1 for all i,j,k? No, that's not right either.
            
            # Actually, the standard way: the matrix multiplication tensor T is a 3-tensor with T[i,j,k] = 1
            # for all i, j, k. Then C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k] = sum_j A[i,j] * B[j,k]
            
            # So T[i,j,k] = 1 for all i, j, k!
            
            # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
            
            # And the output is: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
            # Wait, that's not quite right either. Let me be more careful.
            
            # The standard formulation: we have bilinear forms.
            # For each c, we compute:
            #   U_c = sum_j U[i*m+j, c] * A[i,j]  (for each i)
            #   V_c = sum_j V[j*p+k, c] * B[j,k]  (for each k)
            #   C[i,k] += W[k*n+i, c] * U_c * V_c
            
            # So C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
            
            # Expanding: C[i,k] = sum_c W[k*n+i, c] * sum_{j1,j2} U[i*m+j1, c] * V[j2*p+k, c] * A[i,j1] * B[j2,k]
            
            # For this to equal sum_j A[i,j] * B[j,k], we need:
            # sum_c W[k*n+i, c] * sum_{j1,j2} U[i*m+j1, c] * V[j2*p+k, c] * delta_{j1,j2} ... 
            # No, that's not how it works. The correct condition is:
            
            # sum_c W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c] = delta_{j1, j2} for all i, k, j1, j2
            
            # Wait, that's not right either. Let me look at this more carefully.
            
            # C[i,k] = sum_c W[k*n+i, c] * (sum_{j1} U[i*m+j1, c] * A[i,j1]) * (sum_{j2} V[j2*p+k, c] * B[j2,k])
            
            # = sum_c W[k*n+i, c] * sum_{j1, j2} U[i*m+j1, c] * V[j2*p+k, c] * A[i,j1] * B[j2,k]
            
            # We want this to equal sum_j A[i,j] * B[j,k] for all A, B.
            
            # So we need: sum_c W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c] = delta_{j1, j2}
            
            # for all i, k, j1, j2.
            
            # This is the correct condition! Let me verify with the schoolbook:
            # U[i*m+j, c] = 1 if c = i*m*p + j*p + j' ... no, let me use the original code.
            # In the original: U[i*m+j, column] = 1, V[j*p+k, column] = 1, W[k*n+i, column] = 1
            # where column = i*m*p + j*p + k (approximately)
            
            # For a given column c corresponding to (i0, j0, k0):
            # U[i*m+j, c] = 1 if i=i0 and j=j0
            # V[j*p+k, c] = 1 if j=j0 and k=k0
            # W[k*n+i, c] = 1 if k=k0 and i=i0
            
            # So sum_c W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c]
            # = sum over c where all three are 1
            # = 1 if there exists c with i=i0=i, j1=j0, j2=j0, k=k0=k
            # = 1 if j1=j2 (and the c exists, which it does when i=i0, k=k0, j1=j2)
            # Wait, this gives 1 if j1=j2, regardless of i,k. That's not delta_{j1,j2} in the right sense.
            
            # Hmm, I think I'm overcomplicating this. Let me just use the condition:
            # For all i, k, j1, j2: sum_c W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c] = delta_{j1, j2}
            
            # Let me re-examine. The number of equations is n * p * m * m = 2*6*5*5 = 300
            # The number of variables is r * (nm + mp + pn) = r * (10 + 30 + 12) = 52r
            
            # For r=30: 52*30 = 1560 variables, 300 equations. Very underdetermined.
            # For r=10: 520 variables, 300 equations. Still underdetermined.
            # For r=6: 312 variables, 300 equations. Close!
            # For r=5: 260 variables, 300 equations. Overdetermined.
            
            # So the minimum possible rank might be around 6-10.
            
            # Let's try to solve this as a linear system for small r.
            # For fixed r, we have 52r unknowns and 300 equations.
            # We can use least squares or check if a solution exists.
            
            # Actually, since we need exact integer/half-integer solutions, let's use a different approach.
            
            # For each r, try to find U, V, W such that the condition holds.
            # This is a system of bilinear equations (trilinear in U, V, W).
            
            # Let's try a simpler approach: fix U and V, then solve for W.
            # For fixed U, V, the condition becomes:
            # For all i, k, j1, j2: sum_c W[k*n+i, c] * (U[i*m+j1, c] * V[j2*p+k, c]) = delta_{j1, j2}
            
            # This is linear in W! For each (i, k), we have m*m = 25 equations in r unknowns (W[k*n+i, 0..r-1]).
            # Total: n*p*m*m = 300 equations in n*p*r = 12r unknowns.
            
            # For r=30: 300 equations in 360 unknowns. Underdetermined.
            # For r=25: 300 equations in 300 unknowns. Square system!
            # For r=24: 300 equations in 288 unknowns. Overdetermined.
            
            # So let's try r from 25 upward, fix random U, V, and solve for W.
            
            for r in range(25, best_rank):
                if _time_remaining() < 1.0:
                    return False
                if r >= best_rank:
                    break
                
                num_trials_r = max(10, int(_time_remaining() / 2))
                for trial in range(num_trials_r):
                    if _time_remaining() < 0.5:
                        break
                    
                    # Generate random U and V with entries in {-1, 0, 1}
                    U_t = np.zeros((nm, r))
                    V_t = np.zeros((mp, r))
                    
                    for c in range(r):
                        # Sparse random
                        u_idx = np.random.choice(nm, size=min(3, nm), replace=False)
                        for idx in u_idx:
                            U_t[idx, c] = np.random.choice([-1, 1])
                        
                        v_idx = np.random.choice(mp, size=min(3, mp), replace=False)
                        for idx in v_idx:
                            V_t[idx, c] = np.random.choice([-1, 1])
                    
                    # Build linear system for W
                    # Unknowns: W[k*n+i, c] for i in 0..n-1, k in 0..p-1, c in 0..r-1
                    # Total: p*n*r = 12*r unknowns
                    # Equations: for each i, k, j1, j2: sum_c W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c] = delta_{j1,j2}
                    
                    num_eq = n * p * m * m
                    A_mat = np.zeros((num_eq, p * n * r))
                    b_vec = np.zeros(num_eq)
                    
                    eq_idx = 0
                    for i in range(n):
                        for k in range(p):
                            for j1 in range(m):
                                for j2 in range(m):
                                    rhs = 1.0 if j1 == j2 else 0.0
                                    b_vec[eq_idx] = rhs
                                    for c in range(r):
                                        col_idx = (k * n + i) * r + c
                                        A_mat[eq_idx, col_idx] = U_t[i * m + j1, c] * V_t[j2 * p + k, c]
                                    eq_idx += 1
                    
                    # Solve least squares
                    try:
                        W_flat, residuals, rank_A, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        W_t = W_flat.reshape(p * n, r)
                        
                        # Check if the solution is exact (within tolerance)
                        # Reconstruct and check
                        max_err = 0.0
                        for i in range(n):
                            for k in range(p):
                                for j1 in range(m):
                                    for j2 in range(m):
                                        val = sum(W_t[k*n+i, c] * U_t[i*m+j1, c] * V_t[j2*p+k, c] for c in range(r))
                                        expected = 1.0 if j1 == j2 else 0.0
                                        max_err = max(max_err, abs(val - expected))
                        
                        if max_err < 1e-9:
                            # Found a valid decomposition!
                            best_U, best_V, best_W = U_t, V_t, W_t
                            best_rank = r
                            return True
                    except np.linalg.LinAlgError:
                        continue
            
            break  # If we've tried all r values, break out
        
        return False
    
    # Try to find lower rank
    try_rank(25)
    
    # Convert to lists for output
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
