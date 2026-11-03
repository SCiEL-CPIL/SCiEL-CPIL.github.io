import time
import numpy as np
from fractions import Fraction
import itertools
import sys

def construct():
    n, m, p = 2, 3, 5
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    start_time = time.time()
    deadline = start_time + 115.0

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        r = len(U[0])
        # U is (nm, r), V is (mp, r), W is (pn, r)
        # Indexing:
        # U[i*m+j, c]
        # V[j*p+k, c]
        # W[k*n+i, c]
        # Sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    def is_half_integer(x):
        # x is a Fraction
        y = x * 2
        return y.denominator == 1

    def try_rank(rank):
        if time.time() > deadline:
            return None
        
        # We use a structured approach.
        # The tensor is T_{ijk} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = 1
        # We can fix some U and V to 1 or -1 to reduce the linear system size.
        # A common technique is to set U and V to be "basis-like" or sparse.
        
        # Let's try a specific structured pattern:
        # Assume U and V have entries in {-1, 0, 1}.
        # We will iterate over a subset of possible U, V matrices? No, that's too large.
        
        # Alternative: 
        # Fix U to be the identity-like matrix for the first 'rank' components?
        # If rank >= 6, we can set U to have 1s on the diagonal of the first 6 columns?
        # Let's try setting U[i*m+j, c] = 1 if c == i*m+j and c < 6, else 0?
        # This forces the first 6 columns to correspond to specific (i,j) pairs.
        
        # Let's generalize:
        # We pick a set of "active" columns for U and V.
        
        # Strategy:
        # 1. Guess U and V are sparse with entries in {-1, 1}.
        # 2. Systematically try patterns where U and V are "block diagonal" or similar?
        
        # Let's try a different structural assumption:
        # The matrix multiplication tensor for 2x3 * 3x5 has rank at most 30.
        # Known lower bounds are around 28-29?
        
        # Let's try to find a solution of rank R by assuming U and V have a specific sparse structure.
        # For each column c, let u_c be a vector in {-1,0,1}^6 and v_c be a vector in {-1,0,1}^15.
        # The equation for (i,j,k) is sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1.
        
        # If we fix u_c and v_c, we have a linear system for w_c.
        # Since w_c must be half-integers, we can solve the linear system over rationals and check.
        
        # To make this tractable, we restrict the search space of u_c and v_c.
        # We can use the fact that the tensor is symmetric-ish? No, it's not fully symmetric.
        
        # Let's try a heuristic search with exact arithmetic.
        # We will generate random sparse U and V, solve for W exactly, and check.
        
        num_trials = 5000
        for t in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Generate random U, V
            # Bias towards sparsity
            density = 0.2 + (t % 10) * 0.05 # vary density
            
            U = np.zeros((nm, rank), dtype=int)
            V = np.zeros((mp, rank), dtype=int)
            
            # Randomly assign -1, 0, 1
            for c in range(rank):
                # U column
                mask_u = np.random.rand(nm) < density
                U[mask_u, c] = np.random.choice([-1, 1], size=mask_u.sum())
                
                # V column
                mask_v = np.random.rand(mp) < density
                V[mask_v, c] = np.random.choice([-1, 1], size=mask_v.sum())
            
            # Build linear system for W
            # Variables: W[k*n+i, c] for all k,i,c
            # Total variables: pn * rank
            # Equations: n*m*p = 30
            # Wait, the equations are coupled across c?
            # No, the sum is over c.
            # Sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # This is a linear system in W.
            # The matrix A has size 30 x (10 * rank).
            # A[row, col] where row corresponds to (i,j,k) and col corresponds to (k',i',c)
            # A[(i,j,k), (k',i',c)] = U[i*m+j, c] * V[j*p+k, c] if k==k' and i==i' else 0?
            # No, W index is k*n+i. The term is W[k*n+i, c].
            # So for a fixed (i,j,k), the variable involved is W[k*n+i, c] for each c.
            # So the system is:
            # For each (i,j,k): sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1
            
            # This means the variable W[k*n+i, c] only appears in the equation for (i,j,k) where the indices match?
            # Let's check:
            # The term is W[k*n+i, c]. The index of W is (k, i).
            # The equation is for (i, j, k).
            # So W[k*n+i, c] appears in the equation for (i, j, k).
            # Does it appear in any other equation?
            # The equation for (i', j', k') involves W[k'*n+i', c].
            # So W[k*n+i, c] appears in the equation for (i, j, k) for ALL j?
            # Yes! For a fixed (i, k), the variable W[k*n+i, c] appears in the equations for all j (j=0,1,2).
            # So the equations are coupled through j.
            
            # Let's re-verify the tensor definition.
            # C[i,k] = sum_j A[i,j] B[j,k]
            # T_{i,j,k} is the coefficient of A[i,j] B[j,k] in C[i,k].
            # So T_{i,j,k} = 1 if we consider the full tensor.
            # Decomposition: T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}
            
            # Equation for (i,j,k):
            # sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = 1
            
            # Variable W_{k,i,c} depends on k and i.
            # It appears in the sum for a specific (i,k) pair.
            # But the sum is over c.
            # So for a fixed (i,k), we have:
            # sum_j ( sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} ) ? No.
            # The equation is for each triple (i,j,k).
            
            # So we have 30 equations.
            # Variables are W_{k,i,c}.
            # There are 10 * rank variables.
            
            # Let's build the matrix A.
            # Rows: 30 (indexed by i, j, k)
            # Cols: 10 * rank (indexed by (k, i, c))
            
            # A[row(i,j,k), col(k',i',c')] = 
            # If k==k' and i==i' and c==c':
            #   Then the term is U_{i,j,c} * V_{j,k,c} * W_{k,i,c}.
            #   So coefficient is U_{i,j,c} * V_{j,k,c}.
            # Else 0.
            
            # So A is block diagonal with respect to (k,i)?
            # No, the row is (i,j,k). The col is (k,i,c).
            # For a fixed row (i,j,k), the non-zero columns are those with k'=k, i'=i.
            # There are 'rank' such columns (for c=0..rank-1).
            # So each row has 'rank' non-zeros.
            
            # We can solve this system.
            # Since the system is underdetermined (10*rank vars, 30 eqs), we have many solutions.
            # We need a solution where W entries are half-integers.
            
            # We can use least squares to find the minimum norm solution, then check if it's close to half-integers?
            # Or we can fix some W variables to 0 or 1 to make it determined?
            
            # Let's try:
            # 1. Solve A W = 1 using np.linalg.lstsq.
            # 2. Check if the solution is close to half-integers.
            # 3. If not, try adding constraints?
            
            # To improve chances, we can also randomize the "basis" of the solution space.
            # But lstsq gives a specific solution.
            
            # Let's try this simple approach first.
            
            A = np.zeros((30, 10 * rank))
            b = np.ones(30)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row_idx = i * m * p + j * p + k
                        # Columns for this row: k, i, c
                        for c in range(rank):
                            col_idx = (k * n + i) * rank + c
                            A[row_idx, col_idx] = U[i * m + j, c] * V[j * p + k, c]
            
            # Solve
            try:
                W_sol, residuals, rank_svd, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            W_sol = W_sol.reshape(10, rank)
            
            # Check half-integer
            W_check = W_sol * 2
            if np.allclose(W_check, np.round(W_check), atol=1e-5):
                W_list = (np.round(W_check) / 2).tolist()
                U_list = U.tolist()
                V_list = V.tolist()
                
                # Verify exactly
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
            
            # Optional: If close, snap to nearest half-integer and verify
            W_snapped = np.round(W_check) / 2
            if np.allclose(W_sol, W_snapped, atol=1e-3):
                W_list = W_snapped.tolist()
                U_list = U.tolist()
                V_list = V.tolist()
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}

        return None

    # Try ranks from low to high
    # Schoolbook is 30.
    # Lower bound is likely 28 or 29.
    for r in range(28, 31):
        if time.time() > deadline:
            break
        res = try_rank(r)
        if res:
            return res
            
    return schoolbook()

if __name__ == "__main__":
    construct()
