import numpy as np
import time
import sys
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Schoolbook solution as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Pre-compute indices for equations to speed up loop
    # Equation index: (i, k, i2, k2, j, j2)
    # We need to map this to row index in A and value in b
    
    # Number of equations: n*n*p*p*m*m = 9*49*16 = 7056
    num_eqs = n*n*p*p*m*m
    
    # Pre-compute U indices and V indices for each equation
    # For a fixed (i, k), we vary i2, k2, j, j2
    # Coefficient for W[k*n+i, r] in equation (i2, k2, j, j2) is U[i2*m+j, r] * V[j2*p+k2, r]
    
    # Let's restructure the search to be more efficient.
    # Instead of building the full A matrix in a nested Python loop every time,
    # we can pre-compute the structure.
    # A[row, col] where col = r*21 + (k*n+i)
    # row is determined by (i2, k2, j, j2) for a fixed (i, k)? No, row is global.
    
    # Let's just optimize the inner loop construction.
    # The current loop is very slow in Python.
    # We can vectorize the construction of A for a given U, V.
    
    # A is (7056, 21*R)
    # b is (7056,)
    
    # Let's try to construct A using Kronecker products or broadcasting.
    # This is tricky because the mapping of indices is complex.
    
    # Alternative: Use a sparse solver? Or just accept the speed limit and try fewer, better trials?
    # Or, use a different initialization strategy.
    
    # Let's try to optimize the construction of A.
    # For each r, the column block is 21*R.
    # Actually, let's just speed up the Python loop by pre-computing the non-zero pattern.
    # But the values depend on U and V.
    
    # Let's try a different approach:
    # Instead of random U, V, use a structured initialization?
    # Or, use a smaller subset of equations?
    # The system is overdetermined (7056 eqs, ~1600 vars).
    # We can randomly select a subset of equations to make it square or underdetermined,
    # solve, and then verify against ALL equations.
    # This might be faster to solve.
    
    # Let's stick to the full system but optimize the construction.
    # We can pre-compute the indices.
    
    # Let's try to reduce the number of trials but make them smarter.
    # Or, use a different solver.
    
    # Let's try to vectorize the A construction.
    # For a fixed r, the column is W[:, r].
    # A[:, r*21 + (k*n+i)] = sum over (i2, k2, j, j2) of ...?
    # No, A is linear in W.
    # The equation is: sum_r A[row, col_r] * W[col_r] = b[row]
    # where col_r corresponds to the r-th rank component.
    
    # Let's rewrite the loop to be slightly faster.
    # We can pre-compute the (i2, k2, j, j2) indices.
    
    idx_list = []
    for i in range(n):
        for k in range(p):
            for i2 in range(n):
                for k2 in range(p):
                    for j in range(m):
                        for j2 in range(m):
                            idx_list.append((i, k, i2, k2, j, j2))
                            
    # Now, for each trial:
    for R in range(rank_sb - 1, 69, -1):
        if time.time() > deadline - 5.0:
            break
            
        max_trials = 20 # Reduced trials, but maybe faster construction?
        found = False
        
        for trial in range(max_trials):
            if time.time() > deadline - 5.0:
                break
                
            U = np.random.randint(-1, 2, size=(n*m, R))
            V = np.random.randint(-1, 2, size=(m*p, R))
            
            A = np.zeros((num_eqs, 21*R))
            b = np.zeros(num_eqs)
            
            # Vectorized construction?
            # For each equation (i, k, i2, k2, j, j2):
            # b = 1 if i2==i, k2==k, j==j2 else 0
            # A[row, r*21 + (k*n+i)] = U[i2*m+j, r] * V[j2*p+k2, r]
            
            # This is still hard to vectorize directly without iterating.
            # Let's try to use np.einsum or similar?
            # Or just iterate but with pre-computed indices.
            
            for idx, (i, k, i2, k2, j, j2) in enumerate(idx_list):
                if i2 == i and k2 == k and j == j2:
                    b[idx] = 1.0
                else:
                    b[idx] = 0.0
                    
                # A[idx, :] = U[i2*m+j, :] * V[j2*p+k2, :] ?
                # Wait, the column index is r*21 + (k*n+i).
                # So for a fixed (i, k), the columns are interleaved with other (i, k) pairs.
                # We need to place U[i2*m+j, r] * V[j2*p+k2, r] into A[idx, r*21 + (k*n+i)].
                
                # We can do this by reshaping.
                # Let's create a temporary matrix M of shape (R, 21)
                # M[r, k*n+i] = U[i2*m+j, r] * V[j2*p+k2, r]
                # Then A[idx, :] = M.flatten()
                
                # This is still a loop over R if we do it naively.
                # But we can vectorize the inner R loop.
                
                # prod = U[i2*m+j, :] * V[j2*p+k2, :]  # shape (R,)
                # We need to scatter this into A[idx, :] at positions r*21 + (k*n+i)
                
                # We can pre-compute the column indices for each (i, k).
                # col_indices[i, k] = np.array([r*21 + (k*n+i) for r in range(R)])
                
                # But R changes. So we need to do this inside the R loop.
                
                # Let's pre-compute the base indices for W.
                # w_indices = np.arange(21*R).reshape(R, 21)
                # Then for a specific (i, k), the columns are w_indices[:, k*n+i]
                
                w_indices = np.arange(21*R).reshape(R, 21)
                cols_for_ik = w_indices[:, k*n+i]
                
                prod = U[i2*m+j, :] * V[j2*p+k2, :]
                A[idx, cols_for_ik] = prod
            
            try:
                w, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            if len(residuals) > 0 and residuals[0] > 1e-9:
                continue
                
            w2 = w * 2
            w2_rounded = np.round(w2)
            if np.allclose(w2, w2_rounded, atol=1e-9):
                W_sol = w.reshape((21, R))
                
                if R < best_rank:
                    best_rank = R
                    best_U = U
                    best_V = V
                    best_W = W_sol
                    found = True
                    break
        
        if found:
            pass

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
