"""Search-based solver for rank-minimal exact tensor decomposition of 3x5x6 matmul."""
import time
import numpy as np
import itertools
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s kill
    best_rank = n * m * p
    best_result = None

    # Schoolbook baseline (always valid)
    best_result = _schoolbook(n, m, p, best_rank)

    # Verify the baseline
    if _verify(n, m, p, best_result):
        pass  # baseline is valid

    # Strategy: Use structured construction based on block decomposition
    # For 3x5x6, we can try to construct decompositions with shared structure
    # across the 3 output rows.

    # Approach 1: Try to find a decomposition where U and V have a block structure
    # that allows sharing of computation across i=0,1,2
    
    def try_structured_rank(r, time_budget):
        """Try to find a decomposition of rank r via structured construction."""
        nonlocal best_rank, best_result
        if time.time() > time_budget:
            return False
        
        # Strategy: Use the fact that we have 3 output rows (i=0,1,2)
        # Each row i needs to compute: result[i,k] = sum_j A[i,j] * B[j,k]
        # This is like 3 separate 1x5x6 matmuls, but we can share B-side structure
        
        # For a 1x5x6 matmul, the tensor is T[j,k] = A[j] * B[j,k]
        # We need: sum_c U[j,c] * V[j,k,c] * W[k,c] = A[j] * B[j,k]
        # Wait, that's not quite right. Let me think more carefully.
        
        # The matmul tensor for (i,j,k) is: M[i,j,k] = delta (i.e., the tensor 
        # such that C[i,k] = sum_j A[i,j] * B[j,k] = sum_{j,c} A[i,j] * (sum_c U[i,j,c]*V[j,k,c]) * W[k,i,c]... 
        
        # Actually, the standard decomposition is:
        # C[i,k] = sum_j A[i,j] B[j,k] = sum_c (sum_j A[i,j] U[i,j,c]) * (sum_k' ... ) 
        # No, that's not right either.
        
        # The correct form: We want factors U, V, W such that:
        # For all (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # This means: C[i,k] = sum_j A[i,j] B[j,k] = sum_{j,c} A[i,j] B[j,k] * (sum over nothing)
        # Actually: C[i,k] = sum_j A[i,j] B[j,k]
        # We want: C[i,k] = sum_c W[k*n+i, c] * (sum_j A[i,j] U[i*m+j, c]) * (sum over... )
        
        # Let me be more careful. The tensor T[i,j,k] = 1 for all (i,j,k) in the 
        # "multiplication tensor" sense. We want:
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        
        # So for the actual computation:
        # C[i,k] = sum_j A[i,j] B[j,k] = sum_j A[i,j] B[j,k] * 1
        #        = sum_j A[i,j] B[j,k] * sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        #        = sum_c W[k,i,c] * (sum_j A[i,j] U[i,j,c]) * (sum_j' B[j',k] V[j',k,c])
        # Wait, that doesn't work because j and j' are different indices.
        
        # The correct identity is:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # We decompose as:
        # C[i,k] = sum_c W[k*n+i, c] * P_c[i] * Q_c[k]
        # where P_c[i] = sum_j A[i,j] U[i*m+j, c]  (this is a linear combination of A's row i)
        # and Q_c[k] = sum_j B[j,k] V[j*p+k, c]  (this is a linear combination of B's column k)
        
        # For this to work for ALL A and B, we need:
        # sum_c W[k*n+i, c] * (sum_j A[i,j] U[i*m+j, c]) * (sum_j B[j,k] V[j*p+k, c]) = sum_j A[i,j] B[j,k]
        
        # Expanding: sum_c W[k*n+i, c] * sum_j A[i,j] U[i*m+j, c] * sum_j' B[j',k] V[j'*p+k, c]
        # = sum_{j,j'} A[i,j] B[j',k] * sum_c W[k*n+i, c] U[i*m+j, c] V[j'*p+k, c]
        
        # For this to equal sum_j A[i,j] B[j,k] for all A,B, we need:
        # sum_c W[k*n+i, c] U[i*m+j, c] V[j'*p+k, c] = delta_{j,j'}
        
        # So the constraint is: for all (i,j,j',k):
        # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] = delta_{j,j'}
        
        # This is the correct constraint! Let me verify with schoolbook:
        # In schoolbook, U[i*m+j, c] = 1 only when c corresponds to (i,j,k)
        # V[j*p+k, c] = 1 only when c corresponds to (i,j,k)  
        # W[k*n+i, c] = 1 only when c corresponds to (i,j,k)
        # So sum_c ... = 1 if j=j' and the same (i,j,k) triple, else 0.
        # Yes, that gives delta_{j,j'}.
        
        # So the correct constraint is:
        # For all (i, j, j', k): sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] = delta_{j,j'}
        
        # This is a system of equations. Let's try to solve it.
        
        # For fixed i and k, we have m x m = 25 equations (for j, j' pairs)
        # The unknowns are the rank-r vectors.
        
        # Strategy: Fix U and V to be sparse/structured, then solve for W.
        # Or fix U and W, solve for V. Etc.
        
        # Let's try: fix U to have a specific structure, then for each (i,k),
        # we have a linear system to solve for W[k*n+i, :].
        
        # For fixed (i,k): sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] = delta_{j,j'}
        # Let A_{j,j',c} = U[i*m+j, c] * V[j'*p+k, c]
        # Then: sum_c A_{j,j',c} * W[k*n+i, c] = delta_{j,j'}
        
        # This is: (A * w)_{j,j'} = delta_{j,j'} where w = W[k*n+i, :]
        # A is a (m*m) x r matrix, w is r-vector, result is m*m vector.
        
        # If r >= m*m = 25, we can potentially solve this.
        # But we want r < 90, so r could be as low as ~25-30.
        
        # Let's try: for each (i,k), we need to find w_{i,k} (length r) such that
        # the linear combination of rank-1 matrices U[i*m+j,:] * V[j'*p+k,:]^T 
        # (weighted by w_{i,k}[c]) gives the identity matrix.
        
        # This means: sum_c w_{i,k}[c] * outer(U[i*m+j,:], V[j'*p+k,:]) = I_m
        # Wait, that's not quite right. Let me re-derive.
        
        # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] = delta_{j,j'}
        # Let u_j = U[i*m+j, :] (length r)
        # Let v_{j'} = V[j'*p+k, :] (length r)  
        # Let w = W[k*n+i, :] (length r)
        # Then: sum_c u_j[c] * v_{j'}[c] * w[c] = delta_{j,j'}
        # This is: u_j^T * (v_{j'} * w) ... no.
        # It's: sum_c (u_j[c] * v_{j'}[c]) * w[c] = <u_j * v_{j'}, w> where * is element-wise
        # So: <u_j * v_{j'}, w> = delta_{j,j'}
        
        # For fixed (i,k), we have m^2 = 25 equations in r unknowns (the components of w).
        # The "equation matrix" has rows indexed by (j,j') and columns by c:
        # M_{(j,j'), c} = U[i*m+j, c] * V[j'*p+k, c]
        # We need: M * w = delta_vector (length m^2)
        
        # If M has full row rank (rank = min(r, m^2)), we can solve.
        # If r >= m^2 = 25, we might be able to find a solution.
        
        # Strategy: 
        # 1. Choose U and V randomly (sparse, half-integer)
        # 2. For each (i,k), check if the system M_{i,k} * w = delta is consistent
        # 3. If yes for all (i,k), we have a valid decomposition
        
        # Let's implement this.
        
        def check_and_solve(n, m, p, r, U, V, max_time):
            """Check if given U,V allow solving for W, and return W if so."""
            start = time.time()
            W = np.zeros((p * n, r))
            for i in range(n):
                for k in range(p):
                    if time.time() > max_time:
                        return None
                    # Build the equation matrix M for this (i,k)
                    # M_{(j,j'), c} = U[i*m+j, c] * V[j'*p+k, c]
                    M = np.zeros((m * m, r))
                    for j in range(m):
                        for jp in range(m):
                            M[j * m + jp, :] = U[i * m + j, :] * V[jp * p + k, :]
                    # Right-hand side: delta_{j,j'}
                    rhs = np.zeros(m * m)
                    for j in range(m):
                        rhs[j * m + j] = 1.0
                    # Solve M * w = rhs
                    # Use least squares if r < m*m, or direct solve if r >= m*m
                    try:
                        if r >= m * m:
                            # Use lstsq to find a solution
                            w, residuals, rank, sv = np.linalg.lstsq(M, rhs, rcond=None)
                            # Check if the solution is exact
                            if np.linalg.norm(M @ w - rhs) > 1e-9:
                                return None
                        else:
                            # r < m*m: need to check consistency
                            # Use lstsq and check residual
                            w, residuals, rank, sv = np.linalg.lstsq(M, rhs, rcond=None)
                            if len(residuals) > 0 and residuals[0] > 1e-9:
                                return None
                            if np.linalg.norm(M @ w - rhs) > 1e-9:
                                return None
                    except np.linalg.LinAlgError:
                        return None
                    W[k * n + i, :] = w
            return W
        
        # Try different rank values from low to high
        # Start from a reasonable lower bound
        # For m=5, we need at least m^2=25 per (i,k) pair, but W is shared across i,k
        # Actually, W has p*n = 18 rows, each of length r.
        # The total "degrees of freedom" in W is 18*r.
        # We have n*p*m^2 = 3*6*25 = 450 equations.
        # But they're coupled through U and V.
        
        # Let's try ranks from 30 to 89
        for r in range(30, min(90, best_rank)):
            if time.time() > deadline - 10:
                break
            
            # Try a few random U,V configurations
            num_trials = 50
            for trial in range(num_trials):
                if time.time() > deadline - 10:
                    break
                
                # Generate sparse U and V with half-integer entries
                U = np.zeros((n * m, r))
                V = np.zeros((m * p, r))
                
                # Make U sparse: each row has a few non-zeros
                for idx in range(n * m):
                    num_nz = random.randint(1, 3)
                    cols = random.sample(range(r), num_nz)
                    for c in cols:
                        U[idx, c] = random.choice([0.5, 1.0, -0.5, -1.0, 1.5, -1.5])
                
                for idx in range(m * p):
                    num_nz = random.randint(1, 3)
                    cols = random.sample(range(r), num_nz)
                    for c in cols:
                        V[idx, c] = random.choice([0.5, 1.0, -0.5, -1.0, 1.5, -1.5])
                
                # Try to solve for W
                W = check_and_solve(n, m, p, r, U, V, deadline - 15)
                if W is not None:
                    # Verify the full decomposition
                    if _verify_numpy(n, m, p, U, V, W, r):
                        result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        if r < best_rank:
                            best_rank = r
                            best_result = result
                        return True
        
        return False
    
    def _verify_numpy(n, m, p, U, V, W, r):
        """Verify decomposition using numpy."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(W[k*n+i, :], U[i*m+j, :] * V[j*p+k, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try structured construction
    try_structured_rank(best_rank, deadline - 20)
    
    # If we haven't improved, try a different strategy: 
    # Use the fact that we can pad to a larger problem with known decomposition
    # For example, 4x4x4 has rank 7 (Strassen-like). Can we embed 3x5x6 into 4x4x4?
    # Not directly, but we can try block constructions.
    
    # Alternative: try to find a decomposition by solving a linear system
    # where we fix some entries and solve for others
    
    # Strategy 2: Use iterative refinement
    # Start with schoolbook (rank 90) and try to merge terms
    
    # For now, return the best we found
    return best_result

def _schoolbook(n, m, p, rank):
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, result):
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
