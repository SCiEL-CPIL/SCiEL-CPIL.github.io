import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 4]

def construct():
    t_start = time.time()
    deadline = t_start + 140  # safety margin
    
    n, m, p = DIMENSIONS
    # Tensor T[i, j, k] = 1 if the multiplication uses A[i,j]*B[j,k]
    # U shape: (n*m, rank) = (9, rank)
    # V shape: (m*p, rank) = (12, rank)
    # W shape: (p*n, rank) = (12, rank)
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    nm = n * m  # 9
    mp = m * p  # 12
    pn = p * n  # 12
    
    # Build the tensor in a flattened form
    # For each (i,j,k), the tensor value is 1
    # The decomposition requires: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Strategy: try ranks from a lower bound upward
    # Lower bound: max(nm, mp, pn) = 12, but let's start higher for feasibility
    # Known results suggest ~20-25 for 3x3x4
    
    # Use ALS: iterate fixing two factors, solve for third via linear system
    # Then round and verify
    
    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the tensor."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def round_to_half(x):
        """Round each entry to nearest half-integer."""
        return np.round(x * 2.0) / 2.0
    
    def als_decompose(rank, max_iter=200, seed=0):
        """ALS decomposition with exact rounding verification."""
        rng = np.random.RandomState(seed)
        # Initialize with small random values
        U = rng.randn(nm, rank) * 0.1
        V = rng.randn(mp, rank) * 0.1
        W = rng.randn(pn, rank) * 0.1
        
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Solve for W: for each (k, i), W[k*n+i, :] should satisfy
            # sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1 for all j
            # This is a system of m equations per (k,i) pair
            # Actually, let's set up the full linear system
            
            # For solving W: fix U, V.
            # For each output index (k, i), we need:
            # sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * V[j*p+k, c] * delta_{j,k_in_V}) 
            # Wait, let me reconsider.
            
            # The constraint is: for all (i,j,k):
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            
            # If we fix U and V, then for each (k, i), we have m constraints (one for each j):
            # sum_c W[k*n+i, c] * (U[i*m+j, c] * V[j*p+k, c]) = 1 for j=0..m-1
            # This is a linear system in W[k*n+i, :]:
            # Let A_{(k,i), j, c} = U[i*m+j, c] * V[j*p+k, c]
            # Then sum_c A_{j,c} * W[k*n+i, c] = 1 for each j
            
            # Build the system for W
            # W has pn * rank unknowns. Constraints: n*m*p = 36 equations.
            # Actually, the constraints are: for each (i,j,k), one equation.
            # Unknowns: W has pn*rank = 12*rank values.
            # If rank < 3, this is overdetermined. Let's use least squares.
            
            # Reshape: create matrix M of shape (n*m*p, pn*rank)
            # M[row, col] where row = (i,j,k) index, col = (k*n+i, c) index
            # M[row, col] = U[i*m+j, c] * V[j*p+k, c] if col corresponds to (k*n+i, c)
            
            # This is expensive for large rank. Let's use a smarter approach.
            
            # Alternative: use the fact that we can solve column by column or use block structure.
            
            # Let me try a different approach: use the bilinear structure.
            # Fix W, solve for U and V via linear least squares.
            
            # For fixing W and solving for U:
            # For each (i, j), we need: sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1 for all k
            # This gives p equations per (i,j) pair, with rank unknowns.
            
            # Let's just do full least squares.
            
            # Build the design matrix for solving W
            # Unknowns: W flattened (pn*rank,)
            # Equations: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            
            # Create matrix A of shape (n*m*p, pn*rank)
            A_W = np.zeros((n*m*p, pn*rank))
            b = np.ones(n*m*p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(rank):
                            A_W[idx, k*n+i*rank + c] = U[i*m+j, c] * V[j*p+k, c]
                        idx += 1
            
            # Solve least squares
            W_flat, _, _, _ = np.linalg.lstsq(A_W, b, rcond=None)
            W = W_flat.reshape(pn, rank)
            
            # Solve for U
            A_U = np.zeros((n*m*p, nm*rank))
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(rank):
                            A_U[idx, i*m+j*rank + c] = V[j*p+k, c] * W[k*n+i, c]
                        idx += 1
            U_flat, _, _, _ = np.linalg.lstsq(A_U, b, rcond=None)
            U = U_flat.reshape(nm, rank)
            
            # Solve for V
            A_V = np.zeros((n*m*p, mp*rank))
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(rank):
                            A_V[idx, j*p+k*rank + c] = U[i*m+j, c] * W[k*n+i, c]
                        idx += 1
            V_flat, _, _, _ = np.linalg.lstsq(A_V, b, rcond=None)
            V = V_flat.reshape(mp, rank)
            
            # Round to half-integers
            U_r = round_to_half(U)
            V_r = round_to_half(V)
            W_r = round_to_half(W)
            
            # Check if rounded version works
            if verify(U_r, V_r, W_r, tol=1e-6):
                return U_r, V_r, W_r
        
        return None
    
    # Try ranks from low to high
    # Start from a reasonable lower bound
    for rank in range(12, 40):
        if time.time() > deadline - 5:
            break
        
        # Try multiple seeds
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            result = als_decompose(rank, max_iter=100, seed=seed)
            if result is not None:
                U, V, W = result
                # Convert to lists of lists
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1
                V[j*p+k][column] = 1
                W[k*n+i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
