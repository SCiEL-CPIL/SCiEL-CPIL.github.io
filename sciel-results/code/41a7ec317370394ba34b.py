import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    def verify_decomposition(U, V, W, rank):
        """Verify that U, V, W form a valid rank-r decomposition.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        Check: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
        """
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        # T_calc[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        # Use einsum for efficiency
        T_calc = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        # Target is all ones
        max_err = np.max(np.abs(T_calc - 1.0))
        return max_err < 1e-9, max_err

    def als_refine(U, V, W, rank, max_iter=200, tol=1e-12):
        """Alternating Least Squares refinement for tensor decomposition.
        For fixed V and W, solve for U via linear least squares.
        For fixed U and W, solve for V.
        For fixed U and V, solve for W.
        
        The key insight: for a rank-r decomposition of the matrix multiplication tensor,
        we need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k.
        
        For fixed V and W, this is linear in U. We can vectorize:
        For each r, the contribution to T[i,j,k] is U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        
        Let's restructure. Define for each r:
        A_r = U[:,r] reshaped to (n,m)
        B_r = V[:,r] reshaped to (m,p)  
        C_r = W[:,r] reshaped to (p,n)
        
        T[i,j,k] = sum_r A_r[i,j] * B_r[j,k] * C_r[k,i]
        
        For fixed V, W: we want to find U such that this holds.
        This is a linear system. Let's flatten U to a vector of length n*m*rank.
        For each (i,j,k), the equation is:
        sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1
        
        So for each (i,j,k): sum_r M_{(i,j,k),r} * U_flat[i*m*rank + j*rank + r] = 1
        where M_{(i,j,k),r} = V[j*p+k,r] * W[k*n+i,r]
        
        This is a linear system: M @ U_flat = 1
        where M has shape (n*m*p, n*m*rank)
        
        Similarly for V and W.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # --- Solve for U ---
            # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # Build linear system for U_flat
            # M_U has shape (n*m*p, n*m*rank)
            # M_U[(i,j,k), (i',j',r)] = V[j'*p+k,r] * W[k*n+i',r] if i==i' and j==j' else 0
            
            # More efficiently: for each r, define matrix Q_r of shape (n*m, n*m*p)
            # Actually, let's think of it differently.
            
            # For fixed V and W, the equation for U is linear.
            # Let's build the system block by block.
            
            # U_flat is a vector of length n*m*rank.
            # The linear system is: for each (i,j,k),
            #   sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
            
            # So the coefficient matrix A_U has entries:
            # A_U[(i,j,k), (i,j,r)] = V[j*p+k, r] * W[k*n+i, r]
            # and A_U[(i,j,k), (i',j',r')] = 0 if (i',j') != (i,j)
            
            # This means A_U is block diagonal in the (i,j) index!
            # Each block corresponds to a fixed (i,j) and has shape (p, rank)
            # Block (i,j): A_U_block[(k, r)] = V[j*p+k, r] * W[k*n+i, r]
            # The RHS for block (i,j) is a vector of p ones.
            
            # So we can solve p separate rank x rank systems!
            # For each (i,j): solve A_U_block @ U[i*m+j, :] = ones(p)
            
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # A_U_block has shape (p, rank)
                    # A_U_block[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    A_block = V_r[j, :, :] * W_r[:, i, :]  # shape (p, rank)
                    # Solve A_block @ x = ones(p) for x of shape (rank,)
                    # Use least squares
                    try:
                        x, _, _, _ = np.linalg.lstsq(A_block, np.ones(p), rcond=None)
                        U_new[i*m + j, :] = x
                    except np.linalg.LinAlgError:
                        U_new[i*m + j, :] = U[i*m + j, :]
            
            # --- Solve for V ---
            # For fixed U and W: for each (i,j,k),
            #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # Coefficient for V[j*p+k, r] is U[i*m+j, r] * W[k*n+i, r]
            # For fixed (j,k), the equation is:
            #   sum_r V[j*p+k, r] * (U[i*m+j, r] * W[k*n+i, r]) = 1 for all i
            # So block (j,k): A_V_block[i, r] = U[i*m+j, r] * W[k*n+i, r], shape (n, rank)
            # Solve A_V_block @ V[j*p+k, :] = ones(n)
            
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A_block = U_r[:, j, :] * W_r[k, :, :]  # shape (n, rank)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A_block, np.ones(n), rcond=None)
                        V_new[j*p + k, :] = x
                    except np.linalg.LinAlgError:
                        V_new[j*p + k, :] = V[j*p + k, :]
            
            # --- Solve for W ---
            # For fixed U and V: for each (i,j,k),
            #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # Coefficient for W[k*n+i, r] is U[i*m+j, r] * V[j*p+k, r]
            # For fixed (k,i), the equation is:
            #   sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1 for all j
            # So block (k,i): A_W_block[j, r] = U[i*m+j, r] * V[j*p+k, r], shape (m, rank)
            # Solve A_W_block @ W[k*n+i, :] = ones(m)
            
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A_block = U_r[i, :, :] * V_r[:, k, :]  # shape (m, rank)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A_block, np.ones(m), rcond=None)
                        W_new[k*n + i, :] = x
                    except np.linalg.LinAlgError:
                        W_new[k*n + i, :] = W[k*n + i, :]
            
            # Check convergence
            U_diff = np.max(np.abs(U_new - U))
            V_diff = np.max(np.abs(V_new - V))
            W_diff = np.max(np.abs(W_new - W))
            U = U_new
            V = V_new
            W = W_new
            
            if max(U_diff, V_diff, W_diff) < tol:
                break
        
        # Final error
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T_calc = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        err = np.max(np.abs(T_calc - 1.0))
        
        return U, V, W, err

    # Try to find a decomposition with rank < 24
    found = None
    
    # Strategy 1: Try rank 23 with random initialization + ALS
    for trial in range(20):
        if time.time() > deadline - 20.0:
            break
        
        np.random.seed(42 + trial)
        U = np.random.randn(n * m, 23)
        V = np.random.randn(m * p, 23)
        W = np.random.randn(p * n, 23)
        
        U, V, W, err = als_refine(U, V, W, 23, max_iter=100)
        
        if err < 1e-9:
            valid, _ = verify_decomposition(U, V, W, 23)
            if valid:
                found = (23, U, V, W)
                break
    
    # Strategy 2: Start from rank-24 schoolbook, drop one column, then ALS
    if found is None and time.time() < deadline - 15.0:
        rank24 = n * m * p
        U24 = np.zeros((n * m, rank24))
        V24 = np.zeros((m * p, rank24))
        W24 = np.zeros((p * n, rank24))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U24[i * m + j, col] = 1
                    V24[j * p + k, col] = 1
                    W24[k * n + i, col] = 1
                    col += 1
        
        # Try dropping each column and running ALS
        for drop_col in range(rank24):
            if time.time() > deadline - 10.0:
                break
            
            U = np.delete(U24, drop_col, axis=1)
            V = np.delete(V24, drop_col, axis=1)
            W = np.delete(W24, drop_col, axis=1)
            
            # Add small random perturbation to break symmetry
            np.random.seed(100 + drop_col)
            U += np.random.randn(n * m, 23) * 0.01
            V += np.random.randn(m * p, 23) * 0.01
            W += np.random.randn(p * n, 23) * 0.01
            
            U, V, W, err = als_refine(U, V, W, 23, max_iter=200)
            
            if err < 1e-9:
                valid, _ = verify_decomposition(U, V, W, 23)
                if valid:
                    found = (23, U, V, W)
                    break
    
    if found is not None:
        rank, U, V, W = found
        # Final verification
        valid, err = verify_decomposition(U, V, W, rank)
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook decomposition with rank = n*m*p = 24
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify the schoolbook solution
    valid, err = verify_decomposition(U, V, W, rank)
    if not valid:
        raise RuntimeError(f"Schoolbook decomposition failed verification: err={err}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
