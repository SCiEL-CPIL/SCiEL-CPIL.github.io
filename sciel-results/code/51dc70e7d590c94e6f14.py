import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    total_entries = nm * mp * pn
    
    # Precompute the tensor T as a flat vector of length total_entries
    # T[u, v, w] = 1 if indices match, else 0
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    
    T = np.zeros(total_entries)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                idx = (u * mp + v) * pn + w
                T[idx] = 1.0

    # We want to find U (nm x R), V (mp x R), W (pn x R)
    # such that sum_r U[u,r] V[v,r] W[w,r] = T[u,v,w]
    
    # Strategy:
    # 1. Try ranks from lower bound to 29.
    # 2. For each rank, try random U, V and solve for W.
    # 3. Check if W is integer/half-integer.
    
    # Lower bound estimation
    # Rank must be at least max(nm, mp, pn)? No.
    # Rank must be at least ceil(nmp / (sqrt(nm)+sqrt(mp)+sqrt(pn)))?
    # Let's just try from 20 upwards.
    
    start_time = time.time()
    max_time = 110.0 # Safety margin
    
    # Define a function to check if a matrix is close to integer/half-integer
    def is_int_or_half(x, tol=1e-6):
        # x is a numpy array
        # Check if 2*x is close to an integer
        two_x = 2.0 * x
        rounded = np.round(two_x)
        if np.allclose(two_x, rounded, atol=tol):
            # Convert to integer/half-integer representation
            # The problem allows integer/half-integer.
            # We can return the values as is, or convert to fractions?
            # The output should be 'U', 'V', 'W' with entries.
            # If 2*x is integer, x is k/2.
            return True
        return False

    # Linear system setup for W
    # Unknowns: W_flat of length pn * R
    # Equations: for each (u, v), we have pn equations for w
    # A[u, v, w, (w', r)] = ?
    # Actually, for a fixed (u, v), the equation is:
    # sum_r (U[u, r] V[v, r]) W[w, r] = T[u, v, w] for all w
    # This is a system of size (nm * mp) x (pn * R)
    
    nm_mp = nm * mp
    A = np.zeros((nm_mp, pn * 30)) # Max R 30? No, we build for specific R
    
    # Precompute U*V products for random U, V?
    # No, U, V change each iteration.
    
    for R in range(20, 30):
        if time.time() - start_time > max_time:
            break
            
        # Build the structure of A which depends on R
        # A has shape (nm_mp, pn * R)
        # A[(u*mp+v), (w*R + r)] = 1 if we are solving for W[w,r]
        # Wait, the coefficient is U[u,r] V[v,r].
        # So A is not fixed. It depends on U and V.
        
        # We can build A efficiently.
        # A is block matrix?
        # For each (u,v), the row block is pn x R.
        # The entry at (w, r) in that block is U[u,r] V[v,r].
        
        # Let's iterate
        for trial in range(500):
            if time.time() - start_time > max_time:
                break
                
            # Generate random U, V with small integers
            # Using -1, 0, 1 to keep numbers small
            U = np.random.randint(-1, 2, size=(nm, R))
            V = np.random.randint(-1, 2, size=(mp, R))
            
            # Construct A
            # A is (nm * mp) x (pn * R)
            # We can build it by iterating u, v
            A = np.zeros((nm_mp, pn * R))
            # For each u, v, the block is pn x R
            # Block[w, r] = U[u, r] * V[v, r]
            
            # Vectorized construction
            # U_flat: (nm, R)
            # V_flat: (mp, R)
            # We need outer product of U[u] and V[v] for each pair?
            # This is a 4D tensor U[u,r] * V[v,r] -> (nm, mp, pn, R)? No.
            # It's (nm, mp, R).
            # We need to reshape it to (nm*mp, pn*R)?
            # No, the column index is (w, r).
            # The row index is (u, v).
            # The value is U[u, r] * V[v, r].
            # This is independent of w!
            # Wait, the equation is:
            # sum_r (U[u, r] * V[v, r]) * W[w, r] = T[u, v, w]
            # So for a fixed (u, v), we have a system of pn equations:
            # For w in 0..pn-1:
            #   sum_r (U[u, r] * V[v, r]) * W[w, r] = T[u, v, w]
            
            # This means the matrix A for the whole system is:
            # Row (u, v, w), Col (w', r)
            # A[(u,v,w), (w',r)] = 1 if w==w', else 0?
            # No, the unknown is W[w', r].
            # The coefficient of W[w', r] in the equation for T[u, v, w] is:
            # U[u, r] * V[v, r] if w == w', else 0.
            
            # So A is block diagonal with respect to w?
            # No, the rows are indexed by (u, v, w).
            # The columns are indexed by (w', r).
            # If w != w', the coefficient is 0.
            # If w == w', the coefficient is U[u, r] * V[v, r].
            
            # So A is a block diagonal matrix where each block corresponds to a w?
            # No, the rows are grouped by (u, v).
            # For a fixed (u, v), we have pn rows (for w=0..pn-1).
            # And pn*R columns (for w'=0..pn-1, r=0..R-1).
            # The block for w is a row vector of length R: [U[u,0]V[v,0], ..., U[u,R-1]V[v,R-1]].
            # This row vector is placed at column indices corresponding to (w, r).
            
            # So for each (u, v), the submatrix is:
            # [ 0 ... 0 | C_u_v | 0 ... 0 ]
            # [ 0 ... 0 | 0 ... 0 | C_u_v | 0 ... 0 ]
            # ...
            # Where C_u_v is the row vector U[u] * V[v] (element-wise).
            
            # This structure allows efficient solving.
            # Actually, we can just build the full A matrix for small sizes.
            # nm_mp = 90. pn*R = 10*R.
            # A is 900 x 100 (for R=10).
            
            # Let's build A
            # A is zero initialized
            for u in range(nm):
                for v in range(mp):
                    row_base = u * mp + v
                    # The vector C = U[u] * V[v]
                    C = U[u] * V[v] # length R
                    # For each w, the row is at index (row_base * pn + w)
                    # The non-zero columns are at indices (w * R + r)
                    for w in range(pn):
                        row_idx = row_base * pn + w
                        col_start = w * R
                        A[row_idx, col_start:col_start+R] = C
            
            # RHS vector b
            b = T.reshape(nm_mp * pn)
            
            # Solve A W_flat = b
            # Use least squares or exact solve if possible
            # Since we expect exact integer solutions, we can use np.linalg.lstsq
            # Or better, check if the system is consistent and solve.
            try:
                W_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                # Check residuals
                if np.max(np.abs(A @ W_flat - b)) > 1e-6:
                    continue
                    
                # Reshape W_flat to (pn, R)
                W = W_flat.reshape(pn, R)
                
                # Check if W is integer/half-integer
                if is_int_or_half(W):
                    # Verify the decomposition
                    # Reconstruct T
                    T_recon = np.zeros(total_entries)
                    for r in range(R):
                        # Outer product U[:, r] (nm) x V[:, r] (mp) x W[:, r] (pn)
                        # This is slow, but we only do it once
                        pass # We trust the linear solve if residuals are 0
                    
                    # However, we must ensure U, V, W are the ones returned.
                    # The problem asks for U, V, W.
                    # We found U, V random, W solved.
                    # We must verify that the entries are valid.
                    
                    # Convert to list of lists for output
                    U_out = U.tolist()
                    V_out = V.tolist()
                    W_out = W.tolist()
                    
                    return {"U": U_out, "V": V_out, "W": W_out}
            except np.linalg.LinAlgError:
                continue

    # Fallback to schoolbook if no better rank found
    rank = n * m * p
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
# EVOLVE-BLOCK-END
