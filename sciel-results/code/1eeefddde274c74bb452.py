import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 58  # Try to find a decomposition of rank 58
    max_iters = 200
    
    # The tensor T has dimensions (n*m, m*p, p*n)
    # T[(i*m+j), (j'*p+k'), (k''*n+i'')] = 1 if j==j', k'==k'', i''==i else 0
    
    # We want U (nm x R), V (mp x R), W (pn x R)
    # such that sum_r U[:,r] V[:,r] W[:,r] == T_flattened?
    # No, the tensor is T_{alpha, beta, gamma}.
    # The decomposition is T_{alpha, beta, gamma} = sum_r U[alpha, r] V[beta, r] W[gamma, r].
    
    # We can solve for W given U and V.
    # The equation is:
    # For all alpha, beta, gamma:
    # sum_r U[alpha, r] V[beta, r] W[gamma, r] = T[alpha, beta, gamma]
    
    # This is a linear system for W if U and V are fixed.
    # Let's vectorize.
    # Let M be a matrix where rows correspond to (alpha, beta) pairs and columns to (gamma, r) pairs?
    # No, W is shared across all (alpha, beta).
    # The constraint is:
    # For each r, the outer product U[:,r] V[:,r] contributes to the tensor slice.
    # Actually, for a fixed r, the term is U[:,r] \otimes V[:,r] \otimes W[:,r].
    # The tensor T is the sum of these.
    
    # We can flatten the tensor T into a matrix T_mat of size (nm * mp) x (pn).
    # T_mat[alpha * mp + beta, gamma] = T[alpha, beta, gamma].
    # The decomposition becomes:
    # T_mat = sum_r (U[:,r] \otimes V[:,r])_flattened * W[:,r]^T
    # Let A_r = (U[:,r] \otimes V[:,r])_flattened, which is a vector of size nm*mp.
    # Let W be the matrix pn x R.
    # Then T_mat (nm*mp x pn) = A (nm*mp x R) * W (R x pn).
    # Where A[:,r] = (U[:,r] \otimes V[:,r])_flattened.
    
    # So we need to find A and W such that A * W = T_mat.
    # And A must be of the form where each column is a Kronecker product of two vectors.
    
    # Strategy:
    # 1. Randomly initialize U and V with small integers.
    # 2. Construct A from U and V.
    # 3. Solve for W using least squares: W = A^+ T_mat.
    # 4. Check if A * W approx T_mat exactly.
    # 5. If yes, round W to integers and verify.
    
    # To increase success rate, we can try to fix some entries of U and V to be sparse.
    
    # Construct T_mat
    T_mat = np.zeros((n*m * m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T_mat[alpha * (m*p) + beta, gamma] = 1.0
                
    # Try to find decomposition of rank target_rank
    best_U, best_V, best_W = None, None, None
    
    for _ in range(max_iters):
        # Random U, V with entries in {-1, 0, 1}
        U = np.random.randint(-1, 2, size=(n*m, target_rank))
        V = np.random.randint(-1, 2, size=(m*p, target_rank))
        
        # Construct A
        # A[:, r] = kron(U[:, r], V[:, r])
        # Note: The flattening order must match T_mat.
        # T_mat index: alpha * (mp) + beta.
        # kron(u, v) index: i * len(v) + j.
        # Here u is length nm, v is length mp.
        # So kron(u, v) has length nm * mp.
        # Index is i * mp + j. This matches alpha * mp + beta.
        
        A = np.zeros((n*m * m*p, target_rank))
        for r in range(target_rank):
            A[:, r] = np.kron(U[:, r], V[:, r])
            
        # Solve for W: A * W = T_mat
        # W = A^+ T_mat
        # Using lstsq
        try:
            W, residuals, rank, s = np.linalg.lstsq(A, T_mat, rcond=None)
        except np.linalg.LinAlgError:
            continue
            
        # Check residual
        recon = A @ W
        if np.allclose(recon, T_mat, atol=1e-9):
            # Found a solution!
            # Round W to nearest integer
            W_int = np.round(W)
            recon_int = A @ W_int
            if np.allclose(recon_int, T_mat, atol=1e-9):
                best_U = U
                best_V = V
                best_W = W_int
                break
                
    if best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        # Fallback to schoolbook rank 60
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
