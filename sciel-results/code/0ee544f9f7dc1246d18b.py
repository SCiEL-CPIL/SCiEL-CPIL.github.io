import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Target tensor T[i, j, j_prime, k, k_prime, i_prime]
    # Note: In the problem statement, W index is (k, i).
    # So the term is U[i,j,c] * V[j',k',c] * W[k,i',c].
    # Target: delta(i,i') * delta(j,j') * delta(k,k')
    
    # Flatten indices for linear system
    # Equations indexed by (i, j, j_prime, k, k_prime, i_prime)
    # Variables for W: (k, i_prime, c)
    
    def build_system_U(V, W):
        # Solve for U
        # Sum_c U[i,j,c] V[j',k',c] W[k,i',c] = T
        # For each (i, j), we have variables U[i,j,c] for c=1..r.
        # Equation for (i, j, j', k, k', i'):
        # Sum_c U[i,j,c] * (V[j',k',c] * W[k,i',c]) = T[i,j,j',k',k,i']
        
        # This is a linear system for U[i,j,:].
        # Let's vectorize U as (n*m, r).
        # For each (i,j), the equation is:
        # M[i,j] @ U[i,j,:] = b[i,j]
        # where M[i,j] is (num_eqs, r) and b[i,j] is (num_eqs,)
        
        # This is too big to build explicitly for all i,j at once if we do it naively.
        # But we can solve for each (i,j) separately?
        # No, the equations couple all i,j?
        # No, U[i,j,c] only appears in equations where the first two indices are i,j.
        # So we can solve for each (i,j) independently!
        
        U = np.zeros((n*m, r))
        for i in range(n):
            for j in range(m):
                # Collect all equations involving U[i,j,c]
                # These are all (j', k, k', i')
                # Coefficient for U[i,j,c] is V[j',k',c] * W[k,i',c]
                
                # We need to form the matrix A of shape (m*p*p*n, r)
                # and vector b of shape (m*p*p*n,)
                
                A = np.zeros((m*p*p*n, r))
                b = np.zeros(m*p*p*n)
                
                idx = 0
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                # Coeffs for c
                                # V[jp, kp, c] * W[k, ip, c]
                                # V is (m, p, r), W is (p, n, r)
                                coeffs = V[jp, kp, :] * W[k, ip, :]
                                A[idx, :] = coeffs
                                
                                # Target value
                                # T[i, j, jp, k, kp, ip]
                                # Non-zero if i==ip and j==jp and k==kp
                                if i == ip and j == jp and k == kp:
                                    b[idx] = 1.0
                                else:
                                    b[idx] = 0.0
                                idx += 1
                                
                # Solve A @ x = b
                x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                U[i*m + j, :] = x
                
        return U

    def build_system_V(U, W):
        V = np.zeros((m*p, r))
        for jp in range(m):
            for kp in range(p):
                A = np.zeros((n*m*p*n, r)) # i, j, k, ip
                b = np.zeros(n*m*p*n)
                
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for ip in range(n):
                                coeffs = U[i, j, :] * W[k, ip, :]
                                A[idx, :] = coeffs
                                
                                if i == ip and j == jp and k == kp:
                                    b[idx] = 1.0
                                else:
                                    b[idx] = 0.0
                                idx += 1
                                
                x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                V[jp*p + kp, :] = x
        return V

    def build_system_W(U, V):
        W = np.zeros((p*n, r))
        for k in range(p):
            for ip in range(n):
                A = np.zeros((n*m*m*p, r)) # i, j, jp, kp
                b = np.zeros(n*m*m*p)
                
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            for kp in range(p):
                                coeffs = U[i, j, :] * V[jp, kp, :]
                                A[idx, :] = coeffs
                                
                                if i == ip and j == jp and k == kp:
                                    b[idx] = 1.0
                                else:
                                    b[idx] = 0.0
                                idx += 1
                                
                x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                W[k*n + ip, :] = x
        return W

    def check_solution(U, V, W):
        U_h = np.round(U * 2) / 2
        V_h = np.round(V * 2) / 2
        W_h = np.round(W * 2) / 2
        
        if not np.allclose(U, U_h) or not np.allclose(V, V_h) or not np.allclose(W, W_h):
            return False, None, None, None
            
        # Verify
        # Compute residual
        r = U.shape[1]
        U_r = U_h.T.reshape(r, n, m)
        V_r = V_h.T.reshape(r, m, p)
        W_r = W_h.T.reshape(r, p, n)
        
        U_b = U_r[:, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        V_b = V_r[:, np.newaxis, :, :, np.newaxis, np.newaxis]
        W_b = W_r[:, np.newaxis, np.newaxis, np.newaxis, :, :]
        
        Res = np.sum(U_b * V_b * W_b, axis=0)
        
        # Target
        T = np.zeros((n, m, m, p, p, n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, j, k, k, i] = 1.0
        
        if np.allclose(Res, T, atol=1e-9):
            return True, U_h, V_h, W_h
        return False, None, None, None

    # Try ALS for rank 64
    r = 64
    np.random.seed(42)
    U = np.random.randn(n*m, r)
    V = np.random.randn(m*p, r)
    W = np.random.randn(p*n, r)
    
    for it in range(20):
        W = build_system_W(U, V)
        V = build_system_V(U, W)
        U = build_system_U(V, W)
        
        valid, U_h, V_h, W_h = check_solution(U, V, W)
        if valid:
            return {"U": U_h.tolist(), "V": V_h.tolist(), "W": W_h.tolist()}
            
    # Fallback to schoolbook
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
