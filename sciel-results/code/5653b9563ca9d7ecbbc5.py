import time
import numpy as np

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin

    # 1. Schoolbook fallback (Rank 28)
    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k, col] = 1
                    V[k * p + j, col] = 1
                    W[j * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, r

    best_result, best_rank = schoolbook()

    # 2. Verification function
    def verify(U, V, W):
        """
        U: (n*m, r)
        V: (m*p, r)
        W: (p*n, r)
        Check if sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] == 1
        """
        r = U.shape[1]
        u_r = U.T.reshape(n, m, r)
        v_r = V.T.reshape(m, p, r)
        w_r = W.T.reshape(p, n, r)
        
        # The tensor identity for matrix multiplication C = A * B is:
        # sum_r U[i*m+k, r] V[k*p+j, r] W[j*n+i, r] = 1 for all i, j, k
        # This corresponds to the all-ones tensor of shape (n, p, m)
        T = np.einsum('ikr,kjr,jir->ijk', u_r, v_r, w_r)
        
        # Check if T is exactly 1 (or very close, allowing for float errors)
        # The problem requires exact decomposition, but floating point might introduce noise.
        # We check closeness first, then we can try to round if needed.
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        
        # Check if entries are close to integers or half-integers
        # The prompt says "integer/half-integer entries"
        # We will attempt to round and re-verify if close.
        
        U_rounded = np.round(U * 2) / 2.0
        V_rounded = np.round(V * 2) / 2.0
        W_rounded = np.round(W * 2) / 2.0
        
        # Verify the rounded version
        u_r2 = U_rounded.T.reshape(n, m, r)
        v_r2 = V_rounded.T.reshape(m, p, r)
        w_r2 = W_rounded.T.reshape(p, n, r)
        T2 = np.einsum('ikr,kjr,jir->ijk', u_r2, v_r2, w_r2)
        
        if np.allclose(T2, 1.0, atol=1e-9):
            return True
            
        return False

    # 3. Attempt to construct a rank-25 decomposition
    # Strategy: Merge terms from different columns.
    # The schoolbook has 28 terms: 2 (i) * 2 (k) * 7 (j) = 28.
    # We want to reduce this to 25.
    # This implies sharing 3 rank-1 terms between different columns j.
    
    # Let's try a specific algebraic construction.
    # C[i, j] = sum_k A[i, k] B[k, j]
    # 
    # Consider the first column j=0.
    # C[0, 0] = A[0, 0]B[0, 0] + A[0, 1]B[1, 0]
    # C[1, 0] = A[1, 0]B[0, 0] + A[1, 1]B[1, 0]
    #
    # Can we share terms?
    # If we use Strassen-like ideas, we might combine products.
    # However, for 2x2x7, the known optimal rank is 25.
    # There is a known algorithm for 2x2x7 with rank 25.
    # One such construction is based on the "Winograd" algorithm or similar.
    
    # Let's try to find the rank-25 decomposition numerically but with a better initialization 
    # or a different search method, OR just try to hardcode a known one if I can recall/derive it.
    
    # Since I don't have the exact coefficients memorized for 2x2x7 rank 25, 
    # and numerical search in the previous step failed (score 1.0), 
    # let's try a more robust numerical search using a different method.
    
    # The previous attempt used Nelder-Mead on 25*4 + 25*14 + 25*14 = 100 + 350 + 350 = 800 parameters.
    # This is too high-dimensional for Nelder-Mead to converge reliably.
    
    # Alternative: Use the fact that the tensor is all-ones.
    # We are looking for matrices U, V, W such that:
    # sum_r U[i*m+k, r] V[k*p+j, r] W[j*n+i, r] = 1
    
    # This is equivalent to finding a low-rank approximation of the identity map in a specific tensor product space.
    # Actually, it's the tensor T = sum_{i,k,j} e_{i,k} \otimes e_{k,j} \otimes e_{j,i}.
    
    # Let's try a random search with local refinement using gradient descent (if possible) or just more random restarts with a smarter objective.
    
    # Objective: Minimize || T(U,V,W) - 1 ||_F^2
    # T(U,V,W) is computed via einsum.
    
    # To make it faster, we can fix one matrix, say W, to be the identity-like or random, and solve for U and V?
    # No, it's a 3-way decomposition.
    
    # Let's try a simple iterative method:
    # 1. Initialize U, V, W randomly.
    # 2. Update U by fixing V, W: Minimize || T - 1 ||_F^2 w.r.t U.
    #    T[i,j,k] = sum_r U[i*m+k, r] V[k*p+j, r] W[j*n+i, r]
    #    For fixed i, j, k, this is linear in U[i*m+k, r].
    #    Actually, for fixed i, k, the term U[i*m+k, r] is multiplied by (sum_j V[k*p+j, r] W[j*n+i, r]).
    #    Let M_ik[r] = sum_j V[k*p+j, r] W[j*n+i, r].
    #    Then T[i,j,k] = sum_r U[i*m+k, r] * (V[k*p+j, r] W[j*n+i, r]).
    #    This is not a simple linear least squares for U because the coefficient depends on j.
    #    
    #    Let's reshape.
    #    For a fixed i, k, we have a vector u_ik of length r.
    #    For each j, we have a vector v_kj_w_ji of length r (element-wise product of V[k*p+j, :] and W[j*n+i, :]).
    #    The contribution to T[i, :, k] is u_ik^T * diag(v_kj_w_ji)? No.
    #    T[i, j, k] = dot(u_ik, v_kj * w_ji).
    #    We want T[i, j, k] = 1.
    #    So we want to find u_ik such that dot(u_ik, v_kj * w_ji) = 1 for all j.
    #    This is a system of p linear equations for u_ik (length r).
    #    If p <= r, we can solve this exactly (or least squares).
    #    Here p=7, r=25. So we can solve for u_ik exactly!
    
    # Algorithm:
    # 1. Initialize V and W randomly.
    # 2. For each i, k:
    #    - Construct the system A u_ik = 1, where A[j, r] = V[k*p+j, r] * W[j*n+i, r].
    #    - Solve for u_ik using least squares (np.linalg.lstsq).
    #    - Set U[i*m+k, :] = u_ik.
    # 3. Check if the resulting T is close to 1.
    # 4. If not, update V and W? 
    #    This is a 2-stage optimization. We can fix U, V and solve for W, etc.
    #    Or use an alternating least squares (ALS) approach.
    
    def als_step(r, U, V, W, max_iter=100):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # We want to minimize || T - 1 ||_F^2
        # T[i, j, k] = sum_r U[i*m+k, r] V[k*p+j, r] W[j*n+i, r]
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
                
            # Update U
            for i in range(n):
                for k in range(m):
                    # A[j, r] = V[k*p+j, r] * W[j*n+i, r]
                    A = V[k*p:(k+1)*p, :] * W[i::n, :] # Wait, W index is j*n+i.
                    # W shape is (p*n, r).
                    # W[j*n+i, :] for j=0..p-1.
                    # Indices: i, i+n, i+2n, ... i+(p-1)n.
                    # So W[i::n, :] gives rows i, i+n, ..., i+(p-1)n.
                    # This matches j*n+i for j=0..p-1.
                    
                    # We want U[i*m+k, :] such that A @ U[i*m+k, :]^T = 1 (vector of ones)
                    # Actually, T[i, j, k] = dot(U[i*m+k, :], A[j, :])
                    # So we want A @ u = 1, where u is column vector.
                    try:
                        u, residuals, rank, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i*m+k, :] = u
                    except:
                        pass
            
            # Update V
            for k in range(m):
                for j in range(p):
                    # For fixed k, j, we want V[k*p+j, :].
                    # T[i, j, k] = sum_r U[i*m+k, r] V[k*p+j, r] W[j*n+i, r]
                    # Let B[i, r] = U[i*m+k, r] * W[j*n+i, r]
                    # We want dot(B[i, :], V[k*p+j, :]) = 1 for all i.
                    # So B @ v = 1, where v is column vector of length r.
                    B = U[:, k::m] * W[i::n, :] # Wait, U index is i*m+k.
                    # U[i*m+k, :] for i=0..n-1.
                    # Indices: k, k+m, k+2m.
                    # So U[k::m, :] gives rows k, k+m, ...
                    # This matches i*m+k for i=0..n-1.
                    
                    B = U[k::m, :] * W[i::n, :]
                    
                    try:
                        v, residuals, rank, sv = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        V[k*p+j, :] = v
                    except:
                        pass
                        
            # Update W
            for j in range(p):
                for i in range(n):
                    # For fixed j, i, we want W[j*n+i, :].
                    # T[i, j, k] = sum_r U[i*m+k, r] V[k*p+j, r] W[j*n+i, r]
                    # Let C[k, r] = U[i*m+k, r] * V[k*p+j, r]
                    # We want dot(C[k, :], W[j*n+i, :]) = 1 for all k.
                    # So C @ w = 1, where w is column vector of length r.
                    
                    C = U[i*m::m, :] * V[j::p, :] # Wait, V index is k*p+j.
                    # V[k*p+j, :] for k=0..m-1.
                    # Indices: j, j+p, j+2p.
                    # So V[j::p, :] gives rows j, j+p, ...
                    # This matches k*p+j for k=0..m-1.
                    
                    C = U[i*m::m, :] * V[j::p, :]
                    
                    try:
                        w, residuals, rank, sv = np.linalg.lstsq(C, np.ones(m), rcond=None)
                        W[j*n+i, :] = w
                    except:
                        pass
            
            # Check convergence
            u_r = U.T.reshape(n, m, r)
            v_r = V.T.reshape(m, p, r)
            w_r = W.T.reshape(p, n, r)
            T = np.einsum('ikr,kjr,jir->ijk', u_r, v_r, w_r)
            error = np.sum((T - 1.0)**2)
            if error < 1e-8:
                break
                
        return U, V, W, error

    # Run ALS with several random restarts
    target_rank = 25
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        np.random.seed(100 + restart)
        U = np.random.randn(n*m, target_rank) * 0.1
        V = np.random.randn(m*p, target_rank) * 0.1
        W = np.random.randn(p*n, target_rank) * 0.1
        
        U_opt, V_opt, W_opt, error = als_step(target_rank, U, V, W, max_iter=50)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U_opt, V_opt, W_opt
            
            # Try to round and verify
            U_rounded = np.round(best_U * 2) / 2.0
            V_rounded = np.round(best_V * 2) / 2.0
            W_rounded = np.round(best_W * 2) / 2.0
            
            if verify(U_rounded, V_rounded, W_rounded):
                best_result = {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
                best_rank = target_rank
                break

    return best_result
# EVOLVE-BLOCK-END
