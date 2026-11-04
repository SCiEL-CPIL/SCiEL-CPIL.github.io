"""
Rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
Uses structured random search with ALS optimization and exact verification.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 by a 4x9 matrix.
    
    The tensor T is defined by:
    For A (n x m) * B (m x p) = C (n x p):
      C[i,k] = sum_j A[i,j] * B[j,k]
    
    The decomposition uses U (n*m x rank), V (m*p x rank), W (p*n, rank) such that:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    where T[a,b,c] = 1 iff (a%m == b//p) and (a//m == c%n)
    """
    start_time = time.time()
    deadline = start_time + 110  # 110 second deadline (10s safety margin)
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    # Pre-compute target tensor T as a 3D array
    T = np.zeros((n * m, m * p, p * n))
    for a in range(n * m):
        i = a // m
        j = a % m
        for b in range(m * p):
            j2 = b // p
            if j == j2:
                for c in range(p * n):
                    i2 = c % n
                    if i2 == i:
                        T[a, b, c] = 1.0
    
    def verify_full(U, V, W):
        """Full verification - vectorized."""
        rank = len(U[0])
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        try:
            S = np.einsum('ak,bk,ck->abc', U, V, W)
            if np.max(np.abs(S - T)) > 1e-8:
                return False
        except:
            return False
        return True
    
    def schoolbook():
        """Return the schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def als_optimize(U, V, W, max_iters=30, tol=1e-8):
        """
        Alternating Least Squares optimization for tensor decomposition.
        Returns the final error.
        """
        rank = len(U[0])
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update W: for each c, solve B W[c] = t_c where B[a,b] = U[a] * V[b]
            # B is (n*m*m*p, rank), t_c is (n*m*m*p,)
            B = np.zeros((n * m * m * p, rank))
            idx = 0
            for a in range(n * m):
                for b in range(m * p):
                    B[idx] = U[a] * V[b]
                    idx += 1
            
            # Solve for each c
            for c in range(p * n):
                # t_c is the flattened T[:, :, c]
                t_c = T[:, :, c].flatten()
                try:
                    W[c] = np.linalg.lstsq(B, t_c, rcond=None)[0]
                except:
                    pass
            
            # Update U: for each a, solve C U[a] = u_a where C[b,c] = V[b] * W[c]
            C = np.zeros((m * p * p * n, rank))
            idx = 0
            for b in range(m * p):
                for c in range(p * n):
                    C[idx] = V[b] * W[c]
                    idx += 1
            
            for a in range(n * m):
                u_a = T[a, :, :].flatten()
                try:
                    U[a] = np.linalg.lstsq(C, u_a, rcond=None)[0]
                except:
                    pass
            
            # Update V: for each b, solve D V[b] = v_b where D[a,c] = U[a] * W[c]
            D = np.zeros((n * m * p * n, rank))
            idx = 0
            for a in range(n * m):
                for c in range(p * n):
                    D[idx] = U[a] * W[c]
                    idx += 1
            
            for b in range(m * p):
                v_b = T[:, b, :].flatten()
                try:
                    V[b] = np.linalg.lstsq(D, v_b, rcond=None)[0]
                except:
                    pass
            
            # Check convergence
            S = np.einsum('ak,bk,ck->abc', U, V, W)
            err = np.linalg.norm(S - T)
            if err < tol:
                break
        
        return err, U, V, W
    
    best_rank = n * m * p  # schoolbook
    best_result = None
    
    # Try to find lower rank solutions
    # Start from rank 143 down to 100
    for target_rank in range(143, 99, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(100 + target_rank * 10 + attempt)
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            err, U, V, W = als_optimize(U, V, W, max_iters=30)
            
            if err < 1e-6:
                # Verify exactly
                if verify_full(U, V, W):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_result = {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                    found = True
                    break
        
        if found:
            # We found a solution at this rank.
            # Continue searching lower ranks if time permits.
            pass
    
    if best_result is not None:
        return best_result
    
    return schoolbook()
# EVOLVE-BLOCK-END
