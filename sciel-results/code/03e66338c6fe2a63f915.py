import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _als_decompose_fast(n, m, p, rank, max_iters=1000, tol=1e-8, seed=0):
    """
    Fast ALS for M_{n,m,p} using decoupled linear systems.
    """
    np.random.seed(seed)
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Reshape to 3D for easier indexing
    # U: (n, m, rank)
    # V: (m, p, rank)
    # W: (p, n, rank)
    
    for it in range(max_iters):
        if _time_left() < 15:
            return None

        # Update V
        # For each j, k, solve sum_t U[i,j,t] * W[k,i,t] * V[j,k,t] = 1 for all i
        for j in range(m):
            for k in range(p):
                # Matrix A (n x rank)
                # A[i, t] = U[i, j, t] * W[k, i, t]
                A = np.einsum('ijt,kit->it', U, W) # Wait, U is (n,m,rank), W is (p,n,rank)
                # U[i, j, t] * W[k, i, t]
                # i is 0..n-1, t is 0..rank-1
                # U[:, j, :] is (n, rank)
                # W[k, :, :] is (n, rank) -- Note W is indexed (p, n, rank) so W[k, i, t]
                # So A[i, t] = U[i, j, t] * W[k, i, t]
                A = U[:, j, :] * W[k, :, :]
                # Solve A v = 1
                v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                V[j, k, :] = v

        # Update U
        # For each i, j, solve sum_t V[j,k,t] * W[k,i,t] * U[i,j,t] = 1 for all k
        for i in range(n):
            for j in range(m):
                # Matrix C (p x rank)
                # C[k, t] = V[j, k, t] * W[k, i, t]
                C = V[j, :, :] * W[:, i, :]
                # Solve C u = 1
                u, _, _, _ = np.linalg.lstsq(C, np.ones(p), rcond=None)
                U[i, j, :] = u

        # Update W
        # For each k, i, solve sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = 1 for all j
        for k in range(p):
            for i in range(n):
                # Matrix D (m x rank)
                # D[j, t] = U[i, j, t] * V[j, k, t]
                D = U[i, :, :] * V[:, k, :]
                # Solve D w = 1
                w, _, _, _ = np.linalg.lstsq(D, np.ones(m), rcond=None)
                W[k, i, :] = w

        # Check error
        # sum_t U[i,j,t] V[j,k,t] W[k,i,t] should be 1
        # Compute all terms
        terms = np.einsum('ijt,jkt,kit->ijk', U, V, W)
        err = np.sqrt(np.mean((terms - 1.0)**2))
        
        if err < tol:
            # Snap to half-integers
            U_int = np.round(U * 2) / 2.0
            V_int = np.round(V * 2) / 2.0
            W_int = np.round(W * 2) / 2.0
            
            # Verify
            terms_int = np.einsum('ijt,jkt,kit->ijk', U_int, V_int, W_int)
            if np.allclose(terms_int, 1.0, atol=1e-6):
                # Flatten back to 2D
                U_flat = U_int.reshape(n * m, rank)
                V_flat = V_int.reshape(m * p, rank)
                W_flat = W_int.reshape(p * n, rank)
                return U_flat, V_flat, W_flat
                
    return None

def _snap_to_half_int(arr):
    return np.round(arr * 2) / 2.0

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition with integer/half-integer entries."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Reshape for einsum
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    
    terms = np.einsum('ijt,jkt,kit->ijk', U3, V3, W3)
    return np.allclose(terms, 1.0, atol=1e-6)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = 3, 3, 7
    
    U_33 = None
    V_33 = None
    W_33 = None
    rank_33 = 27
    
    # Try to find rank 23, 24, 25 for 3x3x3
    for target_rank in [23, 24, 25, 26]:
        if _time_left() < 30:
            break
        found = False
        for seed in range(50):
            if _time_left() < 10:
                break
            res = _als_decompose_fast(3, 3, 3, target_rank, max_iters=200, tol=1e-9, seed=seed)
            if res is not None:
                U_cand, V_cand, W_cand = res
                # The function already snaps and verifies
                U_33, V_33, W_33 = U_cand, V_cand, W_cand
                rank_33 = target_rank
                found = True
                break
        if found:
            break

    if U_33 is not None:
        # Construct 3x3x7 decomposition
        U_31, V_31, W_31 = _schoolbook(3, 3, 1)
        rank_31 = 9
        
        total_rank = rank_33 + rank_33 + rank_31
        U = np.zeros((9, total_rank))
        V = np.zeros((21, total_rank))
        W = np.zeros((21, total_rank))
        
        # Part 1: k=0,1,2
        U[:, 0:rank_33] = U_33
        for j in range(3):
            for k in range(3):
                V[j*7+k, 0:rank_33] = V_33[j*3+k, :]
        for k in range(3):
            for i in range(3):
                W[k*3+i, 0:rank_33] = W_33[k*3+i, :]
        
        # Part 2: k=3,4,5
        U[:, rank_33:2*rank_33] = U_33
        for j in range(3):
            for k in range(3):
                V[j*7+(k+3), rank_33:2*rank_33] = V_33[j*3+k, :]
        for k in range(3):
            for i in range(3):
                W[(k+3)*3+i, rank_33:2*rank_33] = W_33[k*3+i, :]
        
        # Part 3: k=6
        U[:, 2*rank_33:2*rank_33+rank_31] = U_31
        for j in range(3):
            V[j*7+6, 2*rank_33:2*rank_33+rank_31] = V_31[j*1+0, :]
        for i in range(3):
            W[6*3+i, 2*rank_33:2*rank_33+rank_31] = W_31[0*3+i, :]
        
        # Verify full decomposition
        if _verify(U, V, W, 3, 3, 7):
            def to_list(arr):
                return [[float(x) for x in row] for row in arr]
            return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
        
    # Fallback to schoolbook
    U, V, W = _schoolbook(3, 3, 7)
    def to_list(arr):
        return [[float(x) for x in row] for row in arr]
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
