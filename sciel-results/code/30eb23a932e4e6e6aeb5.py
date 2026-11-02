"""Rewritten program for matrix multiplication 2x2 by 2x5 tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 5]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin for the 150s limit

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=2, m=2, p=5):
    """
    Verify if the given U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                computed = 0.0
                for r in range(rank):
                    computed += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(computed - target) > 1e-9:
                    return False
    return True

def _build_target(n=2, m=2, p=5):
    """Build the target tensor T[u,v,w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0
    return T

def _als_solve_U(U, V, W, T, R, n, m, p):
    """Solve for U given V, W, T."""
    # For each u: T[u, v, w] = sum_r U[u,r] * V[v,r] * W[w,r]
    # So U[u,r] should satisfy: sum_r U[u,r] * (V[v,r]*W[w,r]) = T[u,v,w]
    # Let M_u[v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
    # We need to solve for U[u,:] given V, W
    # T[u] = U[u] @ (V^T * W^T) element-wise... 
    # Actually: T[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
    # Let B[v,w] = sum_r V[v,r] * W[w,r]  -- no, that's not right either
    # T[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
    # = (U[u] ⊙ (V*W elementwise summed over r))... no
    # For fixed u: T[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
    # This is a linear system in U[u,:]. Let's vectorize:
    # T_flat[u] = A_u @ U_flat[u] where A_u is (m*p*p*n) x R
    # A_u[v*p*n + w, r] = V[v,r] * W[w,r]
    npU = np.array(U, dtype=np.float64)
    npV = np.array(V, dtype=np.float64)
    npW = np.array(W, dtype=np.float64)
    npT = np.array(T, dtype=np.float64)
    
    for u in range(n*m):
        # Build the matrix A: (m*p * p*n) x R
        # A[v * (p*n) + w, r] = V[v, r] * W[w, r]
        A = np.einsum('vr,wr->vwr', npV, npW).reshape(m*p, p*n, R)
        A = A.transpose(0, 2, 1).reshape(m*p * p*n, R)
        b = npT[u].reshape(-1)
        sol, res, rank_s, sv = np.linalg.lstsq(A, b, rcond=None)
        npU[u] = sol
    
    U = npU.tolist()
    return U

def _als_solve_V(U, V, W, T, R, n, m, p):
    """Solve for V given U, W, T."""
    npU = np.array(U, dtype=np.float64)
    npV = np.array(V, dtype=np.float64)
    npW = np.array(W, dtype=np.float64)
    npT = np.array(T, dtype=np.float64)
    
    for v in range(m*p):
        # T[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
        # For fixed v: T[:,v,:] = (U * W) @ V[v,:]... 
        # A[u * (p*n) + w, r] = U[u,r] * W[w,r]
        A = np.einsum('ur,wr->uwr', npU, npW).reshape(n*m, p*n, R)
        A = A.transpose(0, 2, 1).reshape(n*m * p*n, R)
        b = npT[:, v, :].reshape(-1)
        sol, res, rank_s, sv = np.linalg.lstsq(A, b, rcond=None)
        npV[v] = sol
    
    V = npV.tolist()
    return V

def _als_solve_W(U, V, W, T, R, n, m, p):
    """Solve for W given U, V, T."""
    npU = np.array(U, dtype=np.float64)
    npV = np.array(V, dtype=np.float64)
    npW = np.array(W, dtype=np.float64)
    npT = np.array(T, dtype=np.float64)
    
    for w in range(p*n):
        # T[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
        # For fixed w: T[:,:,w] = (U * V) @ W[w,:]
        # A[u * (m*p) + v, r] = U[u,r] * V[v,r]
        A = np.einsum('ur,vr->uvr', npU, npV).reshape(n*m, m*p, R)
        A = A.transpose(0, 2, 1).reshape(n*m * m*p, R)
        b = npT[:, :, w].reshape(-1)
        sol, res, rank_s, sv = np.linalg.lstsq(A, b, rcond=None)
        npW[w] = sol
    
    W = npW.tolist()
    return W

def _als_iterate(U, V, W, T, R, n, m, p, max_iter=200, tol=1e-12):
    """Run ALS iterations."""
    prev_err = np.inf
    for it in range(max_iter):
        U = _als_solve_U(U, V, W, T, R, n, m, p)
        V = _als_solve_V(U, V, W, T, R, n, m, p)
        W = _als_solve_W(U, V, W, T, R, n, m, p)
        
        # Compute error
        npU = np.array(U)
        npV = np.array(V)
        npW = np.array(W)
        result = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(R):
            result += np.einsum('u,v,w->uvw', npU[:,r], npV[:,r], npW[:,r])
        err = np.linalg.norm(result - np.array(T))
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W, err

def _check_and_rationalize(U, V, W, T, R, n, m, p):
    """Check if decomposition is exact and try to rationalize entries."""
    npU = np.array(U)
    npV = np.array(V)
    npW = np.array(W)
    npT = np.array(T)
    
    result = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', npU[:,r], npV[:,r], npW[:,r])
    
    if not np.allclose(result, npT, atol=1e-8):
        return None
    
    # Try to rationalize with various denominators
    for denom in [1, 2, 4, 8, 16, 32]:
        U_s = np.round(npU * denom) / denom
        V_s = np.round(npV * denom) / denom
        W_s = np.round(npW * denom) / denom
        
        result_s = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(R):
            result_s += np.einsum('u,v,w->uvw', U_s[:,r], V_s[:,r], W_s[:,r])
        
        if np.allclose(result_s, npT, atol=1e-8):
            U_list = [[float(x) for x in row] for row in U_s]
            V_list = [[float(x) for x in row] for row in V_s]
            W_list = [[float(x) for x in row] for row in W_s]
            
            for mat in [U_list, V_list, W_list]:
                for i in range(len(mat)):
                    for j in range(len(mat[i])):
                        if abs(mat[i][j] - round(mat[i][j])) < 1e-10:
                            mat[i][j] = int(round(mat[i][j]))
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return as floats if valid
    U_list = [[float(x) for x in row] for row in npU]
    V_list = [[float(x) for x in row] for row in npV]
    W_list = [[float(x) for x in row] for row in npW]
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _construct_rank18(n=2, m=2, p=5):
    """
    Construct a rank-18 decomposition:
    - Strassen (rank 7) for B columns 0,1
    - Strassen (rank 7) for B columns 2,3
    - Schoolbook (rank 4) for B column 4
    """
    rank = 18
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1
        W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
        
        r = base + 2
        U[0][r] = 1
        V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
        W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 3
        U[3][r] = 1
        V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
        W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
        
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[1*p+col1][r] = 1
        W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
        
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
        W[col1*n+1][r] = 1
        
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1
    
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    # Schoolbook for column 4
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None

def _get_rank18_np(n=2, m=2, p=5):
    """Get rank-18 decomposition as numpy arrays."""
    rank = 18
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0, r] = 1; U[3, r] = 1
        V[0*p+col0, r] = 1; V[1*p+col1, r] = 1
        W[col0*n+0, r] = 1; W[col1*n+1, r] = 1
        
        r = base + 1
        U[2, r] = 1; U[3, r] = 1
        V[0*p+col0, r] = 1
        W[col0*n+1, r] = 1; W[col1*n+1, r] = -1
        
        r = base + 2
        U[0, r] = 1
        V[0*p+col1, r] = 1; V[1*p+col1, r] = -1
        W[col1*n+0, r] = 1; W[col1*n+1, r] = 1
        
        r = base + 3
        U[3, r] = 1
        V[1*p+col0, r] = 1; V[0*p+col0, r] = -1
        W[col0*n+0, r] = 1; W[col0*n+1, r] = 1
        
        r = base + 4
        U[0, r] = 1; U[1, r] = 1
        V[1*p+col1, r] = 1
        W[col0*n+0, r] = -1; W[col1*n+0, r] = 1
        
        r = base + 5
        U[2, r] = 1; U[0, r] = -1
        V[0*p+col0, r] = 1; V[0*p+col1, r] = 1
        W[col1*n+1, r] = 1
        
        r = base + 6
        U[1, r] = 1; U[3, r] = -1
        V[1*p+col0, r] = 1; V[1*p+col1, r] = 1
        W[col0*n+0, r] = 1
    
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    # Schoolbook for column 4
    r = 14
    U[0, r] = 1; V[0*p+4, r] = 1; W[4*n+0, r] = 1
    r = 15
    U[1, r] = 1; V[1*p+4, r] = 1; W[4*n+0, r] = 1
    r = 16
    U[2, r] = 1; V[0*p+4, r] = 1; W[4*n+1, r] = 1
    r = 17
    U[3, r] = 1; V[1*p+4, r] = 1; W[4*n+1, r] = 1
    
    return U, V, W

def _search_rank17(n=2, m=2, p=5, timeout=60):
    """
    Search for a rank-17 decomposition by perturbing the rank-18 structure.
    """
    start_time = time.time()
    T = _build_target(n, m, p)
    R = 17
    
    for trial in range(40):
        if time.time() - start_time > timeout - 5:
            break
        
        if trial < 10:
            # Start from rank-18, drop last column, add noise
            U18, V18, W18 = _get_rank18_np(n, m, p)
            U = (U18[:, :17] + np.random.randn(n*m, 17) * 0.3).tolist()
            V = (V18[:, :17] + np.random.randn(m*p, 17) * 0.3).tolist()
            W = (W18[:, :17] + np.random.randn(p*n, 17) * 0.3).tolist()
        elif trial < 20:
            # Drop a middle schoolbook term
            U18, V18, W18 = _get_rank18_np(n, m, p)
            cols = list(range(14)) + [15, 16, 17]
            U = (U18[:, cols] + np.random.randn(n*m, 17) * 0.2).tolist()
            V = (V18[:, cols] + np.random.randn(m*p, 17) * 0.2).tolist()
            W = (W18[:, cols] + np.random.randn(p*n, 17) * 0.2).tolist()
        elif trial < 30:
            # Random perturbation of rank-18 with varying noise
            U18, V18, W18 = _get_rank18_np(n, m, p)
            noise_scale = 0.3 + 0.05 * (trial - 20)
            U = (U18[:, :17] + np.random.randn(n*m, 17) * noise_scale).tolist()
            V = (V18[:, :17] + np.random.randn(m*p, 17) * noise_scale).tolist()
            W = (W18[:, :17] + np.random.randn(p*n, 17) * noise_scale).tolist()
        else:
            # Pure random
            U = (np.random.randn(n*m, R) * 0.5).tolist()
            V = (np.random.randn(m*p, R) * 0.5).tolist()
            W = (np.random.randn(p*n, R) * 0.5).tolist()
        
        U, V, W, err = _als_iterate(U, V, W, T, R, n, m, p, max_iter=200)
        
        result = _check_and_rationalize(U, V, W, T, R, n, m, p)
        if result is not None:
            return result
    
    return None

def _search_rank16(n=2, m=2, p=5, timeout=40):
    """
    Search for a rank-16 decomposition.
    """
    start_time = time.time()
    T = _build_target(n, m, p)
    R = 16
    
    for trial in range(20):
        if time.time() - start_time > timeout - 5:
            break
        
        if trial < 8:
            # Start from rank-18, drop 2 columns
            U18, V18, W18 = _get_rank18_np(n, m, p)
            U = (U18[:, :16] + np.random.randn(n*m, 16) * 0.3).tolist()
            V = (V18[:, :16] + np.random.randn(m*p, 16) * 0.3).tolist()
            W = (W18[:, :16] + np.random.randn(p*n, 16) * 0.3).tolist()
        elif trial < 14:
            # Drop schoolbook terms for col 4
            U18, V18, W18 = _get_rank18_np(n, m, p)
            cols = list(range(14)) + [15, 16]
            U = (U18[:, cols] + np.random.randn(n*m, 16) * 0.3).tolist()
            V = (V18[:, cols] + np.random.randn(m*p, 16) * 0.3).tolist()
            W = (W18[:, cols] + np.random.randn(p*n, 16) * 0.3).tolist()
        else:
            # Pure random
            U = (np.random.randn(n*m, R) * 0.5).tolist()
            V = (np.random.randn(m*p, R) * 0.5).tolist()
            W = (np.random.randn(p*n, R) * 0.5).tolist()
        
        U, V, W, err = _als_iterate(U, V, W, T, R, n, m, p, max_iter=200)
        
        result = _check_and_rationalize(U, V, W, T, R, n, m, p)
        if result is not None:
            return result
    
    return None

def _search_rank15(n=2, m=2, p=5, timeout=30):
    """
    Search for a rank-15 decomposition.
    """
    start_time = time.time()
    T = _build_target(n, m, p)
    R = 15
    
    for trial in range(10):
        if time.time() - start_time > timeout - 5:
            break
        
        if trial < 5:
            # Start from rank-18, drop 3 columns
            U18, V18, W18 = _get_rank18_np(n, m, p)
            U = (U18[:, :15] + np.random.randn(n*m, 15) * 0.3).tolist()
            V = (V18[:, :15] + np.random.randn(m*p, 15) * 0.3).tolist()
            W = (W18[:, :15] + np.random.randn(p*n, 15) * 0.3).tolist()
        else:
            # Pure random
            U = (np.random.randn(n*m, R) * 0.5).tolist()
            V = (np.random.randn(m*p, R) * 0.5).tolist()
            W = (np.random.randn(p*n, R) * 0.5).tolist()
        
        U, V, W, err = _als_iterate(U, V, W, T, R, n, m, p, max_iter=150)
        
        result = _check_and_rationalize(U, V, W, T, R, n, m, p)
        if result is not None:
            return result
    
    return None

def _construct_schoolbook(n=2, m=2, p=5):
    """
    Construct the standard schoolbook decomposition of rank n*m*p.
    """
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

def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT
    
    # Try rank 17 search first
    if time.time() < deadline - 80:
        result = _search_rank17(n, m, p, timeout=min(45, deadline - time.time() - 80))
        if result is not None:
            return result
    
    # Try rank 16 search
    if time.time() < deadline - 50:
        result = _search_rank16(n, m, p, timeout=min(30, deadline - time.time() - 50))
        if result is not None:
            return result
    
    # Try rank 15 search
    if time.time() < deadline - 25:
        result = _search_rank15(n, m, p, timeout=min(15, deadline - time.time() - 25))
        if result is not None:
            return result
    
    # Try rank 18 (deterministic)
    result = _construct_rank18(n, m, p)
    if result is not None:
        return result
    
    # Fallback: Schoolbook (rank 20)
    return _construct_schoolbook(n, m, p)

# EVOLVE-BLOCK-END
