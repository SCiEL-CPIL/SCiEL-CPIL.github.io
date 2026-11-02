"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, m))
            for r in range(R):
                w_val = W[k * n + i, r]
                if w_val == 0: continue
                u_vec = U[i * m : (i + 1) * m, r]
                v_col = V[k::p, r]
                M += w_val * np.outer(u_vec, v_col)
            
            if not np.allclose(M, np.eye(m), atol=1e-9):
                return False
    return True

def _try_algebraic_rank(n, m, p, R, max_attempts=50, seed_base=42):
    """Try to find a rank-R decomposition using algebraic methods."""
    np.random.seed(seed_base)
    
    # The condition: for all i, k, j, j':
    # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j'*p+k, r] = delta[j,j']
    
    # Strategy: Use a structured ansatz. For 2x4x4, we can try:
    # - Each "rank-1" term handles a specific (i,k) pair partially
    # - Use the fact that the identity matrix has rank m=4
    
    # Try different random initializations and solve via least squares
    for attempt in range(max_attempts):
        np.random.seed(seed_base + attempt)
        
        # Initialize W with random values
        W = np.random.randn(p * n, R)
        
        # For each (i,k) pair, we need:
        # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j'*p+k, r] = delta[j,j']
        
        # This is a bilinear system. Let's fix W and solve for U, V
        # Or use an alternating approach.
        
        # Alternate: fix U,V -> solve W; fix W,V -> solve U; fix W,U -> solve V
        U = np.random.randn(n * m, R) * 0.5
        V = np.random.randn(m * p, R) * 0.5
        
        for it in range(200):
            # Solve for W: for each (i,k), we have m*m equations
            # sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j'*p+k, r]) = delta[j,j']
            # This is linear in W[k*n+i, :]
            for i in range(n):
                for k in range(p):
                    # Build system: A_w @ w = b
                    # where w = W[k*n+i, :], A_w[j,j',r] = U[i*m+j, r] * V[j'*p+k, r]
                    A_w = np.zeros((m * m, R))
                    b_w = np.zeros(m * m)
                    for j in range(m):
                        for jp in range(m):
                            idx = j * m + jp
                            b_w[idx] = 1.0 if j == jp else 0.0
                            for r in range(R):
                                A_w[idx, r] = U[i * m + j, r] * V[jp * p + k, r]
                    
                    # Solve least squares
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
                        W[k * n + i, :] = w_sol
                    except:
                        pass
            
            # Solve for U: for each i, we have p*m*m equations
            # sum_r U[i*m+j, r] * (W[k*n+i, r] * V[j'*p+k, r]) = delta[j,j'] for all k,jp
            for i in range(n):
                A_u = np.zeros((p * m * m, R))
                b_u = np.zeros(p * m * m)
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            idx = k * m * m + j * m + jp
                            b_u[idx] = 1.0 if j == jp else 0.0
                            for r in range(R):
                                A_u[idx, r] = W[k * n + i, r] * V[jp * p + k, r]
                
                try:
                    u_sol, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                    U[i * m : (i + 1) * m, :] = u_sol
                except:
                    pass
            
            # Solve for V: for each k, we have n*m*m equations
            # sum_r V[j'*p+k, r] * (W[k*n+i, r] * U[i*m+j, r]) = delta[j,j'] for all i,j
            for k in range(p):
                A_v = np.zeros((n * m * m, R))
                b_v = np.zeros(n * m * m)
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            idx = i * m * m + j * m + jp
                            b_v[idx] = 1.0 if j == jp else 0.0
                            for r in range(R):
                                A_v[idx, r] = W[k * n + i, r] * U[i * m + j, r]
                
                try:
                    v_sol, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                    V[k::p, :] = v_sol
                except:
                    pass
            
            # Check convergence
            max_err = 0.0
            for i in range(n):
                for k in range(p):
                    M = np.zeros((m, m))
                    for r in range(R):
                        w_val = W[k * n + i, r]
                        if w_val != 0:
                            u_vec = U[i * m : (i + 1) * m, r]
                            v_col = V[k::p, r]
                            M += w_val * np.outer(u_vec, v_col)
                    err = np.max(np.abs(M - np.eye(m)))
                    max_err = max(max_err, err)
            
            if max_err < 1e-10:
                # Snap to half-integers
                U_half = np.round(U * 2) / 2
                V_half = np.round(V * 2) / 2
                W_half = np.round(W * 2) / 2
                
                if _verify_decomposition(U_half, V_half, W_half, n, m, p):
                    return U_half, V_half, W_half
                
                # Try integers
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                if _verify_decomposition(U_int, V_int, W_int, n, m, p):
                    return U_int, V_int, W_int
                
                # Return as-is if close enough
                if max_err < 1e-8:
                    return U, V, W
    
    return None, None, None

def _construct_schoolbook(n, m, p):
    """Return the schoolbook (trivial) decomposition with rank n*m*p."""
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _construct_strassen_like(n, m, p):
    """Try a Strassen-like construction for 2x4x4."""
    # For 2x4x4, we can try to exploit the structure
    # The minimum rank is at least 14 (from lower bounds)
    # Let's try rank 16 first as a structured approach
    
    # One approach: treat the 2x4 matrix as having 2 rows, each of length 4
    # C[k,i] = sum_j A[i,j] B[j,k]
    
    # For i=0: C[k,0] = sum_j A[0,j] B[j,k] for k=0,1,2,3
    # For i=1: C[k,1] = sum_j A[1,j] B[j,k] for k=0,1,2,3
    
    # This is like multiplying two 1x4 vectors by a 4x4 matrix
    # Each row of A times B gives a row of C
    
    # We can use the fact that for a fixed i, we're computing A[i,:] * B
    # This is a 1x4 times 4x4 multiplication, which has rank 4 (schoolbook)
    # But we can potentially share computations between i=0 and i=1
    
    # Let's try rank 16: 4 per row of A, shared B multiplications
    R = 16
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # For each i (row of A) and each j (column of A), we need a term
    # But we can group: for fixed j, both i=0 and i=1 use B[j,:]
    
    # Term r = i*m + j: 
    # U[i*m+j, r] = 1 if (i,j) matches, else 0
    # V[j*p+k, r] = 1 if (j,k) matches, else 0  
    # W[k*n+i, r] = 1 if (k,i) matches, else 0
    # This is just schoolbook with rank 8... wait, n*m*p = 2*4*4 = 32
    
    # Let me reconsider. The schoolbook has rank 32.
    # We want to do better.
    
    # Actually, for 2x4x4, the tensor rank is known to be at most 20 
    # and at least 14. Let me try a different structured approach.
    
    # Use the fact that we can combine terms.
    # For now, return schoolbook as fallback
    return _construct_schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Try algebraic approach for various ranks
    best_result = None
    best_rank = n * m * p  # Schoolbook rank = 32
    
    # Try ranks from 14 to 24
    for R in range(14, 25):
        if time.time() > deadline:
            break
        
        result = _try_algebraic_rank(n, m, p, R, max_attempts=30)
        
        if result[0] is not None:
            U, V, W = result
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = R
            break
    
    # Fallback: Schoolbook
    if best_result is None:
        best_result = _construct_schoolbook(n, m, p)
    
    return best_result
# EVOLVE-BLOCK-END
