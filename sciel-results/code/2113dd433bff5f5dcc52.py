"""Search for a rank-minimal exact tensor decomposition of 4x4 * 4x5 matrix multiplication.
Uses block decomposition with Strassen (rank 7) for 2x2x2 and hybrid/search (rank 11 or 9) for 2x2x3.
Falls back to schoolbook (rank 80) if no better decomposition is found in time."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify_fast(U, V, W, n=4, m=4, p=5):
    """Fast verification using numpy with float64, checking tolerance."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = Ur[:, :, c]
        vc = Vr[:, :, c]
        wc = Wr[:, :, c]
        wcT = wc.T
        for j in range(m):
            T[:, j, :] = T[:, j, :] + np.outer(uc[:, j], vc[j, :]) * wcT
    return np.allclose(T, 1.0, atol=1e-9)

def _strassen_2x2():
    """Return Strassen decomposition for 2x2 * 2x2 as factor matrices.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    U = np.zeros((4, 7), dtype=np.float64)
    U[0, 0] = 1; U[0, 2] = 1; U[0, 4] = 1; U[0, 5] = -1
    U[1, 2] = 1; U[1, 4] = 1; U[1, 6] = 1
    U[2, 1] = 1; U[2, 5] = 1
    U[3, 0] = 1; U[3, 1] = 1; U[3, 3] = 1; U[3, 5] = 1; U[3, 6] = -1
    
    V = np.zeros((4, 7), dtype=np.float64)
    V[0, 0] = 1; V[0, 1] = 1; V[0, 3] = -1; V[0, 5] = 1
    V[1, 0] = 1; V[1, 2] = -1; V[1, 5] = 1
    V[2, 0] = 1; V[2, 3] = 1; V[2, 6] = 1
    V[3, 0] = 1; V[3, 4] = 1; V[3, 6] = 1
    
    W = np.zeros((4, 7), dtype=np.float64)
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    W[1, 2] = 1; W[1, 4] = 1
    W[2, 1] = 1; W[2, 3] = 1
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W

def _hybrid_2x2x3():
    """Construct a rank-11 decomposition for 2x2 * 2x3.
    Uses Strassen (rank 7) for first 2 cols and Schoolbook (rank 4) for 3rd col.
    U: (4, 11), V: (6, 11), W: (6, 11)
    """
    U2, V2, W2 = _strassen_2x2()
    U = np.zeros((4, 11), dtype=np.float64)
    V = np.zeros((6, 11), dtype=np.float64)
    W = np.zeros((6, 11), dtype=np.float64)
    
    # Copy Strassen part (cols 0, 1 of B, cols 0, 1 of C)
    U[:, :7] = U2
    # V2 is (4, 7) for B (2x2). V is (6, 11) for B (2x3).
    # V[j*2+k, c] -> V[j*3+k, c]
    V[0, :7] = V2[0, :7] # j=0, k=0
    V[1, :7] = V2[1, :7] # j=0, k=1
    V[2, :7] = V2[2, :7] # j=1, k=0
    V[3, :7] = V2[3, :7] # j=1, k=1
    # V[4, :7] and V[5, :7] are 0 (k=2)
    
    # W2 is (4, 7) for C (2x2). W is (6, 11) for C (2x3).
    # W[k*2+i, c] -> W[k*2+i, c] ... wait.
    # For 2x2x2, W index is k*2+i.
    # For 2x2x3, W index is k*2+i.
    # k=0,1 from Strassen. k=2 from Schoolbook.
    W[0, :7] = W2[0, :7] # k=0, i=0
    W[1, :7] = W2[1, :7] # k=0, i=1
    W[2, :7] = W2[2, :7] # k=1, i=0
    W[3, :7] = W2[3, :7] # k=1, i=1
    # W[4, :7] and W[5, :7] are 0 (k=2)
    
    # Add Schoolbook part for col 2 (k=2)
    # C[i, 2] = A[i, 0]B[0, 2] + A[i, 1]B[1, 2]
    # i=0, j=0, k=2
    U[0, 7] = 1
    V[2, 7] = 1 # j=0, k=2 -> index 0*3+2 = 2
    W[4, 7] = 1 # k=2, i=0 -> index 2*2+0 = 4
    
    # i=0, j=1, k=2
    U[1, 8] = 1
    V[5, 8] = 1 # j=1, k=2 -> index 1*3+2 = 5
    W[4, 8] = 1
    
    # i=1, j=0, k=2
    U[2, 9] = 1
    V[2, 9] = 1
    W[5, 9] = 1 # k=2, i=1 -> index 2*2+1 = 5
    
    # i=1, j=1, k=2
    U[3, 10] = 1
    V[5, 10] = 1
    W[5, 10] = 1
    
    return U, V, W

def _search_2x2x3_rank9():
    """Attempt to find a rank-9 decomposition for 2x2 * 2x3.
    Returns None if not found."""
    np.random.seed(42)
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Try a few random initializations
    for _ in range(5):
        U = np.random.randn(4, 9) * 0.1
        V = np.random.randn(6, 9) * 0.1
        W = np.random.randn(6, 9) * 0.1
        
        lr = 0.01
        for iter in range(100):
            # Compute errors
            U_err = np.zeros_like(U)
            V_err = np.zeros_like(V)
            W_err = np.zeros_like(W)
            
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        s = np.sum(U[i*2+j, :] * V[j*3+k, :] * W[k*2+i, :])
                        error = s - 1.0
                        U_err[i*2+j, :] -= lr * error * V[j*3+k, :] * W[k*2+i, :]
                        V_err[j*3+k, :] -= lr * error * U[i*2+j, :] * W[k*2+i, :]
                        W_err[k*2+i, :] -= lr * error * U[i*2+j, :] * V[j*3+k, :]
            
            U += U_err
            V += V_err
            W += W_err
            
            # Check error
            max_err = 0
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        s = np.sum(U[i*2+j, :] * V[j*3+k, :] * W[k*2+i, :])
                        max_err = max(max_err, abs(s - 1.0))
            
            if max_err < best_err:
                best_err = max_err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            if max_err < 1e-5:
                # Try to round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                max_err_r = 0
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            s = np.sum(U_r[i*2+j, :] * V_r[j*3+k, :] * W_r[k*2+i, :])
                            max_err_r = max(max_err_r, abs(s - 1.0))
                            
                if max_err_r < 1e-9:
                    return U_r, V_r, W_r
                else:
                    # Return the float solution if it's good enough, but we need exact.
                    # If we can't find exact, we might have to fail.
                    pass
    
    return None

def _assemble_4x4x5(sub_222, sub_223):
    """Assemble 4x4x5 from 4 copies of 2x2x2 and 4 copies of 2x2x3."""
    U222, V222, W222 = sub_222
    U223, V223, W223 = sub_223
    
    rank_222 = U222.shape[1]
    rank_223 = U223.shape[1]
    total_rank = 4 * rank_222 + 4 * rank_223
    
    U = np.zeros((16, total_rank), dtype=np.float64)
    V = np.zeros((20, total_rank), dtype=np.float64)
    W = np.zeros((20, total_rank), dtype=np.float64)
    
    col = 0
    for bi in range(2):
        for bj in range(2):
            # Block (bi, bj) for A and B
            # A rows: bi*2 .. bi*2+1, cols: bj*2 .. bj*2+1
            # B rows: bj*2 .. bj*2+1, cols: 0..4
            
            # 1. 2x2x2 part (k=0,1)
            for c in range(rank_222):
                for ii in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            # U index: (bi*2+ii)*4 + (bj*2+jj)
                            U[(bi*2+ii)*4 + (bj*2+jj), col+c] = U222[ii*2+jj, c]
                            # V index: (bj*2+jj)*5 + kk
                            V[(bj*2+jj)*5 + kk, col+c] = V222[jj*2+kk, c]
                            # W index: kk*4 + (bi*2+ii)
                            W[kk*4 + (bi*2+ii), col+c] = W222[kk*2+ii, c]
            col += rank_222
            
            # 2. 2x2x3 part (k=2,3,4)
            for c in range(rank_223):
                for ii in range(2):
                    for jj in range(2):
                        for kk in range(3):
                            # U index: (bi*2+ii)*4 + (bj*2+jj)
                            U[(bi*2+ii)*4 + (bj*2+jj), col+c] = U223[ii*2+jj, c]
                            # V index: (bj*2+jj)*5 + (2+kk)
                            V[(bj*2+jj)*5 + (2+kk), col+c] = V223[jj*3+kk, c]
                            # W index: (2+kk)*4 + (bi*2+ii)
                            W[(2+kk)*4 + (bi*2+ii), col+c] = W223[kk*2+ii, c]
            col += rank_223
            
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    # 1. Get Strassen 2x2x2
    sub_222 = _strassen_2x2()
    
    # 2. Try to get Rank 9 2x2x3
    sub_223 = None
    if time.time() < deadline - 20:
        try:
            sub_223 = _search_2x2x3_rank9()
        except Exception:
            sub_223 = None
            
    if sub_223 is None:
        # Fallback to Hybrid (rank 11)
        sub_223 = _hybrid_2x2x3()
        
    # 3. Assemble
    try:
        U, V, W = _assemble_4x4x5(sub_222, sub_223)
        if _verify_fast(U, V, W, 4, 4, 5):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
        
    # 4. Fallback to Schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
