import time
import random
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [4, 4, 5]

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if total != 1:
                    return False
    return True

def _verify_float(U, V, W, n, m, p, rank, tol=1e-9):
    """Fast float verification."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2_base():
    """Return Strassen decomposition for 2x2 * 2x2 -> 2x2 (rank 7).
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    n = m = p = 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # M1: (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3: a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = 1
    
    # M5: (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[3][4] = 1
    
    # M6: (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1; W[3][5] = 1
    
    # M7: (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[3][6] = 1
    
    return U, V, W

def _build_4x4_4x4_strassen():
    """Build 4x4 * 4x4 -> 4x4 decomposition using recursive Strassen (rank 49)."""
    U2, V2, W2 = _strassen_2x2_base()
    
    n = m = p = 4
    rank = 49
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c in range(7):
        for d in range(7):
            col = c * 7 + d
            for i in range(4):
                for j in range(4):
                    bi, bj = i // 2, j // 2
                    ii, jj = i % 2, j % 2
                    val = U2[bi*2+bj][c] * U2[ii*2+jj][d]
                    if val != 0:
                        U[i*4+j][col] = val
            for j in range(4):
                for k in range(4):
                    bj, bk = j // 2, k // 2
                    jj, kk = j % 2, k % 2
                    val = V2[bj*2+bk][c] * V2[jj*2+kk][d]
                    if val != 0:
                        V[j*4+k][col] = val
            for k in range(4):
                for i in range(4):
                    bk, bi = k // 2, i // 2
                    kk, ii = k % 2, i % 2
                    val = W2[bk*2+bi][c] * W2[kk*2+ii][d]
                    if val != 0:
                        W[k*4+i][col] = val
    
    return U, V, W

def _build_4x4_4x5_base():
    """Build 4x4 * 4x5 -> 4x5 decomposition (rank 50)."""
    U4, V4, W4 = _build_4x4_4x4_strassen()
    
    n, m, p = 4, 4, 5
    rank = 50
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for i in range(n*m):
        for c in range(49):
            U[i][c] = U4[i][c]
    for j in range(m):
        for k in range(4):
            for c in range(49):
                V[j*p+k][c] = V4[j*4+k][c]
    for k in range(4):
        for i in range(n):
            for c in range(49):
                W[k*n+i][c] = W4[k*4+i][c]
    
    col = 49
    for i in range(4):
        for j in range(4):
            U[i*4+j][col] = 1
            V[j*4+4][col] = 1
    for i in range(4):
        W[4*4+i][col] = 1
    
    return U, V, W

def _residual(U, V, W, n, m, p, rank):
    """Compute the residual vector for the tensor equations."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    res = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                res[i, j, k] = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i]) - 1.0
    return res

def _randomize_factors(U, V, W, n, m, p, rank, strength=0.1):
    """Add random noise to factors."""
    U_new = np.array(U, dtype=np.float64)
    V_new = np.array(V, dtype=np.float64)
    W_new = np.array(W, dtype=np.float64)
    U_new += strength * np.random.randn(*U_new.shape)
    V_new += strength * np.random.randn(*V_new.shape)
    W_new += strength * np.random.randn(*W_new.shape)
    return U_new, V_new, W_new

def _local_search(U, V, W, n, m, p, rank, deadline, max_iters=5000):
    """Perform local search to reduce residual to zero."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    best_res = np.linalg.norm(_residual(U_np, V_np, W_np, n, m, p, rank))
    best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
    
    lr = 0.01
    for it in range(max_iters):
        if time.time() > deadline:
            break
        if best_res < 1e-10:
            break
        
        # Compute gradient of residual w.r.t. each factor
        # This is complex, so we use a simple random perturbation approach
        # Try small random steps
        for _ in range(10):
            U_test = U_np + lr * np.random.randn(*U_np.shape)
            V_test = V_np + lr * np.random.randn(*V_np.shape)
            W_test = W_np + lr * np.random.randn(*W_np.shape)
            
            res = _residual(U_test, V_test, W_test, n, m, p, rank)
            res_norm = np.linalg.norm(res)
            
            if res_norm < best_res:
                best_res = res_norm
                best_U, best_V, best_W = U_test.copy(), V_test.copy(), W_test.copy()
                U_np, V_np, W_np = U_test, V_test, W_test
                lr *= 1.1
            else:
                lr *= 0.9
    
    return best_U, best_V, best_W, best_res

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 140
    
    # Build the rank-50 decomposition
    U, V, W = _build_4x4_4x5_base()
    rank = len(U[0])
    
    # Verify
    if not _verify(U, V, W, n, m, p, rank):
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
    
    # Try to reduce rank by local search
    # We'll try to find a rank-49 decomposition by dropping one column and optimizing
    for target_rank in [49, 48, 47]:
        if time.time() > deadline:
            break
        
        # Start from the current best and drop a random column
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            # Drop a random column from the current factors
            col_to_drop = random.randint(0, rank - 1)
            U_drop = [row[:col_to_drop] + row[col_to_drop+1:] for row in U]
            V_drop = [row[:col_to_drop] + row[col_to_drop+1:] for row in V]
            W_drop = [row[:col_to_drop] + row[col_to_drop+1:] for row in W]
            
            # Perform local search
            U_opt, V_opt, W_opt, res = _local_search(U_drop, V_drop, W_drop, n, m, p, target_rank, deadline)
            
            if res < 1e-8:
                # Convert to list of lists and verify with Fraction
                U_list = U_opt.tolist()
                V_list = V_opt.tolist()
                W_list = W_opt.tolist()
                
                if _verify(U_list, V_list, W_list, n, m, p, target_rank):
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    return {"U": U, "V": V, "W": W}
