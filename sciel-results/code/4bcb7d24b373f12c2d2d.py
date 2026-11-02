import time
import random
import numpy as np

def _build_strassen_rank25():
    """Build the rank-25 Strassen decomposition for 2x2 * 2x7."""
    n, m, p = 2, 2, 7
    rank = 25
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # For each pair of output columns (ks, ks+1), apply 7 Strassen multiplications
    # on the 2x2 x 2x2 sub-block
    # A is 2x2: A[i,j] indexed by i (row of A), j (col of A = row of B)
    # B is 2x2: B[j,k] indexed by j (row of B), k (col of B)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Strassen 7-multiplication scheme for 2x2*2x2:
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = (A[1,0]+A[1,1]) * B[1,0]
    # M4 = A[0,0] * (B[1,1]-B[0,1])
    # M5 = A[1,1] * (B[0,1]-B[0,0])
    # M6 = (A[0,0]+A[0,1]) * B[1,0]
    # M7 = (A[0,1]-A[1,1]) * (B[0,0]+B[1,1])
    #
    # C[0,0] = M1 - M5 + M7
    # C[0,1] = M4 + M5
    # C[1,0] = M2 - M3 + M6
    # C[1,1] = M1 - M2 + M3
    
    def add_strassen_block(col_start, k_start):
        ks = k_start
        ks1 = k_start + 1
        # B[0,ks] is V index 7*0+ks, B[1,ks] is V index 7*1+ks
        # B[0,ks1] is V index 7*0+ks1, B[1,ks1] is V index 7*1+ks1
        # A[0,0] is U index 0, A[0,1] is U index 1
        # A[1,0] is U index 2, A[1,1] is U index 3
        # C[i,ks] is W index 2*ks+i, C[i,ks1] is W index 2*ks1+i
        
        c = col_start
        # M1 = (A00+A11)(B0,ks + B1,ks1)
        U[0, c] += 1
        U[3, c] += 1
        V[0*7+ks, c] += 1
        V[1*7+ks1, c] += 1
        W[2*ks+0, c] += 1   # C[0,ks]
        W[2*ks+1, c] += 1   # C[1,ks]
        W[2*ks1+0, c] -= 1  # C[0,ks1]
        W[2*ks1+1, c] -= 1  # C[1,ks1]
        
        # M2 = (A10+A11) * B0,ks
        U[2, c+1] += 1
        U[3, c+1] += 1
        V[0*7+ks, c+1] += 1
        W[2*ks+1, c+1] += 1   # C[1,ks]
        W[2*ks1+1, c+1] -= 1  # C[1,ks1]
        
        # M3 = (A10+A11) * B1,ks
        U[2, c+2] += 1
        U[3, c+2] += 1
        V[1*7+ks, c+2] += 1
        W[2*ks+1, c+2] -= 1   # C[1,ks]
        W[2*ks1+1, c+2] += 1  # C[1,ks1]
        
        # M4 = A00 * (B1,ks1 - B0,ks1)
        U[0, c+3] += 1
        V[1*7+ks1, c+3] += 1
        V[0*7+ks1, c+3] -= 1
        W[2*ks1+0, c+3] += 1  # C[0,ks1]
        
        # M5 = A11 * (B0,ks1 - B0,ks)
        U[3, c+4] += 1
        V[0*7+ks1, c+4] += 1
        V[0*7+ks, c+4] -= 1
        W[2*ks+0, c+4] -= 1   # C[0,ks]
        W[2*ks1+0, c+4] += 1  # C[0,ks1]
        
        # M6 = (A00+A01) * B1,ks
        U[0, c+5] += 1
        U[1, c+5] += 1
        V[1*7+ks, c+5] += 1
        W[2*ks+1, c+5] += 1   # C[1,ks]
        
        # M7 = (A01-A11) * (B0,ks + B1,ks1)
        U[1, c+6] += 1
        U[3, c+6] -= 1
        V[0*7+ks, c+6] += 1
        V[1*7+ks1, c+6] += 1
        W[2*ks+0, c+6] += 1   # C[0,ks]
    
    add_strassen_block(0, 0)
    add_strassen_block(7, 2)
    add_strassen_block(14, 4)
    
    # Schoolbook for column 6 (only one column, so 4 multiplications)
    col_start = 21
    # C[0,6] = A[0,0]*B[0,6] + A[0,1]*B[1,6]
    U[0, col_start+0] += 1
    V[0*7+6, col_start+0] += 1
    W[2*6+0, col_start+0] += 1
    U[1, col_start+1] += 1
    V[1*7+6, col_start+1] += 1
    W[2*6+0, col_start+1] += 1
    # C[1,6] = A[1,0]*B[0,6] + A[1,1]*B[1,6]
    U[2, col_start+2] += 1
    V[0*7+6, col_start+2] += 1
    W[2*6+1, col_start+2] += 1
    U[3, col_start+3] += 1
    V[1*7+6, col_start+3] += 1
    W[2*6+1, col_start+3] += 1
    
    return U.tolist(), V.tolist(), W.tolist()

def _verify(U, V, W, n=2, m=2, p=7):
    """Verify tensor decomposition: for each (i,j,k), sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] == 1."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _try_rank_reduction(U, V, W, target_rank, time_budget):
    """
    Try to reduce rank from current to target_rank by structured search.
    Uses the known Strassen structure and tries small perturbations.
    """
    start_time = time.time()
    n, m, p = 2, 2, 7
    current_rank = len(U[0])
    
    # Strategy: try random small modifications to the rank-25 solution
    # to see if we can get rank 24 by merging two terms
    
    # First, try direct random search with small integer entries
    for size in [3, 5]:
        if time.time() - start_time > time_budget:
            break
        
        low = -size // 2
        high = (size - 1) // 2
        
        max_attempts = 3000
        for attempt in range(max_attempts):
            if time.time() - start_time > time_budget:
                break
            
            U_rand = [[random.randint(low, high) for _ in range(target_rank)] for _ in range(n*m)]
            V_rand = [[random.randint(low, high) for _ in range(target_rank)] for _ in range(m*p)]
            W_rand = [[random.randint(low, high) for _ in range(target_rank)] for _ in range(p*n)]
            
            if _verify(U_rand, V_rand, W_rand, n, m, p):
                return U_rand, V_rand, W_rand
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Step 1: Build rank-25 Strassen decomposition
    U25, V25, W25 = _build_strassen_rank25()
    
    # Verify
    if not _verify(U25, V25, W25):
        # Fallback to schoolbook
        n, m, p = 2, 2, 7
        rank = 28
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_U, best_V, best_W = U25, V25, W25
    best_rank = 25
    
    # Step 2: Try to find rank 24
    if time.time() < deadline - 30:
        time_budget = min(deadline - 30 - time.time(), 60)
        result = _try_rank_reduction(U25, V25, W25, 24, time_budget)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 24
    
    # Step 3: If still time, try rank 23
    if best_rank == 24 and time.time() < deadline - 20:
        time_budget = min(deadline - 20 - time.time(), 40)
        result = _try_rank_reduction(best_U, best_V, best_W, 23, time_budget)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 23
    
    return {"U": best_U, "V": best_V, "W": best_W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    A = np.random.rand(2, 2)
    B = np.random.rand(2, 7)
    C_expected = A @ B
    
    C_W = np.zeros(14)
    for r in range(U.shape[1]):
        SA = np.dot(U[:, r], A.flatten())
        SB = np.dot(V[:, r], B.flatten())
        for idx in range(14):
            C_W[idx] += W[idx, r] * SA * SB
    
    C_mat = np.zeros((2, 7))
    for k in range(7):
        for i in range(2):
            C_mat[i, k] = C_W[2*k+i]
    
    if np.allclose(C_mat, C_expected):
        print("Verification PASSED, rank =", U.shape[1])
    else:
        print("Verification FAILED")
        print(C_mat)
        print(C_expected)
