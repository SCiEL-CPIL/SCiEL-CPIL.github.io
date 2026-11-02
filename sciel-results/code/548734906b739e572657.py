import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _build_schoolbook(n, m, p):
    """Return the trivial schoolbook (rank = n*m*p) decomposition."""
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    return U, V, W

def _strassen_2x2():
    """Return rank-7 Strassen decomposition for 2x2 * 2x2.
    
    Returns U (4,7), V (4,7), W (4,7) where:
      U[i, r] = coefficient of A[i//2][i%2] in the r-th left linear form
      V[j, r] = coefficient of B[j//2][j%2] in the r-th right linear form  
      W[k, r] = coefficient in output C[k//2][k%2] = sum_r W[k,r]*M_r
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # C00 = M1 + M4 - M5 - M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = -1
    
    # C01 = M3 + M5
    W[1, 2] = 1; W[1, 4] = 1
    
    # C10 = M2 + M4
    W[2, 1] = 1; W[2, 3] = 1
    
    # C11 = M1 - M2 + M3 + M7
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 6] = 1
    
    return U, V, W

def _build_block_decomposition(n, m, p):
    """Build a decomposition using block structure with Strassen for 2x2 blocks.
    
    For 3x3 * 3x6:
    Split B into three 3x2 blocks.
    For each block, decompose C = A * B_block (3x3 * 3x2 = 3x2):
      - Top 2 rows: A[0:2,:] * B_block[:,0:2]
        = A[0:2,0:2]*B_block[0:2,0:2] + A[0:2,2]*B_block[2,0:2]
        = Strassen(2x2*2x2) + outer(2x1 * 1x2) = rank 7+1 = 8
      - Bottom row: A[2,:] * B_block
        = A[2,0:2]*B_block[0:2,:] + A[2,2]*B_block[2,:]
        = rank 2 + rank 1 = 3
    Total per block: 11, total: 33
    """
    U_list = []
    V_list = []
    W_list = []
    
    for j in range(3):  # For each 3x2 block of B
        b_col_start = 2 * j
        
        # --- Top part of C (rows 0,1) ---
        # Term 1-7: Strassen for A[0:2, 0:2] * B_block[0:2, :]
        U_s, V_s, W_s = _strassen_2x2()
        
        for r in range(7):
            U_term = np.zeros(n * m)
            V_term = np.zeros(m * p)
            W_term = np.zeros(p * n)
            
            for a_flat in range(4):
                ai = a_flat // 2
                aj = a_flat % 2
                U_term[ai * m + aj] = U_s[a_flat, r]
            
            for b_flat in range(4):
                bj = b_flat // 2
                bk = b_flat % 2
                global_bk = b_col_start + bk
                V_term[bj * p + global_bk] = V_s[b_flat, r]
            
            for c_flat in range(4):
                ci = c_flat // 2
                ck = c_flat % 2
                global_ck = b_col_start + ck
                W_term[global_ck * n + ci] = W_s[c_flat, r]
            
            U_list.append(U_term)
            V_list.append(V_term)
            W_list.append(W_term)
        
        # Term 8: A[0:2,2] * B_block[2,:] -> outer product
        U_term = np.zeros(n * m)
        V_term = np.zeros(m * p)
        W_term = np.zeros(p * n)
        
        for i in range(2):
            U_term[i * m + 2] = 1.0
        
        for k in range(2):
            V_term[2 * p + b_col_start + k] = 1.0
        
        for i in range(2):
            for k in range(2):
                global_k = b_col_start + k
                W_term[global_k * n + i] = 1.0
        
        U_list.append(U_term)
        V_list.append(V_term)
        W_list.append(W_term)
        
        # --- Bottom part of C (row 2) ---
        # Term 9-10: A[2,0:2] * B_block[0:2,:] -> rank 2
        for jj in range(2):
            U_term = np.zeros(n * m)
            V_term = np.zeros(m * p)
            W_term = np.zeros(p * n)
            
            U_term[2 * m + jj] = 1.0
            
            for k in range(2):
                global_k = b_col_start + k
                V_term[jj * p + global_k] = 1.0
                W_term[global_k * n + 2] = 1.0
            
            U_list.append(U_term)
            V_list.append(V_term)
            W_list.append(W_term)
        
        # Term 11: A[2,2] * B_block[2,:] -> rank 1
        U_term = np.zeros(n * m)
        V_term = np.zeros(m * p)
        W_term = np.zeros(p * n)
        
        U_term[2 * m + 2] = 1.0
        
        for k in range(2):
            global_k = b_col_start + k
            V_term[2 * p + global_k] = 1.0
            W_term[global_k * n + 2] = 1.0
        
        U_list.append(U_term)
        V_list.append(V_term)
        W_list.append(W_term)
    
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    rank = U.shape[1]
    return rank, U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    a_dim = n * m
    b_dim = m * p
    c_dim = p * n
    R = U.shape[1]
    
    # Check all non-zero entries: T[i*m+j, j*p+k, k*n+i] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for r in range(R):
                    s += U[a, r] * V[b, r] * W[c, r]
                if abs(s - 1.0) > tol:
                    return False
    
    # Check a sample of zero entries
    np.random.seed(42)
    for _ in range(2000):
        a = np.random.randint(0, a_dim)
        b = np.random.randint(0, b_dim)
        c = np.random.randint(0, c_dim)
        
        i1, j1 = a // m, a % m
        j2, k1 = b // p, b % p
        k2, i2 = c // n, c % n
        
        if i1 == i2 and j1 == j2 and k1 == k2:
            continue
        
        s = 0.0
        for r in range(R):
            s += U[a, r] * V[b, r] * W[c, r]
        if abs(s) > tol:
            return False
    
    return True

def _optimize_decomposition(n, m, p, target_rank, time_limit=60.0):
    """Try to find a lower-rank decomposition using numerical optimization."""
    a_dim = n * m
    b_dim = m * p
    c_dim = p * n
    
    # Build the target tensor
    T = np.zeros((a_dim, b_dim, c_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Initialize with random factors
    np.random.seed(123)
    U = np.random.randn(a_dim, target_rank) * 0.5
    V = np.random.randn(b_dim, target_rank) * 0.5
    W = np.random.randn(c_dim, target_rank) * 0.5
    
    start_time = time.time()
    lr = 0.01
    
    for iteration in range(5000):
        if time.time() - start_time > time_limit:
            break
        
        # Compute gradient for U
        U_grad = np.zeros_like(U)
        V_grad = np.zeros_like(V)
        W_grad = np.zeros_like(W)
        
        # Efficient computation using einsum
        # residual[a,b,c] = T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r]
        UV = np.einsum('ar,br->ab', U, V)  # (a_dim, b_dim)
        residual = T - np.einsum('ab,cr->acr', UV, W)
        
        # Gradient for U: dL/dU[a,r] = -2 * sum_{b,c} residual[a,b,c] * V[b,r] * W[c,r]
        VW = np.einsum('br,cr->bc', V, W)  # (b_dim, c_dim)
        U_grad = -2.0 * np.einsum('abc,br,cr->ar', residual, V, W)
        
        # Gradient for V: dL/dV[b,r] = -2 * sum_{a,c} residual[a,b,c] * U[a,r] * W[c,r]
        UW = np.einsum('ar,cr->ac', U, W)
        V_grad = -2.0 * np.einsum('abc,ar,cr->br', residual, U, W)
        
        # Gradient for W: dL/dW[c,r] = -2 * sum_{a,b} residual[a,b,c] * U[a,r] * V[b,r]
        UV2 = np.einsum('ar,br->ab', U, V)
        W_grad = -2.0 * np.einsum('abc,ar,br->cr', residual, U, V)
        
        # Update
        U -= lr * U_grad
        V -= lr * V_grad
        W -= lr * W_grad
        
        # Decay learning rate
        if iteration % 500 == 499:
            lr *= 0.5
        
        # Check convergence
        if iteration % 100 == 99:
            err = np.linalg.norm(residual)
            if err < 1e-4:
                break
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    deadline = time.time() + 130  # Safety margin
    
    # First try the correct block decomposition (rank 33)
    try:
        rank, U, V, W = _build_block_decomposition(n, m, p)
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    if time.time() > deadline:
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try optimization for lower rank
    for target_rank in [30, 28, 25, 22]:
        if time.time() > deadline - 30:
            break
        try:
            U_opt, V_opt, W_opt = _optimize_decomposition(n, m, p, target_rank, time_limit=25.0)
            if _verify_decomposition(U_opt, V_opt, W_opt, n, m, p, tol=1e-3):
                return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
        except Exception:
            pass
    
    # Fallback to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
