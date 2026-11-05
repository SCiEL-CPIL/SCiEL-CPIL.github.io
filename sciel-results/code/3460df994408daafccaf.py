import numpy as np
import time

DIMENSIONS = [5, 5, 5]
DEADLINE_SECONDS = 140.0

def _schoolbook(n, m, p):
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

def _verify(U, V, W, n, m, p, tol=1e-10):
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_expected = A @ B
    
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    L = U_arr @ A_flat # Shape (rank,)
    R = V_arr @ B_flat # Shape (rank,)
    
    C_computed = np.zeros((n, p))
    # W[k*n+i, r] convention from task description
    # C[i,k] = sum_r W[k*n+i, r] L_r R_r
    for i in range(n):
        for k in range(p):
            idx = k * n + i
            C_computed[i, k] = np.sum(W_arr[idx, :] * L * R)
            
    if np.max(np.abs(C_computed - C_expected)) > tol:
        # Try the other convention: W[i*n+k, r]
        C_computed2 = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                idx = i * n + k
                C_computed2[i, k] = np.sum(W_arr[idx, :] * L * R)
        if np.max(np.abs(C_computed2 - C_expected)) > tol:
            return False
        return True
    return True

def _construct_block_decomposition(n, m, p):
    # Partition 5 into 2, 2, 1
    sizes = [2, 2, 1]
    
    # Pre-compute start indices
    starts = [0]
    for s in sizes:
        starts.append(starts[-1] + s)
        
    U = [[0] * (n * m * p) for _ in range(n * m)]
    V = [[0] * (n * m * p) for _ in range(m * p)]
    W = [[0] * (n * m * p) for _ in range(p * n)]
    
    rank = 0
    
    def add_strassen_block(a_i, a_j, b_k, b_l, c_i, c_k, r_start):
        """
        Adds Strassen decomposition for A[a_i:a_i+2, a_j:a_j+2] * B[b_k:b_k+2, b_l:b_l+2]
        contributing to C[c_i:c_i+2, c_k:c_k+2]
        """
        # M1 = (A00 + A11) * (B00 + B11)
        U[a_i * m + a_j][r_start] += 1
        U[(a_i+1) * m + (a_j+1)][r_start] += 1
        V[b_k * p + b_l][r_start] += 1
        V[(b_k+1) * p + (b_l+1)][r_start] += 1
        W[c_k * n + c_i][r_start] += 1
        W[(c_k+1) * n + (c_i+1)][r_start] += 1
        r_start += 1
        
        # M2 = (A10 + A11) * B00
        U[(a_i+1) * m + a_j][r_start] += 1
        U[(a_i+1) * m + (a_j+1)][r_start] += 1
        V[b_k * p + b_l][r_start] += 1
        W[c_k * n + (c_i+1)][r_start] += 1
        r_start += 1
        
        # M3 = A00 * (B01 - B11)
        U[a_i * m + a_j][r_start] += 1
        V[b_k * p + (b_l+1)][r_start] += 1
        V[(b_k+1) * p + (b_l+1)][r_start] -= 1
        W[c_k * n + (c_i+1)][r_start] += 1
        W[(c_k+1) * n + (c_i+1)][r_start] += 1
        r_start += 1
        
        # M4 = A11 * (B10 - B00)
        U[(a_i+1) * m + (a_j+1)][r_start] += 1
        V[(b_k+1) * p + b_l][r_start] += 1
        V[b_k * p + b_l][r_start] -= 1
        W[c_k * n + c_i][r_start] += 1
        W[c_k * n + (c_i+1)][r_start] += 1
        r_start += 1
        
        # M5 = (A00 + A01) * B11
        U[a_i * m + a_j][r_start] += 1
        U[a_i * m + (a_j+1)][r_start] += 1
        V[(b_k+1) * p + (b_l+1)][r_start] += 1
        W[c_k * n + c_i][r_start] -= 1
        W[c_k * n + (c_i+1)][r_start] += 1
        r_start += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[(a_i+1) * m + a_j][r_start] += 1
        U[a_i * m + a_j][r_start] -= 1
        V[b_k * p + b_l][r_start] += 1
        V[b_k * p + (b_l+1)][r_start] += 1
        W[(c_k+1) * n + (c_i+1)][r_start] += 1
        r_start += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[a_i * m + (a_j+1)][r_start] += 1
        U[(a_i+1) * m + (a_j+1)][r_start] -= 1
        V[(b_k+1) * p + b_l][r_start] += 1
        V[(b_k+1) * p + (b_l+1)][r_start] += 1
        W[c_k * n + c_i][r_start] += 1
        r_start += 1
        
        return r_start

    def add_schoolbook_block(a_i, a_j, b_k, b_l, c_i, c_k, r_start, size_a, size_b, size_c):
        """
        Adds schoolbook decomposition for A[a_i:a_i+size_a, a_j:a_j+size_b] * B[b_k:b_k+size_b, b_l:b_l+size_c]
        contributing to C[c_i:c_i+size_a, c_k:c_k+size_c]
        """
        for di in range(size_a):
            for dj in range(size_b):
                for dk in range(size_c):
                    U[(a_i+di) * m + (a_j+dj)][r_start] = 1
                    V[(b_k+dj) * p + (b_l+dk)][r_start] = 1
                    W[(c_k+dk) * n + (c_i+di)][r_start] = 1
                    r_start += 1
        return r_start

    for i in range(3):
        for j in range(3):
            for k in range(3):
                # A block i,k
                # B block k,j
                # C block i,j
                
                a_i = starts[i]
                a_j = starts[k]
                b_k = starts[k]
                b_l = starts[j]
                c_i = starts[i]
                c_k = starts[j]
                
                size_a = sizes[i]
                size_b = sizes[k]
                size_c = sizes[j]
                
                if size_a == 2 and size_b == 2 and size_c == 2:
                    rank = add_strassen_block(a_i, a_j, b_k, b_l, c_i, c_k, rank)
                else:
                    rank = add_schoolbook_block(a_i, a_j, b_k, b_l, c_i, c_k, rank, size_a, size_b, size_c)
                    
    # Trim to actual rank
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Try the block decomposition
    try:
        result = _construct_block_decomposition(n, m, p)
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
        
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
