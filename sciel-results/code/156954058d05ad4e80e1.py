import time
import numpy as np

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 125  # Safety margin before 150s kill

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def verify_decomposition(U, V, W, n=2, m=4, p=4):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
    
    T_target = np.ones((n, m, p), dtype=float)
    max_err = np.max(np.abs(T_recon - T_target))
    return max_err < 1e-9, max_err

def strassen_decomposition():
    """Return the Strassen-based decomposition (rank 28) - clean version."""
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_strassen(A_r0, A_c0, B_r0, B_c0, C_r0, C_c0, start_r):
        a11 = A_r0 * m + A_c0
        a12 = A_r0 * m + A_c0 + 1
        a21 = (A_r0 + 1) * m + A_c0
        a22 = (A_r0 + 1) * m + A_c0 + 1
        
        b11 = B_r0 * p + B_c0
        b12 = B_r0 * p + B_c0 + 1
        b21 = (B_r0 + 1) * p + B_c0
        b22 = (B_r0 + 1) * p + B_c0 + 1
        
        c11_idx = C_c0 * n + C_r0
        c12_idx = (C_c0 + 1) * n + C_r0
        c21_idx = C_c0 * n + (C_r0 + 1)
        c22_idx = (C_c0 + 1) * n + (C_r0 + 1)
        
        r = start_r
        
        # M1 = (A11 + A22)(B11 + B22)
        U[a11][r] += 1
        U[a22][r] += 1
        V[b11][r] += 1
        V[b22][r] += 1
        W[c11_idx][r] += 1
        W[c22_idx][r] += 1
        r += 1
        
        # M2 = (A21 + A22)B11
        U[a21][r] += 1
        U[a22][r] += 1
        V[b11][r] += 1
        W[c21_idx][r] += 1
        W[c22_idx][r] += -1
        r += 1
        
        # M3 = A11(B12 - B22)
        U[a11][r] += 1
        V[b12][r] += 1
        V[b22][r] += -1
        W[c12_idx][r] += 1
        W[c22_idx][r] += 1
        r += 1
        
        # M4 = A22(B21 - B11)
        U[a22][r] += 1
        V[b21][r] += 1
        V[b11][r] += -1
        W[c11_idx][r] += 1
        W[c21_idx][r] += 1
        r += 1
        
        # M5 = (A11 + A12)B22
        U[a11][r] += 1
        U[a12][r] += 1
        V[b22][r] += 1
        W[c11_idx][r] += -1
        W[c12_idx][r] += 1
        r += 1
        
        # M6 = (A21 - A11)(B11 + B12)
        U[a21][r] += 1
        U[a11][r] += -1
        V[b11][r] += 1
        V[b12][r] += 1
        W[c22_idx][r] += 1
        r += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        U[a12][r] += 1
        U[a22][r] += -1
        V[b21][r] += 1
        V[b22][r] += 1
        W[c11_idx][r] += 1
        r += 1
        
        return r
    
    r = add_strassen(0, 0, 0, 0, 0, 0, 0)
    r = add_strassen(0, 2, 2, 0, 0, 0, r)
    r = add_strassen(0, 0, 0, 2, 0, 2, r)
    r = add_strassen(0, 2, 2, 2, 0, 2, r)
    
    assert r == 28, f"Expected rank 28, got {r}"
    return U, V, W

def construct():
    n, m, p = 2, 4, 4
    
    # Start with Strassen
    U, V, W = strassen_decomposition()
    
    # Verify to be safe
    valid, err = verify_decomposition(U, V, W, n, m, p)
    if not valid:
        # Fallback: naive decomposition
        rank = 32
        U = [[0.0] * rank for _ in range(n*m)]
        V = [[0.0] * rank for _ in range(m*p)]
        W = [[0.0] * rank for _ in range(p*n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1.0
                    V[j*p+k][idx] = 1.0
                    W[k*n+i][idx] = 1.0
                    idx += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print(f"Best rank: {len(result['U'][0])}")
    valid, err = verify_decomposition(result['U'], result['V'], result['W'])
    print(f"Valid: {valid}, Max error: {err}")
