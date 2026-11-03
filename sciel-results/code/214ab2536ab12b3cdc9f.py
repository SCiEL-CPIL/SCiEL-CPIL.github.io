import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 6]

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2 * 2x2 = 2x2 using Strassen (rank 7).
    """
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 1, 0, 0, 0]
    ], dtype=float)
    V = np.array([
        [1, 1, -1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, -1, 1, 0, 1]
    ], dtype=float)
    W = np.array([
        [1, 0, 1, 0, -1, -1, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [1, -1, 0, 1, 0, 0, 1]
    ], dtype=float)
    return U, V, W

def _construct_2x2x2x3_rank11():
    """
    Constructs a rank-11 decomposition for 2x2 * 2x3 = 2x3.
    """
    U22, V22, W22 = _strassen_2x2x2()
    
    U = np.zeros((4, 11), dtype=float)
    V = np.zeros((6, 11), dtype=float)
    W = np.zeros((6, 11), dtype=float)
    
    # Part 1: Strassen for Y[:, 0:2]
    U[:, 0:7] = U22
    
    for r in range(7):
        for j in range(2):
            for k_sub in range(2):
                V[j*3 + k_sub, r] = V22[j*2 + k_sub, r]
    
    for r in range(7):
        for k_sub in range(2):
            for i in range(2):
                W[k_sub*2 + i, r] = W22[k_sub*2 + i, r]
    
    # Part 2: Schoolbook for Y[:, 2]
    U[0, 7] = 1
    V[2, 7] = 1
    W[4, 7] = 1
    
    U[1, 8] = 1
    V[5, 8] = 1
    W[4, 8] = 1
    
    U[2, 9] = 1
    V[2, 9] = 1
    W[5, 9] = 1
    
    U[3, 10] = 1
    V[5, 10] = 1
    W[5, 10] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, num_trials=5):
    """
    Verify if U, V, W correctly decompose the matrix multiplication tensor.
    """
    R = U.shape[1]
    for _ in range(num_trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p), dtype=float)
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        for r in range(R):
            u_val = np.dot(U[:, r], A_flat)
            v_val = np.dot(V[:, r], B_flat)
            if u_val == 0 or v_val == 0:
                continue
            w_mat = W[:, r].reshape(p, n)
            C_computed += u_val * v_val * w_mat.T
        
        if not np.allclose(C_computed, C_expected, atol=1e-8):
            return False
    return True

def _search_rank10_2x2x2x3(deadline):
    """
    Attempts to find a rank-10 decomposition for 2x2 * 2x3 = 2x3.
    Returns (U, V, W) if found, else None.
    """
    # This is a placeholder for a potential numerical search.
    # Given the complexity, we skip this for now and rely on Rank 11.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Use Rank 11 blocks (Total 44)
    U11, V11, W11 = _construct_2x2x2x3_rank11()
    
    rank = 44
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:3] -> C[0:2, 0:3]
    # U indices: i*4+j (i in 0,1; j in 0,1)
    # V indices: j*6+k (j in 0,1; k in 0,1,2)
    # W indices: k*2+i (k in 0,1,2; i in 0,1)
    for r in range(11):
        col = r
        for i in range(2):
            for jj in range(2):
                U[i * 4 + jj, col] = U11[i*2+jj, r]
        for jj in range(2):
            for kk in range(3):
                V[jj * 6 + kk, col] = V11[jj*3+kk, r]
        for kk in range(3):
            for i in range(2):
                W[kk * 2 + i, col] = W11[kk*2+i, r]
                
    # Block 2: A[0:2, 2:4] * B[2:4, 0:3] -> C[0:2, 0:3]
    # U indices: i*4+j (i in 0,1; j in 2,3)
    # V indices: j*6+k (j in 2,3; k in 0,1,2)
    # W indices: k*2+i (k in 0,1,2; i in 0,1)
    for r in range(11):
        col = 11 + r
        for i in range(2):
            for jj in range(2):
                U[i * 4 + (2 + jj), col] = U11[i*2+jj, r]
        for jj in range(2):
            for kk in range(3):
                V[(2 + jj) * 6 + kk, col] = V11[jj*3+kk, r]
        for kk in range(3):
            for i in range(2):
                W[kk * 2 + i, col] = W11[kk*2+i, r]

    # Block 3: A[0:2, 0:2] * B[0:2, 3:6] -> C[0:2, 3:6]
    # U indices: i*4+j (i in 0,1; j in 0,1)
    # V indices: j*6+k (j in 0,1; k in 3,4,5)
    # W indices: k*2+i (k in 0,1,2; i in 0,1) -> Note: Output col 3,4,5 corresponds to k=0,1,2 in local block
    for r in range(11):
        col = 22 + r
        for i in range(2):
            for jj in range(2):
                U[i * 4 + jj, col] = U11[i*2+jj, r]
        for jj in range(2):
            for kk in range(3):
                V[jj * 6 + (3 + kk), col] = V11[jj*3+kk, r]
        for kk in range(3):
            for i in range(2):
                W[kk * 2 + i, col] = W11[kk*2+i, r]

    # Block 4: A[0:2, 2:4] * B[2:4, 3:6] -> C[0:2, 3:6]
    # U indices: i*4+j (i in 0,1; j in 2,3)
    # V indices: j*6+k (j in 2,3; k in 3,4,5)
    # W indices: k*2+i (k in 0,1,2; i in 0,1)
    for r in range(11):
        col = 33 + r
        for i in range(2):
            for jj in range(2):
                U[i * 4 + (2 + jj), col] = U11[i*2+jj, r]
        for jj in range(2):
            for kk in range(3):
                V[(2 + jj) * 6 + (3 + kk), col] = V11[jj*3+kk, r]
        for kk in range(3):
            for i in range(2):
                W[kk * 2 + i, col] = W11[kk*2+i, r]

    if _verify_decomposition(U, V, W, n, m, p, num_trials=3):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Schoolbook Rank 48
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

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
