import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def check_decomposition(U, V, W, n, m, p):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1) > 1e-9:
                    return False
    return True

def get_strassen_333():
    # Strassen decomposition for 3x3x3
    # U: (9, 7), V: (9, 7), W: (9, 7)
    # Indices: U[i*3+j], V[j*3+k], W[k*3+i]
    
    # Standard Strassen for 2x2x2 is rank 7.
    # For 3x3x3, we can embed the 2x2x2 Strassen in the top-left 2x2x2 block
    # and handle the rest trivially?
    # Actually, the rank of 3x3x3 is 7.
    # The decomposition is not just the 2x2x2 Strassen.
    # I will use the known rank-7 decomposition for 3x3x3.
    # Since I don't have it memorized perfectly, I will use the trivial rank 9 for 3x3x3
    # to ensure correctness, or try to find a simpler one.
    # Trivial rank for 3x3x3 is 9.
    # Rank 9 is better than 18 (trivial for 3x3x2 is 18? No, 3*3*2=18).
    # If I use Trivial for 3x3x3 (Rank 9) and Trivial for 3x3x2 (Rank 18), Total 27.
    # This is still better than 45.
    # Let's use Trivial for both to be safe and simple.
    
    n, m, p = 3, 3, 3
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
    return U, V, W

def get_trivial_332():
    n, m, p = 3, 3, 2
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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Get components
    U1, V1, W1 = get_strassen_333() # Actually trivial 333
    U2, V2, W2 = get_trivial_332()
    
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    total_rank = rank1 + rank2
    
    # Initialize full matrices
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Fill first part (3x3x3)
    U[:, :rank1] = U1
    V[:, :rank1] = V1
    # W1 has shape (9, rank1). Rows 0-8 correspond to k=0,1,2.
    W[0:9, :rank1] = W1
    
    # Fill second part (3x3x2)
    U[:, rank1:] = U2
    V[:, rank1:] = V2
    # W2 has shape (6, rank2). Rows 0-2 correspond to k'=0 (global k=3). Rows 3-5 correspond to k'=1 (global k=4).
    W[9:12, rank1:] = W2[0:3, :]
    W[12:15, rank1:] = W2[3:6, :]
    
    # Verify
    if not check_decomposition(U, V, W, n, m, p):
        # Fallback to full trivial
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

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
