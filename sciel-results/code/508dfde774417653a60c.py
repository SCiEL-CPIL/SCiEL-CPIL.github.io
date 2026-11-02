import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    """
    Constructs a rank-minimal (trivial) exact tensor decomposition for 
    multiplying a 3x5 matrix by a 5x7 matrix.
    
    The tensor T has indices (a, b, c) where:
    a indexes A (3x5, row-major: i*m + j)
    b indexes B (5x7, row-major: j*p + k)
    c indexes C (3x7, column-major per task convention: k*n + i)
    
    T[a, b, c] = 1 if and only if:
    a = i*m + j
    b = j*p + k
    c = k*n + i
    
    This yields a decomposition of rank n*m*p = 105.
    """
    n, m, p = DIMENSIONS
    
    # Deadline guard: 120s limit, keep 5s margin
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Trivial rank
    rank = n * m * p
    
    # Initialize U, V, W as zero matrices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in A (row-major)
                a = i * m + j
                # Index in B (row-major)
                b = j * p + k
                # Index in C (column-major: k*n + i)
                c = k * n + i
                
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
                
                # Check deadline occasionally (not every iteration for speed, but this loop is small)
                if time.time() > deadline:
                    # This shouldn't happen for rank 105, but safety first
                    break
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break

    # Return the factor matrices
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Simple self-test
    res = construct()
    U, V, W = res["U"], res["V"], res["W"]
    
    n, m, p = DIMENSIONS
    
    # Verify decomposition
    # T_approx = U @ V.T @ W.T ? No.
    # Tensor T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Reconstruct T from factors
    # T is a 3D tensor of shape (n*m, m*p, p*n)
    # We can check a few specific entries or the full tensor if small.
    
    # Full tensor verification
    T_reconstructed = np.zeros((n*m, m*p, p*n), dtype=np.int8)
    for r in range(U.shape[1]):
        T_reconstructed += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
        
    # Construct expected T
    T_expected = np.zeros((n*m, m*p, p*n), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_expected[a, b, c] = 1
                
    if np.array_equal(T_reconstructed, T_expected):
        print("Verification successful. Rank:", U.shape[1])
    else:
        print("Verification FAILED.")
