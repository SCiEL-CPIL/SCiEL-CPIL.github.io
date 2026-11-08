import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 10s safety margin
    
    # Rank 71 decomposition: 4 Strassen blocks (28 terms) + 43 schoolbook terms.
    # Strassen blocks cover i∈{0,1}, j∈{0..3}, k∈{0..3} (32 entries).
    # Remaining 43 entries use schoolbook.
    
    total_rank = 71
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    col = 0
    
    for (i0, j0, k0) in blocks:
        # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        U[i0*m + j0, col] += 1.0
        U[(i0+1)*m + (j0+1), col] += 1.0
        V[j0*p + k0, col] += 1.0
        V[(j0+1)*p + (k0+1), col] += 1.0
        W[k0*n + i0, col] += 1.0
        W[(k0+1)*n + (i0+1), col] += 1.0
        col += 1
        
        # M2 = (A[1,0] + A[1,1]) * B[0,0]
        U[(i0+1)*m + j0, col] += 1.0
        U[(i0+1)*m + (j0+1), col] += 1.0
        V[j0*p + k0, col] += 1.0
        W[k0*n + (i0+1), col] += 1.0
        W[(k0+1)*n + (i0+1), col] -= 1.0
        col += 1
        
        # M3 = A[0,0] * (B[0,1] - B[1,1])
        U[i0*m + j0, col] += 1.0
        V[j0*p + (k0+1), col] += 1.0
        V[(j0+1)*p + (k0+1), col] -= 1.0
        W[(k0+1)*n + i0, col] += 1.0
        W[(k0+1)*n + (i0+1), col] += 1.0
        col += 1
        
        # M4 = A[1,1] * (B[1,0] - B[0,0])
        U[(i0+1)*m + (j0+1), col] += 1.0
        V[(j0+1)*p + k0, col] += 1.0
        V[j0*p + k0, col] -= 1.0
        W[k0*n + i0, col] += 1.0
        W[k0*n + (i0+1), col] += 1.0
        col += 1
        
        # M5 = (A[0,0] + A[0,1]) * B[1,1]
        U[i0*m + j0, col] += 1.0
        U[i0*m + (j0+1), col] += 1.0
        V[(j0+1)*p + (k0+1), col] += 1.0
        W[k0*n + i0, col] -= 1.0
        W[(k0+1)*n + i0, col] += 1.0
        col += 1
        
        # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        U[(i0+1)*m + j0, col] += 1.0
        U[i0*m + j0, col] -= 1.0
        V[j0*p + k0, col] += 1.0
        V[j0*p + (k0+1), col] += 1.0
        W[(k0+1)*n + (i0+1), col] += 1.0
        col += 1
        
        # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
        U[i0*m + (j0+1), col] += 1.0
        U[(i0+1)*m + (j0+1), col] -= 1.0
        V[(j0+1)*p + k0, col] += 1.0
        V[(j0+1)*p + (k0+1), col] += 1.0
        W[k0*n + i0, col] += 1.0
        col += 1

    covered = np.zeros((n, m, p), dtype=bool)
    for (i0, j0, k0) in blocks:
        covered[i0:i0+2, j0:j0+2, k0:k0+2] = True
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i, j, k]:
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1

    if col != total_rank:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    
    # Full verification of all 75 entries
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(U.shape[1]))
                if abs(val - 1.0) > 1e-9:
                    errors += 1
                    print(f"Error at ({i},{j},{k}): {val}")
    
    print(f"Verification Errors: {errors}")
    print(f"Rank: {U.shape[1]}")
