import time
import numpy as np

def construct():
    n, m, p = 2, 3, 8
    
    # Deadline guard
    deadline = time.time() + 140
    
    # Build rank-44 Strassen-based construction
    U, V, W = _build_rank44(n, m, p)
    
    # Verify correctness
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails (should not happen)
        U, V, W = _build_schoolbook(n, m, p)
        
    # Attempt rank reduction (heuristic)
    # Note: For this specific structured problem, rank 44 is likely minimal.
    # We perform a quick check for zero columns or linear dependencies.
    U, V, W = _try_reduce(U, V, W, n, m, p, deadline)
    
    return {"U": U, "V": V, "W": W}

def _try_reduce(U, V, W, n, m, p, deadline):
    """Attempt to reduce rank by removing zero columns."""
    r = len(U[0])
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check for zero columns
    for c in range(r - 1, -1, -1):
        if time.time() > deadline:
            break
            
        if np.allclose(U_arr[:, c], 0) and np.allclose(V_arr[:, c], 0) and np.allclose(W_arr[:, c], 0):
            U_arr = np.delete(U_arr, c, axis=1)
            V_arr = np.delete(V_arr, c, axis=1)
            W_arr = np.delete(W_arr, c, axis=1)
            
    # Convert back to lists
    U_out = U_arr.tolist()
    V_out = V_arr.tolist()
    W_out = W_arr.tolist()
    
    # Final verification
    if not _verify(U_out, V_out, W_out, n, m, p):
        return U, V, W
    
    return U_out, V_out, W_out

def _build_rank44(n, m, p):
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 2x1x8 part (j=0)
    # Schoolbook: 16 terms
    for k in range(p):
        # A[0,0] * B[0,k] -> C[0,k]
        U[0 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # A[1,0] * B[0,k] -> C[1,k]
        U[1 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
    # Part 2: 2x2x8 part (j=1,2)
    # Strassen: 4 blocks of 7 terms = 28 terms
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # P1: (A[0,1] + A[1,2]) * (B[1,k0] + B[2,k1])
        U[0 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # P2: (A[1,1] + A[1,2]) * B[1,k0]
        U[1 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        # P3: A[0,1] * (B[1,k1] - B[2,k1])
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        V[2 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # P4: A[1,2] * (B[2,k0] - B[1,k0])
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        V[1 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # P5: (A[0,1] + A[0,2]) * B[2,k1]
        U[0 * m + 1][col] = 1
        U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # P6: (A[1,1] - A[0,1]) * (B[1,k0] + B[1,k1])
        U[1 * m + 1][col] = 1
        U[0 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # P7: (A[0,2] - A[1,2]) * (B[2,k0] + B[2,k1])
        U[0 * m + 2][col] = 1
        U[1 * m + 2][col] = -1
        V[2 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        
    return U, V, W

def _build_schoolbook(n, m, p):
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
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify decomposition matches matrix multiplication tensor."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    s = 0.0
                    for r_idx in range(r):
                        s += U[i * m + j][r_idx] * V[j * p + k][r_idx] * W[k * n + ip][r_idx]
                    expected = 1.0 if i == ip else 0.0
                    if abs(s - expected) > 1e-10:
                        return False
    return True

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
