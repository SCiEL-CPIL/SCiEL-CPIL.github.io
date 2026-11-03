import time
import numpy as np

def construct():
    n, m, p = 2, 3, 8
    dim_U = n * m  # 6
    dim_V = m * p  # 24
    dim_W = p * n  # 16
    
    t_start = time.time()
    deadline = t_start + 100.0  # Safety margin

    # Build the known rank-44 decomposition
    def build_rank44():
        rank = 44
        U = np.zeros((dim_U, rank), dtype=float)
        V = np.zeros((dim_V, rank), dtype=float)
        W = np.zeros((dim_W, rank), dtype=float)
        col = 0
        
        # Part 1: j=0, 16 terms (2 rows * 8 cols)
        # A[:, 0] * B[0, :]
        for k in range(p):
            # Term 1: A[0,0] * B[0,k] -> C[0,k]
            U[0 * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + 0, col] = 1
            col += 1
            
            # Term 2: A[1,0] * B[0,k] -> C[1,k]
            U[1 * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + 1, col] = 1
            col += 1
        
        # Part 2: j=1,2 in 4 blocks of 2x2x2, 7 terms each = 28
        # Using Strassen's algorithm for each 2x2x2 block
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            # P1: (A[0,1] + A[1,2]) * (B[1,k0] + B[2,k1])
            # U: A[0,1]=1, A[1,2]=1
            U[0 * m + 1, col] = 1
            U[1 * m + 2, col] = 1
            # V: B[1,k0]=1, B[2,k1]=1
            V[1 * p + k0, col] = 1
            V[2 * p + k1, col] = 1
            # W: C[0,k0]=1, C[1,k1]=1
            W[k0 * n + 0, col] = 1
            W[k1 * n + 1, col] = 1
            col += 1
            
            # P2: (A[1,1] + A[1,2]) * B[1,k0]
            U[1 * m + 1, col] = 1
            U[1 * m + 2, col] = 1
            V[1 * p + k0, col] = 1
            # W: C[1,k0]=1, C[1,k1]=-1
            W[k0 * n + 1, col] = 1
            W[k1 * n + 1, col] = -1
            col += 1
            
            # P3: A[0,1] * (B[1,k1] - B[2,k1])
            U[0 * m + 1, col] = 1
            V[1 * p + k1, col] = 1
            V[2 * p + k1, col] = -1
            # W: C[0,k1]=1, C[1,k1]=1
            W[k1 * n + 0, col] = 1
            W[k1 * n + 1, col] = 1
            col += 1
            
            # P4: A[1,2] * (B[2,k0] - B[1,k0])
            U[1 * m + 2, col] = 1
            V[2 * p + k0, col] = 1
            V[1 * p + k0, col] = -1
            # W: C[0,k0]=1, C[1,k0]=1
            W[k0 * n + 0, col] = 1
            W[k0 * n + 1, col] = 1
            col += 1
            
            # P5: (A[0,1] + A[0,2]) * B[2,k1]
            U[0 * m + 1, col] = 1
            U[0 * m + 2, col] = 1
            V[2 * p + k1, col] = 1
            # W: C[0,k0]=-1, C[0,k1]=1
            W[k0 * n + 0, col] = -1
            W[k1 * n + 0, col] = 1
            col += 1
            
            # P6: (A[1,1] - A[0,1]) * (B[1,k0] + B[1,k1])
            U[1 * m + 1, col] = 1
            U[0 * m + 1, col] = -1
            V[1 * p + k0, col] = 1
            V[1 * p + k1, col] = 1
            # W: C[1,k1]=1
            W[k1 * n + 1, col] = 1
            col += 1
            
            # P7: (A[0,2] - A[1,2]) * (B[2,k0] + B[2,k1])
            U[0 * m + 2, col] = 1
            U[1 * m + 2, col] = -1
            V[2 * p + k0, col] = 1
            V[2 * p + k1, col] = 1
            # W: C[0,k0]=1
            W[k0 * n + 0, col] = 1
            col += 1
        
        return U, V, W, rank
    
    U, V, W, rank = build_rank44()
    
    # Optional: Quick verification to ensure no bugs in construction
    # This is fast for rank 44
    def verify(U, V, W, tol=1e-9):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                    if abs(np.sum(prod) - 1.0) > tol:
                        return False
        # Check cross terms are zero
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        uv = U[i*m+j, :] * V[jp*p+k, :]
                        result = W.T @ uv  # shape (p*n,)
                        for ip in range(n):
                            for kp in range(p):
                                expected = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                                if abs(result[kp*n+ip] - expected) > tol:
                                    return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook if something went wrong (should not happen)
        rank = 48
        U = np.zeros((dim_U, rank), dtype=float)
        V = np.zeros((dim_V, rank), dtype=float)
        W = np.zeros((dim_W, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        # Return fallback
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Convert to list format as required by interface
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}

def _verify(U, V, W):
    n, m, p = 2, 3, 8
    R = len(U[0]) if U else 0
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                for i in range(n):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            if i == i0 and k == k0:
                                val += W_arr[k * n + i, r] * U_arr[i * m + j0, r] * V_arr[j0 * p + k, r]
                        expected_val = 1 if (i == i0 and k == k0) else 0
                        if abs(val - expected_val) > 1e-9:
                            return False
    return True

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
    if _verify(res["U"], res["V"], res["W"]):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
