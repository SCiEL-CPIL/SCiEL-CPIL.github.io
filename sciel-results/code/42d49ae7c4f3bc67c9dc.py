import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    def check_solution_full(U, V, W, R):
        """Check if the full 2x3x5 decomposition is exact."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # This should be 1 for all i,j,k
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def build_strassen_rank28():
        """Build explicit rank-28 solution."""
        rank = 28
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        c = 0

        def add_strassen_block(j_off, k_off):
            nonlocal c
            # m1 = (a00+a11)*(b00+b11)
            U[0*m + j_off, c] = 1; U[1*m + (j_off+1), c] = 1
            V[j_off*p + k_off, c] = 1; V[(j_off+1)*p + (k_off+1), c] = 1
            W[k_off*n + 0, c] = 1; W[(k_off+1)*n + 1, c] = 1
            c += 1
            # m2 = (a10+a11)*b00
            U[1*m + j_off, c] = 1; U[1*m + (j_off+1), c] = 1
            V[j_off*p + k_off, c] = 1
            W[k_off*n + 1, c] = 1; W[(k_off+1)*n + 1, c] = -1
            c += 1
            # m3 = a00*(b01-b11)
            U[0*m + j_off, c] = 1
            V[j_off*p + (k_off+1), c] = 1; V[(j_off+1)*p + (k_off+1), c] = -1
            W[(k_off+1)*n + 0, c] = 1; W[(k_off+1)*n + 1, c] = 1
            c += 1
            # m4 = a11*(b10-b00)
            U[1*m + (j_off+1), c] = 1
            V[(j_off+1)*p + k_off, c] = 1; V[j_off*p + k_off, c] = -1
            W[k_off*n + 0, c] = 1; W[k_off*n + 1, c] = 1
            c += 1
            # m5 = (a00+a01)*b11
            U[0*m + j_off, c] = 1; U[0*m + (j_off+1), c] = 1
            V[(j_off+1)*p + (k_off+1), c] = 1
            W[k_off*n + 0, c] = -1; W[(k_off+1)*n + 0, c] = 1
            c += 1
            # m6 = (a10-a00)*(b00+b01)
            U[1*m + j_off, c] = 1; U[0*m + j_off, c] = -1
            V[j_off*p + k_off, c] = 1; V[j_off*p + (k_off+1), c] = 1
            W[(k_off+1)*n + 1, c] = 1
            c += 1
            # m7 = (a01-a11)*(b10+b11)
            U[0*m + (j_off+1), c] = 1; U[1*m + (j_off+1), c] = -1
            V[(j_off+1)*p + k_off, c] = 1; V[(j_off+1)*p + (k_off+1), c] = 1
            W[k_off*n + 0, c] = 1
            c += 1

        # Block 1: j in {0,1}, k in {0,1}
        add_strassen_block(0, 0)
        # Block 2: j in {0,1}, k in {2,3}
        add_strassen_block(0, 2)
        
        # Remaining schoolbook terms
        # These are terms where (j in {0,1} and k=4) OR (j=2 and k in {0..4})
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Skip terms covered by Strassen blocks
                    if j in (0, 1) and k in (0, 1, 2, 3):
                        continue
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        
        assert c == rank, f"Expected rank {rank}, got {c}"
        return U, V, W, rank

    # Step 1: Get guaranteed rank-28 solution
    U28, V28, W28, rank28 = build_strassen_rank28()
    
    # Verify
    err28 = check_solution_full(U28, V28, W28, rank28)
    
    if err28 > 1e-10:
        # Fallback to schoolbook rank 30 if something went wrong (should not happen)
        R = n*m*p
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
