import numpy as np
import time

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    # Verified Strassen terms for 2x2x2.
    # Each term is (a_vec, b_vec, c_vec)
    # a_vec: coefficients for A[0,0], A[0,1], A[1,0], A[1,1]
    # b_vec: coefficients for B[0,0], B[0,1], B[1,0], B[1,1] (local 2x2 block)
    # c_vec: coefficients for C[0,0], C[1,0], C[0,1], C[1,1] (local 2x2 block)
    # Note: C index in W is k*n + i.
    # Local C indices:
    # C[0,0] -> i=0, k_local=0
    # C[1,0] -> i=1, k_local=0
    # C[0,1] -> i=0, k_local=1
    # C[1,1] -> i=1, k_local=1
    # So c_vec order is [C00, C10, C01, C11] which maps to indices:
    # (k_local=0, i=0), (k_local=0, i=1), (k_local=1, i=0), (k_local=1, i=1)
    
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]

    num_blocks = p // 2  # 5 blocks
    R = 7 * num_blocks  # rank 35

    U = np.zeros((n*m, R), dtype=float)
    V = np.zeros((m*p, R), dtype=float)
    W = np.zeros((p*n, R), dtype=float)

    col = 0
    for q in range(num_blocks):
        for a_vec, b_vec, c_vec in terms:
            # U indices: i*m + j
            # a_vec is [A00, A01, A10, A11]
            # idx 0: A[0,0] -> i=0, j=0 -> 0
            # idx 1: A[0,1] -> i=0, j=1 -> 1
            # idx 2: A[1,0] -> i=1, j=0 -> 2
            # idx 3: A[1,1] -> i=1, j=1 -> 3
            U[0, col] = a_vec[0]
            U[1, col] = a_vec[1]
            U[2, col] = a_vec[2]
            U[3, col] = a_vec[3]

            # V indices: j*p + k
            # b_vec is [B00, B01, B10, B11] for the local block q
            # Local B:
            # B[0, 2q] -> j=0, k=2q
            # B[0, 2q+1] -> j=0, k=2q+1
            # B[1, 2q] -> j=1, k=2q
            # B[1, 2q+1] -> j=1, k=2q+1
            V[0*p + 2*q, col] = b_vec[0]
            V[0*p + 2*q+1, col] = b_vec[1]
            V[1*p + 2*q, col] = b_vec[2]
            V[1*p + 2*q+1, col] = b_vec[3]

            # W indices: k*n + i
            # c_vec is [C00, C10, C01, C11]
            # C[0, 2q] -> i=0, k=2q -> idx 2q*2 + 0 = 4q
            # C[1, 2q] -> i=1, k=2q -> idx 2q*2 + 1 = 4q+1
            # C[0, 2q+1] -> i=0, k=2q+1 -> idx (2q+1)*2 + 0 = 4q+2
            # C[1, 2q+1] -> i=1, k=2q+1 -> idx (2q+1)*2 + 1 = 4q+3
            W[4*q + 0, col] = c_vec[0]
            W[4*q + 1, col] = c_vec[1]
            W[4*q + 2, col] = c_vec[2]
            W[4*q + 3, col] = c_vec[3]

            col += 1

    # Quick Verification
    if time.time() < deadline - 5:
        np.random.seed(12345)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_true = A @ B

        C_decomp = np.zeros((n, p))
        # Efficient check:
        # C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) (sum_j V[j*p+k, r] B[j,k]) W[k*n+i, r]
        # We can compute the intermediate sums.
        # U_r = [sum_j U[i*m+j, r] A[i,j] for i in range(n)] -> shape (n, R)
        # V_r = [sum_j V[j*p+k, r] B[j,k] for k in range(p)] -> shape (p, R)
        
        # U_contrib[i, r] = sum_j U[i*m+j, r] A[i,j]
        U_contrib = np.zeros((n, R))
        for i in range(n):
            for r in range(R):
                U_contrib[i, r] = sum(U[i*m+j, r] * A[i, j] for j in range(m))
        
        # V_contrib[k, r] = sum_j V[j*p+k, r] B[j,k]
        V_contrib = np.zeros((p, R))
        for k in range(p):
            for r in range(R):
                V_contrib[k, r] = sum(V[j*p+k, r] * B[j, k] for j in range(m))
        
        for i in range(n):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U_contrib[i, r] * V_contrib[k, r] * W[k*n + i, r]
                C_decomp[i, k] = s

        if not np.allclose(C_true, C_decomp, atol=1e-10):
            # Fallback to schoolbook if Strassen fails (should not happen)
            R_school = n * m * p
            U = np.zeros((n*m, R_school), dtype=float)
            V = np.zeros((m*p, R_school), dtype=float)
            W = np.zeros((p*n, R_school), dtype=float)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m + j, col] = 1.0
                        V[j*p + k, col] = 1.0
                        W[k*n + i, col] = 1.0
                        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
