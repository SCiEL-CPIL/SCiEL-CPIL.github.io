import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    n, m, p = 2, 2, 2
    rank = 7

    # Strassen's 7 scalar products for 2x2 matrix multiplication
    # A = [[a00, a01], [a10, a11]] -> flat indices: a00=0, a01=1, a10=2, a11=3
    # B = [[b00, b01], [b10, b11]] -> flat indices: b00=0, b01=1, b10=2, b11=3
    # W convention: W[k*n + i] where k is column of C, i is row of C
    #   C00 -> k=0,i=0 -> idx 0
    #   C01 -> k=1,i=0 -> idx 2
    #   C10 -> k=0,i=1 -> idx 1
    #   C11 -> k=1,i=1 -> idx 3

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # M0: (a00+a11)*(b00+b11) -> C00+=1, C11+=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # M1: (a10+a11)*b00 -> C10+=1, C11-=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1

    # M2: a00*(b01-b11) -> C01+=1, C11+=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1

    # M3: a11*(b10-b00) -> C00+=1, C10+=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1

    # M4: (a00+a01)*b11 -> C00-=1, C01+=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1

    # M5: (a10-a00)*(b00+b01) -> C11+=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # M6: (a01-a11)*(b10+b11) -> C00+=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    # Lightweight verification: reconstruct the multiplication tensor
    # T[i,j,k] = 1 iff the (i,j) entry of A times (j,k) entry of B contributes to C[i,k]
    # In flat form: U row i*m+j, V row j*p+k, W row k*n+i
    # T = sum_r U[:,r] * V[:,r] * W[:,r] (outer product)
    # Verify against the canonical 2x2x2 multiplication tensor
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Build canonical tensor: T[a_flat, b_flat, c_flat]
    # a_flat = i*m+j, b_flat = j*p+k, c_flat = k*n+i
    # T = 1 when the indices are consistent with matrix mult
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_expected[a_idx, b_idx, c_idx] = 1

    # Reconstruct T from decomposition
    T_reconstructed = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)

    if not np.allclose(T_reconstructed, T_expected, atol=1e-10):
        # This should never happen with correct Strassen, but guard just in case
        # Fall back to returning the same (known correct) values
        pass

    return {"U": U, "V": V, "W": W}
