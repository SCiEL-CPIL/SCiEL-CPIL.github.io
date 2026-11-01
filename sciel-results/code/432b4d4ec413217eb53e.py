import time
import numpy as np

_START_TIME = time.time()
_TIMEOUT = 120


def construct():
    """
    Return a rank-minimal exact tensor decomposition for 2x2x2 matrix multiplication.
    
    Uses Strassen's algorithm which achieves rank 7, the known minimal rank for
    2x2 matrix multiplication over fields of characteristic != 2.
    
    Output factor follows the official trace convention W[k*n+i, r].
    
    U: (n*m, rank) = (4, 7)
    V: (m*p, rank) = (4, 7)
    W: (p*n, rank) = (4, 7)
    
    Validity: for all (i,j,k) in {0,1}^3:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    if time.time() - _START_TIME > _TIMEOUT:
        raise TimeoutError("Time limit exceeded")

    n, m, p = 2, 2, 2
    rank = 7

    # Strassen's 7 products for A=[[a,b],[c,d]], B=[[e,f],[g,h]]:
    # M0 = (a+d)(e+h)
    # M1 = (c+d)e
    # M2 = a(f-h)
    # M3 = d(g-e)
    # M4 = (a+b)h
    # M5 = (c-a)(e+f)
    # M6 = (b-d)(g+h)
    #
    # C[0,0] = M0 + M3 - M4 + M6
    # C[0,1] = M2 + M4
    # C[1,0] = M1 + M3
    # C[1,1] = M0 - M1 + M2 + M5
    #
    # Index conventions:
    #   U row: i*m + j  -> a:0, b:1, c:2, d:3
    #   V row: j*p + k  -> e:0, f:1, g:2, h:3
    #   W row: k*n + i  -> C[0,0]:0, C[1,0]:1, C[0,1]:2, C[1,1]:3

    U = [
        [1, 0, 1, 0, 1, -1, 0],  # a
        [0, 0, 0, 0, 1, 0, 1],   # b
        [0, 1, 0, 0, 0, 1, 0],   # c
        [1, 1, 0, 1, 0, 0, -1]   # d
    ]

    V = [
        [1, 1, 0, -1, 0, 1, 0],  # e
        [0, 0, 1, 0, 0, 1, 0],   # f
        [0, 0, 0, 1, 0, 0, 1],   # g
        [1, 0, -1, 0, 1, 0, 1]   # h
    ]

    # W rows indexed by k*n + i:
    # Row 0 (k=0,i=0) -> C[0,0] = M0 + M3 - M4 + M6
    # Row 1 (k=0,i=1) -> C[1,0] = M1 + M3
    # Row 2 (k=1,i=0) -> C[0,1] = M2 + M4
    # Row 3 (k=1,i=1) -> C[1,1] = M0 - M1 + M2 + M5
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # C[0,0]
        [0, 1, 0, 1, 0, 0, 0],    # C[1,0]
        [0, 0, 1, 0, 1, 0, 0],    # C[0,1]
        [1, -1, 1, 0, 0, 1, 0]    # C[1,1]
    ]

    # Verification: check tensor decomposition identity
    # For all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i * m + j]
                v_row = V_arr[j * p + k]
                w_row = W_arr[k * n + i]
                val = np.dot(u_row, v_row * w_row)
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback: return schoolbook rank-8 if verification fails (shouldn't happen)
        # But since Strassen is correct, this is just a safety net
        raise ValueError("Strassen decomposition verification failed")

    # Ensure integer entries
    U = [[int(v) for v in row] for row in U]
    V = [[int(v) for v in row] for row in V]
    W = [[int(v) for v in row] for row in W]

    return {"U": U, "V": V, "W": W}
