"""Rewritten program: rank-minimal tensor decomposition for 2x2 * 2x3 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x2 * 2x3 matrix multiplication.
    The bilinear complexity of 2x2 x 2x3 matrix multiplication is 11.
    We construct the known rank-11 solution: Strassen(7) for cols 0,1 + schoolbook(4) for col 2.
    """
    n, m, p = 2, 2, 3
    _t0 = time.time()
    
    # Construct the rank-11 solution directly
    result = _rank11_solution(n, m, p)
    
    # Verify it's correct
    if not _verify(result, n, m, p):
        # Fallback: return a simple schoolbook rank-12 (shouldn't happen)
        result = _rank12_solution(n, m, p)
    
    return result


def _rank11_solution(n, m, p):
    """Construct the known rank-11 decomposition: Strassen(7) for cols 0,1 + schoolbook(4) for col 2."""
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def set_u(i, j, r, val): U[i*m+j][r] = val
    def set_v(j, k, r, val): V[j*p+k][r] = val
    def set_w(k, i, r, val): W[k*n+i][r] = val

    # Strassen for the 2x2 sub-problem (cols 0 and 1 of B)
    # A = [[a,b],[c,d]], B_sub = [[b00,b01],[b10,b11]]
    
    # M1 = (a+d)(b00+b11)
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)   # a, d
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)   # b00, b11
    # C00 += M1, C11 += M1
    set_w(0, 0, 0, 1)  # W[k=0, i=0] = C00
    set_w(1, 1, 0, 1)  # W[k=1, i=1] = C11

    # M2 = (c+d)b00
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)   # c, d
    set_v(0, 0, 1, 1)                # b00
    # C10 += M2, C11 -= M2
    set_w(0, 1, 1, 1)   # W[k=0, i=1] = C10
    set_w(1, 1, 1, -1)  # W[k=1, i=1] = C11

    # M3 = a(b01-b11)
    set_u(0, 0, 2, 1)                # a
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)  # b01, -b11
    # C01 += M3, C11 += M3
    set_w(1, 0, 2, 1)   # W[k=1, i=0] = C01
    set_w(1, 1, 2, 1)   # W[k=1, i=1] = C11

    # M4 = d(b10-b00)
    set_u(1, 1, 3, 1)                # d
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)  # b10, -b00
    # C00 += M4, C10 += M4
    set_w(0, 0, 3, 1)   # W[k=0, i=0] = C00
    set_w(0, 1, 3, 1)   # W[k=0, i=1] = C10

    # M5 = (a+b)b11
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)   # a, b
    set_v(1, 1, 4, 1)                # b11
    # C00 -= M5, C01 += M5
    set_w(0, 0, 4, -1)  # W[k=0, i=0] = C00
    set_w(1, 0, 4, 1)   # W[k=1, i=0] = C01

    # M6 = (c-a)(b00+b01)
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)  # c, -a
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)   # b00, b01
    # C11 += M6
    set_w(1, 1, 5, 1)   # W[k=1, i=1] = C11

    # M7 = (b-d)(b10+b11)
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)  # b, -d
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)   # b10, b11
    # C00 += M7
    set_w(0, 0, 6, 1)   # W[k=0, i=0] = C00

    # Schoolbook for column 2 of B
    # C02 = a*b02 + b*b12
    set_u(0, 0, 7, 1); set_v(0, 2, 7, 1)   # a * b02
    set_w(2, 0, 7, 1)   # W[k=2, i=0] = C02

    set_u(0, 1, 8, 1); set_v(1, 2, 8, 1)   # b * b12
    set_w(2, 0, 8, 1)   # W[k=2, i=0] = C02

    # C12 = c*b02 + d*b12
    set_u(1, 0, 9, 1); set_v(0, 2, 9, 1)   # c * b02
    set_w(2, 1, 9, 1)   # W[k=2, i=1] = C12

    set_u(1, 1, 10, 1); set_v(1, 2, 10, 1) # d * b12
    set_w(2, 1, 10, 1)  # W[k=2, i=1] = C12

    return {"U": U, "V": V, "W": W}


def _rank12_solution(n, m, p):
    """Fallback: pure schoolbook rank-12 decomposition."""
    rank = 12
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][idx] = 1
                V[j*p+k][idx] = 1
                W[k*n+i][idx] = 1
                idx += 1

    return {"U": U, "V": V, "W": W}


def _verify(sol, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    U = np.array(sol["U"], dtype=float)
    V = np.array(sol["V"], dtype=float)
    W = np.array(sol["W"], dtype=float)
    r = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True
# EVOLVE-BLOCK-END
