"""Tensor decomposition for 4x4 * 4x5 matrix multiplication.

Uses the Kronecker-product (Kronecker) construction:
  - 4x4x4x4 block: Strassen rank 7 via Kronecker (rank 49)
  - 4x4x4x1 block: schoolbook (rank 16)
Total rank = 65.

Index conventions:
  U: rows indexed by (i, j) -> i*m + j, where i in [0,4), j in [0,4)
  V: rows indexed by (j, k) -> j*p + k, where j in [0,4), k in [0,5)
  W: rows indexed by (k, i) -> k*n + i, where k in [0,5), i in [0,4)

The decomposition satisfies:
  sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
  if (i,j,k) corresponds to a valid multiplication entry, else 0.
"""

DIMENSIONS = [4, 4, 5]

# Strassen 2x2x2x2 rank-7 decomposition
# U2: rows (i,j) -> 2*i+j, columns 0..6
# V2: rows (j,k) -> 2*j+k, columns 0..6
# W2: rows (k,i) -> 2*k+i, columns 0..6

# Strassen's 7 terms for 2x2 matrix multiplication.
# Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
# M1 = (a+d)(e+h)
# M2 = (c+d)e
# M3 = a(f-h)
# M4 = d(g-e)
# M5 = (a+b)h
# M6 = (c-a)(e+f)
# M7 = (b-d)(g+h)
# C11 = M1 + M4 - M5 + M7
# C12 = M3 + M5
# C21 = M2 + M4
# C22 = M1 - M2 + M3 + M6

U2 = [
    [1, 0, 0, 0, 0, 0, 0],  # (0,0)
    [0, 0, 1, 0, 1, 0, 1],  # (0,1)
    [0, 1, 0, 0, 0, 1, 0],  # (1,0)
    [1, 1, -1, 1, 0, -1, -1],  # (1,1)
]

V2 = [
    [1, 0, 0, 0, 0, 0, 0],  # (0,0)
    [0, 1, 0, 1, 0, 1, 0],  # (0,1)
    [1, 1, -1, 0, 1, 0, 1],  # (1,0)
    [0, 0, 1, -1, 0, -1, -1],  # (1,1)
]

W2 = [
    [0, 0, 0, 0, 0, 0, 0],  # (0,0) - unused in standard Strassen? Let me check.
    # Actually let me recompute W2 properly.
    # W2[k, i] = coefficient of M_{k,i} in the output.
    # C = [[C11, C12], [C21, C22]]
    # W2[0,0] = C11 coefficients: M1 + M4 - M5 + M7 -> [1, 0, 0, 1, -1, 0, 1]
    # W2[0,1] = C12 coefficients: M3 + M5 -> [0, 0, 1, 0, 1, 0, 0]
    # W2[1,0] = C21 coefficients: M2 + M4 -> [0, 1, 0, 1, 0, 0, 0]
    # W2[1,1] = C22 coefficients: M1 - M2 + M3 + M6 -> [1, -1, 1, 0, 0, 1, 0]
]

# Redefine W2 with correct indices: row = 2*k + i
W2 = [
    [1, 0, 0, 1, -1, 0, 1],   # (0,0): C11
    [0, 0, 1, 0, 1, 0, 0],    # (0,1): C12
    [0, 1, 0, 1, 0, 0, 0],    # (1,0): C21
    [1, -1, 1, 0, 0, 1, 0],   # (1,1): C22
]


def _verify_strassen():
    """Verify that U2, V2, W2 correctly decompose 2x2 matrix multiplication."""
    import numpy as np
    U2n = np.array(U2, dtype=float)
    V2n = np.array(V2, dtype=float)
    W2n = np.array(W2, dtype=float)
    # Check: sum_c U2[i,j,c] * V2[j,k,c] * W2[k,i,c] should be 1 for all i,j,k
    # and the decomposition should be exact.
    # Actually the tensor T[i,j,k] = 1 for all i,j,k in {0,1}
    # and we need sum_c U2[i*2+j, c] * V2[j*2+k, c] * W2[k*2+i, c] = 1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = sum(U2[i * 2 + j][c] * V2[j * 2 + k][c] * W2[k * 2 + i][c] for c in range(7))
                assert abs(s - 1) < 1e-10, f"Strassen verify failed at i={i},j={j},k={k}: got {s}"
    return True


def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    
    # We decompose p=5 as 4+1.
    # Part 1: 4x4 x 4x4, rank 49 (Kronecker of Strassen 2x2)
    # Part 2: 4x4 x 4x1, rank 16 (schoolbook)
    # Total rank = 65
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
    # Part 1: 4x4 x 4x4 using Kronecker of Strassen
    # Rank 49, columns 0..48
    # U4[i*4+j, c] where c = c1*7 + c2
    # i = 2*i1 + i2, j = 2*j1 + j2
    # U4[8*i1 + 4*i2 + 2*j1 + j2, c1*7+c2] = U2[2*i1+j1, c1] * U2[2*i2+j2, c2]
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    row_u = 8 * i1 + 4 * i2 + 2 * j1 + j2
                    for c1 in range(7):
                        for c2 in range(7):
                            c = c1 * 7 + c2
                            val = U2[2 * i1 + j1][c1] * U2[2 * i2 + j2][c2]
                            if val != 0:
                                U[row_u][c] = val
    
    # V4[j*4+k, c] where j = 2*j1+j2, k = 2*k1+k2
    # V4[8*j1 + 4*j2 + 2*k1 + k2, c1*7+c2] = V2[2*j1+k1, c1] * V2[2*j2+k2, c2]
    
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    row_v = 8 * j1 + 4 * j2 + 2 * k1 + k2
                    for c1 in range(7):
                        for c2 in range(7):
                            c = c1 * 7 + c2
                            val = V2[2 * j1 + k1][c1] * V2[2 * j2 + k2][c2]
                            if val != 0:
                                V[row_v][c] = val
    
    # W4[k*4+i, c] where k = 2*k1+k2, i = 2*i1+i2
    # W4[8*k1 + 4*k2 + 2*i1 + i2, c1*7+c2] = W2[2*k1+i1, c1] * W2[2*k2+i2, c2]
    
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    row_w = 8 * k1 + 4 * k2 + 2 * i1 + i2
                    for c1 in range(7):
                        for c2 in range(7):
                            c = c1 * 7 + c2
                            val = W2[2 * k1 + i1][c1] * W2[2 * k2 + i2][c2]
                            if val != 0:
                                W[row_w][c] = val
    
    # Part 2: 4x4 x 4x1 using schoolbook, rank 16, columns 49..64
    # For each (i, j) in 0..3 x 0..3, k=4 (the 5th row of B, i.e., k index 4)
    # Wait, p=5, so k ranges 0..4. The "4x1" part is k=4.
    # U41[i*4+j, c] = 1 if c corresponds to (i,j)
    # V41[j*5+4, c] = 1 if c corresponds to (i,j)  -- note: V row is j*p+k = j*5+4
    # W41[4*4+i, c] = 1 if c corresponds to (i,j)  -- note: W row is k*n+i = 4*4+i
    
    offset = 49
    col = 0
    for i in range(4):
        for j in range(4):
            c = offset + col
            U[i * 4 + j][c] = 1
            V[j * 5 + 4][c] = 1  # j*p + k where k=4
            W[4 * 4 + i][c] = 1  # k*n + i where k=4
            col += 1
    
    return {"U": U, "V": V, "W": W}


# Quick self-test
if __name__ == "__main__":
    import numpy as np
    _verify_strassen()
    result = construct()
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    n, m, p = DIMENSIONS
    # Verify: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(U.shape[1]))
                expected = 1.0
                err = abs(s - expected)
                if err > max_err:
                    max_err = err
    print(f"Max error: {max_err}")
    assert max_err < 1e-10, f"Verification failed with max error {max_err}"
    print("All checks passed.")
