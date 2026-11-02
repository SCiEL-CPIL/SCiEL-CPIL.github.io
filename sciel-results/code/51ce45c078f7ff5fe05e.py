import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 8, 32, 16

    # Rank-56 block-Strassen decomposition.
    # The tensor for 2x4x8 matrix multiplication is a direct sum of 8 
    # independent 2x2x2 multiplication tensors.
    # Each 2x2x2 tensor has minimal rank 7 (Strassen).
    # Total rank = 8 * 7 = 56.
    
    rank = 56
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)

    # Strassen coefficients for 2x2 multiplication
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # We iterate over the 8 blocks.
    # Block defined by a (row-pair of A, col-pair of A) and b (col-pair of B, col-pair of C)
    # a in {0, 1} corresponds to A columns {0,1} and {2,3}
    # b in {0, 1, 2, 3} corresponds to B/C columns {0,1}, {2,3}, {4,5}, {6,7}
    # Note: The 'row' of B used is determined by 'a' as well in the standard block partition 
    # where we split the inner dimension m=4 into two halves.
    # Half 0: j in {0,1}. Uses A cols 0,1.
    # Half 1: j in {2,3}. Uses A cols 2,3.
    
    for a in range(2):
        for b in range(4):
            base = 7 * (4 * a + b)

            # A indices (i, j)
            # a=0 -> j=0,1; a=1 -> j=2,3
            a_j0 = 0 * m + (2 * a)
            a_j1 = 0 * m + (2 * a + 1)
            a_j2 = 1 * m + (2 * a)
            a_j3 = 1 * m + (2 * a + 1)
            
            # B indices (j, k)
            # j is determined by a: a=0 -> j=0,1; a=1 -> j=2,3
            # k is determined by b: b -> k=2b, 2b+1
            b_j0_k0 = (2 * a) * p + 2 * b
            b_j0_k1 = (2 * a) * p + 2 * b + 1
            b_j1_k0 = (2 * a + 1) * p + 2 * b
            b_j1_k1 = (2 * a + 1) * p + 2 * b + 1

            # C indices (k, i) -> flattened as k*n + i
            # k is determined by b: b -> k=2b, 2b+1
            # i is 0 or 1
            c_k0_i0 = (2 * b) * n + 0
            c_k1_i0 = (2 * b + 1) * n + 0
            c_k0_i1 = (2 * b) * n + 1
            c_k1_i1 = (2 * b + 1) * n + 1

            # Term 0: M1
            U[a_j0, base] += 1
            U[a_j3, base] += 1
            V[b_j0_k0, base] += 1
            V[b_j1_k1, base] += 1
            W[c_k0_i0, base] += 1
            W[c_k1_i1, base] += 1

            # Term 1: M2
            U[a_j2, base + 1] += 1
            U[a_j3, base + 1] += 1
            V[b_j0_k0, base + 1] += 1
            W[c_k0_i1, base + 1] += 1
            W[c_k1_i1, base + 1] -= 1

            # Term 2: M3
            U[a_j0, base + 2] += 1
            V[b_j0_k1, base + 2] += 1
            V[b_j1_k1, base + 2] -= 1
            W[c_k1_i0, base + 2] += 1
            W[c_k1_i1, base + 2] += 1

            # Term 3: M4
            U[a_j3, base + 3] += 1
            V[b_j1_k0, base + 3] += 1
            V[b_j0_k0, base + 3] -= 1
            W[c_k0_i0, base + 3] += 1
            W[c_k0_i1, base + 3] += 1

            # Term 4: M5
            U[a_j0, base + 4] += 1
            U[a_j1, base + 4] += 1
            V[b_j1_k1, base + 4] += 1
            W[c_k0_i0, base + 4] -= 1
            W[c_k1_i0, base + 4] += 1

            # Term 5: M6
            U[a_j2, base + 5] += 1
            U[a_j0, base + 5] -= 1
            V[b_j0_k0, base + 5] += 1
            V[b_j0_k1, base + 5] += 1
            W[c_k1_i1, base + 5] += 1

            # Term 6: M7
            U[a_j1, base + 6] += 1
            U[a_j3, base + 6] -= 1
            V[b_j1_k0, base + 6] += 1
            V[b_j1_k1, base + 6] += 1
            W[c_k0_i0, base + 6] += 1

    # Quick verification
    def verify(U, V, W, tol=1e-9):
        # Check a few random points or all points if fast
        # 2*4*8 = 64 points. Very fast.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to trivial rank 64 if something went wrong (should not happen)
        rank = 64
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
