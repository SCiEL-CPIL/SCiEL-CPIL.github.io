import time
import sys
import numpy as np
from collections import defaultdict

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        r = U_mat.shape[1]
        # Check non-zeros
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = np.dot(U_mat[a], V_mat[b] * W_mat[c])
                    if abs(s - 1.0) > tol:
                        return False
        # Check zeros
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    i_w = c % n
                    k_w = c // n
                    if i_a == i_w and j_a == j_b and k_b == k_w:
                        continue
                    s = np.dot(U_mat[a], V_mat[b] * W_mat[c])
                    if abs(s) > tol:
                        return False
        return True

    def add_strassen_block(U, V, W, i_start, j_start, k_start, offset):
        # Local A indices: (i, j) -> global i*m + j
        a00 = i_start * m + j_start
        a01 = i_start * m + j_start + 1
        a10 = (i_start + 1) * m + j_start
        a11 = (i_start + 1) * m + j_start + 1

        # Local B indices: (j, k) -> global j*p + k
        b00 = j_start * p + k_start
        b01 = j_start * p + k_start + 1
        b10 = (j_start + 1) * p + k_start
        b11 = (j_start + 1) * p + k_start + 1

        # Local W indices: (k, i) -> global k*n + i
        w00 = k_start * n + i_start
        w01 = (k_start + 1) * n + i_start
        w10 = k_start * n + i_start + 1
        w11 = (k_start + 1) * n + i_start + 1

        # Term 0: M1 = (A00+A11)(B00+B11)
        U[a00, offset] += 1.0; U[a11, offset] += 1.0
        V[b00, offset] += 1.0; V[b11, offset] += 1.0
        W[w00, offset] += 1.0; W[w11, offset] += 1.0

        # Term 1: M2 = (A10+A11)B00
        U[a10, offset+1] += 1.0; U[a11, offset+1] += 1.0
        V[b00, offset+1] += 1.0
        W[w10, offset+1] += 1.0; W[w11, offset+1] -= 1.0

        # Term 2: M3 = A00(B01-B11)
        U[a00, offset+2] += 1.0
        V[b01, offset+2] += 1.0; V[b11, offset+2] -= 1.0
        W[w01, offset+2] += 1.0; W[w11, offset+2] += 1.0

        # Term 3: M4 = A11(B10-B00)
        U[a11, offset+3] += 1.0
        V[b10, offset+3] += 1.0; V[b00, offset+3] -= 1.0
        W[w00, offset+3] += 1.0; W[w10, offset+3] += 1.0

        # Term 4: M5 = (A00+A01)B11
        U[a00, offset+4] += 1.0; U[a01, offset+4] += 1.0
        V[b11, offset+4] += 1.0
        W[w00, offset+4] -= 1.0; W[w01, offset+4] += 1.0

        # Term 5: M6 = (A10-A00)(B00+B01)
        U[a10, offset+5] += 1.0; U[a00, offset+5] -= 1.0
        V[b00, offset+5] += 1.0; V[b01, offset+5] += 1.0
        W[w11, offset+5] += 1.0

        # Term 6: M7 = (A01-A11)(B10+B11)
        U[a01, offset+6] += 1.0; U[a11, offset+6] -= 1.0
        V[b10, offset+6] += 1.0; V[b11, offset+6] += 1.0
        W[w00, offset+6] += 1.0

    # Attempt 1: Rank 24 construction (split m into 4 single columns)
    # This works because the tensor entries for C[i,k] depend on A[i,j] and B[j,k]
    # via a product that can be factored into rank-1 terms per (j,k) pair.
    # Specifically, for each j, k:
    # Term r: U[:, r] = e_{i0,j} + e_{i1,j} (vector with 1s at row i*m+j for i=0,1)
    #         V[:, r] = e_{j,k}
    #         W[:, r] = e_{k,0} + e_{k,1} (vector with 1s at col k*n+i for i=0,1)
    # Then sum_r U[a,r] V[b,r] W[c,r] = sum_{j,k: a=(i0,j) or (i1,j), b=(j,k), c=(k,i0) or (k,i1)} 1
    # For a fixed (i, j, k), the term exists if b=(j,k).
    # The sum over r for a specific (a,b,c) where a=(i,j), b=(j,k), c=(k,i) is:
    # Only the term for this specific (j,k) contributes.
    # U[a,r] = 1 if a corresponds to j. V[b,r] = 1 if b corresponds to j,k. W[c,r] = 1 if c corresponds to k,i.
    # So the product is 1.
    # For any other combination, at least one factor is 0.
    
    rank_try = 24
    U_try = np.zeros((nm, rank_try), dtype=float)
    V_try = np.zeros((mp, rank_try), dtype=float)
    W_try = np.zeros((pn, rank_try), dtype=float)
    
    col = 0
    for j in range(m):
        for k in range(p):
            # U: 1s at (0, j) and (1, j)
            U_try[0 * m + j, col] = 1.0
            U_try[1 * m + j, col] = 1.0
            # V: 1 at (j, k)
            V_try[j * p + k, col] = 1.0
            # W: 1s at (k, 0) and (k, 1)
            W_try[k * n + 0, col] = 1.0
            W_try[k * n + 1, col] = 1.0
            col += 1

    if verify(U_try, V_try, W_try):
        return {"U": U_try.tolist(), "V": V_try.tolist(), "W": W_try.tolist()}

    # Attempt 2: Rank 42 Strassen construction
    rank_init = 42
    U = np.zeros((nm, rank_init), dtype=float)
    V = np.zeros((mp, rank_init), dtype=float)
    W = np.zeros((pn, rank_init), dtype=float)

    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_block(U, V, W, i_start, j_start, k_start, offset)
                offset += 7

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Rank 48 Schoolbook
    rank0 = n * m * p
    U = np.zeros((nm, rank0), dtype=float)
    V = np.zeros((mp, rank0), dtype=float)
    W = np.zeros((pn, rank0), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * (m * p) + j * p + k
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
