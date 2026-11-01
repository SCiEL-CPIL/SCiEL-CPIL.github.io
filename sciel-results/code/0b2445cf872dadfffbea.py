"""
Rank-minimal tensor decomposition for 2x2 * 2x14 matrix multiplication.

Strategy:
- Schoolbook rank is 2*2*14 = 56.
- Split B (2x14) into 7 blocks of 2x2.
- Each 2x2 * 2x2 block uses Strassen's rank-7 algorithm.
- Total rank = 7 * 7 = 49.

Strassen for 2x2 * 2x2 (A=[[a,b],[c,d]], B=[[e,f],[g,h]]):
  P1 = (a+d)(e+h)
  P2 = (c+d)e
  P3 = a(f-h)
  P4 = d(g-e)
  P5 = (a+b)h
  P6 = (c-a)(e+f)
  P7 = (b-d)(g+h)
  C00 = P1+P4-P5+P7, C01 = P3+P5, C10 = P2+P4, C11 = P1-P2+P3+P6

Tensor convention:
  U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = 1 for all i,j,k.
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 145.0

    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 7
    R = 7  # Strassen rank per 2x2 block
    rank = num_blocks * R  # 49

    # Strassen terms: (U_coeffs, V_local_coeffs, W_local_coeffs)
    # U rows: 0=A00, 1=A01, 2=A10, 3=A11
    # V local: 0=B00, 1=B01, 2=B10, 3=B11
    # W local: 0=C00, 1=C10, 2=C01, 3=C11
    terms = [
        (([0, 3], [1, 1]), ([0, 3], [1, 1]), ([0, 3], [1, 1])),
        (([2, 3], [1, 1]), ([0], [1]), ([1, 3], [1, -1])),
        (([0], [1]), ([1, 3], [1, -1]), ([2, 3], [1, 1])),
        (([3], [1]), ([2, 0], [1, -1]), ([0, 1], [1, 1])),
        (([0, 1], [1, 1]), ([3], [1]), ([0, 2], [-1, 1])),
        (([2, 0], [1, -1]), ([0, 1], [1, 1]), ([3], [1])),
        (([1, 3], [1, -1]), ([2, 3], [1, 1]), ([0], [1])),
    ]

    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    for blk in range(num_blocks):
        k0 = 2 * blk
        k1 = k0 + 1
        for s in range(R):
            col = blk * R + s
            u_idx, u_val, v_idx, v_val, w_idx, w_val = terms[s]

            for ui, uv in zip(u_idx, u_val):
                U[ui, col] = uv

            for vi, vv in zip(v_idx, v_val):
                j = vi // 2
                kc = k0 + (vi % 2)
                V[j * p + kc, col] = vv

            for wi, wv in zip(w_idx, w_val):
                i = wi % 2
                kc = k0 + (wi // 2)
                W[kc * n + i, col] = wv

    # Verify: for each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    # Use einsum for speed
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Reshape U to (n,m,rank), V to (m,p,rank), W to (p,n,rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    max_err = float(np.max(np.abs(T - 1.0)))

    if max_err > 1e-9:
        # Fallback: schoolbook rank 56
        rb = n * m * p
        U_fb = np.zeros((n * m, rb), dtype=np.float64)
        V_fb = np.zeros((m * p, rb), dtype=np.float64)
        W_fb = np.zeros((p * n, rb), dtype=np.float64)
        c = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_fb[ii * m + jj, c] = 1.0
                    V_fb[jj * p + kk, c] = 1.0
                    W_fb[kk * n + ii, c] = 1.0
                    c += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
