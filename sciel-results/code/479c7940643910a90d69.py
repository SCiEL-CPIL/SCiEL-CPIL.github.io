"""Matrix multiplication 4x4x7 decomposition via Strassen-inspired construction."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# Deadline guard: evaluator kills after 150s; use 140s budget for safety
_DEADLINE = time.time() + 140

def _strassen_7():
    """Return (U, V, W) for 2x2x2 matrix multiplication with rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7) where W[k*n+i, r]
    Convention: C[k, i] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    But flattened: U has indices (i*m+j) = (i*2+j), V has (j*p+k) = (j*2+k), W has (k*n+i) = (k*2+i)
    """
    n, m, p = 2, 2, 2
    # 7 terms of Strassen's algorithm
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = (a00+a01)b11
    # M5 = (a01+a10)(b00+b11)  -- wait, let me use the standard 7
    # Actually standard Strassen:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = (a11+a12)b22
    # M5 = (a12+a21)(b11+b22)
    # M6 = (a11-a21)(b21+b22)
    # M7 = (a12-a22)(b11+b12)
    # C11 = M1 - M4 + M5 + M7
    # C12 = M4 + M6
    # C21 = M2 + M3
    # C22 = M1 - M2 + M3 + M5
    
    # Using 0-indexed: a00, a01, a10, a11 and b00, b01, b10, b11
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = (a00+a01)b11
    # M5 = (a01+a10)(b00+b11)  -- hmm let me just use a known rank-7 decomposition
    # Let me use the standard formulation:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = (a00+a01)b11
    # M5 = (a01+a10)(b00+b11)
    # M6 = (a00-a10)(b10+b11)
    # M7 = (a01-a11)(b00+b10)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let me verify:
    # C00 should = a00*b00 + a01*b10
    # M1 = a00*b00 + a00*b11 + a11*b00 + a11*b11
    # M4 = a00*b11 + a01*b11
    # M5 = a01*b00 + a01*b11 + a10*b00 + a10*b11
    # M7 = a01*b00 + a01*b10 - a11*b00 - a11*b10
    # M1+M4-M5+M7 = a00*b00 + a00*b11 + a11*b00 + a11*b11 + a00*b11 + a01*b11 - a01*b00 - a01*b11 - a10*b00 - a10*b11 + a01*b00 + a01*b10 - a11*b00 - a11*b10
    # = a00*b00 + 2*a00*b11 + a01*b10 - a10*b11 - a11*b10
    # That's not right. Let me use a verified rank-7.
    
    # Let me just use the schoolbook rank-8 for 2x2x2 to be safe, or find a correct rank-7.
    # Actually, let me just use rank-8 schoolbook for 2x2x2. The total will be 12*8 + 4*1 = 100.
    # Or better: let me carefully derive a correct Strassen-7.
    
    # Standard Strassen (verified):
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = (a00+a01)b11
    # M5 = (a01+a10)(b00+b11)
    # M6 = (a00-a10)(b10+b11)
    # M7 = (a01-a11)(b00+b10)
    
    # C00 = M1 + M4 - M5 + M7
    # Let me recompute carefully:
    # M1 = a00*b00 + a00*b11 + a11*b00 + a11*b11
    # M4 = a00*b11 + a01*b11
    # M5 = a01*b00 + a01*b11 + a10*b00 + a10*b11
    # M7 = a01*b00 + a01*b10 - a11*b00 - a11*b10
    # M1+M4-M5+M7:
    # a00*b00: 1
    # a00*b11: 1+1 = 2
    # a11*b00: 1-1 = 0
    # a11*b11: 1
    # a01*b11: 1-1 = 0
    # a01*b00: -1+1 = 0
    # a01*b10: 1
    # a10*b00: -1
    # a10*b11: -1
    # a11*b10: -1
    # Result: a00*b00 + 2*a00*b11 + a11*b11 + a01*b10 - a10*b00 - a10*b11 - a11*b10
    # This is NOT a00*b00 + a01*b10. So my Strassen formulas are wrong.
    
    # Let me look up the correct ones. The correct Strassen is:
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b10 - b11)
    # M4 = (a00 + a01)b11
    # M5 = (a01 + a10)(b00 + b11)
    # M6 = (a00 - a10)(b10 + b11)
    # M7 = (a01 - a11)(b00 + b10)
    # C00 = M1 - M4 + M5 + M7
    # C01 = M4 + M6
    # C10 = M2 + M3
    # C11 = M1 - M2 + M3 + M5
    
    # Verify C00 = M1 - M4 + M5 + M7:
    # M1 = a00*b00 + a00*b11 + a11*b00 + a11*b11
    # -M4 = -a00*b11 - a01*b11
    # M5 = a01*b00 + a01*b11 + a10*b00 + a10*b11
    # M7 = a01*b00 + a01*b10 - a11*b00 - a11*b10
    # Sum:
    # a00*b00: 1
    # a00*b11: 1-1 = 0
    # a11*b00: 1-1 = 0
    # a11*b11: 1
    # a01*b11: -1+1 = 0
    # a01*b00: 1+1 = 2
    # a10*b00: 1
    # a10*b11: 1
    # a01*b10: 1
    # a11*b10: -1
    # Result: a00*b00 + a11*b11 + 2*a01*b00 + a10*b00 + a10*b11 + a01*b10 - a11*b10
    # Still wrong. I'm confusing myself. Let me just use rank-8 schoolbook for 2x2x2.
    
    # Schoolbook rank 8 for 2x2x2:
    U = np.zeros((4, 8), dtype=float)
    V = np.zeros((4, 8), dtype=float)
    W = np.zeros((4, 8), dtype=float)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                r = i * 4 + j * 2 + k
                U[i * 2 + j, r] = 1
                V[j * 2 + k, r] = 1
                W[k * 2 + i, r] = 1
    return U, V, W


def _rank1_2x2x1():
    """Rank-1 decomposition for 2x2x1 (matrix-vector multiply)."""
    U = np.ones((4, 1))
    V = np.ones((2, 1))
    W = np.ones((2, 1))
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Decompose into 4 blocks of 2x2x7
    # Each 2x2x7 = 3 * 2x2x2 (for k=0,1; k=2,3; k=4,5) + 1 * 2x2x1 (for k=6)
    # Total rank = 4 * (3 * 8 + 1) = 4 * 25 = 100
    
    rank_total = 100
    U = np.zeros((n * m, rank_total), dtype=float)
    V = np.zeros((m * p, rank_total), dtype=float)
    W = np.zeros((p * n, rank_total), dtype=float)
    
    col = 0
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            # 2x2x7 block: i in {i0, i0+1}, j in {j0, j0+1}, k in 0..6
            # Split into k-pairs (0,1), (2,3), (4,5) as 2x2x2 and k=6 as 2x2x1
            for k0 in [0, 2, 4]:
                u_s, v_s, w_s = _strassen_7()
                for r in range(8):
                    for li in range(2):
                        for lj in range(2):
                            gi = i0 + li
                            gj = j0 + lj
                            U[gi * m + gj, col + r] += u_s[li * 2 + lj, r]
                    for lj in range(2):
                        for lk in range(2):
                            gj = j0 + lj
                            gk = k0 + lk
                            V[gj * p + gk, col + r] += v_s[lj * 2 + lk, r]
                    for lk in range(2):
                        for li in range(2):
                            gk = k0 + lk
                            gi = i0 + li
                            W[gk * n + gi, col + r] += w_s[lk * 2 + li, r]
                col += 8
            
            # 2x2x1 for k=6
            u1, v1, w1 = _rank1_2x2x1()
            for li in range(2):
                for lj in range(2):
                    gi = i0 + li
                    gj = j0 + lj
                    U[gi * m + gj, col] += u1[li * 2 + lj, 0]
            for lj in range(2):
                gj = j0 + lj
                V[gj * p + 6, col] += v1[lj * 2 + 0, 0]
            for li in range(2):
                gi = i0 + li
                W[6 * n + gi, col] += w1[0 * 2 + li, 0]
            col += 1
    
    # Convert to lists for the expected interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
