"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS  # 2, 4, 8

    # Strassen rank-7 tensor decomposition for 2x2x2
    # Indices: U[i*2+j] for A[i,j], V[j*2+k] for B[j,k], W[k*2+i] for C[k,i] (trace convention)
    # Strassen products:
    # P1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # P2 = (A10+A11)B00       -> contributes to C10, C11
    # P3 = A00(B01-B11)       -> contributes to C01, C11
    # P4 = A11(B10-B00)       -> contributes to C00, C10
    # P5 = (A00+A01)B11       -> contributes to C00, C10
    # P6 = (A10-A00)(B00+B01) -> contributes to C11
    # P7 = (A01-A11)(B10+B11) -> contributes to C00
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6

    # Each strassen entry: (U_mask, V_mask, W_mask)
    # U_mask: dict from flat A index (i*2+j) to coeff
    # V_mask: dict from flat B index (j*2+k) to coeff
    # W_mask: dict from flat C index (k*2+i) to coeff

    s = [
        # P1: (A00+A11)(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # P2: (A10+A11)B00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # P3: A00(B01-B11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # P4: A11(B10-B00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # P5: (A00+A01)B11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),
        # P6: (A10-A00)(B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # P7: (A01-A11)(B10+B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    # 8 independent 2x2x2 blocks:
    # A is 2x4, B is 4x8, C is 2x8
    # Block (ar, ac, br, bc): A[ar:ar+2, ac:ac+2] @ B[br:br+2, bc:bc+2] -> C[ar:ar+2, bc:bc+2]
    # ar in {0}, ac in {0, 2}, br in {0, 2}, bc in {0, 2, 4, 6}
    # Total: 1 * 2 * 2 * 4 = 16 blocks, each rank 7 -> total rank 112
    # Wait, that's worse than schoolbook 64. Let me reconsider.
    
    # Actually the correct decomposition:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Group j into pairs: j = 2*q + r where q in {0,1}, r in {0,1}
    # Group k into pairs: k = 2*s + t where s in {0,1,2,3}, t in {0,1}
    # Then C[i, 2*s+t] = sum_q sum_r A[i, 2*q+r] * B[2*q+r, 2*s+t]
    # = sum_q (A[i, 2*q] * B[2*q, 2*s] + A[i, 2*q+1] * B[2*q+1, 2*s]
    #          + A[i, 2*q] * B[2*q, 2*s+1] + A[i, 2*q+1] * B[2*q+1, 2*s+1])
    
    # This is still 8 independent 2x2x2 blocks if we group properly.
    # Let's define blocks by (q, s) where q in {0,1} (j-group) and s in {0,1,2,3} (k-group)
    # For fixed (q, s): C[:, 2*s:2*s+2] += A[:, 2*q:2*q+2] @ B[2*q:2*q+2, 2*s:2*s+2]
    # This is a 2x2 @ 2x2 = 2x2 multiplication, rank 7 via Strassen.
    # Total: 2 * 4 = 8 blocks, each rank 7, total rank 56.
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=np.int64)  # 8 x 56
    V = np.zeros((m * p, rank), dtype=np.int64)  # 32 x 56
    W = np.zeros((p * n, rank), dtype=np.int64)  # 16 x 56

    col = 0
    for q in range(2):       # j-group: 2*q, 2*q+1
        for s in range(4):   # k-group: 2*s, 2*s+1
            # A block: rows 0:2, cols 2*q:2*q+2 -> flat indices (i, 2*q), (i, 2*q+1) for i in 0,1
            # B block: rows 2*q:2*q+2, cols 2*s:2*s+2
            # C block: rows 0:2, cols 2*s:2*s+2
            for uc, vc, wc in s:
                for ai, coef in uc.items():
                    # ai is flat index in 2x2 A block: ai = i*2 + j_local, j_local in {0,1}
                    i = ai // 2
                    j_local = ai % 2
                    gj = 2 * q + j_local
                    U[i * m + gj, col] = coef
                for bj, coef in vc.items():
                    # bj is flat index in 2x2 B block: bj = j_local*2 + k_local
                    j_local = bj // 2
                    k_local = bj % 2
                    gj = 2 * q + j_local
                    gk = 2 * s + k_local
                    V[gj * p + gk, col] = coef
                for ci, coef in wc.items():
                    # ci is flat index in 2x2 C block: ci = k_local*2 + i
                    k_local = ci // 2
                    i = ci % 2
                    gk = 2 * s + k_local
                    W[gk * n + i, col] = coef
                col += 1

    assert col == rank

    if time.time() > deadline:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
