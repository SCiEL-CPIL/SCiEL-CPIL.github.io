import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        rank = U_mat.shape[1]
        # Check non-zeros
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r in range(rank):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s - 1.0) > tol:
                        return False
        # Check zeros
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i = a // m
                    j = a % m
                    j_b = b // p
                    k = b % p
                    i_w = c % n
                    k_w = c // n
                    if i == i_w and j == j_b and k == k_w:
                        continue
                    s = 0.0
                    for r in range(rank):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s) > tol:
                        return False
        return True

    # Attempt rank 27 construction
    # Strategy:
    # Split p=6 into 3 blocks of 2: k_blocks = [0,1], [2,3], [4,5]
    # Split m=4 into:
    #   j=0,1 (2x2x2 block) -> Rank 7 (Strassen)
    #   j=2 (2x1x2 block) -> Rank 1
    #   j=3 (2x1x2 block) -> Rank 1
    # Total per k-block: 7 + 1 + 1 = 9
    # Total rank: 3 * 9 = 27

    rank = 27
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)

    def add_strassen_222(i_start, j_start, k_start, offset):
        # i_start: row block of A (always 0 since n=2)
        # j_start: col block of A (0 or 2? No, j is 0..1 or 2..3? No, we pass specific j's)
        # Here j_start is the first column index in A for the 2x2 block.
        # k_start: col block of B (first col index in B for the 2x2 block)
        
        # A indices
        a00 = (0) * m + (j_start + 0)
        a01 = (0) * m + (j_start + 1)
        a10 = (1) * m + (j_start + 0)
        a11 = (1) * m + (j_start + 1)
        
        # B indices
        b00 = (j_start + 0) * p + (k_start + 0)
        b01 = (j_start + 0) * p + (k_start + 1)
        b10 = (j_start + 1) * p + (k_start + 0)
        b11 = (j_start + 1) * p + (k_start + 1)
        
        # W indices
        # C is n x p.
        # A block rows: 0,1. Cols: j_start, j_start+1.
        # B block rows: j_start, j_start+1. Cols: k_start, k_start+1.
        # Product C block: rows 0,1. Cols k_start, k_start+1.
        # Global C index for local (i_loc, k_loc) is (k_start + k_loc)*n + (0 + i_loc).
        w00 = (k_start + 0) * n + 0
        w01 = (k_start + 1) * n + 0
        w10 = (k_start + 0) * n + 1
        w11 = (k_start + 1) * n + 1
        
        # Term 0 (M1): (A00+A11)(B00+B11)
        U[a00, offset] += 1.0
        U[a11, offset] += 1.0
        V[b00, offset] += 1.0
        V[b11, offset] += 1.0
        W[w00, offset] += 1.0
        W[w11, offset] += 1.0
        
        # Term 1 (M2): (A10+A11)B00
        U[a10, offset + 1] += 1.0
        U[a11, offset + 1] += 1.0
        V[b00, offset + 1] += 1.0
        W[w10, offset + 1] += 1.0
        W[w11, offset + 1] -= 1.0
        
        # Term 2 (M3): A00(B01-B11)
        U[a00, offset + 2] += 1.0
        V[b01, offset + 2] += 1.0
        V[b11, offset + 2] -= 1.0
        W[w01, offset + 2] += 1.0
        W[w11, offset + 2] += 1.0
        
        # Term 3 (M4): A11(B10-B00)
        U[a11, offset + 3] += 1.0
        V[b10, offset + 3] += 1.0
        V[b00, offset + 3] -= 1.0
        W[w00, offset + 3] += 1.0
        W[w10, offset + 3] += 1.0
        
        # Term 4 (M5): (A00+A01)B11
        U[a00, offset + 4] += 1.0
        U[a01, offset + 4] += 1.0
        V[b11, offset + 4] += 1.0
        W[w00, offset + 4] -= 1.0
        W[w01, offset + 4] += 1.0
        
        # Term 5 (M6): (A10-A00)(B00+B01)
        U[a10, offset + 5] += 1.0
        U[a00, offset + 5] -= 1.0
        V[b00, offset + 5] += 1.0
        V[b01, offset + 5] += 1.0
        W[w11, offset + 5] += 1.0
        
        # Term 6 (M7): (A01-A11)(B10+B11)
        U[a01, offset + 6] += 1.0
        U[a11, offset + 6] -= 1.0
        V[b10, offset + 6] += 1.0
        V[b11, offset + 6] += 1.0
        W[w00, offset + 6] += 1.0

    def add_rank1_212(j_idx, k_start, offset):
        # A is 2x1 (col j_idx), B is 1x2 (row j_idx, cols k_start, k_start+1)
        # C is 2x2 (rows 0,1, cols k_start, k_start+1)
        
        # A indices
        a0 = 0 * m + j_idx
        a1 = 1 * m + j_idx
        
        # B indices
        b0 = j_idx * p + k_start
        b1 = j_idx * p + (k_start + 1)
        
        # W indices
        w00 = k_start * n + 0
        w01 = (k_start + 1) * n + 0
        w10 = k_start * n + 1
        w11 = (k_start + 1) * n + 1
        
        # Single term
        U[a0, offset] = 1.0
        U[a1, offset] = 1.0
        V[b0, offset] = 1.0
        V[b1, offset] = 1.0
        W[w00, offset] = 1.0
        W[w01, offset] = 1.0
        W[w10, offset] = 1.0
        W[w11, offset] = 1.0

    # Build the decomposition
    offset = 0
    for k_block in range(3):
        k_start = k_block * 2
        
        # Block 1: j=0,1 (2x2x2)
        add_strassen_222(0, 0, k_start, offset)
        offset += 7
        
        # Block 2: j=2 (2x1x2)
        add_rank1_212(2, k_start, offset)
        offset += 1
        
        # Block 3: j=3 (2x1x2)
        add_rank1_212(3, k_start, offset)
        offset += 1

    if verify(U, V, W):
        print(f"Found valid rank {rank} decomposition", file=sys.stderr)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        print(f"Rank {rank} verification failed", file=sys.stderr)
        # Fallback to rank 42 if 27 fails (should not happen)
        # Re-using the logic from the previous best known (Strassen blocks)
        rank_42 = 42
        U42 = np.zeros((nm, rank_42), dtype=float)
        V42 = np.zeros((mp, rank_42), dtype=float)
        W42 = np.zeros((pn, rank_42), dtype=float)
        
        def add_strassen_block_42(i_start, j_start, k_start, offset):
            a00 = (i_start + 0) * m + (j_start + 0)
            a01 = (i_start + 0) * m + (j_start + 1)
            a10 = (i_start + 1) * m + (j_start + 0)
            a11 = (i_start + 1) * m + (j_start + 1)
            
            b00 = (j_start + 0) * p + (k_start + 0)
            b01 = (j_start + 0) * p + (k_start + 1)
            b10 = (j_start + 1) * p + (k_start + 0)
            b11 = (j_start + 1) * p + (k_start + 1)
            
            w00 = (k_start + 0) * n + (i_start + 0)
            w01 = (k_start + 1) * n + (i_start + 0)
            w10 = (k_start + 0) * n + (i_start + 1)
            w11 = (k_start + 1) * n + (i_start + 1)
            
            U42[a00, offset] += 1.0; U42[a11, offset] += 1.0; V42[b00, offset] += 1.0; V42[b11, offset] += 1.0; W42[w00, offset] += 1.0; W42[w11, offset] += 1.0
            U42[a10, offset + 1] += 1.0; U42[a11, offset + 1] += 1.0; V42[b00, offset + 1] += 1.0; W42[w10, offset + 1] += 1.0; W42[w11, offset + 1] -= 1.0
            U42[a00, offset + 2] += 1.0; V42[b01, offset + 2] += 1.0; V42[b11, offset + 2] -= 1.0; W42[w01, offset + 2] += 1.0; W42[w11, offset + 2] += 1.0
            U42[a11, offset + 3] += 1.0; V42[b10, offset + 3] += 1.0; V42[b00, offset + 3] -= 1.0; W42[w00, offset + 3] += 1.0; W42[w10, offset + 3] += 1.0
            U42[a00, offset + 4] += 1.0; U42[a01, offset + 4] += 1.0; V42[b11, offset + 4] += 1.0; W42[w00, offset + 4] -= 1.0; W42[w01, offset + 4] += 1.0
            U42[a10, offset + 5] += 1.0; U42[a00, offset + 5] -= 1.0; V42[b00, offset + 5] += 1.0; V42[b01, offset + 5] += 1.0; W42[w11, offset + 5] += 1.0
            U42[a01, offset + 6] += 1.0; U42[a11, offset + 6] -= 1.0; V42[b10, offset + 6] += 1.0; V42[b11, offset + 6] += 1.0; W42[w00, offset + 6] += 1.0

        offset = 0
        for i_start in range(0, n, 2):
            for j_start in range(0, m, 2):
                for k_start in range(0, p, 2):
                    add_strassen_block_42(i_start, j_start, k_start, offset)
                    offset += 7
        
        if verify(U42, V42, W42):
            print(f"Found valid rank {rank_42} decomposition", file=sys.stderr)
            return {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}
        
        # Final fallback: Schoolbook rank 48
        rank_48 = 48
        U48 = np.zeros((nm, rank_48), dtype=float)
        V48 = np.zeros((mp, rank_48), dtype=float)
        W48 = np.zeros((pn, rank_48), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U48[i*m+j, col] = 1.0
                    V48[j*p+k, col] = 1.0
                    W48[k*n+i, col] = 1.0
                    col += 1
        return {"U": U48.tolist(), "V": V48.tolist(), "W": W48.tolist()}

BET: {mechanism: "Decomposing 2x4x6 into disjoint blocks of 2x2x2 (rank 7) and 2x1x2 (rank 1) yields a total rank of 27, significantly lower than the previous best of 42.", q: 0.95, delta: [0.6, 0.7]}
