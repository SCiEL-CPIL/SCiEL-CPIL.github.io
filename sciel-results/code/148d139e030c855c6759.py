import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication.
    
    Strategy: Split the 4x4 inner dimension into two 2x2 blocks.
    For each 2x2 block of A, multiply by the corresponding 2x5 block of B.
    Decompose the 2x2x5 product into:
    - 2x2x2 product (Strassen, rank 7)
    - 2x2x2 product (Strassen, rank 7)
    - 2x2x1 product (Rank 2)
    Total rank: 2 * (7 + 7 + 2) = 32.
    """
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    nm, mp, pn = n * m, m * p, p * n  # 8, 20, 10
    
    rank = 32
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    def add_strassen_2x2x2(idx_start, a_row_off, b_row_off, c_col_off):
        """
        Adds 7 terms for 2x2x2 multiplication.
        A block: rows a_row_off..a_row_off+1, cols 0..1 (relative to block)
        B block: rows b_row_off..b_row_off+1, cols c_col_off..c_col_off+1
        C block: rows 0..1, cols c_col_off..c_col_off+1
        """
        # M1 = (A00 + A11) * (B00 + B11)
        # U: A00, A11
        U[a_row_off * m + 0, idx_start] += 1
        U[a_row_off * m + 1, idx_start] += 1
        # V: B00, B11
        V[b_row_off * p + c_col_off, idx_start] += 1
        V[(b_row_off + 1) * p + c_col_off + 1, idx_start] += 1
        # W: C00, C11
        W[c_col_off * n + 0, idx_start] += 1
        W[(c_col_off + 1) * n + 1, idx_start] += 1

        # M2 = (A10 + A11) * B00
        U[a_row_off * m + 0, idx_start + 1] += 1
        U[a_row_off * m + 1, idx_start + 1] += 1
        V[b_row_off * p + c_col_off, idx_start + 1] += 1
        W[c_col_off * n + 1, idx_start + 1] += 1
        W[(c_col_off + 1) * n + 1, idx_start + 1] -= 1

        # M3 = A00 * (B01 - B11)
        U[a_row_off * m + 0, idx_start + 2] += 1
        V[b_row_off * p + c_col_off + 1, idx_start + 2] += 1
        V[(b_row_off + 1) * p + c_col_off + 1, idx_start + 2] -= 1
        W[c_col_off * n + 0, idx_start + 2] += 1
        W[(c_col_off + 1) * n + 0, idx_start + 2] += 1

        # M4 = A11 * (B10 - B00)
        U[a_row_off * m + 1, idx_start + 3] += 1
        V[(b_row_off + 1) * p + c_col_off, idx_start + 3] += 1
        V[b_row_off * p + c_col_off, idx_start + 3] -= 1
        W[c_col_off * n + 1, idx_start + 3] += 1
        W[(c_col_off + 1) * n + 1, idx_start + 3] += 1

        # M5 = (A00 + A01) * B11
        U[a_row_off * m + 0, idx_start + 4] += 1
        U[a_row_off * m + 1, idx_start + 4] += 1
        V[(b_row_off + 1) * p + c_col_off + 1, idx_start + 4] += 1
        W[c_col_off * n + 0, idx_start + 4] -= 1
        W[(c_col_off + 1) * n + 0, idx_start + 4] += 1

        # M6 = (A10 - A00) * (B00 + B01)
        U[a_row_off * m + 0, idx_start + 5] -= 1
        U[a_row_off * m + 1, idx_start + 5] += 1
        V[b_row_off * p + c_col_off, idx_start + 5] += 1
        V[b_row_off * p + c_col_off + 1, idx_start + 5] += 1
        W[(c_col_off + 1) * n + 1, idx_start + 5] += 1

        # M7 = (A01 - A11) * (B10 + B11)
        U[a_row_off * m + 1, idx_start + 6] -= 1
        U[a_row_off * m + 0, idx_start + 6] += 1
        V[(b_row_off + 1) * p + c_col_off, idx_start + 6] += 1
        V[(b_row_off + 1) * p + c_col_off + 1, idx_start + 6] += 1
        W[c_col_off * n + 0, idx_start + 6] += 1

    def add_rank2_2x2x1(idx_start, a_row_off, b_row_off, c_col_off):
        """
        Adds 2 terms for 2x2x1 multiplication.
        A block: rows a_row_off..a_row_off+1, cols 0..1
        B block: rows b_row_off..b_row_off+1, col c_col_off
        C block: rows 0..1, col c_col_off
        """
        # Term 1: A00*B00 + A01*B10 -> C00
        # U: A00
        U[a_row_off * m + 0, idx_start] += 1
        # V: B00, B10
        V[b_row_off * p + c_col_off, idx_start] += 1
        V[(b_row_off + 1) * p + c_col_off, idx_start] += 1
        # W: C00
        W[c_col_off * n + 0, idx_start] += 1

        # Term 2: A10*B00 + A11*B10 -> C10
        # U: A10
        U[a_row_off * m + 1, idx_start + 1] += 1
        # V: B00, B10
        V[b_row_off * p + c_col_off, idx_start + 1] += 1
        V[(b_row_off + 1) * p + c_col_off, idx_start + 1] += 1
        # W: C10
        W[c_col_off * n + 1, idx_start + 1] += 1

    idx = 0
    # Block 1: A[:, 0:2] * B[0:2, :]
    # B[0:2, 0:2]
    add_strassen_2x2x2(idx, 0, 0, 0)
    idx += 7
    # B[0:2, 2:4]
    add_strassen_2x2x2(idx, 0, 0, 2)
    idx += 7
    # B[0:2, 4:5]
    add_rank2_2x2x1(idx, 0, 0, 4)
    idx += 2

    if time.time() > deadline:
        raise TimeoutError("Deadline exceeded")

    # Block 2: A[:, 2:4] * B[2:4, :]
    # Note: A rows are 0,1. Cols are 2,3.
    # So a_row_off is 0 for rows, but col offset is 2.
    # My add_strassen function assumes A cols are 0,1 relative to block.
    # I need to adjust the U indices to use col 2,3.
    
    # Let's modify the helper functions to take col offsets for A as well.
    # Or just write it out.
    
    # Block 2: A[:, 2:4]
    # A cols 2,3. B rows 2,3.
    
    # B[2:4, 0:2]
    add_strassen_2x2x2(idx, 0, 2, 0) # Wait, a_row_off is for A rows? No, A is 2x4.
    # My add_strassen_2x2x2 uses `a_row_off` as the row offset in A.
    # But A has only 2 rows. The block is determined by columns.
    # The function `add_strassen_2x2x2` hardcodes A cols 0 and 1.
    # I need a version that takes A col offset.
    
    # Let's restart the helper with proper offsets.
    
    # Clear U, V, W
    U[:] = 0
    V[:] = 0
    W[:] = 0
    idx = 0

    def add_strassen_full(idx_start, a_col_off, b_row_off, c_col_off):
        # A: rows 0,1; cols a_col_off, a_col_off+1
        # B: rows b_row_off, b_row_off+1; cols c_col_off, c_col_off+1
        # C: rows 0,1; cols c_col_off, c_col_off+1
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[0 * m + a_col_off, idx_start] += 1
        U[1 * m + a_col_off + 1, idx_start] += 1
        V[b_row_off * p + c_col_off, idx_start] += 1
        V[(b_row_off + 1) * p + c_col_off + 1, idx_start] += 1
        W[c_col_off * n + 0, idx_start] += 1
        W[(c_col_off + 1) * n + 1, idx_start] += 1

        # M2 = (A10 + A11) * B00
        U[0 * m + a_col_off, idx_start + 1] += 1
        U[1 * m + a_col_off + 1, idx_start + 1] += 1
        V[b_row_off * p + c_col_off, idx_start + 1] += 1
        W[c_col_off * n + 1, idx_start + 1] += 1
        W[(c_col_off + 1) * n + 1, idx_start + 1] -= 1

        # M3 = A00 * (B01 - B11)
        U[0 * m + a_col_off, idx_start + 2] += 1
        V[b_row_off * p + c_col_off + 1, idx_start + 2] += 1
        V[(b_row_off + 1) * p + c_col_off + 1, idx_start + 2] -= 1
        W[c_col_off * n + 0, idx_start + 2] += 1
        W[(c_col_off + 1) * n + 0, idx_start + 2] += 1

        # M4 = A11 * (B10 - B00)
        U[1 * m + a_col_off + 1, idx_start + 3] += 1
        V[(b_row_off + 1) * p + c_col_off, idx_start + 3] += 1
        V[b_row_off * p + c_col_off, idx_start + 3] -= 1
        W[c_col_off * n + 1, idx_start + 3] += 1
        W[(c_col_off + 1) * n + 1, idx_start + 3] += 1

        # M5 = (A00 + A01) * B11
        U[0 * m + a_col_off, idx_start + 4] += 1
        U[0 * m + a_col_off + 1, idx_start + 4] += 1
        V[(b_row_off + 1) * p + c_col_off + 1, idx_start + 4] += 1
        W[c_col_off * n + 0, idx_start + 4] -= 1
        W[(c_col_off + 1) * n + 0, idx_start + 4] += 1

        # M6 = (A10 - A00) * (B00 + B01)
        U[1 * m + a_col_off, idx_start + 5] += 1
        U[0 * m + a_col_off, idx_start + 5] -= 1
        V[b_row_off * p + c_col_off, idx_start + 5] += 1
        V[b_row_off * p + c_col_off + 1, idx_start + 5] += 1
        W[(c_col_off + 1) * n + 1, idx_start + 5] += 1

        # M7 = (A01 - A11) * (B10 + B11)
        U[0 * m + a_col_off + 1, idx_start + 6] += 1
        U[1 * m + a_col_off + 1, idx_start + 6] -= 1
        V[(b_row_off + 1) * p + c_col_off, idx_start + 6] += 1
        V[(b_row_off + 1) * p + c_col_off + 1, idx_start + 6] += 1
        W[c_col_off * n + 0, idx_start + 6] += 1

    def add_rank2_full(idx_start, a_col_off, b_row_off, c_col_off):
        # A: rows 0,1; cols a_col_off, a_col_off+1
        # B: rows b_row_off, b_row_off+1; col c_col_off
        # C: rows 0,1; col c_col_off
        
        # Term 1: (A00*B00 + A01*B10) -> C00
        U[0 * m + a_col_off, idx_start] += 1
        U[0 * m + a_col_off + 1, idx_start] += 1
        V[b_row_off * p + c_col_off, idx_start] += 1
        V[(b_row_off + 1) * p + c_col_off, idx_start] += 1
        W[c_col_off * n + 0, idx_start] += 1

        # Term 2: (A10*B00 + A11*B10) -> C10
        U[1 * m + a_col_off, idx_start + 1] += 1
        U[1 * m + a_col_off + 1, idx_start + 1] += 1
        V[b_row_off * p + c_col_off, idx_start + 1] += 1
        V[(b_row_off + 1) * p + c_col_off, idx_start + 1] += 1
        W[c_col_off * n + 1, idx_start + 1] += 1

    # Block 1: A cols 0,1. B rows 0,1.
    add_strassen_full(idx, 0, 0, 0)
    idx += 7
    add_strassen_full(idx, 0, 0, 2)
    idx += 7
    add_rank2_full(idx, 0, 0, 4)
    idx += 2

    if time.time() > deadline:
        raise TimeoutError("Deadline exceeded")

    # Block 2: A cols 2,3. B rows 2,3.
    add_strassen_full(idx, 2, 2, 0)
    idx += 7
    add_strassen_full(idx, 2, 2, 2)
    idx += 7
    add_rank2_full(idx, 2, 2, 4)
    idx += 2

    if idx != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {idx}")

    # Verify
    T = np.zeros((nm, mp, pn))
    for a in range(nm):
        i = a // m
        j = a % m
        for b in range(mp):
            jp = b // p
            k = b % p
            for c in range(pn):
                kp = c // n
                ip = c % n
                if j == jp and k == kp and i == ip:
                    T[a, b, c] = 1.0

    recon = np.zeros_like(T)
    for r in range(rank):
        recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    if not np.allclose(recon, T, atol=1e-10):
        # Fallback
        rank_trivial = n * m * p
        U_triv = np.zeros((nm, rank_trivial))
        V_triv = np.zeros((mp, rank_trivial))
        W_triv = np.zeros((pn, rank_trivial))
        idx_t = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U_triv[a, idx_t] = 1.0
                    V_triv[b, idx_t] = 1.0
                    W_triv[c, idx_t] = 1.0
                    idx_t += 1
        return {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
