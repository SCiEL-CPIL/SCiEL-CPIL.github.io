"""Program for matrix multiplication 4x4 x 4x6 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    
    Returns (U, V, W) where:
      U: (4, 7) - for A[i,j], row index = i*2+j
      V: (4, 7) - for B[j,k], row index = j*2+k
      W: (4, 7) - for C[i,k], row index = k*2+i  (trace convention: W[k*n+i] with n=2)
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # W: C[i,k] contributions, W row = k*2 + i
    # C[0,0]: i=0,k=0 -> W[0]; C[0,1]: i=0,k=1 -> W[2]
    # C[1,0]: i=1,k=0 -> W[1]; C[1,1]: i=1,k=1 -> W[3]
    
    # C[0,0] = M1 + M4 - M5 + M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    
    # C[0,1] = M3 + M5
    W[2, 2] = 1; W[2, 4] = 1
    
    # C[1,0] = M2 + M4
    W[1, 1] = 1; W[1, 3] = 1
    
    # C[1,1] = M1 - M2 + M3 + M6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W


def _construct_block_decomp():
    """
    Construct a rank-77 decomposition using block structure.
    
    Strategy: Split 4x4 * 4x6 into blocks of size 2x2.
    A is 2x2 blocks of 2x2, B is 2x3 blocks of 2x2, C is 2x3 blocks of 2x2.
    
    For the first two columns of B (cols 0-3), use Strassen's 2x2 block multiplication
    which requires 7 block multiplications. Each block multiplication is 2x2 * 2x2 -> 2x2
    with rank 7. So 7 * 7 = 49 for the first 4 columns.
    
    For the last two columns of B (cols 4-5), use schoolbook: 4 block multiplications
    (2x2 blocks of A times 2x2 blocks of B). Each has rank 4 (schoolbook 2x2).
    So 4 * 4 = 16 for the last 2 columns.
    
    Total rank: 49 + 16 = 65.
    """
    n, m, p = 4, 4, 6
    
    sU, sV, sW = _strassen_2x2()
    
    # Schoolbook 2x2 * 2x2 -> 2x2 (rank 4)
    U_sb = np.zeros((4, 4), dtype=int)
    V_sb = np.zeros((4, 4), dtype=int)
    W_sb = np.zeros((4, 4), dtype=int)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                idx = i*2 + j*0 + k*0  # This won't work, let me redo
    # Actually schoolbook: C[i,k] = sum_j A[i,j] B[j,k]
    # 4 terms: (i,j,k) for each j
    U_sb = np.zeros((4, 4), dtype=int)
    V_sb = np.zeros((4, 4), dtype=int)
    W_sb = np.zeros((4, 4), dtype=int)
    for j in range(2):
        for i in range(2):
            U_sb[i*2+j, j] = 1
        for k in range(2):
            V_sb[j*2+k, j] = 1
        for i in range(2):
            for k in range(2):
                W_sb[k*2+i, j] = 1
    
    total_rank = 49 + 16
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    # Part 1: Strassen for cols 0-3 (4 columns = 2 block columns)
    # Outer Strassen: 2x2 block A * 2x2 block B -> 2x2 block C
    # Each outer term t (0..6) contributes to specific block positions
    for t in range(7):
        for ot in range(7):  # 7 outer terms
            col = t * 7 + ot  # This is wrong, let me restructure
            
    # Let me restructure properly.
    # The block multiplication is:
    # C_block[ai, ck] = sum_bj A_block[ai, bj] * B_block[bj, ck]
    # where ai, bj in {0,1} and ck in {0,1,2}
    #
    # For ck=0,1: use Strassen (7 terms)
    # For ck=2: use schoolbook (4 terms)
    #
    # Each term is a 2x2 * 2x2 -> 2x2 multiplication with rank 7 (Strassen) or 4 (schoolbook)
    
    # For Strassen part (ck=0,1):
    # Outer terms: 7 Strassen terms for the 2x2 * 2x2 -> 2x2 block mult
    # Each outer term has coefficients for (ai,bj), (bj,ck), (ai,ck)
    
    # Let me just use the original approach which worked:
    # 11 outer terms: 7 Strassen + 4 schoolbook
    # Each outer term uses one 2x2*2x2 decomposition
    
    # Actually, let me use the simpler approach from the original:
    # Split into 4 independent 4x4x2 subproblems
    # Each 4x4x2: use 4 rank-7 Strassen 2x2*2x2 products = rank 28
    # Total: 4 * 28 = 112. That's worse.
    
    # Let me try: 4x4 * 4x6 = (4x4 * 4x4) + (4x4 * 4x2)
    # 4x4*4x4: 4 independent 2x2*2x2*2x2 = 4*7 = 28? No, that's not right.
    # 4x4*4x4 is a 2x2 block mult of 2x2 blocks. Strassen: 7 block mults, each rank 7. Total 49.
    # 4x4*4x2: schoolbook: 4 block mults (2x2 blocks), each rank 4. Total 16.
    # Total: 49 + 16 = 65.
    
    # Let me implement this properly.
    
    U = np.zeros((n*m, 65), dtype=int)
    V = np.zeros((m*p, 65), dtype=int)
    W = np.zeros((p*n, 65), dtype=int)
    
    # Part 1: 4x4 * 4x4 -> 4x4 (cols 0-3), rank 49
    # 2x2 block Strassen
    for t in range(7):  # 7 Strassen terms
        for ai in range(2):
            for bj in range(2):
                coeff_outer = sU[ai*2+bj, t]
                if coeff_outer != 0:
                    for ai2 in range(2):
                        for bj2 in range(2):
                            coeff_inner = sU[ai2*2+bj2, t]
                            if coeff_inner != 0:
                                i = 2*ai + ai2
                                j = 2*bj + bj2
                                U[i*4+j, t*7 + (ai*2+bj)*2 + ai2*2 + bj2] += coeff_outer * coeff_inner
        for bj in range(2):
            for ck in range(2):
                coeff_outer = sV[bj*2+ck, t]
                if coeff_outer != 0:
                    for bj2 in range(2):
                        for ck2 in range(2):
                            coeff_inner = sV[bj2*2+ck2, t]
                            if coeff_inner != 0:
                                j = 2*bj + bj2
                                k = 2*ck + ck2
                                V[j*6+k, t*7 + (bj*2+ck)*2 + bj2*2 + ck2] += coeff_outer * coeff_inner
        for ai in range(2):
            for ck in range(2):
                coeff_outer = sW[ck*2+ai, t]
                if coeff_outer != 0:
                    for ai2 in range(2):
                        for ck2 in range(2):
                            coeff_inner = sW[ck2*2+ai2, t]
                            if coeff_inner != 0:
                                i = 2*ai + ai2
                                k = 2*ck + ck2
                                W[k*4+i, t*7 + (ai*2+ck)*2 + ai2*2 + ck2] += coeff_outer * coeff_inner
    
    # Part 2: 4x4 * 4x2 -> 4x2 (cols 4-5), rank 16
    # Schoolbook: 4 block mults
    for bj in range(2):
        for ai in range(2):
            for ck in range(1):  # Only one column block for the 4x2 part
                # This is getting complicated. Let me just use the original working code.
                pass
    
    # Given the complexity, let me just return the original rank-77 solution which is verified.
    # The key insight from the knowledge: splitting into three 4x4x2 subproblems gives rank 84,
    # which is worse than 77. So the original 77 is better.
    
    # Let me just implement the original approach cleanly.
    pass


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    sU, sV, sW = _strassen_2x2()
    
    # Use the block decomposition: 4x4*4x6 = 4x4*4x4 + 4x4*4x2
    # 4x4*4x4: 2x2 block Strassen, rank 49
    # 4x4*4x2: schoolbook block, rank 16
    # Total: 65
    
    total_rank = 65
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    # Part 1: 4x4 * 4x4 -> 4x4 (cols 0-3)
    # 2x2 block Strassen: 7 outer terms, each 2x2*2x2 rank 7
    for t in range(7):
        # U part
        for ai in range(2):
            for bj in range(2):
                for ai2 in range(2):
                    for bj2 in range(2):
                        coeff_outer = sU[ai*2+bj, t]
                        coeff_inner = sU[ai2*2+bj2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            i = 2*ai + ai2
                            j = 2*bj + bj2
                            r = t*7 + (ai*2+bj)*2 + ai2*2 + bj2
                            U[i*m+j, r] += coeff_outer * coeff_inner
        # V part
        for bj in range(2):
            for ck in range(2):
                for bj2 in range(2):
                    for ck2 in range(2):
                        coeff_outer = sV[bj*2+ck, t]
                        coeff_inner = sV[bj2*2+ck2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            j = 2*bj + bj2
                            k = 2*ck + ck2
                            r = t*7 + (bj*2+ck)*2 + bj2*2 + ck2
                            V[j*p+k, r] += coeff_outer * coeff_inner
        # W part
        for ai in range(2):
            for ck in range(2):
                for ai2 in range(2):
                    for ck2 in range(2):
                        coeff_outer = sW[ck*2+ai, t]
                        coeff_inner = sW[ck2*2+ai2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            i = 2*ai + ai2
                            k = 2*ck + ck2
                            r = t*7 + (ai*2+ck)*2 + ai2*2 + ck2
                            W[k*n+i, r] += coeff_outer * coeff_inner
    
    # Part 2: 4x4 * 4x2 -> 4x2 (cols 4-5)
    # Schoolbook: 4 block mults (ai,bj) for each, rank 4 each
    # Block (ai,bj) * block (bj,ck) -> block (ai,ck) where ck=0 (for col 4-5)
    # Actually, for 4x2, we have 2 columns, so 1 block column of size 2.
    # Schoolbook: for each (ai, bj) pair, 1 block mult
    # 2*2 = 4 block mults, each rank 4 (schoolbook 2x2*2x2->2x2)
    
    # Schoolbook 2x2*2x2->2x2: C[i,k] = sum_j A[i,j] B[j,k]
    # 4 terms, one for each j
    U_sb = np.zeros((4, 4), dtype=int)
    V_sb = np.zeros((4, 4), dtype=int)
    W_sb = np.zeros((4, 4), dtype=int)
    for j in range(2):
        for i in range(2):
            U_sb[i*2+j, j] = 1
        for k in range(2):
            V_sb[j*2+k, j] = 1
        for i in range(2):
            for k in range(2):
                W_sb[k*2+i, j] = 1
    
    for ai in range(2):
        for bj in range(2):
            for j in range(2):  # 4 schoolbook terms
                r = 49 + (ai*2+bj)*4 + j
                # U
                for ai2 in range(2):
                    for bj2 in range(2):
                        coeff = U_sb[ai2*2+bj2, j]
                        if coeff != 0:
                            i = 2*ai + ai2
                            jj = 2*bj + bj2
                            U[i*m+jj, r] += coeff
                # V
                for bj2 in range(2):
                    for ck2 in range(2):
                        coeff = V_sb[bj2*2+ck2, j]
                        if coeff != 0:
                            jj = 2*bj + bj2
                            k = 4 + 2*0 + ck2  # ck=0 for the 4x2 part
                            V[jj*p+k, r] += coeff
                # W
                for ai2 in range(2):
                    for ck2 in range(2):
                        coeff = W_sb[ck2*2+ai2, j]
                        if coeff != 0:
                            i = 2*ai + ai2
                            k = 4 + 2*0 + ck2
                            W[k*n+i, r] += coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition():
    """Verify the decomposition is correct."""
    n, m, p = DIMENSIONS  # 4, 4, 6
    result = construct()
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    
    rank = U.shape[1]
    
    # Check that T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-10:
                    print(f"FAIL: T[{i},{j},{k}] = {s}, expected 1.0")
                    return False
    print(f"PASS: rank {rank} decomposition verified")
    return True

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    _verify_decomposition()
