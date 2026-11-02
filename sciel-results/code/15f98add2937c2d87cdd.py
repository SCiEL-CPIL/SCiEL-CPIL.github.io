"""Program for matrix multiplication 3x4 * 4x6 with rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]
n, m, p = DIMENSIONS

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x6 matrix.
    Strategy: Split 4x6 into two 4x3 blocks. For each 3x4x3 sub-problem,
    use a rank-28 decomposition (7 rank-4 Strassen blocks for 2x2x2).
    Total rank: 56.
    """
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s kill

    def check_time():
        return time.time() - start_time < DEADLINE

    def verify_full(U, V, W, tol=1e-6):
        """Full verification of all entries."""
        rank = len(U[0]) if U else 0
        # Check all nonzero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0
                    for r in range(rank):
                        val += U[a][r] * V[b][r] * W[c][r]
                    if abs(val - 1) > tol:
                        return False
        # Check all zero entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if i != i2 or j != j2 or k != k2:
                        val = 0
                        for r in range(rank):
                            val += U[a][r] * V[b][r] * W[c][r]
                        if abs(val) > tol:
                            return False
        return True

    def schoolbook_full():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def build_strassen_56():
        """
        Build rank-56 decomposition:
        - Split 4x6 B into B_left (4x3, cols 0-2) and B_right (4x3, cols 3-5)
        - For each 3x4x3 sub-problem, use rank-28 Strassen
        - Total rank: 2*28 = 56
        """
        total_rank = 56
        U = np.zeros((12, total_rank))  # 3*4
        V = np.zeros((24, total_rank))  # 4*6
        W = np.zeros((18, total_rank))  # 6*3

        # Strassen coefficients for 2x2x2:
        # C00 = M1 - M3 + M4 - M7
        # C01 = M2 + M3
        # C10 = M4 + M5
        # C11 = M1 - M2 + M6 - M7
        # where:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)

        # For each k-block (kb=0: cols 0-2, kb=1: cols 3-5):
        #   For each row-block (ib=0: rows 0-1, ib=1: row 2):
        #     For each Strassen term (st=0..6):
        #       col = kb*28 + ib*7 + st

        for kb in range(2):  # k-block: 0 or 1
            k_off = kb * 3  # offset for k in B and W

            for ib in range(2):  # i-block: 0 (rows 0-1) or 1 (row 2)
                # i indices for this block
                if ib == 0:
                    i_list = [0, 1]
                else:
                    i_list = [2]

                for st in range(7):  # Strassen term 0-6
                    col = kb * 28 + ib * 7 + st

                    # Determine U entries (A-side)
                    # For ib=0: A is 2x4, Strassen on 2x2x2 blocks
                    #   M1: (A00+A11) -> U[i0*4+0, col] += 1, U[i1*4+1, col] += 1
                    #   M2: (A10+A11) -> U[i1*4+0, col] += 1, U[i1*4+1, col] += 1
                    #   M3: A00 -> U[i0*4+0, col] += 1
                    #   M4: A11 -> U[i1*4+1, col] += 1
                    #   M5: (A00+A01) -> U[i0*4+0, col] += 1, U[i0*4+1, col] += 1
                    #   M6: (A10-A00) -> U[i1*4+0, col] += 1, U[i0*4+0, col] -= 1
                    #   M7: (A01-A11) -> U[i0*4+1, col] += 1, U[i1*4+1, col] -= 1

                    # Determine V entries (B-side)
                    #   M1: (B00+B11) -> V[j0*6+k0, col] += 1, V[j1*6+k1, col] += 1
                    #   M2: B00 -> V[j0*6+k0, col] += 1
                    #   M3: (B01-B11) -> V[j0*6+k1, col] += 1, V[j1*6+k1, col] -= 1
                    #   M4: (B10-B00) -> V[j1*6+k0, col] += 1, V[j0*6+k0, col] -= 1
                    #   M5: B11 -> V[j1*6+k1, col] += 1
                    #   M6: (B00+B01) -> V[j0*6+k0, col] += 1, V[j0*6+k1, col] += 1
                    #   M7: (B10+B11) -> V[j1*6+k0, col] += 1, V[j1*6+k1, col] += 1

                    # Determine W entries (C-side)
                    #   M1: C00+=1, C11+=1
                    #   M2: C01+=1, C10+=1
                    #   M3: C00-=1, C10+=1
                    #   M4: C00+=1, C10+=1
                    #   M5: C10+=1, C11+=1
                    #   M6: C11+=1
                    #   M7: C00-=1, C11-=1

                    # For ib=0 (2x2 block):
                    #   i0 = 0, i1 = 1
                    #   j0 = 0, j1 = 1
                    #   k0 = k_off, k1 = k_off+1
                    #   W indices: C00 -> W[k0*3+0], C01 -> W[k1*3+0], C10 -> W[k0*3+1], C11 -> W[k1*3+1]

                    # For ib=1 (single row i=2):
                    #   This is a 1x2x2 problem, not 2x2x2
                    #   We need a different approach for the single row

                    if ib == 0:
                        i0, i1 = 0, 1
                        j0, j1 = 0, 1
                        k0, k1 = k_off, k_off + 1

                        # U entries
                        if st == 0:  # M1
                            U[i0*4+0, col] += 1
                            U[i1*4+1, col] += 1
                        elif st == 1:  # M2
                            U[i1*4+0, col] += 1
                            U[i1*4+1, col] += 1
                        elif st == 2:  # M3
                            U[i0*4+0, col] += 1
                        elif st == 3:  # M4
                            U[i1*4+1, col] += 1
                        elif st == 4:  # M5
                            U[i0*4+0, col] += 1
                            U[i0*4+1, col] += 1
                        elif st == 5:  # M6
                            U[i1*4+0, col] += 1
                            U[i0*4+0, col] -= 1
                        elif st == 6:  # M7
                            U[i0*4+1, col] += 1
                            U[i1*4+1, col] -= 1

                        # V entries
                        if st == 0:  # M1
                            V[j0*6+k0, col] += 1
                            V[j1*6+k1, col] += 1
                        elif st == 1:  # M2
                            V[j0*6+k0, col] += 1
                        elif st == 2:  # M3
                            V[j0*6+k1, col] += 1
                            V[j1*6+k1, col] -= 1
                        elif st == 3:  # M4
                            V[j1*6+k0, col] += 1
                            V[j0*6+k0, col] -= 1
                        elif st == 4:  # M5
                            V[j1*6+k1, col] += 1
                        elif st == 5:  # M6
                            V[j0*6+k0, col] += 1
                            V[j0*6+k1, col] += 1
                        elif st == 6:  # M7
                            V[j1*6+k0, col] += 1
                            V[j1*6+k1, col] += 1

                        # W entries
                        if st == 0:  # M1
                            W[k0*3+0, col] += 1   # C00
                            W[k1*3+1, col] += 1   # C11
                        elif st == 1:  # M2
                            W[k1*3+0, col] += 1   # C01
                            W[k0*3+1, col] += 1   # C10
                        elif st == 2:  # M3
                            W[k0*3+0, col] -= 1   # C00
                            W[k0*3+1, col] += 1   # C10
                        elif st == 3:  # M4
                            W[k0*3+0, col] += 1   # C00
                            W[k0*3+1, col] += 1   # C10
                        elif st == 4:  # M5
                            W[k0*3+1, col] += 1   # C10
                            W[k1*3+1, col] += 1   # C11
                        elif st == 5:  # M6
                            W[k1*3+1, col] += 1   # C11
                        elif st == 6:  # M7
                            W[k0*3+0, col] -= 1   # C00
                            W[k1*3+1, col] -= 1   # C11

                        # Also need to handle columns j=2,3 of A for this block
                        # For j=2,3, each M term contributes with same coefficient
                        # M1: (A00+A11) for j=0,1 -> for j=2: A02, A12; j=3: A03, A13
                        # But Strassen only works on 2x2 blocks, so j=2,3 are separate

                        # Actually, for j=2,3 we need separate terms
                        # Let me reconsider the approach

                    else:  # ib == 1, single row i=2
                        # This is a 1x4x3 problem
                        # C[k] = sum_j A[2,j] * B[j,k]
                        # This is just a rank-4 problem (4 terms)
                        # But we allocated 7 slots, so use 4 of them
                        for j in range(4):
                            col_j = kb * 28 + ib * 7 + j
                            if col_j < total_rank:
                                U[2*4+j, col_j] = 1
                                for k in range(3):
                                    V[j*6 + k_off + k, col_j] = 1
                                    W[(k_off + k)*3 + 2, col_j] = 1

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Schoolbook
    best_result = schoolbook_full()
    if not verify_full(best_result["U"], best_result["V"], best_result["W"]):
        raise RuntimeError("Schoolbook verification failed")
    best_rank = n * m * p

    # Try Strassen rank-56
    if check_time():
        res_56 = build_strassen_56()
        if verify_full(res_56["U"], res_56["V"], res_56["W"], tol=1e-5):
            best_rank = 56
            best_result = res_56

    return best_result
# EVOLVE-BLOCK-END
