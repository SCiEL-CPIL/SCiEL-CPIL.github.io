import numpy as np
import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 2, 5, 5

    # Build target tensor as sparse list of (a,b,c) indices where T[a,b,c]=1
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))

    def build_rank46():
        rank = 46
        U = np.zeros((10, rank))
        V = np.zeros((25, rank))
        W = np.zeros((10, rank))

        col = 0

        def add_term(i, j, k):
            nonlocal col
            if col >= rank:
                raise ValueError("Rank exceeded")
            U[i * 5 + j, col] = 1
            V[j * 5 + k, col] = 1
            W[k * 2 + i, col] = 1
            col += 1

        def add_strassen_block(ja, jb, ka, kb):
            nonlocal col
            # M1 = (A00+A11)(B00+B11)
            U[0 * 5 + ja, col] = 1
            U[1 * 5 + jb, col] = 1
            V[ja * 5 + ka, col] = 1
            V[jb * 5 + kb, col] = 1
            W[ka * 2 + 0, col] = 1
            W[kb * 2 + 1, col] = 1
            col += 1

            # M2 = (A10+A11)B00
            U[1 * 5 + ja, col] = 1
            U[1 * 5 + jb, col] = 1
            V[ja * 5 + ka, col] = 1
            W[ka * 2 + 1, col] = 1
            W[kb * 2 + 1, col] = -1
            col += 1

            # M3 = A00(B01-B11)
            U[0 * 5 + ja, col] = 1
            V[ja * 5 + kb, col] = 1
            V[jb * 5 + kb, col] = -1
            W[kb * 2 + 0, col] = 1
            W[kb * 2 + 1, col] = 1
            col += 1

            # M4 = A11(B10-B00)
            U[1 * 5 + jb, col] = 1
            V[jb * 5 + ka, col] = 1
            V[ja * 5 + ka, col] = -1
            W[ka * 2 + 0, col] = 1
            W[ka * 2 + 1, col] = 1
            col += 1

            # M5 = (A00+A01)B11
            U[0 * 5 + ja, col] = 1
            U[0 * 5 + jb, col] = 1
            V[jb * 5 + kb, col] = 1
            W[ka * 2 + 0, col] = -1
            W[kb * 2 + 0, col] = 1
            col += 1

            # M6 = (A10-A00)(B00+B01)
            U[1 * 5 + ja, col] = 1
            U[0 * 5 + ja, col] = -1
            V[ja * 5 + ka, col] = 1
            V[ja * 5 + kb, col] = 1
            W[kb * 2 + 1, col] = 1
            col += 1

            # M7 = (A01-A11)(B10+B11)
            U[0 * 5 + jb, col] = 1
            U[1 * 5 + jb, col] = -1
            V[jb * 5 + ka, col] = 1
            V[jb * 5 + kb, col] = 1
            W[ka * 2 + 0, col] = 1
            col += 1

        # Four 2x2 Strassen blocks covering j in {0,1,2,3} and k in {0,1,2,3}
        add_strassen_block(0, 1, 0, 1)
        add_strassen_block(2, 3, 0, 1)
        add_strassen_block(0, 1, 2, 3)
        add_strassen_block(2, 3, 2, 3)

        # Tail: j=4 (column 4 of A) times all k except k=4
        for i in [0, 1]:
            for k in [0, 1, 2, 3]:
                add_term(i, 4, k)

        # Tail: all j except j=4 times k=4 (row 4 of B)
        for i in [0, 1]:
            for j in [0, 1, 2, 3]:
                add_term(i, j, 4)

        # Tail: (i, 4, 4)
        for i in [0, 1]:
            add_term(i, 4, 4)

        return U, V, W, rank

    U46, V46, W46, _ = build_rank46()

    # Verify
    T_recon = np.einsum('ar,br,cr->abc', U46, V46, W46)
    T_target = np.zeros((10, 25, 10))
    for (a, b, c) in target_indices:
        T_target[a, b, c] = 1.0

    error46 = np.max(np.abs(T_recon - T_target))

    if error46 > 1e-9:
        # Fallback to schoolbook
        rank_school = 50
        U_s = np.zeros((10, rank_school))
        V_s = np.zeros((25, rank_school))
        W_s = np.zeros((10, rank_school))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * 5 + j, c] = 1
                    V_s[j * 5 + k, c] = 1
                    W_s[k * 2 + i, c] = 1
                    c += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    # Try to find lower rank by checking if any pair of terms can be merged
    # or if a different block structure yields lower rank.
    # For 2x5x5, the known lower bound is 40 and upper bound is 46.
    # Let's try a few alternative structured decompositions.

    if time.time() < deadline - 10:
        # Try rank 45: attempt to merge one tail term with a Strassen term
        # This requires careful analysis. For now, try a few random perturbations
        # of the rank-46 solution and check validity.
        
        # Alternative: try different block sizes
        # What if we use one 3x3 block and one 2x2 block?
        # 3x3 matrix multiplication has rank 19 (vs 27 schoolbook)
        # 2x2 has rank 7 (vs 8 schoolbook)
        # But our problem is 2x5x5, not 5x5x5.
        
        # The structure is: A is 2x5, B is 5x5, C is 2x5.
        # Think of it as: for each column j of A (5 columns), we compute
        # a contribution to C. The j-th column of A (size 2) multiplies
        # the j-th row of B (size 5) to give a 2x5 contribution to C.
        # So it's like 5 independent 2x1x5 products, but they share the
        # same A columns and B rows in a structured way.
        
        # Actually, the tensor is:
        # T[i*5+j, j*5+k, k*2+i] = 1
        # For fixed j: T[i*5+j, j*5+k, k*2+i] = 1 for i in {0,1}, k in {0,4}
        # This is a 2x5x2 tensor (i, k, i) -> rank at most 4 per j, but shared.
        
        # The rank-46 solution uses:
        # - 4 Strassen blocks of size 2x2 in (j,k) space: 4*7 = 28 terms
        # - 18 schoolbook tail terms
        # Total: 46
        
        # Can we do better? Try 3+2 split:
        # j in {0,1,2} as a 3x3 block, j in {3,4} as a 2x2 block
        # k in {0,1,2} as a 3x3 block, k in {3,4} as a 2x2 block
        # This gets complicated. Let's just stick with 46.
        
        pass

    return {"U": U46.tolist(), "V": V46.tolist(), "W": W46.tolist()}
