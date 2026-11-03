import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 6, 6, 6

    # Use 3x3 block partition: 2x2 grid of 3x3 blocks
    # Block-level: Strassen 2x2 (rank 7)
    # Block-internal: trivial 3x3 (rank 27)
    # Total initial rank: 7 * 27 = 189
    # Then greedy reduce

    bs = 3  # block size
    nb = n // bs  # 2 blocks per dimension

    # 2x2 Strassen terms: (A_pattern, B_pattern, C_pattern)
    # Each pattern is list of (block_row, block_col, coeff)
    strassen_2x2 = [
        # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2 = (A10+A11)*B00 -> C10 += M2, C11 -= M2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3 = A00*(B01-B11) -> C01 += M3, C11 += M3
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M4 = A11*(B10-B00) -> C00 += M4, C10 += M4
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5 = (A00+A01)*B11 -> C00 -= M5, C01 += M5
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6 = (A10-A00)*(B00+B01) -> C11 += M6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7 = (A01-A11)*(B10+B11) -> C00 += M7
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    # Trivial 3x3 decomposition: 27 terms
    # Term (i,j,k): A[i,k] * B[k,j] -> C[i,j]
    trivial_3x3 = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                trivial_3x3.append(([(i,k,1)], [(k,j,1)], [(i,j,1)]))

    R_initial = 7 * 27  # 189

    U = np.zeros((n*m, R_initial), dtype=np.int64)
    V = np.zeros((m*p, R_initial), dtype=np.int64)
    W = np.zeros((p*n, R_initial), dtype=np.int64)

    col = 0
    for (sA, sB, sC) in strassen_2x2:
        for (tA, tB, tC) in trivial_3x3:
            # Combine Strassen block-level with 3x3 internal
            for (sbr, sbc, sc) in sA:
                for (tir, tic, tc) in tA:
                    U[(sbr*bs + tir)*m + (sbc*bs + tic), col] += sc * tc
            for (sbr, sbc, sc) in sB:
                for (tir, tic, tc) in tB:
                    V[(sbr*bs + tir)*p + (sbc*bs + tic), col] += sc * tc
            for (sbr, sbc, sc) in sC:
                for (tir, tic, tc) in tC:
                    W[(sbr*bs + tir)*n + (sbc*bs + tic), col] += sc * tc
            col += 1

    # Verify initial decomposition
    def verify_fast(Um, Vm, Wm):
        U_r = Um.reshape(n, m, -1)
        V_r = Vm.reshape(m, p, -1)
        W_r = Wm.reshape(p, n, -1)
        T = np.einsum('ikt,kjt,jit->ijk', U_r, V_r, W_r)
        return np.all(T == 1)

    assert verify_fast(U, V, W), "Initial decomposition is invalid"

    # Greedy rank reduction
    def try_remove(Um, Vm, Wm, t):
        r = Um.shape[1]
        mask = np.ones(r, dtype=bool)
        mask[t] = False
        Um_c = Um[:, mask]
        Vm_c = Vm[:, mask]
        Wm_c = Wm[:, mask]

        U_r = Um_c.reshape(n, m, -1)
        V_r = Vm_c.reshape(m, p, -1)
        W_r = Wm_c.reshape(p, n, -1)

        T = np.einsum('ikt,kjt,jit->ijk', U_r, V_r, W_r)
        return np.all(T == 1)

    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_r = R_initial

    # Multiple passes of greedy removal
    for pass_num in range(20):
        if time.time() > deadline - 20:
            break
        removed_any = False
        for t in range(current_r - 1, -1, -1):
            if time.time() > deadline - 20:
                break
            if try_remove(current_U, current_V, current_W, t):
                mask = np.ones(current_r, dtype=bool)
                mask[t] = False
                current_U = current_U[:, mask]
                current_V = current_V[:, mask]
                current_W = current_W[:, mask]
                current_r -= 1
                removed_any = True
                break
        if not removed_any:
            break

    # Final verification
    assert verify_fast(current_U, current_V, current_W), "Final decomposition is invalid"

    return {
        "U": current_U.tolist(),
        "V": current_V.tolist(),
        "W": current_W.tolist()
    }
