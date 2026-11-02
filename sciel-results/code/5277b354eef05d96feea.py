import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill

    def time_left():
        return deadline - (time.time() - start_time)

    # Build a schoolbook decomposition for an (nn x mm x pp) multiplication.
    # Returns U (nn*mm, r), V (mm*pp, r), W (pp*nn, r) with integer entries.
    def schoolbook(nn, mm, pp):
        r = nn * mm * pp
        U = np.zeros((nn * mm, r), dtype=object)
        V = np.zeros((mm * pp, r), dtype=object)
        W = np.zeros((pp * nn, r), dtype=object)
        col = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i * mm + j, col] = 1
                    V[j * pp + k, col] = 1
                    W[k * nn + i, col] = 1
                    col += 1
        return U, V, W

    # Build a Strassen 2x2x2 decomposition (rank 7).
    # Returns U (2*2=4, 7), V (2*2=4, 7), W (2*2=4, 7).
    def strassen_2x2x2():
        # The 7 Strassen products:
        # M1 = (A00+A11)(B01+B10)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # U rows indexed by (i,j) with i*2+j, V by (j,k) with j*2+k, W by (k,i) with k*2+i
        # We need U[i,j]*V[j,k]*W[k,i] summed over r = 1 for the tensor T[i,j,k]=1.
        U = np.zeros((4, 7), dtype=object)
        V = np.zeros((4, 7), dtype=object)
        W = np.zeros((4, 7), dtype=object)
        # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        # For each of 7 terms, set coefficients.
        # Term r: contributes sum over (i,j,k) of U[i,j]*V[j,k]*W[k,i]
        # We want total to be 1 for all (i,j,k).
        # Standard Strassen: C00 = M1+M4-M5-M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3-M6
        # where C is 2x2 output, A is 2x2 input (Aij), B is 2x2 input (Bjk).
        # C[ci][ck] = sum_{cj} A[cj][cj]... no. C[ci][ck] = sum_j A[ci][j] B[j][ck].
        # So the tensor T[ci, j, ck] = 1 for all ci,j,ck.
        # We need: sum_r U[ci,j] V[j,ck] W[ck,ci] = 1 for all ci,j,ck.
        # Let's assign each M_r a coefficient pattern.
        # M1: A00+A11 times B01+B10. So U[0,0]+=c, U[1,1]+=c; V[0,1]+=c, V[1,0]+=c; W[?].
        # Actually W[ck,ci] multiplies. For M1 to contribute to C[0,0]=M1 and C[1,1]=M1,
        # we need W[0,0]=1 and W[1,1]=1 for M1's coefficient... but M1 appears in C00 with +1 and C11 with +1.
        # Let me define: for term r, U_r[i,j], V_r[j,k], W_r[k,i].
        # C[i,k] gets contribution sum_j (sum_r U_r[i,j] V_r[j,k]) W_r[k,i]... no, the rank-1 term is
        # sum_r (U_r[i,j])(V_r[j,k])(W_r[k,i]) and this must equal 1 for each (i,j,k).
        # So it's a 3-way. Let me just verify by construction and test.
        # M1 = (A00+A11)(B01+B10): 
        #   U: A00 -> U[0,0]=1, A11 -> U[1,1]=1
        #   V: B01 -> V[0,1]=1, B10 -> V[1,0]=1
        #   W: must be 1 for all (k,i) so that M1 contributes to all C? No.
        # Hmm, the issue: W_r[k,i] is the same for all j. So M1's W must make it contribute correctly.
        # Let me reconsider. The rank-1 tensor is U_r[i,j] * V_r[j,k] * W_r[k,i].
        # For M1: we want the product (A00+A11)(B01+B10) to appear in C.
        # C[0,0] = A00*B00 + A01*B10 ... standard. 
        # This is getting complex. Let me just use a known-correct Strassen assignment and verify numerically.
        # Known: the 7 terms with their (i,j,k) coefficient patterns:
        # I'll define each term by which (i,j) of U, (j,k) of V, (k,i) of W get +/-1.
        # Term 1: U[0,0]=1,U[1,1]=1; V[0,1]=1,V[1,0]=1; W[0,0]=1,W[1,1]=1
        # Term 2: U[1,0]=1,U[1,1]=1; V[0,0]=1; W[0,1]=1,W[1,1]=1
        # Term 3: U[0,0]=1; V[1,0]=1,V[1,1]=-1; W[0,0]=1
        # Term 4: U[1,1]=1; V[0,1]=1,V[0,0]=-1; W[1,0]=1
        # Term 5: U[0,0]=1,U[0,1]=1; V[1,1]=1; W[0,1]=1
        # Term 6: U[1,0]=1,U[0,0]=-1; V[0,0]=1,V[0,1]=1; W[1,1]=1
        # Term 7: U[0,1]=1,U[1,1]=-1; V[1,0]=1,V[1,1]=1; W[0,0]=1
        terms = [
            # (list of (i,j,coeff)), (list of (j,k,coeff)), (list of (k,i,coeff))
            ([(0,0,1),(1,1,1)], [(0,1,1),(1,0,1)], [(0,0,1),(1,1,1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,1)]),
            ([(0,0,1)], [(1,0,1),(1,1,-1)], [(0,0,1)]),
            ([(1,1,1)], [(0,1,1),(0,0,-1)], [(1,0,1)]),
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,1,1)]),
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
        ]
        for r, (u_list, v_list, w_list) in enumerate(terms):
            for (i, j, c) in u_list:
                U[i * 2 + j, r] += c
            for (j, k, c) in v_list:
                V[j * 2 + k, r] += c
            for (k, i, c) in w_list:
                W[k * 2 + i, r] += c
        return U, V, W

    def verify_decomp(U, V, W, nn, mm, pp, tol=1e-9):
        # U: (nn*mm, r), V: (mm*pp, r), W: (pp*nn, r)
        r = U.shape[1]
        Uf = np.array(U, dtype=np.float64)
        Vf = np.array(V, dtype=np.float64)
        Wf = np.array(W, dtype=np.float64)
        U3 = Uf.T.reshape(nn, mm, r)
        V3 = Vf.T.reshape(mm, pp, r)
        W3 = Wf.T.reshape(pp, nn, r)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        err = np.max(np.abs(T - 1.0))
        return err < tol

    # Verify Strassen 2x2x2
    sU, sV, sW = strassen_2x2x2()
    s_ok = verify_decomp(sU, sV, sW, 2, 2, 2)

    # Build 4x2x6 using Strassen: split 4->2+2, 6->2+2+2, 2 stays.
    # For each of the 3 output k-blocks (size 2), use one Strassen 2x2x2 for the 2x2x2.
    # But the input j-dim is 2 (single block). So for each (i-block of 2, k-block of 2), 
    # we have a 2x2x2 multiplication. There are 2 i-blocks and 3 k-blocks = 6 such 2x2x2's.
    # Each uses rank 7. Total rank 42. That's worse than schoolbook 48? No, 42 < 48.
    # Wait, but they share the same input A (4x2) and B (2x6). The i-blocks and k-blocks are independent
    # in the output, so we can use separate Strassen for each. But the input factors are shared,
    # meaning we can't just concatenate naively because U factors differ per block.
    # Actually for block decomposition: C[i_block, k_block] = A[i_block, :] B[:, k_block].
    # Each is a 2x2x2 problem. We solve each independently and stack.
    # U for block (ib, kb): the A-part is A[ib*2:(ib+1)*2, :] (2x2), so U rows are (2*2)=4 for that block.
    # We place them in the global U at rows corresponding to (i in ib, j in 0..1).
    # Similarly V for block: B[:, kb*2:(kb+1)*2] (2x2), V rows (j, k in kb).
    # W for block: C[ib, kb] (2x2), W rows (k in kb, i in ib).
    # Since blocks are independent, we can just use the same Strassen pattern for each and offset columns.
    def build_4x2x6():
        # 4x2x6: n=4, m=2, p=6
        n, m, p = 4, 2, 6
        r_total = 0
        U = np.zeros((n * m, 0), dtype=object)
        V = np.zeros((m * p, 0), dtype=object)
        W = np.zeros((p * n, 0), dtype=object)
        for ib in range(n // 2):  # 2 i-blocks
            for kb in range(p // 2):  # 3 k-blocks
                sU, sV, sW = strassen_2x2x2()
                # sU: (4, 7) rows (i_local*2+j), i_local in 0..1, j in 0..1
                # sV: (4, 7) rows (j*2+k_local), k_local in 0..1
                # sW: (4, 7) rows (k_local*2+i_local)
                for col in range(7):
                    # U column
                    u_col = np.zeros(n * m, dtype=object)
                    for row in range(4):
                        i_local = row // 2
                        j = row % 2
                        i_global = ib * 2 + i_local
                        u_col[i_global * m + j] = sU[row, col]
                    # V column
                    v_col = np.zeros(m * p, dtype=object)
                    for row in range(4):
                        j = row // 2
                        k_local = row % 2
                        k_global = kb * 2 + k_local
                        v_col[j * p + k_global] = sV[row, col]
                    # W column
                    w_col = np.zeros(p * n, dtype=object)
                    for row in range(4):
                        k_local = row // 2
                        i_local = row % 2
                        k_global = kb * 2 + k_local
                        i_global = ib * 2 + i_local
                        w_col[k_global * n + i_global] = sW[row, col]
                    U = np.hstack([U, u_col.reshape(-1, 1)])
                    V = np.hstack([V, v_col.reshape(-1, 1)])
                    W = np.hstack([W, w_col.reshape(-1, 1)])
                r_total += 7
        return U, V, W, r_total

    # Build 4x3x6 using schoolbook: n=4, m=3, p=6, rank 72
    def build_4x3x6():
        return schoolbook(4, 3, 6)

    # Combine 4x2x6 and 4x3x6 into 4x5x6.
    # The 4x2x6 handles j in 0..1, the 4x3x6 handles j in 2..4.
    # U rows: 4x5=20. For 4x2x6, U has 4*2=8 rows (j in 0..1). For 4x3x6, U has 4*3=12 rows (j in 2..4).
    # We need to merge: global U[i*5+j, r]. 
    # 4x2x6 U[i*2+j, r] for j in 0..1 -> global row i*5+j.
    # 4x3x6 U[i*3+j, r] for j in 0..2 -> global row i*5+(j+2).
    def build_full():
        U1, V1, W1, r1 = build_4x2x6()
        U2, V2, W2 = build_4x3x6()
        r2 = U2.shape[1]
        r_total = r1 + r2
        n, m, p = 4, 5, 6
        U = np.zeros((n * m, r_total), dtype=object)
        V = np.zeros((m * p, r_total), dtype=object)
        W = np.zeros((p * n, r_total), dtype=object)
        # Copy U1 into columns 0..r1-1
        for i in range(4):
            for j in range(2):
                for col in range(r1):
                    U[i * m + j, col] = U1[i * 2 + j, col]
        # Copy U2 into columns r1..r_total-1
        for i in range(4):
            for j in range(3):
                for col in range(r2):
                    U[i * m + (j + 2), r1 + col] = U2[i * 3 + j, col]
        # V1: rows (j in 0..1, k in 0..5) -> V1[j*6+k, col]. Global V[j*6+k, col].
        for j in range(2):
            for k in range(6):
                for col in range(r1):
                    V[j * p + k, col] = V1[j * 6 + k, col]
        # V2: rows (j in 0..2, k in 0..5) -> V2[j*6+k, col]. Global V[(j+2)*6+k, r1+col].
        for j in range(3):
            for k in range(6):
                for col in range(r2):
                    V[(j + 2) * p + k, r1 + col] = V2[j * 6 + k, col]
        # W1: rows (k in 0..5, i in 0..3) -> W1[k*4+i, col]. Global W[k*4+i, col].
        for k in range(6):
            for i in range(4):
                for col in range(r1):
                    W[k * n + i, col] = W1[k * 4 + i, col]
        # W2: rows (k in 0..5, i in 0..3) -> W2[k*4+i, col]. Global W[k*4+i, r1+col].
        for k in range(6):
            for i in range(4):
                for col in range(r2):
                    W[k * n + i, r1 + col] = W2[k * 4 + i, col]
        return U, V, W, r_total

    # Build and verify
    U, V, W, rank = build_full()
    ok = verify_decomp(U, V, W, 4, 5, 6)

    if ok and rank < 120:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook 4x5x6 rank 120
    U, V, W = schoolbook(4, 5, 6)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
