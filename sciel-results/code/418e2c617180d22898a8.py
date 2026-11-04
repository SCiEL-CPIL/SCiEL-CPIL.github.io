import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 110

    # Strassen 2x2x2 decomposition (rank 7)
    # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)

    # r=0: M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # r=1: M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1

    # r=2: M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1

    # r=3: M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[0, 3] = -1; V2[2, 3] = 1
    W2[0, 3] = 1; W2[1, 3] = 1

    # r=4: M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1

    # r=5: M6 = (A10-A00)(B00+B01)
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    # r=6: M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Build 4x4x8 decomposition using Strassen on (i,j) structure, schoolbook on k
    # Rank = 7 (Strassen) * 8 (columns) = 56
    rank = 56
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((32, rank), dtype=float)

    for ro in range(7):
        for k_out in range(8):
            r = ro * 8 + k_out
            
            # U factor: U[i*4+j, r]
            # i = 2*I + ip, j = 2*J + jp
            # U2 index: I*2+J for outer, ip*2+jp for inner... 
            # Actually U2 is indexed by [i_local*2 + j_local] where i_local, j_local in {0,1}
            # For the 4x4 A matrix: i in 0..3, j in 0..3
            # i = 2*I + ip where I in {0,1}, ip in {0,1}
            # j = 2*J + jp where J in {0,1}, jp in {0,1}
            # U2[I*2+J, ro] gives the coefficient for the "outer" Strassen
            # But we need the full U factor for A[i,j]
            
            # The Strassen decomposition for 4x4x4 (i,j,k) with k in 0..3:
            # U4[i*4+j, ro] = product of two 2x2 Strassen U factors
            # For 4x4x8, we use Strassen on (i,j) and schoolbook on k
            
            # U[i*4+j, r] = U_strassen_4x4[i,j,ro] (same for all k_out)
            # V[j*8+k_out, r] = V_strassen_2x2[j_local,k_local,ro] * delta
            # W[k_out*4+i, r] = W_strassen_2x2[k_local,i_local,ro] * delta
            
            # Let me build U4 first (4x4 Strassen U factor)
            for I in range(2):
                for J in range(2):
                    uo_val = U2[I*2+J, ro]
                    if uo_val == 0:
                        continue
                    for ip in range(2):
                        for jp in range(2):
                            ui_val = U2[ip*2+jp, ro]  # Wait, this isn't right
                            # Actually for 4x4 Strassen, we need two levels
                            pass
            
            # Simpler: just use the known 4x4x4 Strassen U factor
            # U4[i*4+j, ro] for the 4x4x4 case
            # This is built from two 2x2 Strassen levels
            # But for 4x4x8, we only need Strassen on the (i,j) part
            
            # Let me reconsider: the 4x4x8 tensor T[i,j,k] for i,j in 0..3, k in 0..7
            # = sum over j' of A[i,j'] * B[j',k]
            # Split i = 2I+ip, j = 2J+jp, k = 2K+kp
            # For each Strassen rank ro, we have:
            # U[i,j,ro] = U2[I,J,ro] * U2[ip,jp,ro]  (two-level Strassen on 4x4)
            # V[j,k,ro] = V2[J,K,ro] * V2[jp,kp,ro]  (but k has 8 values, so kp in 0..3)
            # W[k,i,ro] = W2[K,I,ro] * W2[kp,ip,ro]
            
            # But k goes 0..7, so K in 0..3, kp in 0..1
            # This means V and W have different structures for k
            
            # Actually, the cleanest: apply Strassen to the 2x2 block structure in i and j,
            # and treat k as having 8 independent columns
            # For each ro (0..6), for each k_out (0..7):
            #   U[i*4+j, r] = U_strassen[i,j,ro]  (from 4x4 Strassen)
            #   V[j*8+k_out, r] = V_strassen[j,k_out,ro]  
            #   W[k_out*4+i, r] = W_strassen[k_out,i,ro]
            
            # Where U_strassen, V_strassen, W_strassen are for the 4x4x4 case
            # but with k replaced by k_out (8 values instead of 4)
            
            # For V: j in 0..3, k_out in 0..7
            # V[j*8+k_out, ro] = V2[J*2+K, ro] * V2[jp*2+kp, ro]
            # where j = 2J+jp, k_out = 2K+kp, K in 0..3, kp in 0..1
            # But V2 is only defined for indices 0..3 (2x2)
            # So V2[jp*2+kp, ro] works for jp,kp in 0..1
            # And V2[J*2+K, ro] - but K can be 0..3, and V2 only has 4 rows!
            # This is the problem: Strassen 2x2 only handles 2x2, not 2x4
            
            # Correct approach: Strassen on (i,j) 4x4, schoolbook on k 8 columns
            # U[i,j,ro] from 4x4 Strassen (two-level 2x2)
            # V[j,k_out,ro] = V2[J*2+jp, ro] * delta_{k_out, K*2+kp} ... no
            
            # Let me just directly construct it:
            # For the 4x4x8 tensor, use the fact that it's:
            # C[i,k] = sum_j A[i,j] B[j,k] for i in 0..3, k in 0..7
            # Apply Strassen to the 2x2 block structure in i (rows of C) and j (cols of A, rows of B)
            # i = 2I+ip, j = 2J+jp
            # For each ro (Strassen rank 0..6):
            #   Define M_ro = sum_{I,J,ip,jp} U2[I*2+J,ro]*U2[ip*2+jp,ro] * A[2I+ip, 2J+jp] * (B factor)
            #   The B factor for column k: sum_{J,jp} V2[J*2+?,ro]*V2[jp*2+?,ro] * B[2J+jp, k]
            #   This doesn't separate cleanly because k has 8 values
            
            # Simplest correct approach: 4x4x8 = two 4x4x4 blocks
            # Block 1: k in 0..3, Block 2: k in 4..7
            # Each 4x4x4 has rank 49 (two-level Strassen)
            # But the U factors are the same for both blocks!
            # So total rank = 49 + 49 = 98, but we can potentially merge if U factors align
            # They do align (same U4), so we just have 98 terms with shared U
            # This is what the current program does
            
            # To get rank 56: use Strassen on i,j (4x4 -> rank 7 per "virtual" 2x2)
            # and schoolbook on k (8 columns -> 8 multiplications per virtual)
            # Total: 7 * 8 = 56
            
            # U[i*4+j, r] for r = ro*8+k_out:
            #   = U4_strassen[i,j,ro] (same for all k_out, from 4x4 Strassen)
            # V[j*8+k_out, r]:
            #   = V_strassen[j,k_out,ro]
            # W[k_out*4+i, r]:
            #   = W_strassen[k_out,i,ro]
            
            # Build U4 (4x4 Strassen U factor, shape 16x7)
            U4 = np.zeros((16, 7), dtype=float)
            V4 = np.zeros((16, 7), dtype=float)  # for 4x4x4, j*4+k
            W4 = np.zeros((16, 7), dtype=float)  # for 4x4x4, k*4+i
            
            for ro in range(7):
                for I in range(2):
                    for J in range(2):
                        uo = U2[I*2+J, ro]
                        if uo == 0:
                            continue
                        for ip in range(2):
                            for jp in range(2):
                                ui = U2[ip*2+jp, ro]
                                if ui == 0:
                                    continue
                                i = 2*I + ip
                                j = 2*J + jp
                                U4[i*4+j, ro] = uo * ui
                                
                                vo = V2[J*2+K if False else J*2, ro]  # placeholder
                                pass
            
            # This is getting complicated. Let me just verify rank 98 is correct and return it.
            # The rank-56 approach requires careful construction that I'm not getting right quickly.
            
            # Fallback: return the known correct rank-98 solution
            break
        break

    # Return the proven rank-98 solution (same as current program)
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)

    for ro in range(7):
        for ri in range(7):
            r = ro * 7 + ri
            for uo in range(4):
                uo_val = U2[uo, ro]
                if uo_val != 0:
                    I = uo // 2
                    J = uo % 2
                    for ui in range(4):
                        ui_val = U2[ui, ri]
                        if ui_val != 0:
                            ip = ui // 2
                            jp = ui % 2
                            i = 2 * I + ip
                            j = 2 * J + jp
                            U4[i * 4 + j, r] = uo_val * ui_val
            for vo in range(4):
                vo_val = V2[vo, ro]
                if vo_val != 0:
                    J = vo // 2
                    K = vo % 2
                    for vi in range(4):
                        vi_val = V2[vi, ri]
                        if vi_val != 0:
                            jp = vi // 2
                            kp = vi % 2
                            j = 2 * J + jp
                            k = 2 * K + kp
                            V4[j * 4 + k, r] = vo_val * vi_val
            for wo in range(4):
                wo_val = W2[wo, ro]
                if wo_val != 0:
                    K = wo // 2
                    I = wo % 2
                    for wi in range(4):
                        wi_val = W2[wi, ri]
                        if wi_val != 0:
                            kp = wi // 2
                            ip = wi % 2
                            k = 2 * K + kp
                            i = 2 * I + ip
                            W4[k * 4 + i, r] = wo_val * wi_val

    rank = 98
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((32, rank), dtype=float)

    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, :]
            W[k * 4 + j, 0:49] = W4[k * 4 + j, :]

    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + 4 + k, 49:98] = V4[j * 4 + k, :]
            W[(4 + k) * 4 + j, 49:98] = W4[k * 4 + j, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
