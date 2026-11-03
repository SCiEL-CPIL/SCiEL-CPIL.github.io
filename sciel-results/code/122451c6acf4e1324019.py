import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    deadline = time.time() + 140  # 10s safety margin

    # Strassen components for 2x2 * 2x2
    # Each component: (U_entries, V_entries, W_entries)
    # Indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),
    ]
    
    def emit_block(a_cs, b_rs, b_cs, c_cs, off, U, V, W, rank):
        for s in range(7):
            col = off + s
            if col >= rank:
                continue
            uc, vc, wc = strassen[s]
            for li, c in uc.items():
                r, cc = li // 2, li % 2
                U[r * m + a_cs + cc][col] += c
            for li, c in vc.items():
                r, cc = li // 2, li % 2
                V[(b_rs + r) * p + b_cs + cc][col] += c
            for li, c in wc.items():
                r, cc = li // 2, li % 2
                W[(c_cs + cc) * n + r][col] += c

    def build_strassen_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        # 4 blocks of 2x2
        # Block (0,0): A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        emit_block(0, 0, 0, 0, 0, U, V, W, rank)
        # Block (0,1): A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        emit_block(0, 0, 2, 2, 7, U, V, W, rank)
        # Block (1,0): A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        emit_block(2, 2, 0, 0, 14, U, V, W, rank)
        # Block (1,1): A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
        emit_block(2, 2, 2, 2, 21, U, V, W, rank)
        return U, V, W

    def verify_decomposition_fast(U, V, W, n, m, p):
        rank = len(U[0])
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        
        U_3d = U_np.reshape(n, m, rank)
        V_3d = V_np.reshape(m, p, rank)
        W_3d = W_np.reshape(p, n, rank)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        if not np.all(T == 1):
            return False
        return True

    U28, V28, W28 = build_strassen_28()

    if not verify_decomposition_fast(U28, V28, W28, n, m, p):
        # Fallback to schoolbook if Strassen construction is buggy (should not happen)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
        return {"U": U, "V": V, "W": W}

    best_U, best_V, best_W = U28, V28, W28
    current_rank = len(best_U[0])
    
    # Attempt to reduce rank via local merges (2-term)
    # This is a heuristic and may not find anything if 28 is optimal.
    # We keep it but with a tight time budget.
    
    def try_merge_two_terms(U, V, W, n, m, p):
        rank = len(U[0])
        nm, mp, pn = n * m, m * p, p * n
        
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)

        for a in range(rank):
            if time.time() > deadline - 5:
                break
            for b in range(a + 1, rank):
                ua = U_np[:, a]
                ub = U_np[:, b]
                va = V_np[:, a]
                vb = V_np[:, b]
                wa = W_np[:, a]
                wb = W_np[:, b]

                T = np.einsum('i,j,k->ijk', ua, va, wa) + np.einsum('i,j,k->ijk', ub, vb, wb)
                
                # Check if T is rank 1
                # Find first non-zero
                i0, j0, k0 = -1, -1, -1
                T000 = 0
                for i in range(nm):
                    if T000 != 0: break
                    for j in range(mp):
                        if T000 != 0: break
                        for k in range(pn):
                            if T[i, j, k] != 0:
                                T000 = T[i, j, k]
                                i0, j0, k0 = i, j, k
                                break
                if T000 == 0:
                    # Sum is zero, can remove both? No, we are merging 2 into 1.
                    # If sum is 0, we can remove both and reduce rank by 2.
                    # But we only reduce by 1 in this function.
                    # We would need a 2-removal logic.
                    continue

                u_vec = T[:, j0, k0].astype(float) / T000
                v_vec = T[i0, :, k0].astype(float) / T000
                w_vec = T[i0, j0, :].astype(float) / T000
                
                T_recon = np.einsum('i,j,k->ijk', u_vec, v_vec, w_vec)
                
                if not np.allclose(T, T_recon, atol=1e-9):
                    continue
                    
                if not (np.allclose(u_vec, np.round(u_vec), atol=1e-9) and
                        np.allclose(v_vec, np.round(v_vec), atol=1e-9) and
                        np.allclose(w_vec, np.round(w_vec), atol=1e-9)):
                    continue
                    
                u_prime = np.round(u_vec).astype(int)
                v_prime = np.round(v_vec).astype(int)
                w_prime = np.round(w_vec).astype(int)

                new_rank = rank - 1
                U_new = [[0] * new_rank for _ in range(nm)]
                V_new = [[0] * new_rank for _ in range(mp)]
                W_new = [[0] * new_rank for _ in range(pn)]

                col_map = [r for r in range(rank) if r != a and r != b]
                for new_c, old_c in enumerate(col_map):
                    for i in range(nm):
                        U_new[i][new_c] = U[i][old_c]
                    for i in range(mp):
                        V_new[i][new_c] = V[i][old_c]
                    for i in range(pn):
                        W_new[i][new_c] = W[i][old_c]

                last_c = new_rank - 1
                for i in range(nm):
                    U_new[i][last_c] = int(u_prime[i])
                for i in range(mp):
                    V_new[i][last_c] = int(v_prime[i])
                for i in range(pn):
                    W_new[i][last_c] = int(w_prime[i])

                if verify_decomposition_fast(U_new, V_new, W_new, n, m, p):
                    return U_new, V_new, W_new

        return None

    if current_rank > 2 and time.time() < deadline - 10:
        merged = try_merge_two_terms(best_U, best_V, best_W, n, m, p)
        if merged is not None:
            best_U, best_V, best_W = merged
            current_rank = len(best_U[0])

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
