import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    deadline = time.time() + 140
    
    n, m, p = DIMENSIONS
    nb = 3
    bs = 2
    
    # Strassen coefficients for 2x2
    # M_k = (sum a_r A_r) (sum b_c B_c) -> C contributions
    # We need U (A coeffs), V (B coeffs), W (C coeffs)
    
    # Indexing: U[i*m+j, k], V[j*p+k, r], W[k*n+i, r]
    # For each rank-1 term r:
    # U_r is a 6x6 matrix (flattened), V_r is 6x6, W_r is 6x6
    # T = sum_r U_r \otimes V_r \otimes W_r (appropriately permuted)
    
    # We use the block structure.
    # There are 27 blocks. Each block uses 7 Strassen terms.
    # Total 189 terms.
    
    strassen_terms = []
    # Each term is (A_mask, B_mask, C_mask, coeff)
    # A_mask is 2x2, etc.
    
    # We will generate the matrices directly.
    
    rank_est = nb * nb * nb * 7
    U = np.zeros((n*m, rank_est), dtype=np.int64)
    V = np.zeros((m*p, rank_est), dtype=np.int64)
    W = np.zeros((p*n, rank_est), dtype=np.int64)
    
    col = 0
    for ib in range(nb):
        for jb in range(nb):
            for kb in range(nb):
                # Generate 7 Strassen terms for this block
                # Strassen for 2x2:
                # M0 = (A00+A11)(B00+B11)
                # M1 = (A10+A11)B00
                # M2 = A00(B01-B11)
                # M3 = A11(B10-B00)
                # M4 = (A00+A01)B11
                # M5 = (A10-A00)(B00+B01)
                # M6 = (A01-A11)(B10+B11)
                
                # C00 = M0 - M4 + M6
                # C01 = M2 + M4
                # C10 = M1 + M3
                # C11 = M0 - M1 + M2 + M5
                
                # We need to map these to U, V, W columns.
                
                # Term 0: (A00+A11) * (B00+B11) -> M0
                U[ib*bs*2 + 0*2 + 0, col] += 1 # A00
                U[ib*bs*2 + 1*2 + 1, col] += 1 # A11
                V[jb*bs*2 + 0*2 + 0, col] += 1 # B00
                V[jb*bs*2 + 1*2 + 1, col] += 1 # B11
                W[kb*bs*2 + 0*2 + 0, col] += 1 # C00
                W[kb*bs*2 + 1*2 + 1, col] += 1 # C11
                col += 1
                
                # Term 1: (A10+A11) * B00 -> M1
                U[ib*bs*2 + 1*2 + 0, col] += 1 # A10
                U[ib*bs*2 + 1*2 + 1, col] += 1 # A11
                V[jb*bs*2 + 0*2 + 0, col] += 1 # B00
                W[kb*bs*2 + 1*2 + 0, col] += 1 # C10
                W[kb*bs*2 + 1*2 + 1, col] -= 1 # C11
                col += 1
                
                # Term 2: A00 * (B01-B11) -> M2
                U[ib*bs*2 + 0*2 + 0, col] += 1 # A00
                V[jb*bs*2 + 0*2 + 1, col] += 1 # B01
                V[jb*bs*2 + 1*2 + 1, col] -= 1 # B11
                W[kb*bs*2 + 0*2 + 1, col] += 1 # C01
                W[kb*bs*2 + 1*2 + 1, col] += 1 # C11
                col += 1
                
                # Term 3: A11 * (B10-B00) -> M3
                U[ib*bs*2 + 1*2 + 1, col] += 1 # A11
                V[jb*bs*2 + 1*2 + 0, col] += 1 # B10
                V[jb*bs*2 + 0*2 + 0, col] -= 1 # B00
                W[kb*bs*2 + 0*2 + 0, col] += 1 # C00
                W[kb*bs*2 + 1*2 + 0, col] += 1 # C10
                col += 1
                
                # Term 4: (A00+A01) * B11 -> M4
                U[ib*bs*2 + 0*2 + 0, col] += 1 # A00
                U[ib*bs*2 + 0*2 + 1, col] += 1 # A01
                V[jb*bs*2 + 1*2 + 1, col] += 1 # B11
                W[kb*bs*2 + 0*2 + 0, col] -= 1 # C00
                W[kb*bs*2 + 0*2 + 1, col] += 1 # C01
                col += 1
                
                # Term 5: (A10-A00) * (B00+B01) -> M5
                U[ib*bs*2 + 1*2 + 0, col] += 1 # A10
                U[ib*bs*2 + 0*2 + 0, col] -= 1 # A00
                V[jb*bs*2 + 0*2 + 0, col] += 1 # B00
                V[jb*bs*2 + 0*2 + 1, col] += 1 # B01
                W[kb*bs*2 + 1*2 + 1, col] += 1 # C11
                col += 1
                
                # Term 6: (A01-A11) * (B10+B11) -> M6
                U[ib*bs*2 + 0*2 + 1, col] += 1 # A01
                U[ib*bs*2 + 1*2 + 1, col] -= 1 # A11
                V[jb*bs*2 + 1*2 + 0, col] += 1 # B10
                V[jb*bs*2 + 1*2 + 1, col] += 1 # B11
                W[kb*bs*2 + 0*2 + 0, col] += 1 # C00
                col += 1
    
    # Verify
    def fast_verify(U_mat, V_mat, W_mat):
        rank_val = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for r in range(rank_val):
                        u_val = U_mat[u_idx, r]
                        if u_val != 0:
                            v_val = V_mat[v_idx, r]
                            if v_val != 0:
                                w_val = W_mat[w_idx, r]
                                if w_val != 0:
                                    total += u_val * v_val * w_val
                    if total != 1:
                        return False
        return True

    if not fast_verify(U, V, W):
        # Fallback
        rank_school = n * m * p
        U_s = np.zeros((n*m, rank_school), dtype=np.int64)
        V_s = np.zeros((m*p, rank_school), dtype=np.int64)
        W_s = np.zeros((p*n, rank_school), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m + j, col] = 1
                    V_s[j*p + k, col] = 1
                    W_s[k*n + i, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    # Rank Reduction
    # Loop until no more merges or time out
    current_U, current_V, current_W = U, V, W
    while time.time() < deadline - 5:
        merged_any = False
        r = current_U.shape[1]
        used = [True] * r
        for i in range(r):
            if not used[i]:
                continue
            for j in range(i+1, r):
                if not used[j]:
                    continue
                if time.time() > deadline - 5:
                    break
                
                # Check if col j is proportional to col i
                # Find ratio ru, rv such that U_j = ru * U_i, V_j = rv * V_i
                ru = None
                rv = None
                valid = True
                
                # Check U
                for idx in range(n*m):
                    ui = current_U[idx, i]
                    uj = current_U[idx, j]
                    if ui == 0 and uj == 0:
                        continue
                    if ui == 0 or uj == 0:
                        valid = False
                        break
                    if uj % ui != 0:
                        valid = False
                        break
                    cand = uj // ui
                    if ru is None:
                        ru = cand
                    elif ru != cand:
                        valid = False
                        break
                if not valid:
                    continue
                if ru is None:
                    # All zero?
                    if np.any(current_U[:, i] != 0) or np.any(current_U[:, j] != 0):
                        continue
                    ru = 1 # Arbitrary, but if all zero, contribution is 0?
                    # If U is all zero, the term is 0. We can just drop it?
                    # But we must maintain the sum.
                    # If U_i is all 0, the term contributes 0 to T.
                    # So we can just remove it?
                    # Yes, if a rank-1 term is zero, it can be removed.
                    # But we are merging.
                    # If U_i is 0, we can't merge into i.
                    # Let's skip merging if vectors are zero.
                    continue

                # Check V
                for idx in range(m*p):
                    vi = current_V[idx, i]
                    vj = current_V[idx, j]
                    if vi == 0 and vj == 0:
                        continue
                    if vi == 0 or vj == 0:
                        valid = False
                        break
                    if vj % vi != 0:
                        valid = False
                        break
                    cand = vj // vi
                    if rv is None:
                        rv = cand
                    elif rv != cand:
                        valid = False
                        break
                if not valid:
                    continue
                if rv is None:
                    continue

                # Merge j into i
                # W_i += ru * rv * W_j
                current_W[:, i] += ru * rv * current_W[:, j]
                used[j] = False
                merged_any = True
        
        if not merged_any:
            break
            
        # Compact matrices
        new_rank = sum(used)
        if new_rank < r:
            idxs = [i for i in range(r) if used[i]]
            current_U = current_U[:, idxs]
            current_V = current_V[:, idxs]
            current_W = current_W[:, idxs]
            
            # Verify after each round of merges
            if not fast_verify(current_U, current_V, current_W):
                # This shouldn't happen if logic is correct, but safe
                break

    return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}
