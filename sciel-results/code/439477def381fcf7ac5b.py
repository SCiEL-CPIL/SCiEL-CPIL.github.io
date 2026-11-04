"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Build the matrix multiplication tensor T[i,j,k] for C[i,k] = sum_j A[i,j]*B[j,k]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Convention: W[k*n + i, r] contributes to C[i,k]
    # So T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # First, build the rank-42 Strassen decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    def add_strassen_block(a_idx, b_idx, w_idx, start_col):
        """
        a_idx: [a00, a01, a10, a11] indices into U
        b_idx: [b00, b01, b10, b11] indices into V
        w_idx: [w00, w01, w10, w11] indices into W for the 2x2 output block
        """
        # Strassen:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        c = start_col
        # M1
        U[a_idx[0]][c] += 1; U[a_idx[3]][c] += 1
        V[b_idx[0]][c] += 1; V[b_idx[3]][c] += 1
        W[w_idx[0]][c] += 1; W[w_idx[3]][c] += 1
        # M2
        U[a_idx[2]][c+1] += 1; U[a_idx[3]][c+1] += 1
        V[b_idx[0]][c+1] += 1
        W[w_idx[2]][c+1] += 1; W[w_idx[3]][c+1] -= 1
        # M3
        U[a_idx[0]][c+2] += 1
        V[b_idx[1]][c+2] += 1; V[b_idx[3]][c+2] -= 1
        W[w_idx[1]][c+2] += 1; W[w_idx[3]][c+2] += 1
        # M4
        U[a_idx[3]][c+3] += 1
        V[b_idx[2]][c+3] += 1; V[b_idx[0]][c+3] -= 1
        W[w_idx[0]][c+3] += 1; W[w_idx[2]][c+3] += 1
        # M5
        U[a_idx[0]][c+4] += 1; U[a_idx[1]][c+4] += 1
        V[b_idx[3]][c+4] += 1
        W[w_idx[0]][c+4] -= 1; W[w_idx[1]][c+4] += 1
        # M6
        U[a_idx[2]][c+5] += 1; U[a_idx[0]][c+5] -= 1
        V[b_idx[0]][c+5] += 1; V[b_idx[1]][c+5] += 1
        W[w_idx[3]][c+5] += 1
        # M7
        U[a_idx[1]][c+6] += 1; U[a_idx[3]][c+6] -= 1
        V[b_idx[2]][c+6] += 1; V[b_idx[3]][c+6] += 1
        W[w_idx[0]][c+6] += 1
    
    # A is 2x4: indices i*m+j
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    a_L = [0, 1, 4, 5]  # a00, a01, a10, a11
    a_R = [2, 3, 6, 7]
    
    # B is 4x6: indices j*p+k
    # For each pair of rows (r1,r2) and pair of cols (c1,c2):
    # b00=B[r1,c1], b01=B[r1,c2], b10=B[r2,c1], b11=B[r2,c2]
    
    # Block structure: split A into left 2 cols and right 2 cols
    # Split B into 3 blocks of 2 cols each
    # For each A block and B col-block, we have a 2x2 * 2x2 multiplication
    
    # B col block 0: cols 0,1
    #   Rows 0,1: b00=B[0,0]=0, b01=B[0,1]=1, b10=B[1,0]=6, b11=B[1,1]=7
    #   Rows 2,3: b00=B[2,0]=12, b01=B[2,1]=13, b10=B[3,0]=18, b11=B[3,1]=19
    b_1_T = [0, 1, 6, 7]
    b_1_B = [12, 13, 18, 19]
    
    # W indices for C cols 0,1: C[i,k] -> W[k*n+i]
    # C[0,0]->W[0], C[0,1]->W[2], C[1,0]->W[1], C[1,1]->W[3]
    w_1 = [0, 2, 1, 3]
    
    add_strassen_block(a_L, b_1_T, w_1, 0)
    add_strassen_block(a_R, b_1_B, w_1, 7)
    
    # B col block 1: cols 2,3
    b_2_T = [2, 3, 8, 9]
    b_2_B = [14, 15, 20, 21]
    w_2 = [4, 6, 5, 7]
    
    add_strassen_block(a_L, b_2_T, w_2, 14)
    add_strassen_block(a_R, b_2_B, w_2, 21)
    
    # B col block 2: cols 4,5
    b_3_T = [4, 5, 10, 11]
    b_3_B = [16, 17, 22, 23]
    w_3 = [8, 10, 9, 11]
    
    add_strassen_block(a_L, b_3_T, w_3, 28)
    add_strassen_block(a_R, b_3_B, w_3, 35)
    
    # Verify the decomposition
    def verify_decomp(U, V, W, rank):
        """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = delta for M_{n,m,p}"""
        # For each (i,j,k), the tensor value should be 1 if this is a valid MM tensor
        # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # Actually T[i,j,k] = 1 for the MM tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True
    
    if not verify_decomp(U, V, W, rank):
        # Fallback: return schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by finding linear dependencies among the 42 rank-1 terms
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # If sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0 with c_{max} != 0, then:
    # u_max ⊗ v_max ⊗ w_max = -sum_{c != max} (c_c/c_max) * (u_c ⊗ v_c ⊗ w_c)
    # This means we can eliminate term max IF u_max, v_max, w_max are each in the
    # span of the remaining {u_c}, {v_c}, {w_c} respectively.
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the matrix of flattened rank-1 tensors: M[:,r] = vec(u_r ⊗ v_r ⊗ w_r)
    flat_terms = []
    for r in range(rank):
        term = np.outer(np.outer(U_np[:, r], V_np[:, r]), W_np[:, r])
        flat_terms.append(term.flatten())
    M = np.array(flat_terms).T  # shape (2304, 42)
    
    best_result = None
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(M, full_matrices=False)
        tol = 1e-9
        effective_rank = int(np.sum(s_vals > tol))
        
        if effective_rank < rank:
            null_space = Vt_svd[effective_rank:, :]  # shape ((rank-eff_rank), rank)
            
            for null_vec in null_space:
                if time.time() > deadline - 15:
                    break
                # Find the column with the largest |coefficient| to eliminate
                candidates = list(range(rank))
                if not candidates:
                    break
                max_idx = max(candidates, key=lambda c: abs(null_vec[c]))
                max_val = abs(null_vec[max_idx])
                if max_val < 1e-8:
                    continue
                
                keep = [c for c in range(rank) if c != max_idx]
                n_keep = len(keep)
                
                # Check if u_max is in span of {u_c : c in keep}
                U_keep = U_np[:, keep]  # (8, n_keep)
                try:
                    U_coeffs, _, _, _ = np.linalg.lstsq(U_keep, U_np[:, max_idx], rcond=1e-10)
                    U_residual = np.linalg.norm(U_keep @ U_coeffs - U_np[:, max_idx])
                except Exception:
                    U_residual = 1e10
                
                # Check if v_max is in span of {v_c : c in keep}
                V_keep = V_np[:, keep]  # (24, n_keep)
                try:
                    V_coeffs, _, _, _ = np.linalg.lstsq(V_keep, V_np[:, max_idx], rcond=1e-10)
                    V_residual = np.linalg.norm(V_keep @ V_coeffs - V_np[:, max_idx])
                except Exception:
                    V_residual = 1e10
                
                # Check if w_max is in span of {w_c : c in keep}
                W_keep = W_np[:, keep]  # (12, n_keep)
                try:
                    W_coeffs, _, _, _ = np.linalg.lstsq(W_keep, W_np[:, max_idx], rcond=1e-10)
                    W_residual = np.linalg.norm(W_keep @ W_coeffs - W_np[:, max_idx])
                except Exception:
                    W_residual = 1e10
                
                if U_residual < 1e-8 and V_residual < 1e-8 and W_residual < 1e-8:
                    # We can eliminate max_idx!
                    # The new decomposition: sum_{c in keep} u'_c ⊗ v'_c ⊗ w'_c = T
                    # where for each c in keep:
                    # u'_c = u_c + (c_c/c_max) * u_max
                    # v'_c = v_c + (c_c/c_max) * v_max  
                    # w'_c = w_c + (c_c/c_max) * w_max
                    # BUT WAIT: this doesn't work because tensor product isn't linear in all three simultaneously.
                    # 
                    # Actually, the correct elimination: if sum c_r (u_r⊗v_r⊗w_r) = 0,
                    # then the remaining sum already equals T. We just drop the max_idx term.
                    # But we need to verify: sum_{c in keep} c_c (u_c⊗v_c⊗w_c) = T?
                    # No! The original decomposition is sum_{r=0}^{rank-1} 1*(u_r⊗v_r⊗w_r) = T
                    # The null space tells us sum_r c_r (u_r⊗v_r⊗w_r) = 0
                    # So we can replace the set {u_r, v_r, w_r}_{r=0}^{rank-1} with
                    # {u_r + c_r/c_max * u_max, v_r + c_r/c_max * v_max, w_r + c_r/c_max * w_max}
                    # NO - that's wrong too.
                    #
                    # The correct way: if sum_r c_r T_r = 0 where T_r = u_r⊗v_r⊗w_r,
                    # then {T_r}_{r in keep} and {T_r}_{r in keep ∪ {max}} span the same space.
                    # We can express T_max = -sum_{c in keep} (c_c/c_max) T_c.
                    # But we can't simply modify the factors.
                    #
                    # HOWEVER: if u_max ∈ span(U_keep), v_max ∈ span(V_keep), w_max ∈ span(W_keep),
                    # then T_max ∈ span of {u_c⊗v_c⊗w_c : c in keep}? NO, that's not true in general.
                    #
                    # The correct approach: we can only eliminate if the tensor T_max is a linear
                    # combination of the remaining tensors. Let's verify directly.
                    
                    # Verify: does sum_{c in keep} (u_c⊗v_c⊗w_c) still equal T?
                    # Original: sum_{all r} T_r = T
                    # Null: sum_r c_r T_r = 0, so T_max = -sum_{c in keep} (c_c/c_max) T_c
                    # Therefore: sum_{c in keep} T_c + T_max = sum_{c in keep} T_c - sum_{c in keep} (c_c/c_max) T_c
                    # = sum_{c in keep} (1 - c_c/c_max) T_c... NO this is wrong.
                    #
                    # Let me re-derive. We have:
                    # (1) sum_{r=0}^{41} T_r = T  (the MM tensor)
                    # (2) sum_{r=0}^{41} c_r T_r = 0  (null space relation)
                    #
                    # From (2): c_max * T_max = -sum_{c in keep} c_c * T_c
                    # So: T_max = -sum_{c in keep} (c_c/c_max) T_c
                    #
                    # From (1): T = sum_{c in keep} T_c + T_max
                    #         = sum_{c in keep} T_c - sum_{c in keep} (c_c/c_max) T_c
                    #         = sum_{c in keep} (1 - c_c/c_max) T_c
                    #
                    # So the new decomposition has factors:
                    # u'_c = u_c, v'_c = v_c, w'_c = w_c for c in keep
                    # but with coefficients (1 - c_c/c_max) instead of 1.
                    #
                    # We can absorb the coefficient into any one of the three factors:
                    # u'_c = (1 - c_c/c_max) * u_c, v'_c = v_c, w'_c = w_c
                    # or split it: u'_c = sqrt(1-c_c/c_max) * u_c, etc.
                    # But we need integer/half-integer entries!
                    
                    new_U = U_np[:, keep].copy()
                    new_V = V_np[:, keep].copy()
                    new_W = W_np[:, keep].copy()
                    
                    for idx, c in enumerate(keep):
                        coeff = 1.0 - null_vec[c] / null_vec[max_idx]
                        new_U[:, idx] *= coeff
                    
                    # Verify the new decomposition
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                total = 0.0
                                for idx in range(n_keep):
                                    total += new_U[i*m+j, idx] * new_V[j*p+k, idx] * new_W[k*n+i, idx]
                                if abs(total - 1.0) > 1e-8:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        best_result = (new_U, new_V, new_W, n_keep)
                        break  # Found a valid elimination, use it
    except Exception:
        pass
    
    if best_result is not None:
        final_U, final_V, final_W, new_rank = best_result
        
        def to_fraction_list(arr):
            result = []
            for row in arr:
                result.append([])
                for val in row:
                    # Try to find the closest fraction with small denominator
                    best_frac = round(val * 2) / 2
                    for denom in range(1, 11):
                        num = round(val * denom)
                        if abs(val * denom - num) < 1e-9:
                            best_frac = num / denom
                            break
                    if abs(best_frac - round(best_frac)) < 1e-10:
                        best_frac = int(round(best_frac))
                    result[-1].append(best_frac)
            return result
        
        return {
            "U": to_fraction_list(final_U),
            "V": to_fraction_list(final_V),
            "W": to_fraction_list(final_W)
        }
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
