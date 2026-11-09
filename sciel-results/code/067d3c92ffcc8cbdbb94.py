"""5x5x5 matrix multiplication tensor decomposition using 4+1 block structure with algebraic cancellation."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

_T0 = time.time()
_LIMIT = 150.0
_MARGIN = 5.0

def _deadline_exceeded():
    return (time.time() - _T0) > (_LIMIT - _MARGIN)

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-reduced exact tensor decomposition for 5x5x5 matrix multiplication.
    
    Uses a 4+1 block partition. 
    The 4x4x4 diagonal block uses recursive Strassen (rank 49).
    Off-diagonal blocks use schoolbook.
    Base total rank: 49 + 61 = 110.
    
    Then applies greedy column removal to find and eliminate linearly dependent columns.
    """
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    columns = []
    
    def add_column(u_entries, v_entries, w_entries):
        u_d = {}
        for idx, val in u_entries:
            u_d[idx] = u_d.get(idx, 0) + val
        v_d = {}
        for idx, val in v_entries:
            v_d[idx] = v_d.get(idx, 0) + val
        w_d = {}
        for idx, val in w_entries:
            w_d[idx] = w_d.get(idx, 0) + val
        # Skip if all zero
        if not u_d or not v_d or not w_d:
            return
        # Check if this column is a scalar multiple of an existing one
        for existing in columns:
            u_ex, v_ex, w_ex = existing
            if u_ex.keys() != u_d.keys() or v_ex.keys() != v_d.keys() or w_ex.keys() != w_d.keys():
                continue
            # Check if same support and scalar multiple
            scale = None
            is_scalar = True
            for k in u_d:
                if u_ex[k] == 0:
                    if u_d[k] != 0:
                        is_scalar = False
                        break
                    continue
                s = u_d[k] / u_ex[k]
                if scale is None:
                    scale = s
                elif abs(s - scale) > 1e-10:
                    is_scalar = False
                    break
            if is_scalar and scale is not None:
                for k in v_d:
                    if v_ex[k] == 0:
                        if v_d[k] != 0:
                            is_scalar = False
                            break
                        continue
                    s = v_d[k] / v_ex[k]
                    if abs(s - scale) > 1e-10:
                        is_scalar = False
                        break
            if is_scalar and scale is not None:
                for k in w_d:
                    if w_ex[k] == 0:
                        if w_d[k] != 0:
                            is_scalar = False
                            break
                        continue
                    s = w_d[k] / w_ex[k]
                    if abs(s - scale) > 1e-10:
                        is_scalar = False
                        break
            if is_scalar and scale is not None and abs(scale) > 1e-10:
                return
        columns.append((u_d, v_d, w_d))
    
    # Block index groups: 4+1 partition
    blocks = [[0, 1, 2, 3], [4]]
    
    def schoolbook_block(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_column(
                        [(i * m + j, 1.0)],
                        [(j * p + k, 1.0)],
                        [(k * n + i, 1.0)]
                    )
                    
    # Standard Strassen 2x2x2 terms
    STRASSEN_2x2_TERMS = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    def strassen_4x4x4(offset_i, offset_j, offset_k):
        I0 = [offset_i + 0, offset_i + 1]
        I1 = [offset_i + 2, offset_i + 3]
        J0 = [offset_j + 0, offset_j + 1]
        J1 = [offset_j + 2, offset_j + 3]
        K0 = [offset_k + 0, offset_k + 1]
        K1 = [offset_k + 2, offset_k + 3]
        
        macro_terms = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        for A_macros, B_macros, C_macros in macro_terms:
            for u_coeffs, v_coeffs, w_coeffs in STRASSEN_2x2_TERMS:
                u_entries = []
                v_entries = []
                w_entries = []
                
                for (li, lj), u_val in u_coeffs.items():
                    for (si, sj), a_val in A_macros.items():
                        total_val = u_val * a_val
                        if total_val != 0:
                            gi = I0[li] if si == 0 else I1[li]
                            gj = J0[lj] if sj == 0 else J1[lj]
                            u_entries.append((gi * m + gj, total_val))
                            
                for (lj, lk), v_val in v_coeffs.items():
                    for (sj, sk), b_val in B_macros.items():
                        total_val = v_val * b_val
                        if total_val != 0:
                            gj = J0[lj] if sj == 0 else J1[lj]
                            gk = K0[lk] if sk == 0 else K1[lk]
                            v_entries.append((gj * p + gk, total_val))
                            
                for (li, lk), w_val in w_coeffs.items():
                    for (si, sk), c_val in C_macros.items():
                        total_val = w_val * c_val
                        if total_val != 0:
                            gi = I0[li] if si == 0 else I1[li]
                            gk = K0[lk] if sk == 0 else K1[lk]
                            w_entries.append((gk * n + gi, total_val))
                            
                add_column(u_entries, v_entries, w_entries)

    # Build all blocks
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                if _deadline_exceeded():
                    break
                i_list = blocks[bi]
                j_list = blocks[bj]
                k_list = blocks[bk]
                
                if len(i_list) == 4 and len(j_list) == 4 and len(k_list) == 4:
                    strassen_4x4x4(i_list[0], j_list[0], k_list[0])
                else:
                    schoolbook_block(i_list, j_list, k_list)
            if _deadline_exceeded():
                break
        if _deadline_exceeded():
            break
    
    rank = len(columns)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c, (u_d, v_d, w_d) in enumerate(columns):
        for idx, val in u_d.items():
            U[idx, c] = val
        for idx, val in v_d.items():
            V[idx, c] = val
        for idx, val in w_d.items():
            W[idx, c] = val
    
    # Compute the reference tensor
    def compute_tensor(U_mat, V_mat, W_mat):
        T = np.zeros((n, m, p))
        for c in range(U_mat.shape[1]):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
        return T
    
    ref_tensor = compute_tensor(U, V, W)
    
    # Greedy column removal
    if not _deadline_exceeded():
        keep = list(range(rank))
        
        # Try removing each column and check if tensor is preserved
        for c in range(rank):
            if _deadline_exceeded():
                break
            if c not in keep:
                continue
            
            # Build matrices without column c
            mask = [i for i in range(rank) if i != c and i in keep]
            if len(mask) == 0:
                continue
                
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            
            T_new = compute_tensor(U_new, V_new, W_new)
            
            if np.allclose(T_new, ref_tensor, atol=1e-10):
                keep.remove(c)
    
    final_rank = len(keep)
    
    U_final = np.zeros((n * m, final_rank))
    V_final = np.zeros((m * p, final_rank))
    W_final = np.zeros((p * n, final_rank))
    
    for new_c, old_c in enumerate(keep):
        U_final[:, new_c] = U[:, old_c]
        V_final[:, new_c] = V[:, old_c]
        W_final[:, new_c] = W[:, old_c]
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
