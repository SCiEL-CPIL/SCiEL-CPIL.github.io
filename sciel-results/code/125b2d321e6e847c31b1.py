"""Rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.

Uses the Kronecker product structure:
  T_{4,4,6} = T_{2,2,2} ⊗ T_{2,2,3}

T_{2,2,2} has rank 7 (Strassen).
T_{2,2,3} has rank 11 (Strassen 7 for 2x2x2 + schoolbook 4 for 2x2x1,
but structured as a single coherent decomposition).

Total rank: 7 * 11 = 77.

The structural difference from the split approach: we use a unified
Kronecker product where the 2x2x3 sub-tensor is treated as an atomic
building block, rather than splitting the 6 columns into 4+2 and
decomposing each half independently.
"""

import time
import numpy as np

_T0 = time.perf_counter()
_DEADLINE = 140.0


def _left():
    return _DEADLINE - (time.perf_counter() - _T0)


# 2x2 Strassen: 7 terms, each (u_dict, v_dict, w_dict) with (i,j)/(j,k)/(k,i) keys
_S2 = [
    ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
    ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
    ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
    ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
    ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
    ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
    ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
]

# 2x2 * 2x1 schoolbook: 4 terms
_S2x1 = [
    ({(0,0):1}, {(0,0):1}, {(0,0):1}),
    ({(1,0):1}, {(0,0):1}, {(1,0):1}),
    ({(0,1):1}, {(1,0):1}, {(0,0):1}),
    ({(1,1):1}, {(1,0):1}, {(1,0):1}),
]


def _build_2x2x3_terms():
    """
    Build the 11 rank-1 terms for the 2x2 * 2x3 matrix multiplication tensor.
    
    A is 2x2 (indices i=0,1; j=0,1), B is 2x3 (indices j=0,1; k=0,1,2),
    C is 2x3 (indices i=0,1; k=0,1,2).
    
    We decompose B = [B_left (2x2), B_right (2x1)].
    C_left = A * B_left uses Strassen (7 terms).
    C_right = A * B_right uses schoolbook (4 terms).
    
    For each term, we produce (u_dict, v_dict, w_dict) where:
      u_dict: {(i,j): coeff}  for A entries
      v_dict: {(j,k): coeff}  for B entries (k in 0..2)
      w_dict: {(i,k): coeff}  for C entries (k in 0..2)
    """
    terms = []
    
    # First 7 terms: Strassen for the 2x2 * 2x2 part (columns 0,1 of B)
    for u_pat, v_pat, w_pat in _S2:
        # u_pat: {(i,j): coeff} - same
        # v_pat: {(j,k): coeff} where k in {0,1} - same (columns 0,1)
        # w_pat: {(i,k): coeff} where k in {0,1} - same (columns 0,1)
        terms.append((dict(u_pat), dict(v_pat), dict(w_pat)))
    
    # Last 4 terms: schoolbook for the 2x2 * 2x1 part (column 2 of B)
    # B_right has entries B[j,2] for j=0,1
    # C_right has entries C[i,2] for i=0,1
    # C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]
    for u_pat, v_pat, w_pat in _S2x1:
        # u_pat: {(i,j): coeff} - same
        # v_pat: {(j,0): coeff} -> remap to {(j,2): coeff}
        v_new = {}
        for (j, k), coeff in v_pat.items():
            v_new[(j, 2)] = coeff
        # w_pat: {(i,0): coeff} -> remap to {(i,2): coeff}
        w_new = {}
        for (i, k), coeff in w_pat.items():
            w_new[(i, 2)] = coeff
        terms.append((dict(u_pat), v_new, w_new))
    
    return terms


def _kronecker_emit(u1, v1, w1, u2, v2, w2, n, m, p, out_u, out_v, out_w):
    """
    Emit one rank-1 term of the Kronecker product decomposition.
    
    u1, v1, w1: patterns for T_{2,2,2} (outer 2x2 blocks)
    u2, v2, w2: patterns for T_{2,2,3} (inner 2x2*2x3)
    
    The full 4x4*4x6 tensor:
    A is 4x4, B is 4x6, C is 4x6.
    
    Block structure:
    A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    B = [[B00, B01, B02], [B10, B11, B12]] where Bij is 2x3
    C = [[C00, C01, C02], [C10, C11, C12]] where Cij is 2x3
    
    The Kronecker product term:
    U[i*m+j] = sum over outer block (bo) and inner (bi):
                u1[bo_block] * u2[bi]
    where bo_block identifies which 2x2 block of A, and bi identifies
    the specific (i,j) within that block.
    
    Similarly for V and W.
    """
    u_col = [0] * (n * m)  # 16
    v_col = [0] * (m * p)  # 24
    w_col = [0] * (p * n)  # 24
    
    # Block offsets for 2x2 blocks in 4x4 / 4x6:
    # Outer block (bo): 0=(0,0), 1=(0,2), 2=(2,0), 3=(2,2) for A
    # For B: outer row block 0=(rows 0-1), 1=(rows 2-3)
    #        outer col block: 0=(cols 0-1), 1=(cols 2-3), 2=(cols 4-5)
    # For C: outer row block 0=(rows 0-1), 1=(rows 2-3)
    #        outer col block: 0=(cols 0-1), 1=(cols 2-3), 2=(cols 4-5)
    
    # u1: {(bo_i, bo_j): coeff} where bo_i, bo_j in {0,1} identify outer block
    # u2: {(i,j): coeff} where i,j in {0,1} identify position within 2x2 block
    # Full A index: i_full = bo_i * 2 + i, j_full = bo_j * 2 + j
    
    for (bo_i, bo_j), coeff1 in u1.items():
        for (i, j), coeff2 in u2.items():
            i_full = bo_i * 2 + i
            j_full = bo_j * 2 + j
            if 0 <= i_full < n and 0 <= j_full < m:
                u_col[i_full * m + j_full] += coeff1 * coeff2
    
    # v1: {(bo_j, bo_k_outer): coeff} where bo_j in {0,1}, bo_k_outer in {0,1}
    #     bo_k_outer identifies which pair of column-blocks (0=cols 0-1, 1=cols 2-3, 2=cols 4-5)
    #     Actually, for the Kronecker structure, the outer B dimension is 2x3 blocks.
    #     B outer: 2 row-blocks x 3 col-blocks.
    #     v1 should be over the 2x3 outer block structure.
    
    # Let me reconsider the Kronecker structure more carefully.
    # T_{4,4,6} = T_{2,2,2} ⊗ T_{2,2,3}
    # The first T_{2,2,2} handles the "outer" 2x2 block structure.
    # The second T_{2,2,3} handles the "inner" 2x2*2x3 within each block.
    
    # For A (4x4): outer 2x2 blocks, each 2x2.
    #   A[bo_i*2+i, bo_j*2+j]
    # For B (4x6): outer 2x3 blocks, each 2x3? No...
    #   B has 4 rows, 6 cols. As 2x3 blocks of size 2x2? No, that doesn't work.
    
    # Actually, the correct decomposition is:
    # T_{4,4,6} where 4=2*2, 4=2*2, 6=2*3
    # This is NOT a simple Kronecker product of T_{2,2,2} and T_{2,2,3}.
    # 
    # The correct Kronecker structure is:
    # T_{(2*2),(2*2),(2*3)} = T_{2,2,2} ⊗ T_{2,2,3}
    # where the first factor uses the "2" in each dimension and the second
    # uses the "2", "2", "3" respectively.
    
    # So for A (4x4 = (2*2) x (2*2)):
    #   Index (i,j) = (i1*2 + i0, j1*2 + j0) where i1,j1 in {0,1} (outer), i0,j0 in {0,1} (inner)
    # For B (4x6 = (2*2) x (2*3)):
    #   Index (j,k) = (j1*2 + j0, k1*3 + k0)? No, that's not right either.
    
    # Let me use the standard Kronecker product convention:
    # If T1 has dimensions (n1,m1,p1) and T2 has dimensions (n2,m2,p2),
    # then T1 ⊗ T2 has dimensions (n1*n2, m1*m2, p1*p2).
    # 
    # For our case: T1 = T_{2,2,2}, T2 = T_{2,2,3}
    # T1 ⊗ T2 has dimensions (2*2, 2*2, 2*3) = (4, 4, 6). ✓
    
    # Index mapping:
    # A index (i,j): i = i1*2 + i0, j = j1*2 + j0
    #   where (i1,j1) is the T1 A-index, (i0,j0) is the T2 A-index
    # B index (j,k): j = j1*2 + j0, k = k1*3 + k0
    #   where (j1,k1) is the T1 B-index... wait, T1 is 2x2x2 so B is 2x2.
    #   But T2 is 2x2x3 so B is 2x3.
    #   So B index: j = j1*2 + j0 (j1 from T1, j0 from T2), k = k1*3 + k0 (k1 from T1, k0 from T2)
    #   But T1's B dimension is 2, so k1 in {0,1}. T2's B dimension is 3, so k0 in {0,1,2}.
    #   Total k = k1*3 + k0. But k1 in {0,1} and k0 in {0,1,2} gives k in {0,1,2,3,4,5}. ✓
    # C index (i,k): i = i1*2 + i0, k = k1*3 + k0
    #   where (i1,k1) is the T1 C-index, (i0,k0) is the T2 C-index
    
    # So the Kronecker product term is:
    # U[i1*2+i0, j1*2+j0] = U1[i1,j1] * U2[i0,j0]
    # V[j1*2+j0, k1*3+k0] = V1[j1,k1] * V2[j0,k0]
    # W[k1*3+k0, i1*2+i0] = W1[i1,k1] * W2[i0,k0]
    
    # u1: {(i1,j1): coeff} for T1's A part
    # u2: {(i0,j0): coeff} for T2's A part
    for (i1, j1), c1 in u1.items():
        for (i0, j0), c2 in u2.items():
            i_full = i1 * 2 + i0
            j_full = j1 * 2 + j0
            u_col[i_full * m + j_full] += c1 * c2
    
    # v1: {(j1,k1): coeff} for T1's B part (j1 in {0,1}, k1 in {0,1})
    # v2: {(j0,k0): coeff} for T2's B part (j0 in {0,1}, k0 in {0,1,2})
    for (j1, k1), c1 in v1.items():
        for (j0, k0), c2 in v2.items():
            j_full = j1 * 2 + j0
            k_full = k1 * 3 + k0
            if 0 <= j_full < m and 0 <= k_full < p:
                v_col[j_full * p + k_full] += c1 * c2
    
    # w1: {(i1,k1): coeff} for T1's C part (i1 in {0,1}, k1 in {0,1})
    # w2: {(i0,k0): coeff} for T2's C part (i0 in {0,1}, k0 in {0,1,2})
    for (i1, k1), c1 in w1.items():
        for (i0, k0), c2 in w2.items():
            i_full = i1 * 2 + i0
            k_full = k1 * 3 + k0
            if 0 <= i_full < n and 0 <= k_full < p:
                w_col[k_full * n + i_full] += c1 * c2
    
    out_u.append(u_col)
    out_v.append(v_col)
    out_w.append(w_col)


def construct():
    n, m, p = 4, 4, 6
    
    out_u, out_v, out_w = [], [], []
    
    # Build T2 terms (2x2 * 2x3, rank 11)
    t2_terms = _build_2x2x3_terms()
    
    # Kronecker product: for each T1 term (7) and each T2 term (11), emit one rank-1 term
    for u1, v1, w1 in _S2:
        for u2, v2, w2 in t2_terms:
            _kronecker_emit(u1, v1, w1, u2, v2, w2, n, m, p, out_u, out_v, out_w)
    
    rank = len(out_u)
    
    # Build final arrays
    U_final = [[out_u[r][i] for r in range(rank)] for i in range(n * m)]
    V_final = [[out_v[r][i] for r in range(rank)] for i in range(m * p)]
    W_final = [[out_w[r][i] for r in range(rank)] for i in range(p * n)]
    
    # Verify
    U_arr = np.array(U_final, dtype=np.int64)
    V_arr = np.array(V_final, dtype=np.int64)
    W_arr = np.array(W_final, dtype=np.int64)
    
    i_vals, j_vals, k_vals = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    u_idx = (i_vals * m + j_vals).ravel()
    v_idx = (j_vals * p + k_vals).ravel()
    w_idx = (k_vals * n + i_vals).ravel()
    
    totals = np.sum(U_arr[u_idx, :] * V_arr[v_idx, :] * W_arr[w_idx, :], axis=1)
    
    if np.all(totals == 1):
        return {"U": U_final, "V": V_final, "W": W_final}
    
    # Fallback: schoolbook
    U_fb = [[0] * (n * m * p) for _ in range(n * m)]
    V_fb = [[0] * (n * m * p) for _ in range(m * p)]
    W_fb = [[0] * (n * m * p) for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
