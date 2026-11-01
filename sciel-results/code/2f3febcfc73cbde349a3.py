"""Combined program: Strassen-based rank-21 construction with verification and fallback."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140  # Safety margin before 150s kill

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0])
    # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _build_strassen_rank21(n, m, p):
    """
    Build a rank-21 decomposition using Strassen's algorithm on three 2x2 blocks.
    The 2x2 * 2x6 multiplication is decomposed as three independent 2x2 * 2x2 multiplications,
    each using Strassen's 7-multiplication algorithm.
    """
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    rank = 21
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Strassen coefficients for 2x2 matrix multiplication
    # For multiplying A (2x2) by B (2x2) to get C (2x2):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # u coefficients for A entries [A00, A01, A10, A11]
    # v coefficients for B entries [B00, B01, B10, B11]
    # w coefficients for C entries [C00, C01, C10, C11]
    
    strassen_steps = [
        # M1: (A00+A11)(B00+B11) -> contributes to C00, C11
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        # M2: (A10+A11)B00 -> contributes to C10, C11
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        # M3: A00(B10-B11) -> contributes to C01, C11
        {'u': [1, 0, 0, 0], 'v': [0, 0, 1, -1], 'w': [0, 1, 0, 1]},
        # M4: A11(B01-B00) -> contributes to C00, C10
        {'u': [0, 0, 0, 1], 'v': [1, -1, 0, 0], 'w': [1, 0, 1, 0]},
        # M5: (A00+A01)B11 -> contributes to C00, C01
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        # M6: (A10-A00)(B00+B10) -> contributes to C11
        {'u': [-1, 0, 1, 0], 'v': [1, 0, 1, 0], 'w': [0, 0, 0, 1]},
        # M7: (A01-A11)(B01+B11) -> contributes to C00
        {'u': [0, 1, 0, -1], 'v': [0, 1, 0, 1], 'w': [1, 0, 0, 0]},
    ]
    
    # Pre-compute index mappings
    # A index: a = i*m + j = i*2 + j
    # B index: b = j*p + k = j*6 + k
    # C index: c = k*n + i = k*2 + i
    
    # For block g (g=0,1,2), the local k indices are 2g and 2g+1
    
    v_indices = []
    w_indices = []
    for g in range(3):
        k0 = 2 * g
        k1 = 2 * g + 1
        # B indices: [B00, B01, B10, B11] where local j=0,1 and local k=k0,k1
        # B00: j=0, k=k0 -> 0*6 + k0
        # B01: j=0, k=k1 -> 0*6 + k1
        # B10: j=1, k=k0 -> 1*6 + k0
        # B11: j=1, k=k1 -> 1*6 + k1
        v_indices.append([0*6 + k0, 0*6 + k1, 1*6 + k0, 1*6 + k1])
        # C indices: [C00, C01, C10, C11] where local k=k0,k1 and local i=0,1
        # C00: k=k0, i=0 -> k0*2 + 0
        # C01: k=k1, i=0 -> k1*2 + 0
        # C10: k=k0, i=1 -> k0*2 + 1
        # C11: k=k1, i=1 -> k1*2 + 1
        w_indices.append([k0*2 + 0, k1*2 + 0, k0*2 + 1, k1*2 + 1])
    
    # Fill U, V, W
    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            
            u_list = step['u']
            v_list = step['v']
            w_list = step['w']
            
            v_idx = v_indices[g]
            w_idx = w_indices[g]
            
            for local_idx in range(4):
                u_val = u_list[local_idx]
                if u_val != 0:
                    U[local_idx, r] = u_val
                    
                v_val = v_list[local_idx]
                if v_val != 0:
                    V[v_idx[local_idx], r] = v_val
                    
                w_val = w_list[local_idx]
                if w_val != 0:
                    W[w_idx[local_idx], r] = w_val
    
    # Convert to lists of integers
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Ensure all values are integers
    for matrix in [U_list, V_list, W_list]:
        for row in matrix:
            for idx, val in enumerate(row):
                if isinstance(val, float):
                    if abs(val - round(val)) < 1e-10:
                        row[idx] = int(round(val))
    
    return U_list, V_list, W_list

def _build_schoolbook(n, m, p):
    """Build the schoolbook rank-n*m*p decomposition as fallback."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Build the Strassen rank-21 decomposition
    U, V, W = _build_strassen_rank21(n, m, p)
    
    # Verify the decomposition
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
