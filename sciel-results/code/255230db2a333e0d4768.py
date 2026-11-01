import time
import itertools

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def _build_strassen_block_decomposition():
    """Fallback: 6 independent Strassen blocks, rank 42 (known valid)."""
    n, m, p = 2, 2, 12
    num_blocks = p // 2
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    strassen_terms = [
        {0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1},
        {2: 1, 3: 1}, {0: 1},        {2: 1, 3: -1},
        {0: 1},       {1: 1, 3: -1}, {1: 1, 3: 1},
        {3: 1},       {2: 1, 0: -1}, {0: 1, 2: 1},
        {0: 1, 1: 1}, {3: 1},        {0: -1, 1: 1},
        {0: -1, 2: 1},{0: 1, 1: 1},  {3: 1},
        {1: 1, 3: -1},{2: 1, 3: 1},  {0: 1},
    ]
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u_coeffs = strassen_terms[s * 3]
            v_coeffs = strassen_terms[s * 3 + 1]
            w_coeffs = strassen_terms[s * 3 + 2]
            for a_idx, val in u_coeffs.items():
                U[a_idx][col] = val
            for local_b, val in v_coeffs.items():
                if local_b == 0:
                    global_b = 2 * t
                elif local_b == 1:
                    global_b = 2 * t + 1
                elif local_b == 2:
                    global_b = m * p // 2 + 2 * t
                else:
                    global_b = m * p // 2 + 2 * t + 1
                V[global_b][col] = val
            for local_c, val in w_coeffs.items():
                if local_c == 0:
                    global_w = (2 * t) * n + 0
                elif local_c == 1:
                    global_w = (2 * t + 1) * n + 0
                elif local_c == 2:
                    global_w = (2 * t) * n + 1
                else:
                    global_w = (2 * t + 1) * n + 1
                W[global_w][col] = val
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n=2, m=2, p=12, tol=1e-9):
    """Check sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c] exactly (within tol).
    T[a,b,c] = 1 iff row(A)=row(C) and col(B)=col(C), else 0.
    a in [0,4): a = 2*rowA + colA
    b in [0,24): b = 2*rowB + colB   (B is m x p = 2 x 12, row-major)
    c in [0,24): c = colC*n + rowC   (W trace convention W[k*n+i], k=colC, i=rowC)
    """
    rank = len(U[0])
    # Precompute for speed: for each r, U[:,r] (4), V[:,r] (24), W[:,r] (24)
    # T is sparse: nonzero only when rowA==rowC and colB==colC.
    # For each (rowA, colA) in A and (rowB, colB) in B and (rowC, colC) in C:
    #   T = 1 if rowA==rowC and colB==colC
    # We check all 4*24*24 = 2304 entries.
    for a in range(4):
        rowA, colA = divmod(a, 1)  # a = colA since n*m=4, A is 2x2: a = rowA*2+colA
        # Actually a indexes A entries: a = rowA * 2 + colA
        rowA, colA = a // 2, a % 2
        for b in range(24):
            rowB, colB = b // 12, b % 12  # B is 2 x 12, row-major: b = rowB*12+colB
            for c in range(24):
                colC, rowC = c // 2, c % 2  # W[k*n+i]: c = colC*2 + rowC
                expected = 1.0 if (rowA == rowC and colB == colC) else 0.0
                s = 0.0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if abs(s - expected) > tol:
                    return False
    return True


def _build_global_strassen(rank_target):
    """Attempt a global rank-r decomposition.
    Structural change: A-coefficients are global 2x2 linear forms (Strassen pattern),
    B and W forms are solved so that all 12 columns are covered with shared structure.
    
    We use the 7 Strassen A-forms. For each, we need B-form (24 coeffs) and W-form (24 coeffs)
    such that the sum reconstructs T. We solve via the constraint that for each output
    column pair, the contribution matches.
    
    Approach: Fix the 7 A-forms (Strassen). Then the tensor equation becomes:
      For each (a,b,c): sum_s A_s[a] * B_s[b] * W_s[c] = T[a,b,c]
    This is bilinear in B_s and W_s. We solve by: for a fixed set of A_s, the equation
    sum_s A_s[a] (B_s ⊗ W_s)[b,c] = T_a[b,c] where T_a[b,c] = T[a,b,c].
    So we need to express each T_a (a 24x24 matrix) as sum_s A_s[a] * B_s[b] W_s[c].
    This is a rank-constrained matrix factorization with shared A_s[a] coefficients.
    
    For rank 7 with the 7 Strassen A-forms, we have 7 unknown B_s (24-dim) and 7 unknown W_s (24-dim).
    The equations: for each a (4 values), T_a = sum_s A_s[a] B_s W_s^T  (outer product).
    This is 4 * 24 * 24 = 2304 equations, 7*24 + 7*24 = 336 unknowns. Overdetermined but the
    structure of T_a (each is a permutation-like sparse matrix) may allow exact solution.
    
    We solve using the fact that T_a has a specific structure. Let's just do a numerical solve
    by linearizing: fix W_s, solve for B_s (linear), iterate. Or use the known Strassen
    structure extended to 12 columns.
    """
    n, m, p = 2, 2, 12
    # 7 Strassen A-forms: A_s[a] for a in 0..3
    strassen_A = [
        [1, 0, 0, 1],   # M1 = a + d
        [0, 0, 1, 1],   # M2 = c + d
        [1, 0, 0, 0],   # M3 = a
        [0, 0, 0, 1],   # M4 = d
        [1, 1, 0, 0],   # M5 = a + b
        [-1, 0, 1, 0],  # M6 = c - a
        [0, 1, 0, -1],  # M7 = b - d
    ]
    r = rank_target
    # We'll use only the first r Strassen forms if r < 7, else use 7.
    # For a global decomposition, we need the B and W forms to cover all 12 columns.
    # Key idea: the B-form for Strassen product s, restricted to column block t,
    # must match the Strassen B-form. But we can let B_s be a 24-vector that is
    # the concatenation over 6 blocks of the block-local Strassen B-form.
    # Similarly for W_s. This gives rank 7 *but* each B_s and W_s is 24-dim,
    # and the A_s are 4-dim. Total rank = 7 (not 42!) because the A-forms are shared.
    # Wait - this is the key: if A_s is shared across blocks, then we only need 7 rank-1
    # terms total, not 42. Let's verify this works.
    
    # Block-local Strassen B-forms (for 2x2x2): indexed by s (0..6) and local_b (0..3)
    strassen_B_block = [
        [1, 1, 0, 0],   # M1: e + h  -> local B: [e,f,g,h] = [1,0,0,1]
        [1, 0, 0, 0],   # M2: e      -> [1,0,0,0]
        [0, 1, 0, -1],  # M3: f - h  -> [0,1,0,-1]
        [-1, 0, 1, 0],  # M4: g - e  -> [-1,0,1,0]
        [0, 0, 0, 1],   # M5: h      -> [0,0,0,1]
        [1, 1, 0, 0],   # M6: e + f  -> [1,1,0,0]
        [0, 0, 1, 1],   # M7: g + h  -> [0,0,1,1]
    ]
    # Block-local Strassen W-forms (for 2x2x2): indexed by s and local_c (0..3)
    # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    strassen_W_block = [
        [1, 0, 0, 0],   # M1 -> C00
        [0, 0, 1, -1],  # M2 -> C10 - C11  (C10=+M2, C11=-M2)
        [0, 1, 0, 1],   # M3 -> C01 + C11
        [1, 0, 1, 0],   # M4 -> C00 + C10
        [-1, 1, 0, 0],  # M5 -> -C00 + C01
        [0, 0, 0, 1],   # M6 -> C11
        [1, 0, 0, 0],   # M7 -> C00
    ]
    # Let me recompute W_block carefully.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So for each M_s, its contribution to (C00, C01, C10, C11):
    # M1: (1, 0, 0, 1)
    # M2: (0, 0, 1, -1)
    # M3: (0, 1, 0, 1)
    # M4: (1, 0, 1, 0)
    # M5: (-1, 1, 0, 0)
    # M6: (0, 0, 0, 1)
    # M7: (1, 0, 0, 0)
    strassen_W_block = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 1],   # M3
        [1, 0, 1, 0],   # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0],   # M7
    ]
    
    num_blocks = p // 2  # 6
    # Build global B_s (24-dim) and W_s (24-dim) by tiling the block forms over 6 blocks.
    # B is 2 x 12, row-major: B[rowB][colB], index = rowB*12 + colB
    # For block t, local B indices 0,1,2,3 map to:
    #   local 0 = B[0][2t]    = rowB=0, colB=2t    -> index 0*12 + 2t
    #   local 1 = B[0][2t+1]  = rowB=0, colB=2t+1  -> index 0*12 + 2t+1
    #   local 2 = B[1][2t]    = rowB=1, colB=2t    -> index 1*12 + 2t
    #   local 3 = B[1][2t+1]  = rowB=1, colB=2t+1  -> index 1*12 + 2t+1
    # W is p x n = 12 x 2, trace convention W[k*n+i] where k=colC, i=rowC, index = colC*2 + rowC
    # For block t, local C indices 0,1,2,3 map to:
    #   local 0 = C00 = rowC=0, colC=2t   -> index (2t)*2 + 0
    #   local 1 = C01 = rowC=0, colC=2t+1 -> index (2t+1)*2 + 0
    #   local 2 = C10 = rowC=1, colC=2t   -> index (2t)*2 + 1
    #   local 3 = C11 = rowC=1, colC=2t+1 -> index (2t+1)*2 + 1
    
    U = [[0] * r for _ in range(4)]
    V = [[0] * r for _ in range(24)]
    W = [[0] * r for _ in range(24)]
    
    for s in range(r):
        # U: A-form s
        for a in range(4):
            U[a][s] = strassen_A[s][a]
        # V: B-form s, tiled over blocks
        for t in range(num_blocks):
            for local_b in range(4):
                val = strassen_B_block[s][local_b]
                if val == 0:
                    continue
                if local_b == 0:
                    gidx = 0 * 12 + 2 * t
                elif local_b == 1:
                    gidx = 0 * 12 + 2 * t + 1
                elif local_b == 2:
                    gidx = 1 * 12 + 2 * t
                else:
                    gidx = 1 * 12 + 2 * t + 1
                V[gidx][s] += val
        # W: W-form s, tiled over blocks
        for t in range(num_blocks):
            for local_c in range(4):
                val = strassen_W_block[s][local_c]
                if val == 0:
                    continue
                if local_c == 0:
                    gidx = (2 * t) * 2 + 0
                elif local_c == 1:
                    gidx = (2 * t + 1) * 2 + 0
                elif local_c == 2:
                    gidx = (2 * t) * 2 + 1
                else:
                    gidx = (2 * t + 1) * 2 + 1
                W[gidx][s] += val
    return {"U": U, "V": V, "W": W}


def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    # STRUCTURAL CHANGE: global Strassen decomposition with shared A-forms.
    # Rank 7 (down from 42). Verify exactly; fall back to rank 42 if invalid.
    best = None
    best_rank = None

    # Try rank 7 global
    for r in [7, 8]:
        if time.time() > deadline:
            break
        cand = _build_global_strassen(r)
        if _verify(cand["U"], cand["V"], cand["W"]):
            if best is None or r < best_rank:
                best = cand
                best_rank = r
            break

    if best is None:
        # Fallback to known-valid rank 42 block decomposition
        best = _build_strassen_block_decomposition()
        best_rank = 42

    return best
# EVOLVE-BLOCK-END
