<<<<<<< SEARCH
"""Matrix multiplication 4x5x5 with structured block decomposition using Strassen."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    n_m = n * m  # 20
    m_p = m * p  # 25
    p_n = p * n  # 20
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        u_col = np.zeros(n_m)
        v_col = np.zeros(m_p)
        w_col = np.zeros(p_n)
        for idx, val in u_vec:
            u_col[idx] = val
        for idx, val in v_vec:
            v_col[idx] = val
        for idx, val in w_vec:
            w_col[idx] = val
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
    
    def schoolbook(n_prime, m_prime, p_prime, i_off, j_off, k_off):
        for i_loc in range(n_prime):
            for j_loc in range(m_prime):
                for k_loc in range(p_prime):
                    ig = i_off + i_loc
                    jg = j_off + j_loc
                    kg = k_off + k_loc
                    I = ig * m + jg
                    J = jg * p + kg
                    K = kg * n + ig
                    add_term([(I, 1)], [(J, 1)], [(K, 1)])
    
    def strassen_2x2x2(i_off, j_off, k_off):
        def gI(i_l, j_l): return (i_off + i_l) * m + (j_off + j_l)
        def gJ(j_l, k_l): return (j_off + j_l) * p + (k_off + k_l)
        def gK(k_l, i_l): return (k_off + k_l) * n + (i_off + i_l)
        
        # M1: (A00+A11)(B00+B11) -> C00 + C11
        add_term(
            [(gI(0,0), 1), (gI(1,1), 1)],
            [(gJ(0,0), 1), (gJ(1,1), 1)],
            [(gK(0,0), 1), (gK(1,1), 1)]
        )
        # M2: (A10+A11)B00 -> C10 - C11
        add_term(
            [(gI(1,0), 1), (gI(1,1), 1)],
            [(gJ(0,0), 1)],
            [(gK(0,1), 1), (gK(1,1), -1)]
        )
        # M3: A00(B01-B11) -> C01 + C11
        add_term(
            [(gI(0,0), 1)],
            [(gJ(0,1), 1), (gJ(1,1), -1)],
            [(gK(1,0), 1), (gK(1,1), 1)]
        )
        # M4: A11(B10-B00) -> C00 + C10
        add_term(
            [(gI(1,1), 1)],
            [(gJ(1,0), 1), (gJ(0,0), -1)],
            [(gK(0,0), 1), (gK(0,1), 1)]
        )
        # M5: (A00+A01)B11 -> -C00 + C01
        add_term(
            [(gI(0,0), 1), (gI(0,1), 1)],
            [(gJ(1,1), 1)],
            [(gK(0,0), -1), (gK(1,0), 1)]
        )
        # M6: (A10-A00)(B00+B01) -> C11
        add_term(
            [(gI(1,0), 1), (gI(0,0), -1)],
            [(gJ(0,0), 1), (gJ(0,1), 1)],
            [(gK(1,1), 1)]
        )
        # M7: (A01-A11)(B10+B11) -> C00
        add_term(
            [(gI(0,1), 1), (gI(1,1), -1)],
            [(gJ(1,0), 1), (gJ(1,1), 1)],
            [(gK(0,0), 1)]
        )
    
    def handle_block(i_off, j_off, k_off, i_sz, j_sz, k_sz):
        if i_sz == 2 and j_sz == 2 and k_sz == 2:
            strassen_2x2x2(i_off, j_off, k_off)
        elif i_sz == 2 and j_sz == 2 and k_sz == 3:
            # 2x2x2 + 2x2x1
            strassen_2x2x2(i_off, j_off, k_off)
            schoolbook(2, 2, 1, i_off, j_off, k_off + 2)
        elif i_sz == 2 and j_sz == 3 and k_sz == 2:
            # 2x2x2 + 2x1x2
            strassen_2x2x2(i_off, j_off, k_off)
            schoolbook(2, 1, 2, i_off, j_off + 2, k_off)
        elif i_sz == 2 and j_sz == 3 and k_sz == 3:
            # 2x2x2 + 2x2x1 + 2x1x2 + 2x1x1
            strassen_2x2x2(i_off, j_off, k_off)
            schoolbook(2, 2, 1, i_off, j_off, k_off + 2)
            schoolbook(2, 1, 2, i_off, j_off + 2, k_off)
            schoolbook(2, 1, 1, i_off, j_off + 2, k_off + 2)
        else:
            schoolbook(i_sz, j_sz, k_sz, i_off, j_off, k_off)
    
    # Partition: n=4 as [2,2], m=5 as [2,3], p=5 as [2,3]
    i_blocks = [(0, 2), (2, 2)]
    j_blocks = [(0, 2), (2, 3)]
    k_blocks = [(0, 2), (2, 3)]
    
    for i_off, i_sz in i_blocks:
        for j_off, j_sz in j_blocks:
            for k_off, k_sz in k_blocks:
                handle_block(i_off, j_off, k_off, i_sz, j_sz, k_sz)
    
    rank = len(U_cols)
    
    if rank == 0:
        # Fallback
        rank = n * m * p
        U = [[0] * rank for _ in range(n_m)]
        V = [[0] * rank for _ in range(m_p)]
        W = [[0] * rank for _ in range(p_n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    # Verify
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n_m)]
        V = [[0] * rank for _ in range(m_p)]
        W = [[0] * rank for _ in range(p_n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
=======
"""Matrix multiplication 4x5x5 with structured block decomposition using Strassen + rank reduction."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def construct():
    t0 = time.time()
    DEADLINE = t0 + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    n_m = n * m  # 20
    m_p = m * p  # 25
    p_n = p * n  # 20
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        u_col = np.zeros(n_m)
        v_col = np.zeros(m_p)
        w_col = np.zeros(p_n)
        for idx, val in u_vec:
            u_col[idx] = val
        for idx, val in v_vec:
            v_col[idx] = val
        for idx, val in w_vec:
            w_col[idx] = val
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
    
    def schoolbook(n_prime, m_prime, p_prime, i_off, j_off, k_off):
        for i_loc in range(n_prime):
            for j_loc in range(m_prime):
                for k_loc in range(p_prime):
                    ig = i_off + i_loc
                    jg = j_off + j_loc
                    kg = k_off + k_loc
                    I = ig * m + jg
                    J = jg * p + kg
                    K = kg * n + ig
                    add_term([(I, 1)], [(J, 1)], [(K, 1)])
    
    def strassen_2x2x2(i_off, j_off, k_off):
        def gI(i_l, j_l): return (i_off + i_l) * m + (j_off + j_l)
        def gJ(j_l, k_l): return (j_off + j_l) * p + (k_off + k_l)
        def gK(k_l, i_l): return (k_off + k_l) * n + (i_off + i_l)
        
        # M1: (A00+A11)(B00+B11) -> C00 + C11
        add_term(
            [(gI(0,0), 1), (gI(1,1), 1)],
            [(gJ(0,0), 1), (gJ(1,1), 1)],
            [(gK(0,0), 1), (gK(1,1), 1)]
        )
        # M2: (A10+A11)B00 -> C10 - C11
        add_term(
            [(gI(1,0), 1), (gI(1,1), 1)],
            [(gJ(0,0), 1)],
            [(gK(0,1), 1), (gK(1,1), -1)]
        )
        # M3: A00(B01-B11) -> C01 + C11
        add_term(
            [(gI(0,0), 1)],
            [(gJ(0,1), 1), (gJ(1,1), -1)],
            [(gK(1,0), 1), (gK(1,1), 1)]
        )
        # M4: A11(B10-B00) -> C00 + C10
        add_term(
            [(gI(1,1), 1)],
            [(gJ(1,0), 1), (gJ(0,0), -1)],
            [(gK(0,0), 1), (gK(0,1), 1)]
        )
        # M5: (A00+A01)B11 -> -C00 + C01
        add_term(
            [(gI(0,0), 1), (gI(0,1), 1)],
            [(gJ(1,1), 1)],
            [(gK(0,0), -1), (gK(1,0), 1)]
        )
        # M6: (A10-A00)(B00+B01) -> C11
        add_term(
            [(gI(1,0), 1), (gI(0,0), -1)],
            [(gJ(0,0), 1), (gJ(0,1), 1)],
            [(gK(1,1), 1)]
        )
        # M7: (A01-A11)(B10+B11) -> C00
        add_term(
            [(gI(0,1), 1), (gI(1,1), -1)],
            [(gJ(1,0), 1), (gJ(1,1), 1)],
            [(gK(0,0), 1)]
        )
    
    def handle_block(i_off, j_off, k_off, i_sz, j_sz, k_sz):
        if i_sz == 2 and j_sz == 2 and k_sz == 2:
            strassen_2x2x2(i_off, j_off, k_off)
        elif i_sz == 2 and j_sz == 2 and k_sz == 3:
            strassen_2x2x2(i_off, j_off, k_off)
            schoolbook(2, 2, 1, i_off, j_off, k_off + 2)
        elif i_sz == 2 and j_sz == 3 and k_sz == 2:
            strassen_2x2x2(i_off, j_off, k_off)
            schoolbook(2, 1, 2, i_off, j_off + 2, k_off)
        elif i_sz == 2 and j_sz == 3 and k_sz == 3:
            strassen_2x2x2(i_off, j_off, k_off)
            schoolbook(2, 2, 1, i_off, j_off, k_off + 2)
            schoolbook(2, 1, 2, i_off, j_off + 2, k_off)
            schoolbook(2, 1, 1, i_off, j_off + 2, k_off + 2)
        else:
            schoolbook(i_sz, j_sz, k_sz, i_off, j_off, k_off)
    
    # Partition: n=4 as [2,2], m=5 as [2,3], p=5 as [2,3]
    i_blocks = [(0, 2), (2, 2)]
    j_blocks = [(0, 2), (2, 3)]
    k_blocks = [(0, 2), (2, 3)]
    
    for i_off, i_sz in i_blocks:
        for j_off, j_sz in j_blocks:
            for k_off, k_sz in k_blocks:
                handle_block(i_off, j_off, k_off, i_sz, j_sz, k_sz)
    
    rank = len(U_cols)
    
    if rank == 0:
        rank = n * m * p
        U = [[0] * rank for _ in range(n_m)]
        V = [[0] * rank for _ in range(m_p)]
        W = [[0] * rank for _ in range(p_n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    # Verify
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        rank = n * m * p
        U = [[0] * rank for _ in range(n_m)]
        V = [[0] * rank for _ in range(m_p)]
        W = [[0] * rank for _ in range(p_n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # --- Rank reduction via rational Gaussian elimination ---
    if time.time() < DEADLINE:
        # Convert to exact rationals
        U_frac = np.array([[Fraction(x).limit_denominator(10**6) for x in row] for row in U], dtype=object)
        V_frac = np.array([[Fraction(x).limit_denominator(10**6) for x in row] for row in V], dtype=object)
        W_frac = np.array([[Fraction(x).limit_denominator(10**6) for x in row] for row in W], dtype=object)
        
        # Find linearly dependent columns using the tensor structure.
        # The tensor T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
        # If columns c1..cr are linearly dependent in a way that preserves T, we can reduce.
        # Strategy: build the "flattened" coefficient matrix and find its rank.
        
        # Alternative: use SVD-like approach on the combined matrix
        # Stack U, V, W into a single matrix and find rank
        combined = np.vstack([U.flatten(), V.flatten(), W.flatten()])  # (65*rank,)
        
        # Better: use the fact that if we can find a non-trivial linear relation
        # sum_c alpha_c * (u_c, v_c, w_c) = 0 that preserves the tensor,
        # we can drop that column.
        
        # Practical approach: try to find the minimum rank by checking
        # if any column can be expressed as a combination of others while
        # preserving the tensor identity.
        
        # Simpler: compute the rank of the "tensor matricizations"
        # T_{ij,k} = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
        
        # Let's try: for each column, check if it's redundant
        # by seeing if the tensor can still be formed without it.
        
        # Most efficient: use numpy SVD on a combined representation
        # and keep only significant components.
        
        # Actually, the cleanest approach: the 92 terms form a valid decomposition.
        # We can try to find linear dependencies among the rank-1 tensors.
        
        # Build matrix M where M[a, c] encodes the c-th term's contribution
        # to the (a)-th "position" in the tensor.
        # Tensor has n*m*p = 100 entries. Each term c contributes to multiple entries.
        
        # For term c: it contributes u_c[i,j] * v_c[j,k] * w_c[k,i] to T[i,j,k]
        # This is a rank-1 tensor in the 3-way sense.
        
        # Flatten: for each term c, create a vector of length 100:
        # vec_c[i*m*n + j*n + k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # But that's not linear in the parameters. The tensor equation is:
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        
        # This is a system of 100 equations in 92*3 = 276 unknowns (with constraints).
        
        # Let's try a different approach: use the structure to find dependencies.
        # In the block decomposition, terms from different blocks are independent
        # (they affect disjoint sets of tensor entries). Within a block, there might
        # be dependencies.
        
        # For the 2x2x2 Strassen block, the 7 terms are known to be rank-minimal.
        # For the schoolbook terms, they are rank-1 and independent.
        
        # Cross-block: since blocks affect disjoint tensor entries, no cross-block
        # dependencies exist. So the total rank is the sum of block ranks.
        
        # Block ranks:
        # 2x2x2 Strassen: 7
        # 2x2x1 schoolbook: 4
        # 2x1x2 schoolbook: 4  
        # 2x1x1 schoolbook: 2
        
        # Total: 4*7 + 4 + 4 + 2 = 28 + 10 = 38? No, let me recount.
        # i_blocks: [2,2], j_blocks: [2,3], k_blocks: [2,3]
        # 8 blocks total:
        # (2,2,2): Strassen -> 7  [4 blocks: (0,0,0), (0,0,2), (2,0,0), (2,0,2)]
        # Wait, j_blocks[0]=2, j_blocks[1]=3; k_blocks[0]=2, k_blocks[1]=3
        # So blocks are:
        # (i=0,j=0,k=0): 2x2x2 -> 7
        # (i=0,j=0,k=2): 2x2x3 -> 7+4=11
        # (i=0,j=2,k=0): 2x3x2 -> 7+4=11
        # (i=0,j=2,k=2): 2x3x3 -> 7+4+4+2=17
        # (i=2,j=0,k=0): 2x2x2 -> 7
        # (i=2,j=0,k=2): 2x2x3 -> 7+4=11
        # (i=2,j=2,k=0): 2x3x2 -> 7+4=11
        # (i=2,j=2,k=2): 2x3x3 -> 7+4+4+2=17
        # Total: 7+11+11+17+7+11+11+17 = 92. Correct.
        
        # Since blocks are disjoint, no cross-block reduction is possible.
        # Within each block, the decomposition is already rank-minimal for that subproblem.
        
        # However, we might be able to do better by using a different decomposition
        # for the 2x3x3 case. Let's check: 2x3x3 matrix multiplication.
        # Schoolbook: 18. Strassen-based: 7+4+4+2=17. 
        # Is 17 minimal for 2x3x3? The trivial lower bound is max(2*3, 3*3, 2*3)=9.
        # Known: M(2,3,3) <= 17 via this method. Could be lower.
        
        # For now, the rank reduction via Gaussian elimination on the full system
        # won't help because of block disjointness. Let's just return the current result.
        
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE

Wait, I need to reconsider. The block disjointness argument means that simple linear dependence elimination won't reduce the rank. Let me think about whether there's a smarter decomposition.

Actually, let me reconsider the 2×3×3 block. Instead of Strassen(2×2×2) + schoolbook remainder, maybe I can use a different factorization. But the key insight from the directive is to try rank reduction. Let me check if the 92 terms actually have any linear dependencies by examining the tensor structure more carefully.

The issue is that blocks are disjoint in tensor entries, so no cross-block dependencies. Within a 2×2×2 Strassen block, the 7 terms are known to be rank-minimal. So the current decomposition is likely already rank-minimal for this block structure.

Let me try a different approach: use a finer or different block partition to see if we can get a lower total rank.

