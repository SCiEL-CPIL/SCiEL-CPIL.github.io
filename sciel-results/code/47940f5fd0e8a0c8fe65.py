"""Tensor decomposition for 4x4 * 4x5 matrix multiplication."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _validate(U, V, W, n=4, m=4, p=5, tol=1e-9):
    """Check that U,V,W form a valid tensor decomposition of matmul(n,m,p)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    M_decomp = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
    if not np.allclose(M_decomp, 1.0, atol=tol):
        return False
    return True

def _strassen_2x2_terms():
    """
    Return the 7 rank-1 tensors for 2x2x2 matrix multiplication (Strassen).
    Each term is (u, v, w) where u, v, w are length-4 vectors (flattened 2x2).
    u[i*2+j], v[j*2+k], w[k*2+i]
    """
    terms = []
    
    # Term 1: M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # u: A00=1, A11=1 -> [1,0,0,1]
    # v: B00=1, B11=1 -> [1,0,0,1]
    # w: C00+=1, C11+=1 -> [1,0,0,1]
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1]
    terms.append((u1, v1, w1))
    
    # Term 2: M2 = (A10+A11)(B00) -> contributes to C10
    # u: A10=1, A11=1 -> [0,0,1,1]
    # v: B00=1 -> [1,0,0,0]
    # w: C10+=1 -> [0,0,1,0]
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, 0]
    terms.append((u2, v2, w2))
    
    # Term 3: M3 = A00(B10-B11) -> contributes to C01
    # u: A00=1 -> [1,0,0,0]
    # v: B10=1, B11=-1 -> [0,1,1,-1]
    # w: C01+=1 -> [0,1,0,0]
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 1, -1]
    w3 = [0, 1, 0, 0]
    terms.append((u3, v3, w3))
    
    # Term 4: M4 = A11(B00-B01) -> contributes to C00, C11
    # u: A11=1 -> [0,0,0,1]
    # v: B00=1, B01=-1 -> [1,-1,0,0]
    # w: C00+=1, C11+=1 -> [1,0,0,1]
    u4 = [0, 0, 0, 1]
    v4 = [1, -1, 0, 0]
    w4 = [1, 0, 0, 1]
    terms.append((u4, v4, w4))
    
    # Term 5: M5 = (A00+A01)B11 -> contributes to C00, C01
    # u: A00=1, A01=1 -> [1,1,0,0]
    # v: B11=1 -> [0,0,0,1]
    # w: C00+=1, C01+=1 -> [1,1,0,0]
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [1, 1, 0, 0]
    terms.append((u5, v5, w5))
    
    # Term 6: M6 = (A10-A00)(B00+B01) -> contributes to C10, C11
    # u: A10=1, A00=-1 -> [-1,0,1,0]
    # v: B00=1, B01=1 -> [1,1,0,0]
    # w: C10+=1, C11+=1 -> [0,0,1,1]
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 1, 1]
    terms.append((u6, v6, w6))
    
    # Term 7: M7 = (A01-A11)(B10+B11) -> contributes to C01, C11
    # u: A01=1, A11=-1 -> [0,1,0,-1]
    # v: B10=1, B11=1 -> [0,0,1,1]
    # w: C01+=1, C11+=1 -> [0,1,0,1]
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [0, 1, 0, 1]
    terms.append((u7, v7, w7))
    
    return terms

def _construct_strassen_4x4_4x5():
    """
    Construct a decomposition for 4x4 * 4x5 using Strassen on the 4x4 part
    and schoolbook for the boundary column.
    
    Strategy:
    - Split 4x4 into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    - For the first 4 columns of B (4x4 part), use Strassen's 7 multiplications
    - Each Strassen multiplication of 2x2 blocks requires 7 scalar 2x2x2 terms
    - Total for 4x4 part: 7 * 7 = 49 terms
    - For the 5th column of B (4x1), use schoolbook: 4*4*1 = 16 terms
    - Total rank: 49 + 16 = 65
    """
    n, m, p = 4, 4, 5
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    strassen_2x2 = _strassen_2x2_terms()
    
    # Block indices: block (bi, bj) corresponds to rows bi*2:(bi+1)*2, cols bj*2:(bj+1)*2
    # For A (4x4): blocks A00, A01, A10, A11
    # For B first 4 cols (4x4): blocks B00, B01, B10, B11
    
    # Strassen formulas for 2x2 block matrix multiplication:
    # C00 = M1 + M4 - M5 + M7  (but we use the additive form with 7 terms)
    # The 7 terms are defined such that:
    # C = A*B where the block structure gives the correct result
    
    # For each of the 7 Strassen multiplies, we need to define:
    # - Which blocks of A are combined (with signs)
    # - Which blocks of B are combined (with signs)
    # - Which blocks of C receive the result (with signs)
    
    # Strassen's 7 products for 2x2 block matrices:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B00 - B01)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Then:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M4 + M7
    
    # For each Strassen product Mk, we apply the 2x2x2 decomposition.
    # The "A" matrix for Mk is a 2x2 block matrix formed from the specified blocks.
    # The "B" matrix for Mk is similarly formed.
    # The result contributes to the specified C blocks.
    
    # Let's define the block mappings for each Strassen product:
    # a_blocks[mk] = list of (block_idx, sign) for A
    # b_blocks[mk] = list of (block_idx, sign) for B
    # c_blocks[mk] = list of (block_idx, sign) for C
    
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # i.e., block (bi, bj) -> idx = bi*2 + bj
    
    # M1: A = A00 + A11, B = B00 + B11
    # A blocks: (0, +1), (3, +1)
    # B blocks: (0, +1), (3, +1)
    # C blocks: (0, +1), (3, +1)
    
    # M2: A = A10 + A11, B = B00
    # A blocks: (2, +1), (3, +1)
    # B blocks: (0, +1)
    # C blocks: (2, +1)
    
    # M3: A = A00, B = B10 - B11
    # A blocks: (0, +1)
    # B blocks: (2, +1), (3, -1)
    # C blocks: (1, +1)
    
    # M4: A = A11, B = B00 - B01
    # A blocks: (3, +1)
    # B blocks: (0, +1), (1, -1)
    # C blocks: (0, +1), (3, +1)
    
    # M5: A = A00 + A01, B = B11
    # A blocks: (0, +1), (1, +1)
    # B blocks: (3, +1)
    # C blocks: (0, +1), (1, +1)
    
    # M6: A = A10 - A00, B = B00 + B01
    # A blocks: (2, +1), (0, -1)
    # B blocks: (0, +1), (1, +1)
    # C blocks: (2, +1), (3, +1)
    
    # M7: A = A01 - A11, B = B10 + B11
    # A blocks: (1, +1), (3, -1)
    # B blocks: (2, +1), (3, +1)
    # C blocks: (1, +1), (3, +1)
    
    strassen_defs = [
        # (a_blocks, b_blocks, c_blocks)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),  # M2
        ([(0, 1)], [(2, 1), (3, -1)], [(1, 1)]),  # M3
        ([(3, 1)], [(0, 1), (1, -1)], [(0, 1), (3, 1)]),  # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, 1), (1, 1)]),  # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(2, 1), (3, 1)]),  # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(1, 1), (3, 1)]),  # M7
    ]
    
    # For each Strassen product, we use the 7 scalar 2x2x2 terms.
    # The scalar terms operate on 2x2 matrices.
    # We need to map the 2x2 indices to the 4x4 block structure.
    
    # For a given Strassen product with:
    # - A formed from blocks a_blocks
    # - B formed from blocks b_blocks
    # - C formed from blocks c_blocks
    #
    # The 2x2 "matrix" for A has entries:
    #   A_scalar[bi, bj] = sum over (block_a, sign) in a_blocks where block_a == (bi, bj) of sign * A[(bi*2):(bi+1)*2, (bj*2):(bj+1)*2]
    # Similarly for B and C.
    #
    # For each scalar term (u, v, w) in strassen_2x2:
    #   u[i*2+j] contributes to A[i,j]
    #   v[j*2+k] contributes to B[j,k]
    #   w[k*2+i] contributes to C[k,i]
    #
    # In the 4x4 structure:
    #   A[i,j] is a 2x2 block at position (i*2, j*2)
    #   So A_scalar[i,j] = A[2i:2i+2, 2j:2j+2]
    #
    # But in our Strassen, the "A" is formed from specific blocks.
    # Let's define the effective A matrix (2x2 of 2x2 blocks):
    #   A_eff[i,j] = sum of signs * A[block] for each (block, sign) in a_blocks where block == (i,j)
    #
    # Actually, each a_block is a specific 2x2 block of the 4x4 A.
    # So A_eff is a 2x2 matrix where each entry is a linear combination of 2x2 blocks.
    #
    # For the scalar term, u[bi*2+bj] is the coefficient for A_eff[bi, bj].
    # A_eff[bi, bj] is itself a sum of 2x2 blocks.
    
    col = 0
    for mk in range(7):
        a_blocks, b_blocks, c_blocks = strassen_defs[mk]
        
        # Build effective block mappings
        # a_map[(bi, bj)] = sum of signs for that block position
        a_map = {}
        for block_idx, sign in a_blocks:
            bi = block_idx // 2
            bj = block_idx % 2
            a_map[(bi, bj)] = a_map.get((bi, bj), 0) + sign
        
        b_map = {}
        for block_idx, sign in b_blocks:
            bi = block_idx // 2
            bj = block_idx % 2
            b_map[(bi, bj)] = b_map.get((bi, bj), 0) + sign
        
        c_map = {}
        for block_idx, sign in c_blocks:
            bi = block_idx // 2
            bj = block_idx % 2
            c_map[(bi, bj)] = c_map.get((bi, bj), 0) + sign
        
        for r in range(7):
            u, v, w = strassen_2x2[r]
            
            # For each 2x2 entry (bi, bj) in A:
            # u[bi*2+bj] is the coefficient
            # This coefficient multiplies A_eff[bi, bj]
            # A_eff[bi, bj] = a_map.get((bi, bj), 0) * A[2*bi:2*bi+2, 2*bj:2*bj+2]
            # So the contribution to U is:
            #   For each (row, col) in the 2x2 block at (2*bi, 2*bj):
            #     U[row*m + col, col_idx] += u[bi*2+bj] * a_map.get((bi, bj), 0)
            
            for bi in range(2):
                for bj in range(2):
                    u_coeff = u[bi * 2 + bj]
                    a_sign = a_map.get((bi, bj), 0)
                    if u_coeff == 0 or a_sign == 0:
                        continue
                    # Block at (2*bi, 2*bj)
                    for di in range(2):
                        for dj in range(2):
                            row = 2 * bi + di
                            col_a = 2 * bj + dj
                            U[row * m + col_a][col] += u_coeff * a_sign
            
            # Similarly for V (B matrix, m x p, but only first 4 cols)
            for bj in range(2):  # row index in B (which is col index in A)
                for bk in range(2):  # col index in B
                    v_coeff = v[bj * 2 + bk]
                    b_sign = b_map.get((bj, bk), 0)
                    if v_coeff == 0 or b_sign == 0:
                        continue
                    # Block at (2*bj, 2*bk) in B
                    for dj in range(2):
                        for dk in range(2):
                            row_v = 2 * bj + dj
                            col_v = 2 * bk + dk
                            V[row_v * p + col_v][col] += v_coeff * b_sign
            
            # Similarly for W (C matrix, p x n, first 4 rows)
            for bk in range(2):  # row index in C
                for bi in range(2):  # col index in C
                    w_coeff = w[bk * 2 + bi]
                    c_sign = c_map.get((bk, bi), 0)
                    if w_coeff == 0 or c_sign == 0:
                        continue
                    # Block at (2*bk, 2*bi) in C
                    for dk in range(2):
                        for di in range(2):
                            row_w = 2 * bk + dk
                            col_w = 2 * bi + di
                            W[row_w * n + col_w][col] += w_coeff * c_sign
            
            col += 1
    
    # Schoolbook for the 5th column of B (k=4)
    # For each (i, j): C[i, 4] += A[i, j] * B[j, 4]
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}

def _greedy_reduce(U, V, W, n, m, p, max_iterations=100, time_limit=20):
    """
    Try to reduce the rank by finding redundant or mergeable terms.
    """
    start_time = time.time()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    def _is_valid(U_mat, V_mat, W_mat, r):
        if r <= 0:
            return False
        U3 = U_mat[:, :r].reshape(n, m, -1)
        V3 = V_mat[:, :r].reshape(m, p, -1)
        W3 = W_mat[:, :r].reshape(p, n, -1)
        M_decomp = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
        return np.allclose(M_decomp, 1.0, atol=1e-9)
    
    # Try removing individual terms
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if rank <= 1:
            break
        
        found = False
        for t in range(rank):
            if time.time() - start_time > time_limit:
                break
            
            U_test = U.copy()
            V_test = V.copy()
            W_test = W.copy()
            U_test[:, t] = 0
            V_test[:, t] = 0
            W_test[:, t] = 0
            
            if _is_valid(U_test, V_test, W_test, rank):
                # Remove term t
                keep = list(range(rank))
                keep.pop(t)
                U = U[:, keep]
                V = V[:, keep]
                W = W[:, keep]
                rank -= 1
                found = True
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
        
        if not found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}, best_rank

def construct():
    """Construct a rank-minimal tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with the Strassen-embedded construction (rank 65)
    result = _construct_strassen_4x4_4x5()
    rank = len(result["U"][0]) if result["U"] else 0
    
    # Validate
    if not _validate(result["U"], result["V"], result["W"], n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        result = {"U": U, "V": V, "W": W}
    
    # Try to reduce the rank with greedy search
    remaining = deadline - time.time()
    if remaining > 5:
        try:
            reduced, reduced_rank = _greedy_reduce(
                result["U"], result["V"], result["W"], n, m, p,
                max_iterations=200, time_limit=min(60, remaining - 5)
            )
            if reduced_rank < rank:
                if _validate(reduced["U"], reduced["V"], reduced["W"], n, m, p):
                    result = reduced
                    rank = reduced_rank
        except Exception:
            pass
    
    return result

# EVOLVE-BLOCK-END
