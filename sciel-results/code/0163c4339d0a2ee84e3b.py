"""
Matrix multiplication 4x4x4 tensor decomposition.

Strategy:
1. Start with the known rank-49 decomposition from Strassen's algorithm applied
   to 2x2 block matrices (each 2x2 block product uses 7 scalar multiplications).
2. Attempt to reduce the rank by eliminating linearly dependent terms in the
   tensor decomposition using integer linear algebra over the rationals.

The rank-49 decomposition is constructed by:
- Partitioning A and B into 2x2 blocks of 2x2 matrices
- Using Strassen's 7-product formula at the block level (7 block products)
- Each block product is itself a 2x2 matmul, expanded via Strassen again (7 scalar mults each)
- Total: 7 * 7 = 49 scalar multiplications

Then we try to find linear dependencies among the 49 terms to reduce rank.
"""

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 4]


def _strassen_2x2_decomposition():
    """
    Return the rank-49 tensor decomposition for 4x4 matrix multiplication
    using Strassen's algorithm on 2x2 blocks.
    
    Returns U, V, W as lists of lists with rational entries.
    """
    n = 4
    rank = 49
    
    # U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    # Indexing:
    #   A[i,j] -> flat index i*4+j, row in U
    #   B[j,k] -> flat index j*4+k, row in V  
    #   C[i,k] -> flat index k*4+i (trace convention W[k*n+i, r]), row in W
    
    U = [[Fraction(0)] * rank for _ in range(16)]
    V = [[Fraction(0)] * rank for _ in range(16)]
    W = [[Fraction(0)] * rank for _ in range(16)]
    
    # 2x2 Strassen formulas:
    # For matrices X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # Z = X * Y where:
    # z1 = (x00+x11)(y00+y11)
    # z2 = (x10+x11)y00
    # z3 = x00(y01-y11)
    # z4 = x11(y10-y00)
    # z5 = (x00+x01)y11
    # z6 = (x10-x00)(y00+y01)
    # z7 = (x01-x11)(y10+y11)
    # Z[0,0] = z1 - z4 - z5 + z7
    # Z[0,1] = z3 + z5
    # Z[1,0] = z2 + z4
    # Z[1,1] = z1 - z2 + z3 + z6
    
    # Strassen terms: each is (X_linear_form, Y_linear_form, Z_contribution)
    # X_linear_form: dict mapping (row, col) -> coeff for the 2x2 input X
    # Y_linear_form: dict mapping (row, col) -> coeff for the 2x2 input Y
    # Z_contribution: dict mapping (row, col) -> coeff for the 2x2 output Z
    
    strassen_terms = [
        # z1 = (x00+x11)(y00+y11)
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        # z2 = (x10+x11)y00
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1, (1,1): -1}
        ),
        # z3 = x00(y01-y11)
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}
        ),
        # z4 = x11(y10-y00)
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}
        ),
        # z5 = (x00+x01)y11
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}
        ),
        # z6 = (x10-x00)(y00+y01)
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        # z7 = (x01-x11)(y10+y11)
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        ),
    ]
    
    # Block structure: A is divided into 4 blocks of 2x2:
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # Similarly for B and C.
    
    # At the block level, we compute:
    # C00 = f(A00, A01, A10, A11, B00, B01, B10, B11)
    # etc.
    
    # The block-level Strassen products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Then:
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    block_products = [
        # (X_blocks, Y_blocks, Z_blocks)
        # X_blocks: dict mapping block_pos -> coeff
        # Y_blocks: dict mapping block_pos -> coeff
        # Z_blocks: dict mapping block_pos -> coeff
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1, (1,1): -1}
        ),
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}
        ),
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}
        ),
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}
        ),
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        ),
    ]
    
    # Now expand each block product into 7 scalar multiplications using Strassen
    # For each block product Mk = Xk * Yk, where Xk and Yk are 2x2 matrices of
    # linear combinations of A and B entries, we apply the 2x2 Strassen formula.
    
    # A block (br, bc) corresponds to:
    #   A rows 2*br to 2*br+1, cols 2*bc to 2*bc+1
    #   Flat index for A[i,j] = i*4+j
    
    def block_to_entries(block_pos):
        """Return list of flat indices for a 2x2 block."""
        br, bc = block_pos
        entries = []
        for i in range(2):
            for j in range(2):
                row = 2*br + i
                col = 2*bc + j
                entries.append(row * 4 + col)
        return entries  # [top_left, top_right, bottom_left, bottom_right]
    
    # For each block product, we have X as a sum of blocks with coefficients
    # and Y as a sum of blocks with coefficients.
    # The 2x2 matrix Xk has entries that are linear combinations of A entries.
    # Similarly for Yk with B entries.
    
    # Let's build the full 4x4 linear forms for each scalar multiplication term.
    
    all_terms = []  # Each term is (U_vec, V_vec, W_vec) where each is a dict
    
    for bp_idx, (X_blocks, Y_blocks, Z_blocks) in enumerate(block_products):
        # For each of the 7 Strassen sub-terms within this block product:
        for st_idx, (x_form, y_form, z_form) in enumerate(strassen_terms):
            # x_form maps (r,c) in 2x2 -> coeff
            # This means the 2x2 matrix Xk has entries:
            # Xk[r,c] = sum over blocks in X_blocks of coeff * A_block[block][r,c]
            
            # Build U vector: linear form on A entries
            u_dict = {}
            for (r, c), xc in x_form.items():
                for (br, bc), xbc in X_blocks.items():
                    # A_block(br,bc)[r,c] corresponds to A[2*br+r, 2*bc+c]
                    flat = (2*br + r) * 4 + (2*bc + c)
                    coeff = xc * xbc
                    u_dict[flat] = u_dict.get(flat, Fraction(0)) + coeff
            
            # Build V vector: linear form on B entries
            v_dict = {}
            for (r, c), yc in y_form.items():
                for (br, bc), ybc in Y_blocks.items():
                    # B_block(br,bc)[r,c] corresponds to B[2*br+r, 2*bc+c]
                    flat = (2*br + r) * 4 + (2*bc + c)
                    coeff = yc * ybc
                    v_dict[flat] = v_dict.get(flat, Fraction(0)) + coeff
            
            # Build W vector: linear form on C entries (trace convention: W[k*4+i])
            w_dict = {}
            for (r, c), zc in z_form.items():
                for (br, bc), zbc in Z_blocks.items():
                    # C_block(br,bc)[r,c] corresponds to C[2*br+r, 2*bc+c]
                    # Trace convention: W index = k*4 + i where k=col, i=row
                    # C[i,j] -> W[j*4+i]
                    i = 2*br + r
                    j = 2*bc + c
                    flat = j * 4 + i
                    coeff = zc * zbc
                    w_dict[flat] = w_dict.get(flat, Fraction(0)) + coeff
            
            all_terms.append((u_dict, v_dict, w_dict))
    
    # Now fill in U, V, W matrices
    for term_idx, (u_dict, v_dict, w_dict) in enumerate(all_terms):
        for flat, coeff in u_dict.items():
            if coeff != 0:
                U[flat][term_idx] = coeff
        for flat, coeff in v_dict.items():
            if coeff != 0:
                V[flat][term_idx] = coeff
        for flat, coeff in w_dict.items():
            if coeff != 0:
                W[flat][term_idx] = coeff
    
    return U, V, W


def _verify_decomposition(U, V, W, n=4, m=4, p=4):
    """
    Verify that the decomposition exactly reproduces matrix multiplication.
    
    The tensor T[i,j,k] = delta[i,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    # Convert to numpy for verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    # Check: for each (i, j, k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == (1 if i==k else 0)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if i == k else 0.0
                actual = 0.0
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                for r in range(rank):
                    actual += U_np[u_row, r] * V_np[v_row, r] * W_np[w_row, r]
                if abs(actual - expected) > 1e-10:
                    return False
    return True


def _try_reduce_rank(U, V, W, max_time=60):
    """
    Try to reduce the rank by finding linear dependencies among the columns.
    
    A term r is redundant if there exist coefficients c_s (s != r) such that
    for all (i,j): U[i*m+j, r] = sum_s c_s * U[i*m+j, s]
    for all (j,k): V[j*p+k, r] = sum_s c_s * V[j*p+k, s]
    for all (k,i): W[k*n+i, r] = sum_s c_s * W[k*n+i, s]
    
    This is a system of bilinear equations, which is hard in general.
    Instead, we try a simpler approach: check if any column is a linear combination
    of others in U, V, AND W simultaneously.
    """
    start_time = time.time()
    n, m, p = 4, 4, 4
    rank = len(U[0])
    
    # First, check for exact duplicates or zero columns
    new_terms = []
    for r in range(rank):
        if time.time() - start_time > max_time:
            break
        
        # Check if column r is all zeros
        is_zero = all(U[i][r] == 0 for i in range(16)) and \
                  all(V[i][r] == 0 for i in range(16)) and \
                  all(W[i][r] == 0 for i in range(16))
        if is_zero:
            continue
        
        # Check if column r is a scalar multiple of a previous column
        is_dup = False
        for prev_idx, prev_r in enumerate(new_terms):
            # Check if U[:, r] = c * U[:, prev_r] for some c
            c = None
            valid = True
            for i in range(16):
                if U[prev_r][i] != 0:
                    c = U[r][i] / U[prev_r][i]
                    break
                elif U[r][i] != 0:
                    valid = False
                    break
            if not valid:
                continue
            
            # Verify U
            u_ok = all(U[r][i] == c * U[prev_r][i] for i in range(16))
            if not u_ok:
                continue
            
            # Verify V
            v_ok = all(V[r][i] == c * V[prev_r][i] for i in range(16))
            if not v_ok:
                continue
            
            # Verify W
            w_ok = all(W[r][i] == c * W[prev_r][i] for i in range(16))
            if not w_ok:
                continue
            
            is_dup = True
            break
        
        if not is_dup:
            new_terms.append(r)
    
    if len(new_terms) == rank:
        return U, V, W, rank
    
    # Filter to keep only non-redundant terms
    new_rank = len(new_terms)
    new_U = [[U[i][r] for r in new_terms] for i in range(16)]
    new_V = [[V[i][r] for r in new_terms] for i in range(16)]
    new_W = [[W[i][r] for r in new_terms] for i in range(16)]
    
    return new_U, new_V, new_W, new_rank


def construct():
    """
    Construct a tensor decomposition for 4x4 matrix multiplication.
    Returns {'U': U, 'V': V, 'W': W} with integer/half-integer entries.
    """
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # Step 1: Construct the rank-49 Strassen-based decomposition
    U, V, W = _strassen_2x2_decomposition()
    
    # Step 2: Verify the decomposition
    if time.time() < deadline - 5:
        if not _verify_decomposition(U, V, W):
            # Fallback to schoolbook if verification fails
            n, m, p = 4, 4, 4
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}
    
    # Step 3: Try to reduce rank
    if time.time() < deadline - 10:
        U, V, W, rank = _try_reduce_rank(U, V, W, max_time=min(50, deadline - time.time() - 10))
    
    # Step 4: Convert to float if all entries are integers or half-integers
    # The entries should already be rational; convert to float for output
    U_out = [[float(x) for x in row] for row in U]
    V_out = [[float(x) for x in row] for row in V]
    W_out = [[float(x) for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print(f"U shape: ({len(result['U'])}, {len(result['U'][0])})")
    print(f"V shape: ({len(result['V'])}, {len(result['V'][0])})")
    print(f"W shape: ({len(result['W'])}, {len(result['W'][0])})")
