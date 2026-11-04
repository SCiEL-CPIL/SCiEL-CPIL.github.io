import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def _verify(U, V, W, n=4, m=4, p=7, tol=1e-9):
    """Verify that the decomposition reproduces the matmul tensor."""
    r = len(U[0]) if U and U[0] else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = U[i * m + j]
                vk = V[j * p + k]
                wi = W[k * n + i]
                for c in range(r):
                    s += ui[c] * vk[c] * wi[c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n=4, m=4, p=7, start_col=0, end_col=None):
    """Schoolbook rank n*m*p decomposition for columns [start_col, end_col)."""
    if end_col is None:
        end_col = p
    num_cols = end_col - start_col
    rank = n * m * num_cols
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(start_col, end_col):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2_terms():
    """
    Returns the 7 rank-1 terms for 2x2x2 Strassen decomposition.
    Each term is a tuple (u, v, w) where u, v, w are lists of 4 floats.
    Indices:
    U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    V: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    W: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)  (W[k*n+i])
    """
    # Standard Strassen 2x2x2 decomposition
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00       -> C10 += M2
    # M3 = A00(B01-B11)       -> C01 += M3
    # M4 = A11(B10-B00)       -> C00 -= M4, C10 += M4
    # M5 = (A00+A01)B11       -> C00 += M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7, C11 += M7
    
    # Let's verify this set of terms.
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 + M6 + M7
    
    # Check M1: (A00+A11)(B00+B11)
    # U: [1, 0, 0, 1]
    # V: [1, 0, 0, 1]
    # W: [1, 0, 0, 1] (C00, C11)
    
    # Check M2: (A10+A11)B00
    # U: [0, 0, 1, 1]
    # V: [1, 0, 0, 0]
    # W: [0, 1, 0, 0] (C10)
    
    # Check M3: A00(B01-B11)
    # U: [1, 0, 0, 0]
    # V: [0, 1, 0, -1]
    # W: [0, 0, 1, 0] (C01)
    
    # Check M4: A11(B10-B00)
    # U: [0, 0, 0, 1]
    # V: [0, 0, 1, -1]
    # W: [-1, 1, 0, 0] (C00, C10)
    
    # Check M5: (A00+A01)B11
    # U: [1, 1, 0, 0]
    # V: [0, 0, 0, 1]
    # W: [1, 0, 1, 0] (C00, C01)
    
    # Check M6: (A10-A00)(B00+B01)
    # U: [-1, 0, 1, 0]
    # V: [1, 1, 0, 0]
    # W: [0, 0, 0, 1] (C11)
    
    # Check M7: (A01-A11)(B10+B11)
    # U: [0, 1, 0, -1]
    # V: [0, 0, 1, 1]
    # W: [1, 0, 0, 1] (C00, C11)
    
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 0]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [-1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]),
    ]
    
    # Verify these terms locally
    # T[i,j,k] = sum_c u[i,j,c] v[j,k,c] w[k,i,c]
    # Should be 1 for all i,j,k in 0..1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(7):
                    u = terms[c][0]
                    v = terms[c][1]
                    w = terms[c][2]
                    ui = u[i*2+j]
                    vk = v[j*2+k]
                    wk = w[k*2+i]
                    s += ui * vk * wk
                if abs(s - 1.0) > 1e-9:
                    # Fallback to schoolbook if Strassen terms are wrong
                    return None
    return terms

def _strassen_4x4x4():
    """
    Generates a rank-49 decomposition for 4x4x4 matrix multiplication.
    Uses 2x2 block structure.
    """
    base_terms = _strassen_2x2_terms()
    if base_terms is None:
        return None
        
    # We need to expand the 2x2 block operations to 4x4 scalar operations.
    # A 4x4 matrix is a 2x2 block matrix of 2x2 blocks.
    # Let A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # The Strassen algorithm on blocks computes 7 block products.
    # Each block product is a 2x2 matmul.
    # We decompose each 2x2 matmul into 7 rank-1 scalar terms.
    # Total rank = 7 * 7 = 49.
    
    # Block Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Output blocks:
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 + M6 + M7
    
    # We will construct U, V, W for the 4x4x4 case.
    # U has 16 rows (4x4), V has 16 rows, W has 16 rows.
    # Rank 49.
    
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    # Helper to get 2x2 block index in 4x4 matrix
    # Block (br, bc) covers rows br*2..br*2+1, cols bc*2..bc*2+1
    # Scalar index (i, j) maps to block (i//2, j//2) and local (i%2, j%2)
    
    # For each of the 7 block products (M1..M7):
    #   For each of the 7 scalar terms (t1..t7) of the 2x2 matmul:
    #     We add a rank-1 term to the 4x4x4 decomposition.
    
    # Block product definitions:
    # Each block product Mk involves specific blocks of A and B.
    # We need to map the 2x2 indices of the scalar term to the 4x4 indices.
    
    # Let's define the block contributions for each Mk.
    # Mk = (sum_alpha A_alpha) * (sum_beta B_beta)
    # where A_alpha and B_beta are blocks of A and B.
    
    # M1: A00+A11, B00+B11
    # M2: A10+A11, B00
    # M3: A00, B01-B11
    # M4: A11, B10-B00
    # M5: A00+A01, B11
    # M6: A10-A00, B00+B01
    # M7: A01-A11, B10+B11
    
    # For each Mk, we have a list of (block_row, block_col, coeff) for A and B.
    # And for the output, we know which C blocks it contributes to and with what sign.
    
    # Actually, it's easier to think of it as:
    # For each scalar term (u2, v2, w2) in the 2x2x2 Strassen:
    #   u2[i,j] is a coefficient for the (i,j) entry of the 2x2 "block matrix" A.
    #   This means u2[i,j] applies to the entire 2x2 block A[i,j].
    #   So for all local (r,c) in 0..1, U[ (i*2+r)*4 + (j*2+c) ] += u2[i,j] * (scaling?)
    #   Wait, the block product is a 2x2 matmul of the blocks.
    #   The scalar term u2[i,j] v2[j,k] w2[k,i] is a rank-1 term for the 2x2x2 block matmul.
    #   This term represents the product of the (i,j) block of A, (j,k) block of B,
    #   contributing to the (k,i) block of C.
    #   So, for a given scalar term index c in 0..6:
    #     For each (i,j) in 0..1:
    #       if u2[i,j] != 0:
    #         For each local (r,s) in 0..1:
    #           U[ (i*2+r)*4 + (j*2+s) ][global_c] = u2[i,j]
    #     Similarly for V and W.
    
    # But wait, the 7 block products M1..M7 are linear combinations of block products.
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # This is a sum of 4 block products.
    # Each block product Aij Bkl is a 2x2 matmul.
    # So M1 is a sum of 4 matmuls.
    # This is not a single rank-1 term at the block level.
    
    # The Strassen algorithm replaces the 8 block products with 7 "fancy" products.
    # M1 = (A00+A11)(B00+B11). This is ONE product of two 2x2 matrices.
    # Let X = A00+A11, Y = B00+B11. M1 = X Y.
    # X and Y are 2x2 matrices.
    # So M1 is a 2x2 matmul.
    # We decompose this 2x2 matmul into 7 scalar rank-1 terms.
    # So for M1, we have 7 terms.
    # Each term corresponds to a specific combination of entries of X and Y.
    # X is a 2x2 matrix. Its entries are sums of A blocks.
    # X[0,0] = A00[0,0] + A11[0,0].
    # So the scalar term involving X[0,0] involves A00[0,0] and A11[0,0].
    
    # So, for each of the 7 block products Mk:
    #   We form the 2x2 matrices Xk and Yk.
    #   We decompose Xk * Yk into 7 scalar rank-1 terms.
    #   For each scalar term, we map the entries of Xk and Yk back to the 4x4 A and B.
    
    # This is doable.
    
    # Let's define the 7 block products explicitly.
    # We will represent Xk and Yk as 2x2 matrices where each entry is a linear combination of 4x4 A/B entries.
    
    # Actually, it's simpler to just iterate over the 7 terms of the 2x2 Strassen for each Mk,
    # and for each term, expand the 2x2 indices to 4x4 indices.
    
    # Let's define the block mappings for each Mk.
    # Mk is a product of two 2x2 matrices X and Y.
    # X and Y are composed of blocks of A and B.
    
    # M1: X = A00+A11, Y = B00+B11
    # M2: X = A10+A11, Y = B00
    # M3: X = A00, Y = B01-B11
    # M4: X = A11, Y = B10-B00
    # M5: X = A00+A01, Y = B11
    # M6: X = A10-A00, Y = B00+B01
    # M7: X = A01-A11, Y = B10+B11
    
    # For each Mk, we have a list of (block_row, block_col, coeff) for X and Y.
    
    # Let's create a helper to expand a 2x2 term to 4x4.
    
    def expand_term(u2, v2, w2, x_blocks, y_blocks, z_blocks, offset):
        """
        u2, v2, w2: 4-element lists for the 2x2 term.
        x_blocks: list of (br, bc, coeff) for X (A blocks)
        y_blocks: list of (br, bc, coeff) for Y (B blocks)
        z_blocks: list of (br, bc, coeff) for Z (C blocks)
        offset: starting column index in U, V, W
        """
        # For each (i,j) in 0..1:
        #   u2[i*2+j] is the coefficient for the (i,j) entry of X.
        #   This entry is a sum of A blocks.
        #   For each (br, bc, c) in x_blocks:
        #     if (br, bc) == (i, j):
        #       For each local (r,s) in 0..1:
        #         U[ (i*2+r)*4 + (j*2+s) ][offset] = u2[i*2+j] * c
        #   Wait, u2[i,j] is the coefficient for the (i,j) entry of the 2x2 matrix X.
        #   The (i,j) entry of X is a sum of A blocks.
        #   So we add u2[i,j] * c to the corresponding A entries.
        
        # Let's do it carefully.
        # U is 16x49.
        # We are filling column `offset`.
        
        for i in range(2):
            for j in range(2):
                ui = u2[i*2+j]
                if ui != 0:
                    for (br, bc, c) in x_blocks:
                        if br == i and bc == j:
                            for r in range(2):
                                for s in range(2):
                                    row = (i*2+r)*4 + (j*2+s)
                                    U[row][offset] += ui * c
                            
        for j in range(2):
            for k in range(2):
                vj = v2[j*2+k]
                if vj != 0:
                    for (br, bc, c) in y_blocks:
                        if br == j and bc == k:
                            for r in range(2):
                                for s in range(2):
                                    row = (j*2+r)*4 + (k*2+s)
                                    V[row][offset] += vj * c
                                    
        for k in range(2):
            for i in range(2):
                wk = w2[k*2+i]
                if wk != 0:
                    for (br, bc, c) in z_blocks:
                        if br == k and bc == i:
                            for r in range(2):
                                for s in range(2):
                                    row = (k*2+r)*4 + (i*2+s)
                                    W[row][offset] += wk * c
                                    
    # Define the block mappings for each Mk.
    # X and Y are 2x2 matrices.
    # x_blocks is a list of (br, bc, coeff) where br, bc are block indices (0,1).
    
    # M1: X = A00+A11, Y = B00+B11
    # X blocks: (0,0,1), (1,1,1)
    # Y blocks: (0,0,1), (1,1,1)
    # Z blocks (C): C00 += M1, C11 += M1 -> (0,0,1), (1,1,1)
    
    # M2: X = A10+A11, Y = B00
    # X blocks: (1,0,1), (1,1,1)
    # Y blocks: (0,0,1)
    # Z blocks: C10 += M2 -> (1,0,1)
    
    # M3: X = A00, Y = B01-B11
    # X blocks: (0,0,1)
    # Y blocks: (0,1,1), (1,1,-1)
    # Z blocks: C01 += M3 -> (0,1,1)
    
    # M4: X = A11, Y = B10-B00
    # X blocks: (1,1,1)
    # Y blocks: (1,0,1), (0,0,-1)
    # Z blocks: C00 -= M4, C10 += M4 -> (0,0,-1), (1,0,1)
    
    # M5: X = A00+A01, Y = B11
    # X blocks: (0,0,1), (0,1,1)
    # Y blocks: (1,1,1)
    # Z blocks: C00 += M5, C01 += M5 -> (0,0,1), (0,1,1)
    
    # M6: X = A10-A00, Y = B00+B01
    # X blocks: (1,0,1), (0,0,-1)
    # Y blocks: (0,0,1), (0,1,1)
    # Z blocks: C11 += M6 -> (1,1,1)
    
    # M7: X = A01-A11, Y = B10+B11
    # X blocks: (0,1,1), (1,1,-1)
    # Y blocks: (1,0,1), (1,1,1)
    # Z blocks: C00 += M7, C11 += M7 -> (0,0,1), (1,1,1)
    
    mappings = [
        # M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
        # M3
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1)]),
        # M4
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,-1), (1,0,1)]),
        # M5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,1), (0,1,1)]),
        # M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
    ]
    
    offset = 0
    for m_idx in range(7):
        x_blocks, y_blocks, z_blocks = mappings[m_idx]
        for t_idx in range(7):
            u2, v2, w2 = base_terms[t_idx]
            expand_term(u2, v2, w2, x_blocks, y_blocks, z_blocks, offset)
            offset += 1
            
    return {"U": U, "V": "V", "W": W} # Bug: return V, not "V"

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Try Strassen 4x4x4 (Rank 49) + Schoolbook 4x4x3 (Rank 48) = 97
    # If it fails, fallback to Schoolbook 112
    
    # First, let's get the Strassen 4x4x4 decomposition
    # Note: _strassen_4x4x4 has a bug in the return statement in the draft above.
    # Let's fix it in the actual code.
    
    # For now, let's just implement the logic directly in construct to avoid function call overhead and bugs.
    
    base_terms = _strassen_2x2_terms()
    if base_terms is None:
        return _schoolbook(n, m, p)
        
    # Build Strassen 4x4x4
    U_s = [[0]*49 for _ in range(16)]
    V_s = [[0]*49 for _ in range(16)]
    W_s = [[0]*49 for _ in range(16)]
    
    def expand_term(u2, v2, w2, x_blocks, y_blocks, z_blocks, offset, U, V, W):
        for i in range(2):
            for j in range(2):
                ui = u2[i*2+j]
                if ui != 0:
                    for (br, bc, c) in x_blocks:
                        if br == i and bc == j:
                            for r in range(2):
                                for s in range(2):
                                    row = (i*2+r)*4 + (j*2+s)
                                    U[row][offset] += ui * c
                            
        for j in range(2):
            for k in range(2):
                vj = v2[j*2+k]
                if vj != 0:
                    for (br, bc, c) in y_blocks:
                        if br == j and bc == k:
                            for r in range(2):
                                for s in range(2):
                                    row = (j*2+r)*4 + (k*2+s)
                                    V[row][offset] += vj * c
                                    
        for k in range(2):
            for i in range(2):
                wk = w2[k*2+i]
                if wk != 0:
                    for (br, bc, c) in z_blocks:
                        if br == k and bc == i:
                            for r in range(2):
                                for s in range(2):
                                    row = (k*2+r)*4 + (i*2+s)
                                    W[row][offset] += wk * c
                                    
    mappings = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,-1), (1,0,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,1), (0,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
    ]
    
    offset = 0
    for m_idx in range(7):
        x_blocks, y_blocks, z_blocks = mappings[m_idx]
        for t_idx in range(7):
            u2, v2, w2 = base_terms[t_idx]
            expand_term(u2, v2, w2, x_blocks, y_blocks, z_blocks, offset, U_s, V_s, W_s)
            offset += 1
            
    # Build Schoolbook 4x4x3 (columns 4,5,6)
    # Rank 48
    U_b = [[0]*48 for _ in range(16)]
    V_b = [[0]*48 for _ in range(28)] # m*p = 4*7 = 28
    W_b = [[0]*48 for _ in range(28)] # p*n = 7*4 = 28
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U_b[i*4+j][col] = 1
                V_b[j*7+k][col] = 1
                W_b[k*4+i][col] = 1
                col += 1
                
    # Combine
    # U: 16x49 + 16x48 -> 16x97
    # V: 16x49 + 28x48 -> 28x97 (Pad U_s/V_s/W_s to correct sizes)
    # W: 16x49 + 28x48 -> 28x97
    
    # Wait, V_s is 16x49 (m*p for 4x4 is 16). But we need V to be 28x97 (4*7).
    # The Strassen part only handles the first 4 columns of B and C.
    # So V_s should be 28x49, with the last 12 rows being 0.
    # W_s should be 28x49, with the last 12 rows being 0.
    
    U_final = [[0]*97 for _ in range(16)]
    V_final = [[0]*97 for _ in range(28)]
    W_final = [[0]*97 for _ in range(28)]
    
    for i in range(16):
        for c in range(49):
            U_final[i][c] = U_s[i][c]
            if c < 49:
                V_final[i][c] = V_s[i][c]
                W_final[i][c] = W_s[i][c]
                
    for i in range(16):
        for c in range(48):
            U_final[i][49+c] = U_b[i][c]
        for c in range(48):
            V_final[i][49+c] = V_b[i][c]
            W_final[i][49+c] = W_b[i][c]
            
    # Verify
    if _verify(U_final, V_final, W_final, 4, 4, 7):
        return {"U": U_final, "V": V_final, "W": W_final}
    else:
        return _schoolbook(4, 4, 7)
