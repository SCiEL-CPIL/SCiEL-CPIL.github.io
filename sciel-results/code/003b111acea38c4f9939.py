"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Recursive Strassen decomposition for 4x4 matrix multiplication, rank 49."""
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4)
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # (16, 49)
    V = [[0] * rank for _ in range(m * p)]  # (16, 49)
    W = [[0] * rank for _ in range(p * n)]  # (16, 49)
    
    deadline = time.time() + 120
    
    # Define Strassen 2x2 multiplication terms
    # For 2x2 matrices A, B, C:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For the inner 2x2 multiplication, each "scalar" is actually a linear combination
    # of the actual scalar entries.
    
    # Outer Strassen: 4x4 as 2x2 blocks of 2x2
    # A = [[A00, A01], [A10, A11]] where Aij are 2x2
    # B = [[B00, B01], [B10, B11]] where Bij are 2x2
    # C = [[C00, C01], [C10, C11]] where Cij are 2x2
    
    # Outer M1 = (A00+A11)(B00+B11)
    # Outer M2 = (A10+A11)B00
    # Outer M3 = A00(B01-B11)
    # Outer M4 = A11(B10-B00)
    # Outer M5 = (A00+A01)B11
    # Outer M6 = (A10-A00)(B00+B01)
    # Outer M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each outer Mi, it's a 2x2 matrix product. We compute it with inner Strassen.
    
    # Helper: add inner Strassen for a 2x2 product
    # X is a 2x2 matrix (linear combo of A entries), Y is 2x2 (linear combo of B entries)
    # We add 7 rank-1 terms for X*Y
    
    def a_idx(block_r, block_c, local_r, local_c):
        return (block_r * 2 + local_r) * m + (block_c * 2 + local_c)
    
    def b_idx(block_r, block_c, local_r, local_c):
        return (block_r * 2 + local_r) * p + (block_c * 2 + local_c)
    
    def w_idx(block_r, block_c, local_r, local_c):
        row = block_r * 2 + local_r
        col = block_c * 2 + local_c
        return col * n + row
    
    # For inner Strassen, X[i,j] and Y[i,j] are linear combinations of actual A/B entries
    # Represent as dict: (global_row, global_col) -> coeff
    
    # Define the 7 outer Strassen terms
    # Each term: X (2x2 of A-entry linear combos), Y (2x2 of B-entry linear combos)
    # X[i][j] = list of (a_global_row, a_global_col, coeff)
    # Y[i][j] = list of (b_global_row, b_global_col, coeff)
    
    # A block (br, bc): rows br*2 to br*2+1, cols bc*2 to bc*2+1
    # A00: block (0,0), A01: block (0,1), A10: block (1,0), A11: block (1,1)
    # B00: block (0,0), B01: block (0,1), B10: block (1,0), B11: block (1,1)
    
    def block_entries(block_r, block_c):
        """Return list of (global_row, global_col) for a 2x2 block"""
        return [(block_r*2+i, block_c*2+j) for i in range(2) for j in range(2)]
    
    # For a linear combination of blocks, represent as list of (global_idx, coeff)
    # where global_idx = row*4+col for A, or row*4+col for B
    
    # Outer M1: X = A00 + A11, Y = B00 + B11
    # X[i][j] = A00[i][j] + A11[i][j]
    # A00[i][j] is at (i, j), A11[i][j] is at (2+i, 2+j)
    
    # Let me define X and Y for each outer term as 2x2 arrays of lists of (a_idx, coeff)
    
    outer_terms = []
    
    # M1: X = A00+A11, Y = B00+B11
    X = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(i, j, 1), (2+i, 2+j, 1)])
        X.append(row)
    Y = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(i, j, 1), (2+i, 2+j, 1)])
        Y.append(row)
    outer_terms.append((X, Y))
    
    # M2: X = A10+A11, Y = B00
    X = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(2+i, j, 1), (2+i, 2+j, 1)])
        X.append(row)
    Y = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(i, j, 1)])
        Y.append(row)
    outer_terms.append((X, Y))
    
    # M3: X = A00, Y = B01-B11
    X = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(i, j, 1)])
        X.append(row)
    Y = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(i, 2+j, 1), (2+i, 2+j, -1)])
        Y.append(row)
    outer_terms.append((X, Y))
    
    # M4: X = A11, Y = B10-B00
    X = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(2+i, 2+j, 1)])
        X.append(row)
    Y = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(2+i, j, 1), (i, j, -1)])
        Y.append(row)
    outer_terms.append((X, Y))
    
    # M5: X = A00+A01, Y = B11
    X = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(i, j, 1), (i, 2+j, 1)])
        X.append(row)
    Y = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(2+i, 2+j, 1)])
        Y.append(row)
    outer_terms.append((X, Y))
    
    # M6: X = A10-A00, Y = B00+B01
    X = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(2+i, j, 1), (i, j, -1)])
        X.append(row)
    Y = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(i, j, 1), (i, 2+j, 1)])
        Y.append(row)
    outer_terms.append((X, Y))
    
    # M7: X = A01-A11, Y = B10+B11
    X = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(i, 2+j, 1), (2+i, 2+j, -1)])
        X.append(row)
    Y = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(2+i, j, 1), (2+i, 2+j, 1)])
        Y.append(row)
    outer_terms.append((X, Y))
    
    # C targets for each outer term
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    c_targets = [
        [(0,0,1), (1,1,1)],       # M1 -> C00, C11
        [(1,0,1), (1,1,-1)],      # M2 -> C10, C11
        [(0,1,1), (1,1,1)],       # M3 -> C01, C11
        [(0,0,1), (1,0,1)],       # M4 -> C00, C10
        [(0,0,-1), (0,1,1)],      # M5 -> C00, C01
        [(1,1,1)],                 # M6 -> C11
        [(0,0,1)],                 # M7 -> C00
    ]
    
    # For each outer term, apply inner Strassen to compute X*Y
    col = 0
    for t in range(7):
        X = outer_terms[t][0]
        Y = outer_terms[t][1]
        
        # Inner Strassen: compute X*Y using 7 multiplications
        # Inner M1 = (X00+X11)(Y00+Y11)
        # Inner M2 = (X10+X11)Y00
        # Inner M3 = X00(Y01-Y11)
        # Inner M4 = X11(Y10-Y00)
        # Inner M5 = (X00+X01)Y11
        # Inner M6 = (X10-X00)(Y00+Y01)
        # Inner M7 = (X01-X11)(Y10+Y11)
        # Z00 = M1 + M4 - M5 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        
        # For each inner term, X[i][j] is a list of (a_row, a_col, coeff)
        # We need to form linear combinations for the inner Strassen
        
        # Helper: add two linear combinations
        def add_combos(c1, c2):
            result = {}
            for (r, c, v) in c1:
                result[(r,c)] = result.get((r,c), 0) + v
            for (r, c, v) in c2:
                result[(r,c)] = result.get((r,c), 0) + v
            return [(r, c, v) for (r,c), v in result.items() if v != 0]
        
        # Helper: subtract
        def sub_combos(c1, c2):
            return add_combos(c1, [(r, c, -v) for (r,c,v) in c2])
        
        # Inner M1: (X00+X11)(Y00+Y11)
        X_in = add_combos(X[0][0], X[1][1])
        Y_in = add_combos(Y[0][0], Y[1][1])
        # Z00 += 1, Z11 += 1
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c][col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c][col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 0, 0)][col] += coeff
            W[w_idx(br, bc, 1, 1)][col] += coeff
        col += 1
        
        # Inner M2: (X10+X11)Y00
        X_in = add_combos(X[1][0], X[1][1])
        Y_in = Y[0][0]
        # Z10 += 1, Z11 -= 1
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c][col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c][col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 1, 0)][col] += coeff
            W[w_idx(br, bc, 1, 1)][col] -= coeff
        col += 1
        
        # Inner M3: X00(Y01-Y11)
        X_in = X[0][0]
        Y_in = sub_combos(Y[0][1], Y[1][1])
        # Z01 += 1, Z11 += 1
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c][col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c][col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 0, 1)][col] += coeff
            W[w_idx(br, bc, 1, 1)][col] += coeff
        col += 1
        
        # Inner M4: X11(Y10-Y00)
        X_in = X[1][1]
        Y_in = sub_combos(Y[1][0], Y[0][0])
        # Z00 += 1, Z10 += 1
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c][col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c][col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 0, 0)][col] += coeff
            W[w_idx(br, bc, 1, 0)][col] += coeff
        col += 1
        
        # Inner M5: (X00+X01)Y11
        X_in = add_combos(X[0][0], X[0][1])
        Y_in = Y[1][1]
        # Z00 -= 1, Z01 += 1
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c][col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c][col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 0, 0)][col] -= coeff
            W[w_idx(br, bc, 0, 1)][col] += coeff
        col += 1
        
        # Inner M6: (X10-X00)(Y00+Y01)
        X_in = sub_combos(X[1][0], X[0][0])
        Y_in = add_combos(Y[0][0], Y[0][1])
        # Z11 += 1
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c][col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c][col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 1, 1)][col] += coeff
        col += 1
        
        # Inner M7: (X01-X11)(Y10+Y11)
        X_in = sub_combos(X[0][1], X[1][1])
        Y_in = add_combos(Y[1][0], Y[1][1])
        # Z00 += 1
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c][col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c][col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 0, 0)][col] += coeff
        col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify tensor identity before attempting reduction
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Tensor T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # For matrix mult: T[a_row*m+a_col, b_row*p+b_col, c_col*n+c_row] = delta
    # Check a few random entries
    def tensor_entry(ar, ac, br, bc):
        """Should be 1 if (ar,ac) in A multiplies (br,bc) in B to give C[ar,bc]"""
        i = ar * m + ac
        j = br * p + bc
        k = bc * n + ar  # W index: col*n + row
        val = 0.0
        for r in range(rank):
            val += U_np[i, r] * V_np[j, r] * W_np[k, r]
        return val
    
    # Quick verification
    test_pass = True
    for ar in range(4):
        for ac in range(4):
            for br in range(4):
                for bc in range(4):
                    expected = 1.0 if ac == br else 0.0
                    actual = tensor_entry(ar, ac, br, bc)
                    if abs(actual - expected) > 1e-10:
                        test_pass = False
                        break
                if not test_pass:
                    break
            if not test_pass:
                break
        if not test_pass:
            break
    
    if not test_pass:
        # Return as-is if verification fails
        return {"U": U, "V": V, "W": W}
    
    if time.time() > deadline - 30:
        return {"U": U, "V": V, "W": W}
    
    # Attempt rank reduction by trying to remove columns one at a time
    # and verifying the tensor identity still holds.
    
    def verify_decomposition(U_mat, V_mat, W_mat):
        """Verify tensor identity: sum_r U[i,r]*V[j,r]*W[k,r] = delta_{i%4, j//4}"""
        # i = ar*4 + ac, j = br*4 + bc, k = bc*4 + ar
        # Expected: 1 if ac == br, else 0
        for ar in range(4):
            for ac in range(4):
                for br in range(4):
                    expected = 1.0 if ac == br else 0.0
                    i = ar * m + ac
                    j = br * p + ac  # j = br*4 + bc where bc varies... wait
                    # Actually j = br*p + bc, and we need to check all bc
                    pass
        # Full check
        for ar in range(4):
            for ac in range(4):
                for br in range(4):
                    for bc in range(4):
                        expected = 1.0 if ac == br else 0.0
                        i = ar * m + ac
                        j = br * p + bc
                        k = bc * n + ar
                        val = np.sum(U_mat[i, :] * V_mat[j, :] * W_mat[k, :])
                        if abs(val - expected) > 1e-9:
                            return False
        return True
    
    # Try removing columns greedily with fast verification
    current_U = U_np.copy()
    current_V = V_np.copy()
    current_W = W_np.copy()
    current_rank = rank
    
    def fast_verify(U_mat, V_mat, W_mat):
        """Fast verification using batched matrix ops.
        For each (ar, bc): M[ac, br] = sum_r U[ar*4+ac,r]*V[br*4+bc,r]*W[bc*4+ar,r]
        should equal I_4.
        """
        R = U_mat.shape[1]
        for ar in range(4):
            # U rows for this ar: indices ar*4 .. ar*4+3
            U_ar = U_mat[ar*4:(ar+1)*4, :]  # 4 x R
            for bc in range(4):
                # W vector: W[bc*4+ar, :]
                W_vec = W_mat[bc*4+ar, :]  # R
                # V rows: V[br*4+bc, :] for br=0..3 -> rows bc, 4+bc, 8+bc, 12+bc
                V_bc = V_mat[bc::4, :]  # 4 x R
                # M[ac, br] = sum_r U_ar[ac,r] * W_vec[r] * V_bc[br,r]
                # = (U_ar * W_vec[None,:]) @ V_bc.T
                M = (U_ar * W_vec[None, :]) @ V_bc.T  # 4 x 4
                if not np.allclose(M, np.eye(4), atol=1e-9):
                    return False
        return True
    
    # Greedy removal: try multiple passes
    improved = True
    while improved:
        improved = False
        for col_to_try in range(current_rank - 1, -1, -1):
            if time.time() > deadline - 15:
                break
            
            # Try removing this column
            keep_cols = [c for c in range(current_rank) if c != col_to_try]
            U_new = current_U[:, keep_cols]
            V_new = current_V[:, keep_cols]
            W_new = current_W[:, keep_cols]
            
            if fast_verify(U_new, V_new, W_new):
                current_U = U_new
                current_V = V_new
                current_W = W_new
                current_rank -= 1
                improved = True
                break  # restart scan from top
    
    if current_rank < rank:
        U_final = current_U.tolist()
        V_final = current_V.tolist()
        W_final = current_W.tolist()
        return {"U": U_final, "V": V_final, "W": W_final}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
