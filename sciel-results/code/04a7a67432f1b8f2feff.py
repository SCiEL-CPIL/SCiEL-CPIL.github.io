"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Recursive Strassen decomposition for 4x4 matrix multiplication.
    Rank: 49 (optimal for 4x4 matrix multiplication).
    
    The 4x4 multiplication is decomposed as a 2x2 block structure where each
    block is 2x2. The outer level uses Strassen (rank 7) and the inner level
    also uses Strassen (rank 7), giving total rank 49.
    
    U: (16, 49) - linear combinations of A entries
    V: (16, 49) - linear combinations of B entries  
    W: (16, 49) - linear combinations for C entries (trace convention W[k*n+i, r])
    """
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4)
    
    _t0 = time.time()
    _DEADLINE = 120.0
    _MARGIN = 5.0
    
    rank = 49
    U = np.zeros((n * m, rank))  # (16, 49)
    V = np.zeros((m * p, rank))  # (16, 49)
    W = np.zeros((p * n, rank))  # (16, 49)
    
    # W index: W[k*n + i, r] where C[i, k] is the output element
    def w_idx(i, k):
        """Convert (row i, col k) of C to W index: k*n + i"""
        return k * n + i
    
    # A entry index: row * m + col = row * 4 + col
    # B entry index: row * p + col = row * 4 + col
    
    # Define outer Strassen terms for 2x2 block structure
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # Each block is 2x2.
    
    # For each outer term t, X_t[i][j] and Y_t[i][j] are 2x2 matrices
    # where each entry is a linear combination of A or B entries.
    # Represent each entry as dict: (global_row, global_col) -> coeff
    
    def make_block(br, bc, sign=1):
        """Create a 2x2 matrix representing block (br,bc) with given sign.
        Each entry is a dict: (global_row, global_col) -> coeff"""
        mat = []
        for i in range(2):
            row = []
            for j in range(2):
                gr = br * 2 + i
                gc = bc * 2 + j
                row.append({(gr, gc): sign})
            mat.append(row)
        return mat
    
    def add_blocks(b1, b2):
        """Add two 2x2 block matrices (linear combos)"""
        result = []
        for i in range(2):
            row = []
            for j in range(2):
                d = {}
                for key, val in b1[i][j].items():
                    d[key] = d.get(key, 0) + val
                for key, val in b2[i][j].items():
                    d[key] = d.get(key, 0) + val
                row.append({k: v for k, v in d.items() if v != 0})
            result.append(row)
        return result
    
    def sub_blocks(b1, b2):
        """Subtract two 2x2 block matrices"""
        neg_b2 = [[{k: -v for k, v in b2[i][j].items()} for j in range(2)] for i in range(2)]
        return add_blocks(b1, neg_b2)
    
    def add_entries(mat, i1, j1, i2, j2):
        """Create 2x2 matrix with only entry (i1,j1) = mat[i1][j1] + mat[i2][j2]"""
        result = [[{} for _ in range(2)] for _ in range(2)]
        d = {}
        for key, val in mat[i1][j1].items():
            d[key] = d.get(key, 0) + val
        for key, val in mat[i2][j2].items():
            d[key] = d.get(key, 0) + val
        result[i1][j1] = {k: v for k, v in d.items() if v != 0}
        return result
    
    def sub_entries(mat, i1, j1, i2, j2):
        """Create 2x2 matrix with only entry (i1,j1) = mat[i1][j1] - mat[i2][j2]"""
        result = [[{} for _ in range(2)] for _ in range(2)]
        d = {}
        for key, val in mat[i1][j1].items():
            d[key] = d.get(key, 0) + val
        for key, val in mat[i2][j2].items():
            d[key] = d.get(key, 0) - val
        result[i1][j1] = {k: v for k, v in d.items() if v != 0}
        return result
    
    def single_entry(mat, i, j):
        """Create 2x2 matrix with only entry (i,j) = mat[i][j]"""
        result = [[{} for _ in range(2)] for _ in range(2)]
        result[i][j] = dict(mat[i][j])
        return result
    
    # Define the 7 outer Strassen terms
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    A00 = make_block(0, 0)
    A01 = make_block(0, 1)
    A10 = make_block(1, 0)
    A11 = make_block(1, 1)
    
    B00 = make_block(0, 0)
    B01 = make_block(0, 1)
    B10 = make_block(1, 0)
    B11 = make_block(1, 1)
    
    outer_X = [
        add_blocks(A00, A11),       # M1
        add_blocks(A10, A11),       # M2
        A00,                        # M3
        A11,                        # M4
        add_blocks(A00, A01),       # M5
        sub_blocks(A10, A00),       # M6
        sub_blocks(A01, A11),       # M7
    ]
    outer_Y = [
        add_blocks(B00, B11),       # M1
        B00,                        # M2
        sub_blocks(B01, B11),       # M3
        sub_blocks(B10, B00),       # M4
        B11,                        # M5
        add_blocks(B00, B01),       # M6
        add_blocks(B10, B11),       # M7
    ]
    
    # C targets: how each outer term contributes to C blocks
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    c_targets = [
        [(0, 0, 1), (1, 1, 1)],     # M1 -> C00(+1), C11(+1)
        [(1, 0, 1), (1, 1, -1)],    # M2 -> C10(+1), C11(-1)
        [(0, 1, 1), (1, 1, 1)],     # M3 -> C01(+1), C11(+1)
        [(0, 0, 1), (1, 0, 1)],     # M4 -> C00(+1), C10(+1)
        [(0, 0, -1), (0, 1, 1)],    # M5 -> C00(-1), C01(+1)
        [(1, 1, 1)],                 # M6 -> C11(+1)
        [(0, 0, 1)],                 # M7 -> C00(+1)
    ]
    
    # Inner Strassen targets: how each inner term contributes to Z (2x2 result)
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    inner_targets = [
        [(0, 0, 1), (1, 1, 1)],     # M1 -> Z00(+1), Z11(+1)
        [(1, 0, 1), (1, 1, -1)],    # M2 -> Z10(+1), Z11(-1)
        [(0, 1, 1), (1, 1, 1)],     # M3 -> Z01(+1), Z11(+1)
        [(0, 0, 1), (1, 0, 1)],     # M4 -> Z00(+1), Z10(+1)
        [(0, 0, -1), (0, 1, 1)],    # M5 -> Z00(-1), Z01(+1)
        [(1, 1, 1)],                 # M6 -> Z11(+1)
        [(0, 0, 1)],                 # M7 -> Z00(+1)
    ]
    
    # For inner Strassen, we need to form the 7 inner terms
    # Given X (2x2 of linear combos) and Y (2x2 of linear combos):
    # Inner M1: (X00+X11)(Y00+Y11)
    # Inner M2: (X10+X11)Y00
    # Inner M3: X00(Y01-Y11)
    # Inner M4: X11(Y10-Y00)
    # Inner M5: (X00+X01)Y11
    # Inner M6: (X10-X00)(Y00+Y01)
    # Inner M7: (X01-X11)(Y10+Y11)
    
    col = 0
    for t in range(7):
        X = outer_X[t]
        Y = outer_Y[t]
        
        # Compute the 7 inner Strassen factors
        # Inner M1: (X00+X11)(Y00+Y11)
        X1 = add_entries(X, 0, 0, 1, 1)
        Y1 = add_entries(Y, 0, 0, 1, 1)
        
        # Inner M2: (X10+X11)Y00
        X2 = add_entries(X, 1, 0, 1, 1)
        Y2 = single_entry(Y, 0, 0)
        
        # Inner M3: X00(Y01-Y11)
        X3 = single_entry(X, 0, 0)
        Y3 = sub_entries(Y, 0, 1, 1, 1)
        
        # Inner M4: X11(Y10-Y00)
        X4 = single_entry(X, 1, 1)
        Y4 = sub_entries(Y, 1, 0, 0, 0)
        
        # Inner M5: (X00+X01)Y11
        X5 = add_entries(X, 0, 0, 0, 1)
        Y5 = single_entry(Y, 1, 1)
        
        # Inner M6: (X10-X00)(Y00+Y01)
        X6 = sub_entries(X, 1, 0, 0, 0)
        Y6 = add_entries(Y, 0, 0, 0, 1)
        
        # Inner M7: (X01-X11)(Y10+Y11)
        X7 = sub_entries(X, 0, 1, 1, 1)
        Y7 = add_entries(Y, 1, 0, 1, 1)
        
        inner_factors = [
            (X1, Y1),
            (X2, Y2),
            (X3, Y3),
            (X4, Y4),
            (X5, Y5),
            (X6, Y6),
            (X7, Y7),
        ]
        
        for k in range(7):
            Xf, Yf = inner_factors[k]
            
            # Fill U from Xf
            for i in range(2):
                for j in range(2):
                    for (a_r, a_c), a_v in Xf[i][j].items():
                        U[a_r * m + a_c, col] += a_v
            
            # Fill V from Yf
            for i in range(2):
                for j in range(2):
                    for (b_r, b_c), b_v in Yf[i][j].items():
                        V[b_r * p + b_c, col] += b_v
            
            # Fill W: for each outer target and inner target
            for (br, bc, c_coeff) in c_targets[t]:
                for (li, lj, z_coeff) in inner_targets[k]:
                    global_i = br * 2 + li
                    global_k = bc * 2 + lj
                    W[w_idx(global_i, global_k), col] += c_coeff * z_coeff
            
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify with numpy if time permits
    if time.time() < _DEADLINE - _MARGIN:
        np.random.seed(42)
        A_test = np.random.randn(4, 4)
        B_test = np.random.randn(4, 4)
        C_expected = A_test @ B_test
        
        # C[i,k] = sum_r (sum_j U[i*4+j,r]*A[i,j]) * (sum_j V[j*4+k,r]*B[j,k]) * W[k*4+i,r]
        # Vectorized: for each r, compute u_r = U[:,r] @ A_flat, v_r = V[:,r] @ B_flat
        A_flat = A_test.flatten()
        B_flat = B_test.flatten()
        
        # U is (16, 49), A_flat is (16,) -> U @ A_flat gives (49,)
        # But we need: for each r, sum_j U[i*4+j, r] * A[i,j] which is NOT U @ A_flat
        # because U[i*4+j, r] multiplies A[i,j], not A_flat[i*4+j]
        # Actually A_flat[i*4+j] = A[i,j], so U @ A_flat[r] = sum_j U[i*4+j, r] * A_flat[i*4+j]
        # = sum_j U[i*4+j, r] * A[i,j] -- but this sums over ALL j for ALL i, not per-i!
        
        # We need per-row: for each i, u_i_r = sum_j U[i*4+j, r] * A[i,j]
        # This is: reshape U to (4, 4, 49), then for each r: sum over axis 1 of U_res * A[:, :, None]
        
        U_r = U.reshape(4, 4, 49)  # [i, j, r]
        V_r = V.reshape(4, 4, 49)  # [j, k, r]
        W_r = W.reshape(4, 4, 49)  # [k, i, r] -- W[k*4+i, r]
        
        # u[i, r] = sum_j U[i, j, r] * A[i, j]
        # = np.einsum('ijr,ij->ir', U_r, A_test)
        u_vals = np.einsum('ijr,ij->ir', U_r, A_test)  # (4, 49)
        v_vals = np.einsum('jkr,jk->kr', V_r, B_test)  # (4, 49)
        
        # C[i,k] = sum_r u[i,r] * v[k,r] * W[k,i,r]
        # W is indexed as W[k*4+i, r], so W_r[k, i, r] = W[k*4+i, r]
        C_computed = np.einsum('ir,kr,kir->ik', u_vals, v_vals, W_r)
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Decomposition is incorrect
            pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
