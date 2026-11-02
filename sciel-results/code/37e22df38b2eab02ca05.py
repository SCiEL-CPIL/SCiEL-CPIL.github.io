import numpy as np

def construct():
    n, m, p = 2, 3, 9
    
    # We will build U, V, W as lists of columns.
    # Each column corresponds to one term in the decomposition.
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vals, v_vals, w_vals):
        """
        u_vals: dict mapping (i, j) to coefficient for A[i, j]
        v_vals: dict mapping (j, k) to coefficient for B[j, k]
        w_vals: dict mapping (k, i) to coefficient for C[k, i] (output index k*n+i)
        """
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        
        for (i, j), val in u_vals.items():
            u_vec[i * m + j] = val
        for (j, k), val in v_vals.items():
            v_vec[j * p + k] = val
        for (k, i), val in w_vals.items():
            w_vec[k * n + i] = val
            
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # Helper to add Strassen 2x2x2 decomposition
    # A is 2x2, B is 2x2
    # A indices: (i, j) in {0,1}x{0,1}
    # B indices: (j, k) in {0,1}x{0,1}
    # C indices: (k, i) in {0,1}x{0,1}
    # Note: The global indices are offset.
    # We assume this block is at A[:, a0:a0+2] and B[b0:b0+2, k0:k0+2]
    # And output C[:, k0:k0+2]
    
    def add_strassen(a_row_offset, b_row_offset, b_col_offset):
        # A[i, a_row_offset + j]
        # B[b_row_offset + j, b_col_offset + k]
        # C[i, b_col_offset + k] -> W index (b_col_offset + k)*n + i
        
        # M1 = (A00 + A11)(B00 + B11)
        # A00 is (0, a0), A11 is (1, a0+1)
        # B00 is (b0, b0c), B11 is (b0+1, b0c+1)
        
        a0, a1 = a_row_offset, a_row_offset + 1
        b0, b1 = b_row_offset, b_row_offset + 1
        c0, c1 = b_col_offset, b_col_offset + 1
        
        # M1
        add_term(
            {(0, a0): 1, (1, a1): 1},
            {(b0, c0): 1, (b1, c1): 1},
            {(c0, 0): 1, (c0, 1): 1, (c1, 0): 1, (c1, 1): 1} # Wait, C indices are (k, i)
            # C00 (k=0, i=0) gets +1 from M1?
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            # So M1 contributes to C00 and C11.
            # C00 index: k=0, i=0 -> 0*2+0=0.
            # C11 index: k=1, i=1 -> 1*2+1=3.
            # But we are adding terms to the global W.
            # The term for M1 should contribute to the final C.
            # So in the term for M1, W should be 1 at positions corresponding to C00 and C11.
            # However, the "term" is a rank-1 update.
            # The product U*V gives the intermediate M1.
            # Then W must route M1 to the correct output entries.
            # So for the term corresponding to M1:
            # U encodes (A00+A11)
            # V encodes (B00+B11)
            # W must encode the coefficients of M1 in the output.
            # M1 is added to C00 (coeff 1) and C11 (coeff 1).
            # So W should be 1 at C00 and C11.
        )
        # Let's refine the add_term logic.
        # The term is U * V * W.
        # The sum of U*V over c gives the intermediate products.
        # Actually, the decomposition is T = sum U_c V_c W_c.
        # For Strassen, we define 7 intermediate products.
        # Each product corresponds to a term.
        # For M1:
        # U1: 1 at A00, A11.
        # V1: 1 at B00, B11.
        # W1: 1 at C00, C11.
        # This term contributes 1 * 1 * 1 = 1 to C00 and C11.
        # This is correct.
        
        # Let's re-implement add_term to take explicit lists or dicts.
        # I will rewrite the function to be more flexible.
        
        pass

    # Reset
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_dict, v_dict, w_dict):
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        
        for (i, j), val in u_dict.items():
            u_vec[i * m + j] = val
        for (j, k), val in v_dict.items():
            v_vec[j * p + k] = val
        for (k, i), val in w_dict.items():
            w_vec[k * n + i] = val
            
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    def add_strassen(a_col_start, b_row_start, b_col_start):
        # A is 2x2 at A[:, a_col_start : a_col_start+2]
        # B is 2x2 at B[b_row_start : b_row_start+2, b_col_start : b_col_start+2]
        # C is 2x2 at C[:, b_col_start : b_col_start+2]
        
        ac0 = a_col_start
        ac1 = a_col_start + 1
        
        br0 = b_row_start
        br1 = b_row_start + 1
        
        bc0 = b_col_start
        bc1 = b_col_start + 1
        
        # M1 = (A00 + A11)(B00 + B11)
        # A00=(0,ac0), A11=(1,ac1)
        # B00=(br0,bc0), B11=(br1,bc1)
        # C00=(bc0,0), C11=(bc1,1)
        add_term(
            {(0, ac0): 1, (1, ac1): 1},
            {(br0, bc0): 1, (br1, bc1): 1},
            {(bc0, 0): 1, (bc1, 1): 1}
        )
        
        # M2 = (A10 + A11)B00
        # A10=(1,ac0), A11=(1,ac1)
        # B00=(br0,bc0)
        # C10=(bc0,1)
        add_term(
            {(1, ac0): 1, (1, ac1): 1},
            {(br0, bc0): 1},
            {(bc0, 1): 1}
        )
        
        # M3 = A00(B01 - B11)
        # A00=(0,ac0)
        # B01=(br0,bc1), B11=(br1,bc1)
        # C01=(bc1,0)
        add_term(
            {(0, ac0): 1},
            {(br0, bc1): 1, (br1, bc1): -1},
            {(bc1, 0): 1}
        )
        
        # M4 = A11(B10 - B00)
        # A11=(1,ac1)
        # B10=(br1,bc0), B00=(br0,bc0)
        # C00=(bc0,0), C10=(bc0,1)
        add_term(
            {(1, ac1): 1},
            {(br1, bc0): 1, (br0, bc0): -1},
            {(bc0, 0): 1, (bc0, 1): 1}
        )
        
        # M5 = (A00 + A01)B11
        # A00=(0,ac0), A01=(0,ac1)
        # B11=(br1,bc1)
        # C00=(bc0,0), C01=(bc1,0)
        add_term(
            {(0, ac0): 1, (0, ac1): 1},
            {(br1, bc1): 1},
            {(bc0, 0): -1, (bc1, 0): 1} # M5 is -1 in C00, +1 in C01
        )
        
        # M6 = (A10 - A00)(B00 + B01)
        # A10=(1,ac0), A00=(0,ac0)
        # B00=(br0,bc0), B01=(br0,bc1)
        # C11=(bc1,1)
        add_term(
            {(1, ac0): 1, (0, ac0): -1},
            {(br0, bc0): 1, (br0, bc1): 1},
            {(bc1, 1): 1}
        )
        
        # M7 = (A01 - A11)(B10 + B11)
        # A01=(0,ac1), A11=(1,ac1)
        # B10=(br1,bc0), B11=(br1,bc1)
        # C00=(bc0,0), C11=(bc1,1)
        add_term(
            {(0, ac1): 1, (1, ac1): -1},
            {(br1, bc0): 1, (br1, bc1): 1},
            {(bc0, 0): 1, (bc1, 1): 1}
        )

    def add_outer_2x1x2(a_col_idx, b_row_idx, b_col_start):
        # A is 2x1 at A[:, a_col_idx]
        # B is 1x2 at B[b_row_idx, b_col_start : b_col_start+2]
        # C is 2x2 at C[:, b_col_start : b_col_start+2]
        # Product is outer product.
        # C[i, k] = A[i, a] * B[b, b_col_start + k]
        # Term: U[i, a]=1, V[b, b_col_start+k]=1, W[b_col_start+k, i]=1
        
        bc0 = b_col_start
        bc1 = b_col_start + 1
        
        # We can do this in 1 term?
        # U: 1 at (0, a), (1, a)
        # V: 1 at (b, bc0), (b, bc1)
        # W: 1 at (bc0, 0), (bc0, 1), (bc1, 0), (bc1, 1)
        # Check:
        # C[0, bc0] = U[0,a] V[b,bc0] W[bc0,0] = 1*1*1 = 1. (Should be A[0,a]B[b,bc0])
        # Wait, the term value is U*V*W.
        # If U=1, V=1, W=1, the term is 1.
        # But we need A[i,a] * B[b,k].
        # The tensor decomposition sums U*V*W.
        # If we set U[i,a]=1, V[b,k]=1, W[k,i]=1.
        # Then sum is 1.
        # But we need the product of the input values.
        # The input values are "extracted" by the indices.
        # The term is a rank-1 tensor.
        # $T_{i,j,k} = \sum U_{i,j,c} V_{j,k,c} W_{k,i,c}$.
        # If we set $U_{0,a,c}=1, U_{1,a,c}=1$, $V_{b,bc0,c}=1, V_{b,bc1,c}=1$, $W_{bc0,0,c}=1, W_{bc0,1,c}=1, W_{bc1,0,c}=1, W_{bc1,1,c}=1$.
        # Then for output $(i, k)$, the contribution is $U_{i,a,c} V_{b,k,c} W_{k,i,c}$.
        # For $i=0, k=bc0$: $1 \cdot 1 \cdot 1 = 1$.
        # This term adds 1 to $C[0, bc0]$.
        # But $C[0, bc0]$ should be $A[0,a] B[b, bc0]$.
        # The decomposition works because the "1"s in U, V, W select the specific entries of A and B.
        # So yes, 1 term is sufficient for the outer product.
        
        add_term(
            {(0, a_col_idx): 1, (1, a_col_idx): 1},
            {(b_row_idx, bc0): 1, (b_row_idx, bc1): 1},
            {(bc0, 0): 1, (bc0, 1): 1, (bc1, 0): 1, (bc1, 1): 1}
        )

    def add_outer_2x2x1(a_col_start, b_row_start, b_col_idx):
        # A is 2x2 at A[:, a_col_start : a_col_start+2]
        # B is 2x1 at B[b_row_start : b_row_start+2, b_col_idx]
        # C is 2x1 at C[:, b_col_idx]
        # C[i, k] = sum_j A[i, j] B[j, k]
        # Here k is fixed.
        # This is a dot product for each row i.
        # Can be done in 1 term?
        # U[i, j]=1, V[j, k]=1, W[k, i]=1.
        # Contribution to C[i, k] is 1.
        # But we need sum_j A[i,j] B[j,k].
        # If we use 1 term, we get 1.
        # We need to sum over j.
        # So we need 2 terms? One for j=0, one for j=1?
        # Term 1: U[i,0]=1, V[0,k]=1, W[k,i]=1.
        # Term 2: U[i,1]=1, V[1,k]=1, W[k,i]=1.
        # Sum: 2? No.
        # Term 1 contributes 1 to C[i,k].
        # Term 2 contributes 1 to C[i,k].
        # Total 2.
        # But we want A[i,0]B[0,k] + A[i,1]B[1,k].
        # The "1"s in U and V select the entries.
        # So Term 1 effectively computes A[i,0]B[0,k].
        # Term 2 computes A[i,1]B[1,k].
        # So yes, 2 terms are needed for 2x2x1.
        
        ac0 = a_col_start
        ac1 = a_col_start + 1
        br0 = b_row_start
        br1 = b_row_start + 1
        
        # Term for j=0
        add_term(
            {(0, ac0): 1, (1, ac0): 1},
            {(br0, b_col_idx): 1},
            {(b_col_idx, 0): 1, (b_col_idx, 1): 1}
        )
        # Term for j=1
        add_term(
            {(0, ac1): 1, (1, ac1): 1},
            {(br1, b_col_idx): 1},
            {(b_col_idx, 0): 1, (b_col_idx, 1): 1}
        )

    def add_scalar_2x1x1(a_col_idx, b_row_idx, b_col_idx):
        # A is 2x1, B is 1x1, C is 2x1
        # C[i, k] = A[i, a] B[b, k]
        # 1 term.
        add_term(
            {(0, a_col_idx): 1, (1, a_col_idx): 1},
            {(b_row_idx, b_col_idx): 1},
            {(b_col_idx, 0): 1, (b_col_idx, 1): 1}
        )

    # Now assemble for 2x3x9
    # A is 2x3. Columns 0,1,2.
    # B is 3x9. Rows 0,1,2. Columns 0..8.
    # We split B into 3 blocks of 3 columns: 0-2, 3-5, 6-8.
    
    for block in range(3):
        b_col_start = block * 3
        
        # Block B_sub is B[0:3, b_col_start : b_col_start+3]
        # Split B_sub into:
        # B_TL: B[0:2, b_col_start : b_col_start+2]
        # B_TR: B[0:2, b_col_start+2 : b_col_start+3]
        # B_BL: B[2, b_col_start : b_col_start+2]
        # B_BR: B[2, b_col_start+2 : b_col_start+3]
        
        # A_L: A[:, 0:2]
        # a_R: A[:, 2]
        
        # 1. A_L B_TL (2x2x2)
        add_strassen(0, 0, b_col_start)
        
        # 2. a_R B_BL (2x1x2)
        add_outer_2x1x2(2, 2, b_col_start)
        
        # 3. A_L B_TR (2x2x1)
        add_outer_2x2x1(0, 0, b_col_start + 2)
        
        # 4. a_R B_BR (2x1x1)
        add_scalar_2x1x1(2, 2, b_col_start + 2)

    rank = len(U_cols)
    U = [list(col) for col in zip(*U_cols)] if U_cols else []
    V = [list(col) for col in zip(*V_cols)] if V_cols else []
    W = [list(col) for col in zip(*W_cols)] if W_cols else []
    
    # Ensure shapes are correct
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    return {"U": U, "V": V, "W": W}
