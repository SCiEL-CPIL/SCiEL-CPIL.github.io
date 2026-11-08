import numpy as np

# Dimensions: A (2x3), B (3x9), C (2x9)
N, M, P = 2, 3, 9

def construct():
    # Initialize factor matrices with zeros
    # U: (N*M, Rank)
    # V: (M*P, Rank)
    # W: (P*N, Rank)
    
    # We estimate the rank needed. 
    # Strassen for 2x2 * 2x2 takes 7.
    # We have 4 such blocks in the 2x2 * 2x9 part (pairs of columns).
    # 1 remaining 2x2 * 2x1 block (4 ops).
    # 1 2x1 * 1x9 block (18 ops).
    # Total = 4*7 + 4 + 18 = 28 + 4 + 18 = 50.
    RANK = 50
    
    U = [[0] * RANK for _ in range(N * M)]
    V = [[0] * RANK for _ in range(M * P)]
    W = [[0] * RANK for _ in range(P * N)]
    
    current_rank = 0
    
    def add_rank1(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        """Add a rank-1 contribution to the decomposition."""
        nonlocal current_rank
        U[u_idx][current_rank] += u_val
        V[v_idx][current_rank] += v_val
        W[w_idx][current_rank] += w_val
        current_rank += 1

    def apply_strassen_2x2(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        """
        Apply Strassen's algorithm for 2x2 matrix multiplication.
        Inputs are indices into U, V, W matrices.
        A = [[a00, a01], [a10, a11]]
        B = [[b00, b01], [b10, b11]]
        C = [[c00, c01], [c10, c11]]
        """
        # M1 = (a00 + a11) * (b00 + b11)
        # To avoid extra additions in U/V, we usually expand the terms.
        # However, Strassen reduces *multiplications*. 
        # In tensor decomposition, a "rank-1" term is a scalar product of a U-vector, V-vector, W-vector.
        # Standard Strassen uses 7 products of sums.
        # We can implement this by creating intermediate rank-1 terms for the sums? No, that increases rank.
        # The standard way to map Strassen to tensor decomposition is to use the 7 products directly as rank-1 terms,
        # where the U and V vectors contain the coefficients of the sums.
        
        # M1 = (A00 + A11)(B00 + B11)
        add_rank1(a00, b00, c00, 1, 1, 1)
        add_rank1(a00, b11, c00, 1, 1, 1)
        add_rank1(a11, b00, c00, 1, 1, 1)
        add_rank1(a11, b11, c00, 1, 1, 1)
        # Wait, this is 4 rank-1 terms for M1. That's not efficient.
        # Actually, in tensor decomposition, a term like (A00+A11)(B00+B11) contributes to multiple C entries.
        # Let's stick to the standard 7 multiplications.
        # The trick is that each "multiplication" in Strassen is a bilinear form.
        # M1 = (A00+A11)(B00+B11) is a single bilinear form.
        # In the U,V,W representation, a single bilinear form (X)(Y) where X is a linear combo of A's and Y is a linear combo of B's
        # corresponds to ONE rank-1 term in the decomposition of the tensor?
        # No. A rank-1 tensor is u v^T w^T.
        # If we have term M1 * C00, that's one rank-1.
        # But M1 is used in C00, C01, C10, C11.
        # So M1 contributes to 4 different rank-1 tensors?
        # Yes. The total rank is the sum of the ranks of the contributions.
        # If we calculate M1 as a "product", we need to represent it.
        # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
        # This requires 4 rank-1 terms if we don't share structure.
        # BUT, Strassen's insight is that we can compute the 7 intermediate values, and then combine them.
        # The "rank" of the multiplication tensor is the minimum number of rank-1 tensors.
        # Strassen shows that for 2x2, rank <= 7.
        # This means we can represent the 8 products using 7 rank-1 terms?
        # No, that's for the *algorithm* complexity (number of scalar multiplications).
        # Tensor rank is different.
        # For 2x2x2, the tensor rank is 7.
        # This means C00, C01, C10, C11 can be written as linear combinations of 7 rank-1 terms.
        
        # Let's define the 7 Strassen products explicitly as rank-1 contributions to the specific C entries.
        # This is tricky because a single Strassen product M1 affects multiple C entries.
        # In the U,V,W representation:
        # Term t: U_t, V_t, W_t.
        # Contribution to C_ij is sum_t U_t[i,j] * V_t[...]* W_t[...].
        # Actually, the standard mapping is:
        # The multiplication tensor T has entries T[i,j,k] = delta_{i,j+k} (for 1D). For 2D matrices, it's more complex.
        # Let's use the known rank-7 decomposition for 2x2.
        
        # The 7 products are:
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B10-B11)
        # P4 = A11(B01-B00)
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        
        # And the outputs:
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # To implement this with rank-1 terms, we need to represent each Pk as a rank-1 term?
        # No, Pk is a product of two linear combinations.
        # (A00+A11)(B00+B11) is NOT a rank-1 tensor in the (A,B,C) space. It's a product of two linear forms.
        # However, the *contribution* of Pk to a specific C entry is a rank-1 term?
        # Let's look at C00 = P1 + ...
        # P1 = (A00+A11)(B00+B11).
        # This term contributes to C00.
        # Can we represent the whole 2x2 multiplication with 7 rank-1 tensors?
        # Yes, that is the definition of tensor rank 7.
        # The 7 rank-1 tensors are not simply the 7 Strassen products mapped to 1 C entry.
        # The 7 rank-1 tensors correspond to the 7 Strassen products, but each one has non-zero entries in U, V, and W for ALL relevant indices.
        
        # Let's define the 7 basis vectors for U, V, W.
        # For each Strassen product k (1..7):
        # U_k has non-zeros at A indices involved.
        # V_k has non-zeros at B indices involved.
        # W_k has non-zeros at C indices involved, with coefficients from the reconstruction.
        
        # Example: P1 = (A00+A11)(B00+B11)
        # U_1: A00=1, A11=1
        # V_1: B00=1, B11=1
        # This product P1 is used in C00 (coeff 1), C01 (coeff 0), C10 (coeff 0), C11 (coeff 1).
        # Wait, C01 = P3+P5. P1 is not in C01.
        # C00 = P1 + P4 - P5 + P7
        # C11 = P1 - P2 + P3 + P6
        
        # So, for rank-1 term 1 (corresponding to P1):
        # U_1[0,0] = 1, U_1[1,1] = 1
        # V_1[0,0] = 1, V_1[1,1] = 1
        # W_1[0,0] = 1 (for C00)
        # W_1[1,1] = 1 (for C11)
        # Other W entries are 0.
        
        # Let's verify this logic.
        # Sum over k of U_k[i,j] V_k[j,k] W_k[k,l] ? No, indices are mixed.
        # The decomposition is C[i,l] = sum_k U_k[i,j] V_k[j,k] ... no.
        # The tensor is T[i, j, k] where i is row of A, j is col of A/row of B, k is col of B.
        # Wait, the standard matrix mult tensor is T_{i,j,k} = delta_{i,j} ... no.
        # C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # So the tensor is T_{i,j,k}.
        # A rank-1 decomposition is T_{i,j,k} = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} ... no, W is separate.
        # T_{i,j,k} = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)}? No, that's not right.
        # The factorization is:
        # C = sum_r (U_r * V_r) * W_r? No.
        # The interface is U (NM x R), V (MP x R), W (PN x R).
        # This implies:
        # C_{k, i} (flattened) = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}?
        # No, the indices in U, V, W are flattened.
        # U has size N*M. Index is i*M + j.
        # V has size M*P. Index is j*P + k.
        # W has size P*N. Index is k*N + i.
        
        # The reconstruction formula is:
        # C_{i, k} = sum_{j, r} A_{i, j} B_{j, k} ...
        # In the factorization:
        # C_{i, k} = sum_r ( sum_j U_{i*M+j, r} A_{i, j} ) * ( sum_j V_{j*P+k, r} B_{j, k} ) * W_{k*N+i, r} ?
        # No, that would be cubic.
        
        # Correct interpretation of U, V, W factors for bilinear maps:
        # The map is (A, B) -> C.
        # C_{i, k} = sum_{j, r} U_{i, j, r} V_{j, k, r} ...
        # But U is flattened (N*M, R). So U_{idx_u, r} where idx_u = i*M+j.
        # V is flattened (M*P, R). So V_{idx_v, r} where idx_v = j*P+k.
        # W is flattened (P*N, R). So W_{idx_w, r} where idx_w = k*N+i.
        
        # The standard decomposition for matrix multiplication is:
        # C_{i, k} = sum_{j, r} A_{i, j} B_{j, k}
        # This can be written as:
        # C_{i, k} = sum_r W_{k*N+i, r} * ( sum_j U_{i*M+j, r} A_{i, j} ) * ( sum_j V_{j*P+k, r} B_{j, k} ) ?
        # No, this structure implies that for a fixed r, the term is:
        # W_{k*N+i, r} * (linear combo of A) * (linear combo of B).
        # This is exactly a rank-1 term in the tensor T_{i,j,k} if we fix j?
        # No, j is summed over.
        
        # Let's check the Strassen rank-7 decomposition for 2x2.
        # It is known that the 2x2 matrix multiplication tensor has rank 7.
        # This means we can write:
        # C_{i, k} = sum_{r=1}^7 W_{i, k, r} ( sum_{j=0}^1 U_{i, j, r} A_{i, j} ) ( sum_{j=0}^1 V_{j, k, r} B_{j, k} )
        # Wait, the indices in U and V must be consistent.
        # Usually, the decomposition is:
        # C_{i, k} = sum_{r} ( sum_{j} U_{i, j, r} A_{i, j} ) ( sum_{j} V_{j, k, r} B_{j, k} ) W_{i, k, r}
        # Note that j is summed separately in U and V parts? No, that would be wrong.
        # The correct form for a bilinear map decomposition is:
        # C_{i, k} = sum_{r, j} U_{i, j, r} A_{i, j} V_{j, k, r} B_{j, k} ... no.
        
        # Let's look at the provided code structure.
        # U[u_a][col] += 1
        # V[v_e][col] += 1
        # W[w_00][col] += 1
        # This implies that the term for column `col` is:
        # Contribution to C[w_00] is U[u_a] * V[v_e] * W[w_00].
        # Here u_a is a specific index in A, v_e is a specific index in B.
        # This means the decomposition is of the form:
        # C_{target} = sum_{r} W_{target, r} ( sum_{u} U_{u, r} A_u ) ( sum_{v} V_{v, r} B_v )
        # This is a rank-r decomposition of the *bilinear form* mapping (A,B) to C_target.
        # Since C has multiple entries, we have a system of equations.
        # The "rank" is the number of such bilinear forms needed to represent the entire map.
        
        # So, for each rank-1 term r:
        # It defines a bilinear form f_r(A, B) = (sum_u U_{u,r} A_u) (sum_v V_{v,r} B_v).
        # Then C_{i,k} = sum_r W_{i,k,r} f_r(A, B).
        
        # This matches the Strassen algorithm!
        # Strassen computes 7 intermediate products P_1...P_7.
        # Each P_r is a bilinear form.
        # P_1 = (A00+A11)(B00+B11).
        # U for P_1: A00=1, A11=1. (Indices 0 and 3 in flattened 2x2? No, 0,1,2,3).
        # A00 is index 0. A11 is index 3.
        # V for P_1: B00=1, B11=1. Indices 0 and 3.
        # Then C00 = 1*P1 + ...
        # C11 = 1*P1 + ...
        # So W for P_1: C00=1, C11=1.
        
        # This works perfectly.
        
        pass # Logic confirmed.

    # Helper to apply a specific Strassen-like rank-1 term
    def apply_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        """
        u_indices: list of indices in U (flattened N*M)
        u_vals: list of values
        v_indices: list of indices in V (flattened M*P)
        v_vals: list of values
        w_indices: list of indices in W (flattened P*N)
        w_vals: list of values
        """
        nonlocal current_rank
        for idx, val in zip(u_indices, u_vals):
            U[idx][current_rank] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][current_rank] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][current_rank] += val
        current_rank += 1

    # --- Block 1: A_L (2x2) * B_L (2x9) ---
    # A_L indices in U:
    # A00: 0*M+0 = 0
    # A01: 0*M+1 = 1
    # A10: 1*M+0 = 3
    # A11: 1*M+1 = 4
    
    a00, a01, a10, a11 = 0, 1, 3, 4
    
    # B_L is 2x9. Rows 0 and 1 of B.
    # B indices in V:
    # B_{row, col} is at index row*P + col.
    # Row 0: 0*9+k = k
    # Row 1: 1*9+k = 9+k
    
    # C indices in W:
    # C_{i, k} is at index k*N + i.
    # i=0: k*2 + 0 = 2k
    # i=1: k*2 + 1 = 2k+1
    
    # We process B_L column by column, grouping pairs for Strassen.
    # Pairs: (0,1), (2,3), (4,5), (6,7). Remainder: 8.
    
    for pair_idx in range(4):
        k0 = pair_idx * 2
        k1 = k0 + 1
        
        # B indices for this pair
        b00, b01 = k0, k1          # Row 0, cols k0, k1
        b10, b11 = 9 + k0, 9 + k1   # Row 1, cols k0, k1
        
        # C indices for this pair
        c00, c01 = 2*k0, 2*k1       # Row 0, cols k0, k1
        c10, c11 = 2*k0+1, 2*k1+1   # Row 1, cols k0, k1
        
        # Apply Strassen 2x2
        # P1 = (A00+A11)(B00+B11)
        apply_term([a00, a11], [1, 1], [b00, b11], [1, 1], [c00, c11], [1, 1])
        
        # P2 = (A10+A11)B00
        apply_term([a10, a11], [1, 1], [b00], [1], [c10], [1])
        
        # P3 = A00(B10-B11)
        apply_term([a00], [1], [b10, b11], [1, -1], [c01], [1])
        
        # P4 = A11(B01-B00)
        apply_term([a11], [1], [b01, b00], [1, -1], [c00, c10], [1, 1])
        
        # P5 = (A00+A01)B11
        apply_term([a00, a01], [1, 1], [b11], [1], [c00, c01], [-1, 1])
        
        # P6 = (A10-A00)(B00+B01)
        apply_term([a10, a00], [1, -1], [b00, b01], [1, 1], [c11], [1])
        
        # P7 = (A01-A11)(B10+B11)
        apply_term([a01, a11], [1, -1], [b10, b11], [1, 1], [c00, c11], [1, 1])
        
    # Handle remaining column k=8 for A_L * B_L
    k8 = 8
    b08 = k8
    b18 = 9 + k8
    c08 = 2*k8
    c18 = 2*k8 + 1
    
    # Direct multiplication for 2x2 * 2x1
    # C08 = A00*B08 + A01*B18
    # C18 = A10*B08 + A11*B18
    
    # Term 1: A00 * B08 -> C08
    apply_term([a00], [1], [b08], [1], [c08], [1])
    # Term 2: A01 * B18 -> C08
    apply_term([a01], [1], [b18], [1], [c08], [1])
    # Term 3: A10 * B08 -> C18
    apply_term([a10], [1], [b08], [1], [c18], [1])
    # Term 4: A11 * B18 -> C18
    apply_term([a11], [1], [b18], [1], [c18], [1])
    
    # --- Block 2: A_R (2x1) * B_R (1x9) ---
    # A_R is column 2 of A.
    # A02 index: 0*M+2 = 2
    # A12 index: 1*M+2 = 5
    
    a02, a12 = 2, 5
    
    # B_R is row 2 of B.
    # B_{2, k} index: 2*P + k = 18 + k
    
    for k in range(9):
        b2k = 18 + k
        c0k = 2*k
        c1k = 2*k + 1
        
        # C0k += A02 * B2k
        apply_term([a02], [1], [b2k], [1], [c0k], [1])
        
        # C1k += A12 * B2k
        apply_term([a12], [1], [b2k], [1], [c1k], [1])
        
    return {"U": U, "V": V, "W": W}
