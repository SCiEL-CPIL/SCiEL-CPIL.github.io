import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 4, 4, 5
    
    def time_remaining():
        return deadline - time.time()

    def verify(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        if U_arr.ndim != 2 or V_arr.ndim != 2 or W_arr.ndim != 2:
            return False
        
        rank = U_arr.shape[1]
        if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
            return False
        
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        # Check against the identity tensor for matrix multiplication
        # T[i,j,k] should be 1 if k == j? No.
        # The tensor for C = A * B is T_{ijk} such that C_{ik} = sum_j A_{ij} B_{jk} T_{ijk} ?
        # No, the decomposition is sum_r U_{ijr} V_{jkr} W_{kir} = 1 for all i,j,k?
        # Wait, the standard definition is:
        # C_{ik} = sum_j A_{ij} B_{jk}
        # The tensor T has indices i (row of A), j (col of A / row of B), k (col of B).
        # T_{ijk} = 1 if the multiplication A_{ij} * B_{jk} contributes to C_{ik}.
        # Actually, the "multiplication tensor" T is defined such that
        # C = A \times B is given by the trilinear form.
        # The condition for the factors U, V, W is:
        # sum_r U_{ijr} V_{jkr} W_{kir} = 1 for all i, j, k.
        # This corresponds to the fact that A_{ij} B_{jk} is the term in the sum for C_{ik}.
        # So T_{ijk} is 1 for all i, j, k?
        # Let's check the schoolbook case.
        # U_{ijr} = 1 if r encodes (i,j). V_{jkr} = 1 if r encodes (j,k). W_{kir} = 1 if r encodes (k,i).
        # If r = (i,j,k), then U=1, V=1, W=1. Sum is 1.
        # If r != (i,j,k), then at least one is 0.
        # So yes, the target tensor T_{ijk} is all ones.
        
        if not np.allclose(T, 1.0, atol=1e-8):
            return False
        return True

    def get_schoolbook():
        schoolbook_rank = n * m * p
        U = np.zeros((n * m, schoolbook_rank), dtype=float)
        V = np.zeros((m * p, schoolbook_rank), dtype=float)
        W = np.zeros((p * n, schoolbook_rank), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    # Construct the 4x4x4 Strassen decomposition (rank 49)
    # We use the recursive structure: 7 outer terms, each decomposed into 7 inner terms.
    
    # Define the 7 Strassen factors for 2x2 matrices.
    # U2, V2, W2 are 4x7 matrices (flattened 2x2 indices).
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 0: (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # Term 1: (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # Term 2: A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # Term 3: A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # Term 4: (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # Term 5: (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # Term 6: (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Now, construct the 4x4x4 factors.
    # The 4x4 matrices are partitioned into four 2x2 blocks.
    # Block (br, bc) corresponds to rows 2*br..2*br+1 and cols 2*bc..2*bc+1.
    # Global index for A: i*m + j.
    # Global index for B: j*p + k.
    # Global index for C: k*n + i.
    
    # We need to map the 2x2 Strassen factors (U2, V2, W2) to the 4x4 context.
    # For a specific outer term s (0..6), we have effective 4x4 matrices L_s and R_s.
    # L_s and R_s are formed by linear combinations of the 2x2 blocks of A and B.
    # Let's define the block coefficients for L_s and R_s.
    # L_s[br, bc] is a 2x2 matrix. In our case, L_s is either 0, I, or combinations.
    # Actually, L_s and R_s are 4x4 matrices that are sparse/block-diagonal or block-sparse.
    
    # Let's define the 7 outer terms as 4x4 matrices of coefficients for the blocks.
    # coeff_L[s, br, bc] is the scalar coefficient for the block (br, bc) of A.
    # coeff_R[s, br, bc] is the scalar coefficient for the block (br, bc) of B.
    
    coeff_L = np.zeros((7, 2, 2))
    coeff_R = np.zeros((7, 2, 2))
    
    # s=0: L = A00 + A11, R = B00 + B11
    coeff_L[0, 0, 0] = 1; coeff_L[0, 1, 1] = 1
    coeff_R[0, 0, 0] = 1; coeff_R[0, 1, 1] = 1
    
    # s=1: L = A10 + A11, R = B00
    coeff_L[1, 1, 0] = 1; coeff_L[1, 1, 1] = 1
    coeff_R[1, 0, 0] = 1
    
    # s=2: L = A00, R = B01 - B11
    coeff_L[2, 0, 0] = 1
    coeff_R[2, 0, 1] = 1; coeff_R[2, 1, 1] = -1
    
    # s=3: L = A11, R = B10 - B00
    coeff_L[3, 1, 1] = 1
    coeff_R[3, 1, 0] = 1; coeff_R[3, 0, 0] = -1
    
    # s=4: L = A00 + A01, R = B11
    coeff_L[4, 0, 0] = 1; coeff_L[4, 0, 1] = 1
    coeff_R[4, 1, 1] = 1
    
    # s=5: L = A10 - A00, R = B00 + B01
    coeff_L[5, 1, 0] = 1; coeff_L[5, 0, 0] = -1
    coeff_R[5, 0, 0] = 1; coeff_R[5, 0, 1] = 1
    
    # s=6: L = A01 - A11, R = B10 + B11
    coeff_L[6, 0, 1] = 1; coeff_L[6, 1, 1] = -1
    coeff_R[6, 1, 0] = 1; coeff_R[6, 1, 1] = 1

    # Initialize 4x4x4 factors (rank 49)
    U44 = np.zeros((16, 49))
    V44 = np.zeros((16, 49))
    W44 = np.zeros((16, 49))
    
    col_idx = 0
    for s in range(7):
        # For each outer term s, we apply the inner 2x2 Strassen decomposition.
        # The inner Strassen operates on the 2x2 blocks of L_s and R_s.
        # Let X be the 2x2 matrix of blocks for L_s, and Y for R_s.
        # X[br, bc] is a 2x2 matrix (the block of A).
        # Y[br, bc] is a 2x2 matrix (the block of B).
        
        # We iterate over the 7 inner terms t (0..6).
        for t in range(7):
            # The inner Strassen term t contributes:
            # U_inner = U2[:, t] (4x1 vector, flattened 2x2)
            # V_inner = V2[:, t] (4x1 vector, flattened 2x2)
            # W_inner = W2[:, t] (4x1 vector, flattened 2x2)
            
            # We need to expand these into the 4x4 global indices.
            # U_global[i*m+j, col] += coeff * U2[local_i, local_j, t]
            # The local index (li, lj) in U2 corresponds to the position within the 2x2 block structure.
            # Wait, U2 is for a 2x2 matrix multiplication.
            # U2[li*2+lj, t] is the coefficient for the element at row li, col lj of the 2x2 "effective" A matrix.
            # The "effective" A matrix X has blocks X[br, bc].
            # X[br, bc] is a 2x2 matrix.
            # The element at global row i, col j of A contributes to X[br, bc] at local row li, col lj if:
            # i = 2*br + li, j = 2*bc + lj.
            # And the coefficient is coeff_L[s, br, bc] * U2[li*2+lj, t].
            
            # So, for each br, bc where coeff_L[s, br, bc] != 0:
            #   For each li, lj in 0..1:
            #     global_i = 2*br + li
            #     global_j = 2*bc + lj
            #     U44[global_i*4 + global_j, col_idx] += coeff_L[s, br, bc] * U2[li*2+lj, t]
            
            # Similarly for V:
            #   For each br, bc where coeff_R[s, br, bc] != 0:
            #     For each li, lj in 0..1:
            #       global_j = 2*br + li  # Row of B
            #       global_k = 2*bc + lj  # Col of B
            #       V44[global_j*5 + global_k, col_idx] += coeff_R[s, br, bc] * V2[li*2+lj, t]
            #       Note: V index is j*p + k. Here p=4 for the 4x4x4 part.
            
            # And for W:
            #   W_inner = W2[:, t].
            #   W2[li*2+lj, t] is the coefficient for the element at row li, col lj of the 2x2 "effective" C matrix.
            #   The effective C matrix Z = X * Y has blocks Z[br, bc].
            #   Z[br, bc] is a 2x2 matrix.
            #   The element at global row i, col k of C corresponds to Z[br, bc] at local row li, col lj if:
            #   i = 2*br + li, k = 2*bc + lj.
            #   And the coefficient is W2[li*2+lj, t].
            
            #   So, for each li, lj in 0..1:
            #     global_i = 2*li  # Wait, br and bc are the block indices.
            #     The block index for C is determined by the outer Strassen combination?
            #     No, the inner Strassen computes the product of the 2x2 block matrices X and Y.
            #     The result Z is a 2x2 matrix of 2x2 blocks.
            #     Z[br, bc] = sum_{mid} X[br, mid] Y[mid, bc].
            #     The inner Strassen decomposition of the 2x2 matrix multiplication gives:
            #     Z[br, bc] = sum_t (sum_{li,lj} U2[li*2+lj, t] X[li, lj]) * (sum_{mi,mk} V2[mi*2+mk, t] Y[mi, mk]) * W2[br*2+bc, t] ?
            #     No, that's not right.
            
            # Let's re-evaluate the inner Strassen application.
            # We are multiplying two 2x2 matrices X and Y (where entries are 2x2 blocks).
            # The result Z is a 2x2 matrix of 2x2 blocks.
            # The Strassen algorithm for 2x2 matrices gives 7 scalar multiplications M_t.
            # M_t = (linear combo of X entries) * (linear combo of Y entries).
            # The entries of Z are linear combinations of M_t.
            # Z[br, bc] = sum_t coeff_Z[br, bc, t] * M_t.
            # M_t is a 2x2 matrix (product of two 2x2 block linear combos).
            # Wait, X entries are 2x2 matrices. So M_t is a 2x2 matrix.
            # So Z[br, bc] is a sum of 2x2 matrices.
            
            # The factorization of the tensor for 2x2 matrix multiplication is:
            # sum_t U2[li, lj, t] V2[mi, mk, t] W2[br, bc, t] = delta_{li,mi} delta_{lj,mk} ? No.
            # The tensor T_{li, mi, br} for 2x2 mult is such that Z_{li, br} = sum_mi X_{li, mi} Y_{mi, br}.
            # The decomposition is sum_t U_{li, mi, t} V_{mi, br, t} W_{br, li, t} = 1.
            # In our case, the "scalars" are 2x2 matrices.
            # So we have a "block" Strassen.
            # The U factor for the block Strassen is U2[li, mi, t] (scalar) * Identity? No.
            # The U factor for the block Strassen at position (li, mi) is U2[li*2+mi, t] * (the block X[li, mi]).
            # This is getting complicated to map directly to the global 4x4 indices without explicit block multiplication.
            
            # Simpler approach:
            # Just compute the full 4x4 matrices L_s and R_s explicitly.
            # Then, for each s, we have L_s and R_s.
            # We want to decompose L_s * R_s using the 2x2 Strassen factors applied to the 2x2 blocks of L_s and R_s.
            # This is equivalent to saying:
            # (L_s * R_s)_{ik} = sum_j (L_s)_{ij} (R_s)_{jk}
            # We use the Strassen decomposition for the 4x4 product.
            # The Strassen decomposition for 4x4 is the tensor product of the 2x2 Strassen decomposition with itself?
            # No, it's the recursive application.
            # The 49 terms are generated by:
            # For each outer term s (0..6):
            #   Form L_s and R_s (4x4 matrices).
            #   Decompose L_s * R_s using the 7-term Strassen algorithm for 2x2 matrices,
            #   where the "scalars" are 2x2 blocks.
            #   This generates 7 terms M_t (each is a 2x2 matrix).
            #   Then decompose each M_t (2x2 matrix) using the 7-term Strassen algorithm for 2x2 matrices,
            #   where the "scalars" are scalars.
            #   This generates 7 scalar terms.
            #   Total 49 terms.
            
            # To implement this efficiently:
            # 1. Construct L_s and R_s as 4x4 arrays.
            # 2. For each s, compute the 7 block-combinations X_t and Y_t (2x2 matrices of 2x2 blocks, i.e., 4x4 matrices).
            # 3. For each t, we have X_t and Y_t (4x4 matrices).
            # 4. Decompose X_t * Y_t using the standard 2x2 Strassen factors U2, V2, W2, but applied to the 4x4 matrices?
            #    No, X_t and Y_t are 4x4 matrices. We can decompose their product using the rank-49 decomposition? No, that's circular.
            #    We decompose the product of two 2x2 block matrices using the 2x2 Strassen factors.
            #    The factors U2, V2, W2 are for 2x2 matrices.
            #    So, for the block multiplication Z_t = X_t * Y_t (where X_t, Y_t are 2x2 block matrices):
            #    Z_t[br, bc] = sum_t' U2[li, mi, t'] X_t[li, mi] Y_t[mi, bc] W2[bc, br, t'] ?
            #    Indices: U2 is 4x7 (flattened 2x2).
            #    Let's use the explicit block indices.
            #    U2[li, mi, t'] is the coefficient for X[li, mi].
            #    V2[mi, bc, t'] is the coefficient for Y[mi, bc].
            #    W2[bc, br, t'] is the coefficient for Z[br, bc].
            
            # So, for a fixed s and t':
            # The contribution to the global U, V, W is:
            # U_global[i, j] += U2[li, mi, t'] * (L_s)_{ij, li, mi} ?
            # No, L_s is a 4x4 matrix.
            # L_s[2*li+di, 2*mi+dj] is the element at block (li, mi), local pos (di, dj).
            # The "scalar" in the block Strassen is the entire 2x2 block.
            # So the U factor for the block Strassen is U2[li, mi, t'] * (the 2x2 block of L_s at (li, mi)).
            # This means:
            # U_global[2*li+di, 2*mi+dj] += U2[li*2+mi, t'] * L_s[2*li+di, 2*mi+dj]
            
            # Similarly for V:
            # V_global[2*mi+dk, 2*bc+dl] += V2[mi*2+bc, t'] * R_s[2*mi+dk, 2*bc+dl]
            
            # And for W:
            # W_global[2*bc+dl, 2*br+di] += W2[bc*2+br, t'] * Z_t[2*br+di, 2*bc+dl] ?
            # No, Z_t is the result. The W factor in the tensor decomposition is just W2.
            # The W factor for the global tensor is W2[bc*2+br, t'] * (the 2x2 block of Identity? No).
            # The W factor tells us which C entry this term contributes to.
            # In the block Strassen, the term M_t' contributes to Z[br, bc] with coefficient W2[bc*2+br, t'].
            # Z[br, bc] is a 2x2 block of C.
            # So the term contributes to all 4 entries of C[2*br:2*br+2, 2*bc:2*bc+2] with the same coefficient W2[bc*2+br, t'].
            # So:
            # W_global[2*bc+dl, 2*br+di] += W2[bc*2+br, t']
            
            # This logic holds.
            
            pass # Placeholder, will implement below

    # Implementation of the 4x4x4 construction
    U44 = np.zeros((16, 49))
    V44 = np.zeros((16, 49))
    W44 = np.zeros((16, 49))
    
    col_idx = 0
    for s in range(7):
        # Construct L_s and R_s (4x4)
        L_s = np.zeros((4, 4))
        R_s = np.zeros((4, 4))
        for br in range(2):
            for bc in range(2):
                cL = coeff_L[s, br, bc]
                cR = coeff_R[s, br, bc]
                if cL != 0:
                    L_s[2*br:2*br+2, 2*bc:2*bc+2] += cL * np.eye(2)
                if cR != 0:
                    R_s[2*br:2*br+2, 2*bc:2*bc+2] += cR * np.eye(2)
        
        # Apply inner Strassen (2x2 block Strassen)
        # We iterate over the 7 inner terms t' (0..6) of the block Strassen.
        for t_prime in range(7):
            # Get the 2x2 Strassen factors for this term
            # U2[li*2+mi, t_prime]
            # V2[mi*2+bc, t_prime]
            # W2[bc*2+br, t_prime]
            
            # Fill U44
            for li in range(2):
                for mi in range(2):
                    coeff_u = U2[li*2+mi, t_prime]
                    if coeff_u != 0:
                        # Block (li, mi) of L_s
                        # Global rows 2*li..2*li+1, cols 2*mi..2*mi+1
                        U44[2*li:2*li+2, 2*mi:2*mi+2, col_idx] += coeff_u * L_s[2*li:2*li+2, 2*mi:2*mi+2]
            
            # Fill V44
            for mi in range(2):
                for bc in range(2):
                    coeff_v = V2[mi*2+bc, t_prime]
                    if coeff_v != 0:
                        # Block (mi, bc) of R_s
                        # Global rows 2*mi..2*mi+1, cols 2*bc..2*bc+1
                        V44[2*mi:2*mi+2, 2*bc:2*bc+2, col_idx] += coeff_v * R_s[2*mi:2*mi+2, 2*bc:2*bc+2]
            
            # Fill W44
            for br in range(2):
                for bc in range(2):
                    coeff_w = W2[bc*2+br, t_prime]
                    if coeff_w != 0:
                        # Block (br, bc) of C
                        # Global rows 2*br..2*br+1, cols 2*bc..2*bc+1
                        # W index is k*n + i. Here n=4.
                        # C[i, k] corresponds to W[k*4 + i].
                        # Block (br, bc) of C covers rows i=2*br..2*br+1 and cols k=2*bc..2*bc+1.
                        for i_local in range(2):
                            for k_local in range(2):
                                i_global = 2*br + i_local
                                k_global = 2*bc + k_local
                                W44[k_global*4 + i_global, col_idx] += coeff_w
            
            col_idx += 1

    # Now we have U44, V44, W44 for the 4x4x4 part (rank 49).
    # We need to extend this to 4x4x5.
    # The full U is 16x65, V is 20x65, W is 20x65.
    # The first 49 columns are the 4x4x4 part.
    # The last 16 columns are for the 5th column of C (k=4).
    
    U_full = np.zeros((16, 65))
    V_full = np.zeros((20, 65))
    W_full = np.zeros((20, 65))
    
    # Copy the 4x4x4 part
    U_full[:, :49] = U44
    V_full[:16, :49] = V44 # V for 4x4x4 uses only first 4 cols of B (indices 0..15)
    W_full[:16, :49] = W44 # W for 4x4x4 uses only first 4 cols of C (indices 0..15)
    
    # Add the 16 schoolbook terms for the 5th column (k=4)
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    for i in range(4):
        for j in range(4):
            col = 49 + i*4 + j
            U_full[i*4 + j, col] = 1.0
            V_full[j*5 + 4, col] = 1.0
            W_full[4*4 + i, col] = 1.0 # W index: k=4, i=i -> 4*4 + i = 16 + i
    
    # Verify
    if verify(U_full, V_full, W_full):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    else:
        # Fallback to schoolbook
        U_sb, V_sb, W_sb = get_schoolbook()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

