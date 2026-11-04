import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal tensor decomposition for 4x4x5 matrix multiplication.
    
    Strategy:
    1. Use the Strassen algorithm for the 4x4x4 part (rank 49).
    2. Use schoolbook multiplication for the 5th column (rank 16).
    Total rank: 65.
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    t_start = time.time()
    
    # We will build U, V, W as lists of lists (or numpy arrays)
    # U: (16, rank), V: (20, rank), W: (20, rank)
    
    # Helper to add a rank-1 term
    # a_idx: list of (index, coeff) for U
    # b_idx: list of (index, coeff) for V
    # c_idx: list of (index, coeff) for W
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(a_sparse, b_sparse, c_sparse):
        """
        a_sparse: dict {index: coeff}
        b_sparse: dict {index: coeff}
        c_sparse: dict {index: coeff}
        """
        u_row = [0.0] * 16
        v_row = [0.0] * 20
        w_row = [0.0] * 20
        
        for idx, val in a_sparse.items():
            u_row[idx] = val
        for idx, val in b_sparse.items():
            v_row[idx] = val
        for idx, val in c_sparse.items():
            w_row[idx] = val
            
        U_cols.append(u_row)
        V_cols.append(v_row)
        W_cols.append(w_row)
        
    # 1. Strassen 4x4x4 part
    # This handles C[:, 0:4] = A * B[:, 0:4]
    
    # Strassen 2x2x2 decomposition
    # Local indices for 2x2 matrix:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    # We define the 7 terms of Strassen.
    # Each term is a tuple (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs is a dict of local_idx -> coeff
    # B_coeffs is a dict of local_idx -> coeff
    # C_coeffs is a dict of local_idx -> coeff
    
    strassen_2x2x2_terms = [
        # M1 = (A00+A11)*(B00+B11) -> C00+=M1, C11+=M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10+A11)*B00 -> C10+=M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1}),
        # M3 = A00*(B10-B00) -> C01+=M3, C11+=M3
        ({0: 1}, {2: 1, 0: -1}, {1: 1, 3: 1}),
        # M4 = A11*(B01-B11) -> C00+=M4, C10+=M4
        ({3: 1}, {1: 1, 3: -1}, {0: 1, 2: 1}),
        # M5 = (A00+A01)*B11 -> C01+=M5, C00-=M5
        ({0: 1, 1: 1}, {3: 1}, {1: 1, 0: -1}),
        # M6 = (A01-A11)*(B00+B01) -> C11+=M6
        ({1: 1, 3: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A00-A10)*(B10+B11) -> C00+=M7
        ({0: 1, 2: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Block structure for 4x4x4
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    
    # Strassen steps for block level:
    # S1 = (A00+A11) * (B00+B11)
    # S2 = (A10+A11) * B00
    # S3 = A00 * (B10-B00)
    # S4 = A11 * (B01-B11)
    # S5 = (A00+A01) * B11
    # S6 = (A01-A11) * (B00+B01)
    # S7 = (A00-A10) * (B10+B11)
    
    # C block contributions:
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    
    # We iterate over the 7 block steps.
    # For each block step, we have specific A blocks and B blocks involved.
    # We also have specific C blocks that receive the result.
    
    block_steps = [
        # (A_blocks, B_blocks, C_blocks)
        # A_blocks: list of (block_idx, coeff)
        # B_blocks: list of (block_idx, coeff)
        # C_blocks: list of (block_idx, coeff)
        
        # S1: (A00+A11) * (B00+B11)
        # A: 00 (+1), 11 (+1)
        # B: 00 (+1), 11 (+1)
        # C: 00 (+1), 11 (+1)
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]),
        
        # S2: (A10+A11) * B00
        # A: 10 (+1), 11 (+1)
        # B: 00 (+1)
        # C: 10 (+1)
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1) ]),
        
        # S3: A00 * (B10-B00)
        # A: 00 (+1)
        # B: 10 (+1), 00 (-1)
        # C: 01 (+1), 11 (+1)
        ([ (0, 1) ], [ (2, 1), (0, -1) ], [ (1, 1), (3, 1) ]),
        
        # S4: A11 * (B01-B11)
        # A: 11 (+1)
        # B: 01 (+1), 11 (-1)
        # C: 00 (+1), 10 (+1)
        ([ (3, 1) ], [ (1, 1), (3, -1) ], [ (0, 1), (2, 1) ]),
        
        # S5: (A00+A01) * B11
        # A: 00 (+1), 01 (+1)
        # B: 11 (+1)
        # C: 01 (+1), 00 (-1)
        ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (1, 1), (0, -1) ]),
        
        # S6: (A01-A11) * (B00+B01)
        # A: 01 (+1), 11 (-1)
        # B: 00 (+1), 01 (+1)
        # C: 11 (+1)
        ([ (1, 1), (3, -1) ], [ (0, 1), (1, 1) ], [ (3, 1) ]),
        
        # S7: (A00-A10) * (B10+B11)
        # A: 00 (+1), 10 (-1)
        # B: 10 (+1), 11 (+1)
        # C: 00 (+1)
        ([ (0, 1), (2, -1) ], [ (2, 1), (3, 1) ], [ (0, 1) ]),
    ]
    
    # Helper to map block index and local index to global index
    # Block index 0: rows 0-1, cols 0-1
    # Block index 1: rows 0-1, cols 2-3
    # Block index 2: rows 2-3, cols 0-1
    # Block index 3: rows 2-3, cols 2-3
    
    def block_local_to_global(block_idx, local_r, local_c, is_B=False):
        """
        Map block index and local (row, col) to global (row, col).
        For A and C (n x m or n x p), row is i, col is j/k.
        For B (m x p), row is j, col is k.
        """
        br = block_idx // 2
        bc = block_idx % 2
        gr = br * 2 + local_r
        gc = bc * 2 + local_c
        return gr, gc
    
    # Construct the 49 terms
    for step in block_steps:
        a_blocks, b_blocks, c_blocks = step
        
        # We need to combine the A blocks into a single "virtual" 2x2 matrix
        # and B blocks into a single "virtual" 2x2 matrix.
        # Then apply Strassen 2x2x2 to these virtual matrices.
        # Then map the resulting C virtual matrix back to the actual C blocks.
        
        # Virtual A 2x2 matrix entries:
        # vA[0] = sum of A_block[0] coeff
        # vA[1] = sum of A_block[1] coeff
        # etc.
        # But wait, the Strassen 2x2x2 terms are fixed.
        # We are effectively doing:
        # For each Strassen term t in strassen_2x2x2_terms:
        #   U_t = sum_{(ba, ca) in a_blocks} ca * (U_t mapped to block ba)
        #   V_t = sum_{(bb, cb) in b_blocks} cb * (V_t mapped to block bb)
        #   W_t = sum_{(bc, cc) in c_blocks} cc * (W_t mapped to block bc)
        
        for t_idx in range(7):
            a_local_coeffs, b_local_coeffs, c_local_coeffs = strassen_2x2x2_terms[t_idx]
            
            # Construct global U, V, W sparse dicts for this term
            u_sparse = {}
            v_sparse = {}
            w_sparse = {}
            
            # Process A
            # a_local_coeffs is dict {local_idx: coeff}
            # local_idx 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
            for a_block_idx, a_block_coeff in a_blocks:
                for loc_idx, loc_coeff in a_local_coeffs.items():
                    if loc_coeff == 0:
                        continue
                    # Determine local row/col from loc_idx
                    lr = loc_idx // 2
                    lc = loc_idx % 2
                    gr, gc = block_local_to_global(a_block_idx, lr, lc)
                    global_idx = gr * 4 + gc
                    val = a_block_coeff * loc_coeff
                    u_sparse[global_idx] = u_sparse.get(global_idx, 0) + val
            
            # Process B
            for b_block_idx, b_block_coeff in b_blocks:
                for loc_idx, loc_coeff in b_local_coeffs.items():
                    if loc_coeff == 0:
                        continue
                    lr = loc_idx // 2
                    lc = loc_idx % 2
                    gr, gc = block_local_to_global(b_block_idx, lr, lc)
                    # B is 4x5, but we are only using cols 0-3 for this part.
                    # So gc is 0-3.
                    global_idx = gr * 5 + gc
                    val = b_block_coeff * loc_coeff
                    v_sparse[global_idx] = v_sparse.get(global_idx, 0) + val
            
            # Process C
            for c_block_idx, c_block_coeff in c_blocks:
                for loc_idx, loc_coeff in c_local_coeffs.items():
                    if loc_coeff == 0:
                        continue
                    lr = loc_idx // 2
                    lc = loc_idx % 2
                    gr, gc = block_local_to_global(c_block_idx, lr, lc)
                    # C is 4x5. gc is 0-3.
                    # W index is k * n + i. Here k=gc, i=gr.
                    global_idx = gc * 4 + gr
                    val = c_block_coeff * loc_coeff
                    w_sparse[global_idx] = w_sparse.get(global_idx, 0) + val
            
            # Add the term if non-zero
            if any(u_sparse.values()) and any(v_sparse.values()) and any(w_sparse.values()):
                add_term(u_sparse, v_sparse, w_sparse)
                
    # 2. Schoolbook for the 5th column
    # C[:, 4] = A * B[:, 4]
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # 16 terms
    
    for i in range(4):
        for j in range(4):
            u_sparse = {i * 4 + j: 1}
            v_sparse = {j * 5 + 4: 1}
            # C index: k=4, i=i -> 4 * 4 + i = 16 + i
            w_sparse = {16 + i: 1}
            add_term(u_sparse, v_sparse, w_sparse)
            
    # Convert lists to numpy arrays
    U = np.array(U_cols).T  # (16, rank)
    V = np.array(V_cols).T  # (20, rank)
    W = np.array(W_cols).T  # (20, rank)
    
    rank = U.shape[1]
    
    # Verification
    # Check if U @ V @ W^T reconstructs the tensor
    # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r] ?
    # Wait, the tensor is T[i,j,k] = A[i,j] B[j,k].
    # Our decomposition is sum_r U_r (A coeffs) * V_r (B coeffs) * W_r (C coeffs).
    # This implies C = sum_r (U_r outer V_r) * W_r? No.
    # The tensor T is a 3-tensor.
    # T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r} is NOT the matrix multiplication tensor.
    # Matrix multiplication tensor is T_{i,j,k} = A_{i,j} B_{j,k}.
    # The decomposition is T = sum_r u_r \otimes v_r \otimes w_r.
    # This means T_{i,j,k} = sum_r u_{r,i} v_{r,j} w_{r,k}.
    # Here u_{r,i} is the coefficient of A_{i,j}? No.
    # u_r is a vector in the space of A (dim 16).
    # v_r is a vector in the space of B (dim 20).
    # w_r is a vector in the space of C (dim 20).
    # So T_{i,j,k} = sum_r U[i,r] V[j,r] W[k,r].
    # This must equal the indicator function of valid (i,j,k) tuples?
    # No, it must equal 1 if the indices form a valid multiplication term, 0 otherwise?
    # Actually, the tensor T is defined such that for any A, B, C=AB:
    # <T, A \otimes B \otimes C> = 0? No.
    # The tensor T represents the bilinear map.
    # T_{i,j,k} = 1 if we are computing C_{i,k} from A_{i,j} and B_{j,k}.
    # So T_{i,j,k} = 1 for all i,j,k? No, that's not a tensor of rank 1.
    # The tensor is the sum of basis tensors e_i \otimes e_j \otimes e_k where the multiplication is defined.
    # Specifically, T = sum_{i,j,k} e_i \otimes e_j \otimes e_k is not right.
    # The multiplication tensor is T = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}^* ?
    # Let's check the reconstruction.
    # We want C = A B.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # Our decomposition gives C = sum_r (U_r \otimes V_r) * W_r ?
    # No, the standard form is:
    # C_{i,k} = sum_r W_{i,k,r} ( sum_j U_{i,j,r} A_{i,j} ) ( sum_k V_{j,k,r} B_{j,k} ) ?
    # No, the rank-r decomposition of the tensor T (where T_{i,j,k} is 1 if (i,j,k) is a valid term in the sum) is:
    # T_{i,j,k} = sum_{r=1}^R U_{i,j,r} V_{j,k,r} W_{i,k,r}.
    # Here U, V, W are the factor matrices.
    # U has shape (16, R). Index (i*4+j, r).
    # V has shape (20, R). Index (j*5+k, r).
    # W has shape (20, R). Index (k*4+i, r).
    
    # Let's verify with a random matrix.
    if time.time() < deadline:
        np.random.seed(42)
        A_test = np.random.rand(4, 4)
        B_test = np.random.rand(4, 5)
        C_test = A_test @ B_test
        
        # Reconstruct C
        C_recon = np.zeros((4, 5))
        for r in range(rank):
            u_vec = U[:, r].reshape(4, 4)
            v_vec = V[:, r].reshape(4, 5)
            w_vec = W[:, r]
            # The term is (u_vec * A_test) * (v_vec * B_test) * w_vec?
            # No, the formula is:
            # C_{i,k} += W_{k*4+i, r} * (sum_j U_{i*4+j, r} A_{i,j}) * (sum_j V_{j*5+k, r} B_{j,k})?
            # Wait, the contraction is over j for A and B?
            # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r}.
            # This implies that for a fixed r, we have a term:
            # Term_r = (sum_{i,j} U_{i,j,r} A_{i,j}) * (sum_{j,k} V_{j,k,r} B_{j,k}) * ... ?
            # No, that's not how tensor decomposition works for matrix multiplication.
            
            # Correct interpretation:
            # The tensor T is a 3-way array T[i,j,k].
            # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] is WRONG.
            # The indices of U, V, W must match the indices of T.
            # T has indices (i, j, k).
            # U has indices (i, j, r).
            # V has indices (j, k, r).
            # W has indices (i, k, r).
            # This is not a standard CP decomposition where all tensors share the same index set.
            # This is a specific decomposition for the multiplication tensor.
            
            # The condition is:
            # sum_r U[i*4+j, r] * V[j*5+k, r] * W[k*4+i, r] should be 1 if we are summing for C[i,k]?
            # No, the tensor T is defined such that:
            # C[i,k] = sum_j A[i,j] B[j,k].
            # This is linear in A and B.
            # The tensor T_{i,j,k} is 1 for all i,j,k? No.
            # The tensor is the sum of rank-1 tensors.
            # A rank-1 tensor for matrix multiplication is of the form:
            # u \otimes v \otimes w where u is a vector in A-space, v in B-space, w in C-space.
            # The product is (u^T A_flat) * (v^T B_flat) * w?
            # No, the term in the sum is:
            # (sum_{i,j} U_{i,j,r} A_{i,j}) * (sum_{j,k} V_{j,k,r} B_{j,k}) ... this doesn't make sense dimensionally.
            
            # Let's look at the schoolbook.
            # U[i*4+j, col] = 1.
            # V[j*5+k, col] = 1.
            # W[k*4+i, col] = 1.
            # Term = 1 * 1 * 1 = 1.
            # This term contributes to C[i,k].
            # The sum is over r.
            # So C[i,k] = sum_r W[k*4+i, r] * (sum_{j'} U[i*4+j', r] A[i,j']) * (sum_{j'} V[j'*5+k, r] B[j',k])?
            # No, the contraction is not like that.
            
            # The correct check is:
            # For each r, let a_r = sum_{i,j} U[i*4+j, r] A[i,j]
            # let b_r = sum_{j,k} V[j*5+k, r] B[j,k]
            # This is not correct.
            
            # The tensor decomposition T = sum_r u_r \otimes v_r \otimes w_r means:
            # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r} is NOT the definition.
            # The definition is that the bilinear map (A,B) -> C is given by:
            # C_{i,k} = sum_r W_{i,k,r} ( sum_{j} U_{i,j,r} A_{i,j} ) ( sum_{j} V_{j,k,r} B_{j,k} ) ?
            # No, that would be trilinear.
            
            # Actually, the standard form is:
            # C_{i,k} = sum_r W_{i,k,r} M_r
            # where M_r = (sum_{j} U_{i,j,r} A_{i,j}) * (sum_{j} V_{j,k,r} B_{j,k}) is wrong.
            
            # Let's stick to the simplest verification:
            # The tensor T_{i,j,k} is 1 if the indices (i,j,k) form a valid path in the multiplication?
            # No, T_{i,j,k} is 1 for all i,j,k in the schoolbook?
            # In schoolbook, each term r corresponds to a specific (i,j,k).
            # U_{i,j,r}=1, V_{j,k,r}=1, W_{i,k,r}=1.
            # So T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r} = 1 for that specific (i,j,k).
            # And 0 for others.
            # So the tensor T is the sum of basis tensors e_{i,j} \otimes e_{j,k} \otimes e_{i,k}.
            # This is exactly the matrix multiplication tensor.
            
            # So we just need to check if:
            # sum_r U[i*4+j, r] V[j*5+k, r] W[k*4+i, r] == 1 for all i,j,k?
            # No, it should be 1 if the term is present in the decomposition of the tensor.
            # The tensor T has T_{i,j,k} = 1 for all i,j,k?
            # No, T_{i,j,k} is 1 if we are computing C_{i,k} using A_{i,j} and B_{j,k}.
            # Since every pair (i,j) and (j,k) contributes to C_{i,k}, T_{i,j,k} = 1 for all i,j,k.
            # Wait, if T_{i,j,k} = 1 for all i,j,k, then the tensor is the all-ones tensor.
            # That is not the matrix multiplication tensor.
            
            # The matrix multiplication tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k ?
            # No, it is T = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}^*.
            # The tensor is defined on the dual space.
            
            # Let's just verify the matrix multiplication directly.
            # C_recon = np.zeros((4,5))
            # for r in range(rank):
            #     # Compute the scalar product for A and B
            #     # This is tricky because U, V are not just vectors.
            
            # Let's use the property:
            # C = sum_r (U_r \otimes V_r) \circ W_r ?
            
            # Easiest check:
            # For random A, B:
            # C_calc = np.zeros((4,5))
            # for r in range(rank):
            #     # The r-th term is:
            #     # M_r = (sum_{i,j} U[i*4+j, r] A[i,j]) * (sum_{j,k} V[j*5+k, r] B[j,k])
            #     # This is a scalar.
            #     # Then C += M_r * W_r_matrix?
            #     # No, W is a vector.
            
            # Actually, the term is:
            # C_{i,k} += W[k*4+i, r] * (sum_{j} U[i*4+j, r] A[i,j]) * (sum_{j} V[j*5+k, r] B[j,k])
            # This is not correct.
            
            # Let's look at the schoolbook again.
            # U[i*4+j, col] = 1.
            # V[j*5+k, col] = 1.
            # W[k*4+i, col] = 1.
            # If we plug this into the formula:
            # C_{i,k} = sum_col W[k*4+i, col] * (sum_{j'} U[i*4+j', col] A[i,j']) * (sum_{j'} V[j'*5+k, col] B[j',k])
            # For the specific col corresponding to (i,j,k):
            # U[i*4+j, col] = 1, others 0. So sum_{j'} ... = A[i,j].
            # V[j*5+k, col] = 1, others 0. So sum_{j'} ... = B[j,k].
            # W[k*4+i, col] = 1.
            # Term = 1 * A[i,j] * B[j,k].
            # Sum over all cols gives sum_{j} A[i,j] B[j,k].
            # This works!
            
            # So the formula is:
            # C_{i,k} = sum_r W[k*4+i, r] * (sum_{j} U[i*4+j, r] A[i,j]) * (sum_{j} V[j*5+k, r] B[j,k])
            
            C_recon = np.zeros((4, 5))
            for r in range(rank):
                # Precompute sums for this r
                # sum_A[i] = sum_j U[i*4+j, r] A[i,j]
                sum_A = np.zeros(4)
                for i in range(4):
                    for j in range(4):
                        sum_A[i] += U[i*4+j, r] * A_test[i,j]
                        
                sum_B = np.zeros(5)
                for k in range(5):
                    for j in range(4):
                        sum_B[k] += V[j*5+k, r] * B_test[j,k]
                        
                for i in range(4):
                    for k in range(5):
                        C_recon[i,k] += W[k*4+i, r] * sum_A[i] * sum_B[k]
                        
            if not np.allclose(C_recon, C_test):
                # Fallback to schoolbook
                print("Verification failed, falling back to schoolbook")
                rank = 80
                U = [[0] * rank for _ in range(16)]
                V = [[0] * rank for _ in range(20)]
                W = [[0] * rank for _ in range(20)]
                column = 0
                for i in range(4):
                    for j in range(4):
                        for k in range(5):
                            U[i*4+j][column] = 1
                            V[j*5+k][column] = 1
                            W[k*4+i][column] = 1
                            column += 1
                return {"U": U, "V": V, "W": W}
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
