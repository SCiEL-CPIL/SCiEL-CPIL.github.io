import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-49 decomposition for 4x4 matrix multiplication.
    This is the known optimal rank for 4x4 matrix multiplication.
    
    The decomposition is constructed by applying Strassen's algorithm recursively:
    1. Split 4x4 matrices into 2x2 blocks of 2x2 matrices.
    2. Apply Strassen at the block level (7 block multiplications).
    3. Apply Strassen at the scalar level for each block multiplication (7 scalar multiplications each).
    Total rank = 7 * 7 = 49.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    # U shape: (n*m, rank) = (16, 49)
    # V shape: (m*p, rank) = (16, 49)
    # W shape: (p*n, rank) = (16, 49)
    # Convention: 
    # U[i*m + k, r] corresponds to A[i, k]
    # V[k*p + j, r] corresponds to B[k, j]
    # W[j*n + i, r] corresponds to C[i, j] (Trace convention: W[k*n+i, r] where k is col, i is row? 
    # Let's check the prompt: "W[k*n+i,r]". 
    # Standard matrix mult: C[i,j] = sum_k A[i,k] B[k,j].
    # Tensor T[i,j,k] = A[i,k] B[k,j] C[i,j].
    # The prompt says: "output factor follows the official trace convention W[k*n+i,r]".
    # This usually implies the third index of the tensor is mapped to the row index of W.
    # Let's assume the tensor indices are (i, j, k) corresponding to (A_row, B_col, A_col/B_row).
    # Wait, standard tensor for matmul is T_{i,j,k} such that C_{i,j} = sum_k T_{i,j,k} A_{i,k} B_{k,j}.
    # The decomposition is T = sum_r U_{i,k}^{(r)} V_{k,j}^{(r)} W_{i,j}^{(r)}? No.
    # The prompt says: "exactly decompose the matrix multiplication tensor".
    # Usually, the tensor is defined as T[i, j, k] = 1 if it contributes to C[i,j] via A[i,k]B[k,j].
    # The factors are U, V, W.
    # U has shape (n*m, rank). Index: i*m + k? Or i*m + j?
    # Let's look at the seed code:
    # U[i*4+j, r] -> A[i,j]
    # V[j*4+k, r] -> B[j,k]
    # W[k*4+i, r] -> C[i,k]? 
    # Seed code comment: "Tensor: T[i*4+j, j*4+k, k*4+i] = sum_r U[i*4+j,r] * V[j*4+k,r] * W[k*4+i,r] = 1"
    # This implies the tensor indices are (A_idx, B_idx, C_idx).
    # A_idx = i*4+j (row i, col j of A)
    # B_idx = j*4+k (row j, col k of B) -- Note: The 'j' here is the shared index.
    # C_idx = k*4+i (row i, col k of C) -- Note: The 'k' here is the col of C, 'i' is row of C.
    # Wait, C[i,k] = sum_j A[i,j] B[j,k].
    # So the term in the sum is A[i,j] * B[j,k] * 1.
    # The tensor T is non-zero at (A_idx, B_idx, C_idx) where A_idx=i*4+j, B_idx=j*4+k, C_idx=k*4+i.
    # And T[A_idx, B_idx, C_idx] = 1.
    # The decomposition is T[A_idx, B_idx, C_idx] = sum_r U[A_idx, r] * V[B_idx, r] * W[C_idx, r].
    # So we need to populate U, V, W such that for all i,j,k:
    # sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] = 1 if the indices match the matmul pattern, else 0?
    # Actually, the tensor T is defined such that T[i,j,k] (using standard indices i,j,k for A[i,k], B[k,j], C[i,j]) is 1.
    # Let's stick to the seed's index mapping to be safe, as it scored 1.3.
    # Seed:
    # U_idx = i*4 + j  (A[i,j])
    # V_idx = j*4 + k  (B[j,k])
    # W_idx = k*4 + i  (C[i,k] -> row i, col k. Index is k*4+i. This is column-major for C? 
    # If C is 4x4, C[i,k] is at linear index i*4+k in row-major.
    # The seed uses W[k*4+i]. This is column-major index for C, or simply swapping i and k.
    # Let's verify the tensor identity with this mapping.
    # We need: sum_r U[i*4+j, r] V[j*4+k, r] W[k*4+i, r] = 1 for all i,j,k?
    # No, the tensor for matrix multiplication is not 1 everywhere.
    # The tensor T is the structure tensor of the algebra.
    # T_{i,j,k} is 1 if the product A_{i,k} B_{k,j} contributes to C_{i,j}.
    # In the seed's notation:
    # A element is A[i,j] (using j as the shared index). Let's call shared index s.
    # A[i,s], B[s,k], C[i,k].
    # U index: i*4+s
    # V index: s*4+k
    # W index: k*4+i
    # We need sum_r U[i*4+s, r] V[s*4+k, r] W[k*4+i, r] = 1 for all i,s,k.
    # This is the correct definition of the matrix multiplication tensor with this specific index permutation.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 decomposition
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We will build the 49 terms.
    # Each term r corresponds to a product X_r * Y_r.
    # X_r is a linear combination of A elements.
    # Y_r is a linear combination of B elements.
    # The contribution to C is determined by which C element(s) the product contributes to, with what coefficient.
    
    # Helper to add to U, V, W
    def add_term(A_coeffs, B_coeffs, C_coeffs):
        """
        A_coeffs: dict { (i, s): coeff } for A[i,s]
        B_coeffs: dict { (s, k): coeff } for B[s,k]
        C_coeffs: dict { (i, k): coeff } for C[i,k]
        """
        nonlocal rank
        # Find an available rank slot? No, we are constructing fixed rank 49.
        # We need to iterate through the 49 logical terms and assign them to rank indices 0..48.
        pass

    # We will generate the 49 terms explicitly.
    # Level 1: 7 block terms.
    # Level 2: For each block term, 7 scalar terms.
    
    # Block Term Definitions (L, R, C_contrib)
    # L and R are 2x2 matrices of linear combinations.
    # We represent a 2x2 matrix of linear combos as a list of 4 dicts.
    # L[r][c] = dict of A coeffs.
    
    # Define the 7 block-level Strassen steps
    # Step 1: P1 = (A11+A22)(B11+B22)
    # A11 is A[0:2, 0:2], A22 is A[2:4, 2:4]
    # B11 is B[0:2, 0:2], B22 is B[2:4, 2:4]
    
    # We need to map global indices (0..3) to local block indices (0..1).
    # Block (bi, bj) covers rows 2*bi..2*bi+1 and cols 2*bj..2*bj+1.
    
    def get_A_idx(bi, bj, li, lj):
        return (2*bi + li, 2*bj + lj)
        
    def get_B_idx(bi, bj, li, lj):
        return (2*bi + li, 2*bj + lj)
        
    def get_C_idx(bi, bj, li, lj):
        return (2*bi + li, 2*bj + lj)

    # We will create a list of 49 tuples: (U_dict, V_dict, W_dict)
    # U_dict: {(i,s): coeff}
    # V_dict: {(s,k): coeff}
    # W_dict: {(i,k): coeff}
    
    terms = []
    
    # Define the 7 block-level operations
    # Each operation is defined by:
    # L: 2x2 matrix of A-linear-combos
    # R: 2x2 matrix of B-linear-combos
    # C_map: mapping from local C indices (0,0)..(1,1) to global C indices and coefficients.
    #        Specifically, C_global = sum_{local_c} coeff * P[local_c]
    
    # Block 0: P1 = (A11+A22)(B11+B22)
    # L0:
    # L0[0][0] = A11[0][0] + A22[0][0] -> A[0,0] + A[2,2]
    # L0[0][1] = A11[0][1] + A22[0][1] -> A[0,1] + A[2,3]
    # L0[1][0] = A11[1][0] + A22[1][0] -> A[1,0] + A[3,2]
    # L0[1][1] = A11[1][1] + A22[1][1] -> A[1,1] + A[3,3]
    # R0:
    # R0[0][0] = B11[0][0] + B22[0][0] -> B[0,0] + B[2,2]
    # ... and so on.
    
    # Instead of hardcoding, we generate them.
    
    block_defs = [
        # (L_blocks, R_blocks, C_contribs)
        # L_blocks: list of 4 dicts, R_blocks: list of 4 dicts
        # C_contribs: list of 4 dicts, mapping local C pos to global C coeffs
        # Local C pos 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        
        # P1
        (
            [
                {get_A_idx(0,0,0,0): 1, get_A_idx(1,1,0,0): 1}, # L00
                {get_A_idx(0,0,0,1): 1, get_A_idx(1,1,0,1): 1}, # L01
                {get_A_idx(0,0,1,0): 1, get_A_idx(1,1,1,0): 1}, # L10
                {get_A_idx(0,0,1,1): 1, get_A_idx(1,1,1,1): 1}  # L11
            ],
            [
                {get_B_idx(0,0,0,0): 1, get_B_idx(1,1,0,0): 1}, # R00
                {get_B_idx(0,0,0,1): 1, get_B_idx(1,1,0,1): 1}, # R01
                {get_B_idx(0,0,1,0): 1, get_B_idx(1,1,1,0): 1}, # R10
                {get_B_idx(0,0,1,1): 1, get_B_idx(1,1,1,1): 1}  # R11
            ],
            [
                {get_C_idx(0,0,0,0): 1, get_C_idx(1,1,0,0): 1}, # C00 gets P00 + P11? No.
                # C11 = P1[0][0]? No.
                # Block C11 = P1. So C[0:2, 0:2] = P1.
                # P1[0][0] contributes to C[0,0]. P1[0][1] to C[0,1]. Etc.
                # But wait, the block-level Strassen formula gives:
                # C11 = P1 + P4 - P5 + P7
                # So P1 contributes +1 to C11 (all 4 elements).
                # P1 is a 2x2 matrix. P1[li, lj] contributes to C[2*0+li, 2*0+lj] with coeff 1.
                # So for P1, the contribution to global C is:
                # C[0,0] += P1[0,0]
                # C[0,1] += P1[0,1]
                # C[1,0] += P1[1,0]
                # C[1,1] += P1[1,1]
                # AND
                # C[2,0] += P1[0,0]
                # C[2,1] += P1[0,1]
                # C[3,0] += P1[1,0]
                # C[3,1] += P1[1,1]
                # This is handled by the C_contribs structure.
                # Let's refine C_contribs to be a dict: { local_pos: { global_C_idx: coeff } }
                # local_pos 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                
                # For P1, local_pos (li, lj) contributes to global C (0+li, 0+lj) and (2+li, 2+lj) with coeff 1.
                {
                    0: {get_C_idx(0,0,0,0): 1, get_C_idx(1,1,0,0): 1},
                    1: {get_C_idx(0,0,0,1): 1, get_C_idx(1,1,0,1): 1},
                    2: {get_C_idx(0,0,1,0): 1, get_C_idx(1,1,1,0): 1},
                    3: {get_C_idx(0,0,1,1): 1, get_C_idx(1,1,1,1): 1}
                }
            ]
        ),
        # P2 = (A21+A22)B11
        (
            [
                {get_A_idx(1,0,0,0): 1, get_A_idx(1,1,0,0): 1},
                {get_A_idx(1,0,0,1): 1, get_A_idx(1,1,0,1): 1},
                {get_A_idx(1,0,1,0): 1, get_A_idx(1,1,1,0): 1},
                {get_A_idx(1,0,1,1): 1, get_A_idx(1,1,1,1): 1}
            ],
            [
                {get_B_idx(0,0,0,0): 1},
                {get_B_idx(0,0,0,1): 1},
                {get_B_idx(0,0,1,0): 1},
                {get_B_idx(0,0,1,1): 1}
            ],
            [
                # C21 = P2 + P4. C22 = P1 - P2 + P3 + P6.
                # P2 contributes +1 to C21 (rows 2-3, cols 0-1) and -1 to C22 (rows 2-3, cols 2-3).
                {
                    0: {get_C_idx(1,0,0,0): 1, get_C_idx(1,1,0,0): -1},
                    1: {get_C_idx(1,0,0,1): 1, get_C_idx(1,1,0,1): -1},
                    2: {get_C_idx(1,0,1,0): 1, get_C_idx(1,1,1,0): -1},
                    3: {get_C_idx(1,0,1,1): 1, get_C_idx(1,1,1,1): -1}
                }
            ]
        ),
        # P3 = A11(B12-B22)
        (
            [
                {get_A_idx(0,0,0,0): 1},
                {get_A_idx(0,0,0,1): 1},
                {get_A_idx(0,0,1,0): 1},
                {get_A_idx(0,0,1,1): 1}
            ],
            [
                {get_B_idx(0,1,0,0): 1, get_B_idx(1,1,0,0): -1},
                {get_B_idx(0,1,0,1): 1, get_B_idx(1,1,0,1): -1},
                {get_B_idx(0,1,1,0): 1, get_B_idx(1,1,1,0): -1},
                {get_B_idx(0,1,1,1): 1, get_B_idx(1,1,1,1): -1}
            ],
            [
                # C12 = P3 + P5. C22 = P1 - P2 + P3 + P6.
                # P3 contributes +1 to C12 (rows 0-1, cols 2-3) and +1 to C22 (rows 2-3, cols 2-3).
                {
                    0: {get_C_idx(0,1,0,0): 1, get_C_idx(1,1,0,0): 1},
                    1: {get_C_idx(0,1,0,1): 1, get_C_idx(1,1,0,1): 1},
                    2: {get_C_idx(0,1,1,0): 1, get_C_idx(1,1,1,0): 1},
                    3: {get_C_idx(0,1,1,1): 1, get_C_idx(1,1,1,1): 1}
                }
            ]
        ),
        # P4 = A22(B21-B11)
        (
            [
                {get_A_idx(1,1,0,0): 1},
                {get_A_idx(1,1,0,1): 1},
                {get_A_idx(1,1,1,0): 1},
                {get_A_idx(1,1,1,1): 1}
            ],
            [
                {get_B_idx(1,0,0,0): 1, get_B_idx(0,0,0,0): -1},
                {get_B_idx(1,0,0,1): 1, get_B_idx(0,0,0,1): -1},
                {get_B_idx(1,0,1,0): 1, get_B_idx(0,0,1,0): -1},
                {get_B_idx(1,0,1,1): 1, get_B_idx(0,0,1,1): -1}
            ],
            [
                # C11 = P1 + P4 - P5 + P7. C21 = P2 + P4.
                # P4 contributes +1 to C11 (rows 0-1, cols 0-1) and +1 to C21 (rows 2-3, cols 0-1).
                {
                    0: {get_C_idx(0,0,0,0): 1, get_C_idx(1,0,0,0): 1},
                    1: {get_C_idx(0,0,0,1): 1, get_C_idx(1,0,0,1): 1},
                    2: {get_C_idx(0,0,1,0): 1, get_C_idx(1,0,1,0): 1},
                    3: {get_C_idx(0,0,1,1): 1, get_C_idx(1,0,1,1): 1}
                }
            ]
        ),
        # P5 = (A11+A12)B22
        (
            [
                {get_A_idx(0,0,0,0): 1, get_A_idx(0,1,0,0): 1},
                {get_A_idx(0,0,0,1): 1, get_A_idx(0,1,0,1): 1},
                {get_A_idx(0,0,1,0): 1, get_A_idx(0,1,1,0): 1},
                {get_A_idx(0,0,1,1): 1, get_A_idx(0,1,1,1): 1}
            ],
            [
                {get_B_idx(1,1,0,0): 1},
                {get_B_idx(1,1,0,1): 1},
                {get_B_idx(1,1,1,0): 1},
                {get_B_idx(1,1,1,1): 1}
            ],
            [
                # C11 = P1 + P4 - P5 + P7. C12 = P3 + P5.
                # P5 contributes -1 to C11 and +1 to C12.
                {
                    0: {get_C_idx(0,0,0,0): -1, get_C_idx(0,1,0,0): 1},
                    1: {get_C_idx(0,0,0,1): -1, get_C_idx(0,1,0,1): 1},
                    2: {get_C_idx(0,0,1,0): -1, get_C_idx(0,1,1,0): 1},
                    3: {get_C_idx(0,0,1,1): -1, get_C_idx(0,1,1,1): 1}
                }
            ]
        ),
        # P6 = (A21-A11)(B11+B12)
        (
            [
                {get_A_idx(1,0,0,0): 1, get_A_idx(0,0,0,0): -1},
                {get_A_idx(1,0,0,1): 1, get_A_idx(0,0,0,1): -1},
                {get_A_idx(1,0,1,0): 1, get_A_idx(0,0,1,0): -1},
                {get_A_idx(1,0,1,1): 1, get_A_idx(0,0,1,1): -1}
            ],
            [
                {get_B_idx(0,0,0,0): 1, get_B_idx(0,1,0,0): 1},
                {get_B_idx(0,0,0,1): 1, get_B_idx(0,1,0,1): 1},
                {get_B_idx(0,0,1,0): 1, get_B_idx(0,1,1,0): 1},
                {get_B_idx(0,0,1,1): 1, get_B_idx(0,1,1,1): 1}
            ],
            [
                # C22 = P1 - P2 + P3 + P6.
                # P6 contributes +1 to C22.
                {
                    0: {get_C_idx(1,1,0,0): 1},
                    1: {get_C_idx(1,1,0,1): 1},
                    2: {get_C_idx(1,1,1,0): 1},
                    3: {get_C_idx(1,1,1,1): 1}
                }
            ]
        ),
        # P7 = (A12-A22)(B21+B22)
        (
            [
                {get_A_idx(0,1,0,0): 1, get_A_idx(1,1,0,0): -1},
                {get_A_idx(0,1,0,1): 1, get_A_idx(1,1,0,1): -1},
                {get_A_idx(0,1,1,0): 1, get_A_idx(1,1,1,0): -1},
                {get_A_idx(0,1,1,1): 1, get_A_idx(1,1,1,1): -1}
            ],
            [
                {get_B_idx(1,0,0,0): 1, get_B_idx(1,1,0,0): 1},
                {get_B_idx(1,0,0,1): 1, get_B_idx(1,1,0,1): 1},
                {get_B_idx(1,0,1,0): 1, get_B_idx(1,1,1,0): 1},
                {get_B_idx(1,0,1,1): 1, get_B_idx(1,1,1,1): 1}
            ],
            [
                # C11 = P1 + P4 - P5 + P7.
                # P7 contributes +1 to C11.
                {
                    0: {get_C_idx(0,0,0,0): 1},
                    1: {get_C_idx(0,0,0,1): 1},
                    2: {get_C_idx(0,0,1,0): 1},
                    3: {get_C_idx(0,0,1,1): 1}
                }
            ]
        )
    ]
    
    # Now, for each block term, apply Strassen to the 2x2 product L*R.
    # L and R are 2x2 matrices of linear combos.
    # Let L = [[L00, L01], [L10, L11]] and R = [[R00, R01], [R10, R11]].
    # Strassen terms for this 2x2 product:
    # M1 = (L00+L11)(R00+R11)
    # M2 = (L10+L11)R00
    # M3 = L00(R01-R11)
    # M4 = L11(R10-R00)
    # M5 = (L00+L01)R11
    # M6 = (L10-L00)(R00+R01)
    # M7 = (L01-L11)(R10+R11)
    
    # Result:
    # P00 = M1 + M4 - M5 + M7
    # P01 = M3 + M5
    # P10 = M2 + M4
    # P11 = M1 - M2 + M3 + M6
    
    # We need to map the local P positions (00, 01, 10, 11) to global C contributions.
    # The C_contribs in block_defs gives for each local P position (0,1,2,3) a dict of global C coeffs.
    # We will create 7 sub-terms for each block term.
    
    for b_idx, (L, R, C_contribs) in enumerate(block_defs):
        # Extract L, R elements
        L00, L01, L10, L11 = L[0], L[1], L[2], L[3]
        R00, R01, R10, R11 = R[0], R[1], R[2], R[3]
        
        # Define the 7 Strassen sub-products
        # Each is a tuple (U_dict, V_dict)
        
        # M1
        U_M1 = {}
        for k, v in L00.items(): U_M1[k] = U_M1.get(k, 0) + v
        for k, v in L11.items(): U_M1[k] = U_M1.get(k, 0) + v
        V_M1 = {}
        for k, v in R00.items(): V_M1[k] = V_M1.get(k, 0) + v
        for k, v in R11.items(): V_M1[k] = V_M1.get(k, 0) + v
        
        # M2
        U_M2 = {}
        for k, v in L10.items(): U_M2[k] = U_M2.get(k, 0) + v
        for k, v in L11.items(): U_M2[k] = U_M2.get(k, 0) + v
        V_M2 = dict(R00)
        
        # M3
        U_M3 = dict(L00)
        V_M3 = {}
        for k, v in R01.items(): V_M3[k] = V_M3.get(k, 0) + v
        for k, v in R11.items(): V_M3[k] = V_M3.get(k, 0) - v
        
        # M4
        U_M4 = dict(L11)
        V_M4 = {}
        for k, v in R10.items(): V_M4[k] = V_M4.get(k, 0) + v
        for k, v in R00.items(): V_M4[k] = V_M4.get(k, 0) - v
        
        # M5
        U_M5 = {}
        for k, v in L00.items(): U_M5[k] = U_M5.get(k, 0) + v
        for k, v in L01.items(): U_M5[k] = U_M5.get(k, 0) + v
        V_M5 = dict(R11)
        
        # M6
        U_M6 = {}
        for k, v in L10.items(): U_M6[k] = U_M6.get(k, 0) + v
        for k, v in L00.items(): U_M6[k] = U_M6.get(k, 0) - v
        V_M6 = {}
        for k, v in R00.items(): V_M6[k] = V_M6.get(k, 0) + v
        for k, v in R01.items(): V_M6[k] = V_M6.get(k, 0) + v
        
        # M7
        U_M7 = {}
        for k, v in L01.items(): U_M7[k] = U_M7.get(k, 0) + v
        for k, v in L11.items(): U_M7[k] = U_M7.get(k, 0) - v
        V_M7 = {}
        for k, v in R10.items(): V_M7[k] = V_M7.get(k, 0) + v
        for k, v in R11.items(): V_M7[k] = V_M7.get(k, 0) + v
        
        sub_products = [
            (U_M1, V_M1, [1, 0, 0, 1]), # Coeffs for P00, P01, P10, P11
            (U_M2, V_M2, [0, 0, 1, -1]),
            (U_M3, V_M3, [0, 1, 0, 1]),
            (U_M4, V_M4, [1, 0, 1, 0]),
            (U_M5, V_M5, [-1, 1, 0, 0]),
            (U_M6, V_M6, [0, 0, 0, 1]),
            (U_M7, V_M7, [1, 0, 0, 0])
        ]
        
        for s_idx, (U_sub, V_sub, coeffs) in enumerate(sub_products):
            rank_idx = b_idx * 7 + s_idx
            
            # Add to U
            for (i, s), coeff in U_sub.items():
                u_idx = i * 4 + s
                U[u_idx][rank_idx] += coeff
                
            # Add to V
            for (s, k), coeff in V_sub.items():
                v_idx = s * 4 + k
                V[v_idx][rank_idx] += coeff
                
            # Add to W
            # coeffs[p_idx] is the coefficient for local P position p_idx
            # C_contribs[p_idx] is a dict of global C coeffs
            for p_idx in range(4):
                sub_coeff = coeffs[p_idx]
                if sub_coeff == 0: continue
                global_c_contribs = C_contribs[p_idx]
                for (i, k), c_coeff in global_c_contribs.items():
                    w_idx = k * 4 + i # Trace convention: W[k*n+i]
                    W[w_idx][rank_idx] += sub_coeff * c_coeff
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
