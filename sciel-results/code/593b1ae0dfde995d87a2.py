"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_block_2x2(i0, j0, k0):
    """Return list of 7 (u_idx, v_idx, w_idx, coeff) tuples for a 2x2 Strassen block.
    i0, j0, k0 are the top-left corner indices.
    U index: i*m + j, V index: j*p + k, W index: k*n + i
    """
    n, m, p = DIMENSIONS
    terms = []
    # M1 = (A[i0,j0]+A[i0+1,j0]) * (B[j0,k0]+B[j0+1,k0])
    # M2 = (A[i0,j0+1]+A[i0+1,j0+1]) * B[j0,k0]
    # M3 = A[i0,j0] * (B[j0+1,k0]-B[j0,k0])
    # M4 = A[i0+1,j0] * (B[j0+1,k0+1]-B[j0,k0])
    # M5 = (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
    # M6 = (A[i0+1,j0+1]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
    # M7 = (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0+1]-B[j0,k0])
    
    # For each of the 7 products, we need to add contributions to U, V, W
    # U[i,j] gets coefficient a, V[j,k] gets coefficient b, W[k,i] gets coefficient c
    # such that sum over all terms: U[i,j]*V[j,k] contributes to W[k,i]
    
    # M1: (A00+A10)(B00+B10) -> contributes to C00, C01, C10, C11
    # C00 += M1, C01 -= M1 (wait, let me re-derive)
    # Standard Strassen:
    # C00 = M1 - M4 + M6 - M7
    # C01 = M2 + M4
    # C10 = M3 + M5
    # C11 = M1 - M3 + M5 - M6
    
    # So for each M_r, we add its contribution to the appropriate C entries.
    # M1: +C00, +C11
    # M2: +C01
    # M3: +C10, -C11
    # M4: -C00, +C01
    # M5: +C10, +C11
    # M6: +C00, -C11
    # M7: -C00
    
    # For each M_r, we have U factors and V factors.
    # M1: U has (i0,j0):1, (i0+1,j0):1; V has (j0,k0):1, (j0+1,k0):1
    #     This single term contributes to all four C entries with different signs.
    #     But in tensor decomposition, each term is a rank-1 tensor.
    #     So we need to split M1 into separate terms for each C entry it contributes to.
    
    # Actually, the standard way: each M_r is a product of two linear combinations.
    # In the tensor decomposition, each "M_r" corresponds to one rank-1 term where
    # U has the A-side linear combo, V has the B-side linear combo, and W has the
    # C-side linear combo (with appropriate signs).
    
    # M1: U: (i0,j0):1, (i0+1,j0):1; V: (j0,k0):1, (j0+1,k0):1; W: (k0,i0):1, (k0+1,i0+1):1
    # M2: U: (i0,j0+1):1, (i0+1,j0+1):1; V: (j0,k0):1; W: (k0,i0+1):1
    # M3: U: (i0,j0):1; V: (j0+1,k0):1, (j0,k0):-1; W: (k0+1,i0):1, (k0,i0+1):-1
    # M4: U: (i0+1,j0):1; V: (j0+1,k0+1):1, (j0,k0):-1; W: (k0+1,i0):-1, (k0+1,i0+1):1
    # M5: U: (i0,j0):1, (i0,j0+1):1; V: (j0+1,k0+1):1; W: (k0+1,i0):1, (k0+1,i0+1):1
    # M6: U: (i0+1,j0+1):1, (i0,j0):-1; V: (j0,k0):1, (j0,k0+1):1; W: (k0,i0):1, (k0,i0+1):-1
    # M7: U: (i0,j0+1):1, (i0+1,j0+1):-1; V: (j0+1,k0+1):1, (j0+1,k0):-1; W: (k0,i0):-1
    
    # Each of these is one rank-1 term in the decomposition.
    
    # Let's define them as (U_dict, V_dict, W_dict) where each is a dict from index to coeff
    M1_U = {(i0, j0): 1, (i0+1, j0): 1}
    M1_V = {(j0, k0): 1, (j0+1, k0): 1}
    M1_W = {(k0, i0): 1, (k0+1, i0+1): 1}
    
    M2_U = {(i0, j0+1): 1, (i0+1, j0+1): 1}
    M2_V = {(j0, k0): 1}
    M2_W = {(k0, i0+1): 1}
    
    M3_U = {(i0, j0): 1}
    M3_V = {(j0+1, k0): 1, (j0, k0): -1}
    M3_W = {(k0+1, i0): 1, (k0, i0+1): -1}
    
    M4_U = {(i0+1, j0): 1}
    M4_V = {(j0+1, k0+1): 1, (j0, k0): -1}
    M4_W = {(k0+1, i0): -1, (k0+1, i0+1): 1}
    
    M5_U = {(i0, j0): 1, (i0, j0+1): 1}
    M5_V = {(j0+1, k0+1): 1}
    M5_W = {(k0+1, i0): 1, (k0+1, i0+1): 1}
    
    M6_U = {(i0+1, j0+1): 1, (i0, j0): -1}
    M6_V = {(j0, k0): 1, (j0, k0+1): 1}
    M6_W = {(k0, i0): 1, (k0, i0+1): -1}
    
    M7_U = {(i0, j0+1): 1, (i0+1, j0+1): -1}
    M7_V = {(j0+1, k0+1): 1, (j0+1, k0): -1}
    M7_W = {(k0, i0): -1}
    
    blocks = [
        (M1_U, M1_V, M1_W),
        (M2_U, M2_V, M2_W),
        (M3_U, M3_V, M3_W),
        (M4_U, M4_V, M4_W),
        (M5_U, M5_V, M5_W),
        (M6_U, M6_V, M6_W),
        (M7_U, M7_V, M7_W),
    ]
    return blocks

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Safety margin before 120s limit
    
    # Collect all terms: list of (U_dict, V_dict, W_dict)
    all_terms = []
    
    # Use Strassen blocks for the 2x2 submatrices in the top-left 2x2x2 region
    # i in {0,1}, j in {0,1}, k in {0,1}
    strassen_blocks = _strassen_block_2x2(0, 0, 0)
    all_terms.extend(strassen_blocks)
    
    # Now handle remaining (i,j,k) triples with schoolbook terms
    # We need to cover all (i,j,k) where the triple is not fully in the 2x2x2 Strassen block
    # The Strassen block covers i in {0,1}, j in {0,1}, k in {0,1}
    # Remaining: any triple where i>=2 or j>=2 or k>=2
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue  # Already covered by Strassen block
                # Schoolbook term
                all_terms.append(
                    ({(i, j): 1}, {(j, k): 1}, {(k, i): 1})
                )
    
    rank = len(all_terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_dict, v_dict, w_dict) in enumerate(all_terms):
        for (i, j), coeff in u_dict.items():
            if 0 <= i < n and 0 <= j < m:
                U[i * m + j][col] = coeff
        for (j, k), coeff in v_dict.items():
            if 0 <= j < m and 0 <= k < p:
                V[j * p + k][col] = coeff
        for (k, i), coeff in w_dict.items():
            if 0 <= k < p and 0 <= i < n:
                W[k * n + i][col] = coeff
    
    # Verify the decomposition
    if time.time() < deadline - 5:
        try:
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            
            # Check: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] should equal W[k*n+i, c]... 
            # Wait, the convention is: (A*B)[i,k] = sum_j A[i,j] * B[j,k]
            # In tensor form: T[i,j,k] = delta_{i,i'}delta_{j,j'}delta_{k,k'} for schoolbook
            # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # Wait, that's not right either. Let me re-read the contract.
            # "output factor follows the official trace convention W[k*n+i,r]"
            # The standard tensor decomposition for matrix mult: 
            # (AB)[i,k] = sum_j A[i,j] B[j,k]
            # As a tensor: T_{i,j,k} where the multiplication tensor has T_{i,j,k} = 1 if we're doing A[i,j]*B[j,k]->C[i,k]
            # Rank-1 decomposition: T = sum_c U_c * V_c * W_c where
            # U_c[i,j] * V_c[j,k] * W_c[k,i]... hmm, the W index convention is W[k*n+i, r]
            # So W_r[k*n+i] is the coefficient for the (k,i) pair in the output.
            # The decomposition should satisfy:
            # For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] = delta_{i,k}... no.
            
            # Actually, the matrix multiplication tensor M has M[i,j,k] = 1 for all i,j,k
            # (since (AB)[i,k] = sum_j A[i,j] B[j,k], each (i,j,k) triple contributes A[i,j]B[j,k] to C[i,k])
            # The rank-1 decomposition: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # Wait, that would be a 3-way product, but matrix mult is bilinear, not trilinear.
            
            # I think the convention is: the multiplication tensor is defined as
            # T[i,j,k] = 1 for all i,j,k (since each input pair (i,j) and (j,k) produces output (i,k))
            # And the rank-1 decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # This is a trilinear form, but for matrix mult, T is actually the identity in the sense that
            # the tensor for matrix multiplication has T_{i,j,k} = 1 for all valid (i,j,k).
            
            # Let me just verify numerically with random matrices.
            np.random.seed(42)
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test
            
            C_computed = np.zeros((n, p))
            for c in range(rank):
                Ac = U_arr[:, c].reshape(n, m)
                Bc = V_arr[:, c].reshape(m, p)
                Wc = W_arr[:, c].reshape(p, n)
                # The contribution to C from term c is: (Ac * Bc) * Wc element-wise?
                # No, I think the decomposition means:
                # C = sum_c (Ac * Bc) where Ac[i,j] = U[i*m+j, c], Bc[j,k] = V[j*p+k, c]
                # and W is used to extract the output. Hmm.
                
            # Let me think differently. The standard formulation:
            # The matrix multiplication tensor T is a 3-tensor with T[i,j,k] = 1 for all i,j,k
            # (each entry is 1 because A[i,j]*B[j,k] contributes to C[i,k])
            # A rank-1 decomposition is T = sum_c u_c * v_c * w_c where
            # u_c is a vector of length n*m, v_c of length m*p, w_c of length p*n
            # and T[i,j,k] = sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
            
            # Let's verify this.
            T_computed = np.zeros((n, m, p))
            for c in range(rank):
                u_c = U_arr[:, c].reshape(n, m)
                v_c = V_arr[:, c].reshape(m, p)
                w_c = W_arr[:, c].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_computed[i, j, k] += u_c[i, j] * v_c[j, k] * w_c[k, i]
            
            T_expected = np.ones((n, m, p))
            if not np.allclose(T_computed, T_expected, atol=1e-10):
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
                return {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
