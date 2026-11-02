"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-26 tensor decomposition for 3x3x3 matrix multiplication.
    
    Strategy:
    1. Use Strassen's algorithm (rank 7) for the 2x2 sub-block (indices 0,1).
    2. Use standard rank-1 terms for all elements involving index 2 (the 'border').
    3. Combine these disjoint supports to form a total rank of 7 + 19 = 26.
    
    Returns:
        Dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    R = 26
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, R), dtype=np.int8)
    V = np.zeros((m * p, R), dtype=np.int8)
    W = np.zeros((p * n, R), dtype=np.int8)
    
    # Helper functions for flattened indices
    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i
    
    # --- Part 1: Strassen Decomposition for 2x2 Block (Indices 0,1) ---
    # Defined as a list of (U_indices, V_indices, W_indices) tuples
    # Each tuple represents one rank-1 term: sum_{(i,j) in U_ind} U[idx_u(i,j)] * V[idx_v(j,k)] * W[idx_w(k,i)]
    # The entries are implicitly 1 unless specified, but here we just list the non-zero positions.
    # Actually, for simplicity and clarity in the AST, we list (u_idx, v_idx, w_idx) where value is 1,
    # and handle negative signs by adding multiple terms or explicit values if needed.
    # Strassen uses +/- 1. We can represent this by having entries in the list.
    
    # Format: ( [(u_idx, val), ...], [(v_idx, val), ...], [(w_idx, val), ...] )
    strassen_terms = [
        # m1 = (a00 + a11)(b00 + b11) -> c00 += 1, c11 += 1
        ([(idx_u(0,0), 1), (idx_u(1,1), 1)],
         [(idx_v(0,0), 1), (idx_v(1,1), 1)],
         [(idx_w(0,0), 1), (idx_w(1,1), 1)]),
         
        # m2 = (a10 + a11)b00 -> c10 += 1, c11 -= 1
        ([(idx_u(1,0), 1), (idx_u(1,1), 1)],
         [(idx_v(0,0), 1)],
         [(idx_w(1,0), 1), (idx_w(1,1), -1)]),
         
        # m3 = a00(b01 - b11) -> c01 += 1, c11 += 1
        ([(idx_u(0,0), 1)],
         [(idx_v(0,1), 1), (idx_v(1,1), -1)],
         [(idx_w(1,0), 1), (idx_w(1,1), 1)]),
         
        # m4 = a11(b10 - b00) -> c00 += 1, c10 += 1
        ([(idx_u(1,1), 1)],
         [(idx_v(1,0), 1), (idx_v(0,0), -1)],
         [(idx_w(0,0), 1), (idx_w(0,1), 1)]),
         
        # m5 = (a00 + a01)b11 -> c00 -= 1, c01 += 1
        ([(idx_u(0,0), 1), (idx_u(0,1), 1)],
         [(idx_v(1,1), 1)],
         [(idx_w(0,0), -1), (idx_w(1,0), 1)]),
         
        # m6 = (a10 - a00)(b00 + b01) -> c11 += 1
        ([(idx_u(1,0), 1), (idx_u(0,0), -1)],
         [(idx_v(0,0), 1), (idx_v(0,1), 1)],
         [(idx_w(1,1), 1)]),
         
        # m7 = (a01 - a11)(b10 + b11) -> c00 += 1
        ([(idx_u(0,1), 1), (idx_u(1,1), -1)],
         [(idx_v(1,0), 1), (idx_v(1,1), 1)],
         [(idx_w(0,0), 1)]),
    ]
    
    # Populate Strassen terms
    for r, (u_terms, v_terms, w_terms) in enumerate(strassen_terms):
        for idx, val in u_terms:
            U[idx, r] = val
        for idx, val in v_terms:
            V[idx, r] = val
        for idx, val in w_terms:
            W[idx, r] = val
            
    # --- Part 2: Standard Terms for Border (Index 2 involved) ---
    # Total standard multiplications needed: 27 (total) - 8 (handled by 2x2 block) = 19.
    # We iterate over all (i, j, k) where i, j, k in {0,1,2}.
    # If i<2 and j<2 and k<2, skip (handled by Strassen).
    # Else, add a rank-1 term e_{(i,j)} e_{(j,k)} e_{(k,i)}.
    
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                
                # Add rank 1 term
                U[idx_u(i, j), col] = 1
                V[idx_v(j, k), col] = 1
                W[idx_w(k, i), col] = 1
                col += 1
                
    if col != R:
        raise ValueError(f"Rank calculation error: expected {R}, got {col}")
    
    # --- Verification ---
    # Check if sum_r U[:, :, r] * V[:, :, r] * W[:, :, r] == Tensor M
    # M[i, j, k] = 1 if i==j and j==k is NOT the condition.
    # Matrix Multiplication Tensor T is defined such that C = A*B.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # The tensor T_{i,j,k} contributes to C_{i,k} when multiplying U_{i,j} V_{j,k} W_{k,i}?
    # Wait, the standard decomposition is T = sum u_r \otimes v_r \otimes w_r.
    # The tensor entry T_{i,j,k} corresponds to the coefficient of x_i y_j z_k in the polynomial?
    # No, for matrix multiplication, the tensor is T_{i,j,k} = 1 if the indices match the multiplication structure.
    # Specifically, the multiplication tensor M has entries M[i, j, k] = 1 if we are computing C_{i,k} using A_{i,j} and B_{j,k}.
    # But the tensor decomposition usually maps to the structure:
    # sum_r U[i, j, r] V[j, k, r] W[k, i, r] = delta_{i,i'} delta_{j,j'} delta_{k,k'}?
    # Let's check the task description: "W[k*n+i,r]".
    # The standard tensor for matmul is T[i, j, k] = 1. No, that's the all-ones tensor.
    # The matmul tensor is defined by T[i, j, k] = 1 if ...?
    # Actually, the tensor is T such that T_{i,j,k} = 1 for all i,j,k in the identity permutation?
    # No. The tensor is T_{i,j,k} = 1 if the indices satisfy the condition for multiplication.
    # Let's verify the identity:
    # We want U[i,j] V[j,k] W[k,i] summed over r to equal 1 if this term is part of the matmul, 0 otherwise?
    # No, the tensor M for matmul has M[i,j,k] = 1 for all i,j,k? No.
    # M[i,j,k] = 1 is the tensor of rank 1? No.
    # The tensor for multiplying two n x n matrices is T_{i,j,k} = 1 if i,j,k are indices in A, B, C.
    # The condition is that the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # So the tensor T has T[i, j, k] = 1 for all valid i,j,k?
    # Yes, the support of the tensor is all (i,j,k). The value is 1.
    # Wait, if T[i,j,k]=1 for all i,j,k, then sum_r U[i,j,r]V[j,k,r]W[k,i,r] must equal 1 for ALL i,j,k.
    # Let's check if our decomposition satisfies this.
    
    T_reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # The target tensor T_target has T_target[i,j,k] = 1 for all i,j,k in range(3).
    T_target = np.ones((n, m, p), dtype=np.int8)
    
    if not np.array_equal(T_reconstructed, T_target):
        # Debugging: Find where they differ
        diffs = np.argwhere(T_reconstructed != T_target)
        # This should not happen if the decomposition is correct.
        # Note: Strassen handles the 2x2 block correctly.
        # The standard terms handle the rest.
        # Are there overlaps?
        # Strassen uses i,j in {0,1} and k in {0,1}.
        # Standard uses cases where i==2 or j==2 or k==2.
        # The sets of (i,j,k) covered by Strassen are {0,1}^3.
        # The sets covered by Standard are {0,1,2}^3 \ {0,1}^3.
        # They are disjoint.
        # Within Strassen, do they sum to 1 for all (i,j,k) in {0,1}^3?
        # Yes, Strassen is an exact decomposition of the 2x2 matmul tensor.
        # Within Standard, each (i,j,k) has exactly one term with 1s, so sum is 1.
        # Thus, the total sum is 1 for all (i,j,k).
        
        # If it fails, we raise an error to fail fast.
        raise AssertionError("Tensor decomposition verification failed.")
        
    # Check deadline
    if time.time() - start_time > 110:
         # This is a static construction, so we shouldn't be close to the limit,
         # but we return immediately to be safe.
         pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
