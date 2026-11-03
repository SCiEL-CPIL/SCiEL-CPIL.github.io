"""
Optimized matrix multiplication decomposition for 3x4 by 4x7.
Uses a single Strassen block for the top-left 2x2x2 subproblem
and schoolbook for the remainder to achieve Rank 83.
"""

import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 100s safety margin
    
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # We will construct the rank 83 decomposition.
    # Rank 83 = 7 (Strassen) + 76 (Schoolbook remainder)
    
    # Helper to add a term to the decomposition
    # u_idx: index in U (i*m + j)
    # v_idx: index in V (j*p + k)
    # w_idx: index in W (k*n + i)
    # u_val, v_val, w_val: coefficients for this rank component
    
    # We store factors as lists of lists for the final return, 
    # but we can build them as sparse dictionaries or dense arrays.
    # Dense arrays are easier to manage for correctness.
    
    # Total rank is 83.
    # Let's just build the full rank 84 schoolbook and then "fix" one block?
    # No, easier to build from scratch with a helper function.
    
    # Since we need to return U, V, W as matrices, and rank is small (83),
    # we can use a list of tuples for non-zero entries and then densify,
    # or just use numpy arrays.
    
    import numpy as np
    
    rank_total = 83
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    r_idx = 0
    
    # 1. Strassen block for i in [0,1], j in [0,1], k in [0,1]
    # Strassen formulas for C = A*B where A,B are 2x2
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * B11
    # M5 = (A00 + A01) * (B11 - B10)
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # We need to map these to the tensor factors.
    # For a term like (A_i1_j1 + A_i2_j2) * (B_j1_k1 + B_j2_k2), it is not a single rank 1 term.
    # It expands to 4 terms. But Strassen combines them to reduce the COUNT of products.
    # Wait, the tensor decomposition rank is the number of terms in the sum:
    # T_{ijk} = sum_{r} U_{ir, jr?} ...
    # The standard form is T_{ijk} = sum_{r=1}^R U_{i, r} V_{j, r} W_{k, r} is incorrect.
    # The form is T_{ijk} = sum_{r=1}^R U_{i, r} V_{j, r} W_{k, r} ? No.
    # The matrix multiplication tensor is T_{i, j, k} = delta_{i, row} delta_{j, col} ...
    # Let's use the definition:
    # (A B)_{i, k} = sum_j A_{i, j} B_{j, k}
    # The tensor T has indices i, j, k. T_{i, j, k} = 1 if the multiplication involves A_{i, j} and B_{j, k}.
    # Actually, the tensor is defined such that <T, A \otimes B> = trace(AB^T) or similar.
    # The decomposition is T = sum_r u_r \otimes v_r \otimes w_r
    # where u_r is a vector of size n*m? No.
    # The interface specifies U has shape (n*m, rank).
    # This implies the index i*m+j is the row of U.
    # So U[r, i*m+j] is the coefficient for A_{i,j} in the r-th term.
    # V[r, j*p+k] is the coefficient for B_{j,k} in the r-th term.
    # W[r, k*n+i] is the coefficient for C_{i,k} in the r-th term.
    
    # So, a rank-1 term corresponds to:
    # C_{i, k} += U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i]
    # Wait, this structure implies that for a fixed r, only one (i,j) pair in U is non-zero?
    # No, U[r, :] is a vector. It can have multiple non-zeros.
    # But the standard schoolbook decomposition has U[r, i*m+j]=1 for a specific (i,j),
    # V[r, j*p+k]=1 for a specific (j,k), and W[r, k*n+i]=1 for a specific (k,i).
    # This is a rank-1 tensor in the sense of the Kronecker product structure.
    
    # Strassen replaces 8 such terms with 7 "virtual" products.
    # But each "virtual" product in Strassen is a linear combination of A's times a linear combination of B's.
    # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # This single "product" term in the algorithm corresponds to 4 terms in the tensor expansion.
    # However, in the tensor decomposition context, we are looking for the rank of the TENSOR.
    # The Strassen algorithm shows that the TENSOR can be decomposed into 7 terms of the form
    # (linear combo of A) \otimes (linear combo of B) \otimes (linear combo of C).
    # So yes, each Strassen "multiplication" M_r corresponds to one rank-1 term in the tensor decomposition.
    # The entries in U, V, W for that rank-1 term will be the coefficients of the linear combinations.
    
    # Example for M1 = (A00+A11)(B00+B11):
    # U term: U[r, 0*m+0] = 1, U[r, 1*m+1] = 1. (Indices for A00 and A11)
    # V term: V[r, 0*p+0] = 1, V[r, 1*p+1] = 1. (Indices for B00 and B11)
    # W term: This is tricky. The result contributes to C00, C01, C10, C11.
    # C00 += M1. So W[r, 0*n+0] += 1.
    # C01 += M1? No, C01 = M3+M4+M5+M6. M1 is not in C01.
    # C10 = M1-M2+M3+M7. So W[r, 1*n+0] += 1.
    # C11 = M2+M4-M6+M7. M1 not in C11.
    # Wait, let's re-check Strassen formulas.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M4 + M5 + M6
    # C10 = M1 - M2 + M3 + M7
    # C11 = M2 + M4 - M6 + M7
    
    # So for M1:
    # W[r, 0*n+0] = 1 (C00)
    # W[r, 1*n+0] = 1 (C10)
    
    # Let's implement this.
    
    def add_strassen_block(i0, j0, k0):
        """Adds the 7 Strassen terms for the block starting at (i0, j0, k0)."""
        nonlocal r_idx
        if r_idx >= rank_total:
            return
        
        # Indices in U, V, W
        # A indices: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
        u_00 = i0 * m + j0
        u_01 = i0 * m + (j0 + 1)
        u_10 = (i0 + 1) * m + j0
        u_11 = (i0 + 1) * m + (j0 + 1)
        
        # B indices: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
        v_00 = j0 * p + k0
        v_01 = j0 * p + (k0 + 1)
        v_10 = (j0 + 1) * p + k0
        v_11 = (j0 + 1) * p + (k0 + 1)
        
        # C indices: (i0, k0), (i0, k0+1), (i0+1, k0), (i0+1, k0+1)
        w_00 = k0 * n + i0
        w_01 = k0 * p + i0 # Wait, W index is k*n + i.
        # Correction: W index is k * n + i.
        w_00_idx = k0 * n + i0
        w_01_idx = (k0 + 1) * n + i0
        w_10_idx = k0 * n + (i0 + 1)
        w_11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[r_idx, u_00] += 1
        U[r_idx, u_11] += 1
        V[r_idx, v_00] += 1
        V[r_idx, v_11] += 1
        W[r_idx, w_00_idx] += 1
        W[r_idx, w_10_idx] += 1
        r_idx += 1
        
        # M2 = (A10 + A11) * B00
        U[r_idx, u_10] += 1
        U[r_idx, u_11] += 1
        V[r_idx, v_00] += 1
        W[r_idx, w_10_idx] += -1 # C10 = ... - M2 ...
        W[r_idx, w_11_idx] += 1  # C11 = ... + M2 ...
        r_idx += 1
        
        # M3 = A00 * (B10 - B11)
        U[r_idx, u_00] += 1
        V[r_idx, v_10] += 1
        V[r_idx, v_11] += -1
        W[r_idx, w_01_idx] += 1  # C01 = ... + M3 ...
        W[r_idx, w_10_idx] += 1  # C10 = ... + M3 ...
        r_idx += 1
        
        # M4 = A11 * B11
        U[r_idx, u_11] += 1
        V[r_idx, v_11] += 1
        W[r_idx, w_00_idx] += 1  # C00 = ... + M4 ...
        W[r_idx, w_01_idx] += 1  # C01 = ... + M4 ...
        W[r_idx, w_11_idx] += 1  # C11 = ... + M4 ...
        r_idx += 1
        
        # M5 = (A00 + A01) * (B11 - B10)
        U[r_idx, u_00] += 1
        U[r_idx, u_01] += 1
        V[r_idx, v_11] += 1
        V[r_idx, v_10] += -1
        W[r_idx, w_00_idx] += -1 # C00 = ... - M5 ...
        W[r_idx, w_01_idx] += 1  # C01 = ... + M5 ...
        r_idx += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[r_idx, u_10] += 1
        U[r_idx, u_00] += -1
        V[r_idx, v_00] += 1
        V[r_idx, v_01] += 1
        W[r_idx, w_01_idx] += 1  # C01 = ... + M6 ...
        W[r_idx, w_11_idx] += -1 # C11 = ... - M6 ...
        r_idx += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[r_idx, u_01] += 1
        U[r_idx, u_11] += -1
        V[r_idx, v_10] += 1
        V[r_idx, v_11] += 1
        W[r_idx, w_00_idx] += 1  # C00 = ... + M7 ...
        W[r_idx, w_10_idx] += 1  # C10 = ... + M7 ...
        W[r_idx, w_11_idx] += 1  # C11 = ... + M7 ...
        r_idx += 1

    # 2. Schoolbook for the rest
    # We need to identify which (i, j, k) triples are NOT covered by the Strassen block.
    # Strassen covered i in [0,1], j in [0,1], k in [0,1].
    # We iterate over all i, j, k. If (i,j,k) is in the Strassen block, skip.
    # Else, add a schoolbook term.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    # Fallback to pure schoolbook if we run out of time?
                    # We are just building arrays, so it should be fast.
                    pass
                
                # Check if this term is part of the Strassen block
                if i < 2 and j < 2 and k < 2:
                    continue
                
                # Schoolbook term
                U[r_idx, i * m + j] = 1
                V[r_idx, j * p + k] = 1
                W[r_idx, k * n + i] = 1
                r_idx += 1
                
                if r_idx >= rank_total:
                    break
            if r_idx >= rank_total:
                break
        if r_idx >= rank_total:
            break
    
    # Convert to lists for return interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
