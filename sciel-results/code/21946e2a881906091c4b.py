import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
_start_time = time.time()
_DEADLINE = 140  # Safety margin from 150s limit

def _deadline_ok():
    return time.time() - _start_time < _DEADLINE

def _verify_decomp(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    T_decomp = np.zeros((n, m, p))
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        # Outer product
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_decomp[i, j, k] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
    
    # Expected tensor: T[i,j,k] = 1 if j == k else 0... 
    # Wait, for matrix multiplication C = A*B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = δ_{j,k}? No...
    # Actually, the multiplication tensor has T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C.
    # So T[i,j,k] = 1 for all valid i,j,k where j is the shared index.
    # But that's not right either. Let me think again.
    
    # The matrix multiplication tensor T ∈ R^{n×m×p} is defined by:
    # T[i,j,k] = 1 if the (i,j) element of first matrix and (j,k) element of second matrix multiply to contribute to (i,k) element of product.
    # Actually, T[i,j,k] = δ_{j,k} is wrong.
    
    # The correct definition: For multiplying A (n×m) by B (m×p) to get C (n×p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i,j,k (since every combination contributes).
    # No wait, that's not right. The tensor is such that:
    # sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = sum_k C[i,k]
    # So T[i,j,k] must be such that when contracted with A and B, we get C.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k? That would mean rank = n*m*p.
    
    # Actually yes! The standard matrix multiplication tensor has T[i,j,k] = 1 for all valid indices.
    # The rank of this tensor is what we're trying to minimize.
    
    # So the expected tensor is all ones:
    T_expected = np.ones((n, m, p))
    
    if not np.allclose(T_decomp, T_expected, atol=1e-10):
        return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
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
    
    return U, V, W, rank

def _strassen_222_block(i0, j0, k0, U, V, W, col_offset):
    """
    Add a Strassen 2x2x2 block at positions (i0, j0, k0).
    This replaces 8 schoolbook terms with 7 Strassen terms.
    Returns the new column offset.
    """
    # Strassen's algorithm for 2x2 matrix multiplication:
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 + M7
    # where:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # The 7 rank-1 terms for Strassen:
    # Each term is: (linear combo of A entries) * (linear combo of B entries) * (contribution to C)
    
    # For the block at (i0, j0, k0), the indices are:
    # A rows: i0, i0+1
    # A cols/B rows: j0, j0+1
    # B cols/C cols: k0, k0+1
    
    # Strassen terms (U coefficient, V coefficient, W coefficient):
    # M1: (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1]) -> contributes to C[i0,k0] and C[i0+1,k0+1]
    # Actually, each Strassen term is a rank-1 tensor, so it has one U, one V, one W vector.
    
    # Let me define the 7 terms properly:
    # Term 1: M1 = (A11+A22)*(B11+B22)
    #   U: [1,0,1,0] for A (i0,j0)=1, (i0,j0+1)=0, (i0+1,j0)=0, (i0+1,j0+1)=1
    #   V: [1,0,0,1] for B (j0,k0)=1, (j0,k0+1)=0, (j0+1,k0)=0, (j0+1,k0+1)=1
    #   W: contributes to C[i0,k0] with +1, C[i0,k0+1] with 0, C[i0+1,k0] with 0, C[i0+1,k0+1] with +1
    #   Wait, M1 contributes to both C11 and C22.
    #   In the tensor decomposition, each rank-1 term is:
    #   u_i * v_j * w_k where u is for A[i,j], v is for B[j,k], w is for C[i,k]
    #   But C is the output, so w should be indexed by (i,k).
    
    # Let me reconsider. The tensor T[i,j,k] has indices:
    # i: row of A (0..n-1)
    # j: col of A / row of B (0..m-1)
    # k: col of B (0..p-1)
    
    # A rank-1 term is: U[i,j] * V[j,k] * W[i,k] (with W indexed by output position)
    # But in the problem statement, W is indexed by (k*n+i, r), so W is p*n x rank.
    # So W[k*n+i, r] corresponds to output position (i,k).
    
    # For Strassen 2x2x2 at offset (i0, j0, k0):
    # The 7 terms are:
    
    terms = []
    
    # M1 = (A11+A22)*(B11+B22) -> contributes +1 to C11, +1 to C22
    # U: A[i0,j0]=1, A[i0+1,j0+1]=1, others 0
    # V: B[j0,k0]=1, B[j0+1,k0+1]=1, others 0
    # W: C[i0,k0]=1, C[i0+1,k0+1]=1, others 0
    terms.append((
        [(i0, j0), (i0+1, j0+1)],           # U positions
        [(j0, k0), (j0+1, k0+1)],           # V positions
        [(i0, k0), (i0+1, k0+1)]            # W positions (as (i,k) pairs)
    ))
    
    # M2 = (A21+A22)*B11 -> contributes +1 to C21
    terms.append((
        [(i0+1, j0), (i0+1, j0+1)],
        [(j0, k0)],
        [(i0+1, k0)]
    ))
    
    # M3 = A11*(B12-B22) -> contributes +1 to C12, -1 to C22
    terms.append((
        [(i0, j0)],
        [(j0, k0+1), (j0+1, k0+1)],
        [(i0, k0+1), (i0+1, k0+1)]
    ))
    # But M3 has coefficients +1 and -1 in V
    # So we need to handle signed coefficients
    
    # Let me redo this with explicit coefficients:
    # Each term is a list of (position, coefficient) for U, V, W
    
    strassen_terms = []
    
    # M1 = (A11+A22)*(B11+B22)
    # U: A[i0,j0]=1, A[i0+1,j0+1]=1
    # V: B[j0,k0]=1, B[j0+1,k0+1]=1
    # W: C[i0,k0]=1, C[i0+1,k0+1]=1
    strassen_terms.append((
        {i0*m + j0: 1, (i0+1)*m + j0+1: 1},
        {j0*p + k0: 1, (j0+1)*p + k0+1: 1},
        {k0*n + i0: 1, (k0+1)*n + i0+1: 1}
    ))
    
    # M2 = (A21+A22)*B11
    # U: A[i0+1,j0]=1, A[i0+1,j0+1]=1
    # V: B[j0,k0]=1
    # W: C[i0+1,k0]=1
    strassen_terms.append((
        {(i0+1)*m + j0: 1, (i0+1)*m + j0+1: 1},
        {j0*p + k0: 1},
        {k0*n + i0+1: 1}
    ))
    
    # M3 = A11*(B12-B22)
    # U: A[i0,j0]=1
    # V: B[j0,k0+1]=1, B[j0+1,k0+1]=-1
    # W: C[i0,k0+1]=1, C[i0+1,k0+1]=-1
    strassen_terms.append((
        {i0*m + j0: 1},
        {j0*p + k0+1: 1, (j0+1)*p + k0+1: -1},
        {k0*n + i0: 1, (k0+1)*n + i0: -1}  # Wait, C[i0+1,k0+1] not C[i0,k0+1]
    ))
    # Correction: W should be C[i0,k0+1]=1 and C[i0+1,k0+1]=-1
    strassen_terms[2] = (
        {i0*m + j0: 1},
        {j0*p + k0+1: 1, (j0+1)*p + k0+1: -1},
        {k0*n + i0: 1, (k0+1)*n + i0+1: -1}
    )
    
    # M4 = A22*(B21-B11)
    # U: A[i0+1,j0+1]=1
    # V: B[j0+1,k0]=1, B[j0,k0]=-1
    # W: C[i0+1,k0]=1
    strassen_terms.append((
        {(i0+1)*m + j0+1: 1},
        {(j0+1)*p + k0: 1, j0*p + k0: -1},
        {k0*n + i0+1: 1}
    ))
    
    # M5 = (A11+A12)*B22
    # U: A[i0,j0]=1, A[i0,j0+1]=1
    # V: B[j0+1,k0+1]=1
    # W: C[i0,k0+1]=1, C[i0+1,k0+1]=1
    strassen_terms.append((
        {i0*m + j0: 1, i0*m + j0+1: 1},
        {(j0+1)*p + k0+1: 1},
        {k0*n + i0: 1, (k0+1)*n + i0: 1}  # Wait, C[i0+1,k0+1]
    ))
    strassen_terms[4] = (
        {i0*m + j0: 1, i0*m + j0+1: 1},
        {(j0+1)*p + k0+1: 1},
        {k0*n + i0: 1, (k0+1)*n + i0+1: 1}
    )
    
    # M6 = (A21-A11)*(B11+B12)
    # U: A[i0+1,j0]=1, A[i0,j0]=-1
    # V: B[j0,k0]=1, B[j0,k0+1]=1
    # W: C[i0+1,k0+1]=1
    strassen_terms.append((
        {(i0+1)*m + j0: 1, i0*m + j0: -1},
        {j0*p + k0: 1, j0*p + k0+1: 1},
        {(k0+1)*n + i0+1: 1}
    ))
    
    # M7 = (A12-A22)*(B21+B22)
    # U: A[i0,j0+1]=1, A[i0+1,j0+1]=-1
    # V: B[j0+1,k0]=1, B[j0+1,k0+1]=1
    # W: C[i0,k0]=-1, C[i0+1,k0+1]=1
    strassen_terms.append((
        {i0*m + j0+1: 1, (i0+1)*m + j0+1: -1},
        {(j0+1)*p + k0: 1, (j0+1)*p + k0+1: 1},
        {k0*n + i0: -1, (k0+1)*n + i0+1: 1}
    ))
    
    # Now add these 7 terms to U, V, W
    for u_dict, v_dict, w_dict in strassen_terms:
        for idx, coeff in u_dict.items():
            U[idx][col_offset] += coeff
        for idx, coeff in v_dict.items():
            V[idx][col_offset] += coeff
        for idx, coeff in w_dict.items():
            W[idx][col_offset] += coeff
        col_offset += 1
    
    return col_offset

def construct():
    n, m, p = DIMENSIONS
    _start_time_global = time.time()
    deadline = _start_time_global + 140
    
    # Start with schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to find 2x2x2 blocks where we can apply Strassen
    # For n=2, m=5, p=5:
    # We can have Strassen blocks at (i0, j0, k0) where i0 in {0}, j0 in {0,1,2,3}, k0 in {0,1,2,3}
    # But i0 must be such that i0+1 < n, so i0=0 only
    # j0 must be such that j0+1 < m, so j0 in {0,1,2,3}
    # k0 must be such that k0+1 < p, so k0 in {0,1,2,3}
    
    # But we can't just apply Strassen independently to overlapping blocks.
    # We need to partition the tensor into non-overlapping 2x2x2 blocks and schoolbook terms.
    
    # Strategy: Try different partitions and see which gives the lowest rank.
    # A 2x2x2 block at (i0,j0,k0) covers:
    # A indices: (i0,j0), (i0,j0+1), (i0+1,j0), (i0+1,j0+1)
    # B indices: (j0,k0), (j0,k0+1), (j0+1,k0), (j0+1,k0+1)
    # C indices: (i0,k0), (i0,k0+1), (i0+1,k0), (i0+1,k0+1)
    
    # For n=2, the i dimension is fully covered by one block (i0=0).
    # So we can have multiple Strassen blocks with different (j0, k0) as long as they don't overlap in j or k.
    
    # Actually, blocks can overlap in the tensor as long as the overall decomposition is correct.
    # But for simplicity, let's try non-overlapping blocks.
    
    # For m=5, p=5, n=2:
    # Possible non-overlapping 2x2 blocks in the j-k plane:
    # We can tile the 5x5 j-k plane with 2x2 blocks and 1x1 or 1x2 or 2x1 remainders.
    
    # Let's try: Use Strassen for 2x2 blocks at (0, j0, k0) where j0 and k0 are even (0, 2)
    # This covers j=0,1 and j=2,3, leaving j=4 uncovered.
    # Similarly for k.
    
    # Partition 1: Strassen at (0,0,0), (0,0,2), (0,2,0), (0,2,2)
    # Covers j=0,1,2,3 and k=0,1,2,3
    # Remaining: j=4 (all k) and k=4 (all j except where covered)
    
    # Let's implement a greedy approach:
    # 1. Start with schoolbook
    # 2. Try to replace 2x2x2 sub-blocks with Strassen
    # 3. Verify the result
    
    # Actually, let me just try a specific decomposition:
    # Use Strassen for the 2x2x2 block at (0,0,0)
    # Use schoolbook for everything else
    
    # But this is tricky because the Strassen block and schoolbook terms would double-count.
    
    # Better approach: Build the decomposition from scratch.
    # For each (i,j,k) triple, we need the sum of contributions to equal 1.
    
    # Let me try a different approach:
    # Decompose the 5x5 into blocks:
    # - 2x2 block at top-left: use Strassen
    # - Remaining parts: use schoolbook
    
    # The 2x5 matrix A: rows 0,1; cols 0-4
    # The 5x5 matrix B: rows 0-4; cols 0-4
    
    # C = A*B, C has rows 0,1; cols 0-4
    
    # Split B into:
    # B = [B_top_left (2x2) | B_top_right (2x3)]
    #     [B_bottom_left (3x2) | B_bottom_right (3x3)]
    
    # C[:,0:2] = A[:,0:2]*B[0:2,:] + A[:,2:5]*B[2:5,:]
    # C[:,2:5] = A[:,0:2]*B[0:2,2:5] + A[:,2:5]*B[2:5,2:5]
    
    # Hmm, this is still complex. Let me just implement the Strassen replacement for one 2x2x2 block and see if verification passes.
    
    # Try: Replace the schoolbook terms for (i=0,1; j=0,1; k=0,1) with Strassen.
    # The schoolbook terms for this block are 8 terms.
    # Strassen uses 7 terms.
    # So we save 1 rank.
    
    # New rank: 50 - 1 = 49
    
    # Let's build this:
    U_new = [[0] * 49 for _ in range(n * m)]
    V_new = [[0] * 49 for _ in range(m * p)]
    W_new = [[0] * 49 for _ in range(p * n)]
    
    col = 0
    
    # Add Strassen block at (0, 0, 0)
    col = _strassen_222_block(0, 0, 0, U_new, V_new, W_new, col)
    
    # Add schoolbook for all other (i,j,k) not in the 2x2x2 block
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if in the Strassen block
                if i in [0, 1] and j in [0, 1] and k in [0, 1]:
                    continue
                U_new[i * m + j][col] = 1
                V_new[j * p + k][col] = 1
                W_new[k * n + i][col] = 1
                col += 1
    
    # Verify
    if _verify_decomp(U_new, V_new, W_new, n, m, p):
        best_rank = 49
        best_U, best_V, best_W = U_new, V_new, W_new
    
    # Try more Strassen blocks
    # Can we add another Strassen block at (0, 2, 0)?
    # This would cover j=2,3 and k=0,1
    # Total Strassen blocks: 2, saving 2 rank
    # New rank: 50 - 2 = 48
    
    if _deadline_ok():
        U_new2 = [[0] * 48 for _ in range(n * m)]
        V_new2 = [[0] * 48 for _ in range(m * p)]
        W_new2 = [[0] * 48 for _ in range(p * n)]
        
        col2 = 0
        col2 = _strassen_222_block(0, 0, 0, U_new2, V_new2, W_new2, col2)
        col2 = _strassen_222_block(0, 2, 0, U_new2, V_new2, W_new2, col2)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Skip if in any Strassen block
                    if i in [0, 1] and j in [0, 1] and k in [0, 1]:
                        continue
                    if i in [0, 1] and j in [2, 3] and k in [0, 1]:
                        continue
                    U_new2[i * m + j][col2] = 1
                    V_new2[j * p + k][col2] = 1
                    W_new2[k * n + i][col2] = 1
                    col2 += 1
        
        if _verify_decomp(U_new2, V_new2, W_new2, n, m, p):
            best_rank = 48
            best_U, best_V, best_W = U_new2, V_new2, W_new2
    
    # Try 3 Strassen blocks: (0,0,0), (0,2,0), (0,0,2)
    if _deadline_ok():
        U_new3 = [[0] * 47 for _ in range(n * m)]
        V_new3 = [[0] * 47 for _ in range(m * p)]
        W_new3 = [[0] * 47 for _ in range(p * n)]
        
        col3 = 0
        col3 = _strassen_222_block(0, 0, 0, U_new3, V_new3, W_new3, col3)
        col3 = _strassen_222_block(0, 2, 0, U_new3, V_new3, W_new3, col3)
        col3 = _strassen_222_block(0, 0, 2, U_new3, V_new3, W_new3, col3)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i in [0, 1] and j in [0, 1] and k in [0, 1]:
                        continue
                    if i in [0, 1] and j in [2, 3] and k in [0, 1]:
                        continue
                    if i in [0, 1] and j in [0, 1] and k in [2, 3]:
                        continue
                    U_new3[i * m + j][col3] = 1
                    V_new3[j * p + k][col3] = 1
                    W_new3[k * n + i][col3] = 1
                    col3 += 1
        
        if _verify_decomp(U_new3, V_new3, W_new3, n, m, p):
            best_rank = 47
            best_U, best_V, best_W = U_new3, V_new3, W_new3
    
    # Try 4 Strassen blocks: (0,0,0), (0,2,0), (0,0,2), (0,2,2)
    if _deadline_ok():
        U_new4 = [[0] * 46 for _ in range(n * m)]
        V_new4 = [[0] * 46 for _ in range(m * p)]
        W_new4 = [[0] * 46 for _ in range(p * n)]
        
        col4 = 0
        col4 = _strassen_222_block(0, 0, 0, U_new4, V_new4, W_new4, col4)
        col4 = _strassen_222_block(0, 2, 0, U_new4, V_new4, W_new4, col4)
        col4 = _strassen_222_block(0, 0, 2, U_new4, V_new4, W_new4, col4)
        col4 = _strassen_222_block(0, 2, 2, U_new4, V_new4, W_new4, col4)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i in [0, 1] and j in [0, 1, 2, 3] and k in [0, 1, 2, 3]:
                        continue
                    U_new4[i * m + j][col4] = 1
                    V_new4[j * p + k][col4] = 1
                    W_new4[k * n + i][col4] = 1
                    col4 += 1
        
        if _verify_decomp(U_new4, V_new4, W_new4, n, m, p):
            best_rank = 46
            best_U, best_V, best_W = U_new4, V_new4, W_new4
    
    # Convert to lists of integers if possible
    def _to_int_list(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                if isinstance(val, Fraction):
                    # Check if integer
                    if val.denominator == 1:
                        new_row.append(int(val))
                    else:
                        new_row.append(val)
                elif isinstance(val, (int, float)):
                    if isinstance(val, float) and val == int(val):
                        new_row.append(int(val))
                    else:
                        new_row.append(val)
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    return {"U": _to_int_list(best_U), "V": _to_int_list(best_V), "W": _to_int_list(best_W)}

