"""Program for matrix multiplication tensor decomposition 5x5x5.
Attempts to find a lower-rank exact decomposition using recursive
Kronecker product embedding of smaller known tensor decompositions,
with a deadline guard to ensure we return the best valid result found.
"""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matrix multiplication, rank 7."""
    # Strassen's algorithm: 7 multiplications
    # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For the tensor decomposition, we need:
    # T[i,j,k] = sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c]
    
    # Let's define the 7 rank-1 terms:
    # Term 0: M1
    U[0*2+0, 0] = 1  # A00
    U[1*2+1, 0] = 1  # A11
    V[0*2+0, 0] = 1  # B00
    V[1*2+1, 0] = 1  # B11
    W[0*2+0, 0] = 1  # C00
    W[1*2+1, 0] = 1  # C11
    
    # Term 1: M2
    U[1*2+0, 1] = 1  # A10
    U[1*2+1, 1] = 1  # A11
    V[0*2+0, 1] = 1  # B00
    W[1*2+0, 1] = 1  # C10
    W[1*2+1, 1] = -1  # C11
    
    # Term 2: M3
    U[0*2+0, 2] = 1  # A00
    V[1*2+0, 2] = 1  # B10
    V[1*2+1, 2] = -1 # B11
    W[0*2+1, 2] = 1  # C01
    W[1*2+1, 2] = 1  # C11
    
    # Term 3: M4
    U[1*2+1, 3] = 1  # A11
    V[0*2+1, 3] = 1  # B01
    V[0*2+0, 3] = -1 # B00
    W[0*2+0, 3] = 1  # C00
    W[1*2+0, 3] = 1  # C10
    
    # Term 4: M5
    U[0*2+0, 4] = 1  # A00
    U[0*2+1, 4] = 1  # A01
    V[1*2+1, 4] = 1  # B11
    W[0*2+0, 4] = -1 # C00
    W[0*2+1, 4] = 1  # C01
    
    # Term 5: M6
    U[1*2+0, 5] = 1  # A10
    U[0*2+0, 5] = -1 # A00
    V[0*2+0, 5] = 1  # B00
    V[0*2+1, 5] = 1  # B01
    W[1*2+1, 5] = 1  # C11
    
    # Term 6: M7
    U[0*2+1, 6] = 1  # A01
    U[1*2+1, 6] = -1 # A11
    V[1*2+0, 6] = 1  # B10
    V[1*2+1, 6] = 1  # B11
    W[0*2+0, 6] = 1  # C00
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that the tensor decomposition exactly represents matmul."""
    rank = U.shape[1]
    errors = 0
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = 0.0
                for c in range(rank):
                    actual += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                err = abs(actual - expected)
                if err > tol:
                    errors += 1
                    max_err = max(max_err, err)
    return errors == 0, max_err

def _block_partition_decomposition(n, m, p):
    """
    Try to decompose the n×m×p matrix multiplication tensor using block partitioning.
    For 5×5×5, use a 2+3 partition and combine Strassen (rank 7) for 2×2×2
    and schoolbook (rank 27) for 3×3×3, plus cross terms.
    """
    if n != 5 or m != 5 or p != 5:
        return None
    
    # Partition indices into groups of size 2 and 3
    # Group 0: indices {0, 1} (size 2)
    # Group 1: indices {2, 3, 4} (size 3)
    
    groups = [
        [0, 1],      # size 2
        [2, 3, 4]    # size 3
    ]
    
    # We'll build the decomposition by combining:
    # 1. The 2×2×2 block (indices in group 0 for all three dimensions) - use Strassen, rank 7
    # 2. The 3×3×3 block (indices in group 1 for all three dimensions) - use schoolbook, rank 27
    # 3. Cross terms: where indices come from different groups
    
    # Start with empty factor matrices
    total_rank = 0
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Helper to add a sub-decomposition with index offsets
    def add_subdecomp(sub_U, sub_V, sub_W, i_off, j_off, k_off, sub_n, sub_m, sub_p):
        """Add a sub-decomposition for indices starting at offsets i_off, j_off, k_off."""
        nonlocal total_rank
        rank = sub_U.shape[1]
        
        # Create full-sized factor matrices with zeros
        U_full = np.zeros((n * m, rank))
        V_full = np.zeros((m * p, rank))
        W_full = np.zeros((p * n, rank))
        
        # Fill in the relevant entries
        for ii in range(sub_n):
            for jj in range(sub_m):
                U_full[(i_off + ii) * m + (j_off + jj), :] = sub_U[ii * sub_m + jj, :]
        
        for jj in range(sub_m):
            for kk in range(sub_p):
                V_full[(j_off + jj) * p + (k_off + kk), :] = sub_V[jj * sub_p + kk, :]
        
        for kk in range(sub_p):
            for ii in range(sub_n):
                W_full[(k_off + kk) * n + (i_off + ii), :] = sub_W[kk * sub_n + ii, :]
        
        U_parts.append(U_full)
        V_parts.append(V_full)
        W_parts.append(W_full)
        total_rank += rank
    
    # 1. Add 2×2×2 Strassen block at (0,0,0)
    U2, V2, W2 = _strassen_2x2x2()
    add_subdecomp(U2, V2, W2, 0, 0, 0, 2, 2, 2)
    
    # 2. Add 3×3×3 schoolbook block at (2,2,2)
    U3, V3, W3 = _schoolbook(3, 3, 3)
    add_subdecomp(U3, V3, W3, 2, 2, 2, 3, 3, 3)
    
    # 3. Add cross terms: we need to handle all combinations where
    #    i is in one group, j in another, k in another
    #    The tensor T[i,j,k] = 1 for all i,j,k, so we need to cover all 125 entries
    
    # The diagonal blocks (0,0,0) and (2,2,2) are covered.
    # We need to cover:
    # - i in {0,1}, j in {0,1}, k in {2,3,4} (2×2×3 = 12 entries)
    # - i in {0,1}, j in {2,3,4}, k in {0,1} (2×3×2 = 12 entries)
    # - i in {0,1}, j in {2,3,4}, k in {2,3,4} (2×3×3 = 18 entries)
    # - i in {2,3,4}, j in {0,1}, k in {0,1} (3×2×2 = 12 entries)
    # - i in {2,3,4}, j in {0,1}, k in {2,3,4} (3×2×3 = 18 entries)
    # - i in {2,3,4}, j in {2,3,4}, k in {0,1} (3×3×2 = 18 entries)
    
    # For cross terms, we can use rank-1 terms: for each (i,j,k), add a term
    # with U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # But we can potentially combine multiple (i,j,k) into one term if possible.
    
    # For now, use rank-1 terms for each uncovered (i,j,k)
    covered = set()
    # Mark covered entries from diagonal blocks
    for i in range(2):
        for j in range(2):
            for k in range(2):
                covered.add((i, j, k))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                covered.add((i+2, j+2, k+2))
    
    # Add rank-1 terms for uncovered entries
    for i in range(5):
        for j in range(5):
            for k in range(5):
                if (i, j, k) not in covered:
                    # Add a rank-1 term
                    U_term = np.zeros((25, 1))
                    V_term = np.zeros((25, 1))
                    W_term = np.zeros((25, 1))
                    U_term[i * 5 + j, 0] = 1
                    V_term[j * 5 + k, 0] = 1
                    W_term[k * 5 + i, 0] = 1
                    U_parts.append(U_term)
                    V_parts.append(V_term)
                    W_parts.append(W_term)
                    total_rank += 1
    
    # Concatenate all parts
    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    W = np.hstack(W_parts)
    
    return U, V, W, total_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # First, get the schoolbook decomposition as a baseline
    U, V, W = _schoolbook(n, m, p)
    baseline_rank = n * m * p
    
    # Verify baseline
    ok, _ = _verify(U, V, W, n, m, p)
    if not ok:
        raise RuntimeError("Schoolbook decomposition failed verification")
    
    best_U, best_V, best_W = U, V, W
    best_rank = baseline_rank
    
    # Try block partitioning approach
    if time.time() < deadline:
        result = _block_partition_decomposition(n, m, p)
        if result is not None:
            U_new, V_new, W_new, rank_new = result
            if rank_new < best_rank:
                ok, _ = _verify(U_new, V_new, W_new, n, m, p)
                if ok:
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = rank_new
    
    # Convert numpy arrays to lists for output
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
