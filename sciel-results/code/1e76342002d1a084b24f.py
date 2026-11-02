import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import itertools

def _strassen_2x2x2():
    """
    Returns the 7 terms of Strassen's algorithm for 2x2x2 matrix multiplication.
    Each term is (u_sparsity, v_sparsity, w_sparsity) where each sparsity is a dict {index: value}.
    
    For A (2x2), B (2x2), C (2x2):
    C[0,0] = M1 + M4 - M5 + M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M2 + M3 + M6
    
    M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    M2 = (A[1,0]+A[1,1]) * B[0,0]
    M3 = A[0,0] * (B[1,0]-B[1,1])
    M4 = A[1,1] * (B[0,1]-B[0,0])
    M5 = (A[0,0]+A[0,1]) * B[1,1]
    M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    M7 = (A[0,1]-A[1,0]) * (B[1,0]+B[1,1])
    """
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    terms = []
    
    # M1 = (A00+A11) * (B00+B11)
    u1 = {0: 1, 3: 1}
    v1 = {0: 1, 3: 1}
    w1 = {0: 1, 3: 1}  # contributes to C00 and C11
    terms.append((u1, v1, w1))
    
    # M2 = (A10+A11) * B00
    u2 = {2: 1, 3: 1}
    v2 = {0: 1}
    w2 = {2: 1}  # contributes to C10
    terms.append((u2, v2, w2))
    
    # M3 = A00 * (B10-B11)
    u3 = {0: 1}
    v3 = {2: 1, 3: -1}
    w3 = {1: 1, 3: 1}  # contributes to C01 and C11
    terms.append((u3, v3, w3))
    
    # M4 = A11 * (B01-B00)
    u4 = {3: 1}
    v4 = {1: 1, 0: -1}
    w4 = {0: 1, 2: 1}  # contributes to C00 and C10
    terms.append((u4, v4, w4))
    
    # M5 = (A00+A01) * B11
    u5 = {0: 1, 1: 1}
    v5 = {3: 1}
    w5 = {0: -1, 1: 1}  # contributes to C00 (negative) and C01
    terms.append((u5, v5, w5))
    
    # M6 = (A10-A00) * (B00+B01)
    u6 = {2: 1, 0: -1}
    v6 = {0: 1, 1: 1}
    w6 = {3: 1}  # contributes to C11
    terms.append((u6, v6, w6))
    
    # M7 = (A01-A10) * (B10+B11)
    u7 = {1: 1, 2: -1}
    v7 = {2: 1, 3: 1}
    w7 = {0: 1}  # contributes to C00
    terms.append((u7, v7, w7))
    
    return terms

def _embed_strassen(n, m, p, row_offset, col_offset_a, col_offset_b, terms):
    """
    Embed Strassen terms into the full n x m x p tensor.
    
    row_offset: starting row in A (and C)
    col_offset_a: starting column in A
    col_offset_b: starting column in B (and C)
    terms: list of (u_sparsity, v_sparsity, w_sparsity) from _strassen_2x2x2()
    
    Returns list of (u_vec, v_vec, w_vec) where each vec is a dict {index: value}
    for the full-sized vectors.
    """
    result = []
    for u_sp, v_sp, w_sp in terms:
        u_full = {}
        v_full = {}
        w_full = {}
        
        # Map local A indices to global
        for local_idx, val in u_sp.items():
            local_i = local_idx // 2
            local_j = local_idx % 2
            global_idx = (row_offset + local_i) * m + (col_offset_a + local_j)
            if global_idx in u_full:
                u_full[global_idx] += val
            else:
                u_full[global_idx] = val
        
        # Map local B indices to global
        for local_idx, val in v_sp.items():
            local_j = local_idx // 2
            local_k = local_idx % 2
            global_idx = (col_offset_a + local_j) * p + (col_offset_b + local_k)
            if global_idx in v_full:
                v_full[global_idx] += val
            else:
                v_full[global_idx] = val
        
        # Map local C indices to global
        for local_idx, val in w_sp.items():
            local_k = local_idx // 2
            local_i = local_idx % 2
            global_idx = (col_offset_b + local_k) * n + (row_offset + local_i)
            if global_idx in w_full:
                w_full[global_idx] += val
            else:
                w_full[global_idx] = val
        
        result.append((u_full, v_full, w_full))
    
    return result

def _schoolbook_terms(n, m, p):
    """Return schoolbook terms as list of (u_sparsity, v_sparsity, w_sparsity)."""
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_sp = {i * m + j: 1}
                v_sp = {j * p + k: 1}
                w_sp = {k * n + i: 1}
                terms.append((u_sp, v_sp, w_sp))
    return terms

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact.
    U: list of dicts, each representing a sparse vector of size n*m
    V: list of dicts, each representing a sparse vector of size m*p  
    W: list of dicts, each representing a sparse vector of size p*n
    
    Returns True if exact, False otherwise.
    """
    R = len(U)
    
    # For each (i,j,k), check that sum_r U[i*m+j][r] * V[j*p+k][r] == delta_{i,k}... 
    # Wait, the tensor T[i,j,k] for matrix mult is: C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # Actually, the multiplication tensor M_{n,m,p} has M[i,j,k] = 1 for all i,j,k
    # No wait. Let me reconsider.
    
    # The matrix multiplication tensor: for A (n x m), B (m x p), C (n x p)
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all valid i,j,k
    # The decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, the convention is W[k*n+i, r]
    
    # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k, i) where k is from B's columns and i is from A's rows
    # This corresponds to C[i,k]
    
    # So the tensor is: for each i in [0,n), j in [0,m), k in [0,p):
    #   T[i*m+j, j*p+k, k*n+i] = 1
    # And we need: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Check a random set of (a,b,c) to verify
    import random
    random.seed(42)
    
    # Check all entries to be safe
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                expected = 1
                actual = 0
                for r in range(R):
                    u_val = U[r].get(a, 0)
                    v_val = V[r].get(b, 0)
                    w_val = W[r].get(c, 0)
                    actual += u_val * v_val * w_val
                if abs(actual - expected) > 1e-9:
                    return False
    
    # Also check that entries that should be 0 are 0
    # T[a,b,c] should be 0 if a, b, c don't form a valid (i,j,k) triple
    # a = i*m+j, b = j'*p+k, c = k'*n+i'
    # For T[a,b,c] to be non-zero, we need j=j' and i=i' and k=k'
    # So if the implied j from a and b differ, or i from a and c differ, or k from b and c differ, T should be 0
    
    # Check some invalid entries
    for _ in range(100):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        c = random.randint(0, p*n-1)
        
        # Extract implied indices
        i_a = a // m
        j_a = a % m
        j_b = b // p
        k_b = b % p
        k_c = c // n
        i_c = c % n
        
        # If indices are consistent, this is a valid entry (should be 1)
        if j_a == j_b and i_a == i_c and k_b == k_c:
            continue  # Already checked
        
        # This should be 0
        actual = 0
        for r in range(R):
            u_val = U[r].get(a, 0)
            v_val = V[r].get(b, 0)
            w_val = W[r].get(c, 0)
            actual += u_val * v_val * w_val
        if abs(actual) > 1e-9:
            return False
    
    return True

def construct():
    """
    Returns an exact tensor decomposition for 3x4x7 matrix multiplication.
    Uses a combination of Strassen blocks and schoolbook terms to achieve lower rank.
    """
    n, m, p = DIMENSIONS  # 3, 4, 7
    start_time = time.time()
    deadline = start_time + 120
    
    # Strategy: Use Strassen for 2x2x2 blocks where possible, schoolbook for the rest
    # Partition:
    # - Rows: {0,1} and {1,2} (overlapping at row 1)
    # - A columns: {0,1} and {2,3}
    # - B columns: {0,1}, {2,3}, {4,5}, {6}
    
    # For each (row_group, a_col_group, b_col_group):
    #   If b_col_group has 2 columns, use Strassen (rank 7)
    #   If b_col_group has 1 column, use schoolbook (rank 4 for 2x2x1)
    
    all_terms = []
    
    # Define row groups
    row_groups = [(0, 1), (1, 2)]
    
    # Define A column groups
    a_col_groups = [(0, 1), (2, 3)]
    
    # Define B column groups
    b_col_groups = [(0, 1), (2, 3), (4, 5), (6, 6)]  # Last one is single column
    
    for row_offset in row_groups:
        for a_col_offset in a_col_groups:
            for b_col_offset in b_col_groups:
                if b_col_offset[0] == b_col_offset[1]:
                    # Single column in B: use schoolbook for 2x2x1
                    # 2 rows, 2 cols of A, 1 col of B
                    # Rank = 2*2*1 = 4
                    for i_local in range(2):
                        for j_local in range(2):
                            u_sp = {(row_offset[0] + i_local) * m + (a_col_offset[0] + j_local): 1}
                            v_sp = {(a_col_offset[0] + j_local) * p + b_col_offset[0]: 1}
                            w_sp = {b_col_offset[0] * n + (row_offset[0] + i_local): 1}
                            all_terms.append((u_sp, v_sp, w_sp))
                else:
                    # Two columns in B: use Strassen
                    strassen_terms = _strassen_2x2x2()
                    embedded = _embed_strassen(n, m, p, row_offset[0], a_col_offset[0], b_col_offset[0], strassen_terms)
                    all_terms.extend(embedded)
    
    # Convert to dict-based sparse representation
    U_dicts = []
    V_dicts = []
    W_dicts = []
    
    for u_sp, v_sp, w_sp in all_terms:
        U_dicts.append(dict(u_sp))
        V_dicts.append(dict(v_sp))
        W_dicts.append(dict(w_sp))
    
    R = len(U_dicts)
    
    # Verify the decomposition
    if not _verify_decomposition(U_dicts, V_dicts, W_dicts, n, m, p):
        # Fall back to schoolbook if Strassen-based decomposition is incorrect
        schoolbook = _schoolbook_terms(n, m, p)
        U_dicts = [dict(t[0]) for t in schoolbook]
        V_dicts = [dict(t[1]) for t in schoolbook]
        W_dicts = [dict(t[2]) for t in schoolbook]
        R = len(U_dicts)
    
    # Convert to dense matrix format
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    
    for r in range(R):
        for idx, val in U_dicts[r].items():
            U[idx][r] = val
        for idx, val in V_dicts[r].items():
            V[idx][r] = val
        for idx, val in W_dicts[r].items():
            W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
