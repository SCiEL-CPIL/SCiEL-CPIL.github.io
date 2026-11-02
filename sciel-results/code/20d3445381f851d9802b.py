"""Matrix multiplication 2x3x7 tensor decomposition with minimal rank."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 7
    start_time = time.time()
    deadline = start_time + 130
    
    # Strategy 1: Build hybrid Strassen/schoolbook decomposition
    decomp = _build_hybrid_decomp(n, m, p)
    if decomp is not None:
        rank = len(decomp["U"][0])
        if _verify_exact(decomp, n, m, p):
            # Try to reduce further with linear algebra
            reduced = _reduce_rank_linear(decomp, n, m, p, start_time, deadline)
            if reduced is not None and _verify_exact(reduced, n, m, p):
                return reduced
            return decomp
    
    # Strategy 2: Pure linear algebra reduction from schoolbook
    schoolbook = _schoolbook_decomp(n, m, p)
    reduced = _reduce_rank_linear(schoolbook, n, m, p, start_time, deadline)
    if reduced is not None and _verify_exact(reduced, n, m, p):
        return reduced
    
    # Fallback: schoolbook
    return schoolbook

def _schoolbook_decomp(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
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

def _verify_exact(decomp, n, m, p):
    """Exact verification using fractions."""
    U, V, W = decomp["U"], decomp["V"], decomp["W"]
    rank = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0, 1)
                for r in range(rank):
                    u_val = Fraction(str(U[i*m+j][r]))
                    v_val = Fraction(str(V[j*p+k][r]))
                    w_val = Fraction(str(W[k*n+i][r]))
                    val += u_val * v_val * w_val
                if val != 1:
                    return False
    return True

def _verify_fast(decomp, n, m, p):
    """Fast verification using numpy."""
    U = np.array(decomp["U"], dtype=np.float64)
    V = np.array(decomp["V"], dtype=np.float64)
    W = np.array(decomp["W"], dtype=np.float64)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-8:
                    return False
    return True

def _build_hybrid_decomp(n, m, p):
    """
    Build a structured decomposition using Strassen for 2x2x2 blocks
    and schoolbook for remaining terms.
    
    For 2x3x7:
    - Split j dimension (3) into j1=2, j2=1
    - Split k dimension (7) into k1=2, k2=2, k3=2, k4=1
    
    Blocks:
    - (i:2, j1:2, k1:2): Strassen, rank 7
    - (i:2, j1:2, k2:2): Strassen, rank 7
    - (i:2, j1:2, k3:2): Strassen, rank 7
    - (i:2, j1:2, k4:1): schoolbook 2x2x1, rank 4
    - (i:2, j2:1, k1:2): schoolbook 2x1x2, rank 4
    - (i:2, j2:1, k2:2): schoolbook 2x1x2, rank 4
    - (i:2, j2:1, k3:2): schoolbook 2x1x2, rank 4
    - (i:2, j2:1, k4:1): schoolbook 2x1x1, rank 2
    
    Total rank: 7+7+7+4+4+4+4+2 = 39
    """
    U_list = []  # Will collect (row, col, val) tuples
    V_list = []
    W_list = []
    col_offset = 0
    
    def add_term(u_row, v_row, w_row, u_val, v_val, w_val):
        nonlocal col_offset
        U_list.append((u_row, col_offset, u_val))
        V_list.append((v_row, col_offset, v_val))
        W_list.append((w_row, col_offset, w_val))
        col_offset += 1
    
    # Strassen block for (i in {0,1}, j in {j0,j0+1}, k in {k0,k0+1})
    def add_strassen_block(i0, j0, k0):
        nonlocal col_offset
        # A indices: A[i0,j0], A[i0,j0+1], A[i0+1,j0], A[i0+1,j0+1]
        # B indices: B[j0,k0], B[j0,k0+1], B[j0+1,k0], B[j0+1,k0+1]
        # C indices: C[i0,k0], C[i0,k0+1], C[i0+1,k0], C[i0+1,k0+1]
        
        A11 = i0 * m + j0
        A12 = i0 * m + j0 + 1
        A21 = (i0+1) * m + j0
        A22 = (i0+1) * m + j0 + 1
        
        B11 = j0 * p + k0
        B12 = j0 * p + k0 + 1
        B21 = (j0+1) * p + k0
        B22 = (j0+1) * p + k0 + 1
        
        C11 = k0 * n + i0
        C12 = (k0+1) * n + i0
        C21 = k0 * n + i0 + 1
        C22 = (k0+1) * n + i0 + 1
        
        # M1 = (A11+A22)(B11+B22)
        # Contributes to C11, C12, C21, C22
        # We need: C11 += A11*B11 + A11*B22 + A22*B11 + A22*B22
        # But Strassen uses combined terms. Let me use the standard 7-term formula.
        
        # Standard Strassen for 2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 - M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 - M3 + M6
        
        # For each M, we add a rank-1 term with appropriate U, V, W coefficients
        # M1: U has A11=1, A22=1; V has B11=1, B22=1; W has C11=1, C12=-1, C21=-1, C22=1
        # Wait, that's not right. Each M is a single rank-1 term.
        # M1 = (A11+A22)(B11+B22) means:
        #   U: A11=1, A22=1
        #   V: B11=1, B22=1
        #   W: contributes to multiple C's with different coefficients
        
        # Actually, in tensor decomposition, each rank-1 term is:
        #   U vector, V vector, W vector
        # The contribution to T[a,b,c] is U[a]*V[b]*W[c]
        
        # For M1 = (A11+A22)(B11+B22):
        #   U[A11] = 1, U[A22] = 1
        #   V[B11] = 1, V[B22] = 1
        #   W[C11] = 1, W[C12] = -1, W[C21] = -1, W[C22] = 1
        #   (because C11 = M1 - M4 - M5 + M7, etc.)
        
        # This is getting complex. Let me use a cleaner formulation.
        # I'll directly construct the 7 terms for each 2x2x2 block.
        
        terms = [
            # (u_coeffs, v_coeffs, w_coeffs)
            # M1 = (A11+A22)(B11+B22)
            ({A11: 1, A22: 1}, {B11: 1, B22: 1}, {C11: 1, C12: -1, C21: -1, C22: 1}),
            # M2 = (A21+A22)B11
            ({A21: 1, A22: 1}, {B11: 1}, {C21: 1, C22: -1}),
            # M3 = A11(B21-B22)
            ({A11: 1}, {B21: 1, B22: -1}, {C12: 1, C22: -1}),
            # M4 = A22(B12-B11)
            ({A22: 1}, {B12: 1, B11: -1}, {C11: -1, C21: 1}),
            # M5 = (A11+A12)B22
            ({A11: 1, A12: 1}, {B22: 1}, {C11: -1, C12: 1}),
            # M6 = (A21-A11)(B11+B12)
            ({A21: 1, A11: -1}, {B11: 1, B12: 1}, {C22: 1}),
            # M7 = (A12-A22)(B21+B22)
            ({A12: 1, A22: -1}, {B21: 1, B22: 1}, {C11: 1, C22: 1}),
        ]
        
        for u_coeffs, v_coeffs, w_coeffs in terms:
            for u_row, u_val in u_coeffs.items():
                pass  # We need to add a single rank-1 term
            
            # Add as single rank-1 term
            col = col_offset
            for u_row, u_val in u_coeffs.items():
                U_list.append((u_row, col, u_val))
            for v_row, v_val in v_coeffs.items():
                V_list.append((v_row, col, v_val))
            for w_row, w_val in w_coeffs.items():
                W_list.append((w_row, col, w_val))
            col_offset += 1
    
    # Schoolbook for small blocks
    def add_schoolbook_block(i0, j0, k0, ni, nj, nk):
        nonlocal col_offset
        for ii in range(ni):
            for jj in range(nj):
                for kk in range(nk):
                    i = i0 + ii
                    j = j0 + jj
                    k = k0 + kk
                    col = col_offset
                    U_list.append((i * m + j, col, 1))
                    V_list.append((j * p + k, col, 1))
                    W_list.append((k * n + i, col, 1))
                    col_offset += 1
    
    # Apply Strassen to 2x2x2 blocks where both j and k have at least 2
    # j=0,1 with k=0,1
    add_strassen_block(0, 0, 0)
    # j=0,1 with k=2,3
    add_strassen_block(0, 0, 2)
    # j=0,1 with k=4,5
    add_strassen_block(0, 0, 4)
    
    # Remaining: j=0,1 with k=6 (2x2x1)
    add_schoolbook_block(0, 0, 6, 2, 2, 1)
    # j=2 with k=0,1 (2x1x2)
    add_schoolbook_block(0, 2, 0, 2, 1, 2)
    # j=2 with k=2,3 (2x1x2)
    add_schoolbook_block(0, 2, 2, 2, 1, 2)
    # j=2 with k=4,5 (2x1x2)
    add_schoolbook_block(0, 2, 4, 2, 1, 2)
    # j=2 with k=6 (2x1x1)
    add_schoolbook_block(0, 2, 6, 2, 1, 1)
    
    # Convert to matrices
    total_rank = col_offset
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for row, col, val in U_list:
        U[row][col] = val
    for row, col, val in V_list:
        V[row][col] = val
    for row, col, val in W_list:
        W[row][col] = val
    
    return {"U": U, "V": V, "W": W}

def _reduce_rank_linear(decomp, n, m, p, start_time, deadline):
    """
    Try to reduce the rank by finding linear dependencies among the rank-1 terms.
    Uses linear algebra to identify and eliminate dependent terms.
    """
    U = np.array(decomp["U"], dtype=np.float64)
    V = np.array(decomp["V"], dtype=np.float64)
    W = np.array(decomp["W"], dtype=np.float64)
    rank = U.shape[1]
    
    # The tensor T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
    # We want to find if some columns are linearly dependent in a way
    # that allows us to remove them while preserving T.
    
    # Approach: For each column t, check if the rank-1 tensor
    # U[:,t] ⊗ V[:,t] ⊗ W[:,t] can be expressed as a linear combination
    # of the others. This is hard in general.
    
    # Simpler approach: Try removing one column at a time and see if we can
    # adjust the remaining columns to compensate.
    
    best_decomp = decomp
    best_rank = rank
    
    # Try removing columns one by one
    for t_remove in range(rank):
        if time.time() > deadline - 10:
            break
        
        # Remove column t_remove
        mask = [i for i in range(rank) if i != t_remove]
        U_sub = U[:, mask]
        V_sub = V[:, mask]
        W_sub = W[:, mask]
        
        # Check if the tensor is still correct
        T_computed = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for t in range(len(mask)):
            T_computed += np.einsum('a,b,c->abc', U_sub[:, t], V_sub[:, t], W_sub[:, t])
        
        T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
        
        if np.max(np.abs(T_computed - T_target)) < 1e-8:
            # Column t_remove was redundant!
            new_rank = len(mask)
            if new_rank < best_rank:
                best_rank = new_rank
                U_new = U_sub.tolist()
                V_new = V_sub.tolist()
                W_new = W_sub.tolist()
                best_decomp = {"U": U_new, "V": V_new, "W": W_new}
    
    if best_rank < rank:
        return best_decomp
    return None
# EVOLVE-BLOCK-END
