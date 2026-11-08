"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import numpy as np

DIMENSIONS = [3, 3, 3]
START_TIME = time.time()
TIME_LIMIT = 120.0

def _verify_decomposition(U, V, W, n=3, m=3, p=3):
    """Verify that the tensor decomposition is correct."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the multiplication tensor
    # T[i,j,k,l] = 1 if the product C[i,k] gets contribution from A[i,j]*B[j,k]
    # Our convention: U has shape (n*m, rank), indexed by i*m+j
    # V has shape (m*p, rank), indexed by j*p+k
    # W has shape (p*n, rank), indexed by k*n+i
    
    # The decomposition says: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{product}
    # For matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] for the multiplication is: T[i,j,k] = 1 (it's a 3-way tensor)
    # Actually, the multiplication tensor has entries T_{i,j,k} where we contract over j.
    # The rank-r decomposition: T_{i,j,k} = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)} is not quite right.
    
    # Let me re-derive. The tensor for matrix multiplication M_{n,m,p} has entries:
    # T[a,b,c] = 1 if a is row of A, b is col of A / row of B, c is col of B
    # The product is: C[i,c] = sum_j A[i,j] B[j,c]
    # So T[i,j,c] = 1 for all valid i,j,c (it's the identity in the middle index)
    
    # The rank-r decomposition: T[i,j,c] = sum_r U[i*m+j, r] * V[j*p+c, r] * W[c*n+i, r]
    # Wait, that's not right either. Let me think again.
    
    # Standard form: The matrix multiplication tensor is a 3-tensor T with T[i,j,k] = 1
    # if we're computing C[i,k] from A[i,j]*B[j,k].
    # A rank-r decomposition: T[i,j,k] = sum_{r=1}^r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}
    # No, that's also wrong.
    
    # Correct form: The multiplication tensor T has indices (i, j, k) where i indexes rows of A,
    # j indexes cols of A (= rows of B), k indexes cols of B.
    # T[i,j,k] = 1 (for all i,j,k in range)
    # A rank-r decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # No! The W factor should have indices matching the output.
    
    # Let me use the convention from the problem:
    # U: shape (n*m, rank), U[i*m + j, r]
    # V: shape (m*p, rank), V[j*p + k, r]  
    # W: shape (p*n, rank), W[k*n + i, r]
    
    # The decomposition asserts: for all i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is because T[i,j,k] = 1 for the multiplication tensor.
    
    # Let's verify this.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _construct_rank26():
    """Construct the known rank-26 decomposition."""
    n, m, p = 3, 3, 3
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1
    
    # 1. A2 B2 (Top-left 2x2 * 2x2) -> Rank 7 (Strassen)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # 2. Cross terms
    add_term(2, 6, 0, 1)
    add_term(2, 7, 3, 1)
    add_term(5, 6, 1, 1)
    add_term(5, 7, 4, 1)
    
    add_term(0, 2, 6, 1)
    add_term(1, 5, 6, 1)
    add_term(3, 2, 7, 1)
    add_term(4, 5, 7, 1)
    
    add_term(2, 8, 6, 1)
    add_term(5, 8, 7, 1)
    
    add_term(6, 0, 2, 1)
    add_term(7, 3, 2, 1)
    add_term(6, 1, 5, 1)
    add_term(7, 4, 5, 1)
    
    add_term(8, 6, 2, 1)
    add_term(8, 7, 5, 1)
    
    add_term(6, 2, 8, 1)
    add_term(7, 5, 8, 1)
    
    add_term(8, 8, 8, 1)
    
    return {"U": U, "V": V, "W": W}

def _construct_rank23():
    """
    Attempt to construct a rank-23 decomposition.
    This uses a more efficient block decomposition.
    """
    n, m, p = 3, 3, 3
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += coeff
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1
    
    # Use a different strategy: exploit symmetry and combine terms
    # The key idea: some cross terms can be combined with the Strassen terms
    
    # Start with the 2x2 Strassen part (rank 7)
    # M1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # M2 = (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    # M3 = A00(B01-B11)
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # M4 = A11(B10-B00)
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # Now handle cross terms more efficiently
    # Instead of 19 separate terms, try to combine
    
    # Group 1: Terms involving A[0:2,2] and B[2,0:2] (4 terms)
    # C00 += A02*B20, C01 += A02*B21, C10 += A12*B20, C11 += A12*B21
    # Can we combine with existing terms? Not directly.
    add_term(2, 6, 0, 1)
    add_term(2, 7, 3, 1)
    add_term(5, 6, 1, 1)
    add_term(5, 7, 4, 1)
    
    # Group 2: A[0:2,0:2] * B[0:2,2] (4 terms)
    add_term(0, 2, 6, 1)
    add_term(1, 5, 6, 1)
    add_term(3, 2, 7, 1)
    add_term(4, 5, 7, 1)
    
    # Group 3: A[0:2,2] * B[2,2] (2 terms)
    add_term(2, 8, 6, 1)
    add_term(5, 8, 7, 1)
    
    # Group 4: A[2,0:2] * B[0:2,0:2] (4 terms)
    add_term(6, 0, 2, 1)
    add_term(7, 3, 2, 1)
    add_term(6, 1, 5, 1)
    add_term(7, 4, 5, 1)
    
    # Group 5: A[2,2] * B[2,0:2] (2 terms)
    add_term(8, 6, 2, 1)
    add_term(8, 7, 5, 1)
    
    # Group 6: A[2,0:2] * B[0:2,2] (2 terms)
    add_term(6, 2, 8, 1)
    add_term(7, 5, 8, 1)
    
    # Group 7: A[2,2] * B[2,2] (1 term)
    add_term(8, 8, 8, 1)
    
    # We've used 7 + 4 + 4 + 2 + 4 + 2 + 2 + 1 = 26 terms, but rank is 23
    # This won't work as-is. The rank-23 construction requires a different approach.
    
    # For now, return the rank-26 result if we can't fit in 23
    if col > rank:
        return None
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # First, try to construct a rank-23 decomposition
    result_23 = _construct_rank23()
    if result_23 is not None:
        if _verify_decomposition(result_23["U"], result_23["V"], result_23["W"], n, m, p):
            return result_23
    
    # Fall back to rank-26
    result_26 = _construct_rank26()
    
    # Verify the rank-26 decomposition
    if not _verify_decomposition(result_26["U"], result_26["V"], result_26["W"], n, m, p):
        # If verification fails, return schoolbook rank-27
        rank = 27
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Deadline guard
    if time.time() - START_TIME > TIME_LIMIT - 5:
        pass
        
    return result_26
# EVOLVE-BLOCK-END
