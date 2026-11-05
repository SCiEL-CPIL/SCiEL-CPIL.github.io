import time
import sys
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 5x5 matrix by a 5x5 matrix.
    
    Strategy:
    1. Start with the trivial schoolbook decomposition (rank 125).
    2. Try to reduce rank by finding linear dependencies in the tensor.
    3. Use Strassen-like recursive blocks where beneficial.
    
    The output follows the trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS
    # n=5, m=5, p=5
    
    # The matrix multiplication tensor T[i,j,k] = 1 if i==k and j==j (always true for the structure)
    # Actually, T[i,j,k] represents the coefficient of A[i,j] * B[j,k] in C[i,k]
    # So T[i,j,k] = 1 for all valid i,j,k where the product contributes to C[i,k]
    # Wait, the tensor for matrix multiplication is T_{i,j,k} = A_{i,j} B_{j,k} C_{i,k}
    # In the standard formulation, the tensor has components T[i,j,k] = 1 for all i,j,k
    # because each term A[i,j]*B[j,k] contributes to C[i,k].
    
    # Let me reconsider. The tensor M_{i,j,k} = A_{i,j} B_{j,k} C_{i,k}
    # For matrix multiplication, M[i,j,k] = 1 for all i,j,k (each triple contributes 1)
    # No wait, that's not right either.
    
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 for all i,j,k
    # where i in [0,n), j in [0,m), k in [0,p), and the product A[i,j]*B[j,k] contributes to C[i,k].
    # This is always the case, so T[i,j,k] = 1 for all i,j,k.
    
    # Wait, I need to be more careful. The tensor for n x m by m x p matrix multiplication
    # has dimensions n*m*p. The entry T[i,j,k] corresponds to the term A[i,j] * B[j,k] * C[i,k].
    # For the multiplication to be correct, we need:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor decomposes as: T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r is a vector in R^{n*m}, V_r in R^{m*p}, W_r in R^{p*n}
    # and the condition is: sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = delta_{i,k} * (something)
    
    # Actually, the standard formulation is:
    # The multiplication tensor T ∈ R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}
    # T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # where e_{i,j} is the basis vector for position (i,j) in the n*m matrix,
    # e_{j,k} is the basis vector for position (j,k) in the m*p matrix,
    # e_{k,i} is the basis vector for position (k,i) in the p*n matrix.
    
    # So the trivial decomposition has rank n*m*p = 125, with:
    # For each (i,j,k): U_r[i,j] = 1, V_r[j,k] = 1, W_r[k,i] = 1
    
    # To reduce rank, we can use linear combinations.
    
    # Let's use the Strassen-based approach for 5x5x5.
    # Partition 5 = 2 + 2 + 1
    
    # For 2x2x2 blocks, use Strassen (rank 7 instead of 8)
    # For other blocks, use schoolbook (rank = product of sizes)
    
    parts = [2, 2, 1]
    offsets = [0, 2, 4]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_col(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Iterate over all block combinations
    for a in range(3):
        for b in range(3):
            for c in range(3):
                size_a = parts[a]
                size_b = parts[b]
                size_c = parts[c]
                off_a = offsets[a]
                off_b = offsets[b]
                off_c = offsets[c]
                
                if size_a == 2 and size_b == 2 and size_c == 2:
                    # Strassen block
                    # 7 factors for 2x2x2
                    
                    # M1 = A00 * B00 -> C00
                    u = [0]*25; v = [0]*25; w = [0]*25
                    u[(off_a+0)*5 + (off_b+0)] = 1
                    v[(off_b+0)*5 + (off_c+0)] = 1
                    w[(off_c+0)*5 + (off_a+0)] = 1
                    add_col(u, v, w)
                    
                    # M2 = (A10-A00) * B00 -> C10
                    u = [0]*25; v = [0]*25; w = [0]*25
                    u[(off_a+1)*5 + (off_b+0)] = 1
                    u[(off_a+0)*5 + (off_b+0)] = -1
                    v[(off_b+0)*5 + (off_c+0)] = 1
                    w[(off_c+0)*5 + (off_a+1)] = 1
                    add_col(u, v, w)
                    
                    # M3 = (A01-A11) * B10 -> C00
                    u = [0]*25; v = [0]*25; w = [0]*25
                    u[(off_a+0)*5 + (off_b+1)] = 1
                    u[(off_a+1)*5 + (off_b+1)] = -1
                    v[(off_b+1)*5 + (off_c+0)] = 1
                    w[(off_c+0)*5 + (off_a+0)] = 1
                    add_col(u, v, w)
                    
                    # M4 = A11 * B11 -> C11
                    u = [0]*25; v = [0]*25; w = [0]*25
                    u[(off_a+1)*5 + (off_b+1)] = 1
                    v[(off_b+1)*5 + (off_c+1)] = 1
                    w[(off_c+1)*5 + (off_a+1)] = 1
                    add_col(u, v, w)
                    
                    # M5 = (A00+A11) * (B00+B11) -> C00, C11
                    u = [0]*25; v = [0]*25; w = [0]*25
                    u[(off_a+0)*5 + (off_b+0)] = 1
                    u[(off_a+1)*5 + (off_b+1)] = 1
                    v[(off_b+0)*5 + (off_c+0)] = 1
                    v[(off_b+1)*5 + (off_c+1)] = 1
                    w[(off_c+0)*5 + (off_a+0)] = 1
                    w[(off_c+1)*5 + (off_a+1)] = 1
                    add_col(u, v, w)
                    
                    # M6 = (A10+A11) * (B01-B00) -> C10, C11
                    u = [0]*25; v = [0]*25; w = [0]*25
                    u[(off_a+1)*5 + (off_b+0)] = 1
                    u[(off_a+1)*5 + (off_b+1)] = 1
                    v[(off_b+0)*5 + (off_c+1)] = 1
                    v[(off_b+0)*5 + (off_c+0)] = -1
                    w[(off_c+1)*5 + (off_a+1)] = 1
                    w[(off_c+0)*5 + (off_a+1)] = -1
                    add_col(u, v, w)
                    
                    # M7 = (A00-A10) * (B10+B11) -> C00, C01
                    u = [0]*25; v = [0]*25; w = [0]*25
                    u[(off_a+0)*5 + (off_b+0)] = 1
                    u[(off_a+1)*5 + (off_b+0)] = -1
                    v[(off_b+1)*5 + (off_c+0)] = 1
                    v[(off_b+1)*5 + (off_c+1)] = 1
                    w[(off_c+0)*5 + (off_a+0)] = 1
                    w[(off_c+1)*5 + (off_a+0)] = 1
                    add_col(u, v, w)
                    
                else:
                    # Schoolbook block
                    for u_idx in range(size_a):
                        for v_idx in range(size_b):
                            for w_idx in range(size_c):
                                u = [0]*25; v = [0]*25; w = [0]*25
                                gi = off_a + u_idx
                                gj = off_b + v_idx
                                gk = off_c + w_idx
                                u[gi*5 + gj] = 1
                                v[gj*5 + gk] = 1
                                w[gk*5 + gi] = 1
                                add_col(u, v, w)
    
    rank = len(U_cols)
    
    # Convert to row-major format
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    for col_idx in range(rank):
        u_col = U_cols[col_idx]
        v_col = V_cols[col_idx]
        w_col = W_cols[col_idx]
        for row_idx in range(25):
            U[row_idx][col_idx] = u_col[row_idx]
            V[row_idx][col_idx] = v_col[row_idx]
            W[row_idx][col_idx] = w_col[row_idx]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
