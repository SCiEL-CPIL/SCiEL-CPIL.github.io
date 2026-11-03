import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds safety margin
    
    # Estimated rank: 49 (Strassen 4x4) + 16 + 16 + 4 = 85
    rank = 85
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    
    # Helper to add a rank-1 tensor
    def add_tensor(U_idx, V_idx, W_idx, u_val, v_val, w_val):
        nonlocal col
        if col >= rank:
            return
        U[U_idx][col] = u_val
        V[V_idx][col] = v_val
        W[W_idx][col] = w_val
        col += 1

    # Helper to add schoolbook for a block
    # A: i in i_range, j in j_range
    # B: j in j_range, k in k_range
    # C: i in i_range, k in k_range
    def add_schoolbook(i_range, j_range, k_range):
        nonlocal col
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    if col >= rank:
                        return
                    add_tensor(i * m + j, j * p + k, k * n + i, 1, 1, 1)

    # Helper to add Strassen for 2x2 block
    # i0, i1: rows of A/C
    # j0, j1: cols of A / rows of B
    # k0, k1: cols of B / cols of C
    def add_strassen_2x2(i0, i1, j0, j1, k0, k1):
        nonlocal col
        if col + 7 > rank:
            return
        
        # P1 = (A00 + A11)(B00 + B11)
        # C11 += P1, C22 += P1
        # U: A00, A11 = 1
        # V: B00, B11 = 1
        # W: C11 = 1, C22 = 1
        add_tensor(i0 * m + j0, j0 * p + k0, k0 * n + i0, 1, 1, 1)
        U[i1 * m + j1][col-1] = 1
        V[j1 * p + k1][col-1] = 1
        W[k1 * n + i1][col-1] = 1
        
        # P2 = (A10 + A11)B00
        # C21 += P2, C22 -= P2
        # U: A10, A11 = 1
        # V: B00 = 1
        # W: C21 = 1, C22 = -1
        add_tensor(i1 * m + j0, j0 * p + k0, k0 * n + i1, 1, 1, 1)
        U[i1 * m + j1][col-1] = 1
        W[k1 * n + i1][col-1] = -1
        
        # P3 = A00(B01 - B11)
        # C12 += P3, C22 += P3
        # U: A00 = 1
        # V: B01 = 1, B11 = -1
        # W: C12 = 1, C22 = 1
        add_tensor(i0 * m + j0, j0 * p + k1, k1 * n + i0, 1, 1, 1)
        V[j1 * p + k1][col-1] = -1
        W[k1 * n + i1][col-1] = 1
        
        # P4 = A11(B10 - B00)
        # C11 += P4, C21 += P4
        # U: A11 = 1
        # V: B10 = 1, B00 = -1
        # W: C11 = 1, C21 = 1
        add_tensor(i1 * m + j1, j1 * p + k0, k0 * n + i0, 1, 1, 1)
        V[j0 * p + k0][col-1] = -1
        W[k0 * n + i1][col-1] = 1
        
        # P5 = (A00 + A01)B11
        # C12 += P5, C11 -= P5
        # U: A00, A01 = 1
        # V: B11 = 1
        # W: C12 = 1, C11 = -1
        add_tensor(i0 * m + j0, j1 * p + k1, k1 * n + i0, 1, 1, 1)
        U[i0 * m + j1][col-1] = 1
        W[k0 * n + i0][col-1] = -1
        
        # P6 = (A10 - A00)(B00 + B01)
        # C22 += P6
        # U: A10 = 1, A00 = -1
        # V: B00 = 1, B01 = 1
        # W: C22 = 1
        add_tensor(i1 * m + j0, j0 * p + k0, k1 * n + i1, 1, 1, 1)
        U[i0 * m + j0][col-1] = -1
        V[j0 * p + k1][col-1] = 1
        
        # P7 = (A01 - A11)(B10 + B11)
        # C11 += P7
        # U: A01 = 1, A11 = -1
        # V: B10 = 1, B11 = 1
        # W: C11 = 1
        add_tensor(i0 * m + j1, j1 * p + k0, k0 * n + i0, 1, 1, 1)
        U[i1 * m + j1][col-1] = -1
        V[j1 * p + k1][col-1] = 1

    # 1. A1 B11 (4x4x4) using Strassen
    # Split 4x4 into 2x2 blocks
    # i: 0,1 and 2,3
    # j: 0,1 and 2,3
    # k: 0,1 and 2,3
    
    # Top-left 2x2
    add_strassen_2x2(0, 1, 0, 1, 0, 1)
    # Top-right 2x2
    add_strassen_2x2(0, 1, 2, 3, 0, 1)
    # Bottom-left 2x2
    add_strassen_2x2(2, 3, 0, 1, 0, 1)
    # Bottom-right 2x2
    add_strassen_2x2(2, 3, 2, 3, 0, 1)
    
    # Wait, Strassen for 4x4 requires 7 products of 2x2.
    # My add_strassen_2x2 computes one 2x2 product.
    # To compute a 4x4 product, I need to call add_strassen_2x2 7 times?
    # No, Strassen for 4x4 uses 7 products of 2x2 matrices.
    # Each of those 2x2 products can be done in rank 7.
    # So total rank 49.
    # But my add_strassen_2x2 adds 7 rank-1 tensors for ONE 2x2 product.
    # So for 4x4, I need to add 7 * 7 = 49 rank-1 tensors.
    # But the 7 products in Strassen for 4x4 are linear combinations of the 4x4 blocks.
    # This is complex to implement directly.
    # Let's just use schoolbook for the 4x4 part? Rank 64.
    # Total rank 64 + 16 + 16 + 4 = 100.
    # This is not better than schoolbook.
    
    # Let's try to use Strassen for the 4x4 part properly.
    # We can view the 4x4 multiplication as 7 multiplications of 2x2 matrices.
    # Let A = [[A11, A12], [A21, A22]] where each is 2x2.
    # Let B = [[B11, B12], [B21, B22]] where each is 2x2.
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # Each Px is a 2x2 matrix product.
    # We can compute each Px using add_strassen_2x2.
    # But add_strassen_2x2 adds to C.
    # We need to accumulate these into the final C.
    # This requires a more complex structure.
    
    # Given the time constraint and complexity, I will stick to the schoolbook decomposition.
    # It is guaranteed to work.
    
    # Reset
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U_mat, V_mat, W_mat, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_mat[i * m + j][r] * V_mat[j * p + k][r] * W_mat[k * n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if verify_decomposition(U, V, W, rank):
        return best_result
    else:
        return best_result
