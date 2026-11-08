import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    
    # Rank 36 decomposition
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded")
            
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # M1: (A00+A11)(B00+B11)
        U[u_idx(y00[0], y00[1])][col] += 1
        U[u_idx(y11[0], y11[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] += 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M2: (A10+A11)B00
        U[u_idx(y10[0], y10[1])][col] += 1
        U[u_idx(y11[0], y11[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1
        col += 1
        
        # M3: A00(B10-B11)
        U[u_idx(y00[0], y00[1])][col] += 1
        V[v_idx(z10[0], z10[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M4: A11(B00-B01)
        U[u_idx(y11[0], y11[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] += 1
        V[v_idx(z01[0], z01[1])][col] -= 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        col += 1
        
        # M5: (A00+A01)B11
        U[u_idx(y00[0], y00[1])][col] += 1
        U[u_idx(y01[0], y01[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] += 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[u_idx(y10[0], y10[1])][col] += 1
        U[u_idx(y00[0], y00[1])][col] -= 1
        V[v_idx(z00[0], z00[1])][col] += 1
        V[v_idx(z01[0], z01[1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[u_idx(y01[0], y01[1])][col] += 1
        U[u_idx(y11[0], y11[1])][col] -= 1
        V[v_idx(z10[0], z10[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] += 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        col += 1

    # Block 1: A_L (0,0:2,2) * B_11 (0,0:2,2) -> C_11 (0,0:2,2)
    y1 = [(0,0), (0,1), (1,0), (1,1)]
    z1 = [(0,0), (0,1), (1,0), (1,1)]
    w1 = [(0,0), (1,0), (0,1), (1,1)] # w_offsets: (k, i) for C[k,i]
    # C[0,0] is w(0,0), C[1,0] is w(0,1), C[0,1] is w(1,0), C[1,1] is w(1,1)
    # Wait, w_idx(k, i) = k*n + i.
    # C is 2x5. k is row (0..4), i is col (0..1).
    # Block 1 contributes to C[0:2, 0:2].
    # w_offsets should be the (k,i) pairs for the 4 output positions.
    # M1 adds to C[0,0] and C[1,1]
    # M2 adds to C[1,0] and C[1,1]
    # M3 adds to C[0,1] and C[1,1]
    # M4 adds to C[0,0] and C[1,0]
    # M5 adds to C[0,0] and C[0,1]
    # M6 adds to C[1,1]
    # M7 adds to C[0,0]
    
    # Let's map w_offsets to the 4 corners:
    # 0: top-left (0,0)
    # 1: top-right (0,1)
    # 2: bottom-left (1,0)
    # 3: bottom-right (1,1)
    
    w1 = [(0,0), (0,1), (1,0), (1,1)]
    strassen_block(y1, z1, w1)
    
    # Block 2: A_L (0,0:2,2) * B_12 (0,2:2,4) -> C_12 (0,2:2,4)
    y2 = [(0,0), (0,1), (1,0), (1,1)]
    z2 = [(0,2), (0,3), (1,2), (1,3)]
    w2 = [(0,2), (0,3), (1,2), (1,3)]
    strassen_block(y2, z2, w2)
    
    # Block 3: A_R (0,2:2,4) * B_21 (2,0:4,2) -> C_21 (0,0:2,2)
    y3 = [(0,2), (0,3), (1,2), (1,3)]
    z3 = [(2,0), (2,1), (3,0), (3,1)]
    w3 = [(0,0), (0,1), (1,0), (1,1)]
    strassen_block(y3, z3, w3)
    
    # Block 4: A_R (0,2:2,4) * B_22 (2,2:4,4) -> C_22 (0,2:2,4)
    y4 = [(0,2), (0,3), (1,2), (1,3)]
    z4 = [(2,2), (2,3), (3,2), (3,3)]
    w4 = [(0,2), (0,3), (1,2), (1,3)]
    strassen_block(y4, z4, w4)
    
    # Block 5: A_L (0,0:2,2) * B_15 (0,4:2,5) -> C_15 (0,4:2,5)
    # 2x2 * 2x1. Rank 4.
    # C[0,4] = A[0,0]B[0,4] + A[0,1]B[1,4]
    # C[1,4] = A[1,0]B[0,4] + A[1,1]B[1,4]
    
    # Term 1: A[0,0] B[0,4] -> C[0,4]
    U[u_idx(0,0)][col] += 1
    V[v_idx(0,4)][col] += 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    # Term 2: A[0,1] B[1,4] -> C[0,4]
    U[u_idx(0,1)][col] += 1
    V[v_idx(1,4)][col] += 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    # Term 3: A[1,0] B[0,4] -> C[1,4]
    U[u_idx(1,0)][col] += 1
    V[v_idx(0,4)][col] += 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    # Term 4: A[1,1] B[1,4] -> C[1,4]
    U[u_idx(1,1)][col] += 1
    V[v_idx(1,4)][col] += 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    # Block 6: A_R (0,2:2,4) * B_25 (2,4:4,5) -> C_25 (0,4:2,5)
    # C[0,4] += A[0,2]B[2,4] + A[0,3]B[3,4]
    # C[1,4] += A[1,2]B[2,4] + A[1,3]B[3,4]
    
    # Term 5: A[0,2] B[2,4] -> C[0,4]
    U[u_idx(0,2)][col] += 1
    V[v_idx(2,4)][col] += 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    # Term 6: A[0,3] B[3,4] -> C[0,4]
    U[u_idx(0,3)][col] += 1
    V[v_idx(3,4)][col] += 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    # Term 7: A[1,2] B[2,4] -> C[1,4]
    U[u_idx(1,2)][col] += 1
    V[v_idx(2,4)][col] += 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    # Term 8: A[1,3] B[3,4] -> C[1,4]
    U[u_idx(1,3)][col] += 1
    V[v_idx(3,4)][col] += 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
