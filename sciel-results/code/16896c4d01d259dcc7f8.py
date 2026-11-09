import time

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0.0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0.0] * rank for _ in range(m * p)]  # 32 rows
    W = [[0.0] * rank for _ in range(p * n)]  # 16 rows
    
    # Strassen terms definitions
    # X coeffs: (x11, x12, x21, x22)
    # Y coeffs: (y11, y12, y21, y22)
    # Z coeffs: (z11, z12, z21, z22)
    strassen_terms = [
        # M1
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),
        # M2
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, 1)),
        # M3
        ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, -1)),
        # M4
        ((0, 0, 0, 1), (0, 0, 1, -1), (-1, 0, 1, 0)),
        # M5
        ((1, 1, 0, 0), (0, 0, 0, 1), (1, 1, 0, 0)),
        # M6
        ((0, 0, 1, -1), (1, 1, 0, 0), (0, 0, 0, -1)),
        # M7
        ((0, 1, 0, -1), (0, 0, 1, 1), (-1, 0, 0, 0)),
    ]
    
    # Map local indices to global indices
    # A indices: (i, j) -> i*4 + j
    # B indices: (j, k) -> j*8 + k
    # C indices: (i, k) -> k*2 + i (W index)
    
    col = 0
    for k in range(4):
        # k corresponds to column pair (2k, 2k+1)
        for block in range(2): # 0: L, 1: R
            # Determine A and B global indices
            # A_L: i in 0,1; j in 0,1
            # A_R: i in 0,1; j in 2,3
            # B_L: j in 0,1; k in 2k, 2k+1
            # B_R: j in 2,3; k in 2k, 2k+1
            
            a_base_j = block * 2
            b_base_j = block * 2
            
            for t in range(7):
                x_c, y_c, z_c = strassen_terms[t]
                
                # x_c corresponds to A[i, j] where (i,j) is (0,0), (0,1), (1,0), (1,1) in local block
                # Local (0,0) -> Global (0, a_base_j)
                # Local (0,1) -> Global (0, a_base_j + 1)
                # Local (1,0) -> Global (1, a_base_j)
                # Local (1,1) -> Global (1, a_base_j + 1)
                
                # y_c corresponds to B[j, k] where (j,k) is (0,0), (0,1), (1,0), (1,1) in local block
                # Local (0,0) -> Global (b_base_j, 2*k)
                # Local (0,1) -> Global (b_base_j, 2*k + 1)
                # Local (1,0) -> Global (b_base_j + 1, 2*k)
                # Local (1,1) -> Global (b_base_j + 1, 2*k + 1)
                
                # z_c corresponds to C[i, k'] where (i,k') is (0,0), (0,1), (1,0), (1,1) in local block
                # Local (0,0) -> Global (0, 2*k)
                # Local (0,1) -> Global (0, 2*k + 1)
                # Local (1,0) -> Global (1, 2*k)
                # Local (1,1) -> Global (1, 2*k + 1)
                
                # Indices in U, V, W
                # U index for A[i, j] is i*4 + j
                # V index for B[j, k] is j*8 + k
                # W index for C[i, k] is k*2 + i
                
                # Add contributions to U, V, W for this column
                # x_c[0] * A[0, a_base_j]
                if x_c[0] != 0:
                    U[0*4 + a_base_j][col] += x_c[0]
                if x_c[1] != 0:
                    U[0*4 + a_base_j + 1][col] += x_c[1]
                if x_c[2] != 0:
                    U[1*4 + a_base_j][col] += x_c[2]
                if x_c[3] != 0:
                    U[1*4 + a_base_j + 1][col] += x_c[3]
                    
                if y_c[0] != 0:
                    V[b_base_j*8 + 2*k][col] += y_c[0]
                if y_c[1] != 0:
                    V[b_base_j*8 + 2*k + 1][col] += y_c[1]
                if y_c[2] != 0:
                    V[(b_base_j+1)*8 + 2*k][col] += y_c[2]
                if y_c[3] != 0:
                    V[(b_base_j+1)*8 + 2*k + 1][col] += y_c[3]
                    
                if z_c[0] != 0:
                    W[2*k*2 + 0][col] += z_c[0]
                if z_c[1] != 0:
                    W[(2*k+1)*2 + 0][col] += z_c[1]
                if z_c[2] != 0:
                    W[2*k*2 + 1][col] += z_c[2]
                if z_c[3] != 0:
                    W[(2*k+1)*2 + 1][col] += z_c[3]
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
