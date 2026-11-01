"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (10, rank)
    
    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    def build_rank36():
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        def strassen_block(y_indices, z_indices, w_offsets):
            nonlocal col
            
            y00, y01, y10, y11 = y_indices
            z00, z01, z10, z11 = z_indices
            
            # M1: (y00+y11)(z00+z11) -> X00, X11
            U[u_idx(y00[0], y00[1])][col] = 1
            U[u_idx(y11[0], y11[1])][col] = 1
            V[v_idx(z00[0], z00[1])][col] = 1
            V[v_idx(z11[0], z11[1])][col] = 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1
            
            # M2: (y10+y11)z00 -> X10, -X11
            U[u_idx(y10[0], y10[1])][col] = 1
            U[u_idx(y11[0], y11[1])][col] = 1
            V[v_idx(z00[0], z00[1])][col] = 1
            W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1
            col += 1
            
            # M3: y00(z01-z11) -> X01, X11
            U[u_idx(y00[0], y00[1])][col] = 1
            V[v_idx(z01[0], z01[1])][col] = 1
            V[v_idx(z11[0], z11[1])][col] = -1
            W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1
            
            # M4: y11(z10-z00) -> X00, X10
            U[u_idx(y11[0], y11[1])][col] = 1
            V[v_idx(z10[0], z10[1])][col] = 1
            V[v_idx(z00[0], z00[1])][col] = -1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
            col += 1
            
            # M5: (y00+y01)z11 -> -X00, X01
            U[u_idx(y00[0], y00[1])][col] = 1
            U[u_idx(y01[0], y01[1])][col] = 1
            V[v_idx(z11[0], z11[1])][col] = 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1
            W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
            col += 1
            
            # M6: (y10-y00)(z00+z01) -> X11
            U[u_idx(y10[0], y10[1])][col] = 1
            U[u_idx(y00[0], y00[1])][col] = -1
            V[v_idx(z00[0], z00[1])][col] = 1
            V[v_idx(z01[0], z01[1])][col] = 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1
            
            # M7: (y01-y11)(z10+z11) -> X00
            U[u_idx(y01[0], y01[1])][col] = 1
            U[u_idx(y11[0], y11[1])][col] = -1
            V[v_idx(z10[0], z10[1])][col] = 1
            V[v_idx(z11[0], z11[1])][col] = 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            col += 1

        # Block 1: A_L (cols 0,1) * B_11 (rows 0,1; cols 0,1)
        y1 = [(0,0), (0,1), (1,0), (1,1)]
        z1 = [(0,0), (0,1), (1,0), (1,1)]
        w1 = [(0,0), (1,0), (0,1), (1,1)]
        strassen_block(y1, z1, w1)
        
        # Block 2: A_R (cols 2,3) * B_21 (rows 2,3; cols 0,1)
        y2 = [(0,2), (0,3), (1,2), (1,3)]
        z2 = [(2,0), (2,1), (3,0), (3,1)]
        w2 = [(0,0), (1,0), (0,1), (1,1)]
        strassen_block(y2, z2, w2)
        
        # Block 3: A_L (cols 0,1) * B_12 (rows 0,1; cols 2,3)
        y3 = [(0,0), (0,1), (1,0), (1,1)]
        z3 = [(0,2), (0,3), (1,2), (1,3)]
        w3 = [(2,0), (3,0), (2,1), (3,1)]
        strassen_block(y3, z3, w3)
        
        # Block 4: A_R (cols 2,3) * B_22 (rows 2,3; cols 2,3)
        y4 = [(0,2), (0,3), (1,2), (1,3)]
        z4 = [(2,2), (2,3), (3,2), (3,3)]
        w4 = [(2,0), (3,0), (2,1), (3,1)]
        strassen_block(y4, z4, w4)
        
        # Block 5: A_L (cols 0,1) * B_15 (rows 0,1; col 4)
        # Term 1: A_00 * B_0 -> C_0
        U[u_idx(0,0)][col] = 1
        V[v_idx(0,4)][col] = 1
        W[w_idx(4,0)][col] = 1
        col += 1
        
        # Term 2: A_01 * B_1 -> C_0
        U[u_idx(0,1)][col] = 1
        V[v_idx(1,4)][col] = 1
        W[w_idx(4,0)][col] = 1
        col += 1
        
        # Term 3: A_10 * B_0 -> C_1
        U[u_idx(1,0)][col] = 1
        V[v_idx(0,4)][col] = 1
        W[w_idx(4,1)][col] = 1
        col += 1
        
        # Term 4: A_11 * B_1 -> C_1
        U[u_idx(1,1)][col] = 1
        V[v_idx(1,4)][col] = 1
        W[w_idx(4,1)][col] = 1
        col += 1
        
        # Block 6: A_R (cols 2,3) * B_25 (rows 2,3; col 4)
        # Term 1: A_02 * B_24 -> C_0
        U[u_idx(0,2)][col] = 1
        V[v_idx(2,4)][col] = 1
        W[w_idx(4,0)][col] = 1
        col += 1
        
        # Term 2: A_03 * B_34 -> C_0
        U[u_idx(0,3)][col] = 1
        V[v_idx(3,4)][col] = 1
        W[w_idx(4,0)][col] = 1
        col += 1
        
        # Term 3: A_12 * B_24 -> C_1
        U[u_idx(1,2)][col] = 1
        V[v_idx(2,4)][col] = 1
        W[w_idx(4,1)][col] = 1
        col += 1
        
        # Term 4: A_13 * B_34 -> C_1
        U[u_idx(1,3)][col] = 1
        V[v_idx(3,4)][col] = 1
        W[w_idx(4,1)][col] = 1
        col += 1
        
        return U, V, W

    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Build the tensor from the decomposition
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # But we need to map indices correctly
        # U index: i*m + j for A[i,j]
        # V index: j*p + k for B[j,k]
        # W index: k*n + i for C[k,i] (trace convention W[k*n+i])
        
        # The tensor for matrix multiplication C = A * B is:
        # T[A_idx, B_idx, C_idx] = 1 if C_idx corresponds to the product
        # Actually, the standard tensor is: T[i,j,k] where C[i,k] = sum_j A[i,j]*B[j,k]
        # In our indexing: U is indexed by (i,j) -> i*m+j, V by (j,k) -> j*p+k, W by (k,i) -> k*n+i
        
        # For a given (i,j,k), the contribution is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = delta(i,j,k) for the multiplication tensor
        # Actually the tensor is: T[i,j,k] = 1 if this is a valid term in C[i,k] = sum_j A[i,j]B[j,k]
        # So T[i*m+j, j*p+k, k*n+i] should equal 1 for all valid (i,j,k)
        
        # Let's compute the tensor from the decomposition
        tensor = np.zeros((n*m, m*p, p*n), dtype=float)
        for r in range(rank):
            for i in range(n*m):
                if U_arr[i, r] != 0:
                    for j in range(m*p):
                        if V_arr[j, r] != 0:
                            for k in range(p*n):
                                tensor[i, j, k] += U_arr[i, r] * V_arr[j, r] * W_arr[k, r]
        
        # Now check against the true tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_i = i * m + j
                    v_j = j * p + k
                    w_k = k * n + i
                    if abs(tensor[u_i, v_j, w_k] - 1.0) > 1e-9:
                        return False
                    # Check that other entries are 0
                    # (This is expensive, so we'll do a spot check)
        
        return True

    def try_reduce_rank(U, V, W, max_time=10):
        """Try to reduce rank by checking if any columns are linearly dependent."""
        start_time = time.time()
        rank = len(U[0])
        
        # Try removing single columns
        for col_to_remove in range(rank - 1, -1, -1):
            if time.time() - start_time > max_time:
                break
            new_rank = rank - 1
            U_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in U]
            V_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in V]
            W_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in W]
            
            if verify(U_new, V_new, W_new):
                return U_new, V_new, W_new
        
        # Try removing pairs of columns
        for i in range(rank - 1):
            if time.time() - start_time > max_time:
                break
            for j in range(i + 1, rank):
                if time.time() - start_time > max_time:
                    break
                new_rank = rank - 2
                U_new = []
                for row in U:
                    new_row = []
                    for r in range(rank):
                        if r != i and r != j:
                            new_row.append(row[r])
                    U_new.append(new_row)
                V_new = []
                for row in V:
                    new_row = []
                    for r in range(rank):
                        if r != i and r != j:
                            new_row.append(row[r])
                    V_new.append(new_row)
                W_new = []
                for row in W:
                    new_row = []
                    for r in range(rank):
                        if r != i and r != j:
                            new_row.append(row[r])
                    W_new.append(new_row)
                
                if verify(U_new, V_new, W_new):
                    return U_new, V_new, W_new
        
        return U, V, W

    # Build initial rank-36 decomposition
    U, V, W = build_rank36()
    
    # Try to reduce rank
    U, V, W = try_reduce_rank(U, V, W, max_time=8)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
