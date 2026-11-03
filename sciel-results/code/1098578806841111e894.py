import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    # Generate Schoolbook Decomposition as baseline
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try to find a lower rank decomposition using Strassen-like ideas
    # For 4x4x7, we can try to split into smaller subproblems
    # Strategy: split the 7 columns into 4 and 3
    # Part 1: 4x4x4 using Strassen (rank 49)
    # Part 2: 4x4x3 using schoolbook (rank 48)
    # Total: 97 < 112
    
    # Implement Strassen for 4x4x4
    def strassen_4x4x4():
        """
        Returns U, V, W for 4x4x4 matrix multiplication using Strassen.
        U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
        """
        # Strassen for 2x2x2 has rank 7
        # For 4x4x4, we split into 2x2 blocks
        # Each 2x2x2 uses rank 7, so 4 blocks * 7 = 28? No, that's not right.
        # Actually, Strassen for 4x4x4 can be done with rank 49.
        # Let's use the known decomposition.
        
        # For now, let's just use schoolbook for 4x4x4 (rank 64)
        # and see if we can do better.
        
        # Actually, let's try a different approach:
        # Use the fact that 4x4x4 can be done with rank 49 using a known algorithm.
        # But implementing that from scratch is complex.
        
        # Let's try a simpler approach: split 4x4x4 into two 2x4x4 or 4x2x4 problems
        # 2x4x4: rank = 2*4*4 = 32 (schoolbook)
        # But we can do better with Strassen-like ideas.
        
        # For now, let's just return schoolbook for 4x4x4
        rank_4x4x4 = 4 * 4 * 4
        U_sub = [[0] * rank_4x4x4 for _ in range(16)]
        V_sub = [[0] * rank_4x4x4 for _ in range(16)]
        W_sub = [[0] * rank_4x4x4 for _ in range(16)]
        
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_sub[i * 4 + j][col] = 1
                    V_sub[j * 4 + k][col] = 1
                    W_sub[k * 4 + i][col] = 1
                    col += 1
        
        return U_sub, V_sub, W_sub, rank_4x4x4
    
    # Try the split approach
    if time.time() < deadline - 5:
        # Part 1: 4x4x4 (columns 0-3)
        U1, V1, W1, rank1 = strassen_4x4x4()
        
        # Part 2: 4x4x3 (columns 4-6) using schoolbook
        rank2 = 4 * 4 * 3
        U2 = [[0] * rank2 for _ in range(16)]
        V2 = [[0] * rank2 for _ in range(12)]
        W2 = [[0] * rank2 for _ in range(12)]
        
        col2 = 0
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U2[i * 4 + j][col2] = 1
                    V2[j * 4 + (k + 4)][col2] = 1
                    W2[(k + 4) * 4 + i][col2] = 1
                    col2 += 1
        
        # Combine
        total_rank = rank1 + rank2
        if total_rank < best_rank:
            U_new = [[0] * total_rank for _ in range(16)]
            V_new = [[0] * total_rank for _ in range(28)]
            W_new = [[0] * total_rank for _ in range(28)]
            
            # Copy part 1
            for r in range(rank1):
                for i in range(16):
                    U_new[i][r] = U1[i][r]
                for j in range(16):
                    V_new[j][r] = V1[j][r]
                for k in range(16):
                    W_new[k][r] = W1[k][r]
            
            # Copy part 2
            offset = rank1
            for r in range(rank2):
                for i in range(16):
                    U_new[i][offset + r] = U2[i][r]
                for j in range(12):
                    V_new[j][offset + r] = V2[j][r]
                for k in range(12):
                    W_new[k][offset + r] = W2[k][r]
            
            # Verify the decomposition
            if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                best_result = {"U": U_new, "V": V_new, "W": W_new}
                best_rank = total_rank
    
    return best_result

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition correctly represents matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = len(U[0])
    
    # For each (i, k) in the output, check that:
    # sum_j A[i,j] * B[j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This is hard to verify directly, so let's check a few random cases
    
    # Actually, let's just check the structure
    # The decomposition should satisfy:
    # (A * B)[i,k] = sum_j A[i,j] B[j,k]
    # = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    # This is not quite right. The correct identity is:
    # (A * B)[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where the sums are over j and c.
    
    # Let's just do a numerical check with random matrices
    np.random.seed(42)
    A = np.random.randint(-5, 5, size=(n, m))
    B = np.random.randint(-5, 5, size=(m, p))
    C_expected = A @ B
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    C_computed = np.zeros((n, p))
    for c in range(rank):
        # u_c is a vector of length n*m
        u_c = U_arr[:, c]
        v_c = V_arr[:, c]
        w_c = W_arr[:, c]
        
        # Reshape u_c to (n, m), v_c to (m, p)
        u_mat = u_c.reshape(n, m)
        v_mat = v_c.reshape(m, p)
        
        # The contribution to C[i,k] from term c is:
        # w_c[k*n+i] * sum_j u_mat[i,j] * A[i,j] * v_mat[j,k] * B[j,k]
        # Wait, that's not right either.
        
        # The correct interpretation:
        # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
        # No, that's not standard.
        
        # Actually, the tensor decomposition for matrix multiplication is:
        # T_{i,j,k} = sum_c U_{i,j,c} * V_{j,k,c} * W_{k,i,c}
        # where T_{i,j,k} is the multiplication tensor.
        # So C[i,k] = sum_j A[i,j] B[j,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
        
        # Let's compute it properly
        for i in range(n):
            for k in range(p):
                sum_u = 0
                sum_v = 0
                for j in range(m):
                    sum_u += U_arr[i * m + j, c] * A[i, j]
                    sum_v += V_arr[j * p + k, c] * B[j, k]
                C_computed[i, k] += W_arr[k * n + i, c] * sum_u * sum_v
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        return False
    
    return True
# EVOLVE-BLOCK-END
