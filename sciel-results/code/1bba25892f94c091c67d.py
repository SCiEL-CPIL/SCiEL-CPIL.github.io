"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    def make_schoolbook():
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

    best_result = make_schoolbook()
    best_rank = n * m * p

    # Precompute index arrays for efficient verification
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx

    def verify(U, V, W, r):
        """Check if U, V, W form a valid tensor decomposition. Returns max error."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        u_vals = Ua[u_idx, :]  # (n*m*p, r)
        v_vals = Va[v_idx, :]  # (n*m*p, r)
        w_vals = Wa[w_idx, :]  # (n*m*p, r)
        
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        
        return np.max(np.abs(result - 1.0))

    def strassen_2x2x2():
        """Return (U, V, W) for 2x2x2 Strassen decomposition, rank 7."""
        # Strassen's 7-term algorithm for 2x2 matrix multiplication
        U = [
            [1, 0, 0, 0, 0, 0, 0],  # (0,0)
            [0, 1, 0, 0, 0, 0, 0],  # (0,1)
            [0, 0, 1, 0, 0, 0, 0],  # (1,0)
            [0, 0, 0, 1, 0, 0, 0],  # (1,1)
        ]
        V = [
            [1, 0, 1, 0, 0, 0, 0],  # (0,0)
            [0, 1, 0, 1, 0, 0, 0],  # (0,1)
            [1, 0, 0, 0, 0, 0, 0],  # (1,0)
            [0, 1, 0, 0, 0, 0, 0],  # (1,1)
        ]
        W = [
            [1, 0, 0, 0, 0, 0, 0],  # (0,0)
            [0, 1, 0, 0, 0, 0, 0],  # (0,1)
            [0, 0, 1, 0, 0, 0, 0],  # (1,0)
            [0, 0, 0, 1, 0, 0, 0],  # (1,1)
        ]
        # Actually, let me use the standard Strassen formulas:
        # M1 = (a00+a11)*(b00+b11)
        # M2 = (a10+a11)*b00
        # M3 = a00*(b01-b11)
        # M4 = a11*(b10-b00)
        # M5 = (a00+a01)*b11
        # M6 = (a10-a00)*(b00+b01)
        # M7 = (a01-a11)*(b10+b11)
        #
        # U[i*m+j, c] where i,j in {0,1}, c in {0,...,6}
        # V[j*p+k, c] where j,k in {0,1}, c in {0,...,6}
        # W[k*n+i, c] where k,i in {0,1}, c in {0,...,6}
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1: (a00+a11)*(b00+b11) -> contributes to all 4 outputs
        U[0, 0] = 1  # a00
        U[3, 0] = 1  # a11
        V[0, 0] = 1  # b00
        V[3, 0] = 1  # b11
        W[0, 0] = 1  # (0,0)
        W[1, 0] = 1  # (0,1)
        W[2, 0] = 1  # (1,0)
        W[3, 0] = 1  # (1,1)
        
        # M2: (a10+a11)*b00 -> contributes to (1,0) and (1,1)
        U[2, 1] = 1  # a10
        U[3, 1] = 1  # a11
        V[0, 1] = 1  # b00
        W[2, 1] = 1  # (1,0)
        W[3, 1] = 1  # (1,1)
        
        # M3: a00*(b01-b11) -> contributes to (0,1) and (1,1) with sign
        U[0, 2] = 1  # a00
        V[1, 2] = 1  # b01
        V[3, 2] = -1  # -b11
        W[1, 2] = 1  # (0,1)
        W[3, 2] = 1  # (1,1)
        
        # M4: a11*(b10-b00) -> contributes to (1,0) and (0,0) with sign
        U[3, 3] = 1  # a11
        V[2, 3] = 1  # b10
        V[0, 3] = -1  # -b00
        W[2, 3] = 1  # (1,0)
        W[0, 3] = 1  # (0,0)
        
        # M5: (a00+a01)*b11 -> contributes to (0,0) and (0,1)
        U[0, 4] = 1  # a00
        U[1, 4] = 1  # a01
        V[3, 4] = 1  # b11
        W[0, 4] = 1  # (0,0)
        W[1, 4] = 1  # (0,1)
        
        # M6: (a10-a00)*(b00+b01) -> contributes to (1,0) and (1,1)
        U[2, 5] = 1  # a10
        U[0, 5] = -1  # -a00
        V[0, 5] = 1  # b00
        V[1, 5] = 1  # b01
        W[2, 5] = 1  # (1,0)
        W[3, 5] = 1  # (1,1)
        
        # M7: (a01-a11)*(b10+b11) -> contributes to (0,1) and (1,1)
        U[1, 6] = 1  # a01
        U[3, 6] = -1  # -a11
        V[2, 6] = 1  # b10
        V[3, 6] = 1  # b11
        W[1, 6] = 1  # (0,1)
        W[3, 6] = 1  # (1,1)
        
        return U.tolist(), V.tolist(), W.tolist()

    def schoolbook_2x2x1():
        """Return (U, V, W) for 2x2x1 schoolbook decomposition, rank 2."""
        # For multiplying 2x2 by 2x1: two independent multiplications
        U = np.zeros((4, 2))
        V = np.zeros((2, 2))
        W = np.zeros((2, 2))
        
        # Output (0,0): a00 * b0
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        
        # Output (1,0): a10 * b0
        U[2, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        
        return U.tolist(), V.tolist(), W.tolist()

    def kronecker_decomp(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
        """
        Combine two decompositions via Kronecker product.
        First: n1 x m1 x p1 with rank r1
        Second: n2 x m2 x p2 with rank r2
        Result: (n1*n2) x (m1*m2) x (p1*p2) with rank r1*r2
        """
        r1 = len(W1[0]) if W1 else 0
        r2 = len(W2[0]) if W2 else 0
        r = r1 * r2
        
        n = n1 * n2
        m = m1 * m2
        p = p1 * p2
        
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        U1a = np.array(U1)
        V1a = np.array(V1)
        W1a = np.array(W1)
        U2a = np.array(U2)
        V2a = np.array(V2)
        W2a = np.array(W2)
        
        for c1 in range(r1):
            for c2 in range(r2):
                c = c1 * r2 + c2
                
                # U[i1*m1+j1, c1] * U[i2*m2+j2, c2] -> U[(i1*n2+i2)*m2+j2, c]
                # Actually: U_index = (i1*n2 + i2) * (m1*m2) + (j1*m2 + j2)
                # Wait, let me think about this more carefully.
                # The first matrix is n1 x m1, second is n2 x m2.
                # Combined: (n1*n2) x (m1*m2)
                # Row index: i_combined = i1 * n2 + i2
                # Col index: j_combined = j1 * m2 + j2
                # So U_combined[(i1*n2+i2)*(m1*m2) + (j1*m2+j2), c] = U1[i1*m1+j1, c1] * U2[i2*m2+j2, c2]
                
                for i1 in range(n1):
                    for j1 in range(m1):
                        val1 = U1a[i1 * m1 + j1, c1]
                        if val1 == 0:
                            continue
                        for i2 in range(n2):
                            for j2 in range(m2):
                                val2 = U2a[i2 * m2 + j2, c2]
                                if val2 == 0:
                                    continue
                                row = (i1 * n2 + i2) * (m1 * m2) + (j1 * m2 + j2)
                                U[row, c] += val1 * val2
                
                # V[j1*p1+k1, c1] * V[j2*p2+k2, c2] -> V[(j1*m2+j2)*p2+k2, c]
                # Combined: (m1*m2) x (p1*p2)
                # Row index: j_combined = j1 * m2 + j2
                # Col index: k_combined = k1 * p2 + k2
                # So V_combined[(j1*m2+j2)*(p1*p2) + (k1*p2+k2), c] = V1[j1*p1+k1, c1] * V2[j2*p2+k2, c2]
                
                for j1 in range(m1):
                    for k1 in range(p1):
                        val1 = V1a[j1 * p1 + k1, c1]
                        if val1 == 0:
                            continue
                        for j2 in range(m2):
                            for k2 in range(p2):
                                val2 = V2a[j2 * p2 + k2, c2]
                                if val2 == 0:
                                    continue
                                row = (j1 * m2 + j2) * (p1 * p2) + (k1 * p2 + k2)
                                V[row, c] += val1 * val2
                
                # W[k1*n1+i1, c1] * W[k2*n2+i2, c2] -> W[(k1*p2+k2)*n2+i2, c]
                # Wait, the W index is k*n+i where n is the number of rows.
                # For combined: p_combined = p1*p2, n_combined = n1*n2
                # W_combined[(k1*p2+k2)*(n1*n2) + (i1*n2+i2), c] = W1[k1*n1+i1, c1] * W2[k2*n2+i2, c2]
                
                for k1 in range(p1):
                    for i1 in range(n1):
                        val1 = W1a[k1 * n1 + i1, c1]
                        if val1 == 0:
                            continue
                        for k2 in range(p2):
                            for i2 in range(n2):
                                val2 = W2a[k2 * n2 + i2, c2]
                                if val2 == 0:
                                    continue
                                row = (k1 * p2 + k2) * (n1 * n2) + (i1 * n2 + i2)
                                W[row, c] += val1 * val2
        
        return U.tolist(), V.tolist(), W.tolist()

    def eliminate_redundant(U, V, W, r, tol=1e-10):
        """
        Try to reduce rank by identifying linearly dependent columns.
        Uses Gaussian elimination on the combined tensor representation.
        """
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        # Check which columns are actually needed
        # A column c is redundant if it can be expressed as a linear combination of others
        # This is complex, so let's just verify and return as-is for now
        # In a more advanced version, we could do rank reduction here
        
        return U, V, W, r

    # Strategy: Use Kronecker product of Strassen(2x2x2) and schoolbook(2x2x1)
    # This gives 4x4x2 with rank 7*2 = 14
    # For 4x4x7, we need to handle the 7 columns differently
    
    # Actually, let's try a different approach:
    # Decompose 4x4x7 as blocks:
    # The 7 columns can be split as 2+2+2+1
    # Use Strassen for 2x2x2 blocks (rank 7 each)
    # Use schoolbook for 2x2x1 block (rank 2)
    # Total: 3*7 + 2 = 23 for the 4x4 part handling 7 columns
    
    # But we need to be careful about how the blocks interact.
    # Let me try a direct construction:
    
    def construct_block_decomp():
        """
        Construct a decomposition for 4x4x7 using block structure.
        Split the 7 output columns into groups: [2, 2, 2, 1]
        For each group of 2, use Strassen-like approach.
        For the group of 1, use schoolbook.
        """
        n, m, p = 4, 4, 7
        
        # We'll build this by combining smaller decompositions
        # First, handle the 4x4x2 part using Strassen (rank 7)
        # Then handle the 4x4x1 part using schoolbook (rank 4)
        # But 2+1 = 3, not 7. We need 7 columns.
        
        # Alternative: Use 4 separate 2x2x2 blocks for the 4x4 part,
        # each contributing to 2 of the 7 columns, plus 1 more column.
        
        # Let me try: decompose as Kronecker of (2x2x2 Strassen) and (2x2x1 schoolbook)
        # This gives 4x4x2 with rank 14
        # Then we need to extend to 4x4x7
        
        # Actually, the cleanest approach:
        # 4x4x7 = sum of 7 independent 4x4x1 problems
        # Each 4x4x1 can be done in rank 4 (schoolbook)
        # Total rank: 7*4 = 28
        
        # But we can do better by sharing computations.
        
        # Let me just use the Kronecker product approach properly:
        # Strassen 2x2x2 (rank 7) for the 4x4 matrix multiply
        # Combined with 2x2x1 schoolbook (rank 2) for the vector multiply
        # This gives 4x4x2 with rank 14
        
        # For 4x4x7, we can think of it as:
        # C = A * B where A is 4x4, B is 4x7
        # Split B into B1 (4x2), B2 (4x2), B3 (4x2), B4 (4x1)
        # C1 = A*B1 (4x2), C2 = A*B2 (4x2), C3 = A*B3 (4x2), C4 = A*B4 (4x1)
        
        # For C1, C2, C3: use Strassen-based 4x4x2 (rank 14 each)
        # For C4: use schoolbook 4x4x1 (rank 4)
        # Total rank: 3*14 + 4 = 46
        
        # That's better than 112!
        
        U_parts = []
        V_parts = []
        W_parts = []
        
        # Get the 4x4x2 Strassen-based decomposition
        U_s, V_s, W_s = strassen_2x2x2()
        U_sb, V_sb, W_sb = schoolbook_2x2x1()
        
        # Kronecker product: 4x4x2 with rank 14
        U_442, V_442, W_442 = kronecker_decomp(U_s, V_s, W_s, U_sb, V_sb, W_sb, 2, 2, 2, 2, 2, 1)
        
        # Get 4x4x1 schoolbook decomposition
        U_441 = np.zeros((16, 4))
        V_441 = np.zeros((4, 4))
        W_441 = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                U_441[i*4+j, j] = 1  # A[i,j]
                V_441[j, j] = 1  # B[j,0]
                W_441[i, i] = 1  # C[i,0]
        
        # Now combine: 3 copies of 4x4x2 and 1 copy of 4x4x1
        # Total rank: 3*14 + 4 = 46
        
        total_rank = 3 * 14 + 4
        U = np.zeros((16, total_rank))
        V = np.zeros((4*7, total_rank))
        W = np.zeros((7*4, total_rank))
        
        col_offset = 0
        
        # Copy 1: columns 0-1 of B
        U[:, col_offset:col_offset+14] = np.array(U_442)
        V[0*4:2*4, col_offset:col_offset+14] = np.array(V_442)  # Rows 0-7 of V (for B cols 0-1)
        W[0*4:2*4, col_offset:col_offset+14] = np.array(W_442)  # Rows 0-7 of W (for C cols 0-1)
        col_offset += 14
        
        # Copy 2: columns 2-3 of B
        U[:, col_offset:col_offset+14] = np.array(U_442)
        V[2*4:4*4, col_offset:col_offset+14] = np.array(V_442)  # Rows 8-15 of V (for B cols 2-3)
        W[2*4:4*4, col_offset:col_offset+14] = np.array(W_442)  # Rows 8-15 of W (for C cols 2-3)
        col_offset += 14
        
        # Copy 3: columns 4-5 of B
        U[:, col_offset:col_offset+14] = np.array(U_442)
        V[4*4:6*4, col_offset:col_offset+14] = np.array(V_442)  # Rows 16-23 of V (for B cols 4-5)
        W[4*4:6*4, col_offset:col_offset+14] = np.array(W_442)  # Rows 16-23 of W (for C cols 4-5)
        col_offset += 14
        
        # Copy 4: column 6 of B
        U[:, col_offset:col_offset+4] = np.array(U_441)
        V[6*4:7*4, col_offset:col_offset+4] = np.array(V_441)  # Rows 24-27 of V (for B col 6)
        W[6*4:7*4, col_offset:col_offset+4] = np.array(W_441)  # Rows 24-27 of W (for C col 6)
        col_offset += 4
        
        return U.tolist(), V.tolist(), W.tolist(), total_rank

    # Try the block decomposition
    try:
        U_block, V_block, W_block, rank_block = construct_block_decomp()
        err = verify(U_block, V_block, W_block, rank_block)
        if err < 1e-8:
            best_rank = rank_block
            best_result = {
                "U": U_block,
                "V": V_block,
                "W": W_block
            }
    except Exception as e:
        pass

    # If we still have the schoolbook, try to improve with random search
    if best_rank == n * m * p:
        # Try random search for better rank
        for r in range(40, min(best_rank, 60)):
            if time.time() > deadline - 10:
                break
            
            for trial in range(10):
                if time.time() > deadline - 5:
                    break
                
                np.random.seed(trial * 7919 + r)
                
                U = np.random.randn(n * m, r) * 0.5
                V = np.random.randn(m * p, r) * 0.5
                W = np.random.randn(p * n, r) * 0.5
                
                # Simple gradient descent
                for it in range(30):
                    if time.time() > deadline - 3:
                        break
                    
                    u_vals = U[u_idx, :]
                    v_vals = V[v_idx, :]
                    w_vals = W[w_idx, :]
                    
                    prod = u_vals * v_vals * w_vals
                    result = prod.sum(axis=1)
                    error = result - 1.0
                    err_norm = np.linalg.norm(error)
                    
                    if err_norm < 1e-8:
                        break
                    
                    err_4d = error.reshape(n, m, p, 1)
                    u_4d = u_vals.reshape(n, m, p, r)
                    v_4d = v_vals.reshape(n, m, p, r)
                    w_4d = w_vals.reshape(n, m, p, r)
                    
                    grad_U = np.sum(2 * err_4d * v_4d * w_4d, axis=2).reshape(n*m, r)
                    grad_V = np.zeros((m*p, r))
                    grad_W = np.zeros((p*n, r))
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                grad_V[j*p+k, :] += 2 * error[i*m*p+j*p+k] * U[i*m+j, :] * W[k*n+i, :]
                                grad_W[k*n+i, :] += 2 * error[i*m*p+j*p+k] * U[i*m+j, :] * V[j*p+k, :]
                    
                    lr = 0.1 / (1.0 + it * 0.2)
                    U -= lr * grad_U
                    V -= lr * grad_V
                    W -= lr * grad_W
                
                err = verify(U, V, W, r)
                if err < 1e-6:
                    best_rank = r
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    break
            
            if best_rank < n * m * p:
                break

    if time.time() > deadline:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
