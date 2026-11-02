import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    
    # Search for rank-10 decomposition of 3x2x2 matrix multiplication
    # A: 3x2 (6 entries), B: 2x2 (4 entries), C: 3x2 (6 entries)
    # C[i,j] = sum_k A[i,k] * B[k,j]
    
    def try_rank10_3x2x2(max_time=60):
        """Search for rank-10 decomposition of 3x2x2 multiplication."""
        rank = 10
        U = np.zeros((6, rank))
        V = np.zeros((4, rank))
        W = np.zeros((6, rank))
        
        # Initialize with Strassen-like structure
        # M1 = (A00+A11)*(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)*B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = A00*(B10-B00)
        U[0, 2] = 1
        V[2, 2] = 1; V[0, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A11*(B01-B11)
        U[3, 3] = 1
        V[1, 3] = 1; V[3, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00+A01)*B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)*(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)*(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        # For row 2, try 3 products with shared B factors:
        # M8 = A[2,0]*(B[0,0]+B[0,1])
        U[4, 7] = 1
        V[0, 7] = 1; V[1, 7] = 1
        W[4, 7] = 1; W[5, 7] = 1
        
        # M9 = A[2,1]*(B[1,0]+B[1,1])
        U[5, 8] = 1
        V[2, 8] = 1; V[3, 8] = 1
        W[4, 8] = 1; W[5, 8] = 1
        
        # M10 = A[2,0]*B[0,0] - but we need to separate C[2,0] and C[2,1]
        # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
        # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
        # From M8: A[2,0]*(B[0,0]+B[0,1]) contributes to both
        # From M9: A[2,1]*(B[1,0]+B[1,1]) contributes to both
        # C[2,0] = M8_part + M9_part where M8_part = A[2,0]*B[0,0], M9_part = A[2,1]*B[1,0]
        # This doesn't work directly.
        
        # Let me try a different approach: optimize the last 3 columns
        # Keep first 7 columns fixed (Strassen for rows 0,1)
        # Optimize columns 7,8,9 for row 2
        
        # Row 2 needs: C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
        #              C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
        
        # Try: M8 = A[2,0]*B[0,0], M9 = A[2,1]*B[1,0], M10 = A[2,0]*B[0,1] + A[2,1]*B[1,1]?
        # No, M10 can't be a sum of two products in a single rank-1 term.
        
        # Alternative: M8 = (A[2,0]+A[2,1])*B[0,0], M9 = A[2,1]*B[1,0], M10 = A[2,0]*B[0,1]
        # Then C[2,0] = M8 + M9 - A[2,1]*B[0,0]... no this gets complicated.
        
        # Let me try: M8 = A[2,0]*B[0,0], M9 = A[2,1]*B[1,0], M10 = (A[2,0]*B[0,1] + A[2,1]*B[1,1])
        # M10 can't be rank 1.
        
        # What about: M8 = A[2,0]*(B[0,0]+B[1,0]), M9 = A[2,1]*B[1,0], M10 = A[2,0]*B[0,1]
        # C[2,0] = M8 + M9 - A[2,1]*B[1,0]... no.
        
        # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
        # If M8 = A[2,0]*B[0,0], M9 = A[2,1]*B[1,0], then C[2,0] = M8 + M9
        # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
        # M10 would need to give both A[2,0]*B[0,1] and A[2,1]*B[1,1]
        
        # Try: M10 = (A[2,0]+A[2,1])*(B[0,1]+B[1,1])? 
        # Then M10 = A[2,0]*B[0,1] + A[2,0]*B[1,1] + A[2,1]*B[0,1] + A[2,1]*B[1,1]
        # C[2,1] = M10 - A[2,0]*B[1,1] - A[2,1]*B[0,1]
        # We don't have those terms.
        
        # Let me try a completely different set of 10 products.
        # Use 4 products for row 0, 4 for row 1, 2 for row 2? No, that's 10 but rows 0,1 need at least 4 each.
        
        # Actually, the minimum rank for 2x2x2 is 7 (Strassen). For 3x2x2, 
        # we can do 7 for the top 2x2x2 and then 3 more for the third row = 10.
        # But the third row needs 4 products in general.
        
        # Unless we can share some B factors. Let me try:
        # M8 = A[2,0]*B[0,0]
        # M9 = A[2,0]*B[0,1]  
        # M10 = A[2,1]*(B[1,0]+B[1,1])
        # Then C[2,0] = M8 + A[2,1]*B[1,0] and C[2,1] = M9 + A[2,1]*B[1,1]
        # We still need A[2,1]*B[1,0] and A[2,1]*B[1,1] separately.
        
        # It seems rank 10 might not be possible for 3x2x2. Let me verify rank 11 is correct.
        
        # Actually, let me try to optimize a rank-10 solution using gradient descent.
        
        # Define the tensor: T[a,b,c] = 1 if a_col == b_row, where a=(a_row,a_col), b=(b_row,b_col), c=(c_row,c_col)
        # a: 0..5 (a_row*2+a_col), b: 0..3 (b_row*2+b_col), c: 0..5 (c_col*3+c_row)
        
        tensor = np.zeros((6, 4, 6))
        for a in range(6):
            a_row, a_col = divmod(a, 2)
            for b in range(4):
                b_row, b_col = divmod(b, 2)
                if a_col == b_row:
                    c = b_col * 3 + a_row
                    tensor[a, b, c] = 1.0
        
        # Try to find U, V, W of shape (6,10), (4,10), (6,10) such that
        # sum_r U[a,r]*V[b,r]*W[c,r] = tensor[a,b,c]
        
        # Use a simple gradient descent approach
        np.random.seed(42)
        U = np.random.randn(6, rank) * 0.1
        V = np.random.randn(4, rank) * 0.1
        W = np.random.randn(6, rank) * 0.1
        
        # Initialize with the known rank-11 solution truncated
        U[:6, :7] = U[:6, :7]  # Keep Strassen part
        V[:4, :7] = V[:4, :7]
        W[:6, :7] = W[:6, :7]
        
        # Optimize the last 3 columns
        lr = 0.01
        for it in range(5000):
            if time.time() > start_time + max_time:
                break
            
            # Compute gradient
            grad_U = np.zeros((6, rank))
            grad_V = np.zeros((4, rank))
            grad_W = np.zeros((6, rank))
            
            loss = 0.0
            for a in range(6):
                for b in range(4):
                    for c in range(6):
                        if tensor[a, b, c] == 0:
                            continue
                        pred = np.sum(U[a, :] * V[b, :] * W[c, :])
                        err = pred - tensor[a, b, c]
                        loss += err * err
                        for r in range(rank):
                            grad_U[a, r] += 2 * err * V[b, r] * W[c, r]
                            grad_V[b, r] += 2 * err * U[a, r] * W[c, r]
                            grad_W[c, r] += 2 * err * U[a, r] * V[b, r]
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            if it % 1000 == 0:
                # Check accuracy
                max_err = 0.0
                for a in range(6):
                    for b in range(4):
                        for c in range(6):
                            pred = np.sum(U[a, :] * V[b, :] * W[c, :])
                            max_err = max(max_err, abs(pred - tensor[a, b, c]))
        
        # Check if we found a solution
        max_err = 0.0
        for a in range(6):
            for b in range(4):
                for c in range(6):
                    pred = np.sum(U[a, :] * V[b, :] * W[c, :])
                    max_err = max(max_err, abs(pred - tensor[a, b, c]))
        
        if max_err < 1e-6:
            # Round to integers/half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify rounded solution
            max_err_r = 0.0
            for a in range(6):
                for b in range(4):
                    for c in range(6):
                        pred = np.sum(U_rounded[a, :] * V_rounded[b, :] * W_rounded[c, :])
                        max_err_r = max(max_err_r, abs(pred - tensor[a, b, c]))
            
            if max_err_r < 1e-6:
                return U_rounded, V_rounded, W_rounded
        
        return None
    
    # Try to find rank-10 solution
    U10, V10, W10 = None, None, None
    if time.time() < deadline - 60:
        result = try_rank10_3x2x2(max_time=50)
        if result is not None:
            U10, V10, W10 = result
    
    if U10 is not None:
        rank = 10
        U_loc, V_loc, W_loc = U10, V10, W10
    else:
        # Fall back to rank 11
        rank = 11
        U_loc = np.zeros((6, 11))
        V_loc = np.zeros((4, 11))
        W_loc = np.zeros((6, 11))
        
        # Strassen part (cols 0-6) for rows 0,1 of A
        U_loc[0, 0] = 1; U_loc[3, 0] = 1
        V_loc[0, 0] = 1; V_loc[3, 0] = 1
        W_loc[0, 0] = 1; W_loc[3, 0] = 1
        
        U_loc[2, 1] = 1; U_loc[3, 1] = 1
        V_loc[0, 1] = 1
        W_loc[2, 1] = 1; W_loc[3, 1] = -1
        
        U_loc[0, 2] = 1
        V_loc[1, 2] = 1; V_loc[3, 2] = -1
        W_loc[1, 2] = 1; W_loc[3, 2] = 1
        
        U_loc[3, 3] = 1
        V_loc[2, 3] = 1; V_loc[0, 3] = -1
        W_loc[0, 3] = 1; W_loc[2, 3] = 1
        
        U_loc[0, 4] = 1; U_loc[1, 4] = 1
        V_loc[3, 4] = 1
        W_loc[0, 4] = -1; W_loc[1, 4] = 1
        
        U_loc[2, 5] = 1; U_loc[0, 5] = -1
        V_loc[0, 5] = 1; V_loc[1, 5] = 1
        W_loc[3, 5] = 1
        
        U_loc[1, 6] = 1; U_loc[3, 6] = -1
        V_loc[2, 6] = 1; V_loc[3, 6] = 1
        W_loc[0, 6] = 1
        
        # Extra part (cols 7-10) for Row 2 of A
        U_loc[4, 7] = 1
        V_loc[0, 7] = 1
        W_loc[4, 7] = 1
        
        U_loc[5, 8] = 1
        V_loc[2, 8] = 1
        W_loc[4, 8] = 1
        
        U_loc[4, 9] = 1
        V_loc[1, 9] = 1
        W_loc[5, 9] = 1
        
        U_loc[5, 10] = 1
        V_loc[3, 10] = 1
        W_loc[5, 10] = 1
    
    total_rank = 4 * rank
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Mapping function
    def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * rank
        
        for r in range(3):
            for c in range(2):
                local_a_idx = r * 2 + c
                global_a_row = a_row_start + r
                global_a_col = a_col_start + c
                global_a_idx = global_a_row * m + global_a_col
                U[global_a_idx, col_offset:col_offset+rank] += U_loc[local_a_idx, :]
                
        for r in range(2):
            for c in range(2):
                local_b_idx = r * 2 + c
                global_b_row = b_row_start + r
                global_b_col = b_col_start + c
                global_b_idx = global_b_row * p + global_b_col
                V[global_b_idx, col_offset:col_offset+rank] += V_loc[local_b_idx, :]
                
        for r in range(3):
            for c in range(2):
                local_c_idx = r * 2 + c
                global_c_row = c_row_start + r
                global_c_col = c_col_start + c
                global_c_idx = global_c_col * n + global_c_row
                W[global_c_idx, col_offset:col_offset+rank] += W_loc[local_c_idx, :]

    add_block(0, 0, 0, 0, 0, 0, 0)
    add_block(1, 0, 2, 2, 0, 0, 0)
    add_block(2, 0, 0, 0, 2, 0, 2)
    add_block(3, 0, 2, 2, 2, 0, 2)
    
    # Verify decomposition is correct
    if time.time() < deadline:
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Vectorized verification
        for a in range(n*m):
            a_row, a_col = divmod(a, m)
            for b in range(m*p):
                b_row, b_col = divmod(b, p)
                if a_col != b_row:
                    continue
                c_row = a_row
                c_col = b_col
                c = c_col * n + c_row
                expected = 1.0
                computed = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                if abs(computed - expected) > 1e-10:
                    # Fallback to trivial rank-48
                    U_fallback = np.zeros((n*m, n*m*p))
                    V_fallback = np.zeros((m*p, n*m*p))
                    W_fallback = np.zeros((p*n, n*m*p))
                    for a_idx in range(n*m):
                        a_r, a_c = divmod(a_idx, m)
                        for b_idx in range(m*p):
                            b_r, b_c = divmod(b_idx, p)
                            if a_c == b_r:
                                c_idx = b_c * n + a_r
                                rank_idx = a_idx * (m*p) + b_idx
                                U_fallback[a_idx, rank_idx] = 1
                                V_fallback[b_idx, rank_idx] = 1
                                W_fallback[c_idx, rank_idx] = 1
                    return {
                        "U": U_fallback.tolist(),
                        "V": V_fallback.tolist(),
                        "W": W_fallback.tolist()
                    }
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
