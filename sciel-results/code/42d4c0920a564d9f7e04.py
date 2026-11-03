import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    
    def _build_3x2x2_rank11():
        """Build the known rank-11 decomposition for 3x2x2 matmul."""
        U = np.zeros((6, 11))
        V = np.zeros((4, 11))
        W = np.zeros((6, 11))
        
        # Strassen part (cols 0-6) for rows 0,1 of A
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        # Extra part (cols 7-10) for Row 2 of A
        U[4, 7] = 1
        V[0, 7] = 1
        W[2, 7] = 1
        
        U[5, 8] = 1
        V[2, 8] = 1
        W[4, 8] = 1
        
        U[4, 9] = 1
        V[1, 9] = 1
        W[5, 9] = 1
        
        U[5, 10] = 1
        V[3, 10] = 1
        W[5, 10] = 1
        
        return U, V, W
    
    def _decompose_error(U, V, W, tensor):
        """Compute max absolute error using vectorized einsum."""
        pred = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.max(np.abs(pred - tensor))
    
    def _build_3x2x2_tensor():
        """Build the 3x2x2 matrix multiplication tensor."""
        tensor = np.zeros((6, 4, 6))
        for a in range(6):
            a_row, a_col = divmod(a, 2)
            for b in range(4):
                b_row, b_col = divmod(b, 2)
                if a_col == b_row:
                    c = b_col * 3 + a_row
                    tensor[a, b, c] = 1.0
        return tensor
    
    def _try_rank_reduction_3x2x2(U_init, V_init, W_init, tensor, target_rank, max_time):
        """Try to reduce rank by removing redundant columns and optimizing."""
        rank = U_init.shape[1]
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        # First try simple column removal
        for col in range(rank - 1, -1, -1):
            if time.time() > start_time + max_time:
                break
            if rank <= target_rank:
                break
            mask = np.ones(rank, dtype=bool)
            mask[col] = False
            U_sub = U[:, mask]
            V_sub = V[:, mask]
            W_sub = W[:, mask]
            if _decompose_error(U_sub, V_sub, W_sub, tensor) < 1e-8:
                U, V, W = U_sub, V_sub, W_sub
                rank -= 1
        
        # If still above target, try optimization to merge columns
        if rank > target_rank:
            np.random.seed(42)
            best_err = _decompose_error(U, V, W, tensor)
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Try to find a lower-rank decomposition by optimizing
            # Reduce to target_rank columns
            U_opt = U[:, :target_rank].copy()
            V_opt = V[:, :target_rank].copy()
            W_opt = W[:, :target_rank].copy()
            
            # Add perturbation
            U_opt[:, -1] += np.random.randn(6) * 0.05
            V_opt[:, -1] += np.random.randn(4) * 0.05
            W_opt[:, -1] += np.random.randn(6) * 0.05
            
            lr = 0.003
            for it in range(5000):
                if time.time() > start_time + max_time:
                    break
                
                pred = np.einsum('ar,br,cr->abc', U_opt, V_opt, W_opt, optimize=True)
                err = pred - tensor
                
                grad_U = 2.0 * np.einsum('abc,br,cr->ar', err, V_opt, W_opt, optimize=True)
                grad_V = 2.0 * np.einsum('abc,ar,cr->br', err, U_opt, W_opt, optimize=True)
                grad_W = 2.0 * np.einsum('abc,ar,br->cr', err, U_opt, V_opt, optimize=True)
                
                U_opt -= lr * grad_U
                V_opt -= lr * grad_V
                W_opt -= lr * grad_W
                
                if it % 1000 == 0:
                    cur_err = _decompose_error(U_opt, V_opt, W_opt, tensor)
                    if cur_err < best_err:
                        best_err = cur_err
                        best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
                    if cur_err < 1e-10:
                        break
            
            if best_err < 1e-6:
                # Try rounding to half-integers
                U_r = np.round(best_U * 2) / 2
                V_r = np.round(best_V * 2) / 2
                W_r = np.round(best_W * 2) / 2
                if _decompose_error(U_r, V_r, W_r, tensor) < 1e-6:
                    return U_r, V_r, W_r
                else:
                    return best_U, best_V, best_W
        
        return U, V, W
    
    def _build_3x2x2_rank10_structured():
        """Build rank-10 decomposition using multi-resolution structure.
        
        Key insight: Apply a 2D wavelet-like decomposition to the 3x2x2 tensor.
        The 'approximation' subspace (rows 0,1 of A) can use Strassen (rank 7).
        The 'detail' subspace (row 2 of A) needs rank 3 if we can decouple it.
        
        The wavelet transform decouples:
        - Approximation: A[0:2,:] * B[0:2,:] -> C[0:2,:] (Strassen, rank 7)
        - Detail: A[2,:] * B[0:2,:] -> C[2,:] (needs rank 3)
        
        For the detail part, we need:
        C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
        C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
        
        Using 3 rank-1 terms with careful coefficient choice:
        M7 = (A[2,0] + A[2,1]) * (B[0,0] + B[1,0]) -> contributes to C[2,0]
        M8 = (A[2,0] - A[2,1]) * (B[0,1] + B[1,1]) -> contributes to C[2,1]  
        M9 = A[2,0] * B[0,1] -> correction
        
        Actually, let's use a different approach. The detail row is a 1x2 * 2x2 = 1x2 matmul.
        This has tensor rank 2 (it's just a dot product with 2 terms).
        But we need to embed it in the 3x2x2 structure.
        
        Let me try:
        M7 = A[2,0] * B[0,0]  (U[4,7]=1, V[0,7]=1, W[2,7]=1)
        M8 = A[2,1] * B[1,0]  (U[5,8]=1, V[2,8]=1, W[2,8]=1)
        M9 = A[2,0] * B[0,1]  (U[4,9]=1, V[1,9]=1, W[5,9]=1)
        M10 = A[2,1] * B[1,1] (U[5,10]=1, V[3,10]=1, W[5,10]=1)
        
        That's 4 terms, giving rank 11 total. For rank 10, we need to merge one.
        
        Try: M9' = A[2,0]*B[0,1] + A[2,1]*B[1,1] as a single rank-1 term?
        That would be (A[2,0] + A[2,1]) * (B[0,1] + B[1,1]) but that gives cross terms.
        
        Alternative: Use the Strassen structure to help with row 2.
        In Strassen, we have terms like (A[0,0]+A[1,1])*(B[0,0]+B[1,1]).
        If we modify U to include row 2 in some Strassen terms, we can save a rank.
        
        Let me try modifying the Strassen decomposition:
        Instead of U[0..3, 0..6] for Strassen, use U[0..5, 0..6] with:
        - Term 0: (A[0,0]+A[1,1]+A[2,0]+A[2,1]) * (B[0,0]+B[1,1]) -> W contributes to C[0,0],C[1,1],C[2,0],C[2,1]
        This gets complicated. Let me just use the numerical approach.
        """
        tensor = _build_3x2x2_tensor()
        U11, V11, W11 = _build_3x2x2_rank11()
        
        # Try to reduce from 11 to 10
        U, V, W = _try_rank_reduction_3x2x2(U11, V11, W11, tensor, 10, max_time=30)
        if U.shape[1] <= 10:
            return U, V, W
        
        # Try from scratch with rank 10 initialization
        np.random.seed(123)
        best_err = float('inf')
        best_U = best_V = best_W = None
        
        for trial in range(10):
            if time.time() > start_time + 60:
                break
            
            U = U11[:, :10].copy()
            V = V11[:, :10].copy()
            W = W11[:, :10].copy()
            
            # Perturb
            U[:, 9] += np.random.randn(6) * 0.1
            V[:, 9] += np.random.randn(4) * 0.1
            W[:, 9] += np.random.randn(6) * 0.1
            
            lr = 0.005
            for it in range(3000):
                if time.time() > start_time + 60:
                    break
                
                pred = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                err = pred - tensor
                
                grad_U = 2.0 * np.einsum('abc,br,cr->ar', err, V, W, optimize=True)
                grad_V = 2.0 * np.einsum('abc,ar,cr->br', err, U, W, optimize=True)
                grad_W = 2.0 * np.einsum('abc,ar,br->cr', err, U, V, optimize=True)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                if it % 500 == 0:
                    cur_err = _decompose_error(U, V, W, tensor)
                    if cur_err < best_err:
                        best_err = cur_err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if cur_err < 1e-10:
                        break
            
            cur_err = _decompose_error(U, V, W, tensor)
            if cur_err < best_err:
                best_err = cur_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-6:
            U_r = np.round(best_U * 2) / 2
            V_r = np.round(best_V * 2) / 2
            W_r = np.round(best_W * 2) / 2
            if _decompose_error(U_r, V_r, W_r, tensor) < 1e-6:
                return U_r, V_r, W_r
            else:
                return best_U, best_V, best_W
        
        return None
    
    # Build the 3x2x2 tensor for verification
    tensor_322 = _build_3x2x2_tensor()
    
    U_loc = V_loc = W_loc = None
    rank = 11
    
    # Try rank 10
    if time.time() < deadline - 50:
        result = _build_3x2x2_rank10_structured()
        if result is not None:
            U_loc, V_loc, W_loc = result
            rank = U_loc.shape[1]
    
    if U_loc is None:
        # Fall back to rank 11
        U_loc, V_loc, W_loc = _build_3x2x2_rank11()
        rank = 11
    
    total_rank = 4 * rank
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Mapping function for 3x2x2 blocks
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
    
    # Fast vectorized verification
    if time.time() < deadline:
        # Check all non-zero entries
        pred = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        
        # Build expected tensor
        full_tensor = np.zeros((n*m, m*p, p*n))
        for a in range(n*m):
            a_row, a_col = divmod(a, m)
            for b in range(m*p):
                b_row, b_col = divmod(b, p)
                if a_col == b_row:
                    c_row = a_row
                    c_col = b_col
                    c = c_col * n + c_row
                    full_tensor[a, b, c] = 1.0
        
        max_err = np.max(np.abs(pred - full_tensor))
        if max_err > 1e-10:
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
