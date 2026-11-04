import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 135  # safety margin before 150s kill

def _elapsed():
    return time.time() - START_TIME

def _deadline_reached():
    return _elapsed() > TIME_LIMIT

def _strassen_2x2x2():
    """Strassen decomposition for 2x2x2 matrix multiplication. Rank 7."""
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[3, 4] = -1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _strassen_2x2x3():
    """
    Strassen-style decomposition for 2x2 x 2x3 matrix multiplication.
    A is 2x2, B is 2x3, C is 2x3.
    Uses 10 scalar multiplications (rank 10).
    """
    n, m, p = 2, 2, 3
    rank = 10
    U = np.zeros((n * m, rank), dtype=np.float64)  # (4, 10)
    V = np.zeros((m * p, rank), dtype=np.float64)  # (6, 10)
    W = np.zeros((p * n, rank), dtype=np.float64)  # (6, 10)
    
    # Indices: A[i,j] -> U[i*2+j], B[j,k] -> V[j*3+k], C[k,i] -> W[k*2+i]
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01, b02], [b10, b11, b12]]
    # C = [[c00, c01, c02], [c10, c11, c12]]
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # 10 multiplications:
    # M1 = (a00+a11)*(b00+b10)
    # M2 = (a00+a11)*(b01+b11)
    # M3 = (a00+a11)*(b02+b12)
    # M4 = (a10+a11)*b00
    # M5 = (a10+a11)*b01
    # M6 = (a10+a11)*b02
    # M7 = a00*(b10-b00)
    # M8 = a00*(b11-b01)
    # M9 = a00*(b12-b02)
    # M10 = a11*(b00-b10)  -- wait, this isn't quite right for 2x2x3
    
    # Let me use a proper 2x2x3 Strassen decomposition.
    # From the literature, multiplying 2x2 by 2x3 can be done in 10 multiplications.
    
    # Let me derive it more carefully.
    # C[0,0] = a00*b00 + a01*b10
    # C[0,1] = a00*b01 + a01*b11
    # C[0,2] = a00*b02 + a01*b12
    # C[1,0] = a10*b00 + a11*b10
    # C[1,1] = a10*b01 + a11*b11
    # C[1,2] = a10*b02 + a11*b12
    
    # Define:
    # P1 = (a00 + a11) * (b00 + b10)
    # P2 = (a00 + a11) * (b01 + b11)
    # P3 = (a00 + a11) * (b02 + b12)
    # P4 = (a10 + a11) * b00
    # P5 = (a10 + a11) * b01
    # P6 = (a10 + a11) * b02
    # P7 = a00 * (b10 - b00)
    # P8 = a00 * (b11 - b01)
    # P9 = a00 * (b12 - b02)
    # P10 = a11 * b00  -- hmm, this doesn't work cleanly
    
    # Actually, let me use a known rank-10 decomposition.
    # The key insight: split B into B_left (2x2) and B_right (2x1).
    # Use Strassen for the 2x2x2 part (7 mults) and 3 more for the 2x2x1 part.
    # But 2x2x1 can be done in 3 mults (schoolbook).
    # Total: 7 + 3 = 10.
    
    # Strassen part (2x2 x 2x2):
    # M1 = (a00+a11)*(b00+b10)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b10-b00)
    # M4 = a11*(b00-b10)  -- wait, standard Strassen uses different terms
    
    # Let me just use the standard 2x2x2 Strassen for columns 0,1 and add 3 more for column 2.
    # For the 2x2x2 part (columns 0,1):
    # M1 = (a00+a11)*(b00+b10)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b10-b00)
    # M4 = a11*(b00-b10)  -- No, standard Strassen:
    # M1 = (a00+a11)*(b00+b10)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b10-b00)  
    # M4 = a11*(b00-b10)
    # M5 = (a00+a01)*b10
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    
    # For column 2, we need:
    # C[0,2] = a00*b02 + a01*b12
    # C[1,2] = a10*b02 + a11*b12
    
    # We can reuse some structure. Let's define:
    # M8 = (a00+a11)*b02
    # M9 = (a10+a11)*b02  -- no, this is getting complicated.
    
    # Simpler approach: Just use 3 schoolbook multiplications for column 2:
    # M8 = a00*b02
    # M9 = a01*b12
    # M10 = a10*b02
    # M11 = a11*b12
    # But that's 4, giving rank 11.
    
    # Better: 
    # M8 = (a00+a11)*b02
    # M9 = a01*b12
    # M10 = (a10-a00)*b02  -- no...
    
    # Let me think differently. For the 2x2x1 part (column 2):
    # C[0,2] = a00*b02 + a01*b12
    # C[1,2] = a10*b02 + a11*b12
    # This is a 2x2 x 1x2 problem. Can be done in 3 mults:
    # M8 = (a00+a10)*b02
    # M9 = (a01+a11)*b12
    # M10 = a00*b12  -- no...
    # Actually: 
    # M8 = (a00+a11)*b02
    # M9 = (a01+a10)*b12
    # M10 = a11*b02  -- still not clean.
    
    # Simplest: use 3 mults for the 2x2x1:
    # M8 = a00*b02
    # M9 = a01*b12  
    # M10 = (a10+a11)*b02  -- no, C[1,2] needs a10*b02 + a11*b12
    
    # OK, let me just use schoolbook for the third column: 4 mults.
    # Total rank = 7 + 4 = 11. Not as good.
    
    # Alternative: Use a different decomposition.
    # Actually, the known result is that 2x2x3 has rank 10.
    # Let me look up the specific construction.
    
    # From Bürgisser's work, one rank-10 decomposition:
    # Let me try:
    # M1 = (a00+a11)*(b00+b10)
    # M2 = (a00+a11)*(b01+b11)
    # M3 = (a00+a11)*(b02+b12)
    # M4 = (a10+a11)*b00
    # M5 = (a10+a11)*b01
    # M6 = (a10+a11)*b02
    # M7 = a00*(b10-b00)
    # M8 = a00*(b11-b01)
    # M9 = a00*(b12-b02)
    # M10 = a11*b00  -- this gives C[1,0] = M4 + M10 - M7 = (a10+a11)*b00 + a11*b00 - a00*(b10-b00)
    #     = a10*b00 + a11*b00 + a11*b00 - a00*b10 + a00*b00
    #     = a10*b00 + 2*a11*b00 - a00*b10 + a00*b00
    #     That's wrong.
    
    # Let me be more careful. I'll use a verified decomposition.
    # Actually, let me just use the simple approach:
    # Split into 2x2x2 (Strassen, rank 7) + 2x2x1 (schoolbook, rank 3) = rank 10.
    # For 2x2x1: C[0,2] = a00*b02 + a01*b12, C[1,2] = a10*b02 + a11*b12
    # 3 multiplications:
    # M8 = (a00+a11)*b02
    # M9 = (a01+a10)*b12  -- no
    # M8 = a00*b02
    # M9 = a01*b12
    # M10 = (a10+a11)*b02  -- C[1,2] = M10 + a11*b12, still need a11*b12
    
    # I'll just use 4 for the last column: rank 11.
    # Or better: use a proper rank-10 construction.
    
    # After research, here's a rank-10 decomposition:
    # M1 = (a00+a11)*(b00+b10)
    # M2 = (a00+a11)*(b01+b11)  
    # M3 = (a00+a11)*(b02+b12)
    # M4 = (a10+a11)*b00
    # M5 = (a10+a11)*b01
    # M6 = (a10+a11)*b02
    # M7 = a00*(b10-b00)
    # M8 = a00*(b11-b01)
    # M9 = a00*(b12-b02)
    # M10 = (a01-a11)*b10  -- hmm
    
    # This is getting complex. Let me just use rank 11 (7+4) for now,
    # or better, use a simpler overall strategy.
    
    # Actually, for the 2x4x6 problem, let me use:
    # - One 2x2x2 Strassen block for A[:,0:2] * B[0:2,:]  (but B is 4x6, so this is 2x2 x 2x6)
    # This is getting complicated. Let me use a cleaner approach.
    
    # NEW APPROACH: Use the fact that 2x4x6 can be decomposed as:
    # - 2x2x6 using rank 10 (2x2x3 Strassen x2 for the 6 columns)
    # - 2x2x6 using rank 10
    # Total: 20. But we need to handle the overlap.
    
    # Actually, the cleanest is:
    # A (2x4) = [A_left (2x2) | A_right (2x2)]
    # B (4x6) = [B_top (2x6); B_bottom (2x6)]
    # C = A_left * B_top + A_right * B_bottom
    # Each 2x2x6 can be done in rank 10 (using 2x2x3 Strassen for each pair of columns)
    # Total rank: 20.
    
    # Wait, 2x2x6: split into three 2x2x2 blocks, each rank 7. Total 21.
    # Or use 2x2x3 Strassen (rank 10) twice: 20.
    
    # Let me implement 2x2x3 with rank 10 properly.
    
    # After more thought, here's a valid rank-10 decomposition for 2x2x3:
    # We need C[i,k] = sum_j A[i,j]*B[j,k] for i=0,1 and k=0,1,2
    
    # Use these 10 products:
    # P1 = (a00+a11)*(b00+b10)
    # P2 = (a00+a11)*(b01+b11)
    # P3 = (a00+a11)*(b02+b12)
    # P4 = (a10+a11)*b00
    # P5 = (a10+a11)*b01
    # P6 = (a10+a11)*b02
    # P7 = a00*(b10-b00)
    # P8 = a00*(b11-b01)
    # P9 = a00*(b12-b02)
    # P10 = a01*b10
    
    # C[0,0] = P1 - P7 - P10 = (a00+a11)(b00+b10) - a00(b10-b00) - a01*b10
    #        = a00*b00 + a00*b10 + a11*b00 + a11*b10 - a00*b10 + a00*b00 - a01*b10
    #        = 2*a00*b00 + a11*b00 + a11*b10 - a01*b10
    # That's wrong.
    
    # I'll use a simpler, verified approach: just use rank 7 for 2x2x2 and rank 4 for 2x2x1.
    # 2x2x6 = 3 * (2x2x2) = rank 21, or 2 * (2x2x3) where 2x2x3 = rank 11.
    
    # Let me just go with the straightforward block approach:
    # 2x4x6 = 2 * (2x2x6)
    # 2x2x6 = 3 * (2x2x2) = rank 21 per 2x2x6 block
    # Total: 2 * 21 = 42. Same as before.
    
    # For improvement, let me try a different block structure:
    # 2x4x6: use 2x2x2 Strassen for the first 4 columns (2 blocks of 2x2x2),
    # and handle the last 2 columns differently.
    
    # Actually, let me just implement the rank-42 solution cleanly and then
    # try to find dependencies using SVD on the tensor terms.
    
    # For now, return a placeholder - I'll fix this.
    raise NotImplementedError("Need proper 2x2x3 decomposition")

def _build_2x4x6_rank42():
    """Build the 2x4x6 matrix multiplication decomposition using Strassen on 2x2 blocks. Rank = 42."""
    n, m, p = 2, 4, 6
    n_j_blocks = m // 2  # 2
    n_k_blocks = p // 2  # 3
    rank = n_j_blocks * n_k_blocks * 7  # 42
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    for jb in range(n_j_blocks):
        for kb in range(n_k_blocks):
            base_col = (jb * n_k_blocks + kb) * 7
            j0 = 2 * jb
            j1 = 2 * jb + 1
            k0 = 2 * kb
            k1 = 2 * kb + 1
            
            for c in range(7):
                col = base_col + c
                for i in range(2):
                    for j_local in range(2):
                        j = j0 + j_local
                        U[i * m + j, col] = U_s[i * 2 + j_local, c]
                for j_local in range(2):
                    j = j0 + j_local
                    for k_local in range(2):
                        k = k0 + k_local
                        V[j * p + k, col] = V_s[j_local * 2 + k_local, c]
                for k_local in range(2):
                    k = k0 + k_local
                    for i in range(2):
                        W[k * n + i, col] = W_s[k_local * 2 + i, c]
    
    return U, V, W

def _build_2x4x6_rank40():
    """
    Build a rank-40 decomposition by using Strassen on 2x2 blocks for 5 of the 6
    column-pairs, and a different structure for the 6th.
    
    Actually, let's try: use 2x2x2 Strassen for all 3 column-pairs in the first
    2x2 row-block (rank 21), and for the second 2x2 row-block, use a mixed approach.
    
    Alternative: Use the observation that some Strassen terms might be shared.
    
    For now, let's try a different block decomposition:
    - Block (0,0): 2x2x2 Strassen, rank 7
    - Block (0,1): 2x2x2 Strassen, rank 7  
    - Block (0,2): 2x2x2 Strassen, rank 7
    - Block (1,0): 2x2x2 Strassen, rank 7
    - Block (1,1): 2x2x2 Strassen, rank 7
    - Block (1,2): 2x2x2 Strassen, rank 7
    Total: 42
    
    To get 40, we need to find 2 linear dependencies.
    """
    # For now, build rank 42 and then try to reduce
    return _build_2x4x6_rank42()

def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify the decomposition is exact."""
    n, m, p = 2, 4, 6
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0
    
    recon = np.einsum('ac,bc,dc->abd', U_arr, V_arr, W_arr)
    
    error = np.linalg.norm(target - recon)
    return error < tol, error

def _try_reduce_rank(U, V, W, target_rank, max_iters=50, tol=1e-9):
    """
    Try to reduce the rank by finding linear dependencies among the terms.
    Uses iterative null-space elimination.
    """
    n, m, p = 2, 4, 6
    current_rank = U.shape[1]
    
    if current_rank <= target_rank:
        return U, V, W
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for iteration in range(max_iters):
        if _deadline_reached():
            break
        if U.shape[1] <= target_rank:
            break
        
        # Build the terms matrix: each column is u_r ⊗ v_r ⊗ w_r
        rank = U.shape[1]
        term_dim = (n * m) * (m * p) * (p * n)
        
        # Only compute if feasible
        if term_dim * rank > 10**8:
            break
            
        terms = np.zeros((term_dim, rank))
        for r in range(rank):
            u_r = U[:, r]
            v_r = V[:, r]
            w_r = W[:, r]
            term = np.outer(np.outer(u_r, v_r), w_r).flatten()
            terms[:, r] = term
        
        # SVD to find null space
        try:
            U_svd, s, Vt = np.linalg.svd(terms, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        
        # Find numerical rank
        if s[0] == 0:
            break
        
        significant = s > tol * s[0]
        num_rank = np.sum(significant)
        
        if num_rank >= rank:
            break
        
        # The null space is spanned by the last (rank - num_rank) rows of Vt
        null_dim = rank - num_rank
        
        # Try to eliminate one term at a time
        # Find a null vector and use it to eliminate a term
        for null_idx in range(null_dim):
            if _deadline_reached():
                break
            if U.shape[1] <= target_rank:
                break
                
            c = Vt[num_rank + null_idx, :]  # null vector
            # Find the largest component to eliminate
            max_idx = np.argmax(np.abs(c))
            if abs(c[max_idx]) < tol:
                continue
            
            # We have sum_r c_r * t_r = 0
            # So t_max = -sum_{r != max} (c_r / c_max) * t_r
            # We can eliminate term max_idx by adjusting others.
            # But this changes the tensor! We need:
            # sum_r t_r = sum_{r != max} t_r + t_max
            # = sum_{r != max} t_r - sum_{r != max} (c_r/c_max) t_r
            # = sum_{r != max} (1 - c_r/c_max) t_r
            
            # Wait, that's not right either. The null vector gives:
            # sum_r c_r * t_r = 0
            # This means the t_r are linearly dependent, but it doesn't directly
            # let us eliminate one without changing the sum.
            
            # Actually, we can use the null vector to create a new set of terms.
            # If c_k is the largest, we can write:
            # t_k = -sum_{r != k} (c_r/c_k) t_r
            # Then: sum_r t_r = sum_{r != k} t_r + t_k = sum_{r != k} (1 - c_r/c_k) t_r
            # But (1 - c_r/c_k) t_r is still a rank-1 tensor (scalar multiple).
            # So we can eliminate term k and scale the others.
            
            # However, this only works if we're allowed to change the individual terms.
            # The new terms are (1 - c_r/c_k) * u_r ⊗ v_r ⊗ w_r for r != k.
            # Each is still rank-1.
            
            # Let's do it:
            c_k = c[max_idx]
            new_U = np.zeros((n * m, rank - 1))
            new_V = np.zeros((m * p, rank - 1))
            new_W = np.zeros((p * n, rank - 1))
            
            col = 0
            for r in range(rank):
                if r == max_idx:
                    continue
                scale = 1.0 - c[r] / c_k
                new_U[:, col] = scale * U[:, r]
                new_V[:, col] = scale * V[:, r]
                new_W[:, col] = scale * W[:, r]
                col += 1
            
            U = new_U
            V = new_V
            W = new_W
            
            # Verify
            valid, err = _verify_decomposition(U, V, W, tol)
            if not valid:
                # Roll back
                break
            break  # Restart the loop with the new rank
    
    return U, V, W

def _try_random_rank_search(target_rank, num_trials=100, tol=1e-9):
    """
    Try to find a decomposition of the given rank using random initialization
    and alternating least squares.
    """
    n, m, p = 2, 4, 6
    
    # Build target tensor
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0
    
    target_flat = target.reshape(-1)
    
    for trial in range(num_trials):
        if _deadline_reached():
            break
        
        rng = np.random.RandomState(trial * 1000 + 42)
        
        U = rng.randn(n * m, target_rank) * 0.5
        V = rng.randn(m * p, target_rank) * 0.5
        W = rng.randn(p * n, target_rank) * 0.5
        
        # Alternating least squares
        for iteration in range(20):
            if _deadline_reached():
                break
            
            # Update U
            # For each a: U[a,:] @ M.T = target[a,:,:].flat
            # where M[b,d,c] = V[b,c] * W[d,c]
            M = V[:, None, :] * W[None, :, :]  # (m*p, p*n, rank)
            M_flat = M.reshape(target_rank, -1)  # (rank, m*p*p*n)
            M_T = M_flat.T  # (m*p*p*n, rank)
            
            # Solve U[a,:] = target[a,:,:].flat @ pinv(M_T)
            try:
                M_pinv = np.linalg.pinv(M_T)
            except np.linalg.LinAlgError:
                break
            
            new_U = np.zeros((n * m, target_rank))
            for a in range(n * m):
                y = target[a, :, :].reshape(-1)
                new_U[a, :] = y @ M_pinv
            U = new_U
            
            if _deadline_reached():
                break
            
            # Update V
            Q = U[:, None, :] * W[None, :, :]  # (n*m, p*n, rank)
            Q_flat = Q.reshape(target_rank, -1)  # (rank, n*m*p*n)
            Q_T = Q_flat.T
            try:
                Q_pinv = np.linalg.pinv(Q_T)
            except np.linalg.LinAlgError:
                break
            
            new_V = np.zeros((m * p, target_rank))
            for b in range(m * p):
                y = target[:, b, :].reshape(-1)
                new_V[b, :] = y @ Q_pinv
            V = new_V
            
            if _deadline_reached():
                break
            
            # Update W
            R = U[:, None, :] * V[None, :, :]  # (n*m, m*p, rank)
            R_flat = R.reshape(target_rank, -1)  # (rank, n*m*m*p)
            R_T = R_flat.T
            try:
                R_pinv = np.linalg.pinv(R_T)
            except np.linalg.LinAlgError:
                break
            
            new_W = np.zeros((p * n, target_rank))
            for d in range(p * n):
                y = target[:, :, d].reshape(-1)
                new_W[d, :] = y @ R_pinv
            W = new_W
        
        # Check convergence
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        error = np.linalg.norm(target - recon)
        
        if error < tol:
            return U, V, W
    
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Construct the guaranteed Rank 42 solution
    U42, V42, W42 = _build_2x4x6_rank42()
    
    # 2. Verify Rank 42
    is_valid_42, _ = _verify_decomposition(U42, V42, W42)
    
    best_U, best_V, best_W = U42, V42, W42
    best_rank = 42
    
    if not is_valid_42:
        # Fallback to Schoolbook Rank 48
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
    
    # 3. Try to reduce rank via null-space elimination
    if not _deadline_reached():
        U_red, V_red, W_red = _try_reduce_rank(U42, V42, W42, target_rank=38, max_iters=20)
        if U_red.shape[1] < best_rank:
            is_valid_red, _ = _verify_decomposition(U_red, V_red, W_red)
            if is_valid_red:
                best_U, best_V, best_W = U_red, V_red, W_red
                best_rank = U_red.shape[1]
    
    # 4. Try random search for lower rank
    if not _deadline_reached() and best_rank > 38:
        for target_rank in [best_rank - 1, best_rank - 2, 38, 37, 36]:
            if _deadline_reached():
                break
            if target_rank < 30:
                break
            U_rand, V_rand, W_rand = _try_random_rank_search(target_rank, num_trials=20)
            if U_rand is not None:
                is_valid_rand, _ = _verify_decomposition(U_rand, V_rand, W_rand)
                if is_valid_rand:
                    best_U, best_V, best_W = U_rand, V_rand, W_rand
                    best_rank = target_rank
                    break
    
    # 5. Return best result
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
