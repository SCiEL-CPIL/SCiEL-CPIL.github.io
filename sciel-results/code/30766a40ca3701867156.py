import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    # 1. Build the full multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # 2. Helper function to verify a decomposition
    def verify(U, V, W, tol=1e-9):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Compute tensor: sum_r U[:,r] V[:,r]^T W[:,r]^T
        # This is equivalent to np.einsum('ir,js,kr->ijs', U, V, W)
        # Or more efficiently: reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        # Then C_ik = sum_j sum_r U_ir V_jr W_kr ... wait.
        # The tensor T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
        # where a = i*m+j, b = j*p+k, c = k*n+i
        # Let's just use einsum for clarity and correctness check
        try:
            U_arr = np.array(U)
            V_arr = np.array(V)
            W_arr = np.array(W)
            rank = U_arr.shape[1]
            # T_calc = sum_{r=0}^{rank-1} U[:,r] * V[:,r]^T * W[:,r]^T
            # This is a 3rd order tensor.
            # We can compute this by iterating r or using einsum.
            # einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
            T_calc = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
            return np.allclose(T_calc, T, atol=tol)
        except:
            return False

    # 3. Helper to construct Schoolbook decomposition
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank

    # 4. Attempt to find a lower rank decomposition using Random Search + Greedy Projection
    # Strategy:
    # Start with rank R (e.g., 70) and try to reduce?
    # Or start with a small rank and add vectors?
    # Given the difficulty of exact integer decomposition, a robust approach is:
    # Try to find a decomposition of rank R for R in [23, 72] for the 3x3x3 block,
    # then combine.
    
    # However, finding an exact integer decomposition of 3x3x3 rank 23 is hard to do from scratch quickly.
    # Let's try a different approach: 
    # We know the rank of 3x3x3 is 23. 
    # We can try to construct a decomposition of rank 23 for 3x3x3 using a known structure or search.
    # If we fail, we fall back to schoolbook.
    
    # Let's try to find a low rank decomposition for the full 3x3x8 tensor directly?
    # Rank bound: R(3,3,8) <= 2 * R(3,3,4) <= 2 * (3*3*4) = 72.
    # Actually, R(3,3,4) is likely lower than 36.
    # R(3,3,3) = 23.
    # R(3,3,8) = R(3,3,3) + R(3,3,3) + R(3,3,2) = 23 + 23 + 18 = 64 (if we can get 18 for 3x3x2).
    # Schoolbook for 3x3x2 is 18. Can we do better?
    # R(3,3,2) >= 3*2 = 6? No, lower bound is max(n*m, m*p, p*n) = max(9, 6, 6) = 9.
    # Actually, R(3,3,2) is known to be 15? Or 16?
    # Let's check: R(2,2,2)=7. R(3,3,3)=23.
    # If R(3,3,2) is 15, total = 23+23+15 = 61.
    # If R(3,3,2) is 16, total = 62.
    # If R(3,3,2) is 18 (schoolbook), total = 64.
    
    # Let's try to find a decomposition for 3x3x2 with rank < 18.
    # And for 3x3x3 with rank < 27.
    
    # We will implement a simple random search for a decomposition of a sub-tensor.
    # For a tensor T_sub of shape (A, B, C), we want U_sub (A x R), V_sub (B x R), W_sub (C x R).
    
    def search_decomposition(T_sub, target_rank, max_iters=5000, seed=0):
        """
        Attempts to find a decomposition of T_sub with rank <= target_rank.
        Returns (U, V, W) or None.
        """
        A, B, C = T_sub.shape
        np.random.seed(seed)
        
        # Initialize with random vectors
        # We use a greedy approach:
        # 1. Start with R=1, add vectors one by one.
        # 2. Or start with R=target_rank and try to zero out columns?
        
        # Let's try a simple iterative refinement on a random initialization of rank target_rank.
        # This is unlikely to converge to exact integers quickly.
        
        # Alternative: Use the fact that we can combine known decompositions.
        # Since we don't have a reliable fast search for exact integers, 
        # we will rely on the block structure and hardcoded known decompositions if possible, 
        # or just return schoolbook if search fails.
        
        # Given the time constraint and the difficulty of finding exact integer decompositions 
        # via random search in Python, we will stick to the block decomposition with 
        # Schoolbook for the 3x3x3 blocks unless we can hardcode the 23-rank decomposition.
        
        # Let's try to hardcode the 23-rank decomposition for 3x3x3.
        # I will use a set of vectors from a known source.
        # Reference: "The rank of 3x3x3 matrix multiplication is 23"
        # One such decomposition is available in literature.
        
        # If I cannot hardcode it correctly, I will return None.
        return None

    # Since search_decomposition is not implemented effectively, 
    # we will use the block decomposition with Schoolbook for 3x3x3 (Rank 27) 
    # and Schoolbook for 3x3x2 (Rank 18).
    # Total Rank = 27 + 27 + 18 = 72.
    
    # Wait, the current program already does this and scores 1.0.
    # To improve, I MUST find a rank < 72.
    # The only way is to use a rank < 27 for 3x3x3 or rank < 18 for 3x3x2.
    
    # Strategy: Use block decomposition with improved ranks for sub-blocks
    # R(3,3,8) = R(3,3,3) + R(3,3,3) + R(3,3,2)
    # Schoolbook: 27 + 27 + 18 = 72
    # Try to find better decompositions for 3x3x3 and 3x3x2
    
    # Build 3x3x3 tensor
    T_333 = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                T_333[i*3+j, j*3+k, k*3+i] = 1.0
    
    # Build 3x3x2 tensor
    T_332 = np.zeros((9, 6, 6))
    for i in range(3):
        for j in range(3):
            for k in range(2):
                T_332[i*3+j, j*2+k, k*3+i] = 1.0
    
    # Function to verify a decomposition
    def verify_decomp(U, V, W, T, tol=1e-9):
        try:
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            return np.allclose(T_calc, T, atol=tol)
        except:
            return False
    
    # Function to find low-rank decomposition using iterative refinement
    def find_low_rank_decomp(T, target_rank, max_iters=200, seed=42):
        """
        Find a decomposition of T with rank close to target_rank.
        Uses gradient-free iterative refinement.
        """
        A, B, C = T.shape
        np.random.seed(seed)
        
        # Initialize with random vectors
        U = np.random.randn(A, target_rank)
        V = np.random.randn(B, target_rank)
        W = np.random.randn(C, target_rank)
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_loss = np.inf
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Compute current tensor
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            loss = np.sum((T_calc - T) ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Check if we've found a good solution
            if loss < 1e-6:
                break
            
            # Update each matrix in turn (coordinate descent)
            # Update U: minimize ||sum_r U[:,r] V[:,r]^T W[:,r]^T - T||^2
            # For each r, the optimal U[:,r] given V,W is:
            # U[:,r] = (T * V[:,r] * W[:,r]) / (||V[:,r]||^2 * ||W[:,r]||^2)
            # But this is approximate. Let's use a simpler approach.
            
            # Alternative: Use alternating least squares style updates
            # For U: T = U @ (V * W)^T where * is element-wise
            # Reshape T to (A, B*C)
            T_2d = T.reshape(A, B * C)
            
            # For each r, we want to find u_r such that:
            # sum_r u_r v_r^T w_r^T ≈ T
            # This is equivalent to: sum_r (u_r ⊗ v_r ⊗ w_r) ≈ vec(T)
            
            # Let's try a different approach: iterative proportional fitting
            # Or just use random restarts with local optimization
            
            # Simple approach: random perturbations
            for r in range(target_rank):
                # Perturb U[:, r]
                U[:, r] += np.random.randn(A) * 0.1
                # Perturb V[:, r]
                V[:, r] += np.random.randn(B) * 0.1
                # Perturb W[:, r]
                W[:, r] += np.random.randn(C) * 0.1
        
        return best_U, best_V, best_W, best_loss
    
    # Try to find rank 23 for 3x3x3
    rank_333_found = 27
    U_333, V_333, W_333 = None, None, None
    
    for target_rank in [23, 24, 25, 26]:
        if time.time() > deadline - 30:
            break
        U_try, V_try, W_try, loss = find_low_rank_decomp(T_333, target_rank, max_iters=50)
        if loss < 1e-6:
            rank_333_found = target_rank
            U_333, V_333, W_333 = U_try, V_try, W_try
            break
    
    # Try to find rank 15 or 16 for 3x3x2
    rank_332_found = 18
    U_332, V_332, W_332 = None, None, None
    
    for target_rank in [14, 15, 16, 17]:
        if time.time() > deadline - 20:
            break
        U_try, V_try, W_try, loss = find_low_rank_decomp(T_332, target_rank, max_iters=50)
        if loss < 1e-6:
            rank_332_found = target_rank
            U_332, V_332, W_332 = U_try, V_try, W_try
            break
    
    # If we didn't find better decompositions, use schoolbook
    if U_333 is None:
        U_333, V_333, W_333, _ = schoolbook(3, 3, 3)
        rank_333_found = 27
    if U_332 is None:
        U_332, V_332, W_332, _ = schoolbook(3, 3, 2)
        rank_332_found = 18
    
    # Combine the decompositions for 3x3x8
    total_rank = 2 * rank_333_found + rank_332_found
    
    # Build full U, V, W matrices
    U_full = np.zeros((9, total_rank))
    V_full = np.zeros((24, total_rank))
    W_full = np.zeros((24, total_rank))
    
    col = 0
    # First 3x3x3 block (k=0,1,2)
    U_full[:, col:col+rank_333_found] = U_333
    V_full[:9, col:col+rank_333_found] = V_333
    W_full[:9, col:col+rank_333_found] = W_333
    col += rank_333_found
    
    # Second 3x3x3 block (k=3,4,5)
    U_full[:, col:col+rank_333_found] = U_333
    V_full[9:18, col:col+rank_333_found] = V_333
    W_full[9:18, col:col+rank_333_found] = W_333
    col += rank_333_found
    
    # Third block: 3x3x2 (k=6,7)
    U_full[:, col:col+rank_332_found] = U_332
    V_full[18:24, col:col+rank_332_found] = V_332
    W_full[18:24, col:col+rank_332_found] = W_332
    col += rank_332_found
    
    # Verify the full decomposition
    if not verify_decomp(U_full, V_full, W_full, T):
        # Fall back to schoolbook
        U, V, W, rank = schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    # Convert to lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
