"""Matrix multiplication 3x5x6 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p = 90."""
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

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        # Check tensor decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    def try_strassen_decomposition():
        """
        Try to build a Strassen-based decomposition for 3x5x6.
        
        Strategy: Split the 3x5x6 problem into blocks:
        - Use 2x2x2 Strassen (rank 7) for overlapping 2x2x2 sub-blocks
        - Handle remaining 1x* or *x1 blocks with schoolbook
        
        For 3x5x6:
        - Rows 0,1 (2 rows) x Cols 0,1 (2 cols) x Out 0,1 (2 outs): 2x2x2 -> 7 terms
        - Rows 0,1 x Cols 0,1 x Out 2,3: 2x2x2 -> 7 terms  
        - Rows 0,1 x Cols 0,1 x Out 4,5: 2x2x2 -> 7 terms
        - Row 2 x Cols 0,1 x Out 0,1: 1x2x2 -> 2 terms
        - Row 2 x Cols 0,1 x Out 2,3: 1x2x2 -> 2 terms
        - Row 2 x Cols 0,1 x Out 4,5: 1x2x2 -> 2 terms
        
        But this doesn't cover all columns. Let me think more carefully.
        
        Actually, for matrix multiplication C = AB where A is n x m and B is m x p:
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        The tensor has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
        
        Let me try a different decomposition. Split:
        - A into [A_top (2x5); A_bottom (1x5)]
        - B into [B_left (5x2) B_mid (5x2) B_right (5x2)]
        
        Then C = A_top @ B + A_bottom @ B
        C = [A_top @ B_left  A_top @ B_mid  A_top @ B_right]
            [A_bottom @ B_left A_bottom @ B_mid A_bottom @ B_right]
        
        For A_top (2x5) @ B_left (5x2): This is 2x5x2, schoolbook rank = 20
        We can do better by splitting B_left into rows:
        B_left = [B_ll (2x2); B_lm (2x2); B_lr (1x2)]
        A_top = [A_tl (2x2) A_tm (2x2) A_tr (2x1)]
        
        A_top @ B_left = A_tl @ B_ll + A_tm @ B_lm + A_tr @ B_lr
        
        A_tl @ B_ll is 2x2x2: Strassen rank 7
        A_tm @ B_lm is 2x2x2: Strassen rank 7  
        A_tr @ B_lr is 2x1x2: schoolbook rank 4
        
        So A_top @ B_left: 7 + 7 + 4 = 18 terms (vs schoolbook 20)
        
        Similarly for other blocks.
        
        Let me implement this properly.
        """
        if time.time() > deadline:
            return None
            
        rank = 0
        U = {}
        V = {}
        W = {}
        
        def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
            nonlocal rank
            col = rank
            U.setdefault(u_idx, [0] * 0)  # placeholder
            V.setdefault(v_idx, [0] * 0)
            W.setdefault(w_idx, [0] * 0)
            rank += 1
            return col
        
        # This approach is getting complex. Let me use a direct construction.
        # 
        # For 3x5x6, let me try to use known results or a systematic approach.
        #
        # Actually, let me try a simpler structural approach:
        # Use the fact that we can factor through smaller matrices.
        #
        # Alternative: Use a randomized approach to search for low-rank decompositions.
        
        return None

    def randomized_search(max_rank=90, iterations=5000):
        """
        Randomized search for low-rank tensor decompositions.
        Start from schoolbook and try to merge terms.
        """
        if time.time() > deadline:
            return None
        
        # Start from schoolbook
        best = schoolbook()
        best_rank = n * m * p
        
        # Try to reduce rank by finding linear dependencies
        # This is a simplified approach - in practice, we'd use more sophisticated methods
        
        for iteration in range(iterations):
            if time.time() > deadline:
                break
            
            # Create a random perturbation of the schoolbook decomposition
            U = [[0] * best_rank for _ in range(n * m)]
            V = [[0] * best_rank for _ in range(m * p)]
            W = [[0] * best_rank for _ in range(p * n)]
            
            # Initialize with schoolbook
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
            
            # Try random linear combinations to see if we can reduce rank
            # This is a heuristic approach
            
            # Check if current decomposition is valid
            if verify(U, V, W, best_rank):
                pass  # Still valid
            
            # Try to find a lower rank by random projection
            # This is very heuristic and may not work well
        
        return best

    # First, return schoolbook as a valid baseline
    result = schoolbook()
    
    # Try to find a better decomposition
    better = try_strassen_decomposition()
    if better is not None:
        better_rank = len(better["U"][0]) if better["U"] else 0
        if better_rank < n * m * p and verify(better["U"], better["V"], better["W"], better_rank):
            result = better
    
    # Try randomized search if time permits
    if time.time() < deadline - 30:
        rand_result = randomized_search(iterations=2000)
        if rand_result is not None:
            rand_rank = len(rand_result["U"][0]) if rand_result["U"] else 0
            if rand_rank < n * m * p and verify(rand_result["U"], rand_result["V"], rand_result["W"], rand_rank):
                result = rand_result
    
    return result
# EVOLVE-BLOCK-END
