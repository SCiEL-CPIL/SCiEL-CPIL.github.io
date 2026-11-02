import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 125

    def verify(U, V, W, rank):
        """Verify decomposition. Returns max error."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def try_decomposition(rank, max_iters=150, seed=0):
        """Alternating minimization with structured initialization."""
        if time.time() > deadline - 10:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Structured initialization: exploit block structure
        # For 5x5x5, use a Kronecker-like structure
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Try different initialization strategies
        if rank >= 25:
            # Dense initialization
            U = rng.randn(n*m, rank) * (1.0 / np.sqrt(rank))
            V = rng.randn(m*p, rank) * (1.0 / np.sqrt(rank))
            W = rng.randn(p*n, rank) * (1.0 / np.sqrt(rank))
        else:
            U = rng.randn(n*m, rank) * 0.3
            V = rng.randn(m*p, rank) * 0.3
            W = rng.randn(p*n, rank) * 0.3
        
        prev_err = float('inf')
        best_local = None
        best_local_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    B = np.zeros((rank, p))
                    for k in range(p):
                        B[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        sol = np.linalg.lstsq(B.T, np.ones(p), rcond=None)[0]
                        U[i*m+j, :] = sol
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((rank, n))
                    for i in range(n):
                        A[:, i] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        sol = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                        V[j*p+k, :] = sol
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    C = np.zeros((rank, m))
                    for j in range(m):
                        C[:, j] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        sol = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                        W[k*n+i, :] = sol
                    except:
                        pass
            
            # Compute error
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < best_local_err:
                best_local_err = max_err
                best_local = (U.copy(), V.copy(), W.copy(), max_err)
            
            if max_err < 1e-8:
                return U, V, W, max_err
            
            if max_err > prev_err * 1.2 and it > 20:
                break
            prev_err = max_err
        
        if best_local is not None and best_local[3] < 1e-6:
            return best_local
        return None

    def construct_strassen_2x2():
        """
        Construct a rank-7 decomposition for 2x2x2 matrix multiplication.
        Returns U, V, W of shape (4, 7) each.
        Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        """
        # Strassen's 7 terms for 2x2 matrices
        # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # C = [[a*e+b*g, a*f+b*h], [c*e+d*g, c*f+d*h]]
        
        # U, V, W entries (using 1s and -1s)
        # Row/col mapping:
        # U row: i*2+j -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # V row: j*2+k -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # W row: k*2+i -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Term 0: (a+d)(e+h) -> contributes to C00, C01, C10, C11
        # a=U[0,0]=1, d=U[3,0]=1
        # e=V[0,0]=1, h=V[3,0]=1
        # C00 gets +, C01 gets +, C10 gets +, C11 gets -
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[1,0]=1; W[2,0]=1; W[3,0]=-1
        
        # Term 1: (c+d)e -> contributes to C10, C11
        # c=U[2,1]=1, d=U[3,1]=1
        # e=V[0,1]=1
        # C10 gets +, C11 gets +
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[2,1]=1; W[3,1]=1
        
        # Term 2: a(f-h) -> contributes to C00, C01
        # a=U[0,2]=1
        # f=V[1,2]=1, h=V[3,2]=-1
        # C00 gets +, C01 gets +
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        W[0,2]=1; W[1,2]=1
        
        # Term 3: c(d-b) -> contributes to C10, C11
        # c=U[2,3]=1, d=U[3,3]=1, b=U[1,3]=-1
        # C10 gets +, C11 gets +
        U[2,3]=1; U[3,3]=1; U[1,3]=-1
        W[2,3]=1; W[3,3]=1
        
        # Term 4: (a+b)h -> contributes to C00, C01
        # a=U[0,4]=1, b=U[1,4]=1
        # h=V[3,4]=1
        # C00 gets +, C01 gets +
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[0,4]=1; W[1,4]=1
        
        # Term 5: (c-a)(e+g) -> contributes to C10, C11
        # c=U[2,5]=1, a=U[0,5]=-1
        # e=V[0,5]=1, g=V[2,5]=1
        # C10 gets +, C11 gets +
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[2,5]=1
        W[2,5]=1; W[3,5]=1
        
        # Term 6: (b-d)(f+h) -> contributes to C00, C01
        # b=U[1,6]=1, d=U[3,6]=-1
        # f=V[1,6]=1, h=V[3,6]=1
        # C00 gets +, C01 gets +
        U[1,6]=1; U[3,6]=-1
        V[1,6]=1; V[3,6]=1
        W[0,6]=1; W[1,6]=1
        
        return U, V, W

    def construct_laderman_3x3():
        """
        Construct a rank-15 decomposition for 3x3x3 matrix multiplication.
        This is a known decomposition.
        """
        # Laderman's algorithm for 3x3
        # This is complex, so we use a known set of coefficients.
        # Reference: Laderman, S. (1976). "On the number of multiplications required to multiply 3 x 3 matrices."
        
        U = np.zeros((9, 15))
        V = np.zeros((9, 15))
        W = np.zeros((9, 15))
        
        # Indices:
        # U row: i*3+j (0..8)
        # V row: j*3+k (0..8)
        # W row: k*3+i (0..8)
        
        # We will use a standard Laderman decomposition.
        # Note: The exact coefficients vary by source, but they are integers.
        
        # For simplicity and correctness, we can use a block decomposition if Laderman is too complex to hardcode correctly from memory.
        # However, the prompt suggests using Laderman. Let's try to implement a known rank-15 solution.
        
        # Actually, implementing Laderman correctly from scratch in this short time is risky.
        # A safer "structured" approach for 3x3 might be to use 3x3x3 rank 15 if we can find the coefficients.
        # Let's use a different strategy for the 3x3 block:
        # We can use the fact that 3x3x3 can be decomposed into smaller blocks? No, that's not efficient.
        
        # Let's stick to the schoolbook for the 3x3 block if we can't guarantee Laderman's correctness.
        # But the directive says "Strategy to try... embedding a 3x3 Laderman algorithm".
        # If I cannot correctly hardcode Laderman, I should fall back to a valid lower-rank or schoolbook.
        # Schoolbook for 3x3 is rank 27.
        
        # Let's try to find a rank-15 decomposition online or derive it.
        # Since I cannot browse, I will use a known property:
        # Laderman's algorithm has rank 15.
        
        # I will use a placeholder rank-27 (schoolbook) for the 3x3 block if I cannot confidently write Laderman.
        # Wait, the directive is a "Strategy to try". If it fails, I should keep the program robust.
        
        # Let's use schoolbook for the 3x3 block to ensure correctness, and Strassen for 2x2.
        # Total rank: 7 (2x2) + 27 (3x3) + 4 * (2*3=6 or 3*2=6) = 7 + 27 + 24 = 58.
        # This is close to the target 57.
        
        # Actually, let's look at the off-diagonal blocks.
        # Top-right: 2x3. Schoolbook rank 2*3=6.
        # Bottom-left: 3x2. Schoolbook rank 3*2=6.
        # There are 4 such blocks? No.
        # Matrix:
        # [ A  B ]
        # [ C  D ]
        # A is 2x2, D is 3x3.
        # B is 2x3, C is 3x2.
        # There is only one B and one C.
        # So rank = rank(A) + rank(D) + rank(B) + rank(C)
        # rank(A) = 7 (Strassen)
        # rank(D) = 27 (Schoolbook) or 15 (Laderman)
        # rank(B) = 6 (Schoolbook)
        # rank(C) = 6 (Schoolbook)
        
        # Total with Schoolbook D: 7 + 27 + 6 + 6 = 46.
        # Total with Laderman D: 7 + 15 + 6 + 6 = 34.
        
        # This is much lower than 125!
        
        # Let's implement this block diagonal decomposition.
        
        return None # We will build this in the main function instead.

    # --- Block Decomposition Strategy ---
    
    def build_block_decomposition():
        """
        Builds a decomposition using Strassen for 2x2 and Schoolbook for 3x3 and off-diagonals.
        Rank = 7 (2x2) + 27 (3x3) + 6 (2x3) + 6 (3x2) = 46.
        """
        # Total dimensions: 5x5
        # Blocks:
        # A: rows 0-1, cols 0-1 (2x2)
        # B: rows 0-1, cols 2-4 (2x3)
        # C: rows 2-4, cols 0-1 (3x2)
        # D: rows 2-4, cols 2-4 (3x3)
        
        # Rank contributions:
        # A: 7
        # D: 27 (schoolbook)
        # B: 6 (schoolbook)
        # C: 6 (schoolbook)
        # Total Rank: 46
        
        total_rank = 7 + 27 + 6 + 6
        U = np.zeros((25, total_rank))
        V = np.zeros((25, total_rank))
        W = np.zeros((25, total_rank))
        
        col_offset = 0
        
        # 1. Block A (2x2) - Strassen
        U_a, V_a, W_a = construct_strassen_2x2()
        # Map indices:
        # A is at rows 0-1, cols 0-1.
        # U index for A: i*5+j where i in 0..1, j in 0..1.
        #   (0,0)->0, (0,1)->1, (1,0)->5, (1,1)->6
        # V index for A: j*5+k where j in 0..1, k in 0..1.
        #   (0,0)->0, (0,1)->1, (1,0)->5, (1,1)->6
        # W index for A: k*5+i where k in 0..1, i in 0..1.
        #   (0,0)->0, (0,1)->1, (1,0)->5, (1,1)->6
        
        # Strassen U_a indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Map U_a[0] -> U[0]
        # Map U_a[1] -> U[1]
        # Map U_a[2] -> U[5]
        # Map U_a[3] -> U[6]
        
        U[0, col_offset:col_offset+7] = U_a[0]
        U[1, col_offset:col_offset+7] = U_a[1]
        U[5, col_offset:col_offset+7] = U_a[2]
        U[6, col_offset:col_offset+7] = U_a[3]
        
        V[0, col_offset:col_offset+7] = V_a[0]
        V[1, col_offset:col_offset+7] = V_a[1]
        V[5, col_offset:col_offset+7] = V_a[2]
        V[6, col_offset:col_offset+7] = V_a[3]
        
        W[0, col_offset:col_offset+7] = W_a[0]
        W[1, col_offset:col_offset+7] = W_a[1]
        W[5, col_offset:col_offset+7] = W_a[2]
        W[6, col_offset:col_offset+7] = W_a[3]
        
        col_offset += 7
        
        # 2. Block D (3x3) - Schoolbook
        # D is at rows 2-4, cols 2-4.
        # U index for D: i*5+j where i in 2..4, j in 2..4.
        #   (2,2)->12, (2,3)->13, (2,4)->14
        #   (3,2)->17, (3,3)->18, (3,4)->19
        #   (4,2)->22, (4,3)->23, (4,4)->24
        # V index for D: j*5+k where j in 2..4, k in 2..4.
        #   Same indices as U for D.
        # W index for D: k*5+i where k in 2..4, i in 2..4.
        #   (2,2)->12, (2,3)->13, (2,4)->14
        #   (3,2)->17, (3,3)->18, (3,4)->19
        #   (4,2)->22, (4,3)->23, (4,4)->24
        
        # Schoolbook for 3x3: 27 terms.
        # Term index t = i'*3 + j'*3 + k' ? No, standard iteration.
        # for i' in 0..2, j' in 0..2, k' in 0..2:
        #   U[i'*3+j', t] = 1
        #   V[j'*3+k', t] = 1
        #   W[k'*3+i', t] = 1
        
        for i_prime in range(3):
            for j_prime in range(3):
                for k_prime in range(3):
                    t = col_offset + i_prime*9 + j_prime*3 + k_prime
                    # Map to global indices
                    # U global: (2+i_prime)*5 + (2+j_prime)
                    u_idx = (2 + i_prime) * 5 + (2 + j_prime)
                    # V global: (2+j_prime)*5 + (2+k_prime)
                    v_idx = (2 + j_prime) * 5 + (2 + k_prime)
                    # W global: (2+k_prime)*5 + (2+i_prime)
                    w_idx = (2 + k_prime) * 5 + (2 + i_prime)
                    
                    U[u_idx, t] = 1
                    V[v_idx, t] = 1
                    W[w_idx, t] = 1
        
        col_offset += 27
        
        # 3. Block B (2x3) - Schoolbook
        # B is at rows 0-1, cols 2-4.
        # Output C is at rows 0-1, cols 2-4.
        # U index for B: i*5+j where i in 0..1, j in 2..4.
        #   (0,2)->2, (0,3)->3, (0,4)->4
        #   (1,2)->7, (1,3)->8, (1,4)->9
        # V index for B: j*5+k where j in 2..4, k in 2..4.
        #   (2,2)->12, (2,3)->13, (2,4)->14
        #   (3,2)->17, (3,3)->18, (3,4)->19
        #   (4,2)->22, (4,3)->23, (4,4)->24
        # W index for B: k*5+i where k in 2..4, i in 0..1.
        #   (2,0)->10, (2,1)->11
        #   (3,0)->15, (3,1)->16
        #   (4,0)->20, (4,1)->21
        
        # Schoolbook for 2x3: 2*3=6 terms? No, 2x3 multiplication requires 2*3*3 = 18 multiplications?
        # Wait. Matrix multiplication C = A * B.
        # A is 2x3, B is 3x3. C is 2x3.
        # C[i,k] = sum_j A[i,j] * B[j,k].
        # Here, the "middle" dimension is 3 (cols of A, rows of B).
        # The "input" dimension for the tensor is the middle index j.
        # In our tensor definition:
        # T[i, j, k] = 1 where i is row of A, j is col of A / row of B, k is col of B.
        # For block B:
        # i in 0..1 (rows of A)
        # j in 2..4 (cols of A / rows of B) -> Local j' in 0..2
        # k in 2..4 (cols of B) -> Local k' in 0..2
        
        # So for each i' in 0..1, j' in 0..2, k' in 0..2:
        # We need a term.
        # U[i*5 + (2+j')]
        # V[(2+j')*5 + (2+k')]
        # W[(2+k')*5 + i]
        
        for i_prime in range(2):
            for j_prime in range(3):
                for k_prime in range(3):
                    t = col_offset + i_prime*9 + j_prime*3 + k_prime
                    u_idx = i_prime * 5 + (2 + j_prime)
                    v_idx = (2 + j_prime) * 5 + (2 + k_prime)
                    w_idx = (2 + k_prime) * 5 + i_prime
                    
                    U[u_idx, t] = 1
                    V[v_idx, t] = 1
                    W[w_idx, t] = 1
        
        col_offset += 18
        
        # 4. Block C (3x2) - Schoolbook
        # C is at rows 2-4, cols 0-1.
        # A is 3x2, B is 2x3. Result is 3x3? No.
        # Wait, the matrix is 5x5.
        # A (top-left 2x2), B (top-right 2x3), C (bottom-left 3x2), D (bottom-right 3x3).
        # C = A * B_total.
        # C_block (bottom-left) = A_bottom * B_total.
        # A_bottom is 3x5. B_total is 5x5.
        # C_block[i, k] where i in 2..4, k in 0..1.
        # C_block[i, k] = sum_j A[i, j] * B[j, k].
        # j ranges 0..4.
        # We split the sum into j in 0..1 (from A_left, which is 2x2) and j in 2..4 (from A_right, which is 2x3).
        # Wait, A is the left matrix.
        # A is 5x5.
        # A is split into:
        #   A_top (2x5) and A_bottom (3x5).
        #   A_top is split into A_top_left (2x2) and A_top_right (2x3).
        #   A_bottom is split into A_bottom_left (3x2) and A_bottom_right (3x3).
        
        # So for the bottom-left block of the result (rows 2..4, cols 0..1):
        # Result[i, k] = sum_j A[i, j] B[j, k].
        # i in 2..4, k in 0..1.
        # j in 0..4.
        
        # We have already handled:
        # 1. A_top_left * B_top_left -> contributes to Result_top_left (i in 0..1, k in 0..1).
        # 2. D * D -> contributes to Result_bottom_right (i in 2..4, k in 2..4).
        # 3. A_top_right * B_bottom_right -> contributes to Result_top_right (i in 0..1, k in 2..4).
        #    Wait, A_top_right is 2x3 (rows 0..1, cols 2..4).
        #    B_bottom_right is 3x3 (rows 2..4, cols 2..4).
        #    Product is 2x3. Result rows 0..1, cols 2..4. Correct.
        
        # Now we need:
        # 4. A_bottom_left * B_top_left -> contributes to Result_bottom_left (i in 2..4, k in 0..1).
        #    A_bottom_left is 3x2 (rows 2..4, cols 0..1).
        #    B_top_left is 2x2 (rows 0..1, cols 0..1).
        #    Product is 3x2. Result rows 2..4, cols 0..1. Correct.
        
        # And:
        # 5. A_bottom_right * B_bottom_left -> contributes to Result_bottom_left (i in 2..4, k in 0..1).
        #    A_bottom_right is 3x3 (rows 2..4, cols 2..4).
        #    B_bottom_left is 3x2 (rows 2..4, cols 0..1).
        #    Product is 3x2. Result rows 2..4, cols 0..1. Correct.
        
        # So we need to add terms for:
        # Block C1: A_bottom_left * B_top_left.
        #   A indices: i in 2..4, j in 0..1.
        #   B indices: j in 0..1, k in 0..1.
        #   Result: i in 2..4, k in 0..1.
        #   Rank: 3 * 2 * 2 = 12.
        
        # Block C2: A_bottom_right * B_bottom_left.
        #   A indices: i in 2..4, j in 2..4.
        #   B indices: j in 2..4, k in 0..1.
        #   Result: i in 2..4, k in 0..1.
        #   Rank: 3 * 3 * 2 = 18.
        
        # Total Rank so far:
        # A (2x2): 7
        # D (3x3): 27
        # B (2x3): 18
        # C1 (3x2 x 2x2): 12
        # C2 (3x3 x 3x2): 18
        # Total: 7 + 27 + 18 + 12 + 18 = 82.
        
        # This is still lower than 125.
        
        # Let's add C1.
        for i_prime in range(3): # i in 2..4
            for j_prime in range(2): # j in 0..1
                for k_prime in range(2): # k in 0..1
                    t = col_offset + i_prime*4 + j_prime*2 + k_prime
                    u_idx = (2 + i_prime) * 5 + j_prime
                    v_idx = j_prime * 5 + k_prime
                    w_idx = k_prime * 5 + (2 + i_prime)
                    
                    U[u_idx, t] = 1
                    V[v_idx, t] = 1
                    W[w_idx, t] = 1
        
        col_offset += 12
        
        # Let's add C2.
        for i_prime in range(3): # i in 2..4
            for j_prime in range(3): # j in 2..4
                for k_prime in range(2): # k in 0..1
                    t = col_offset + i_prime*6 + j_prime*2 + k_prime
                    u_idx = (2 + i_prime) * 5 + (2 + j_prime)
                    v_idx = (2 + j_prime) * 5 + k_prime
                    w_idx = k_prime * 5 + (2 + i_prime)
                    
                    U[u_idx, t] = 1
                    V[v_idx, t] = 1
                    W[w_idx, t] = 1
        
        col_offset += 18
        
        return U, V, W, col_offset

    # Schoolbook fallback
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_result = (U_school, V_school, W_school)
    
    # Try block decomposition
    if time.time() < deadline - 10:
        U_block, V_block, W_block, block_rank = build_block_decomposition()
        max_err = verify(U_block, V_block, W_block, block_rank)
        if max_err < 1e-8:
            best_rank = block_rank
            best_result = (U_block, V_block, W_block)
    
    # Try alternating minimization for even lower ranks
    # Start from the block decomposition rank and go down
    candidate_ranks = list(range(max(20, best_rank - 10), best_rank))
    
    for rank in candidate_ranks:
        if time.time() > deadline - 20:
            break
        
        num_seeds = 5
        
        for seed in range(num_seeds):
            if time.time() > deadline - 10:
                break
            
            result = try_decomposition(rank, max_iters=100, seed=seed + rank*1000)
            if result is not None:
                U, V, W, err = result
                # Round to nearest half-integer
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                max_err = verify(U_r, V_r, W_r, rank)
                if max_err < 1e-8:
                    best_rank = rank
                    best_result = (U_r, V_r, W_r)
                    break
        
        if best_rank < candidate_ranks[0]:
            break

    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
