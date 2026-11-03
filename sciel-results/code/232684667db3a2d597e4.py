import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill

    def get_schoolbook():
        rank = n * m * p  # 30
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

    def validate(U, V, W, num_tests=10):
        rank = len(U[0])
        U_arr = np.array(U, dtype=float).reshape(n, m, rank)
        V_arr = np.array(V, dtype=float).reshape(m, p, rank)
        W_arr = np.array(W, dtype=float).reshape(p, n, rank)
        
        np.random.seed(42)
        for _ in range(num_tests):
            A = np.random.rand(n, m)
            B = np.random.rand(m, p)
            C_true = A @ B
            
            C_comp = np.zeros((n, p))
            for r in range(rank):
                u_val = np.sum(U_arr[:, :, r] * A)
                v_val = np.sum(V_arr[:, :, r] * B)
                w_mat = W_arr[:, :, r].T  # W is (p, n, r), transpose to get (n, p)
                C_comp += u_val * v_val * w_mat
                
            if not np.allclose(C_true, C_comp, atol=1e-9):
                return False
        return True

    def get_strassen_rank28():
        """
        Rank-28 decomposition using Strassen's trick on the 2x2 block structure.
        
        Split A (2x3) into [A1 | A2] where A1 is 2x1 (first col) and A2 is 2x2 (last 2 cols).
        Split B (3x5) into [B1; B2] where B1 is 1x5 (first row) and B2 is 2x5 (last 2 rows).
        
        C = A1 * B1 + A2 * B2
        
        A2 * B2 uses Strassen: 7 products instead of 4.
        A1 * B1 uses standard: 2 products.
        Total: 7 + 2 = 9... wait, that's not 28.
        
        Actually, let me think more carefully.
        
        A is 2x3, B is 3x5.
        C = A @ B is 2x5.
        
        Standard: 2*3*5 = 30 products.
        
        Strassen approach:
        Split A into [A1 | A2] where A1 is 2x1, A2 is 2x2.
        Split B into [B1; B2] where B1 is 1x5, B2 is 2x5.
        
        C = A1 @ B1 + A2 @ B2
        
        A1 @ B1: (2x1) @ (1x5) -> 2x5. This needs 2 products (one for each row of A1).
        A2 @ B2: (2x2) @ (2x5) -> 2x5. 
        
        For A2 @ B2, we can use a variant of Strassen.
        A2 is 2x2, B2 is 2x5.
        
        Let A2 = [[a11, a12], [a21, a22]]
        Let B2 = [[b11, b12, b13, b14, b15], [b21, b22, b23, b24, b25]]
        
        C_top = a11*B2[0,:] + a12*B2[1,:]
        C_bot = a21*B2[0,:] + a22*B2[1,:]
        
        Standard: 4 * 5 = 20 products for the 2x2 @ 2x5.
        Plus 2 * 5 = 10 products for A1 @ B1.
        Total: 30.
        
        To use Strassen on the 2x2 part:
        M1 = (a21-a11) * (b11+b21)
        M2 = a21 * b21
        M3 = (a11+a12) * b21  -- wait, this doesn't quite work for non-square B.
        
        Actually, for 2x2 @ 2x5, we can use a rank-7 decomposition of the 2x2x2 tensor
        and extend to 5 columns.
        
        Let me use the standard Strassen for 2x2 @ 2x2 and then generalize.
        
        For 2x2 @ 2x2:
        M1 = (a11+a22)(b11+b22)
        M2 = (a21+a22)b11
        M3 = a11(b21-b22)
        M4 = a22(b12-b11)
        M5 = (a11+a12)b22
        M6 = (a21-a11)(b11+b12)
        M7 = (a12-a22)(b21+b22)
        
        c11 = M1+M4-M5+M7
        c12 = M3+M5
        c21 = M2+M4
        c22 = M1-M2+M3+M6
        
        For 2x2 @ 2x5, we have 5 columns. We can apply the same 7 products to each pair of columns?
        No, that would give 7 * 3 = 21 or something.
        
        Actually, a simpler approach: use the fact that the matrix multiplication tensor M_{2,2,5}
        has a known minimal rank. But I don't know it off the top of my head.
        
        Let me try a different decomposition.
        
        Actually, the inspiration says rank 28 was achieved. Let me try to construct it directly.
        
        One approach: use the schoolbook for most terms and use Strassen-like tricks to combine some.
        
        Let me try:
        A = [[a1, a2, a3], [b1, b2, b3]]
        B = [[c1, c2, c3, c4, c5],
             [d1, d2, d3, d4, d5],
             [e1, e2, e3, e4, e5]]
        
        C = [[a1*c1 + a2*c2 + a3*c3, a1*c2 + a2*c3 + a3*c4, a1*c3 + a2*c4 + a3*c5, a1*c4 + a2*c5 + a3*0, a1*c5 + a2*0 + a3*0],
             [b1*c1 + b2*c2 + b3*c3, b1*c2 + b2*c3 + b3*c4, b1*c3 + b2*c4 + b3*c5, b1*c4 + b2*c5 + b3*0, b1*c5 + b2*0 + b3*0]]
        
        Wait, that's not right. Let me redo.
        
        A is 2x3, B is 3x5.
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        C[0,0] = a1*c1 + a2*d1 + a3*e1
        C[0,1] = a1*c2 + a2*d2 + a3*e2
        C[0,2] = a1*c3 + a2*d3 + a3*e3
        C[0,3] = a1*c4 + a2*d4 + a3*e4
        C[0,4] = a1*c5 + a2*d5 + a3*e5
        C[1,0] = b1*c1 + b2*d1 + b3*e1
        C[1,1] = b1*c2 + b2*d2 + b3*e2
        C[1,2] = b1*c3 + b2*d3 + b3*e3
        C[1,3] = b1*c4 + b2*d4 + b3*e4
        C[1,4] = b1*c5 + b2*d5 + b3*e5
        
        That's 10 entries, each with 3 terms = 30 terms total.
        
        To reduce to 28, we need to save 2 terms.
        
        Let me try using the Strassen trick on the middle part.
        
        Actually, let me just try to find a rank-28 decomposition by using known results.
        
        For M_{2,3,5}, I believe the rank is at most 28. Let me try to construct it.
        
        One known approach: use the fact that M_{2,3,5} can be decomposed using a combination of
        schoolbook and some clever combinations.
        
        Let me try a specific construction:
        
        Use 28 bilinear forms. I'll try to define them.
        
        Actually, let me just return the schoolbook for now and ensure the code is correct,
        then try to improve.
        """
        # Let me try a specific rank-28 construction
        rank = 28
        U = [[0] * rank for _ in range(n * m)]  # 6 x 28
        V = [[0] * rank for _ in range(m * p)]  # 15 x 28
        W = [[0] * rank for _ in range(p * n)]  # 10 x 28
        
        # A indices: (i,j) -> i*3+j, i in [0,1], j in [0,1,2]
        # B indices: (j,k) -> j*5+k, j in [0,1,2], k in [0,1,2,3,4]
        # W indices: (k,i) -> k*2+i, k in [0,1,2,3,4], i in [0,1]
        
        # I'll use a known rank-28 decomposition pattern
        # This is based on combining Strassen-like tricks
        
        # Let me try to use the following approach:
        # Split into overlapping 2x2 subproblems
        
        # Term 1-7: Strassen on A[0:2,0:2] @ B[0:2,0:2] contribution
        # But this is getting complicated. Let me just verify the schoolbook works and return it.
        
        # Actually, let me try a simpler rank reduction.
        # Use the identity: (x+y)(x-y) = x^2 - y^2 type tricks.
        
        # For now, return schoolbook. The validation will confirm it's correct.
        return None  # Signal to use schoolbook

    def get_strassen_block():
        """
        Construct a rank-28 decomposition using block Strassen.
        
        Split A (2x3) as [A_left | A_right] where A_left is 2x1, A_right is 2x2.
        Split B (3x5) as [B_top; B_bot] where B_top is 1x5, B_bot is 2x5.
        
        C = A_left @ B_top + A_right @ B_bot
        
        A_left @ B_top: 2 terms (each row of A_left times the single row of B_top)
        A_right @ B_bot: use a rank-7 decomposition of the 2x2x5 tensor.
        
        For 2x2 @ 2x5, we can use 7 products per column pair, but we have 5 columns.
        Actually, the tensor M_{2,2,5} can be decomposed with rank 7*? 
        No, we need to be more careful.
        
        Let me use a direct construction.
        
        A_right = [[a1, a2], [a3, a4]]
        B_bot = [[b1, b2, b3, b4, b5], [c1, c2, c3, c4, c5]]
        
        Result:
        r1 = a1*B_bot[0,:] + a2*B_bot[1,:]
        r2 = a3*B_bot[0,:] + a4*B_bot[1,:]
        
        Using Strassen-like:
        M1 = (a3+a4) * (B_bot[0,:]+B_bot[1,:])
        M2 = a3 * B_bot[1,:]
        M3 = a1 * (B_bot[0,:]-B_bot[1,:])  -- wait, this is element-wise, not standard.
        
        Actually, for matrix multiplication, the Strassen products are:
        M1 = (a3+a4) @ (B_bot[0,:] + B_bot[1,:])  -- but these are row vectors, so this is a dot product per column.
        
        Let me just use the standard 4 products for 2x2 @ 2x5, giving 20 products.
        Plus 2 for A_left @ B_top = 22.
        That's 22, not 28. Hmm.
        
        Wait, A_left @ B_top: A_left is 2x1, B_top is 1x5.
        Result is 2x5. Each entry is a product of one A element and one B element.
        So 2*5 = 10 products for A_left @ B_top.
        Plus 20 for A_right @ B_bot = 30. That's schoolbook.
        
        To reduce, we need to combine terms across the boundary.
        
        Let me try a different split.
        
        Actually, let me just implement the schoolbook correctly and return it.
        """
        return None

    # Try the Strassen rank-28 approach
    result = get_strassen_rank28()
    if result is not None:
        if validate(result['U'], result['V'], result['W']):
            return result
    
    # Fall back to schoolbook
    return get_schoolbook()
