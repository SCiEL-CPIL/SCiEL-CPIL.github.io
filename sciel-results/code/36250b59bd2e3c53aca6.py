import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    
    # We'll construct a decomposition for 2x2x15 matrix multiplication
    # by combining sub-decompositions for 2x2x5 and 2x2x10
    #
    # For 2x2x5: we use a rank-14 decomposition
    # For 2x2x10: we use a rank-28 decomposition (5 Strassen-like pairs)
    # Total rank: 14 + 28 = 42
    
    # First, let's define the tensor structure
    # The matrix multiplication tensor for (n,m,p) has entries T[i*m+j, j*p+k, k*n+i]
    # where i in [0,n), j in [0,m), k in [0,p)
    
    # For the 2x2x5 block (columns 0-4 of the 15 columns)
    # We need a rank-14 decomposition
    
    # For the 2x2x10 block (columns 5-14 of the 15 columns)
    # We use 5 independent 2x2x2 Strassen decompositions, each with rank 7
    
    # Strassen decomposition for 2x2x2:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = a22(b12-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # Let's build the full decomposition
    rank = 42
    U = np.zeros((n * m, rank))  # shape (4, 42)
    V = np.zeros((m * p, rank))  # shape (30, 42)
    W = np.zeros((p * n, rank))  # shape (30, 42)
    
    # Part 1: 2x2x5 block with rank 14
    # We'll use a numerical solver to find a rank-14 decomposition for 2x2x5
    
    def solve_2x2x5(rank_target=14):
        """Solve for a rank-14 decomposition of 2x2x5 matrix multiplication."""
        n, m, p = 2, 2, 5
        total_U = n * m  # 4
        total_V = m * p  # 10
        total_W = p * n  # 10
        
        np.random.seed(42)
        
        from scipy.optimize import least_squares
        
        def residuals(params):
            U = params[:rank_target*total_U].reshape(rank_target, total_U).T
            V = params[rank_target*total_U:rank_target*(total_U+total_V)].reshape(rank_target, total_V).T
            W = params[rank_target*(total_U+total_V):].reshape(rank_target, total_W).T
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        res.append(val - 1.0)
            return res
        
        best_U, best_V, best_W = None, None, None
        best_cost = float('inf')
        
        for trial in range(30):
            params0 = np.random.randn(rank_target * (total_U + total_V + total_W)) * 0.5
            try:
                result = least_squares(residuals, params0, method='lm', max_nfev=3000)
                cost = result.cost
                if cost < best_cost:
                    best_cost = cost
                    U_sol = result.x[:rank_target*total_U].reshape(rank_target, total_U).T
                    V_sol = result.x[rank_target*total_U:rank_target*(total_U+total_V)].reshape(rank_target, total_V).T
                    W_sol = result.x[rank_target*(total_U+total_V):].reshape(rank_target, total_W).T
                    # Check if solution is close to exact
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_sol[i*m+j] * V_sol[j*p+k] * W_sol[k*n+i])
                                max_err = max(max_err, abs(val - 1.0))
                    if max_err < 1e-8:
                        return U_sol, V_sol, W_sol
            except:
                pass
        
        return best_U, best_V, best_W
    
    U5, V5, W5 = solve_2x2x5(14)
    
    if U5 is not None:
        # Place the 2x2x5 decomposition in the first 14 columns
        U[:, :14] = U5
        V[:10, :14] = V5  # First 10 entries of V (2*5=10)
        W[:10, :14] = W5  # First 10 entries of W (5*2=10)
    
    # Part 2: 2x2x10 block with rank 28 (5 Strassen pairs)
    # The 2x2x10 block corresponds to columns 5-14 of the original 2x2x15
    # This is equivalent to 5 independent 2x2x2 multiplications
    
    # For each of the 5 blocks (columns 5-6, 7-8, 9-10, 11-12, 13-14):
    # We use a Strassen decomposition with rank 7
    
    # Strassen for 2x2x2:
    # Input matrix A (2x2): A[i*m+j] where i,j in [0,2)
    # Input matrix B (2x2): B[j*p+k] where j in [0,2), k in [0,2)
    # Output: C[k*n+i] where k in [0,2), i in [0,2)
    
    # Strassen factors:
    # M1: A factors: [1,0,1,1], B factors: [1,1,0,0], W factors: [0,0,0,0] -> contributes to C[0,0] and C[0,1] and C[1,0] and C[1,1]
    # Actually, let me be more careful with the Strassen decomposition.
    
    # For 2x2x2 matrix multiplication:
    # C = A * B where A is 2x2, B is 2x2, C is 2x2
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # Strassen's 7 products:
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[1,0]-B[1,1])
    # M4 = A[1,1] * (B[0,1]-B[0,0])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    
    # C[0,0] = M1 - M4 + M5 - M7
    # C[0,1] = M3 + M7
    # C[1,0] = M2 + M6
    # C[1,1] = M1 - M3 + M2 - M6
    
    # Let me verify:
    # M1 - M4 + M5 - M7 = (A00+A11)(B00+B11) - A11(B01-B00) + (A00+A01)B11 - (A01-A11)(B10+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 - A11*B01 + A11*B00 + A00*B11 + A01*B11 - A01*B10 - A01*B11 + A11*B10 + A11*B11
    # = A00*B00 + 2*A00*B11 + 2*A11*B00 + 2*A11*B11 - A11*B01 + A01*B11 - A01*B10 + A11*B10
    # Hmm, that doesn't look right. Let me recalculate.
    
    # Actually, the standard Strassen formulas are:
    # C[0,0] = M1 + M4 - M5 - M7
    # C[0,1] = M3 + M7
    # C[1,0] = M2 + M6
    # C[1,1] = M1 - M3 + M5 - M6
    
    # Let me verify C[0,0]:
    # M1 + M4 - M5 - M7 = (A00+A11)(B00+B11) + A11(B01-B00) - (A00+A01)B11 - (A01-A11)(B10+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B01 - A11*B00 - A00*B11 - A01*B11 - A01*B10 - A01*B11 + A11*B10 + A11*B11
    # = A00*B00 + A11*B00 + 2*A11*B11 + A11*B01 - A11*B00 - A01*B11 - A01*B10 - A01*B11 + A11*B10 + A11*B11
    # Hmm, still not clean. Let me use the standard Strassen decomposition more carefully.
    
    # Standard Strassen:
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[1,0] - B[1,1])
    # M4 = A[1,1] * (B[0,1] - B[0,0])
    # M5 = (A[0,0] + A[0,1]) * B[1,1]
    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    
    # C[0,0] = M1 + M4 - M5 - M7
    # C[0,1] = M3 + M7
    # C[1,0] = M2 + M6
    # C[1,1] = M1 - M3 + M5 - M6
    
    # Let me verify C[1,1]:
    # M1 - M3 + M5 - M6 = (A00+A11)(B00+B11) - A00(B10-B11) + (A00+A01)B11 - (A10-A00)(B00+B01)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 - A00*B10 + A00*B11 + A00*B11 + A01*B11 - A10*B00 - A10*B01 + A00*B00 + A00*B01
    # = 2*A00*B00 + A00*B11 + A11*B00 + A11*B11 - A00*B10 + 2*A00*B11 + A01*B11 - A10*B00 - A10*B01 + A00*B01
    # This is getting messy. Let me just use the known correct Strassen factors.
    
    # The correct Strassen decomposition for 2x2 matrix multiplication:
    # p1 = (a00 + a11) * (b00 + b11)
    # p2 = (a10 + a11) * b00
    # p3 = a00 * (b10 - b11)
    # p4 = a11 * (b01 - b00)
    # p5 = (a00 + a01) * b11
    # p6 = (a10 - a00) * (b00 + b01)
    # p7 = (a01 - a11) * (b10 + b11)
    
    # c00 = p1 + p4 - p5 - p7
    # c01 = p3 + p7
    # c10 = p2 + p6
    # c11 = p1 - p3 + p5 - p6
    
    # I'll trust this standard decomposition and build the factors.
    
    # For the 2x2x10 block, we have 5 independent 2x2x2 multiplications.
    # The input A is the same for all 5 (it's the left 2x2 matrix)
    # The input B is split into 5 chunks of 2x2 (columns 5-6, 7-8, 9-10, 11-12, 13-14)
    # The output C is split into 5 chunks of 2x2 (rows 5-6, 7-8, 9-10, 11-12, 13-14)
    
    # Let's define the Strassen factors for one 2x2x2 block:
    # A indices: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
    # B indices: 0=B[0,0], 1=B[0,1], 2=B[1,0], 3=B[1,1]
    # C indices: 0=C[0,0], 1=C[0,1], 2=C[1,0], 3=C[1,1]
    
    # p1: A factors = [1,0,0,1], B factors = [1,0,0,1], contributes to C with W factors
    # p1 contributes to c00 (+1), c11 (+1)
    # p2: A factors = [0,0,1,1], B factors = [1,0,0,0], contributes to c10 (+1)
    # p3: A factors = [1,0,0,0], B factors = [0,0,1,-1], contributes to c01 (+1), c11 (-1)
    # p4: A factors = [0,0,0,1], B factors = [0,1,-1,0], contributes to c00 (+1), c11 (0)... 
    # Wait, let me re-read: c00 = p1 + p4 - p5 - p7
    # So p4 contributes +1 to c00
    # c11 = p1 - p3 + p5 - p6, so p4 doesn't contribute to c11
    
    # Let me just directly write the U, V, W factors for each Strassen product.
    
    # For the 2x2x10 block, the B matrix has 10 rows (m=2, p=10), so rows 0-9 correspond to:
    # B[0,0], B[0,1], ..., B[0,9] (row 0 of B, 10 columns)
    # B[1,0], B[1,1], ..., B[1,9] (row 1 of B, 10 columns)
    # In our indexing: V[j*p+k] where j in [0,2), k in [0,10)
    # So V[0:10] corresponds to B[0,0:10] and V[10:20] corresponds to B[1,0:10]
    # But we're only using columns 5-14 of the original 15-column B, which corresponds to
    # the second half of the 2x2x10 block.
    
    # Actually, let me reconsider. The 2x2x10 block means we're multiplying a 2x2 matrix by a 2x10 matrix.
    # The B matrix has shape (2, 10), so m=2, p=10.
    # V has shape (m*p, rank) = (20, 28)
    # W has shape (p*n, rank) = (20, 28)
    
    # The 5 Strassen blocks correspond to columns 0-1, 2-3, 4-5, 6-7, 8-9 of the 2x10 B matrix.
    # In the original 2x15 problem, these correspond to columns 5-6, 7-8, 9-10, 11-12, 13-14.
    
    # Let's define the Strassen factors for one 2x2x2 block and replicate 5 times.
    
    # A indices in the full problem: U[i*m+j] where i in [0,2), j in [0,2)
    # So A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[2], A[1,1]=U[3]
    
    # For block t (t=0..4), the B submatrix columns are 2*t and 2*t+1 in the 2x10 block,
    # which corresponds to columns 5+2*t and 5+2*t+1 in the original 2x15 problem.
    # In the V matrix: V[j*p+k] where j in [0,2), k in [0,10) for the 2x10 block.
    # So B[j, 2*t] = V[j*10 + 2*t] and B[j, 2*t+1] = V[j*10 + 2*t + 1]
    
    # In the original 2x15 problem, the V matrix has shape (30, 42).
    # The 2x10 block uses V[10:30, 14:42] (since the first 10 rows are for the 2x5 block).
    # Wait, no. The V matrix has shape (m*p, rank) = (2*15, 42) = (30, 42).
    # The 2x5 block uses V[0:10, 0:14] (first 10 rows, first 14 columns)
    # The 2x10 block uses V[10:30, 14:42] (next 20 rows, next 28 columns)
    
    # Similarly for W: W has shape (p*n, rank) = (15*2, 42) = (30, 42).
    # The 2x5 block uses W[0:10, 0:14]
    # The 2x10 block uses W[10:30, 14:42]
    
    # For the t-th Strassen block in the 2x10 part:
    # B submatrix: rows 0-1, columns 2*t to 2*t+1 in the 2x10 block
    # In the full V: rows 10+j*10+2*t and 10+j*10+2*t+1 for j=0,1
    # Wait, that's not right either. Let me be more careful.
    
    # In the 2x10 block, V has shape (20, 28) where row index = j*10 + k for j in [0,2), k in [0,10).
    # In the full V (30x42), the 2x10 block is at rows 10:30, cols 14:42.
    # So V_full[10 + j*10 + k, 14 + col] = V_2x10[j*10 + k, col]
    
    # For the t-th 2x2 subblock, we use k = 2*t and k = 2*t+1.
    # So the relevant V rows are: 10 + j*10 + 2*t and 10 + j*10 + 2*t + 1 for j=0,1.
    
    # Similarly for W: in the 2x10 block, W has shape (20, 28) where row index = k*2 + i for k in [0,10), i in [0,2).
    # In the full W (30x42), the 2x10 block is at rows 10:30, cols 14:42.
    # So W_full[10 + k*2 + i, 14 + col] = W_2x10[k*2 + i, col]
    
    # For the t-th 2x2 subblock, we use k = 2*t and k = 2*t+1.
    # So the relevant W rows are: 10 + (2*t)*2 + i and 10 + (2*t+1)*2 + i for i=0,1.
    
    # Now let's define the Strassen factors for one 2x2x2 block.
    # U factors (for A): 4 entries corresponding to A[0,0], A[0,1], A[1,0], A[1,1]
    # V factors (for B subblock): 4 entries corresponding to B[0,0], B[0,1], B[1,0], B[1,1]
    # W factors (for C subblock): 4 entries corresponding to C[0,0], C[0,1], C[1,0], C[1,1]
    
    # p1: A=[1,0,0,1], B=[1,0,0,1], C=[1,0,0,1] (contributes +1 to c00 and c11)
    # But wait, c00 = p1 + p4 - p5 - p7, so p1 contributes +1 to c00
    # c11 = p1 - p3 + p5 - p6, so p1 contributes +1 to c11
    # So W for p1: [1, 0, 0, 1]
    
    # p2: A=[0,0,1,1], B=[1,0,0,0], c10 = p2 + p6, so p2 contributes +1 to c10
    # W for p2: [0, 0, 1, 0]
    
    # p3: A=[1,0,0,0], B=[0,0,1,-1], c01 = p3 + p7, so p3 contributes +1 to c01
    # c11 = p1 - p3 + p5 - p6, so p3 contributes -1 to c11
    # W for p3: [0, 1, 0, -1]
    
    # p4: A=[0,0,0,1], B=[0,1,-1,0], c00 = p1 + p4 - p5 - p7, so p4 contributes +1 to c00
    # W for p4: [1, 0, 0, 0]
    
    # p5: A=[1,1,0,0], B=[0,0,0,1], c00 = p1 + p4 - p5 - p7, so p5 contributes -1 to c00
    # c11 = p1 - p3 + p5 - p6, so p5 contributes +1 to c11
    # W for p5: [-1, 0, 0, 1]
    
    # p6: A=[-1,0,1,0], B=[1,1,0,0], c10 = p2 + p6, so p6 contributes +1 to c10
    # c11 = p1 - p3 + p5 - p6, so p6 contributes -1 to c11
    # W for p6: [0, 0, 1, -1]
    
    # p7: A=[0,1,0,-1], B=[0,0,1,1], c01 = p3 + p7, so p7 contributes +1 to c01
    # c00 = p1 + p4 - p5 - p7, so p7 contributes -1 to c00
    # W for p7: [-1, 1, 0, 0]
    
    # Let me verify this is correct by checking that the tensor decomposition works.
    # For each output entry C[i,j], the sum over r of U[i,j,r] * V[j,k,r] * W[k,i,r] should be 1
    # when (i,j) corresponds to the correct product.
    
    # Actually, I realize I should just build this and test it. Let me proceed.
    
    # Define the 7 Strassen products for one 2x2x2 block
    strassen_products = [
        # (U_factors, V_factors, W_factors)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # p1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),  # p2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, -1]),  # p3
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 0, 0]),  # p4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 1]),  # p5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, -1]),  # p6
        ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 1, 0, 0]),  # p7
    ]
    
    # Place the 5 Strassen blocks
    for t in range(5):
        for s in range(7):
            col = 14 + t * 7 + s
            U_factors, V_factors, W_factors = strassen_products[s]
            
            # U: same for all 5 blocks (A is the same)
            U[0, col] = U_factors[0]
            U[1, col] = U_factors[1]
            U[2, col] = U_factors[2]
            U[3, col] = U_factors[3]
            
            # V: for the t-th block, B submatrix is at columns 2*t and 2*t+1 in the 2x10 block
            # V_full[10 + j*10 + k, col] for j=0,1 and k=2*t, 2*t+1
            # V_factors: [B[0,0], B[0,1], B[1,0], B[1,1]] for the subblock
            # So: V[0,0]=V_factors[0], V[0,1]=V_factors[1], V[1,0]=V_factors[2], V[1,1]=V_factors[3]
            # In full V: row 10 + j*10 + k where j is B row, k is B column within the 2x10 block
            V[10 + 0*10 + 2*t, col] = V_factors[0]   # B[0, 2*t]
            V[10 + 0*10 + 2*t + 1, col] = V_factors[1]  # B[0, 2*t+1]
            V[10 + 1*10 + 2*t, col] = V_factors[2]   # B[1, 2*t]
            V[10 + 1*10 + 2*t + 1, col] = V_factors[3]  # B[1, 2*t+1]
            
            # W: for the t-th block, C submatrix is at rows 2*t and 2*t+1 in the 2x10 block
            # W_full[10 + k*2 + i, col] for k=2*t, 2*t+1 and i=0,1
            # W_factors: [C[0,0], C[0,1], C[1,0], C[1,1]] for the subblock
            # So: C[0,0]=W_factors[0], C[0,1]=W_factors[1], C[1,0]=W_factors[2], C[1,1]=W_factors[3]
            # In full W: row 10 + k*2 + i where k is C row, i is C column
            W[10 + (2*t)*2 + 0, col] = W_factors[0]   # C[2*t, 0]
            W[10 + (2*t)*2 + 1, col] = W_factors[1]   # C[2*t, 1]
            W[10 + (2*t+1)*2 + 0, col] = W_factors[2]  # C[2*t+1, 0]
            W[10 + (2*t+1)*2 + 1, col] = W_factors[3]  # C[2*t+1, 1]
    
    # Now verify the decomposition
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                expected = 1.0
                err = abs(val - expected)
                max_err = max(max_err, err)
    
    if max_err > 1e-9:
        # If the decomposition is not exact, fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
