import time
import numpy as np

DIMENSIONS = [2, 3, 3]
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 120s kill

def construct():
    n, m, p = DIMENSIONS
    
    # Build target tensor for validation
    target_tensor = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                target_tensor[u, v, w] = 1.0
    
    def to_half_int_list(matrix):
        scaled = np.rint(matrix * 2.0)
        result = scaled / 2.0
        return [[float(v) for v in row] for row in result]

    def validate(U, V, W):
        """Check if U, V, W decompose the tensor exactly."""
        result = np.zeros((n*m, m*p, p*n))
        for c in range(U.shape[1]):
            result += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        return np.allclose(result, target_tensor, atol=1e-8)
    
    # Construct rank-17 decomposition using Strassen for 2x2 subproblem + schoolbook
    # A is 2x3, B is 3x3, C is 2x3
    # Split: C_left (2x2) uses Strassen (7 mults), C_right (2x1) uses schoolbook (6 mults)
    # Total: 7 + 6 = 13... wait, that's not right either.
    # Actually: Strassen for 2x2x2 gives 7, plus 2x1x3 schoolbook gives 6, total 13? No.
    # Let me reconsider. The known rank is 17.
    # 
    # Better approach: use the known explicit construction.
    # For 2x3x3, rank 17 is achievable. Let me construct it directly.
    
    # Using a known construction:
    # Think of it as: first column of C (2x1) from A(2x3)*B_col0(3x1): 6 terms
    # Second column of C (2x1) from A(2x3)*B_col1(3x1): 6 terms  
    # Third column of C (2x1) from A(2x3)*B_col2(3x1): 6 terms
    # Total schoolbook: 18
    # 
    # To save 1: combine some terms using Strassen-like tricks on pairs.
    # 
    # Let me use a different known construction. The rank of <2,3,3> is 17.
    # One construction: treat B as three 3x1 vectors b0, b1, b2.
    # C = [A*b0, A*b1, A*b2]
    # 
    # Use Strassen on the 2x2 subproblem formed by (A[:,0:2], B[0:2,:]) for columns 0,1:
    # That gives 7 multiplications for C[:,0:2] using only A[:,0:2] and B[0:2,0:2]
    # But we also need contributions from A[:,2] and B[2,0:2]
    # 
    # Let me just hardcode a known rank-17 construction.
    
    rank = 17
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
    # i in {0,1}, j in {0,1,2}, k in {0,1,2}
    
    # Strategy: Use Strassen for the "core" 2x2x2 (i in {0,1}, j in {0,1}, k in {0,1})
    # Then add schoolbook terms for remaining parts.
    
    # Strassen 2x2x2 on (i={0,1}, j={0,1}, k={0,1}):
    # M1 = (A00+A01)*B00 -> contributes to C00
    # M2 = (A00+A01)*B10 -> contributes to C10  
    # M3 = A01*(B00-B10) -> contributes to C00
    # M4 = A00*(B01+B11) -> contributes to C01
    # M5 = (A00+A01)*(B11-B00) -> contributes to C01
    # M6 = A01*(B01+B11) -> contributes to C01
    # M7 = (A00-A01)*B00 -> contributes to C00
    # 
    # C00 = M1 + M3 + M7  ... wait, standard Strassen:
    # C00 = M1 - M3 + M7
    # C01 = M4 + M5
    # C10 = M2 - M5 + M6  ... hmm, let me be more careful.
    #
    # Standard Strassen for 2x2:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = a22(b12-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 - M4
    # C22 = M1 - M2 + M4
    #
    # For our case with i=row of A (2 rows), j=col of A / row of B (using j in {0,1}), k=col of B (2 cols)
    # A submatrix: A[0:2, 0:2] (2x2), B submatrix: B[0:2, 0:2] (2x2)
    # C submatrix: C[0:2, 0:2] (2x2)
    #
    # Using Strassen indexing where a_ij = A[i,j], b_jk = B[j,k], c_ik = C[i,k]:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[1,0]-B[1,1])
    # M4 = A[1,1]*(B[0,1]-B[0,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    # C[0,0] = M1 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 - M4
    # C[1,1] = M1 - M2 + M4
    
    # Now I need to add contributions from:
    # - A[0,2]*B[2,k] for k=0,1,2 (row 0 of A, col 2)
    # - A[1,2]*B[2,k] for k=0,1,2 (row 1 of A, col 2)
    # - A[i,0]*B[0,k] already handled in Strassen for k=0,1
    # - A[i,1]*B[1,k] already handled in Strassen for k=0,1
    # - A[i,0]*B[0,2] for i=0,1
    # - A[i,1]*B[1,2] for i=0,1
    # - A[i,2]*B[2,2] for i=0,1
    #
    # So remaining terms (schoolbook):
    # C[0,0]: A[0,2]*B[2,0]
    # C[0,1]: A[0,2]*B[2,1]
    # C[0,2]: A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,0]: A[1,2]*B[2,0]
    # C[1,1]: A[1,2]*B[2,1]
    # C[1,2]: A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    
    # That's 7 (Strassen) + 9 (remaining) = 16... but we need 17?
    # Wait, let me recount. The remaining terms:
    # (0,2)x(2,0), (0,2)x(2,1), (0,0)x(0,2), (0,1)x(1,2), (0,2)x(2,2),
    # (1,2)x(2,0), (1,2)x(2,1), (1,0)x(0,2), (1,1)x(1,2), (1,2)x(2,2)
    # That's 10 terms. 7 + 10 = 17. Yes!
    
    col = 0
    # Strassen M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # Contributes: +1 to C[0,0], +1 to C[1,1]
    U[0*3+0, col] = 1.0  # A[0,0]
    U[1*3+1, col] = 1.0  # A[1,1]
    V[0*3+0, col] = 1.0  # B[0,0]
    V[1*3+1, col] = 1.0  # B[1,1]
    W[0*2+0, col] = 1.0  # C[0,0]
    W[1*2+1, col] = 1.0  # C[1,1]
    col += 1
    
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # Contributes: -1 to C[1,0], -1 to C[1,1]
    U[1*3+0, col] = 1.0  # A[1,0]
    U[1*3+1, col] = 1.0  # A[1,1]
    V[0*3+0, col] = 1.0  # B[0,0]
    W[0*2+1, col] = -1.0  # C[1,0]
    W[1*2+1, col] = -1.0  # C[1,1]
    col += 1
    
    # M3 = A[0,0]*(B[1,0]-B[1,1])
    # Contributes: +1 to C[0,1]
    U[0*3+0, col] = 1.0  # A[0,0]
    V[1*3+0, col] = 1.0  # B[1,0]
    V[1*3+1, col] = -1.0  # B[1,1]
    W[0*2+1, col] = 1.0  # C[0,1]
    col += 1
    
    # M4 = A[1,1]*(B[0,1]-B[0,0])
    # Contributes: -1 to C[1,0], +1 to C[1,1]
    U[1*3+1, col] = 1.0  # A[1,1]
    V[0*3+1, col] = 1.0  # B[0,1]
    V[0*3+0, col] = -1.0  # B[0,0]
    W[0*2+1, col] = -1.0  # C[1,0]
    W[1*2+1, col] = 1.0  # C[1,1]
    col += 1
    
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # Contributes: -1 to C[0,0], +1 to C[0,1]
    U[0*3+0, col] = 1.0  # A[0,0]
    U[0*3+1, col] = 1.0  # A[0,1]
    V[1*3+1, col] = 1.0  # B[1,1]
    W[0*2+0, col] = -1.0  # C[0,0]
    W[0*2+1, col] = 1.0  # C[0,1]
    col += 1
    
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # This doesn't appear in the standard C formulas I wrote above...
    # Let me recheck. Standard Strassen:
    # C11 = M1 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 - M4
    # C22 = M1 - M2 + M4
    # M6 is not used in the final C! Wait, that can't be right.
    # 
    # Let me look up the correct Strassen formulas again:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = a22(b12-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 - M4
    # C22 = M1 - M2 + M4
    #
    # Yes, M6 is indeed not used in the output! That's a known property of Strassen - 
    # one of the 7 products doesn't directly appear. Wait, that doesn't make sense.
    # Let me verify: 7 multiplications, 4 outputs, each output is a linear combination.
    # C11 = M1 - M5 + M7 (3 terms)
    # C12 = M3 + M5 (2 terms)
    # C21 = M2 - M4 (2 terms)
    # C22 = M1 - M2 + M4 (3 terms)
    # Total appearances: M1:2, M2:2, M3:1, M4:2, M5:2, M6:0, M7:1 = 10 appearances for 4 outputs
    # That's correct! M6 is a "waste" in the sense that it's computed but... 
    # No wait, that can't be. Let me recheck.
    #
    # Actually I think I have the wrong formulas. Let me use a verified source:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)(b11+b12)
    # M3 = (a11-a22)(b21+b22)
    # M4 = a11(b12-b22)
    # M5 = a22(b21-b11)
    # M6 = (a11+a12)(b22-b12)
    # M7 = (a21+a22)(b11-b21)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M2 + M4 - M6 - M7
    # C21 = M2 + M5 - M6 - M7
    # C22 = M1 + M3 - M4 + M6
    #
    # Let me verify with a simple case. a = [[1,0],[0,1]], b = [[1,0],[0,1]]
    # M1 = (1+1)(1+1) = 4
    # M2 = (0+1)(1+0) = 1
    # M3 = (1-1)(0+1) = 0
    # M4 = 1*(0-1) = -1
    # M5 = 1*(0-1) = -1
    # M6 = (1+0)(1-0) = 1
    # M7 = (0+1)(1-0) = 1
    # C11 = 4 + (-1) - (-1) + 1 = 5? Should be 1. Wrong.
    #
    # Let me try yet another formulation. I'll use the Wikipedia version:
    # p1 = (a11+a22)(b11+b22)
    # p2 = (a21+a22)b11
    # p3 = a11(b21-b22)
    # p4 = a22(b12-b11)
    # p5 = (a11+a12)b22
    # p6 = (a21-a11)(b11+b12)
    # p7 = (a12-a22)(b21+b22)
    # c11 = p1 - p5 + p7
    # c12 = p3 + p5
    # c21 = p2 - p4
    # c22 = p1 - p2 + p4
    #
    # Verify: a=[[1,0],[0,1]], b=[[1,0],[0,1]]
    # p1 = (1+1)(1+1) = 4
    # p2 = (0+1)(1) = 1
    # p3 = 1*(0-1) = -1
    # p4 = 1*(0-1) = -1
    # p5 = (1+0)(1) = 1
    # p6 = (0-1)(1+0) = -1
    # p7 = (0-1)(0+1) = -1
    # c11 = 4 - 1 + (-1) = 2? Should be 1. Still wrong!
    #
    # I'm clearly making errors. Let me just use a direct construction instead.
    # 
    # For the 2x2x2 Strassen, let me verify by direct computation.
    # Actually, let me just abandon the Strassen approach and use a known explicit rank-17 construction
    # or find it through a different method.
    
    # Let me reset and use a cleaner approach.
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # I'll use a known result: the rank of 2x3x3 is 17.
    # Construction: partition B into columns b0, b1, b2 (each 3x1).
    # C[:,0] = A*b0, C[:,1] = A*b1, C[:,2] = A*b2
    # 
    # Use the fact that for 2x3x1, rank is 6 (schoolbook is optimal for this).
    # But we can share some computations between the three columns.
    #
    # Alternative: use the algorithm from "The rank of 2x3x3 matrix multiplication"
    # 
    # Let me try a different strategy: use integer linear programming or just 
    # search over small integer coefficients.
    
    # Actually, let me just try to find a rank-17 solution via optimization with 
    # the constraint that entries are in {-1, -0.5, 0, 0.5, 1}.
    
    # For now, let me return the schoolbook rank-18 solution as a safe fallback,
    # but also try a quick search for rank 17.
    
    # Schoolbook rank 18
    U_sb = np.zeros((n*m, 18))
    V_sb = np.zeros((m*p, 18))
    W_sb = np.zeros((p*n, 18))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    
    # Try rank 17 with a simple search over {-1, 0, 1} coefficients
    # This is a small enough problem that we can try random restarts
    
    import random
    random.seed(42)
    np.random.seed(42)
    
    def try_rank_17():
        """Try to find a rank-17 decomposition with entries in {-1, 0, 1}."""
        if time.time() > START_TIME + TIME_LIMIT - 20:
            return None
            
        for attempt in range(5000):
            if time.time() > START_TIME + TIME_LIMIT - 10:
                break
                
            # Random sparse initialization
            U_try = np.zeros((n*m, 17))
            V_try = np.zeros((m*p, 17))
            W_try = np.zeros((p*n, 17))
            
            for c in range(17):
                # Each factor vector should have a few non-zero entries
                nz_u = random.randint(1, 3)
                nz_v = random.randint(1, 3)
                nz_w = random.randint(1, 2)
                
                idx_u = random.sample(range(n*m), nz_u)
                idx_v = random.sample(range(m*p), nz_v)
                idx_w = random.sample(range(p*n), nz_w)
                
                for idx in idx_u:
                    U_try[idx, c] = random.choice([-1, 1])
                for idx in idx_v:
                    V_try[idx, c] = random.choice([-1, 1])
                for idx in idx_w:
                    W_try[idx, c] = random.choice([-1, 1])
            
            # Check
            if validate(U_try, V_try, W_try):
                return (U_try, V_try, W_try)
                
        return None
    
    result = try_rank_17()
    if result:
        U, V, W = result
        return {"U": to_half_int_list(U), "V": to_half_int_list(V), "W": to_half_int_list(W)}
    else:
        return {"U": to_half_int_list(U_sb), "V": to_half_int_list(V_sb), "W": to_half_int_list(W_sb)}
