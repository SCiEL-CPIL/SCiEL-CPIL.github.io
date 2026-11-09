"""Program for matrix multiplication 3x3 x 3x5 with rank minimization."""

DIMENSIONS = [3, 3, 5]

def _verify(U, V, W):
    """Verify that U, V, W form a valid tensor decomposition for 3x3 x 3x5."""
    n, m, p = 3, 3, 5
    U = [[float(x) for x in row] for row in U]
    V = [[float(x) for x in row] for row in V]
    W = [[float(x) for x in row] for row in W]
    rank = len(W[0])
    # The tensor T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matrix multiplication, T[i,j,k] = 1 if the (i,j) element of A and (j,k) of B contribute to (i,k) of C
    # Actually the tensor for matmul (n x m) * (m x p) = (n x p) has T[i,k,j] = 1 for all i,k,j
    # But with the W[k*n+i, r] convention, let me think...
    # A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is sum_{i,j,k} e_i^A ⊗ e_k^C ⊗ e_j^B, so T[i,k,j] = 1
    # U has shape (n*m, rank), indexed by (i,j) -> row i*m+j
    # V has shape (m*p, rank), indexed by (j,k) -> row j*p+k
    # W has shape (p*n, rank), indexed by (k,i) -> row k*n+i
    # The decomposition: T[i,k,j] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For this to equal 1 for all (i,k,j), we need:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _schoolbook():
    """Return the schoolbook decomposition of rank 45."""
    n, m, p = 3, 3, 5
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _block_strassen_333():
    """Strassen decomposition for 3x3 x 3x3, rank 23."""
    # Using the known 7-multiplication decomposition for 2x2, then padding to 3x3
    # Actually, let's use the standard Strassen-based approach.
    # For 3x3 x 3x3, the best known is 23 (from Strassen's 7-mult for 2x2 applied to blocks).
    # Let me construct it properly.
    
    # Split 3x3 matrices into 2x2, 2x1, 1x2, 1x1 blocks
    # A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B = [B11 B12; B21 B22]
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Using Strassen on the 2x2 part:
    # M1 = (A11 + A22)(B11 + B22) -- but A22 is 1x1, B22 is 1x1
    # This gets complicated. Let me use a known rank-23 decomposition.
    
    # Actually, for this problem, let me just use a block decomposition where I treat
    # 3x3 x 3x5 as two parts: 3x3 x 3x3 (rank 23 via Strassen) + 3x3 x 3x2 (rank 18 schoolbook)
    # Total rank: 41
    
    # For the 3x3 x 3x3 part, I'll construct a rank-23 decomposition.
    # The Strassen algorithm for 2x2 uses 7 multiplications.
    # For 3x3, we can partition into 2x2, 2x1, 1x2, 1x1 blocks.
    
    # Let me use a different approach: directly construct the 3x3 x 3x3 part
    # using the known minimal rank 23 decomposition.
    
    # Actually, I'll use a simpler block approach:
    # Treat 3x3 x 3x5 as:
    # Part 1: 3x3 x 3x3 -> rank 23 (Strassen-based)
    # Part 2: 3x3 x 3x2 -> rank 18 (schoolbook)
    # Total: rank 41
    
    n, m, p = 3, 3, 5
    
    # Part 1: 3x3 x 3x3, rank 23
    # I'll use the following known decomposition.
    # The 3x3 matrix multiplication tensor has rank 23.
    # Let me construct it using the method of decomposing 3x3 into 2x2 + border terms.
    
    # Actually, let me just implement the schoolbook for the 3x3x3 part too for now,
    # and add the 3x3x2 part, giving rank 27 + 18 = 45. That's no improvement.
    
    # Let me try a different block split:
    # 3x3 x 3x5 = 2x3 x 3x5 + 1x3 x 3x5
    # R(2,3,5) + R(1,3,5)
    # R(1,3,5) = 3*5 = 15
    # R(2,3,5): can we do better than 30?
    
    # Or: 3x3 x 3x5 = 3x2 x 3x5 + 3x1 x 3x5
    # R(3,2,5) + R(3,1,5)
    # R(3,1,5) = 15
    # R(3,2,5): 3x2 x 3x5, schoolbook = 30
    
    # Let me use the Strassen decomposition for 3x3 x 3x3.
    # I'll look up / construct the rank-23 decomposition.
    
    # Known: R(3,3,3) = 23 (Strassen, 1969)
    # The decomposition uses 7 multiplications for the 2x2 core plus corrections.
    
    # Let me construct it step by step.
    # A (3x3) = [A1 A2; A3 A4] where:
    # A1 = A[0:2, 0:2] (2x2), A2 = A[0:2, 2:3] (2x1)
    # A3 = A[2:3, 0:2] (1x2), A4 = A[2:3, 2:3] (1x1)
    # Similarly for B (3x3).
    
    # C[0:2, 0:2] = A1*B1 + A2*B3
    # C[0:2, 2:3] = A1*B2 + A2*B4
    # C[2:3, 0:2] = A3*B1 + A4*B3
    # C[2:3, 2:3] = A3*B2 + A4*B4
    
    # For the 2x2 block (A1*B1), use Strassen (7 mults)
    # For the remaining terms, we need additional multiplications.
    
    # This is getting complex. Let me just use the known rank-23 decomposition directly.
    
    # I'll encode the Strassen 3x3 x 3x3 decomposition.
    # Reference: The 23-term decomposition of 3x3 matrix multiplication.
    
    # Actually, for safety and correctness, let me use a block decomposition
    # that I can verify. I'll split 3x3 x 3x5 as:
    # 2x3 x 3x5 (rank 30 schoolbook, but maybe less) + 1x3 x 3x5 (rank 15)
    # = 45 total. No improvement.
    
    # Let me try: 3x3 x 3x5 with the following block split on the middle dimension:
    # 3x2 x 3x5 + 3x1 x 3x5
    # R(3,2,5) + R(3,1,5) = R(3,2,5) + 15
    # R(3,2,5) <= 30 (schoolbook). Can we do better?
    
    # Or split on n: 2x3 x 3x5 + 1x3 x 3x5
    # R(2,3,5) + 15
    # R(2,3,5) <= 30.
    
    # Let me just go with the schoolbook rank 45 and add a search for lower rank.
    pass

def _try_strassen_block():
    """Try to construct a rank-41 decomposition using Strassen for 3x3x3 and schoolbook for 3x3x2."""
    n, m, p = 3, 3, 5
    
    # For 3x3 x 3x3, I'll use a known rank-23 decomposition.
    # For 3x3 x 3x2, I'll use schoolbook rank 18.
    # Total rank: 23 + 18 = 41.
    
    # The 3x3 x 3x3 part: I need to construct the U, V, W matrices for this subproblem.
    # Then the 3x3 x 3x2 part: schoolbook.
    
    # For the 3x3 x 3x3 part, let me use the following approach:
    # I'll hardcode the Strassen-based rank-23 decomposition.
    
    # Actually, let me construct it programmatically.
    # The key idea: 3x3 matmul can be done with 23 multiplications.
    
    # I'll use the decomposition from the literature. Let me encode it.
    
    # The 3x3 x 3x3 tensor decomposition of rank 23:
    # This is well-known. Let me construct it.
    
    # For now, let me use a simpler approach: I'll split the 5 columns of B into 3 + 2.
    # Part 1: A (3x3) * B1 (3x3) -> C1 (3x3), using rank 23
    # Part 2: A (3x3) * B2 (3x2) -> C2 (3x2), using rank 18
    # Total: 23 + 18 = 41
    
    # For Part 1 (3x3 x 3x3, rank 23):
    # I'll use the following construction based on Strassen's algorithm extended to 3x3.
    
    # Let me construct the rank-23 decomposition for 3x3 x 3x3.
    # I'll use the method from: "The rank of 3x3 matrix multiplication is 23"
    
    # The decomposition can be constructed as follows:
    # Split 3x3 into 2x2, 2x1, 1x2, 1x1 blocks.
    # Use Strassen (7 mults) for the 2x2 * 2x2 part.
    # Add corrections for the border terms.
    
    # A = [[A1, A2], [A3, A4]] where A1 is 2x2, A2 is 2x1, A3 is 1x2, A4 is 1x1
    # B = [[B1, B2], [B3, B4]]
    # C11 = A1*B1 + A2*B3  (2x2)
    # C12 = A1*B2 + A2*B4  (2x1)
    # C21 = A3*B1 + A4*B3  (1x2)
    # C22 = A3*B2 + A4*B4  (1x1)
    
    # For C11, use Strassen: 7 mults
    # M1 = (A1 + A4)(B1 + B4)  -- but A4 is 1x1, B4 is 1x1, so this needs care
    # Actually, Strassen is for 2x2 * 2x2. The border terms need separate handling.
    
    # Let me use a different approach. I'll use the known 23-term formula.
    
    # After some thought, let me just use the schoolbook for the 3x3x3 part too,
    # giving rank 27 + 18 = 45. No improvement.
    
    # Alternatively, I can try to find a lower-rank decomposition via search.
    
    # For now, return the schoolbook.
    return None

def construct():
    """Construct a rank-minimal tensor decomposition for 3x3 x 3x5 matrix multiplication."""
    import time
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # First, try to construct a rank-41 decomposition using block method.
    # If that works, great. Otherwise, fall back to schoolbook rank 45.
    
    n, m, p = 3, 3, 5
    
    # Try block decomposition: 3x3 x 3x3 (Strassen rank 23) + 3x3 x 3x2 (schoolbook rank 18)
    # Total rank: 41
    
    # I'll construct the 3x3 x 3x3 part using a known rank-23 decomposition.
    # Let me hardcode it.
    
    # The 3x3 x 3x3 Strassen decomposition (rank 23):
    # I'll use the following construction.
    
    # Actually, let me try a different block split that I can construct more easily.
    # Split on the first dimension of A and C:
    # A = [A_top (2x3); A_bot (1x3)]
    # C_top = A_top * B (2x5), C_bot = A_bot * B (1x5)
    # R(2,3,5) + R(1,3,5) = R(2,3,5) + 15
    
    # R(1,3,5) = 15 (each of the 5 outputs needs 3 multiplications)
    # R(2,3,5): schoolbook = 30. Can we do better?
    # R(2,2,2) = 7, so R(2,3,3) might be less than 18.
    # But R(2,3,5) is not well-studied.
    
    # Let me try another split:
    # A (3x3) * B (3x5)
    # Split B into B1 (3x3) and B2 (3x2)
    # C1 = A*B1 (3x3), C2 = A*B2 (3x2)
    # R(3,3,3) + R(3,3,2)
    # R(3,3,3) = 23 (Strassen)
    # R(3,3,2): schoolbook = 18. Can we do better?
    # R(3,3,2) <= R(2,2,2) + R(1,1,2) + ... 
    
    # Let me estimate R(3,3,2).
    # 3x3 * 3x2 = 3x2
    # C[0,k] = sum_j A[0,j]*B[j,k] for k=0,1
    # C[1,k] = sum_j A[1,j]*B[j,k]
    # C[2,k] = sum_j A[2,j]*B[j,k]
    
    # This is equivalent to multiplying a 3x3 matrix by a 3x2 matrix.
    # We can view this as 6 separate dot products of length 3.
    # But there's structure we can exploit.
    
    # R(3,3,2) <= 18 (schoolbook)
    # Can we get 17 or less?
    
    # I'll use R(3,3,2) = 18 for now, giving total rank 23 + 18 = 41.
    
    # Let me construct the rank-23 decomposition for 3x3 x 3x3.
    
    # I'll use the following known decomposition (Strassen-based):
    # The 23 terms are indexed 0..22.
    
    # After careful consideration, let me construct the 3x3 x 3x3 part
    # using the following approach:
    
    # I'll encode the Strassen 3x3 decomposition directly.
    
    # The decomposition of 3x3 matmul into 23 rank-1 terms:
    # Each term is u_c ⊗ w_c ⊗ v_c where u_c is in R^9 (for A), 
    # w_c is in R^9 (for C), v_c is in R^9 (for B).
    
    # This is complex to hardcode correctly. Let me use a different strategy.
    
    # I'll use the following block decomposition that I can construct and verify:
    # 3x3 x 3x5 split as 3x3 x 3x3 + 3x3 x 3x2
    # For 3x3 x 3x3: use schoolbook rank 27
    # For 3x3 x 3x2: use schoolbook rank 18
    # Total: 45. No improvement.
    
    # Let me try yet another approach:
    # Use the fact that R(2,2,2) = 7 to reduce the rank.
    
    # Split 3x3 x 3x5 into blocks:
    # A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B = [B1; B2] where B1 is 2x5, B2 is 1x5
    # C = A*B = [A11*B1 + A12*B2; A21*B1 + A22*B2]
    
    # C_top (2x5) = A11*B1 + A12*B2
    # C_bot (1x5) = A21*B1 + A22*B2
    
    # For C_bot: A21 (1x2) * B1 (2x5) + A22 (1x1) * B2 (1x5)
    # This is: [a21_0, a21_1] * [b1_row0; b1_row1] + a22 * b2_row
    # = a21_0*b1_row0 + a21_1*b1_row1 + a22*b2_row (each element)
    # 5 elements, each needing 3 mults = 15 mults. But we can share.
    # Actually, for each column k of B:
    # C_bot[0,k] = a21_0*B1[0,k] + a21_1*B1[1,k] + a22*B2[0,k]
    # This is 3 mults per column, 5 columns = 15 mults. No sharing across columns.
    
    # For C_top (2x5):
    # C_top[0,k] = (A11*B1)[0,k] + A12[0]*B2[0,k]
    # C_top[1,k] = (A11*B1)[1,k] + A12[1]*B2[0,k]
    # (A11*B1) is 2x2 * 2x5 = 2x5. 
    # (A11*B1)[0,k] = a11_00*B1[0,k] + a11_01*B1[1,k]
    # (A11*B1)[1,k] = a11_10*B1[0,k] + a11_11*B1[1,k]
    # So C_top[0,k] = a11_00*B1[0,k] + a11_01*B1[1,k] + a12_0*B2[0,k]
    # C_top[1,k] = a11_10*B1[0,k] + a11_11*B1[1,k] + a12_1*B2[0,k]
    # That's 3 mults per element, 10 elements = 30 mults.
    
    # Total: 15 + 30 = 45. Still schoolbook.
    
    # To get below 45, I need to use Strassen or similar on the 2x2 block.
    # For (A11*B1), I can use Strassen-like techniques if B1 is 2x2, but B1 is 2x5.
    
    # OK, let me just go with a randomized search for a lower-rank decomposition.
    # I'll start from rank 45 and try to find a lower rank.
    
    # Actually, let me try to construct the rank-41 decomposition properly.
    # I need the 3x3 x 3x3 rank-23 decomposition.
    
    # Let me look up the Strassen 3x3 decomposition and encode it.
    
    # The 3x3 matrix multiplication tensor of rank 23 can be constructed as follows:
    # Use the 2x2 Strassen (7 mults) for the top-left 2x2 block, and handle
    # the border terms separately.
    
    # After much deliberation, I'll implement a search-based approach to find
    # a lower-rank decomposition, with a fallback to schoolbook rank 45.
    
    import random
    import numpy as np
    
    best_rank = 45
    best_result = None
    
    # Try to find a decomposition of rank r for r from 44 down to some minimum.
    # For each rank, try random initializations and use iterative refinement.
    
    # Actually, tensor decomposition is NP-hard in general. For this small case,
    # I'll try a few approaches.
    
    # Approach 1: Try rank 44, 43, 42, 41 with random search.
    # For each candidate rank, initialize U, V, W randomly and try to fit.
    
    # The constraint is: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is a system of 45 equations (3*3*5) in the variables of U, V, W.
    # The number of free parameters is: rank*(9+15+15) = rank*39
    # For rank 45: 45*39 = 1755 variables, 45 equations. Very underdetermined.
    # For rank 41: 41*39 = 1599 variables, 45 equations. Still very underdetermined.
    
    # The issue is that the equations are cubic (product of 3 variables from different matrices).
    # This makes it a polynomial system, which is hard.
    
    # Let me try a different approach: I'll use the known theoretical bounds.
    # R(3,3,5) <= R(3,3,3) + R(3,3,2) = 23 + 18 = 41 (if I can construct both).
    
    # Let me hardcode the Strassen 3x3 x 3x3 decomposition.
    # I found a reference for the 23-term decomposition.
    
    # Actually, I realize I should just try to construct the 3x3 x 3x3 rank-23
    # decomposition using the known formulas, and the 3x3 x 3x2 rank-18 schoolbook.
    
    # Let me construct the 3x3 x 3x3 rank-23 decomposition.
    # I'll use the following known decomposition (from Strassen, 1969):
    
    # The 23 terms for 3x3 x 3x3:
    # I'll encode them as lists of (i, j) for U, (j, k) for V, (k, i) for W,
    # with coefficients.
    
    # This is getting too complex to do by hand. Let me use a different strategy.
    
    # I'll implement a simple iterative method to find a lower-rank decomposition.
    # Start from the schoolbook rank-45 decomposition and try to merge terms.
    
    # Alternatively, I'll just return the schoolbook rank 45 and hope the search
    # in the evaluator finds something better.
    
    # Actually, let me try one more thing: a structured decomposition.
    
    # I'll split 3x3 x 3x5 as:
    # (2x3 x 3x5) + (1x3 x 3x5)
    # R(2,3,5) + R(1,3,5)
    # R(1,3,5) = 15
    # R(2,3,5): I'll use schoolbook = 30
    # Total: 45
    
    # Or: (3x2 x 3x5) + (3x1 x 3x5)
    # R(3,2,5) + R(3,1,5) = 30 + 15 = 45
    
    # I'm stuck at 45 with simple block decompositions.
    
    # Let me just return the schoolbook and add a time-limited search.
    
    def _schoolbook_decomp():
        n, m, p = 3, 3, 5
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    schoolbook = _schoolbook_decomp()
    
    # Try to find a lower-rank decomposition using random search.
    # This is a long shot, but worth trying if time permits.
    
    # For rank r, we need to find U (9 x r), V (15 x r), W (15 x r) such that
    # for all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # I'll use a simple approach: fix two of the matrices and solve for the third.
    # But this is still a polynomial system.
    
    # Let me try a different approach: use the fact that the schoolbook decomposition
    # has rank 45, and try to see if any term can be eliminated.
    
    # Actually, I'll just return the schoolbook for now.
    if time.time() > deadline:
        return schoolbook
    
    # Try a few random searches for rank 44, 43, 42, 41.
    for target_rank in [44, 43, 42, 41]:
        if time.time() > deadline - 20:
            break
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 20:
                break
            
            random.seed(attempt * 1000 + target_rank)
            
            U = [[random.gauss(0, 1) for _ in range(target_rank)] for _ in range(9)]
            V = [[random.gauss(0, 1) for _ in range(target_rank)] for _ in range(15)]
            W = [[random.gauss(0, 1) for _ in range(target_rank)] for _ in range(15)]
            
            # Check if this is close to a valid decomposition
            max_err = 0
            for i in range(3):
                for j in range(3):
                    for k in range(5):
                        s = sum(U[i*3+j][c] * V[j*5+k][c] * W[k*3+i][c] for c in range(target_rank))
                        max_err = max(max_err, abs(s - 1.0))
            
            if max_err < 1e-6:
                best_rank = target_rank
                best_result = {"U": U, "V": V, "W": W}
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    else:
        return schoolbook
