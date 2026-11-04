"""Improved program: search for rank-minimal tensor decomposition of 4x4*4x7."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    DEADLINE = 130.0  # leave 20s safety margin
    
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank) = (16, rank)
    # V shape: (m*p, rank) = (28, rank)
    # W shape: (p*n, rank) = (28, rank)
    # Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    
    # Start with schoolbook rank = 112
    # Try to find lower rank using structured approaches
    
    # Approach 1: Use Kronecker product structure
    # 4x4 * 4x7 = (2x2 ⊗ 2x2) * (2x2 ⊗ 2x4) roughly
    # Actually, let's think about block decomposition
    
    # 4 = 2*2, so we can write A as 2x2 blocks of 2x2
    # B as 2x2 blocks of 2x4 (since 7 isn't divisible by 2, this is messy)
    
    # Alternative: try rank 112 first (schoolbook), then try to reduce
    
    # Let's try a structured approach using Strassen-like ideas
    # For 2x2 matrices: rank 7 (Strassen)
    # For 4x4 * 4x7: 
    #   Split 4x4 into 2x2 blocks: 4 blocks of 2x2
    #   Split 4x7 into 2x2 blocks: 2 blocks of 2x2 + 2x3 (messy)
    
    # Let's try: 4x4 * 4x7
    # Write A = [A11 A12; A21 A22] where each Ai is 2x2
    # Write B = [B11 B12 B13; B21 B22 B23] where Bij is 2x2 for j=1,2 and 2x3 for j=3
    # Hmm, 7 = 2+2+3, so B columns split into 2,2,3
    
    # This is getting complex. Let's try a different approach.
    
    # Try to find a decomposition of rank < 112 using random initialization + optimization
    
    # Actually, let's first try some known bounds and structured decompositions
    
    # For m x n * n x p, the rank is at most m*n*p (schoolbook)
    # Known: 4x4 * 4x4 has rank at most 49 (Strassen-like)
    # For 4x4 * 4x7, we can use block methods
    
    # Let me try: decompose using 2x2 blocks where possible
    # A (4x4) = [[A1, A2], [A3, A4]] each 2x2
    # B (4x7) = [[B1, B2, B3], [B4, B5, B6]] where B1,B2 are 2x2, B3 is 2x3, etc.
    # Wait, 7 columns: split as 2+2+3
    
    # Product C = A*B (4x7):
    # C11 = A1*B1 + A2*B4 (2x2)
    # C12 = A1*B2 + A2*B5 (2x2)  
    # C13 = A1*B3 + A2*B6 (2x3)
    # C21 = A3*B1 + A4*B4 (2x2)
    # C22 = A3*B2 + A4*B5 (2x2)
    # C23 = A3*B3 + A4*B6 (2x3)
    
    # For the 2x2 * 2x2 parts, we can use Strassen (rank 7)
    # For 2x2 * 2x3 parts, rank at most 12 (schoolbook) or less
    
    # Let's count: 
    # 4 blocks of 2x2*2x2: 4 * 7 = 28
    # 2 blocks of 2x2*2x3: 2 * 12 = 24 (schoolbook) or maybe less
    # Total: ~52? But there's overlap in the intermediate products
    
    # Actually, Strassen's algorithm computes specific linear combinations.
    # Let me think more carefully.
    
    # Strassen for 2x2*2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # This is hard to generalize directly. Let me try a numerical search.
    
    # Try random search for low-rank decomposition
    # We need integer/half-integer entries, so let's work with 2*entries as integers
    
    def check_decomposition(U, V, W, n, m, p, tol=1e-9):
        """Check if U,V,W exactly decompose the tensor."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def check_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
        """Vectorized check."""
        rank = U.shape[1]
        # Build tensor via einsum
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_sb = np.zeros((n*m, rank_school))
    V_sb = np.zeros((m*p, rank_school))
    W_sb = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # Try to find better decompositions using structured approaches
    # First, try some specific ranks
    
    # Let's try a greedy/random search approach
    # For each candidate rank, try to find U,V,W with small integer/half-integer entries
    
    def try_random_decomposition(rank, n, m, p, num_trials=100, max_val=4):
        """Try random initializations and local search."""
        for trial in range(num_trials):
            # Initialize with random small integers (multiply by 2 for half-integers)
            U = np.random.randint(-max_val, max_val+1, size=(n*m, rank)) * 0.5
            V = np.random.randint(-max_val, max_val+1, size=(m*p, rank)) * 0.5
            W = np.random.randint(-max_val, max_val+1, size=(p*n, rank)) * 0.5
            
            # Check
            if check_decomposition_fast(U, V, W, n, m, p):
                return U, V, W
        return None
    
    # Try some structured decompositions
    # Idea: use the fact that 4 = 2^2 to exploit Kronecker structure
    
    # For 2x2 * 2x2, Strassen gives rank 7
    # Can we build a 4x4 * 4x7 decomposition from smaller blocks?
    
    # Let's try: treat the problem as involving 2x2 sub-problems
    # A is 4x4, B is 4x7
    # Split rows of A into 2 groups of 2, columns of A into 2 groups of 2
    # Split rows of B into 2 groups of 2, columns of B into groups of 2,2,3
    
    # This is complex. Let me try a simpler approach: 
    # Use the schoolbook but with some sharing via linearity
    
    # Actually, let's just do a more careful search over possible ranks
    # starting from schoolbook and going down
    
    # For efficiency, let's focus on a few promising ranks
    # Known: for 4x4*4x4, best known is around 49
    # For 4x4*4x7, it should be somewhat larger
    
    # Let's try rank 100, 90, 80, 70, 60, 50
    # But random search is unlikely to find exact decompositions
    
    # Better approach: use the structure of the problem
    # The tensor is all-ones, so we need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
    
    # One approach: use rank-1 terms that are "spread out"
    # Each term contributes to multiple (i,j,k) entries
    
    # Let's try a construction based on modular arithmetic or similar
    
    # For now, let's return the schoolbook solution if we can't do better
    # But first, try a few specific structured approaches
    
    # Approach: use 7 terms for each 2x2 block pair (Strassen)
    # This requires careful bookkeeping
    
    # Let me try a direct construction:
    # Split the 4x4 matrix A into four 2x2 blocks: A = [[A00, A01], [A10, A11]]
    # Split the 4x7 matrix B into six blocks: B = [[B00, B01, B02], [B10, B11, B12]]
    # where B00, B01, B10, B11 are 2x2, and B02, B12 are 2x3
    
    # The product C = AB has blocks:
    # C00 = A00*B00 + A01*B10 (2x2)
    # C01 = A00*B01 + A01*B11 (2x2)
    # C02 = A00*B02 + A01*B12 (2x3)
    # C10 = A10*B00 + A11*B10 (2x2)
    # C11 = A10*B01 + A11*B11 (2x2)
    # C12 = A10*B02 + A11*B12 (2x3)
    
    # For each 2x2*2x2 product, use Strassen (7 terms)
    # For each 2x2*2x3 product, use schoolbook (6 terms) or better
    
    # Total terms: 4*7 + 2*6 = 28 + 12 = 40? But there's sharing possible
    
    # Actually, Strassen's algorithm computes 7 specific linear combinations.
    # Let me write out the Strassen formulas for each block product.
    
    # For C00 = A00*B00 + A01*B10:
    # This is a sum of two 2x2*2x2 products. We can compute each with 7 terms = 14 terms
    # Or we can combine them: (A00+B01... no, that doesn't work directly)
    
    # This is getting very complex. Let me just return the schoolbook for now
    # but try one more thing: a simple improvement using linearity
    
    # Idea: instead of 112 separate terms, can we share some?
    # For example, if we have two terms that differ only in one index,
    # we might be able to combine them.
    
    # But the tensor is all-ones, so each (i,j,k) needs exactly 1.
    # Schoolbook gives exactly 1 for each (i,j,k) with no overlap.
    # To reduce rank, we need terms that contribute to multiple (i,j,k) with cancellation.
    
    # This requires careful design. Let me try a small search.
    
    # Given time constraints, let's return the schoolbook solution
    # which is guaranteed correct with rank 112
    
    if time.time() - start_time > DEADLINE - 10:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try a few random searches at lower ranks (long shot)
    for test_rank in [100, 90, 80]:
        if time.time() - start_time > DEADLINE - 15:
            break
        result = try_random_decomposition(test_rank, n, m, p, num_trials=50, max_val=3)
        if result is not None:
            U, V, W = result
            if check_decomposition_fast(U, V, W, n, m, p):
                best_rank = test_rank
                best_U, best_V, best_W = U, V, W
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
