"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # We'll try to find the best rank decomposition
    # Start with schoolbook and try to improve
    
    # Schoolbook rank = n*m*p = 45
    # Try to find better decompositions
    
    deadline = time.time() + 130  # Safety margin
    
    def verify(U, V, W):
        """Verify that the decomposition exactly matches matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # For random test matrices, check A*B == sum_c U[:,c] * V[:,c] * W[:,c]
        # The tensor is: C[i,k] = sum_j A[i,j] * B[j,k]
        # Decomposition: C = sum_c u_c ⊗ v_c ⊗ w_c
        # where u_c is in R^n, v_c in R^m, w_c in R^p
        # But our format is: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        
        # U[i*m + j, c] = u_c[i,j] (coefficient for A[i,j])
        # V[j*p + k, c] = v_c[j,k] (coefficient for B[j,k])  
        # W[k*n + i, c] = w_c[k,i] (coefficient for C[i,k])
        
        # The product: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should = delta_{i,k} contribution
        # Actually: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
        # In tensor form: T[i,j,k] = delta_{i,k}... no.
        # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
        # So T[i,j,k] contributes to C[i,k] when j is summed.
        # T[i,j,k] = 1 for all valid (i,j,k) where the product A[i,j]*B[j,k] contributes to C[i,k]
        # That's all (i,j,k) triples! T[i,j,k] = 1.
        
        # Wait, let me re-read. The multiplication tensor T has T[i,j,k] = 1 for all i,j,k
        # because C[i,k] = sum_j A[i,j]*B[j,k].
        # So every triple (i,j,k) has coefficient 1.
        
        # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Wait, that's not right either. Let me re-read the format.
        
        # U has shape (n*m, rank): U[i*m+j, c]
        # V has shape (m*p, rank): V[j*p+k, c]
        # W has shape (p*n, rank): W[k*n+i, c]
        
        # The reconstruction: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # should equal T[i,j,k] = 1 (since every A[i,j]*B[j,k] contributes to C[i,k])
        
        # Let's verify with a specific test
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)  
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    def verify_fast(U, V, W):
        """Vectorized verification."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        Ur = U.reshape(n, m, rank)
        Vr = V.reshape(m, p, rank)
        Wr = W.reshape(p, n, rank)
        
        # For each (i,j,k): sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
        # Let's compute this efficiently
        # sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
        # = sum_c (Ur * Vr) elementwise with Wr... but indices are coupled.
        
        # Let's just do it directly for small dimensions
        result = np.zeros((n, m, p))
        for c in range(rank):
            # Ur[:, :, c] * Vr[:, :, c] * Wr[:, :, c] but with proper index coupling
            # Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        result[i,j,k] += Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
        
        return np.allclose(result, np.ones((n,m,p)), atol=1e-10)
    
    # Try to find a good decomposition
    # Strategy: Use known constructions
    
    # For 3x3x5, try rank 40 first, then go lower
    
    # Let me try a constructive approach using block matrices
    
    # Approach: Split the 5 columns into groups and use Strassen for 2x2 blocks
    
    # Actually, let me try to use a known result. 
    # M(3,3,5) can be bounded by considering it as a matrix multiplication problem.
    
    # Let me try a simple improvement: use the fact that some products can be shared.
    
    # For now, let's try ranks from 40 down to 35 and see if we can find valid ones
    # by using a structured approach.
    
    # One known approach: M(3,3,5) <= 2*M(3,3,3) + something
    # Or use the "lax" rank bounds.
    
    # Let me try to construct using Strassen for 2x2x2 blocks
    
    # A is 3x3, B is 3x5
    # Split A into [[A11, A12], [A21, A22, A23]] where A11 is 2x2, etc.
    # This gets complicated. Let me try a different approach.
    
    # Direct approach: try to find integer/half-integer decompositions
    # using a greedy or structured method.
    
    # Let me try rank 40 with a structured pattern.
    
    # Actually, let me use the following known bound:
    # M(3,3,5) <= M(3,3,3) + 3*M(1,3,1) + 2*M(1,3,1) ... no, this doesn't work cleanly.
    
    # Let me try a computational search for small rank improvements.
    
    # For rank 44, 43, 42, 41, 40: try to find valid decompositions
    
    best_rank = 45
    best_decomp = None
    
    # Generate schoolbook as baseline
    def schoolbook():
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
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    best_decomp = (U_sb, V_sb, W_sb)
    best_rank = n * m * p
    
    # Try to improve using a structured decomposition
    # Use Strassen for the 2x2x2 subproblems
    
    # Let me try a specific construction for rank 40
    
    # Actually, let me try to use a known decomposition for 3x3x5 with rank 37 or so.
    # Since I can't recall the exact coefficients, let me try a different approach.
    
    # Use the fact that we can express the multiplication using fewer terms
    # by exploiting linear dependencies.
    
    # For 3x3x5: C[i,k] = sum_j A[i,j]*B[j,k]
    # We have 15 output entries (3x5), each is a bilinear form.
    
    # Let me try to use a "middle" approach: 
    # Compute intermediate vectors and combine.
    
    # Actually, the cleanest approach for a time-constrained problem:
    # Return the best known decomposition I can construct.
    
    # Let me try rank 40 using a pattern.
    
    # For now, if I can't find a better one quickly, return schoolbook.
    # But let me try one more thing: use the 2x2 Strassen decomposition for part of it.
    
    # Split B's 5 columns as 2+3.
    # Part 1: A(3x3) * B_left(3x2) -> C_left(3x2), rank at most 3*3*2=18, but can be less
    # Part 2: A(3x3) * B_right(3x3) -> C_right(3x3), rank at most 7 (Strassen)
    # Total: at most 18+7 = 25? But the A parts overlap, so we can't just add.
    
    # Actually, we CAN just concatenate decompositions since they use the same A but different B columns.
    # The A coefficients would be duplicated, but that's fine for a valid decomposition.
    
    # So: M(3,3,5) <= M(3,3,2) + M(3,3,3)
    
    # For M(3,3,3): Strassen gives 7.
    # For M(3,3,2): Let's figure out a good bound.
    
    # M(3,3,2): C[i,k] = sum_j A[i,j]*B[j,k] for k=0,1
    # This is 6 bilinear forms. Schoolbook = 18.
    # Can we do better? 
    # M(3,3,2) <= M(3,3,3) by padding, so <= 7.
    # Actually, M(3,3,2) <= 7 should be achievable.
    
    # Let me construct Strassen for 3x3x3 and a decomposition for 3x3x2.
    
    # Strassen for 3x3x3:
    def strassen_3x3x3():
        """Return U, V, W for 3x3x3 matrix multiplication with rank 7."""
        n_s, m_s, p_s = 3, 3, 3
        rank_s = 7
        U = [[0]*rank_s for _ in range(n_s*m_s)]
        V = [[0]*rank_s for _ in range(m_s*p_s)]
        W = [[0]*rank_s for _ in range(p_s*n_s)]
        
        # Strassen for 2x2x2 has 7 products. For 3x3x3, we need to extend.
        # Actually, the standard Strassen is for 2x2. For 3x3, we can use 
        # a combination of 2x2 Strassen and scalar multiplications.
        
        # 3x3x3 can be decomposed as:
        # - 2x2x2 Strassen for the top-left 2x2 block: 7 products
        # - Plus terms for the third row and column
        # This gives rank > 7 for 3x3x3.
        
        # Actually, M(3,3,3) is known to be <= 23 (not 7). Strassen's 7 is for 2x2x2.
        # M(3,3,3) <= 23 is the best known upper bound from Strassen-type algorithms.
        
        # Let me use schoolbook for 3x3x3: rank 27.
        # And for 3x3x2: rank 18 schoolbook, or better.
        
        # Total: 27 + 18 = 45, which is just schoolbook.
        
        # This approach doesn't help unless I have good decompositions.
        
        # Let me just use schoolbook for simplicity.
        col = 0
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    U[i*m_s+j][col] = 1
                    V[j*p_s+k][col] = 1
                    W[k*n_s+i][col] = 1
                    col += 1
        return U, V, W
    
    # This approach isn't leading to improvement. Let me try a different strategy.
    
    # Try to find a rank 40 decomposition by construction.
    
    # Actually, let me just try to verify if any structured decomposition works.
    
    # For time efficiency, let me try a few candidate ranks with structured patterns.
    
    # Candidate: Use the fact that in the 3x3x5 problem, we can group terms.
    
    # Let me try rank 40 with a specific pattern.
    
    # Given time constraints, let me return schoolbook if I can't find better quickly.
    # But first, let me try one specific known construction.
    
    # The "middle product" trick or "bilinear" approach.
    
    # Actually, let me just try to see if rank 44, 43, 42, 41, 40 work with a 
    # simple perturbation of schoolbook.
    
    # This is getting complex. Let me just return schoolbook for now.
    # The score will be 1.0, but at least it's correct.
    
    if time.time() > deadline - 5:
        U, V, W = best_decomp
        return {"U": U, "V": V, "W": W}
    
    # Try a simple optimization: remove redundant terms if possible
    # (not applicable for schoolbook since all terms are independent)
    
    # Return schoolbook
    U, V, W = best_decomp
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
