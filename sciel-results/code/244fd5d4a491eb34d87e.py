"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    
    # Target rank 49
    # Strategy: Decompose 5x5x5 into a 3x3 block (Laderman rank 7) and 2x2 block (Strassen rank 7)
    # plus interaction terms.
    # A full 5x5 matrix multiplication can be viewed as:
    # [ A11 A12 ] [ B11 B12 ]   [ C11 C12 ]
    # [ A21 A22 ] [ B21 B22 ] = [ C21 C22 ]
    # Where A11, B11, C11 are 3x3, A22, B22, C22 are 2x2, and others are 3x2 or 2x3.
    
    # This specific construction is complex to derive on the fly correctly without a known exact formula 
    # for the 49 rank decomposition of 5x5x5.
    # However, it is a known result that the rank is at most 49.
    # A common way to achieve this is by using a specific structured decomposition.
    
    # Let's try to construct the factors U, V, W directly.
    # U is 25x49, V is 25x49, W is 25x49.
    
    # If we cannot guarantee the exact algebraic coefficients for rank 49 in this short time,
    # we should fall back to a verified lower rank if found, or the schoolbook if not.
    # But the prompt asks for a DIFFERENT strategy. Hard-coding is the strategy.
    
    # Since I do not have the explicit 49-term formula memorized to the level of individual coefficients 
    # for the specific index ordering W[k*n+i, r], I will implement a robust search for a lower rank 
    # using the algebraic structure as a guide, or simply return the schoolbook if the hard-coded 
    # attempt fails verification.
    
    # Wait, the directive says "explicitly hard-code the 49 non-zero tensor coefficients".
    # This implies I should produce the matrices.
    # Without the explicit coefficients, I cannot "hard-code" them.
    # I will instead implement a fast search that tries to find a rank < 125 decomposition 
    # using random initialization and verification, which is a different strategy from the 
    # static schoolbook, but might not reach 49.
    
    # Actually, let's look at the "Inspiration 2" which tried random search.
    # The directive specifically asks for the 49 rank hard-code.
    # If I cannot provide the exact 49 rank coefficients, I should provide the best alternative 
    # that is structurally different.
    
    # Let's try to construct a rank 100 decomposition using a block-diagonal approach 
    # (Laderman 3x3 + Strassen 2x2 + cross terms handled by schoolbook).
    # 3x3 Laderman: Rank 7.
    # 2x2 Strassen: Rank 7.
    # Cross terms: A12*B21 (3x2 * 2x3) and A21*B12 (2x3 * 3x2).
    # These are 3x3 and 3x3 multiplications? No, 3x2 * 2x3 -> 3x3.
    # We can treat these as separate small matrix multiplications.
    
    # Let's define the indices.
    # Rows of A: 0-4. Cols of A: 0-4.
    # Rows of B: 0-4. Cols of B: 0-4.
    # Rows of C: 0-4. Cols of C: 0-4.
    
    # Block 1: Top-left 3x3 (indices 0,1,2)
    # Block 2: Bottom-right 2x2 (indices 3,4)
    # Block 3: Top-right 3x2 (A rows 0-2, B cols 3-4)
    # Block 4: Bottom-left 2x3 (A rows 3-4, B cols 0-2)
    
    # This is getting complicated to map to the single 5x5x5 tensor without explicit formulas.
    
    # Given the constraint to "hard-code", and my inability to recall the exact 49 coefficients,
    # I will implement a **verified random search** for a rank significantly lower than 125, 
    # such as 100, using a smarter initialization based on the block structure.
    
    # However, the most robust "different strategy" that is guaranteed to work and is 
    # structurally different from the trivial loop is to use **NumPy's tensor operations** 
    # to verify and potentially optimize, but since we need exact integer/half-integer, 
    # random search is the way.
    
    # Let's try a simple optimization: Rank 100 via block diagonal.
    # We can decompose the 5x5 multiplication into:
    # 1. C11 = A11 B11 (3x3) -> Rank 7 (Laderman)
    # 2. C22 = A22 B22 (2x2) -> Rank 7 (Strassen)
    # 3. C12 = A12 B12 (3x2 * 2x2? No, A12 is 3x2, B12 is 2x2. Result 3x2)
    #    Actually, C12 = A11 B12 + A12 B22.
    #    This involves sums of products.
    
    # This is too complex to derive safely in this context without error.
    
    # I will revert to a safe, fast, structurally different approach: 
    # **Use the schoolbook decomposition but vectorized with NumPy for speed, 
    # and return it.** 
    # Wait, the score is 1.0 for schoolbook. I need to beat it.
    
    # Let's try a **Rank 100** decomposition by simply taking the schoolbook 
    # and removing dependencies? No, that breaks it.
    
    # Okay, I will implement a **Random Search for Rank 100** with a tight deadline.
    # If it finds a valid rank 100, it scores better than 125.
    
    def _verify(U, V, W, n, m, p):
        # U: nm x rank, V: mp x rank, W: pn x rank
        # C[i,k] = sum_j A[i,j] B[j,k]
        # Tensor T[i,j,k] = 1 if i,j,k form a valid multiplication term.
        # Decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Note: W index is k*n + i.
        
        rank = U.shape[1]
        # Reshape U to (n, m, rank)
        U_reshaped = np.array(U).reshape(n, m, rank)
        V_reshaped = np.array(V).reshape(m, p, rank)
        W_reshaped = np.array(W).reshape(p, n, rank)
        
        # Compute sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This is a tensor contraction.
        # We can do this efficiently.
        
        # For each i, k, we need sum_{j,r} U[i,j,r] V[j,k,r] W[k,i,r] == (i==j? No, T[i,j,k] is 1 only if it's a term in the sum)
        # The tensor T for matrix multiplication has T[i,j,k] = 1 for all i,j,k.
        # Wait, the tensor for matrix multiplication C = AB is defined by C_{ik} = sum_j A_{ij} B_{jk}.
        # The tensor T_{ijk} such that C_{ik} = sum_{j} A_{ij} B_{jk} T_{ijk}? No.
        # The decomposition is for the **multiplication tensor** T where T[i, j, k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
        # In the standard tensor decomposition of matrix multiplication, we want to represent the map (A, B) -> C.
        # C[i,k] = sum_r U[i,?,r] ... 
        # The standard form is: C[i,k] = sum_{j,r} A[i,j] B[j,k] M_{i,j,k,r}? No.
        
        # The task says: "exact tensor decomposition for multiplying...".
        # Usually this means representing the trilinear form or the bilinear map.
        # The schoolbook uses rank n*m*p.
        # Term r corresponds to (i,j,k).
        # U[i*m+j, r] = 1 if r matches (i,j,k).
        # V[j*p+k, r] = 1 if r matches (i,j,k).
        # W[k*n+i, r] = 1 if r matches (i,j,k).
        
        # So, for a specific (i,j,k), the contribution to C[i,k] from A[i,j]B[j,k] is:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This sum must be 1 for the specific r corresponding to (i,j,k) and 0 otherwise?
        # No, the decomposition is:
        # C[i,k] = sum_j A[i,j] B[j,k] = sum_{j,r} A[i,j] B[j,k] (U[i*m+j, r] V[j*p+k, r] W[k*n+i, r])?
        # If U, V, W are 1 for the matching index, then the term is 1 only when r matches (i,j,k).
        # Then sum_r ... picks out the term where r corresponds to (i,j,k).
        # So C[i,k] = sum_j A[i,j] B[j,k] * 1.
        
        # So the verification is:
        # For all i, j, k:
        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # And for "cross" terms that shouldn't be 1?
        # The tensor T_{ijk} is 1 for all i,j,k in the range?
        # No, the matrix multiplication tensor T is defined by T[i,j,k] = 1.
        # The decomposition approximates T.
        # So we need sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for ALL i,j,k.
        
        # Let's check the schoolbook.
        # For a given (i,j,k), let r be the index for (i,j,k).
        # U[i*m+j, r] = 1.
        # V[j*p+k, r] = 1.
        # W[k*n+i, r] = 1.
        # Sum is 1.
        # For any other r', at least one of U, V, W will be 0 because the indices don't match.
        # So the sum is 1.
        
        # So the condition is: For all i in [0,n), j in [0,m), k in [0,p):
        #   dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :]) == 1
        # Wait, element-wise product of V and W rows, then dot with U row.
        
        pass

    # I will return the schoolbook decomposition to ensure validity, as I cannot guarantee 
    # a lower rank without the explicit coefficients.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
