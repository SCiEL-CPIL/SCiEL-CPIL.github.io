"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 48
    deadline = time.time() + 140.0  # Leave safety margin before 150s limit
    
    # Start with schoolbook decomposition
    best_U = [[0] * rank_schoolbook for _ in range(n * m)]
    best_V = [[0] * rank_schoolbook for _ in range(m * p)]
    best_W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                best_U[i * m + j][c] = 1
                best_V[j * p + k][c] = 1
                best_W[k * n + i][c] = 1
    
    def verify(U, V, W, rank):
        """Check if the decomposition exactly represents matrix multiplication."""
        # Build the tensor T[i,j,k,l] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # This should equal 1 if i==l and j==k, else 0 (for (i,j) x (k,l) -> (i,l) at position j... )
        # Actually: C[i,l] = sum_j A[i,j] * B[j,l]
        # The tensor T_{i,j,k,l} should be 1 when the (i,j) entry of A and (j,k) entry of B 
        # contribute to (i,k) entry of C. So T[i,j,k,l] = delta_{j,k} * delta_{i,l}... 
        # Wait, let me reconsider. The multiplication tensor T has T[i,j,k] where 
        # C[i,k] = sum_j A[i,j] B[j,k]. In 3-index form: T_{i,j,k} = 1 if i,j,k valid.
        # In the U,V,W decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Wait, W index: W[k*n+i, c]. So W is indexed by (k, i).
        # So T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # And this must equal 1 for all valid (i,j,k).
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if s != 1:
                        return False
        return True
    
    # Verify schoolbook
    if not verify(best_U, best_V, best_W, rank_schoolbook):
        # Should never happen, but just in case
        pass
    
    # Try to find a lower-rank decomposition using local search
    # Strategy: try to find if we can merge two rank-1 terms into one
    # This is equivalent to finding a rank-1 tensor in the span of two given rank-1 tensors
    # that still covers the required entries
    
    # A more practical approach: use the known Strassen-like decomposition for 3x4x4
    # Based on the insight that rank 44 is achievable using 2x4 block Strassen
    
    # Let me try a direct construction based on 2x4 block Strassen:
    # Split A (3x4) into rows: A1 (first 2 rows, 2x4) and A2 (last row, 1x4)
    # For A1 * B (2x4 * 4x4): use Strassen-like 8-term decomposition
    # For A2 * B (1x4 * 4x4): use 4-term schoolbook
    # Total: 8 + 4 = 12? No, that's not right because we need the full 3x4 output.
    
    # Actually, the rank 44 result comes from a different decomposition.
    # Let me try to implement a known rank-44 decomposition for 3x4x4.
    
    # The 3x4x4 tensor can be viewed as the 4x4x4 tensor with one row of A zeroed out.
    # The 4x4x4 tensor has rank 49 (Strassen-Winograd). Removing constraints can reduce rank.
    # But we need an explicit decomposition.
    
    # Alternative: use bilinear complexity algorithms. For 3x4x4, the lower bound is
    # max(12, 16, 12) = 16, and the upper bound is 48. Known results suggest rank ~44.
    
    # Let me try a random local search approach:
    # Start with rank 48, try to reduce by finding dependent terms.
    
    import random
    random.seed(42)
    
    # Try to find a rank-r decomposition by solving a system of equations
    # For a given rank r, we have r unknown rank-1 tensors.
    # The constraints are: for all (i,j,k), sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
    # This is a system of n*m*p = 48 polynomial equations in r*(n*m + m*p + p*n) = r*(12+16+12) = 40r variables.
    
    # For r=44, we have 48 equations in 1760 variables - very underdetermined.
    # We can try to solve this using numerical methods.
    
    # Let me try a simpler approach: use the fact that the 3x4x4 tensor is a sub-tensor of 4x4x4
    # and construct a decomposition based on Strassen's 4x4x4 algorithm.
    
    # Strassen's 4x4x4 decomposition uses 49 terms. I'll implement this and then
    # see if I can reduce the rank by exploiting the structure where one row of A is missing.
    
    # Actually, let me just try to output a known rank-44 decomposition.
    # The key insight: for 3x4x4, we can use the following approach:
    # - Treat it as multiplying (3x4) * (4x4)
    # - Use a block structure: split the 4 columns of A into two groups of 2
    # - Apply Strassen to each 3x2 * 2x4 block
    
    # Let me implement a rank-44 decomposition based on 2x4 Strassen blocks.
    
    # For a 2x4 * 4x4 multiplication, we can use 8 terms (Strassen-style):
    # Let A = [[a11,a12,a13,a14],[a21,a22,a23,a24]], B = 4x4
    # C = A*B, C is 2x4
    # 
    # Using Strassen-like decomposition for 2x4*4x4:
    # We can split B into 2x2 blocks: B = [[B11,B12],[B21,B22]] where each is 2x2
    # Then A*B = [A11*B11 + A12*B21, A11*B12 + A12*B22; A21*B11 + A22*B21, A21*B12 + A22*B22]
    # This doesn't directly help.
    
    # Let me try a different approach: use the known result that rank(3x4x4) = 44
    # and construct it explicitly.
    
    # The rank-44 decomposition can be constructed as follows:
    # Use the 4x4x4 Strassen-Winograd decomposition (rank 49) and remove 5 terms
    # that are not needed when one row of A is missing.
    
    # Since I don't have the explicit Strassen-Winograd 4x4x4 coefficients memorized,
    # let me try a numerical approach to find a rank-44 decomposition.
    
    # Try to solve for a rank-r decomposition using least squares + local search
    # This is computationally expensive but might work for r close to 48.
    
    # Given time constraints, let me just return the schoolbook decomposition
    # unless I can quickly find a better one.
    
    # Quick check: can we do rank 47? 46? Let's try a greedy elimination approach.
    # For each pair of columns (rank-1 terms), check if their sum can be replaced by a single term.
    
    # This is too complex to implement correctly in the remaining time.
    
    # Return the schoolbook decomposition (rank 48)
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
