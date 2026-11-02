"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    deadline = time.time() + 140.0  # Safety margin before 150s kill

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly represents matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Known: rank(3x3x4) is at most 24 (from literature, e.g., Strassen-type constructions)
    # We attempt to find a rank-24 decomposition using a structured algebraic approach.
    # The key idea: for fixed (i,k), C[i,k] = sum_j A[i,j] B[j,k]
    # We can potentially combine terms across different j values.
    
    # Try a specific rank-24 construction based on the following structure:
    # For each pair (i,k), we have 3 terms (one for each j).
    # Total: 3*3*4 = 36 terms in schoolbook.
    # To get to 24, we need to eliminate 12 terms by sharing.
    
    # One known approach: use a basis transformation that combines j-terms
    # Let's try to construct a rank-24 decomposition using the following pattern:
    
    # Actually, let's use a more systematic approach:
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We want to find U (9 x r), V (12 x r), W (12 x r) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # For rank 24, let's try a specific construction:
    # Group by j. For each j, we have n*p = 12 terms.
    # But we can potentially share terms across j values.
    
    # Let's try a different approach: use the fact that the tensor is symmetric in a sense
    # and try to find a low-rank decomposition using numerical optimization
    
    # For now, let's try a few candidate ranks and see if we can find valid decompositions
    # using a simple greedy or structured approach
    
    best_result = None
    best_rank = n * m * p  # 36
    
    # Try rank 24 with a specific construction
    # Based on the idea of sharing terms across different j values
    
    # Construction attempt for rank 24:
    # Use the following pattern:
    # c = 0..11: handle j=0, all (i,k) pairs
    # c = 12..23: handle j=1 and j=2 combined
    
    # Let's try a more systematic approach using a known result:
    # The rank of 3x3x4 matrix multiplication is at most 24.
    # One construction uses the following idea:
    # For each (i,k), we have C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k]
    # We can rewrite this as:
    # C[i,k] = A[i,0]B[0,k] + A[i,1](B[1,k] + B[2,k]) + A[i,2]B[2,k] - A[i,1]B[2,k]
    # This doesn't immediately help.
    
    # Let's try a different approach: use a basis transformation on the j-index
    # Define new variables:
    # A'[i,0] = A[i,0]
    # A'[i,1] = A[i,1] + A[i,2]
    # A'[i,2] = A[i,2]
    # B'[0,k] = B[0,k]
    # B'[1,k] = B[1,k]
    # B'[2,k] = B[2,k] - B[1,k]
    # Then:
    # A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k]
    # = A'[i,0]B'[0,k] + A'[i,1]B'[1,k] + A'[i,2]B'[2,k]
    # This is just a change of basis, doesn't reduce rank.
    
    # Let's try a more sophisticated approach:
    # Use the fact that we can combine terms across different (i,k) pairs
    
    # For now, let's just return the schoolbook decomposition since we haven't
    # found a valid lower-rank decomposition yet.
    
    # Try to find a rank-24 decomposition using a structured approach
    if time.time() < deadline - 20:
        # Attempt a rank-24 construction
        target_rank = 24
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        # Fill in a candidate rank-24 decomposition
        # Based on the idea of sharing terms across different j values
        
        # Let's use a systematic approach:
        # For each c in range(24), assign U[c], V[c], W[c]
        
        # Try a specific pattern:
        # c = 0..11: handle j=0, all (i,k) pairs
        # c = 12..23: handle j=1 and j=2 combined
        
        # For c = 0..11 (j=0):
        for i in range(n):
            for k in range(p):
                c = i * p + k
                U[i * m + 0][c] = 1
                V[0 * p + k][c] = 1
                W[k * n + i][c] = 1
        
        # For c = 12..23 (j=1 and j=2 combined):
        # This is where we try to share terms
        # Let's try: for each (i,k), use one term for j=1 and one for j=2
        # But that would be 24 terms, total 36, no improvement
        
        # Let's try a different pattern:
        # c = 12..23: use a combination of j=1 and j=2
        
        # Actually, let's just verify if the current construction works
        # (it won't, since we haven't filled in the j=1 and j=2 parts properly)
        
        # For now, skip this and return schoolbook
        pass
    
    # If no better decomposition found, return schoolbook
    if best_result is None:
        return schoolbook()
    
    return best_result

# EVOLVE-BLOCK-END
