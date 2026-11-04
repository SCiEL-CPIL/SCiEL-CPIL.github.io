"""Research program: Attempts rank-14 construction via tensor network contraction, falls back to schoolbook."""
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor T is of shape (n, m, p)
    # T[i, j, k] = 1 for all i, j, k (since C[i,k] = sum_j A[i,j] * B[j,k])
    # We seek a decomposition of rank R such that:
    # T[i, j, k] = sum_{c=0}^{R-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # The directive suggests using the rank-7 decomposition for 3x3 matrix multiplication.
    # Let's define the known rank-7 decomposition for 3x3 matrix multiplication.
    # The tensor for 3x3 multiplication is T_3x3 of shape (3, 3, 3) with all entries 1.
    # A known rank-7 decomposition (from Bini et al. or similar sources) can be represented.
    # However, finding the exact integer/half-integer coefficients for a rank-7 3x3 decomposition
    # is non-trivial to hardcode from memory without reference.
    
    # Instead, let's try a direct approach for the 2x3x3 tensor.
    # The tensor is T[i, j, k] = 1 for i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    
    # Strategy: 
    # 1. Try to find a low-rank decomposition using optimization or known patterns.
    # 2. Since we need exact integer/half-integer entries, we can try small ranks and solve for the factors.
    
    # Let's try ranks from 10 up to 18.
    # For a given rank R, we have:
    # U: (6, R), V: (9, R), W: (6, R)
    # We need U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] to sum to 1 for each (i,j,k).
    
    # This is a system of equations. 
    # Let's try a specific structure: 
    # We can view the 2x3x3 tensor as two 3x3 tensors (one for i=0, one for i=1).
    # T[0, j, k] = 1 and T[1, j, k] = 1.
    # If we have a rank-7 decomposition for 3x3, we could potentially use it twice?
    # But the indices are coupled.
    
    # Let's try to construct a rank-14 decomposition by duplicating a rank-7 3x3 decomposition.
    # Suppose we have a rank-7 decomposition for 3x3:
    # T_3x3[j, k, l] = sum_{c=0}^{6} U3[j, c] * V3[k, c] * W3[l, c]
    # Wait, the indices for 3x3 multiplication are (row, mid, col).
    # So T_3x3[r, m, c] = 1.
    # Decomposition: T_3x3[r, m, c] = sum_{t=0}^{6} U3[r, m, t] * V3[m, c, t] * W3[c, r, t]? No, the standard form is:
    # T_3x3[r, m, c] = sum_t U3[r*m+m, t] * V3[m*c+c, t] * W3[c*3+r, t]
    
    # Let's use a known rank-7 decomposition for 3x3. 
    # One such decomposition uses the following factors (scaled to integers if possible):
    # This is complex to derive from scratch. Let's try a different approach.
    
    # We can try to solve for a rank-R decomposition using linear algebra.
    # Fix the factors U and V, then solve for W.
    # For each (i, k), W[k*n+i, :] is a vector of length R.
    # The condition is: sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j? No.
    # The condition is: for each (i, j, k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
    
    # Let's group by (i, k). For a fixed (i, k), we have 3 equations (for j=0,1,2):
    # sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1 for j=0,1,2.
    # This is a linear system: M_{ik} * W_{ik} = 1, where M_{ik} is a 3xR matrix.
    # M_{ik}[j, c] = U[i*m+j, c] * V[j*p+k, c].
    
    # If we choose U and V such that M_{ik} has full row rank (rank 3), then we can solve for W_{ik}.
    # We need W to have integer/half-integer entries.
    
    # Let's try to find U and V such that the resulting W has simple entries.
    # A simple choice: Let U and V be identity-like matrices.
    
    # Try rank 18 (schoolbook).
    # U[i*m+j, c] = 1 if c == i*m+j, else 0.
    # V[j*p+k, c] = 1 if c == j*p+k, else 0.
    # Then M_{ik}[j, c] = 1 if c == i*m+j and c == j*p+k. This requires i*m+j == j*p+k.
    # This doesn't work directly.
    
    # Let's try a different construction.
    # For each (i, k), we want to represent the vector of ones (length 3) as a linear combination of the rows of M_{ik}.
    # If we set U[i*m+j, c] = delta_{j, c%3} and V[j*p+k, c] = delta_{k, c%3}? No.
    
    # Let's try to use the fact that T is all ones.
    # T[i, j, k] = 1.
    # We can write 1 = sum_{a=0}^{2} delta_{j,a} * delta_{k,a} * delta_{i,0} + ... No.
    
    # Let's try rank 12.
    # For each i (0,1), we have a 3x3 tensor of ones.
    # A 3x3 tensor of ones can be decomposed into rank 3?
    # T[j, k] = 1. This is a matrix, not a tensor. But in the multiplication context, it's part of a 3-index tensor.
    # Actually, for fixed i, T[i, j, k] = 1. This is a 3x3 matrix of ones.
    # The rank of a 3x3 matrix of ones is 1.
    # So for each i, we can use 1 term? But the terms are coupled through the summation over c.
    
    # Let's try rank 6.
    # For i=0, use c=0,1,2. For i=1, use c=3,4,5.
    # U[0*m+j, 0] = 1, U[0*m+j, 1] = 0, ... No, we need to match j.
    
    # Let's try this:
    # For c in 0..2:
    #   U[0*m+j, c] = delta_{j,c}
    #   V[j*p+0, c] = 1  (for k=0)
    #   W[0*n+0, c] = 1
    # This only covers k=0.
    
    # We need to cover all k.
    
    # Let's try rank 18 again but with a different structure.
    # For each (i, j, k), we have a term.
    # This is the schoolbook solution. Rank 18.
    
    # Can we do better?
    # The tensor is T[i, j, k] = 1.
    # This is the tensor for multiplying a 2x3 matrix by a 3x3 matrix.
    # The border rank of this tensor is likely 18, but the rank might be lower.
    
    # Let's try to find a rank-14 decomposition.
    # We can use the following identity:
    # 1 = 1*1*1
    # We can group the terms.
    
    # Let's try to solve for rank 14 using a heuristic.
    # We'll initialize U and V with random values, then solve for W, and check if W has small integer/half-integer entries.
    
    import random
    random.seed(42)
    
    for rank in range(10, 19):
        # Try to find a decomposition of this rank
        best_W = None
        for attempt in range(100):
            # Initialize U and V with small random integers
            U = np.random.randint(-2, 3, size=(n*m, rank))
            V = np.random.randint(-2, 3, size=(m*p, rank))
            
            # Solve for W
            W = np.zeros((p*n, rank))
            valid = True
            for i in range(n):
                for k in range(p):
                    # Build M_{ik}
                    M = np.zeros((m, rank))
                    for j in range(m):
                        for c in range(rank):
                            M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    
                    # Solve M * W_{ik} = [1, 1, 1]
                    try:
                        # Use least squares if rank is insufficient
                        sol, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        # Check if the solution is close to integer/half-integer
                        # And if the residuals are small
                        if np.allclose(M @ sol, np.ones(m), atol=1e-6):
                            # Check if sol is close to integer or half-integer
                            sol_rounded = np.round(sol * 2) / 2
                            if np.allclose(sol, sol_rounded, atol=1e-6):
                                W[k*n+i, :] = sol_rounded
                            else:
                                valid = False
                                break
                        else:
                            valid = False
                            break
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                # Verify the full decomposition
                T_recon = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_recon[i, j, k] = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                
                if np.allclose(T_recon, np.ones((n, m, p)), atol=1e-6):
                    # Found a valid decomposition
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Schoolbook decomposition of rank n*m*p
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col_idx] = 1
                V[j * p + k][col_idx] = 1
                W[k * n + i][col_idx] = 1
                col_idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
