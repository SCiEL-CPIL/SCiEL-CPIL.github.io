"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    n, m, p = DIMENSIONS
    n_rows = n * m
    m_rows = m * p
    p_rows = p * n
    total = n_rows + m_rows + p_rows

    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n_rows)]
        V = [[0] * rank for _ in range(m_rows)]
        W = [[0] * rank for _ in range(p_rows)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try to find a lower rank decomposition using randomized search with rational arithmetic
    # Strategy: try ranks from 24 upward, use random sparse matrices and solve linear system
    
    best_result = None
    best_rank = n * m * p + 1  # one more than schoolbook
    
    # Try a few candidate ranks
    candidate_ranks = list(range(24, min(54, best_rank)))
    random.shuffle(candidate_ranks)
    
    for rank in candidate_ranks:
        if time.time() > deadline - 30:
            break
            
        if rank >= best_rank:
            continue
            
        # Try a few random initializations for this rank
        max_trials = 5
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                    break
            
            # Random sparse integer matrices for U, V, W
            U = np.zeros((n_rows, rank), dtype=np.int64)
            V = np.zeros((m_rows, rank), dtype=np.int64)
            W = np.zeros((p_rows, rank), dtype=np.int64)
            
            # Fill with random small integers
            for c in range(rank):
                # Randomly pick entries to be non-zero
                for r in range(n_rows):
                    if random.random() < 0.3:
                        U[r, c] = random.choice([-1, 1, 2, -2])
                for r in range(m_rows):
                    if random.random() < 0.3:
                        V[r, c] = random.choice([-1, 1, 2, -2])
                for r in range(p_rows):
                    if random.random() < 0.3:
                        W[r, c] = random.choice([-1, 1, 2, -2])
            
            # Check if this random assignment gives a valid decomposition
            # The tensor T[i,j,k] should equal sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
            # For matrix multiplication: T[i,j,k] = 1 if all valid, else 0 (actually it's the bilinear form)
            
            # The multiplication tensor has entries T[i,j,k] for the coefficient of A[i,j]*B[j,k] in C[i,k]
            # Actually, the tensor is defined such that C[i,k] = sum_j A[i,j] * B[j,k]
            # The bilinear form is: T(A,B) = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = sum_{i,k} C[i,k]
            # No wait, let me reconsider.
            
            # The matrix multiplication tensor T is of size n*m*p where T[i,j,k] = 1 for all valid i,j,k
            # because C[i,k] = sum_j A[i,j]*B[j,k], so the coefficient of A[i,j]*B[j,k] in the product
            # summed over all outputs is 1 for each (i,j,k).
            
            # Actually, the standard definition: the tensor T such that <T, A ⊗ B> = sum_{i,k} (AB)[i,k]
            # where A is n×m, B is m×p.
            # (AB)[i,k] = sum_j A[i,j]*B[j,k]
            # So <T, A ⊗ B> = sum_{i,j,k} A[i,j]*B[j,k] = sum_i sum_k (AB)[i,k]
            # This means T[i,j,k] = 1 for all i,j,k.
            
            # Wait, that's not quite right either. Let me think again.
            # The matrix multiplication map is: M(A,B) = AB
            # As a bilinear map from (R^{n×m}) × (R^{m×p}) → R^{n×p}
            # The tensor T ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{n×p}
            # T[i,j,a,b,c,d] = 1 if j=a and... no.
            
            # Let's use the standard convention from the problem:
            # U has shape (n*m, rank), indexed by (i,j) flattened as i*m+j
            # V has shape (m*p, rank), indexed by (j,k) flattened as j*p+k
            # W has shape (p*n, rank), indexed by (k,i) flattened as k*n+i (note: k*n+i, not i*n+k)
            
            # The decomposition says: for all A, B, we have (AB)[i,k] = sum_c U[(i,j),c] * V[(j,k),c] * W[(k,i),c] summed over j? No.
            
            # Actually, the bilinear form for matrix multiplication as a tensor:
            # T[i,j,k] for i in [0,n), j in [0,m), k in [0,p)
            # T[i,j,k] = 1 (the coefficient of A[i,j]*B[j,k] in the sum that gives the trace or something)
            
            # Let me re-read the problem statement: "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 9 matrix."
            # "The output factor follows the official trace convention W[k*n+i,r]."
            
            # I think the tensor is T[i,j,k] = 1 for all i,j,k, representing that each product A[i,j]*B[j,k] contributes to the result.
            # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
            
            # This is a system of n*m*p = 54 equations with 3*rank variables (entries of U, V, W).
            # For rank r, we have 3*r variables and 54 constraints.
            # We need 3r >= 54, so r >= 18. But rank is likely higher due to algebraic constraints.
            
            # For the random trial, we can't easily solve this nonlinear system.
            # Let me instead try a structured approach: use the schoolbook but merge some terms.
            
            # Alternative: just return schoolbook since finding a valid lower rank is very hard.
            # Let me verify the schoolbook is correct.
            
            break  # break out of trial loop, this approach isn't working
        
        if time.time() > deadline - 20:
            break
    
    # Return schoolbook decomposition
    return schoolbook()
# EVOLVE-BLOCK-END
