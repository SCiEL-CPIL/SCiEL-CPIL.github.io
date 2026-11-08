"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x7 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    
    The multiplication tensor T[i, j, k] = e_{i,j} * e_{j,k} = e_{i,k} (in the appropriate subspace)
    can be decomposed into rank-1 tensors. The schoolbook algorithm uses rank n*m*p.
    
    For matrix multiplication C = A * B where A is n x m and B is m x p,
    C[i, k] = sum_j A[i, j] * B[j, k].
    
    The tensor is T[i, j, k] = delta_{i, k} * (something) ... no, that's not right.
    
    Actually, the multiplication tensor T has entries T[i, j, k] such that
    T[i, j, k] = 1 if we are computing C[i, k] from A[i, j] and B[j, k].
    
    The standard decomposition is:
    For each (i, j, k), we have a rank-1 term:
      U[i*m + j, c] = 1
      V[j*p + k, c] = 1  
      W[k*n + i, c] = 1
    
    This gives C[i, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over j.
    
    Wait, let me think about this more carefully.
    
    The matrix multiplication tensor T ∈ R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n} (or similar indexing).
    
    Actually, the standard formulation: We want to compute C = AB where A is n×m, B is m×p, C is n×p.
    
    The tensor T is defined such that T[i, j, k] represents the coefficient for A[i,j] * B[j,k] in computing C[i,k].
    
    So T[i, j, k] = 1 for all valid i, j, k (when j is the shared index).
    
    The rank-1 decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
    
    where u_c is in R^{n*m} (indexed by i*m+j), v_c is in R^{m*p} (indexed by j*p+k), w_c is in R^{p*n} (indexed by k*n+i).
    
    For the schoolbook algorithm, each (i,j,k) triple gives one term:
    u_c has 1 at position i*m+j
    v_c has 1 at position j*p+k
    w_c has 1 at position k*n+i
    
    This gives rank = n*m*p = 4*4*7 = 112.
    
    Can we do better? For matrix multiplication, the best known upper bounds are significantly less than n*m*p for large matrices, but for small matrices like 4x4x7, we need to check if any known decompositions are better.
    
    For 4x4x4, the best known rank is 49 (Laderman's algorithm is for 4x4x4 with rank 49? No, Laderman is for 4x4x4 with rank 49... actually Laderman's algorithm gives rank 49 for 4x4x4? Let me recall. Strassen's is for 2x2x2 with rank 7. For 4x4x4, the best known is around 49 or so? Actually, I think the best known for 4x4x4 is 49 from Laderman.
    
    But our case is 4x4x7, which is not square. Let me think about what the minimal rank could be.
    
    Actually, for the purposes of this task, the schoolbook decomposition with rank 112 is a valid exact decomposition. The question is whether we can find a lower rank exact decomposition.
    
    Given the complexity of finding optimal tensor decompositions, and the time limit, I'll start with the schoolbook decomposition which is guaranteed to be exact.
    
    Let me implement the schoolbook decomposition.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition: rank = n * m * p
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
