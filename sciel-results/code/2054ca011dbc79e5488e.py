"""Matrix multiplication 3x3 by 3x7 - rank minimization."""
import time
import random
import itertools
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # The tensor T for matrix multiplication: T[i,j,k] = 1
    # We want rank-r decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # 
    # Actually, the correct condition for matrix multiplication is:
    # For all i,k: C[i,k] = sum_j A[i,j]*B[j,k]
    # = sum_c (sum_{j1} U[i*m+j1,c]*A[i,j1]) * (sum_{j2} V[j2*p+k,c]*B[j2,k]) * W[k*n+i,c]
    #
    # This must equal sum_j A[i,j]*B[j,k] for all A,B
    # So we need: sum_c U[i*m+j1,c] * V[j2*p+k,c] * W[k*n+i,c] = delta_{j1,j2} for all i,k,j1,j2
    #
    # Wait, that's not right. Let me reconsider.
    #
    # The bilinear algorithm: M_c = (sum_{i,j} u_{i,j,c} A[i,j]) * (sum_{j,k} v_{j,k,c} B[j,k])
    # C[i,k] = sum_c w_{i,k,c} M_c
    #
    # For this to compute C=AB, we need:
    # sum_j A[i,j]B[j,k] = sum_c w_{i,k,c} (sum_{j1} u_{i,j1,c} A[i,j1]) (sum_{j2} v_{j2,k,c} B[j2,k])
    #
    # Matching coefficients: sum_c w_{i,k,c} u_{i,j1,c} v_{j2,k,c} = delta_{j1,j2}
    #
    # In vectorized form: U[i*m+j, c] = u_{i,j,c}, V[j*p+k, c] = v_{j,k,c}, W[k*n+i, c] = w_{i,k,c}
    #
    # So the condition is: for all i,k,j1,j2:
    # sum_c U[i*m+j1, c] * V[j2*p+k, c] * W[k*n+i, c] = delta_{j1,j2}
    
    def verify(U, V, W, rank):
        """Check if the decomposition is valid."""
        # For all i, k, j1, j2: sum_c U[i*m+j1,c] * V[j2*p+k,c] * W[k*n+i,c] = delta_{j1,j2}
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        expected = 1.0 if j1 == j2 else 0.0
                        actual = 0.0
                        for c in range(rank):
                            actual += U[i * m + j1, c] * V[j2 * p + k, c] * W[k * n + i, c]
                        if abs(actual - expected) > 1e-6:
                            return False
        return True
    
    # Try to find lower rank decompositions
    # Strategy: start with schoolbook and try to merge columns
    
    # For each pair of columns, check if they can be merged
    # Two columns c1, c2 can be merged if there exist scalars alpha, beta such that
    # the new column (alpha*c1 + beta*c2) still allows a valid decomposition
    
    # Simpler approach: try random perturbations of the schoolbook decomposition
    # and check if they're still valid
    
    # Actually, let's try a more systematic approach:
    # For 3x3x7, try ranks from 62 down to some minimum
    
    # First, let's try to see if we can reduce by 1
    # The idea: if two columns have the same U pattern or same V pattern, we might merge them
    
    # Let's try a different approach: use the fact that the tensor has structure
    # T[i,j,k] = 1 for all i,j,k (it's a constant tensor in this representation)
    # Wait no, that's not right.
    
    # Let me reconsider the problem. The matrix multiplication tensor for n x m times m x p
    # has components T[i,j,k] where the multiplication is C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # In the tensor decomposition framework, we're decomposing the trilinear form:
    # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C^*[i,k] = sum_{i,j} A[i,j] B[j,k] C^*[i,k] (with appropriate indices)
    #
    # Actually, I think the correct tensor is: T[i,j,k] = delta_{i,i} * delta_{j,j} * delta_{k,k} = 1
    # No wait, that's not right either.
    
    # Let me just trust the verification function and try to find better decompositions.
    
    # Try rank 62: can we merge any two columns?
    # For a merge to work, we need the sum of two rank-1 tensors to be rank-1
    # This happens when the three vectors are linearly dependent in a specific way
    
    # Let's try a computational search: for each pair of columns, try to see if
    # they can be combined into one column while maintaining validity
    
    # This is complex. Let's try a simpler heuristic:
    # Randomly perturb the schoolbook decomposition and check validity
    
    random.seed(42)
    
    # Try various rank values
    for target_rank in range(rank_school - 1, max(20, rank_school - 20), -1):
        if time.time() > deadline - 20:
            break
        
        # Try to find a valid decomposition of this rank
        # Strategy: start with schoolbook and try to eliminate columns
        
        # For each column, check if it's redundant given the others
        # A column c is redundant if T can be expressed without it
        # This is equivalent to checking if the column is in the span of the others
        # in the tensor space
        
        # Simpler: try random initializations and optimize
        
        if target_rank < best_rank:
            # Try a few random initializations
            for attempt in range(3):
                if time.time() > deadline - 15:
                    break
                
                # Initialize with random values
                U = np.random.randn(n * m, target_rank) * 0.1
                V = np.random.randn(m * p, target_rank) * 0.1
                W = np.random.randn(p * n, target_rank) * 0.1
                
                # Check if close to valid
                if verify(U, V, W, target_rank):
                    best_rank = target_rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    break
    
    # If we found a better rank, return it; otherwise return schoolbook
    if best_rank < rank_school:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
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
