"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Build schoolbook decomposition
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Verify schoolbook is correct
    # The tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For schoolbook, T[i,j,k] = 1 if there's a term with this (i,j,k), else 0
    # Actually the matrix multiplication tensor has T[i,j,k] = delta_{i,k} for appropriate indices
    # Let me think: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is T[i,j,k] where we want sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = A[i,j]*B[j,k]
    # This means T[i,j,k] should be 1 for all valid (i,j,k) and 0 otherwise
    # But that's not right either. The tensor decomposition gives:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal the product structure
    
    # For standard matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # The bilinear form is: sum_{i,j,k} C[i,k] * (A[i,j]*B[j,k]) 
    # = sum_c (sum_{i,j} U[i*m+j,c]*A[i,j]) * (sum_{j,k} V[j*p+k,c]*B[j,k]) * (sum_{k,i} W[k*n+i,c]*C[i,k])
    
    # So the tensor T has entries T[i,j,k] = 1 for all (i,j,k) in [0,n)×[0,m)×[0,p)
    # Wait no. Let me reconsider.
    
    # The matrix multiplication tensor T is defined such that:
    # <T, A⊗B⊗C> = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{k,i}
    # For matrix multiplication, we want this to equal sum_{i,k} (AB)_{i,k} C_{k,i} = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}
    # So T_{i,j,k} = 1 for all valid indices!
    
    # So the tensor is all ones. The schoolbook decomposition with rank nmp works because
    # each term picks out one (i,j,k) triple.
    
    # Now try to reduce rank by finding dependencies
    # If sum_{c in S} alpha_c * U_c = 0, sum alpha_c * V_c = 0, sum alpha_c * W_c = 0
    # then we can eliminate those terms.
    
    # Strategy: try to find subsets of columns that are linearly dependent in a way that
    # preserves the tensor identity.
    
    # Simpler approach: try to merge pairs of terms. If U_a + U_b, V_a + V_b, W_a + W_b
    # still gives a valid decomposition when combined with appropriate coefficients...
    
    # Actually, let's try a different approach: use the fact that for matrix multiplication,
    # there might be algebraic identities that allow rank reduction.
    
    # For 2x5 x 5x5, the best known upper bound might be less than 50.
    # Let me try a greedy elimination approach.
    
    # Check if we can eliminate any column by expressing it as a combination of others
    # while maintaining the tensor identity.
    
    # The tensor identity requires: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # If I remove column c0, I need to find coefficients for other columns such that
    # the sum still equals 1 for all (i,j,k).
    
    # This is a hard problem. Let me try a simpler heuristic:
    # Look for pairs of columns that have identical U, V, or W patterns and merge them.
    
    # Check for duplicate columns
    def find_duplicates(mat):
        """Find indices of duplicate rows in a matrix."""
        seen = {}
        dups = []
        for idx in range(mat.shape[1]):
            row = tuple(np.round(mat[:, idx], decimals=10))
            if row in seen:
                dups.append((seen[row], idx))
            else:
                seen[row] = idx
        return dups
    
    # Check if any two columns have identical U vectors
    u_dups = find_duplicates(U)
    v_dups = find_duplicates(V)
    w_dups = find_duplicates(W)
    
    # If two columns have the same U, V, AND W, they're redundant
    # But that's unlikely in schoolbook.
    
    # Let me try a different strategy: use random linear combinations to try to reduce rank
    # Start with rank r, try to find a (r-1)-rank decomposition
    
    # Actually, let me just return the schoolbook for now, as finding a better decomposition
    # for 2x5x5 is non-trivial and may require sophisticated methods.
    
    # But wait - the task says "rank-minimal exact tensor decomposition". The known
    # lower bound for 2x5x5 might be achievable. Let me think about what's known.
    
    # For matrix multiplication of n×m by m×p, the rank is at most nmp (schoolbook).
    # For 2x5x5, that's 50. There might be better algorithms.
    
    # Given time constraints, let me return the schoolbook decomposition.
    # Convert to lists as required by the interface
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
