"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]
import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # Safety margin before 150s kill
    
    # Start with schoolbook as fallback
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize schoolbook
    U = [[0] * best_rank for _ in range(n * m)]
    V = [[0] * best_rank for _ in range(m * p)]
    W = [[0] * best_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower-rank decompositions
    # The matrix multiplication tensor T_{ijk} for (i,j,k) in [0,n)×[0,m)×[0,p)
    # A rank-r decomposition: T_{ijk} = sum_c U_{i*m+j,c} * V_{j*p+k,c} * W_{k*n+i,c}
    # Wait, that's not right. Let me reconsider.
    
    # Actually, for matrix multiplication C = A * B where A is n×m, B is m×p, C is n×p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # 
    # A rank-r bilinear algorithm: C[i,k] = sum_c U_{i,k,c} * (sum_j V_{j,c} * A[i,j]) * (sum_j W_{j,c} * B[j,k])
    # Wait, let me re-read the interface.
    
    # Interface: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # W[k*n+i, r] follows trace convention
    # 
    # The decomposition should satisfy: for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c] ... no that's not right either.
    
    # Let me think again. The matrix multiplication tensor T has entries T[i,j,k] for i in [0,n), j in [0,m), k in [0,p)
    # where T[i,j,k] represents the contribution to C[i,k] from A[i,j]*B[j,k].
    # Actually the standard tensor for matrix multiplication has indices (i, j, k) where:
    # - i indexes rows of A and C (size n)
    # - j indexes cols of A and rows of B (size m)  
    # - k indexes cols of B and C (size p)
    # And T[i,j,k] = 1 means this triple contributes.
    
    # A rank-r decomposition: T[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that would be a trilinear form, but matrix multiplication is bilinear.
    
    # Actually I think the convention here is different. Let me re-read:
    # "U factor matrix of shape (n*m, rank), V of shape (m*p, rank), W of shape (p*n, rank)"
    # "W[k*n+i,r] follows the official trace convention"
    
    # I think the correct interpretation is:
    # For matrix multiplication C = AB where A is n×m and B is m×p:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # A rank-r decomposition expresses each C[i,k] as:
    # C[i,k] = sum_{c=0}^{r-1} (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    # Wait no, that doesn't match the shapes.
    
    # Let me try another interpretation. Perhaps:
    # U[i*m+j, c] are coefficients for A[i,j] in the c-th bilinear form
    # V[j*p+k, c] are coefficients for B[j,k] in the c-th bilinear form  
    # W[k*n+i, c] is the coefficient for C[i,k] from the c-th bilinear form
    # 
    # Then: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    # This doesn't seem right either because the sums over j would be inside.
    
    # Actually, I think the standard tensor decomposition for matrix multiplication is:
    # The multiplication tensor T has T[i,j,k] = 1 for valid (i,j,k)
    # Rank-r decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But this is for the trilinear tensor, and matrix multiplication is bilinear...
    
    # I think I'm overcomplicating this. Let me just use the schoolbook and try to verify it works,
    # then attempt to find lower rank solutions using a search.
    
    # For now, let's just return the schoolbook if we don't find better in time.
    # But let me try a simple search for rank < 36.
    
    # Actually, let me implement a proper verification and search.
    
    def verify(U_mat, V_mat, W_mat, rank):
        """Verify that the decomposition is exact using exact arithmetic."""
        # The decomposition should satisfy:
        # For all i in [0,n), j in [0,m), k in [0,p):
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 if (i,j,k) is a valid triple, 0 otherwise
        # But wait, for matrix multiplication, every (i,j,k) with i<n, j<m, k<p is valid.
        # So we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
        
        # Hmm, but that's a trilinear identity. Let me check with schoolbook:
        # Schoolbook: for each (i,j,k), there's a column c where U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1
        # So for a fixed (i,j,k), only one c has all three non-zero, giving product 1. Correct.
        
        # So the condition is: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        
        # But wait, that can't be right for arbitrary matrices. Let me reconsider.
        
        # Actually, I think the correct condition for the bilinear algorithm is:
        # C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * A[i,j]) * (sum_j' V[j'*p+k,c] * B[j',k])
        # This should equal sum_j A[i,j]*B[j,k] for all A,B.
        
        # Expanding: C[i,k] = sum_c W[k*n+i,c] * sum_j U[i*m+j,c] A[i,j] * sum_j' V[j'*p+k,c] B[j',k]
        # = sum_j,j' A[i,j] B[j',k] * sum_c W[k*n+i,c] U[i*m+j,c] V[j'*p+k,c]
        
        # For this to equal sum_j A[i,j] B[j,k], we need:
        # sum_c W[k*n+i,c] U[i*m+j,c] V[j'*p+k,c] = delta_{j,j'} for all i,k,j,j'
        
        # So the verification condition is:
        # For all i in [0,n), k in [0,p), j in [0,m), j' in [0,m):
        # sum_c W[k*n+i,c] * U[i*m+j,c] * V[j'*p+k,c] = 1 if j==j' else 0
        
        n, m, p = DIMENSIONS
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for j2 in range(m):
                        total = Fraction(0)
                        for c in range(rank):
                            total += Fraction(W_mat[k*n+i][c]) * Fraction(U_mat[i*m+j][c]) * Fraction(V_mat[j2*p+k][c])
                        expected = 1 if j == j2 else 0
                        if total != expected:
                            return False
        return True
    
    # Try random searches for lower rank
    # Start from rank 35 down to some minimum
    min_rank_to_try = 20  # Reasonable lower bound
    
    for target_rank in range(best_rank - 1, max(min_rank_to_try, 0) - 1, -1):
        if time.time() > deadline - 30:  # Stop if running out of time
            break
        
        # Try multiple random initializations for this rank
        max_attempts = 50
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 30:
                break
            
            # Initialize with random small integers/half-integers
            U_try = [[random.choice([0, 0.5, 1, 1.5, 2, -0.5, -1]) for _ in range(target_rank)] for _ in range(n*m)]
            V_try = [[random.choice([0, 0.5, 1, 1.5, 2, -0.5, -1]) for _ in range(target_rank)] for _ in range(m*p)]
            W_try = [[random.choice([0, 0.5, 1, 1.5, 2, -0.5, -1]) for _ in range(target_rank)] for _ in range(p*n)]
            
            # Verify
            if verify(U_try, V_try, W_try, target_rank):
                best_rank = target_rank
                best_U = U_try
                best_V = V_try
                best_W = W_try
                found = True
                break
        
        if found:
            # Found a lower rank, continue searching for even lower
            continue
        else:
            # Didn't find for this rank, try lower rank (but this is expensive)
            # For now, break to save time
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
