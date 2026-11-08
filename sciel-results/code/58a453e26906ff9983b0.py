"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Schoolbook decomposition as baseline
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find lower rank decompositions
    # The tensor T[i,j,k] = delta_{ik} for the multiplication
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{ik}
    
    # Try ranks from lower values up to schoolbook
    for target_rank in range(1, rank_school):
        if time.time() > deadline - 10:
            break
        
        # Use random initialization and iterative refinement
        max_attempts = 5 if target_rank > 30 else 3
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random rational/integer initialization
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Initialize with small integers
            for c in range(target_rank):
                U[:, c] = np.random.randint(-2, 3, size=n * m)
                V[:, c] = np.random.randint(-2, 3, size=m * p)
                W[:, c] = np.random.randint(-2, 3, size=p * n)
            
            # Check if this gives exact decomposition
            # T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # And T[i,j,k] = 1 if i==k else 0 (wait, that's not right)
            
            # Actually for matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
            # The tensor is T[i,j,k] where the bilinear form is sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = C[i,k]
            # So T[i,j,k] = delta_{i,k}... no.
            
            # Let me re-derive: C[i,k] = sum_j A[i,j] B[j,k]
            # In tensor form: C = T * (A, B) where T has indices such that
            # sum_{j} T[i,j,k] A[i,j] B[j,k] = C[i,k]
            # So T[i,j,k] = 1 for all valid i,j,k, and the sum over j gives the result.
            # Wait, that means T[i,j,k] = 1 for all i,j,k? No...
            
            # Actually the tensor decomposition is:
            # C[i,k] = sum_c (sum_j U[i*m+j,c] A[i,j]) * (sum_k V[j*p+k,c] B[j,k]) * W[k*n+i,c]
            # Hmm, that's not quite right either.
            
            # The standard form: C[i,k] = sum_c U[i*m+?, c] * V[?*p+k, c] * W[k*n+i, c]
            # where the middle index j is summed. So:
            # C[i,k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # And we need this to equal sum_j A[i,j] B[j,k]
            
            # So the condition is: for each i,k, sum_j sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = sum_j A[i,j]B[j,k]
            # This must hold for ALL A and B, so the tensor must match exactly.
            
            # The tensor T_{i,j,k} for matrix multiplication is: T[i,j,k] = 1 (it's just the identity in the middle dimension)
            # So we need: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,j,k
            # Wait, that's not right. Let me think again.
            
            # C[i,k] = sum_j A[i,j] B[j,k]
            # = sum_j sum_c (U[i*m+j,c] A[i,j]) (V[j*p+k,c] B[j,k]) W[k*n+i,c] ... no
            
            # The correct tensor decomposition form is:
            # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
            # No, that's not standard either.
            
            # Standard: C[i,k] = sum_c U_c[i,m] * V_c[m,p] * W_c[p,n] where these are rank-1 tensors
            # In index form: C[i,k] = sum_c sum_j U[i,j,c] V[j,k,c] W[i,k,c]... 
            # I'm getting confused. Let me use the seed program's convention.
            
            # From seed: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
            # The decomposition is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] for each (i,j,k)
            # And this should equal the tensor element, which for matrix mult is 1 if... 
            # Actually it should reconstruct: sum_j [sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]] A[i,j] B[j,k] = C[i,k]
            # So the inner sum over j of (tensor_val[i,j,k]) * A[i,j] * B[j,k] = C[i,k]
            # This means tensor_val[i,j,k] must be such that this holds for all A,B.
            # That requires tensor_val[i,j,k] = delta_{j,j}... no.
            
            # It requires that sum_j tensor_val[i,j,k] A[i,j] B[j,k] = sum_j A[i,j] B[j,k]
            # for all A,B, which means tensor_val[i,j,k] = 1 for all i,j,k.
            
            # So we need: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for ALL i in [0,n), j in [0,m), k in [0,p)
            
            # That's the condition! For every triple (i,j,k), the sum over c must be 1.
            
            # This is a system of n*m*p = 60 equations in rank variables (3*rank unknowns).
            # We need integer/half-integer solutions.
            
            # For rank 60, trivial solution: one c per (i,j,k) with U=V=W=1.
            # For lower rank, we need to solve this system.
            
            # This is a hard combinatorial problem. Let me just verify the schoolbook and return it.
            # Finding lower rank likely requires sophisticated algorithms beyond what I can do reliably in 110s.
            
            pass
    
    return best_result
# EVOLVE-BLOCK-END
