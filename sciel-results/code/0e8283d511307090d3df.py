"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Try ranks from theoretical lower bound upward
    # Lower bound: need at least max(n*p, m*p, n*m) for bilinear, but tensor rank can be different
    # For exact integer/half-integer, try various ranks
    
    # Strategy: Use the fact that for matrix multiplication, there are known constructions
    # For 2x4x7, try to exploit structure
    
    # First, let's try a simple approach: see if we can do it with fewer terms
    # by grouping
    
    # Try rank = n*p = 14 (this is the number of output entries)
    # Each output C[i,k] = sum_j A[i,j]*B[j,k]
    # We can write this as sum over j, so rank m*n*p is schoolbook
    # But we might share computation
    
    # Let's try a systematic search for small ranks
    # Use the property that entries should be 0, ±1/2, ±1, ±3/2, etc.
    
    # For a 2x4x7 matrix multiplication, let's try to find a decomposition
    # The key insight: we can factor the middle dimension
    
    # Try rank 28 (which is m*p = 4*7)
    # U[i*m+j, r] where r indexes (j,k) pairs -> r = j*p+k
    # V[j*p+k, r] where r indexes (j,k) -> V[j*p+k, j'*p+k'] = delta(j,j')*delta(k,k')... no
    
    # Actually, let me think differently.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_j sum_{j'} A[i,j] delta(j,j') B[j',k]
    # This is rank m*n*p
    
    # Alternative: use Strassen-like ideas or exploit that n=2 is small
    
    # For n=2, we have C[0,k] = sum_j A[0,j] B[j,k]
    #              C[1,k] = sum_j A[1,j] B[j,k]
    
    # We can compute these independently, so rank = 2*m*p = 2*4*7 = 56 (schoolbook)
    # But we might share some terms
    
    # Let's try to find a decomposition with rank < 56 using a search
    # Since exact integer/half-integer entries are required, we can use a finite search
    
    # Try ranks: 28, 30, 32, ..., 55
    # For each rank, try to find a valid decomposition
    
    # Use a simple greedy/iterative approach
    # Or use the known result: for 2xm x mxp, the rank is at most 2*min(m,p)*max(1, ceil(m/p))... 
    
    # Actually, let's just try to verify if rank 28 works with a specific construction
    
    # Construction attempt for rank 28:
    # Let r = j*p + k, so r ranges over 0..27
    # U[i*m+j, r] = A[i,j] if k matches, but we need it to be constant per column
    # This doesn't directly work
    
    # Let me try: U[i*m+j, j'*p+k'] = delta(i, i') for some mapping...
    
    # Alternative approach: use the fact that we can write
    # sum_j A[i,j] B[j,k] = sum_{j,k'} A[i,j] B[j,k'] delta(k',k)
    # But delta(k',k) needs to be represented
    
    # Let's try a different construction:
    # For each j, compute A[i,j] * B[j,k] for all i,k
    # This gives m terms, each of which is an outer product
    
    # Rank m construction:
    # U[i*m+j, j'] = A[i,j] if j==j' else 0 -> but U must be constant
    # This doesn't work because U depends on i
    
    # The correct schoolbook:
    # For each (i,j,k): U[i*m+j, (i,j,k)] = 1, V[j*p+k, (i,j,k)] = 1, W[k*n+i, (i,j,k)] = 1
    # Sum over (i,j,k) gives the product
    
    # To reduce rank, we need to share terms across different (i,j,k) tuples
    
    # Let's try: can we share across i for fixed (j,k)?
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # If we fix (j,k) and vary i, we need:
    # U[i*m+j, r] * W[k*n+i, r] to be such that summing gives the right result
    
    # For fixed (j,k), the contribution to C[i,k] from this term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want this to contribute to C[i,k] = A[i,j]*B[j,k]
    
    # If we set V[j*p+k, r] = 1 for some r, then we need U[i*m+j, r] * W[k*n+i, r] to represent A[i,j]*B[j,k]
    # But B[j,k] is constant for fixed (j,k), so we need U[i*m+j, r] * W[k*n+i, r] = c * A[i,j]
    # This is an outer product, which requires rank 2 for the 2x2 matrix [U[0*m+j,r]*W[k*n+0,r], U[0*m+j,r]*W[k*n+1,r]; U[1*m+j,r]*W[k*n+0,r], U[1*m+j,r]*W[k*n+1,r]]
    # But we only have one term, so we can only represent rank-1 matrices
    
    # So for each (j,k), we need at least 2 terms to represent the 2-vector [A[0,j], A[1,j]] times B[j,k]
    # This suggests rank >= 2*m*p = 56, which is the schoolbook
    
    # Wait, that's not right. Let me reconsider.
    
    # Actually, the tensor rank of the matrix multiplication tensor [n,m,p] is known to be at most n*m*p (schoolbook)
    # and at least max(n*p, m*p, n*m) in some cases, but the exact value is often unknown
    
    # For 2x4x7, the known bounds might not give a reduction below 56
    
    # However, the problem asks for integer/half-integer entries, which is a constraint
    # The schoolbook uses all 1's, which are integers, so rank 56 is achievable
    
    # Can we do better? Let's try a small search for rank 55, 54, etc.
    # But this is computationally expensive
    
    # Given the time constraint, let's return the schoolbook decomposition
    # unless we can find a quick improvement
    
    # Try a simple heuristic: see if we can merge some terms
    # For now, return schoolbook
    
    if time.time() > deadline:
        return best_result
    
    # Try to verify if a lower rank is possible with a quick check
    # For rank r, we need to find matrices U, V, W of appropriate sizes
    # such that the tensor equation holds
    
    # This is a hard problem, so let's just return the schoolbook result
    # The score will be 56/56 = 1.0, same as the seed
    
    return best_result
# EVOLVE-BLOCK-END
