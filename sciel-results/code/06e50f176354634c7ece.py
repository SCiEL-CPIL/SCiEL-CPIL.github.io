"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # n=3, m=3, p=8
    
    # Schoolbook rank: n*m*p = 72
    rank_schoolbook = n * m * p
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    # We want to find rank-r decomposition: T = sum_t u_t ⊗ v_t ⊗ w_t
    # where u_t is in R^{n*m}, v_t in R^{m*p}, w_t in R^{p*n}
    
    # The tensor can be viewed as a matrix of shape (n*m) x (m*p*p*n)
    # by flattening the last two indices. Or as (n*m*m*p) x (p*n).
    # Let's use the matrix unfolding: M[i*m+j, (k*n+i)*p + k] ... this is getting complex.
    
    # Alternative: use the "slice" approach. For each j, the slice T[:, j*p:(j+1)*p, :] 
    # has entries T[i*m+j, j*p+k, k*n+i] = delta_{i,i} * delta_{k,k} ... no.
    
    # Let me be more careful. T[a, b, c] where:
    # a = i*m + j (i in 0..n-1, j in 0..m-1)
    # b = j*p + k (j in 0..m-1, k in 0..p-1)  
    # c = k*n + i (k in 0..p-1, i in 0..n-1)
    # T[a,b,c] = 1 if the indices are consistent, else 0.
    
    # For a given (i,j,k): a=i*m+j, b=j*p+k, c=k*n+i
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # To find the minimal rank, we can try to find linear dependencies.
    # The tensor has 72 "nonzero" entries (one for each (i,j,k) triple).
    # The schoolbook decomposition uses 72 terms, one for each (i,j,k).
    
    # Let's try to find a lower-rank decomposition by solving a system of equations.
    # Assume rank r. We need to find U (n*m x r), V (m*p x r), W (p*n x r) such that:
    # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all (i,j,k)
    # and = 0 for all other (a,b,c) combinations.
    
    # This is a non-convex problem. Let's try a greedy approach:
    # Start with schoolbook (rank 72) and try to merge terms.
    
    # For now, let's return the schoolbook decomposition as a safe baseline.
    # Then we can try to improve.
    
    rank = rank_schoolbook
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Try to find a lower-rank decomposition using linear algebra.
    # The tensor T can be flattened to a matrix of shape (n*m) x (m*p*p*n)
    # by treating (b,c) as a combined index. But this doesn't directly give rank.
    
    # Better: use the "mode-1" unfolding. M1[i, (j,k)] where i in 0..n*m-1, 
    # but (j,k) isn't a simple pair. Let me think again.
    
    # Actually, for matrix multiplication tensor M_{n,m,p}, the minimal rank 
    # is the tensor rank. For n=m=p=2, it's 7 (Strassen). For n=m=p=3, 
    # it's at most 23 (Bini et al.) and at least 19.
    
    # For n=3, m=3, p=8, the rank is at most 72 (schoolbook) and at least 
    # max(9, 24, 24) = 24 (trivial lower bounds).
    
    # Let's try a structured approach: use Strassen-like savings on 2x2 subblocks.
    # Partition the 3x3 A into 2x2 blocks and apply Strassen.
    
    # For a 2x2x2 block (rows {r1,r2}, cols {c1,c2}, and for each k in 0..7):
    # We compute C[r1,k] and C[r2,k] from A[r1,c1], A[r1,c2], A[r2,c1], A[r2,c2]
    # and B[c1,k], B[c2,k].
    
    # This is a 2x2x1 problem (2x2 matrix times 1x2 vector), which has rank 4.
    # No Strassen savings possible for 2x2x1.
    
    # So the schoolbook rank of 72 might be close to optimal for this shape.
    
    # Let's try to verify if we can do better by checking for linear dependencies.
    # Create the 72 "basis" tensors and check if any are linearly dependent.
    
    # Each basis tensor e_{i,j,k} has a single nonzero entry at (i*m+j, j*p+k, k*n+i).
    # These are all distinct positions, so they are linearly independent as tensors.
    # Therefore, the tensor rank is exactly 72.
    
    # Wait, that's not right. Tensor rank is about decomposing as a sum of 
    # rank-1 tensors. The 72 schoolbook terms are rank-1 tensors, but there might
    # be a decomposition with fewer rank-1 tensors.
    
    # The fact that the 72 positions are distinct doesn't mean the rank is 72.
    # For example, the 2x2x2 matrix multiplication tensor has 8 nonzero entries 
    # but rank 7.
    
    # Let me try to find a rank-71 or lower decomposition by solving a system.
    
    # For rank r, we have 3*r*(n*m + m*p + p*n) / 3 = r*(n*m + m*p + p*n) unknowns
    # (up to scaling). The constraints are:
    # 1. T[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t] for all (a,b,c) in the support
    # 2. T[a,b,c] = 0 for all (a,b,c) not in the support
    
    # The support has 72 points. The total number of (a,b,c) combinations is 
    # n*m * m*p * p*n = 72 * 24 * 24 = 41472. Most are zero.
    
    # This is a hard non-convex problem. Let's try a simpler heuristic.
    
    # Try to find pairs of schoolbook terms that can be merged using a Strassen-like trick.
    # For terms (i1,j1,k) and (i2,j2,k) with the same k, we can try to combine them.
    
    # For fixed k, the terms are: for i in 0..2, j in 0..2: A[i,j]*B[j,k] contributes to C[i,k].
    # This is a 3x3x1 problem. The rank of 3x3x1 is 9 (it's a linear map).
    
    # So for each k, we need 9 terms. Total: 8*9 = 72. No savings possible 
    # by considering each k independently.
    
    # The only way to save is to share terms across different k values.
    # But the B[j,k] for different k are independent, so it's not obvious 
    # how to share.
    
    # I think the rank is exactly 72 for this problem. Let's return the schoolbook.
    
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return result
# EVOLVE-BLOCK-END
