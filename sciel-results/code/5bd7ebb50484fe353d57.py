"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    n, m, p = 3, 3, 7
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a rank-23 decomposition using the 3x3 matrix multiplication structure
    # The 3x3x3 tensor has rank 23. We exploit that the 3x3x7 tensor decomposes as
    # 3 copies of 3x3x3 (sharing the same A factor), so rank = 23.
    
    # Known rank-23 decomposition of 3x3 matrix multiplication (from literature)
    # Using the structure where we compute C = A*B for 3x3 matrices
    # The 23 rank-1 terms have U,V in {0,1,-1} and W in {0,±1/2,±1,±3/2}
    
    # Let me try to find a valid rank-23 decomposition using a search approach
    # based on the structure of 3x3 matrix multiplication.
    
    # Actually, let me use a well-known approach: the 3x3x3 tensor of rank 23
    # can be constructed. For 3x3x7, we use the same 23 terms but with W expanded.
    
    # The key: for multiplying A(3x3) * B(3x7), view B as 3 blocks of 3x3.
    # C[:, 0:3] = A * B[:, 0:3]
    # C[:, 3:6] = A * B[:, 3:6]  
    # C[:, 6:7] = A * B[:, 6:7]  (this is 3x1, not 3x3)
    
    # Hmm, 7 is not a multiple of 3. Let me reconsider.
    
    # Actually, the tensor for (3,3,7) multiplication:
    # T[i,j,k] = delta (meaning: C[i,k] = sum_j A[i,j] * B[j,k])
    # U has shape (9, rank), V has shape (21, rank), W has shape (21, rank)
    
    # The schoolbook rank is 3*3*7 = 63.
    # We need rank < 63.
    
    # Let me try rank 27: use 3 independent 3x3x3 decompositions of rank 9 each
    # No, that's still 27 > 23.
    
    # Better: since all three 3x3 sub-multiplications share the same A,
    # we can reuse the same rank-1 terms. If 3x3x3 has rank 23, then
    # 3x3x7 (as 3 copies sharing A) has rank at most 23.
    
    # Wait - 7 is not 3*something. Let me re-examine.
    # B is 3x7. We can split B into B1 (3x3), B2 (3x3), B3 (3x1).
    # C1 = A*B1 (3x3), C2 = A*B2 (3x3), C3 = A*B3 (3x1)
    # C1 and C2 each need rank-23 terms for 3x3x3.
    # C3 needs rank-9 terms for 3x3x1.
    # But they share the same A, so the U factors are the same.
    
    # Actually, the total rank for 3x3x7 where we use shared A:
    # We need to decompose the full tensor. The rank is at most 23 (from 3x3x3)
    # if 7 <= 3, but 7 > 3 so we need more.
    
    # Let me just try: rank = n*m*ceil(p/m) * something... 
    # Or more directly: the tensor rank of (n,m,p) is at most n*m*p but can be lower.
    
    # For (3,3,7): let me try rank 27 first (3 * 3 * 3 = 27, treating 7 as 3+3+1
    # with the 1-column part needing 3 terms, total 23+23+3 but sharing A means
    # we can do better).
    
    # Actually, the simplest improvement: use the fact that the 3x3x3 part has
    # rank 23. For the 3x3x7 tensor:
    # - Split into (3,3,3) + (3,3,3) + (3,3,1)
    # - The (3,3,3) parts each need rank 23, but they share U
    # - The (3,3,1) part needs rank 3 (schoolbook for 3x3x1)
    # - Total: we need 23 + 23 + 3 = 49 rank-1 terms? No, because they share U.
    
    # Hmm, let me think differently. The tensor is:
    # T_{ijk} where i in [0,3), j in [0,3), k in [0,7)
    # T_{ijk} = 1 if (i,j,k) corresponds to a valid multiplication term
    # Actually T_{ijk} is just the structure tensor.
    
    # The rank is the minimum number of rank-1 tensors that sum to T.
    # A rank-1 tensor is u ⊗ v ⊗ w where u in R^9, v in R^21, w in R^21.
    
    # For the schoolbook: rank = 63, each term picks one (i,j,k).
    
    # To reduce rank, we need rank-1 tensors that "cover" multiple (i,j,k) tuples.
    
    # Key observation: if we fix i and j, then for all k, T[i,j,k] = 1.
    # So a rank-1 tensor with u having a single 1 at position i*m+j, v having a single
    # 1 at position j*p+k0, and w having 1s at positions k0*n+i for all k0... 
    # No, that doesn't work for a rank-1 tensor.
    
    # Let me just use the known result: rank(3,3,p) for p >= 3.
    # rank(3,3,3) = 23
    # rank(3,3,4) = ? 
    # rank(3,3,p) for large p approaches 3*p (since we can do 3 independent 3x1 multiplications)
    
    # For p=7: rank(3,3,7) <= 3*7 = 21? No, that's the lower bound.
    # Actually rank(3,3,p) <= n*m*p but also <= n*p*ceil(m/1)... 
    
    # Let me just try to find a decomposition with rank < 63 using a greedy/search approach.
    
    # Strategy: Start with schoolbook, then try to merge terms.
    # Two schoolbook terms (i1,j1,k1) and (i2,j2,k2) can be merged into one rank-1
    # tensor if there exists u,v,w such that:
    # u⊗v⊗w has 1s exactly at (i1*m+j1, j1*p+k1, k1*n+i1) and (i2*m+j2, j2*p+k2, k2*n+i2)
    
    # This is complex. Let me try a different approach: use the Strassen-like idea.
    
    # For 2x2x2, Strassen reduces rank from 8 to 7.
    # For 3x3x3, the rank is 23 (known result).
    
    # Let me try to use the 3x3x3 rank-23 decomposition and extend to 3x3x7.
    
    # I'll use a simple approach: try rank values from low to high and check if
    # a valid decomposition exists using a constraint satisfaction approach.
    
    # For now, let me return a rank-27 decomposition (3 blocks of 3x3x3, each with
    # rank 9 schoolbook, but that's 27). Actually 3*3*3=27.
    
    # Better: rank 24 = 3*3*3 - 3 + something... 
    
    # Let me just return the schoolbook for now and try to improve.
    # The task says score = rank, so I need to minimize rank.
    
    # Let me try a rank-36 decomposition: use 4 independent 3x3x3 schoolbook
    # No, that's 36. Better than 63.
    
    # Actually, simplest improvement: the 3x3x7 tensor can be seen as computing
    # 3 independent 3x3 matrix multiplications (for the first 3 columns of B)
    # plus 3 independent 3x3 (for columns 3-5) plus 3x1 (for column 6).
    # But they all share the same A, so the U factor is shared.
    
    # The total number of rank-1 terms needed:
    # For each of the 7 columns of B, we need to compute A (3x3) * B_col (3x1) = C_col (3x1)
    # This is a (3,3,1) tensor with rank 3 (schoolbook) or possibly less.
    # But since all 7 columns share the same A, the U factor is the same.
    
    # Total rank = 7 * rank(3,3,1) = 7 * 3 = 21? 
    # Wait, rank(3,3,1) = 3 (since C[i] = sum_j A[i,j]*B[j], which needs 3 terms).
    # And since the U factor is shared, the total rank is 7*3 = 21.
    
    # Let me verify: for column k of B, the terms are:
    # For each i in 0..2, j in 0..2: term (i,j,k) with U[i*3+j]=1, V[j*7+k]=1, W[k*3+i]=1
    # For all 7 columns, the U entries are the same (U[i*3+j] for i,j in 0..2).
    # So we have 3*3*7 = 63 terms in schoolbook, but if we group by (i,j),
    # we have 9 groups, each with 7 terms.
    
    # Can we merge terms within a group? For fixed (i,j), we have terms for k=0..6.
    # Term (i,j,k): U[i*3+j]=1, V[j*7+k]=1, W[k*3+i]=1
    # To merge two terms (i,j,k1) and (i,j,k2) into one rank-1 tensor:
    # We need u⊗v⊗w where u has 1 at i*3+j, v has 1 at j*7+k1 AND j*7+k2,
    # and w has 1 at k1*3+i AND k2*3+i.
    # But a rank-1 tensor v⊗w has entries v[a]*w[b], so we need v[j*7+k1]=1,
    # v[j*7+k2]=1, w[k1*3+i]=1, w[k2*3+i]=1. This works!
    # So we can merge all 7 terms for fixed (i,j) into ONE rank-1 tensor!
    
    # Wait, that means rank = 9 (one per (i,j) pair)?
    # Let me check: rank-1 tensor for (i,j): 
    # u[i*3+j] = 1, u[other] = 0
    # v[j*7+k] = 1 for all k in 0..6, v[other] = 0
    # w[k*3+i] = 1 for all k in 0..6, w[other] = 0
    # Then (u⊗v⊗w)[i*3+j, j*7+k, k*3+i] = 1*1*1 = 1 for all k. ✓
    # And for other indices, it's 0. ✓
    
    # So the rank is at most 9! That's a huge improvement from 63.
    
    # Let me verify this is correct. The tensor T has T[i*3+j, j*7+k, k*3+i] = 1
    # for all i,j in 0..2 and k in 0..6. All other entries are 0.
    # My decomposition: sum over (i,j) in 0..2 x 0..2 of u_ij ⊗ v_ij ⊗ w_ij
    # where u_ij has 1 at i*3+j, v_ij has 1 at j*7+k for all k, w_ij has 1 at k*3+i for all k.
    # The sum has entry at (i*3+j, j*7+k, k*3+i) = sum over (i',j') of 
    # u_{i'j'}[i*3+j] * v_{i'j'}[j*7+k] * w_{i'j'}[k*3+i]
    # = u_ij[i*3+j] * v_ij[j*7+k] * w_ij[k*3+i] (only the (i,j) term is nonzero)
    # = 1*1*1 = 1. ✓
    
    # And for any other index (a,b,c): if a = i'*3+j' for some (i',j'), b = j'*7+k',
    # c = k'*3+i'' where (i'',k') != (i',j') or i'' != i' or k' doesn't match, then
    # the only potentially nonzero term is (i',j'), but v_{i'j'}[b] * w_{i'j'}[c] = 0
    # unless b = j'*7+k and c = k*3+i' for the same k. Since b = j'*7+k' and c = k'*3+i'',
    # we need k = k' and i'' = i'. If i'' = i' and k' = k, then it's the correct entry.
    # Otherwise, it's 0. ✓
    
    # So rank 9 works! Let me implement it.
    
    def rank9_decomposition():
        rank = 9
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                for k in range(p):
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                col += 1
        
        return {"U": U, "V": V, "W": W}
    
    result = rank9_decomposition()
    
    # Verify the decomposition
    if time.time() < deadline - 5:
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        
        # Check: sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] should equal the tensor
        # Tensor T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
        # else 0
        
        rank = len(U[0])
        # Build the tensor from the decomposition
        tensor = np.zeros((n*m, m*p, p*n))
        for c in range(rank):
            tensor += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
        
        # Build the target tensor
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1
        
        if np.allclose(tensor, target):
            return result
        else:
            # Fallback to schoolbook
            return schoolbook()
    
    return result
# EVOLVE-BLOCK-END
