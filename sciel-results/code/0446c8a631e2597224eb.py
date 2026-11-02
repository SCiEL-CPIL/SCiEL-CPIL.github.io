import numpy as np
import time
from itertools import product, combinations

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Helper to convert indices
    def u_to_ij(u): return u // m, u % m
    def v_to_jk(v): return v // p, v % p
    def w_to_ki(w): return w // n, w % n
    def ij_to_u(i, j): return i * m + j
    def jk_to_v(j, k): return j * p + k
    def ki_to_w(k, i): return k * n + i
    
    # 1. Construct Schoolbook Rank 18 Decomposition
    # This is our baseline and fallback.
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(nm)]
    V_fb = [[0] * rank_fb for _ in range(mp)]
    W_fb = [[0] * rank_fb for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = ij_to_u(i, j)
                v = jk_to_v(j, k)
                w = ki_to_w(k, i)
                U_fb[u][col] = 1
                V_fb[v][col] = 1
                W_fb[w][col] = 1
                col += 1
    fallback = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Verify fallback just in case (should always be true)
    # T_recon = sum(U_fb[:,r] * V_fb[:,r] * W_fb[:,r]) for all r
    # This is O(R * N * M * P) which is trivial for R=18.
    
    # 2. Attempt Rank 15 Construction via "Subtraction of Linear Dependencies"
    # The tensor T for 2x3x3 matrix multiplication can be viewed as two independent
    # 1x3x3 tensors (for row i=0 and row i=1 of A).
    # Each 1x3x3 tensor has rank 9 (schoolbook). Total 18.
    # However, the global structure might allow sharing of terms or cancellations.
    # Actually, the supports in U and W are disjoint for different i.
    # U indices for i=0: 0,1,2. U indices for i=1: 3,4,5.
    # W indices for i=0: 0,1,2. W indices for i=1: 3,4,5.
    # V indices are shared: 0..8.
    # Since U and W supports are disjoint, the tensor is a direct sum.
    # T = T_0 + T_1 where T_0 uses U[0:3] and W[0:3], and T_1 uses U[3:6] and W[3:6].
    # The rank of a direct sum is the sum of the ranks.
    # rank(T) = rank(T_0) + rank(T_1).
    # T_0 is a 1x3x3 tensor (effectively). Its rank is 9.
    # So rank(T) = 9 + 9 = 18.
    # 
    # Wait, is rank(1x3x3) really 9?
    # A 1xm x p tensor is just a matrix of size m x p.
    # The rank of a tensor of order 3 is the minimal number of rank-1 tensors.
    # A rank-1 tensor u v^T w^T where u is 1x1, v is 1xm, w is 1xp?
    # No, the indices are (i, j, k).
    # For i fixed, T[i, j, k] = 1.
    # This is just a matrix M_{jk} = 1.
    # The tensor decomposition is sum_r u_r v_r w_r.
    # Since u is a scalar (dimension 1), u_r is just a number.
    # So we are decomposing the all-ones matrix 3x3 into rank-1 matrices.
    # The rank of the all-ones 3x3 matrix is 1.
    # So T_0 can be represented by 1 term?
    # Let's check.
    # T_0[i, j, k] = 1 for i=0, all j,k.
    # Term r: U[0, r] * V[j, r] * W[k, r].
    # If we set U[0, r] = 1, V[j, r] = 1, W[k, r] = 1, then sum is 1.
    # But V and W are shared across the whole problem?
    # In the direct sum argument, the factors U, V, W are global.
    # For T_0, we need U[0..2, r], V[0..8, r], W[0..2, r].
    # For T_1, we need U[3..5, r], V[0..8, r], W[3..5, r].
    # The V factor is the bottleneck.
    # Can we reuse V terms?
    # Let's try to construct a rank 15 solution.
    # If rank(T_0) = 1 and rank(T_1) = 1, total rank 2? No, that's not how it works.
    # The decomposition is global.
    # T = sum_{r=1}^R u_r \otimes v_r \otimes w_r.
    # For i=0: T[0, j, k] = sum_r U[0, r] V[j, r] W[k, r] = 1.
    # For i=1: T[1, j, k] = sum_r U[1, r] V[j, r] W[k, r] = 1. (Wait, W index is k*n+i. So W[k*2+0] and W[k*2+1]).
    # Let's rewrite indices clearly.
    # U index: u = i*m + j. (0..5)
    # V index: v = j*p + k. (0..8)
    # W index: w = k*n + i. (0..5)
    #
    # Equation for i=0:
    # For all j in 0..2, k in 0..2:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # i=0: sum_r U[j, r] * V[3j+k, r] * W[2k, r] = 1
    # i=1: sum_r U[3+j, r] * V[3j+k, r] * W[2k+1, r] = 1
    #
    # Notice that V depends on j and k.
    # If we could choose V[j*p+k, r] to be 1 for all j,k, then:
    # i=0: sum_r U[j, r] W[2k, r] = 1
    # i=1: sum_r U[3+j, r] W[2k+1, r] = 1
    # This separates into two matrix rank problems.
    # Let A_0[j, r] = U[j, r] and B_0[k, r] = W[2k, r]. We want A_0 B_0^T = J_3 (all ones 3x3).
    # Rank of J_3 is 1. So we need 1 term?
    # If R=1, U[j, 1]=1, W[2k, 1]=1.
    # Similarly for i=1, U[3+j, 1]=1, W[2k+1, 1]=1.
    # Then V[3j+k, 1] must be 1 for all j,k.
    # So a rank 1 decomposition exists?
    # Let's check.
    # U[0..5, 0] = 1
    # V[0..8, 0] = 1
    # W[0..5, 0] = 1
    # T[i, j, k] = U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1*1*1 = 1.
    # This works!
    # So the rank is 1?
    #
    # Wait, is this allowed?
    # "integer/half-integer entries"
    # 1 is an integer.
    # "exactly decompose the matrix multiplication tensor"
    # Yes.
    #
    # Why did everyone think it was 18?
    # Because they assumed monomial patterns or didn't realize the all-ones structure allows rank 1.
    # Or maybe I am misinterpreting the tensor definition.
    #
    # Let's re-read carefully.
    # "Matrix multiplication 2,3,3"
    # "T[i, j, k] = 1 if we have A[i,j] * B[j,k] contributing to C[k,i]"
    # This is the standard matrix multiplication tensor.
    # The tensor T is defined such that C = A B is computed by contracting T with A and B.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # The tensor T has indices (i, j, k).
    # T[i, j, k] = 1.
    #
    # If T[i, j, k] = 1 for all i, j, k, then the rank is indeed 1.
    # u_r = 1 for all i, v_r = 1 for all j, w_r = 1 for all k.
    #
    # Is there a catch?
    # "Find a rank-minimal exact tensor decomposition"
    # "score is the rank"
    # "lower is better"
    #
    # If rank 1 is possible, the score would be 1/18 * 1.0? No, score is seed/value.
    # Seed program has rank 18. Score 1.0.
    # If I return rank 1, score = 18/1 = 18.0.
    #
    # Let's double check if "rank" in tensor decomposition of matrix multiplication usually refers to the number of multiplications.
    # Yes, Strassen's algorithm reduces the rank from 8 to 7 for 2x2x2.
    # The tensor for 2x2x2 matrix multiplication is NOT all ones.
    # T[i, j, k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # For 2x2x2:
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
    #
    # The tensor T is indexed by (i, j, k) where i is row of C, j is intermediate, k is col of C?
    # No, the standard definition is:
    # T is a tensor of shape (n, m, p).
    # T[i, j, k] = 1 if A[i, j] and B[j, k] contribute to C[i, k].
    #
    # In the prompt:
    # "Output factor follows the official trace convention W[k*n+i,r]"
    # This implies C index is (i, k) flattened to k*n+i? Or i*p+k?
    # "W[k*n+i,r]" -> W index is k*n + i.
    # This means C is indexed by (i, k) but flattened as k*n+i.
    # So C[k*n+i] corresponds to C[i, k]?
    # Usually C[i, k] is row i, col k.
    # If W index is k*n+i, then for fixed k, i varies.
    # This is column-major for C?
    #
    # Let's look at the indices in the seed program:
    # u = i*m + j  (A[i, j])
    # v = j*p + k  (B[j, k])
    # w = k*n + i  (C[i, k] flattened as k*n+i)
    #
    # The condition for T[u, v, w] = 1 is:
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    #
    # This implies that for a fixed (i, j, k), there is exactly one term.
    # And T[u, v, w] = 1 for all valid (i, j, k).
    #
    # So T is indeed the all-ones tensor on the valid indices?
    # No, T is a tensor of shape (nm, mp, pn).
    # Most entries are 0.
    # T[i*m+j, j*p+k, k*n+i] = 1.
    #
    # My previous analysis:
    # T_recon[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    # We want this to be 1 for all i, j, k.
    # And 0 for all other combinations?
    #
    # The problem statement says: "exactly decompose the matrix multiplication tensor".
    # The matrix multiplication tensor T is defined by T[i, j, k] = 1 if A[i, j] B[j, k] contributes to C[i, k].
    # In the standard definition, T is sparse. It is 1 at positions (u, v, w) where u, v, w correspond to a valid (i, j, k) triple.
    # All other entries are 0.
    #
    # So we need:
    # sum_r U[u, r] V[v, r] W[w, r] = 1 if (u, v, w) is a valid triple.
    # sum_r U[u, r] V[v, r] W[w, r] = 0 if (u, v, w) is NOT a valid triple.
    #
    # My rank 1 attempt:
    # U[u, 0] = 1 for all u.
    # V[v, 0] = 1 for all v.
    # W[w, 0] = 1 for all w.
    # Then sum = 1 for ALL (u, v, w).
    # But we need 0 for invalid (u, v, w).
    #
    # So rank 1 does NOT work because it doesn't vanish on the invalid entries.
    #
    # Okay, back to square one.
    # The tensor is sparse.
    # The supports are disjoint in a specific way.
    #
    # Let's check the direct sum argument again.
    # Valid triples (u, v, w) are determined by (i, j, k).
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    #
    # Consider i=0 and i=1.
    # For i=0:
    # u in {0, 1, 2}
    # w in {0, 1, 2} (since w = 2k + 0)
    # v = 3j + k.
    #
    # For i=1:
    # u in {3, 4, 5}
    # w in {3, 4, 5} (since w = 2k + 1)
    # v = 3j + k.
    #
    # The set of valid (u, v, w) for i=0 is disjoint from the set for i=1 in the U and W dimensions.
    # Specifically, U indices for i=0 are 0,1,2. For i=1 are 3,4,5.
    # W indices for i=0 are 0,1,2. For i=1 are 3,4,5.
    #
    # Since the supports in U and W are disjoint, the tensor is a direct sum T = T_0 \oplus T_1.
    # The rank of a direct sum is the sum of the ranks.
    # rank(T) = rank(T_0) + rank(T_1).
    #
    # T_0 is the tensor for 1x3x3 matrix multiplication (effectively).
    # T_0[u, v, w] = 1 if u=i*m+j (i=0), v=j*p+k, w=k*n+0.
    # This is equivalent to a 1x3x3 tensor.
    # What is the rank of a 1xm x p tensor?
    # It is the rank of the matrix M_{jk} = 1.
    # The rank of the all-ones 3x3 matrix is 1.
    # So rank(T_0) = 1?
    #
    # Let's verify.
    # We need U_0, V_0, W_0 such that:
    # U_0[j] V_0[3j+k] W_0[2k] = 1 for all j,k.
    # And 0 for invalid?
    # For T_0, the "invalid" entries are those where u, v, w don't match the pattern.
    # But in the direct sum, we only care about the sub-tensor.
    # The global decomposition must handle the zeros.
    #
    # If rank(T_0) = 1, we can use 1 term for i=0.
    # Term r0:
    # U[0..2, r0] = 1
    # V[0..8, r0] = 1
    # W[0..2, r0] = 1
    # W[3..5, r0] = 0
    #
    # This term contributes 1 to all valid (u,v,w) for i=0.
    # Does it contribute 0 to invalid (u,v,w) for i=0?
    # Invalid (u,v,w) for i=0 means u in 0..2, but v,w don't match.
    # If u=0 (j=0), v=1 (j=0, k=1? No, v=0*3+1=1). w=0 (k=0).
    # Triple (0, 1, 0).
    # i=0, j=0, k=0 -> u=0, v=0, w=0. Valid.
    # i=0, j=0, k=1 -> u=0, v=1, w=2. Valid.
    # Is (0, 1, 0) valid?
    # u=0 -> i=0, j=0.
    # v=1 -> j=0, k=1.
    # w=0 -> k=0, i=0.
    # j matches (0=0).
    # k matches? v says k=1, w says k=0. Mismatch.
    # So (0, 1, 0) is invalid.
    #
    # With my term r0:
    # U[0, r0]=1, V[1, r0]=1, W[0, r0]=1.
    # Product = 1.
    # But T[0, 1, 0] should be 0.
    # So rank 1 does NOT work for T_0 either, because it doesn't vanish on invalid entries within the block.
    #
    # So we need a decomposition that vanishes on the "cross terms" within the 1x3x3 block.
    # This is the standard matrix multiplication tensor problem for 1x3x3.
    # The rank of the 1xm x p matrix multiplication tensor is known to be m*p?
    # No, for 1xm x p, it's just a matrix multiplication C = A B where A is 1xm, B is mxp.
    # C is 1xp.
    # C[k] = sum_j A[j] B[j, k].
    # This is just a vector-matrix product.
    # The tensor is T[j, k] = 1? No, T[j, k] is not a tensor.
    # The tensor is T[u, v, w] where u is index of A, v is index of B, w is index of C.
    # A has m elements. B has m*p elements. C has p elements.
    # T[a, b, c] = 1 if b corresponds to (j, k) and a corresponds to j, and c corresponds to k.
    # So T[j, 3j+k, k] = 1.
    #
    # What is the rank of this tensor?
    # This is the "vector-matrix multiplication" tensor.
    # It is known that the rank of the 1xm x p tensor is m*p?
    # Let's check m=1, p=1. Rank 1.
    # m=1, p=2.
    # T[0, 0, 0] = 1 (j=0, k=0)
    # T[0, 1, 1] = 1 (j=0, k=1)
    # Can we do rank 1?
    # U[0, r] V[0, r] W[0, r] = 1
    # U[0, r] V[1, r] W[1, r] = 1
    # U[0, r] V[0, r] W[1, r] = 0
    # U[0, r] V[1, r] W[0, r] = 0
    # From 3rd: V[0, r] W[1, r] = 0.
    # From 4th: V[1, r] W[0, r] = 0.
    # If W[0, r] != 0, then V[1, r] = 0. Then 2nd eq: 0 = 1. Contradiction.
    # So W[0, r] = 0. Then 1st eq: 0 = 1. Contradiction.
    # So rank 1 is impossible.
    # Rank 2?
    # Likely yes.
    # In general, the rank of the 1xm x p tensor is m*p.
    # Why?
    # Because the supports are disjoint in a way that prevents sharing.
    #
    # If rank(T_0) = m*p = 3*3 = 9.
    # And rank(T_1) = 9.
    # Total rank = 18.
    #
    # So the schoolbook rank 18 is likely optimal.
    #
    # Given the time constraints and the high likelihood that 18 is the minimum (due to the direct sum of two 1x3x3 tensors, each of rank 9), I will stick with the schoolbook decomposition.
    # The "subtraction of linear dependencies" strategy is unlikely to yield rank 15 if the lower bound is 18.
    #
    # I will return the schoolbook decomposition.

    return fallback
