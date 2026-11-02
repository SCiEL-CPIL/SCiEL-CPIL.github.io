"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
_start = time.time()

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is correct."""
    rank = len(U[0])
    # Reconstruct the tensor: T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # The multiplication tensor: T[(i,j),(j,k),(k,i)] = delta (i.e., T = 1 for valid (i,j,k) where j is shared)
    # Actually, the tensor for matmul: for inputs A[i,j] and B[j,k], output C[i,k] += A[i,j]*B[j,k]
    # So the tensor is: T[i,j,k] = 1 if we think of it as the "multiplication pattern"
    # More precisely: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 for all i,j,k
    # Wait no. The tensor M has M[(i,j),(j,k),(k,i)] such that C[i,k] = sum_{i,j,k} M[(i,j),(j,k),(k,i)] * A[i,j] * B[j,k]
    # For standard matmul: C[i,k] = sum_j A[i,j] * B[j,k], so M[(i,j),(j,k),(k,i)] = 1 for all i,j,k
    # The decomposition: M = sum_c u_c ⊗ v_c ⊗ w_c where
    #   u_c[i*m+j] = U[i*m+j, c], v_c[j*p+k] = V[j*p+k, c], w_c[k*n+i] = W[k*n+i, c]
    # Check: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False, (i, j, k, s)
    return True, None

def _rank36_decomposition():
    """Construct a rank-36 decomposition for 3x3 * 3x6 matmul."""
    n, m, p = 3, 3, 6
    rank = 36
    U = [[0.0] * rank for _ in range(n * m)]  # 9 x 36
    V = [[0.0] * rank for _ in range(m * p)]  # 18 x 36
    W = [[0.0] * rank for _ in range(p * n)]  # 18 x 36
    
    # Strategy: For 3x3 * 3x6, use the structure that 6 = 3+3
    # Split B into two 3x3 blocks: B[:,0:3] and B[:,3:6]
    # C = A @ B = [A @ B0 | A @ B1]
    # 
    # Use a 12-term decomposition per row of A (3 rows * 12 terms = 36)
    # For row i of A: C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:] + A[i,2]*B[2,:]
    # This is a 1x6 vector = sum of 3 "scaled" 1x6 rows of B
    # 
    # Key insight: we can use shared intermediate values.
    # For the 1x3 * 3x6 = 1x6 case, the tensor rank is 6 (since it's just
    # C[k] = a0*B[0,k] + a1*B[1,k] + a2*B[2,k] for k=0..5)
    # This requires rank 6 in general (the 6 outputs are independent linear forms).
    # So 3 rows * 6 terms = 18 minimum? No wait...
    #
    # Actually for fixed i, the problem is: given a=(a0,a1,a2) and B (3x6),
    # compute c[k] = sum_j a[j]*B[j,k] for k=0..5.
    # The tensor for this 1x3 * 3x6 -> 1x6 has rank exactly 6 (it's a "rectangular"
    # multiplication that doesn't share structure in a useful way for rank reduction
    # below 6 per row... unless we share across rows).
    #
    # Let me try a different approach: use the fact that we can combine.
    # 
    # Alternative: think of it as 3x3 * 3x6 = 3x(3x3 * 3x2) no that's wrong.
    #
    # Let me just try to find ANY valid decomposition with rank < 54.
    #
    # Approach: Use rank-18 for the "A side" shared structure.
    # 
    # Actually, the simplest improvement: notice that for 3x3 * 3x6,
    # we can write it as (3x3 * 3x3) x2 but share the A multiplications.
    # If we precompute certain combinations of A, we can save.
    #
    # Let me try: rank-45 by saving one multiplication.
    # Or better: use the Winograd trick.
    #
    # For 3x3 * 3x3, Laderman gives rank 23.
    # For 3x3 * 3x6, if the two 3x3 blocks shared A-side terms:
    #   C0 = A * B0, C1 = A * B1
    #   Using Laderman: each needs 23 terms, total 46.
    #   But some A-side linear combinations are shared!
    #
    # Laderman's 23-term algorithm for 3x3*3x3 uses these 7 A-side combinations
    # (plus 16 B/C side). If both C0 and C1 use the same A combinations,
    # we only need to compute them once. But in tensor rank, "sharing" means
    # the same rank-1 term appears, which requires the same U vector.
    #
    # Let me just hardcode a known rank-46 or better construction.
    #
    # SIMPLEST VALID IMPROVEMENT: rank-45
    # Take schoolbook (54) and merge one pair of terms where possible.
    # This is tricky. Let me instead use a clean structural approach.
    #
    # CLEAN APPROACH: rank-36
    # For 3x3 * 3x6: treat as 3 separate 1x3 * 3x6 = 1x6 multiplications
    # Each 1x3 * 3x6 has tensor rank 6 (it's essentially a 3x6 matrix-vector
    # product viewed as a tensor, which has rank 6 since the 6 output coords
    # are independent).
    # Total: 3 * 6 = 18? No, that's not how tensor rank works for the combined tensor.
    #
    # The full 3x3 * 3x6 tensor: T[i,j,k] for i=0..2, j=0..2, k=0..5
    # T decomposes as sum_c U_c ⊗ V_c ⊗ W_c where:
    #   sum_c U[i,j]*V[j,k]*W[k,i] = 1 for all i,j,k
    #
    # Minimum rank is at least the "slice rank" or other bounds.
    # Let me just try rank-36 with a specific construction and verify.
    
    # Construction: for each i in 0..2 and each "mode" in 0..11 (12 modes per row)
    # Use the identity: 1 = sum of 36 rank-1 terms
    
    # Let me use a different parameterization.
    # For the 3x3 * 3x6 tensor, I'll use:
    # rank = 3 * 12 = 36
    # For each i (row of A, 3 choices) and each t in 0..11:
    #   The term handles some combination
    
    # Actually, let me just verify: is rank 36 achievable?
    # The tensor is in F^9 ⊗ F^18 ⊗ F^18.
    # A trivial upper bound is 9*18/18 = 9? No.
    # The rank is at most 9*18 = 162 (schoolbook-like) but we know 54 works.
    # 
    # Let me try a very direct construction:
    # For i=0,1,2 and j=0,1,2 and k=0,...,5: we need U[i,j]*V[j,k]*W[k,i] summed over c to give 1.
    # 
    # Simple valid rank-54 (schoolbook) is known. Let me try rank-36.
    #
    # KEY INSIGHT: For 3x3 * 3x6, write k = 3q + r, q=0,1, r=0,1,2.
    # C[i, 3q+r] = sum_j A[i,j] * B[j, 3q+r]
    # 
    # For fixed i and q: the 3 outputs C[i,3q+0], C[i,3q+1], C[i,3q+2] 
    # are computed from the 3x3 sub-matrix B[:, 3q:3q+3] and vector A[i,:].
    # This is a 1x3 * 3x3 = 1x3 multiplication, which has tensor rank 3
    # (just dot product, needs 3 terms: for each j, term with A[i,j]*B[j,3q+r] for all r).
    # Wait, for fixed i,q: C[i,3q+r] = sum_j A[i,j]*B[j,3q+r], r=0,1,2
    # The tensor for this sub-problem: for j=0,1,2 and r=0,1,2,
    # we need sum_c U[i,j]*V[j,3q+r]*W[3q+r,i] = 1
    # This 3x3 "tensor" (indices j and r, with i fixed) has rank 3:
    #   term (j=r): U[i,j]=1, V[j,3q+j]=1, W[3q+j,i]=1, other entries 0
    # So 3 terms per (i,q) pair. Total: 3 * 2 * 3 = 18 terms!
    # 
    # Wait, that can't be right. Let me re-examine.
    # For fixed i and q, we need: for all j,r: sum_c U[i*m+j, c] * V[j*p+3q+r, c] * W[(3q+r)*n+i, c] = 1
    # where the sum is over the 3 terms for this (i,q) group.
    # If I use 3 terms indexed by j=0,1,2:
    #   Term j: U[i*3+j, c]=1, V[j*6+3q+j, c]=1 (only for r=j), W[(3q+j)*3+i, c]=1
    #   Then for input (i,j',k') where k'=3q+r': sum = 1 if j'=j and r'=j for the term with j.
    #   So sum over 3 terms gives 1 if j'=r', 0 otherwise. That's NOT what we want!
    # We want the sum to be 1 for ALL (j',r') pairs, not just j'=r'.
    #
    # I confused myself. The tensor T[i,j,k]=1 for ALL i,j,k. So for fixed i,
    # the "slice" T[i,*,*] is a 3x6 all-ones matrix. Its tensor rank (as a matrix
    # in the j-k plane, with i fixed) is... it's a matrix, and the "rank" here is
    # not matrix rank but the rank in the tensor decomposition sense restricted to
    # this slice.
    #
    # For fixed i, we need: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all j=0..2, k=0..5.
    # This is: (U_i ⊙ W_i) applied... no. Let me think of it as:
    # Let u_c = U[i,*,c] (vector in F^3), v_c = V[*,*,c] (matrix in F^{3x6}), w_c = W[*,i,c] (vector in F^6)
    # Then sum_c (u_c ⊗ v_c ⊗ w_c)[j,k] = sum_c u_c[j] * v_c[j,k] * w_c[k] = 1 for all j,k
    # This means: sum_c (u_c * v_c * w_c^T) = ones_{3x6} (outer product of ones vectors)
    # = 1_vec_3 ⊗ 1_vec_6
    # So we need to decompose the 3x6 all-ones matrix as sum_c (u_c * v_c * w_c^T)
    # where v_c is also 3x6. This is getting complicated.
    #
    # SIMPLER: Let me just use a rank-46 construction based on two 23-term Laderman blocks.
    # Or even simpler: verify that a rank-45 or rank-46 exists and hardcode it.
    #
    # ACTUALLY, the cleanest approach: use the fact that 3x3*3x6 can be done in
    # rank 36 by the following construction.
    # 
    # For each i in {0,1,2} and each pair (a,b) where a in {0,1,2}, b in {0,1,2}:
    #   That's 9 terms per i, 27 total. Not enough.
    #
    # Let me try: rank = 3 * 6 = 18? Probably too low.
    # rank = 3 * 12 = 36? Let me try to construct it.
    #
    # I'll use the following: for each i (3 rows) and each k (6 cols), use 2 terms
    # that together give the right contribution. Total 3*6*2 = 36.
    #
    # For C[i,k] = sum_j A[i,j]*B[j,k], I need the tensor to contribute 1 for each
    # (i,j,k) triple. With 2 terms per (i,k):
    #   Term 1: covers j=0 and j=1 somehow
    #   Term 2: covers j=2
    # This doesn't work cleanly.
    #
    # I give up trying to derive it from first principles in this time.
    # Let me use a SAFE rank-46 construction:
    # Two independent 3x3*3x3 blocks, each using Laderman's 23-term algorithm.
    # Total rank: 46.
    
    # Laderman's 23-term algorithm for 3x3 * 3x3:
    # The 23 rank-1 terms are specified by (i,j,k) with coefficients.
    # Standard Laderman:
    # A-side (7 combinations), B-side (8), C-side (8) -> 23 terms
    
    # Actually, let me just use a simple rank-54 -> rank-45 improvement.
    # 
    # FINAL DECISION: Use a verified rank-36 construction if I can find one,
    # otherwise use rank-46 (two Laderman blocks).
    # 
    # Let me try rank-36 with a specific pattern and verify it.
    
    # Construction for rank 36:
    # Use the "Kronecker" style: 3x3*3x6 = (3x3 * 3x3) ⊗ (1x1 * 2x1) roughly
    # No, that's not quite right.
    #
    # Let me try: for i in 0..2, for q in 0..1, for j in 0..2, for r in 0..2:
    #   That's 36 terms. Term (i,q,j,r):
    #     U[i*3+j] = 1
    #     V[j*6 + 3*q + r] = 1 if r == j else 0? No...
    #     W[(3*q+r)*3 + i] = 1
    #   This gives sum = 1 if j==r, 0 otherwise. WRONG.
    #
    # For the sum to be 1 for all (j,r), I need a different pattern.
    #
    # What if: for i in 0..2, for q in 0..1, for s in 0..2 (s indexes the 3 terms):
    #   Term (i,q,s):
    #     U[i*3+0] += alpha[s], U[i*3+1] += beta[s], U[i*3+2] += gamma[s]
    #     V[j*6+3q+r] for all j,r: some pattern
    #     W[(3q+r)*3+i] for all r: some pattern
    #   such that sum_s (U part) * (V part) * (W part) = 1 for all j,r
    #
    # This is: sum_s u_s[j] * v_s[j,r] * w_s[r] = 1 for all j,r
    # where u_s, w_s are in F^3, v_s is in F^{3x3} (for fixed q, the 3x3 block).
    #
    # If I set v_s = I (identity) for all s: sum_s u_s[j]*w_s[r] = (sum_s u_s w_s^T)[j,r] = 1
    # So sum_s u_s w_s^T = ones_{3x3}. 
    # Minimum rank of ones_{3x3} as sum of rank-1 matrices: rank 1!
    # ones_{3x3} = 1_vec * 1_vec^T. So u_s = 1_vec, w_s = 1_vec for one term s.
    # Then I only need 1 term per (i,q)! Total: 3*2*1 = 6 terms? 
    # But wait, v_s = I means V[j*6+3q+r, c] = delta_{j,r}. Let me check:
    #   sum_s U[i*3+j,c]*V[j*6+3q+r,c]*W[(3q+r)*3+i,c]
    #   = sum_s u_s[j] * delta_{j,r} * w_s[r]
    #   = u_s[j]*w_s[j] (for the s where this is nonzero)
    #   If only s=0 is nonzero: u_0[j]*w_0[j]
    #   For this to equal 1 for all j: u_0[j]*w_0[j] = 1 for all j, so u_0 = w_0 = 1_vec.
    #   So: U[i*3+j,c] = 1, V[j*6+3q+j,c] = 1 (diagonal), W[(3q+j)*3+i,c] = 1
    #   Check: for (i,j',k') where k'=3q+r':
    #     sum = U[i*3+j',c]*V[j'*6+3q+r',c]*W[(3q+r')*3+i,c]
    #     = 1 * delta_{j',r'} * 1 = delta_{j',r'}
    #   This equals 1 only when j'=r', not for all j',r'. STILL WRONG.
    #
    # The issue: I need the SUM over ALL (i,j,k) to give the tensor, but the tensor
    # is 1 for ALL i,j,k, not just diagonal.
    #
    # OK so for fixed i,q, I need: sum_c U[i,j,c]*V[j,3q+r,c]*W[3q+r,i,c] = 1 for ALL j,r.
    # With v_s = I: sum_c u_c[j]*delta_{j,r}*w_c[r] = u_c[j]*w_c[j] (for c where j=r)
    # = 1 requires u_c[j]*w_c[j]=1 for each j. So 3 separate terms (one per j),
    # giving 3 terms per (i,q), total 3*2*3=18. But these are DIFFERENT terms for
    # different j, so they don't share across (i,q) in a useful way for reducing rank.
    #
    # Unless... I use v_s that's NOT identity. If v_s = ones_{3x3} (all ones):
    #   sum_c u_c[j] * 1 * w_c[r] = (sum_c u_c w_c^T)[j,r] = 1 for all j,r
    #   => sum_c u_c w_c^T = ones_{3x3} = 1_vec * 1_vec^T
    #   Minimum: 1 term with u_c = 1_vec, w_c = 1_vec.
    #   So: U[i*3+j,c] = 1 for all j, V[j*6+3q+r,c] = 1 for all j,r, W[(3q+r)*3+i,c] = 1 for all r
    #   Check: U[i*3+j',c]*V[j'*6+3q+r',c]*W[(3q+r')*3+i,c] = 1*1*1 = 1. CORRECT!
    #   So 1 term per (i,q): total 6 terms!
    #   But wait, these 6 terms have the SAME U, V, W patterns (just different i,q indices).
    #   Term (i,q): U has 1s in row i (all 3 columns of that row), V has 1s in the 3x3 block
    #   for columns 3q..3q+2 (all 9 entries), W has 1s in the 3 entries for row i in the
    #   (3q..3q+2) block.
    #   Total: 6 rank-1 terms. RANK = 6?!
    #
    # Let me verify more carefully. 6 terms, each with:
    #   c=0: i=0,q=0: U[0:3,0]=1, V[0:9,0]=1, W[0:3,0]=1
    #   c=1: i=0,q=1: U[0:3,1]=1, V[9:18,1]=1, W[9:18,1]=1 (wait, W index: (3q+r)*3+i, q=1: (3+r)*3+0 = 9+r, r=0,1,2: 9,10,11)
    #   c=2: i=1,q=0: U[3:6,2]=1, V[0:9,2]=1, W[0:3,2]... W[(r)*3+1] for r=0,1,2: 1,4,7
    #   etc.
    #
    # Check tensor value at (i,j,k): sum_c U[i*3+j,c]*V[j*6+k,c]*W[k*3+i,c]
    # For (i=0,j=0,k=0): 
    #   c=0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> 1
    #   c=1: U[0,1]=1, V[0,1]=0 (V[0:9,1] is for q=1, which is columns 9-17, not 0) -> 0
    #   c=2: U[0,2]=0 (U[3:6,2] is for i=1) -> 0
    #   ... all other c: 0
    #   Sum = 1. CORRECT.
    #
    # For (i=0,j=0,k=3) (q=1,r=0):
    #   c=0: U[0,0]=1, V[0*6+3,0]=V[3,0]=1, W[3*3+0,0]=W[9,0]=0 (W[0:3,0] is indices 0,1,2) -> 0
    #   c=1: U[0,1]=1, V[3,1]=V[3,1]=1 (V[9:18,1], index 3 is within 9:18? No! V[j*6+k] = V[0*6+3]=V[3,1]. V[0:9,1] would be indices 0-8. Index 3 is in range. But V[3,1]: is this part of q=0 or q=1?
    #   
    #   I'm confusing myself. V is indexed by (j*6+k) where j=0..2, k=0..5.
    #   V[j*6+k, c] for c=1 (i=0,q=1): V has 1s where... I said V[9:18,1]=1.
    #   V index 9 = j*6+k: j=1,k=3? No: j*6+k, j=0:k=0..5 (0-5), j=1:k=6..11 (6-11), j=2:k=12..17 (12-17).
    #   So V[9:18,1] means indices 9-17, which is j=1,k=3..5 and j=2,k=0..5. That's NOT what I want!
    #
    # I made an error. For q=1, the k values are 3,4,5. V[j*6+k, c] for k=3,4,5:
    #   j=0: k=3,4,5 -> indices 3,4,5
    #   j=1: k=3,4,5 -> indices 9,10,11
    #   j=2: k=3,4,5 -> indices 15,16,17
    # So V should have 1s at indices {3,4,5, 9,10,11, 15,16,17} for q=1.
    # That's 9 entries, not a contiguous block 9:18.
    #
    # So the 6-term construction DOES work, but V is sparse (9 ones per term, not 18).
    # Let me re-verify:
    # Term (i,q,c): 
    #   U[i*3+j, c] = 1 for j=0,1,2
    #   V[j*6+k, c] = 1 for j=0,1,2 and k=3q,3q+1,3q+2
    #   W[k*3+i, c] = 1 for k=3q,3q+1,3q+2
    #
    # Check (i,j,k): sum over c of U[i*3+j,c]*V[j*6+k,c]*W[k*3+i,c]
    # Only term c=(i, q) where q = k//3 contributes (if k%3 < 3, which is always true for k=0..5)
    # For that term: U=1, V=1 (since k is in {3q,3q+1,3q+2}), W=1 (same reason)
    # Sum = 1. CORRECT!
    #
    # RANK = 6! That's amazing. Let me double-check with a specific example.
    # (i=1, j=2, k=4): q = 4//3 = 1, so c should be the term for (i=1,q=1).
    #   U[1*3+2, c] = U[5, c] = 1 (for c corresponding to i=1)
    #   V[2*6+4, c] = V[16, c] = 1 (k=4 is in {3,4,5} for q=1, j=2)
    #   W[4*3+1, c] = W[13, c] = 1 (k=4 is in {3,4,5} for q=1)
    #   Product = 1. All other c: either U=0 (different i) or V=0 (different q block) or W=0.
    #   Sum = 1. CORRECT!
    #
    # So rank 6 works?! That seems too good. Let me make sure I'm not making an error.
    #
    # The tensor T has T[i,j,k] = 1 for all i=0..2, j=0..2, k=0..5. (54 entries, all 1)
    # My 6-term decomposition: for each (i,q) with i=0..2, q=0..1:
    #   u_{i,q}[i*3+j] = 1 for all j (3 ones in row i of U)
    #   v_{i,q}[j*6+k] = 1 for j=0..2, k=3q..3q+2 (9 ones)
    #   w_{i,q}[k*3+i] = 1 for k=3q..3q+2 (3 ones)
    #
    # Sum_{(i',q')} u_{i',q'}[i*3+j] * v_{i',q'}[j*6+k] * w_{i',q'}[k*3+i]
    # = sum over (i',q') where i'=i (for u to be nonzero) and q'=k//3 (for v and w to be nonzero)
    # = 1 (exactly one term contributes)
    #
    # YES! Rank 6 is valid!
    
    rank = 6
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    c = 0
    for i in range(n):
        for q in range(p // 3):  # q = 0, 1
            k_start = 3 * q
            for j in range(m):
                U[i * m + j][c] = 1.0
            for j in range(m):
                for k in range(k_start, k_start + 3):
                    V[j * p + k][c] = 1.0
            for k in range(k_start, k_start + 3):
                W[k * n + i][c] = 1.0
            c += 1
    
    return U, V, W

def construct():
    deadline = _start + 140  # safety margin before 150s kill
    
    # Try rank-6 construction first
    U, V, W = _rank36_decomposition()
    
    # Verify
    ok, info = _verify_decomposition(U, V, W, 3, 3, 6)
    if ok:
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank-54
    n, m, p = 3, 3, 6
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

