"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def _verify_decomposition(n, m, p, U, V, W):
    """Verify that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta_{i,k} for all i,j,k.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    """
    rank = len(U[0])
    # Build tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For efficiency with small dims, do it directly
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                expected = 1 if i == k else 0
                if s != expected:
                    return False
    return True

def _to_frac(x):
    """Convert a number to Fraction, handling floats that are close to rationals with small denom."""
    if isinstance(x, (int, np.integer)):
        return Fraction(int(x), 1)
    if isinstance(x, (float, np.floating)):
        xf = float(x)
        # Check if close to integer
        if abs(xf - round(xf)) < 1e-9:
            return Fraction(int(round(xf)), 1)
        # Try small denominators
        for d in range(2, 1000):
            num = round(xf * d)
            if abs(xf - num/d) < 1e-9:
                return Fraction(num, d)
        # Fallback: use Fraction(xf) which gives exact binary representation
        return Fraction(xf)
    return x  # already a Fraction or other

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s deadline, 10s safety margin

    # Schoolbook baseline (rank = 54)
    sb_rank = n * m * p
    U_sb = [[0] * sb_rank for _ in range(n * m)]
    V_sb = [[0] * sb_rank for _ in range(m * p)]
    W_sb = [[0] * sb_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1

    # Known optimal: rank 54 (schoolbook) for 3x3x6
    # But let's try to find a better decomposition using structured search.
    # Strategy: try to exploit the structure. The tensor for multiplying (n x m) by (m x p)
    # has tensor T[i,j,k] = delta_{i,k} (for the "output" index i from first matrix, k from second... 
    # wait, let me re-read the convention).
    
    # The output is W[k*n+i, r]. So the output matrix C[i,k] where i is row of A (0..n-1) and k is col of B (0..p-1).
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T_{i,j,k} = delta_{i,k}? No.
    # The multiplication tensor T_{i,j,k} for (A B) where A is n x m, B is m x p:
    # (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor is sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}... 
    # Actually the standard tensor is: T = sum_{i,j,k} e_i (A) * e_j (B) * e_k (C) where C_{i,k} = sum_j A_{i,j}B_{j,k}
    # So T_{i,j,k} = 1 if the output entry (i,k) depends on A_{i,j} and B_{j,k}, which it does for all i,j,k.
    # Wait no. The multiplication tensor for matrix multiplication (n x m) * (m x p) is:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i^{(A)} ⊗ e_j^{(B)} ⊗ e_k^{(C)}
    # where C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor has a nonzero entry at (i, j, k) for ALL combinations of i, j, k.
    # The rank of this tensor is what we want to minimize.
    
    # For n=3, m=3, p=6: schoolbook rank = 3*3*6 = 54.
    # The known lower bound for 3x3x3 is 19 (Strassen-like), and upper bound is 23 (or lower with recent results).
    # For 3x3x6, the rank should be at least... well, it's a "slice" of 3x3x3x6 or something.
    # Actually, 3x3x6 matrix multiplication can be seen as 6 independent 3x3 matrix-vector products (each column of B).
    # Each 3x3 matrix-vector product (3x3 times 3x1) has rank at most 9 (schoolbook) but can be done in rank 7 
    # using a variant of Strassen or other methods? Actually for matrix-vector it's just 9 scalar mults in schoolbook.
    # But we can share computations across columns.
    
    # Let's try a different approach: use the fact that we're multiplying 3x3 by 3x6.
    # This is equivalent to computing C = A * B where A is 3x3 and B is 3x6.
    # We can view B as 6 column vectors b_1, ..., b_6, and compute A * b_k for each k.
    # The tensor for this is: T = sum_{i,j,k} e_i (A) ⊗ e_j (B_k) ⊗ e_{(k,n)+i} (C)
    # Actually with the W[k*n+i] convention, the output index is k*n+i where k is 0..p-1 and i is 0..n-1.
    
    # Let me try to use known results. For n x n x n matrix multiplication:
    # n=1: rank 1
    # n=2: rank 7 (Strassen)
    # n=3: rank <= 23 (best known), lower bound 19
    
    # For 3x3x6, one approach: think of it as 2 independent 3x3x3 multiplications (splitting the 6 columns into two groups of 3).
    # That would give rank 2 * rank(3x3x3) <= 2 * 23 = 46, which is better than 54.
    # Or we could try to do it more cleverly.
    
    # Actually, let's try to search for a decomposition using random initialization and optimization.
    # But that's complex. Let me try a structured approach first.
    
    # Approach: Use the fact that 3x3x6 = 3x3x3 + 3x3x3 (block diagonal in some sense).
    # If we know a rank-r decomposition for 3x3x3, we can get rank 2r for 3x3x6.
    # Best known rank for 3x3x3 is 23, so rank 46 for 3x3x6.
    
    # But maybe we can do better by sharing across the two blocks.
    # Let's try a direct search with a smaller rank.
    
    # First, let's try to find a 3x3x3 decomposition of rank <= 23, then use it.
    # Actually, for the purpose of this task, let's try to find any improvement over 54.
    
    # Let me try a random search approach for rank 46 (2 * 23) or even lower.
    # Actually, a simpler idea: use rank 36 by viewing 3x3x6 as 3 copies of 3x3x2.
    # 3x3x2 matrix multiplication: rank at most 12 (schoolbook 3*3*2=18), but can be lower.
    # For 3x3x2, the tensor has 18 entries. Can we do better than 18?
    # Actually 3x3x2 is just 6 matrix-vector products of dimension 3, each needing 3 mults = 9 per vector, total 18.
    # But with Strassen-like tricks on the 3x3 part... hmm.
    
    # Let me just try to do a numerical search for a decomposition with rank < 54.
    # I'll use a random initialization and try to solve the tensor decomposition problem.
    
    # Actually, let me try a simpler structured approach:
    # For 3x3x6, we can use the following trick:
    # A (3x3) * B (3x6) = A * [B1 | B2] where B1, B2 are 3x3.
    # = [A*B1 | A*B2]
    # If we have a rank-r decomposition for 3x3x3, we get rank 2r for 3x3x6.
    
    # Known: Strassen gives rank 7 for 2x2x2.
    # For 3x3x3, one can use a "bordered" version or direct algorithms.
    # The best known exact rank for 3x3x3 over Q is 23 (Bini et al.), but I'm not sure of the exact formula.
    
    # Let me try a different approach: use the "Laser Method" style or just try to find 
    # a decomposition programmatically.
    
    # Actually, let me try the simplest possible improvement:
    # Use rank 36 by computing A*B where we treat B as having 6 columns,
    # and use a rank-6 decomposition for the 3x3x2 case... no that doesn't work.
    
    # Let me try: can we do 3x3x6 in rank 48? That's 3x3x3 (rank 24) + 3x3x3 (rank 24).
    # Or rank 46 if we use rank 23 for each 3x3x3.
    
    # For now, let me try to construct a rank-46 decomposition using a known 3x3x3 rank-23 method.
    # Actually, I recall that the 3x3x3 matrix multiplication can be done in rank 23 
    # using a specific algorithm, but I don't have the exact formulas memorized.
    
    # Alternative: use the fact that 3x3x6 can be done by first doing a 3x3x3 multiplication
    # for the first 3 columns, and another for the last 3 columns.
    # If I use the schoolbook for each, that's 9 + 9 = 18 per 3x3x3, total 36. Wait, that's better than 54!
    # Oh wait, I was confused. 3x3x3 schoolbook is rank 9 (3*3*3=27? No, 3*3*3=27).
    # Hmm, 3x3x3 schoolbook rank is 3*3*3 = 27.
    # So 2 * 27 = 54, same as 3x3x6 schoolbook. No improvement.
    
    # But Strassen reduces 2x2x2 from 8 to 7. Can we get a similar reduction for 3x3x3?
    # The best known rank for 3x3x3 is 23 (over C), which is less than 27.
    # So 2 * 23 = 46 < 54. That's an improvement!
    
    # Let me try to find a rank-23 decomposition for 3x3x3.
    # Actually, this is a well-studied problem. Let me try to use a known construction.
    
    # For 3x3x3, one approach is to use the "middle product" or other algebraic methods.
    # But implementing this from scratch is complex.
    
    # Let me try a numerical optimization approach instead.
    # I'll initialize U, V, W randomly with rank r and try to minimize the error.
    # Then round to rationals and verify.
    
    def _try_random_rank(r, max_iters=200, seed=42):
        """Try to find a rank-r decomposition using random init and gradient-like updates."""
        if time.time() > deadline:
            return None
        
        rng = np.random.RandomState(seed)
        # Initialize
        U = rng.randn(n*m, r)
        V = rng.randn(m*p, r)
        W = rng.randn(p*n, r)
        
        # The tensor T[i,j,k] should equal 1 if i==k (in the appropriate sense) else 0.
        # Actually, the multiplication tensor for (n x m) * (m x p) -> (n x p) is:
        # T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # because C_{i,k} = sum_j A_{i,j} B_{j,k}
        # Wait, that means ALL entries of the tensor are 1? No, that's not right.
        
        # Let me re-read the problem. The output is W[k*n+i, r].
        # So the output matrix has entries indexed by (k, i) where k is 0..p-1 (column of result) 
        # and i is 0..n-1 (row of result). The result C[i,k] = sum_j A[i,j] B[j,k].
        
        # The tensor is: sum_{i,j,k} A[i,j] * B[j,k] * C[k,i] ... no.
        # The tensor decomposition is: T = sum_r U_r ⊗ V_r ⊗ W_r
        # where U_r has entries U[i*m+j, r], V_r has entries V[j*p+k, r], W_r has entries W[k*n+i, r].
        # And T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # should equal the multiplication tensor, which is:
        # T[i,j,k] = 1 if the output entry corresponding to (i,k) uses A[i,j] and B[j,k]
        # For matrix multiplication, C[i,k] = sum_j A[i,j] B[j,k], so every (i,j,k) triple contributes.
        # Thus T[i,j,k] = 1 for all i,j,k.
        
        # Wait, that can't be right either, because then the rank would be at least the number of 
        # linearly independent slices, which is n*p = 18 in this case (3*6).
        # Actually, the tensor T[i,j,k] = 1 for all i,j,k is a rank-1 tensor! 
        # Because T[i,j,k] = 1 * 1 * 1 for all i,j,k.
        
        # I think I'm confusing the tensor convention. Let me re-read.
        
        # "output factor follows the official trace convention W[k*n+i,r]"
        # "U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)"
        # "exactly decompose the matrix multiplication tensor"
        
        # The matrix multiplication tensor for multiplying an (n x m) matrix by an (m x p) matrix
        # is T in R^{n x m x p} where T[i,j,k] = delta_{i,k}... no.
        
        # Actually, I think the correct interpretation is:
        # We want C = A * B where A is n x m, B is m x p, C is n x p.
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor is T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the coefficient is 1 if and only if
        # this term appears in the sum for some output entry.
        
        # Actually, the standard matrix multiplication tensor is:
        # T = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_i^{(A)} ⊗ e_j^{(B)} ⊗ e_k^{(C)}
        # where the output C has entries C_{i,k} (i row, k col), and C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # In this tensor, the "C" index corresponds to the pair (i,k), so we have p*n output entries.
        # The tensor entry T[i,j,(k,n)+i] = 1 for all i,j,k, and 0 otherwise.
        
        # Wait, with W[k*n+i, r], the output index is k*n+i where k is 0..p-1 and i is 0..n-1.
        # So W index k*n+i corresponds to output entry C[i,k] (row i, col k).
        # The tensor condition is:
        # T[i, j, k*n+i] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # T[i, j, w] = 0 for w not of the form k*n+i for some k
        
        # So the tensor has n*m*p = 54 nonzero entries (all 1s) at positions (i, j, k*n+i).
        
        # Let me verify this interpretation with the schoolbook:
        # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1 for each (i,j,k).
        # Then sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 (from the term where c corresponds to (i,j,k))
        # and 0 otherwise (since each (i,j,k) has its own column).
        # So T[i,j,k*n+i] = 1 for all i,j,k, and T[i,j,w] = 0 for other w.
        # This matches!
        
        # So the tensor is: T[i,j,w] = 1 if w = k*n+i for some k (i.e., w mod n == i and w < p*n), else 0.
        # More precisely, T[i,j,w] = 1 if w % n == i and w // n < p, else 0.
        
        # Now, what's the rank of this tensor?
        # It's a 3 x 3 x 18 tensor with T[i,j,w] = 1 if w%n == i, else 0.
        # This can be written as: T[i,j,w] = sum_{k=0}^{p-1} delta_{i, w%n} * delta_{k, w//n}
        # = delta_{i, w%n} (independent of j!)
        # Wait, T[i,j,w] doesn't depend on j! It only depends on i and w.
        # T[i,j,w] = 1 if w%n == i, else 0, for all j.
        
        # So the tensor is: T = sum_{i=0}^{n-1} sum_{k=0}^{p-1} e_i ⊗ (sum_j e_j) ⊗ e_{k*n+i}
        # = (sum_i e_i) ⊗ ... no, let me be more careful.
        
        # T[i,j,w] = 1 if w = k*n+i for some k, else 0.
        # This equals sum_{k=0}^{p-1} delta_{w, k*n+i}.
        # So T = sum_{i=0}^{n-1} sum_{k=0}^{p-1} e_i ⊗ (1,1,...,1)^T ⊗ e_{k*n+i}
        # where (1,...,1) is the all-ones vector of length m.
        
        # This is a sum of n*p = 3*6 = 18 rank-1 tensors!
        # So the rank is at most 18!
        
        # Let me verify: can I write this as a sum of 18 rank-1 terms?
        # Term (i,k): e_i (in A-space) ⊗ ones_m (in B-space) ⊗ e_{k*n+i} (in C-space)
        # U column for (i,k): U[i*m+j, (i,k)] = 1 for all j, 0 otherwise.
        # V column for (i,k): V[j*p+k', (i,k)] = 1 for all j, 0 otherwise. Wait, this doesn't separate i and k properly.
        
        # Let me redo this. The rank-1 term for fixed (i,k) is:
        # u_{i,k} in R^{n*m}, v_{i,k} in R^{m*p}, w_{i,k} in R^{p*n}
        # such that u_{i,k} ⊗ v_{i,k} ⊗ w_{i,k} contributes to T[i',j,w'] only when i'=i and w'=k*n+i.
        
        # u_{i,k}[i*m+j] = 1 for all j (since the term doesn't depend on j)
        # v_{i,k}[j*p+k'] = 1 for all j, and... hmm, this needs to be independent of k' too?
        
        # Actually, I made an error. Let me re-examine.
        # T[i,j,w] = 1 if w%n == i, else 0. This is independent of j.
        # So T = sum_{i=0}^{2} sum_{k=0}^{5} e_i ⊗ (1_m) ⊗ e_{k*n+i} where 1_m = (1,1,1).
        # This is a sum of 18 rank-1 tensors, so rank <= 18.
        
        # Can we do better? The tensor T[i,j,w] = delta_{w%n, i} (with w//n < p).
        # As a 3 x (3*18) matrix (fixing i), it's a 3x54 matrix.
        # For each i, the slice T[i,:,:] is a 3x54 matrix with row j all equal to (0,...,0,1,0,...,0,1,0,...)
        # where the 1s are at positions w = k*n+i for k=0..5.
        # So all 3 rows of T[i,:,:] are identical.
        # The rank of T[i,:,:] is 1 (since all rows are the same nonzero vector).
        # But the overall tensor rank is not just the max slice rank.
        
        # Let me try to construct the rank-18 decomposition explicitly.
        # For each (i,k) pair (i in 0..2, k in 0..5):
        #   U column: U[i*m+j, idx] = 1 for all j in 0..2, 0 otherwise.
        #   V column: V[j'*p+k', idx] = 1 for all j' in 0..2 and all k' in 0..5, 0 otherwise.
        #     Wait, that would make V the same for all (i,k), which is fine.
        #     But then the product U*V*W would not separate the k indices properly.
        
        # Let me think again. The rank-1 tensor for (i,k) should be:
        #   u_{i,k} = e_i ⊗ 1_m (length 9, with 1s at positions i*3+j for j=0,1,2)
        #   v_{i,k} = 1_m ⊗ e_k (length 18, with 1s at positions j*6+k for j=0,1,2)
        #   w_{i,k} = e_{k*n+i} (length 18, with 1 at position k*3+i)
        
        # Then (u_{i,k} ⊗ v_{i,k} ⊗ w_{i,k})[i', j, w'] = u_{i,k}[i'*3+j] * v_{i,k}[j*6+k'] * w_{i,k}[k'*3+i']
        # = (1 if i'=i else 0) * (1 if k'=k else 0) * (1 if w'=k'*3+i' else 0)
        # = 1 if i'=i, k'=k, w'=k*3+i, else 0.
        
        # Summing over all (i,k): T[i',j,w'] = sum_{i,k} [i'=i] [k'=k] [w'=k'*3+i']
        # where k' = w'//3 and i' should equal w'%3.
        # = 1 if w'%3 == i' and w'//3 < 6, else 0.
        # This is exactly what we want!
        
        # So the rank-18 decomposition is:
        # For idx = (i,k) -> i*6+k (or some mapping):
        #   U[i*3+j, idx] = 1 for j=0,1,2 (i.e., row i*3+j is 1)
        #   V[j*6+k, idx] = 1 for j=0,1,2 (i.e., column k of V is all 1s)
        #   W[k*3+i, idx] = 1
        
        # Let me verify: sum_{idx} U[i'*3+j, idx] V[j*6+k', idx] W[k'*3+i', idx]
        # = sum_{i,k} [i'=i] [k'=k] [k'*3+i' == k*3+i]  (since W[k*3+i, idx]=1 only when idx corresponds to (i,k) and we're evaluating at W[k'*3+i', idx])
        # Wait, I need to be more careful.
        
        # idx = (i,k), mapped to column c = i*6+k (say).
        # U[i*3+j, c] = 1 for all j, 0 otherwise.
        # V[j*6+k, c] = 1 for all j, 0 otherwise.
        # W[k*3+i, c] = 1.
        
        # sum_c U[i'*3+j, c] * V[j*6+k', c] * W[k'*3+i', c]
        # = sum_{i,k} U[i'*3+j, i*6+k] * V[j*6+k', i*6+k] * W[k'*3+i', i*6+k]
        # = sum_{i,k} [i==i'] * [k==k'] * [k'*3+i' == k*3+i]
        # = [i'==i'] * [k'==k'] * [k'*3+i' == k'*3+i']  (when i=i', k=k')
        # = 1
        
        # Wait, that's always 1, regardless of i', j, k'. That's not right.
        # Let me recheck.
        
        # U[i*3+j, c] where c = i'*6+k' (renaming to avoid confusion):
        # U[row, col] = 1 if row = i_col*3 + j_row for some j_row, and col = i_col*6+k_col
        # So U[i'*3+j, c] = 1 if i' == i_c (the i-index of column c), 0 otherwise.
        
        # V[j*6+k', c] = 1 if k' == k_c (the k-index of column c), 0 otherwise.
        
        # W[k'*3+i', c] = 1 if k'*3+i' == k_c*3+i_c, else 0.
        
        # So the sum is: sum_{c} [i_c == i'] [k_c == k'] [k'*3+i' == k_c*3+i_c]
        # = sum_{c} [i_c == i'] [k_c == k'] [k'*3+i' == k_c*3+i_c]
        # When i_c == i' and k_c == k', the last condition is k'*3+i' == k'*3+i', which is true.
        # So the sum equals 1 if there exists a column c with i_c == i' and k_c == k', which there is (c = i'*6+k').
        # So the sum is always 1!
        
        # But we want T[i',j,w'] = 1 if w'%3 == i', else 0.
        # Here w' = k'*3+i', so w'%3 == i' is always true by construction.
        # So T[i',j,w'] should be 1 for all i',j,k' where w'=k'*3+i'.
        # And 0 for w' not of the form k'*3+i'.
        
        # In our sum, we're only computing T[i',j,w'] for w' = k'*3+i' (since W is indexed by k'*3+i').
        # For other w', W[w', c] = 0 for all c (since W only has 1s at positions k*3+i).
        # So T[i',j,w'] = 0 for w' not of the form k*3+i.
        
        # And for w' = k'*3+i', T[i',j,w'] = 1 as shown.
        
        # So the rank-18 decomposition is correct!
        
        # Let me construct it:
        rank_18 = n * p  # 3 * 6 = 18
        U18 = [[0] * rank_18 for _ in range(n * m)]
        V18 = [[0] * rank_18 for _ in range(m * p)]
        W18 = [[0] * rank_18 for _ in range(p * n)]
        
        for i in range(n):
            for k in range(p):
                c = i * p + k  # column index
                for j in range(m):
                    U18[i * m + j][c] = 1
                    V18[j * p + k][c] = 1
                W18[k * n + i][c] = 1
        
        # Verify
        if _verify_decomposition(n, m, p, U18, V18, W18):
            # Rank 18 is much better than 54!
            # But can we do even better?
            # The tensor T[i,j,w] = 1 if w%n == i, else 0.
            # This is a 3x3x18 tensor. The "slice" T[:,:,w] for fixed w is:
            # If w = k*n+i, then T[:,:,w] has T[i,j,w] = 1 for all j, and 0 for other i.
            # So T[:,:,w] = e_i ⊗ 1_m^T (a rank-1 matrix) for w = k*n+i.
            # For other w (none, since all w in 0..17 are of the form k*3+i), it's 0.
            
            # The tensor can be written as: T = sum_{i,k} e_i ⊗ 1_m ⊗ e_{k*n+i}
            # This is 18 rank-1 terms. Can we merge any of them?
            # Two rank-1 terms can be merged if their u-vectors are proportional, v-vectors are proportional, etc.
            # Here, u_{i,k} = e_i ⊗ 1_m, which depends only on i, not k.
            # v_{i,k} = 1_m ⊗ e_k, which depends only on k, not i.
            # w_{i,k} = e_{k*n+i}, which depends on both i and k.
            
            # Since u_{i,k} = u_i (independent of k) and v_{i,k} = v_k (independent of i),
            # the tensor is T = sum_{i,k} u_i ⊗ v_k ⊗ w_{i,k}
            # This is a "matrix" of rank-1 tensors indexed by (i,k).
            # The rank of this tensor is at least the rank of the matrix [w_{i,k}] as vectors... 
            # actually, it's related to the rank of a certain matrix.
            
            # Specifically, T = sum_{i,k} u_i ⊗ v_k ⊗ w_{i,k}
            # This can be viewed as a bilinear form in (u-space, v-space) with values in W-space.
            # The rank is at least the rank of the matrix M[i,k] = w_{i,k} (as vectors in R^{p*n}).
            # But w_{i,k} = e_{k*n+i} are standard basis vectors, so the matrix [w_{i,k}] has rank p*n = 18.
            # Wait, that's a 3x6 matrix of vectors, so it's a 3x6x18 tensor of 0s and 1s.
            # The "fiber" rank or something... 
            
            # Actually, the rank of T = sum_{i,k} u_i ⊗ v_k ⊗ w_{i,k} is at least 
            # the rank of the matrix (sum_i u_i ⊗ something) ... this is getting complicated.
            
            # Let me just try to see if rank < 18 is possible by numerical search.
            # But given the structure, rank 18 might be optimal.
            
            # For now, return rank 18.
            return {"U": U18, "V": V18, "W": W18}
        
        # If verification failed (shouldn't happen), fall back to schoolbook
        return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
