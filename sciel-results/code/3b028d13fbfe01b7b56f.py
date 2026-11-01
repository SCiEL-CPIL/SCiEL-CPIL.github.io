<<<<<<< SEARCH
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
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
=======
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import time
import itertools
from fractions import Fraction

def _schoolbook(n, m, p):
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

def _verify(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the multiplication tensor T[i,j,k] = delta_{i,k}
    # Decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == delta_{i,k}
    # We check all (i,j,k) entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    u_val = U[i * m + j][c]
                    v_val = V[j * p + k][c]
                    w_val = W[k * n + i][c]
                    s += Fraction(u_val) * Fraction(v_val) * Fraction(w_val)
                expected = Fraction(1) if i == k else Fraction(0)
                if s != expected:
                    return False
    return True

def _try_rank(n, m, p, target_rank, deadline):
    """
    Try to find a rank-`target_rank` decomposition.
    Uses structured search with laser method ideas.
    For 2x2x13, we look for decompositions where the structure exploits
    the small n,m dimensions.
    
    Key insight: For n=2, m=2, p=13, the tensor is T[i,j,k] = delta_{i,k} for i,k in {0,1}.
    This means for each j, the 2x13 slice is the identity padded with zeros.
    
    We use a parameterized ansatz. Each rank-1 term is (u, v, w) where:
    u is in R^{n*m} = R^4, v is in R^{m*p} = R^{26}, w is in R^{p*n} = R^{26}.
    
    The constraint is: sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = delta_{i,k}
    
    For n=2, this gives 2*2*13 = 52 scalar equations (but many are trivially 0=0).
    Non-trivial: for i=k=0: sum_c u_c[i,j]*v_c[j,0]*w_c[0,i] = 1 for each j
    for i=k=1: sum_c u_c[i,j]*v_c[j,1]*w_c[1,i] = 1 for each j
    for i!=k: sum_c u_c[i,j]*v_c[j,k]*w_c[k,i] = 0
    
    This is a system of polynomial equations. We search over small integer/half-integer values.
    """
    start = time.time()
    
    # For 2x2x13, lower bound on rank:
    # The tensor has rank at least max(n*m, m*p/p, ...) 
    # Known: rank(2x2x13) >= 2*13 = 26 (from slice arguments)
    # Actually, for multiplying (2x2)*(2x13), the rank is at least 2*min(2,13) = 4? No.
    # The standard lower bound is n*p for the first factor being n x m.
    # Actually rank >= n*p = 2*13 = 26? Let me think...
    # For A(2x2) * B(2x13) = C(2x13):
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is T_{i,j,k} = delta_{j,k}... wait no.
    # T_{i,j,k} where A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 if this is the identity contribution... 
    # Actually the multiplication tensor has T[i,j,k] such that
    # C[i,k] = sum_j A[i,j] * T[i,j,k] * B[j,k]... no.
    # Standard: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: C = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * e_{i,k}
    # So T[i,j,k] = 1 for all i,j,k? No, that's not right either.
    # 
    # Let me re-read the problem. The output factor follows W[k*n+i, r].
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # 
    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k}
    # 
    # Wait, that doesn't make sense for matrix multiplication. Let me think again.
    # 
    # For matrix multiplication C = A * B where A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The tensor decomposition expresses this as:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    # 
    # No wait, the standard form is:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # 
    # Hmm, but W is also there. Let me look at the schoolbook:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # Product: sum_c A[i,j] * B[j,k] * C[k,i]... that's not right.
    # 
    # I think the convention is:
    # The multiplication tensor T has components T[i*m+j, j*p+k, k*n+i] = 1
    # And the decomposition is: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # where a indexes (i,j), b indexes (j,k), c indexes (k,i)
    # 
    # So: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k}... 
    # No, it should equal 1 when the indices are consistent.
    # 
    # Actually for matrix multiplication, the tensor is:
    # T_{(i,j),(j,k),(k,i)} = 1 (with the understanding that i,j,k are fixed)
    # 
    # Wait, I think the correct statement is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for ALL i, j, k
    # 
    # No, that can't be right because then the decomposition would be trivial.
    # 
    # Let me re-examine. The matrix multiplication map is:
    # (A, B) -> A*B
    # where A is n x m, B is m x p, result is n x p.
    # 
    # The tensor T is in R^{n*m} x R^{m*p} x R^{n*p} (note: n*p, not p*n).
    # T[i*m+j, j*p+k, i*p+k] = 1
    # 
    # But the problem says W has shape (p*n, rank). So W[k*n+i, r].
    # If n=p=2, then p*n = n*p = 4 and k*n+i = i*p+k.
    # But here p=13, n=2, so p*n = 26 and n*p = 26.
    # k*n+i where k in [0,12], i in [0,1]: this gives indices 0..25.
    # i*p+k where i in [0,1], k in [0,12]: this also gives indices 0..25.
    # But k*n+i != i*p+k in general!
    # k*n+i = 2k+i, i*p+k = 13i+k.
    # 
    # So the W index is k*n+i = 2k+i, NOT i*p+k.
    # 
    # This means the tensor being decomposed is NOT the standard matrix multiplication tensor.
    # Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]."
    # 
    # I think the convention is that the third factor of the tensor is indexed by (k,i) 
    # where k is the row of B (which becomes the column of C) and i is the row of A.
    # So W[k*n+i, r] means the index is k*n + i = 2k + i for n=2.
    # 
    # The tensor to decompose: T[a, b, c] where:
    # a = i*m + j (i in [0,n), j in [0,m)) -> indexes A[i,j]
    # b = j*p + k (j in [0,m), k in [0,p)) -> indexes B[j,k]  
    # c = k*n + i (k in [0,p), i in [0,n)) -> indexes C[k,i]?? 
    # 
    # But C should be n x p, so C[i,k] with i in [0,n), k in [0,p).
    # The index k*n + i puts k as the "row" (0..p-1) and i as the "column" (0..n-1).
    # So this is C^T or C with transposed indexing.
    # 
    # The tensor is: T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # 
    # And the decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # 
    # Wait, that means ALL entries of the tensor are 1? That's just the all-ones tensor,
    # which has rank 1. That can't be right.
    # 
    # I must be misunderstanding. Let me look at the schoolbook more carefully.
    # In the schoolbook, for each (i,j,k) triple, we set one term.
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    # 
    # The product U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] is 1 only when 
    # the specific (i,j,k) matches. For a fixed (i,j,k), only the term with
    # that exact (i,j,k) contributes. So the sum gives 1 for each (i,j,k) pair
    # where all three indices match.
    # 
    # But what is the tensor? I think the tensor is:
    # T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    #           = 0 otherwise
    # 
    # This is the standard matrix multiplication tensor! The condition is that
    # the j-index in A and B match, and the i-index in A matches the i-index
    # in C (via k*n+i), and the k-index in B matches the k-index in C.
    # 
    # So: T[i*m+j, j*p+k, k*n+i] = 1, and T[a,b,c] = 0 otherwise.
    # 
    # The decomposition: sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
    # 
    # For the schoolbook: each (i,j,k) gives one rank-1 term, and the product
    # U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] is 1 only at that specific (a,b,c).
    
    # So we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    # And: sum_r U[a,r] * V[b,r] * W[c,r] = 0 for all (a,b,c) not of the form (i*m+j, j*p+k, k*n+i)
    
    # This is the standard matrix multiplication tensor.
    # For n=2, m=2, p=13:
    # - U is 4 x rank
    # - V is 26 x rank  
    # - W is 26 x rank
    # - 52 equations = 1 (the diagonal ones)
    # - Many equations = 0 (the off-diagonal ones)
    
    # Lower bound: The rank of the matrix multiplication tensor for (n,m,p)
    # is at least n*p (by considering a slice). For n=2, p=13: rank >= 26.
    # Also at least m*max(n,p) = 2*13 = 26.
    # 
    # Known results: rank(2x2xp) for various p.
    # For 2x2x4: rank = 7 (Strassen)
    # For 2x2xp: the rank is known to be at most 2p+2 for p >= 4 (using Strassen-like methods)
    # Actually, for 2x2xp, the rank is at most 2p + 2 - 2*log2(p+1) or similar.
    # 
    # Let me think about what's achievable. For 2x2x13:
    # - Schoolbook: 2*2*13 = 52
    # - Lower bound: 2*13 = 26
    # - Strassen for 2x2x2 gives rank 7 instead of 8.
    # - For 2x2xp, one can use the "laser method" or structured approaches.
    
    # Let me try a structured search. For small ranks, we can use the fact that
    # the problem has a lot of structure.
    
    # Approach: Use the fact that for n=2, the tensor has a special form.
    # T[i*m+j, j*p+k, k*n+i] = 1
    # 
    # For i=0: T[j, j*p+k, k] = 1  (since 0*2+j=j, k*2+0=2k... wait)
    # i=0: a=0*2+j=j, b=j*p+k, c=k*2+0=2k
    # i=1: a=1*2+j=2+j, b=j*p+k, c=k*2+1=2k+1
    # 
    # So the non-zero entries are:
    # For i=0: (j, 13j+k, 2k) = 1 for j in {0,1}, k in {0,...,12}
    # For i=1: (2+j, 13j+k, 2k+1) = 1 for j in {0,1}, k in {0,...,12}
    
    # Total non-zero entries: 2 * 2 * 13 = 52
    
    # This is a very sparse tensor. The rank could potentially be much less than 52.
    
    # Let me try to find a decomposition using a parameterized approach.
    # For rank r, we need to find U (4xr), V (26xr), W (26xr) such that
    # sum_c U[a,c] V[b,c] W[c_idx,c] = T[a,b,c_idx]
    
    # This is a system of polynomial equations. For small r, we can try
    # to solve it using the fact that many entries are zero.
    
    # Strategy: Since the tensor is very structured, try ansätze where
    # U, V, W have simple forms.
    
    # For now, let's try ranks from the lower bound up to schoolbook.
    # Start with the lower bound of 26.
    
    # Actually, let me first check if rank 26 is achievable.
    # The lower bound of 26 comes from: consider the 26x26 matrix
    # M[b,c] = T[a0, b, c] for a fixed a0. This matrix must have rank <= r.
    # For a0 = 0 (i=0, j=0): M[b,c] = 1 if b=k and c=2k, else 0.
    #   b = 0*13+k = k, c = 2k. So M[k, 2k] = 1 for k=0..12.
    #   This is a 26x26 matrix with 13 ones on a "diagonal" (not standard diagonal).
    #   Rank of this matrix: it has 13 non-zero rows (rows 0..12), each with one 1.
    #   So rank = 13.
    # 
    # For a0 = 1 (i=0, j=1): b = 1*13+k = 13+k, c = 2k.
    #   M[13+k, 2k] = 1 for k=0..12.
    #   13 non-zero rows, each with one 1. Rank = 13.
    # 
    # For a0 = 2 (i=1, j=0): b = k, c = 2k+1.
    #   M[k, 2k+1] = 1 for k=0..12. Rank = 13.
    # 
    # For a0 = 3 (i=1, j=1): b = 13+k, c = 2k+1.
    #   M[13+k, 2k+1] = 1 for k=0..12. Rank = 13.
    # 
    # So the slice lower bound gives rank >= 13, not 26.
    # 
    # Better lower bound: Consider the 4x676 matrix of all slices.
    # Or use the "slice rank" bound.
    # 
    # Actually, the standard lower bound for matrix multiplication rank(n,m,p)
    # is max(n*p, m*p, n*m) = max(26, 26, 4) = 26.
    # This comes from: the multiplication map has a certain structure.
    # 
    # Let me try a different approach. For 2x2xp, there are known results.
    # The rank of 2x2x4 is 7 (Strassen). 
    # For 2x2xp, the rank is at most 2p+1 for p odd, 2p for p even? 
    # Actually, I recall that rank(2x2xp) <= 2p + 2 - 2*ceil(log2(p+1)) or similar.
    # 
    # For p=13: 2*13 + 2 = 28? Or maybe 2*13+1 = 27?
    # 
    # Let me just try to search for small ranks using a structured approach.
    
    # For the search, I'll use the following strategy:
    # 1. Try rank r from lower_bound to schoolbook_rank
    # 2. For each r, try to find a solution using a parameterized ansatz
    
    # Given time constraints, let me try a few specific ranks.
    
    # First, let's establish that the schoolbook (rank 52) works.
    best = _schoolbook(n, m, p)
    
    # Try to improve. For 2x2x13, try ranks from 26 upward.
    # Use a simple but effective search: try random integer/half-integer entries.
    
    import random
    random.seed(42)
    
    # Try a structured approach for small ranks
    # The key insight: for n=2, we can write the tensor as a sum of
    # "laser" terms.
    
    # For now, try a few random searches for rank < 52.
    for target_rank in range(26, 52):
        if time.time() - start > 120:
            break
        # Try a few random initializations
        found = False
        for attempt in range(100):
            if time.time() - start > 120:
                break
            # Random rational entries from {-1, 0, 1, 1/2, -1/2}
            vals = [Fraction(-1), Fraction(0), Fraction(1), Fraction(1,2), Fraction(-1,2)]
            U_rand = [[random.choice(vals) for _ in range(target_rank)] for _ in range(n*m)]
            V_rand = [[random.choice(vals) for _ in range(target_rank)] for _ in range(m*p)]
            W_rand = [[random.choice(vals) for _ in range(target_rank)] for _ in range(p*n)]
            if _verify(U_rand, V_rand, W_rand, n, m, p):
                found = True
                best = {"U": U_rand, "V": V_rand, "W": W_rand}
                break
        if found:
            break
    
    return best
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, the random search over 5 values for 52*rank entries is extremely unlikely to find a solution. Let me think of a better approach.

For 2×2×13 matrix multiplication, the key structure is that the tensor T[i·2+j, j·13+k, 2k+i] = 1. This is very sparse and structured. Let me use a more principled approach based on the known structure of 2×2×p matrix multiplication.

