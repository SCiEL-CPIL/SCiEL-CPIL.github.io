"""Matrix multiplication 3,3,6 - rank-minimal tensor decomposition search."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 6]
DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
    
    def check_decomp(U, V, W, tol=1e-9):
        """Check if U,V,W exactly decompose the matrix multiplication tensor."""
        # T[i,j,k] = 1 if (i,j,k) valid, else 0
        # For each column c: U[:,c] * V[:,c] * W[:,c] should reconstruct T
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We check: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        # and for invalid (i,j,k) it should be 0
        # Actually the tensor is: T[i,j,k] = 1 for all valid (i,j,k)
        # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
        # This is a rank-3 tensor factorization where T is the all-ones tensor of shape (n,m,p)
        # Wait - the matrix multiplication tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # because (AB)[i,k] = sum_j A[i,j]*B[j,k], so the structure tensor has all entries = 1
        # Actually no. The structure tensor for matrix mult A(n x m) * B(m x p) -> C(n x p) is:
        # T[i, j, k] = 1 where i indexes A's rows, j indexes A's cols/B's rows, k indexes B's cols/C's cols
        # T[i,j,k] = 1 for all valid indices. Yes, it's the all-ones tensor.
        
        # So we need: for all (i,j,k) in [0,n)x[0,m)x[0,p): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # And the factorization must be exact (integer/half-integer entries)
        
        # Check all (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def is_half_integer(arr, tol=1e-9):
        """Check if all entries are integer or half-integer."""
        for val in arr.flat:
            if abs(val * 2 - round(val * 2)) > tol:
                return False
        return True
    
    # Strategy: Search over structured decompositions
    # Key insight: T = all-ones tensor of shape (3,3,6)
    # We need rank-r factorization: T[i,j,k] = sum_{c=1}^r u_{i,j}^{(c)} v_{j,k}^{(c)} w_{k,i}^{(c)}
    
    # Try small ranks with structured patterns
    # Pattern: group by (i,k) pairs and share across j, or group by j and share across (i,k)
    
    best_rank = rank_sb
    best_result = None
    
    # Try rank from lower bounds upward
    # Lower bound: max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # Also: rank >= max(n, m, p) * something... actually rank >= max(n*m, m*p, p*n) is not right
    # The tensor has dimensions (nm, mp, pn) = (9, 18, 18)
    # Rank >= max(9, 18, 18) = 18 is a lower bound from dimension arguments
    # But more carefully: the tensor T of shape (3,3,6) has rank at least...
    # Actually for the all-ones tensor of shape (a,b,c), the tensor rank is min(a,b,c) if... no.
    # The all-ones tensor of shape (a,b,c) has tensor rank = min(a,b,c) only in special cases.
    # Actually, the all-ones tensor 1_{a,b,c} can be written as sum of min(a,b,c) rank-1 tensors:
    # 1_{a,b,c} = sum_{r=1}^{min(a,b,c)} e_r e_r^T e_r^T (with appropriate indexing)
    # Wait, that's not quite right either. Let me think more carefully.
    
    # For all-ones tensor of shape (3,3,6):
    # T[i,j,k] = 1 for all i,j,k
    # Can write as: sum_{r=1}^3 u_r v_r w_r where u_r[i,j]=1, v_r[j,k]=1, w_r[k,i]=1? No that gives 3.
    # Actually: T[i,j,k] = (1,1,1)[i] * (1,1,1)[j] * (1,1,1)[k] is rank 1!
    # Wait - T[i,j,k] = 1 = 1*1*1. So the all-ones tensor IS rank 1!
    # But that can't be right for matrix multiplication...
    
    # Let me reconsider. The structure tensor for matrix multiplication is NOT the all-ones tensor in the naive sense.
    # The structure tensor T has entries T[i, j, k] where:
    # - i indexes the (row, col) pair of A: i = a_row * m + a_col, range [0, n*m)
    # - j indexes the (row, col) pair of B: j = b_row * p + b_col, range [0, m*p)  
    # - k indexes the (row, col) pair of C: k = c_row * p + c_col, range [0, n*p)
    # T[i, j, k] = 1 iff a_row = c_row AND a_col = b_row AND b_col = c_col
    # i.e., T[a_row*m + a_col, b_row*p + b_col, c_row*p + c_col] = 1 iff a_row=c_row, a_col=b_row, b_col=c_col
    
    # So it's NOT the all-ones tensor! It's a sparse tensor.
    # For n=3, m=3, p=6:
    # T[i,j,k] = 1 iff i%3 == k%6... wait let me re-index.
    # i = a_row * m + a_col, so a_row = i // m, a_col = i % m
    # j = b_row * p + b_col, so b_row = j // p, b_col = j % p
    # k = c_row * p + c_col, so c_row = k // p, c_col = k % p
    # Condition: a_row = c_row, a_col = b_row, b_col = c_col
    # i.e.: i//m == k//p, i%m == j//p, j%p == k%p
    
    def check_decomp_correct(U, V, W, tol=1e-9):
        """Check if U,V,W decompose the matrix multiplication tensor correctly."""
        for i in range(nm):
            a_row = i // m
            a_col = i % m
            for j in range(mp):
                b_row = j // p
                b_col = j % p
                for k in range(pn):
                    c_row = k // p
                    c_col = k % p
                    expected = 1.0 if (a_row == c_row and a_col == b_row and b_col == c_col) else 0.0
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i, c] * V[j, c] * W[k, c]
                    if abs(s - expected) > tol:
                        return False
        return True
    
    # Now search for low-rank decompositions
    # Key structural insight: The tensor has a lot of structure.
    # T[i,j,k] = 1 iff i//m == k//p, i%m == j//p, j%p == k%p
    # Let's denote: a=i//m, b=i%m, c=j//p, d=j%p, e=k//p, f=k%p
    # T=1 iff a==e, b==c, d==f
    
    # This means for fixed (a, b, d): T[i,j,k]=1 only when e=a, c=b, f=d
    # So for each (a,b,d) triple, there's exactly one (i,j,k) with T=1
    # Total nonzero entries: n * m * p = 54 (same as schoolbook rank)
    
    # The tensor is a permutation-like tensor. Let's think about what rank is achievable.
    
    # Approach: Try to find decompositions where each rank-1 term covers multiple (i,j,k) entries
    # A rank-1 term U[:,c]*V[:,c]*W[:,c] contributes to T[i,j,k] for all (i,j,k) where
    # U[i,c], V[j,c], W[k,c] are all nonzero.
    
    # Let's try a structured approach:
    # For each (a, b) where a in [0,n), b in [0,m), we need to cover all d in [0,p)
    # The entries are: (i=a*m+b, j=b*p+d, k=a*p+d) for d=0..p-1
    
    # Idea: Use terms that are "diagonal" in d. For a fixed (a,b), the p entries form a line.
    # If we use p terms per (a,b) pair, we get n*m*p = 54 (schoolbook).
    # Can we do better by sharing across different (a,b) pairs?
    
    # Let's try: for each d in [0,p), and each a in [0,n), create a term that covers
    # all b in [0,m) simultaneously.
    # Term (a, d): U[a*m+b, c] = 1 for all b, V[b*p+d, c] = 1 for all b, W[a*p+d, c] = 1
    # This gives: contribution to T[i,j,k] where i=a*m+b, j=b'*p+d', k=a'*p+d'
    # = 1*1*1 = 1 iff a'=a, d'=d, and b=b' (from U and V indices)
    # Wait: U[i,c] is nonzero for i in {a*m, a*m+1, ..., a*m+m-1}
    # V[j,c] is nonzero for j in {0*p+d, 1*p+d, ..., (m-1)*p+d} = {d, p+d, 2p+d, ...}
    # W[k,c] is nonzero for k = a*p+d only
    # So T[i,j,k] gets contribution 1 when:
    # i = a*m+b for some b, j = b'*p+d for some b', k = a*p+d
    # And the product is U[i,c]*V[j,c]*W[k,c] = 1*1*1 = 1
    # But we need this to equal T[i,j,k] which is 1 only when i//m==k//p, i%m==j//p, j%p==k%p
    # Here: i//m = a, k//p = a ✓, i%m = b, j//p = b', j%p = d, k%p = d
    # So we need b = b'. But U and V don't enforce b=b'!
    # The term contributes 1 to ALL (i,j,k) with i in row a of A, j in column d of B, k=(a,d) of C
    # That's m*p entries per term, but only m of them are correct (where b=b')
    
    # So this doesn't work directly. We need more careful construction.
    
    # Let me try a different approach: use the fact that for fixed a and d,
    # we need to realize the vector (1,1,...,1) of length m in the b-index.
    # This requires at least 1 term if we can make it work.
    
    # Actually, let's think about it as: for each (a,d) pair, we need a rank-1 (or low-rank)
    # decomposition of the all-ones vector of length m in the b-index.
    # The all-ones vector of length m has rank 1, so 1 term suffices per (a,d) pair.
    # Total: n * p = 3 * 6 = 18 terms!
    
    # Let's verify: Term for (a, d):
    # U[i, c] = 1 if i // m == a (i.e., i in {a*m, ..., a*m+m-1}), else 0
    # V[j, c] = 1 if j % p == d (i.e., j in {d, p+d, 2p+d, ..., (m-1)*p+d}), else 0  
    # W[k, c] = 1 if k == a*p + d, else 0
    
    # Contribution to T[i,j,k]: U[i,c]*V[j,c]*W[k,c]
    # = 1 iff i//m == a AND j%p == d AND k == a*p+d
    # For this to match T[i,j,k] = 1 iff i//m==k//p, i%m==j//p, j%p==k%p:
    # When W[k,c]=1: k=a*p+d, so k//p=a, k%p=d
    # We need: i//m == a ✓ (from U), j%p == d ✓ (from V), i%m == j//p
    # But i%m == j//p is NOT enforced! U doesn't care about i%m, V doesn't care about j//p.
    # So the term contributes 1 to entries where i%m != j//p as well. WRONG.
    
    # We need to also enforce i%m == j//p. This couples the b-index of i with the b-index of j.
    # This is the fundamental difficulty.
    
    # Alternative: for each (a, d, b) triple, use a term:
    # U[i,c] = 1 iff i == a*m+b
    # V[j,c] = 1 iff j == b*p+d
    # W[k,c] = 1 iff k == a*p+d
    # This is exactly schoolbook: n*m*p = 54 terms.
    
    # Can we merge terms? Two terms (a1,d1,b1) and (a2,d2,b2) can be merged if...
    # Actually, let's think about linear algebra. The tensor T can be viewed as a matrix
    # for each fixed index. T as a (nm x mp) matrix for each k, or (nm x pn) for each j, etc.
    
    # Let's try: fix k. T[:, :, k] is an nm x mp matrix.
    # T[i, j, k] = 1 iff i//m == k//p, i%m == j//p, j%p == k%p
    # For fixed k with k//p = a, k%p = d:
    # T[i, j, k] = 1 iff i//m == a, i%m == j//p, j%p == d
    # So T[:, :, k] has 1's at positions (a*m+b, b*p+d) for b=0..m-1
    # This is a permutation matrix (m ones on a "diagonal") embedded in nm x mp.
    # The rank of this matrix is m (since the m ones are in distinct rows and columns).
    
    # Similarly, for fixed i: T[i, :, :] is an mp x pn matrix.
    # For i = a*m+b: T[i, j, k] = 1 iff j//p == b, j%p == k%p, k//p == a
    # So 1's at (b*p+d, a*p+d) for d=0..p-1. This is a permutation matrix of size p.
    # Rank = p.
    
    # For fixed j: T[:, j, :] is an nm x pn matrix.
    # For j = b*p+d: T[i, j, k] = 1 iff i//m == k//p, i%m == b, k%p == d
    # So 1's at (a*m+b, a*p+d) for a=0..n-1. This is a permutation matrix of size n.
    # Rank = n.
    
    # The tensor rank is at least the maximum of these matrix ranks: max(m, p, n) = max(3,6,3) = 6.
    # But that's a weak lower bound. The actual rank is likely higher.
    
    # Let's try a computational search for small ranks.
    # We'll set up the linear system and try to solve it.
    
    # The constraint is: for all (i,j,k):
    # sum_c U[i,c] * V[j,c] * W[k,c] = T[i,j,k]
    
    # This is nonlinear in (U,V,W). But we can fix two of them and solve for the third.
    # Or use iterative methods.
    
    # Let's try a direct approach: enumerate possible small-rank decompositions
    # with structured (sparse) patterns.
    
    # Given time constraints, let's try a few specific structured patterns:
    
    # Pattern 1: n*p = 18 terms, one per (a,d) pair, with the b-coupling handled by
    # making U and V depend on b in a coordinated way.
    # Actually this doesn't work as shown above.
    
    # Pattern 2: Use m*p = 18 terms, one per (b,d) pair.
    # Term (b,d): U[i,c] = 1 iff i%m == b, V[j,c] = 1 iff j == b*p+d, W[k,c] = 1 iff k%p == d
    # Contribution: 1 iff i%m==b, j==b*p+d, k%p==d
    # T requires: i//m==k//p, i%m==j//p, j%p==k%p
    # Here j//p = b, j%p = d, so we need i%m==b ✓, k%p==d ✓, i//m==k//p
    # But i//m==k//p is NOT enforced. U doesn't constrain i//m, W doesn't constrain k//p.
    # Same problem.
    
    # Pattern 3: n*m = 9 terms, one per (a,b) pair.
    # Term (a,b): U[i,c]=1 iff i==a*m+b, V[j,c]=1 iff j//p==b, W[k,c]=1 iff k//p==a
    # Contribution: 1 iff i==a*m+b, j//p==b, k//p==a
    # T requires: i//m==k//p, i%m==j//p, j%p==k%p
    # Here i//m=a, i%m=b, j//p=b, k//p=a. Need j%p==k%p. NOT enforced.
    
    # It seems like we always miss one constraint. The three constraints are:
    # (1) i//m == k//p  (a == e)
    # (2) i%m == j//p    (b == c)  
    # (3) j%p == k%p     (d == f)
    # Each pair of constraints can be enforced by 2 factors, but the 3rd is missed.
    
    # This suggests we need at least n*m*p terms if we use this simple structure.
    # But maybe with more clever (non-binary, non-sparse) entries we can do better.
    
    # Let's try a numerical optimization approach for small ranks.
    # We'll use the fact that the problem is equivalent to finding a low-rank
    # decomposition of the tensor.
    
    # Actually, let me reconsider. The tensor T has a very specific structure.
    # Let's view it as a 3-tensor and use known results.
    
    # The tensor T[i,j,k] for matrix multiplication n x m by m x p is known to have
    # tensor rank that's been studied extensively. For 3x3 by 3x6:
    # This is related to the rank of matrix multiplication.
    # For square matrices, the best known rank for 3x3 is 23 (Laser's method) or 22 (recent).
    # For rectangular, it's less clear.
    
    # Given the time constraint, let me try a practical search:
    # 1. Try rank 18, 20, 24, 27, 30, 36, 40, 45, 50, 54
    # 2. For each rank, use a randomized/structured initialization and Newton's method
    
    # Actually, let me try a more systematic approach.
    # The tensor can be written as a sum of rank-1 tensors.
    # Each rank-1 tensor is u ⊗ v ⊗ w where u in R^{nm}, v in R^{mp}, w in R^{pn}.
    
    # Key observation: T is a "permutation-like" tensor. For each of the 54 nonzero entries,
    # exactly one (i,j,k) triple is active. The 54 entries are:
    # (a*m+b, b*p+d, a*p+d) for a in [0,3), b in [0,3), d in [0,6)
    
    # Let's try to find a decomposition where each rank-1 term covers multiple entries.
    # A rank-1 term u⊗v⊗w covers entry (i,j,k) if u[i]*v[j]*w[k] != 0.
    # For the term to contribute correctly, we need the sum over all terms to give 1
    # at the 54 active entries and 0 elsewhere.
    
    # Let's try: group entries by their "diagonal" structure.
    # Entry (a*m+b, b*p+d, a*p+d) has the property that:
    # i + j + k = a*m+b + b*p+d + a*p+d = a*(m+p) + b*(1+p) + 2d
    # Hmm, not obviously helpful.
    
    # Let me just do a numerical search with time limit.
    
    def try_rank(r, time_limit=10.0):
        """Try to find a rank-r decomposition."""
        t0 = time.time()
        
        # Set up the problem as: find U (nm x r), V (mp x r), W (pn x r)
        # such that for all (i,j,k): sum_c U[i,c]*V[j,c]*W[k,c] = T[i,j,k]
        
        # We'll use an iterative approach: fix two matrices, solve for the third.
        # Or use gradient-based optimization.
        
        # Actually, let's use a simpler approach: 
        # For a given rank r, the problem is equivalent to finding r vectors u_c, v_c, w_c
        # such that the outer product sum equals T.
        
        # Let's try a structured initialization:
        # Initialize with schoolbook pattern (first r of the 54 terms)
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        
        # Initialize with first r schoolbook terms
        count = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if count >= r:
                        break
                    U[i*m+j, count] = 1
                    V[j*p+k, count] = 1
                    W[k*n+i, count] = 1
                    count += 1
            if count >= r:
                break
            if count >= r:
                break
        
        # This initialization only works if r <= 54
        if r > 54:
            return None
        
        # Now try to optimize. The objective is:
        # min sum_{i,j,k} (sum_c U[i,c]*V[j,c]*W[k,c] - T[i,j,k])^2
        # subject to entries being half-integers (but let's ignore this for now)
        
        # Use alternating least squares:
        # Fix V, W: solve for U. U[i,c] appears linearly in the objective.
        # For fixed i: sum_c U[i,c] * (sum_k V[j,c]*W[k,c]) ... hmm, this is getting complex.
        
        # Actually, ALS for tensor decomposition:
        # Fix V, W: For each i, U[i,:] should satisfy:
        # sum_c U[i,c] * V[j,c] * W[k,c] contributes to T[i,j,k] for all j,k
        # This is: U[i,:] * (V[j,:] ⊙ W[k,:]) = T[i,j,k] for all j,k
        # Where ⊙ is element-wise product.
        # So: U[i,:] * M_jk = T[i,j,k] where M_jk[c] = V[j,c]*W[k,c]
        # This is a linear system: U[i,:] * M = T[i,:,:] (reshaped)
        # But M is mp*pn x r, and we have mp*pn equations for r unknowns.
        # Least squares: U[i,:] = T[i,:,:].reshape(-1) @ pinv(M)
        
        # Let's implement ALS:
        for iteration in range(200):
            if time.time() - t0 > time_limit:
                return None
            
            # Update U
            for i in range(nm):
                # M[j*pn+k, c] = V[j,c] * W[k,c]
                M = np.zeros((mp * pn, r))
                for j in range(mp):
                    for k in range(pn):
                        M[j*pn+k, :] = V[j, :] * W[k, :]
                T_vec = np.zeros(mp * pn)
                for j in range(mp):
                    for k in range(pn):
                        a_row = i // m
                        a_col = i % m
                        b_row = j // p
                        b_col = j % p
                        c_row = k // p
                        c_col = k % p
                        T_vec[j*pn+k] = 1.0 if (a_row == c_row and a_col == b_row and b_col == c_col) else 0.0
                # Solve: U[i,:] @ M = T_vec
                # U[i,:] = T_vec @ pinv(M)
                try:
                    U[i, :] = np.linalg.lstsq(M, T_vec, rcond=None)[0]
                except:
                    return None
            
            # Update V
            for j in range(mp):
                M = np.zeros((nm * pn, r))
                for i in range(nm):
                    for k in range(pn):
                        M[i*pn+k, :] = U[i, :] * W[k, :]
                T_vec = np.zeros(nm * pn)
                a_row = j // p
                a_col = j % p  # This is b_col, not a_col... let me re-check
                # j = b_row * p + b_col
                b_row = j // p
                b_col = j % p
                for i in range(nm):
                    for k in range(pn):
                        i_ar = i // m
                        i_ac = i % m
                        k_cr = k // p
                        k_cc = k % p
                        T_vec[i*pn+k] = 1.0 if (i_ar == k_cr and i_ac == b_row and b_col == k_cc) else 0.0
                try:
                    V[j, :] = np.linalg.lstsq(M, T_vec, rcond=None)[0]
                except:
                    return None
            
            # Update W
            for k in range(pn):
                M = np.zeros((nm * mp, r))
                for i in range(nm):
                    for j in range(mp):
                        M[i*mp+j, :] = U[i, :] * V[j, :]
                T_vec = np.zeros(nm * mp)
                c_row = k // p
                c_col = k % p
                for i in range(nm):
                    for j in range(mp):
                        i_ar = i // m
                        i_ac = i % m
                        j_br = j // p
                        j_bc = j % p
                        T_vec[i*mp+j] = 1.0 if (i_ar == c_row and i_ac == j_br and j_bc == c_col) else 0.0
                try:
                    W[k, :] = np.linalg.lstsq(M, T_vec, rcond=None)[0]
                except:
                    return None
            
            # Check convergence
            if iteration % 20 == 0:
                err = 0.0
                for i in range(nm):
                    for j in range(mp):
                        for k in range(pn):
                            a_row = i // m
                            a_col = i % m
                            b_row = j // p
                            b_col = j % p
                            c_row = k // p
                            c_col = k % p
                            expected = 1.0 if (a_row == c_row and a_col == b_row and b_col == c_col) else 0.0
                            s = np.dot(U[i, :], V[j, :] * W[k, :])
                            err += (s - expected) ** 2
                if err < 1e-10:
                    # Check half-integer constraint
                    if is_half_integer(U) and is_half_integer(V) and is_half_integer(W):
                        return (U, V, W)
                    # Try to round to nearest half-integers and re-check
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if check_decomp_correct(U_r, V_r, W_r):
                        return (U_r, V_r, W_r)
        
        return None
    
    # Try ranks from low to high
    for r in [18, 20, 22, 24, 27, 30, 32, 36, 40, 42, 45, 48, 50, 54]:
        if time.time() > DEADLINE - 20:
            break
        result = try_rank(r, time_limit=10.0)
        if result is not None:
            U, V, W = result
            # Convert to lists
            U_list = [[float(U[i, c]) for c in range(r)] for i in range(nm)]
            V_list = [[float(V[j, c]) for c in range(r)] for j in range(mp)]
            W_list = [[float(W[k, c]) for c in range(r)] for k in range(pn)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    U_list = [[float(U_sb[i, c]) for c in range(rank_sb)] for i in range(nm)]
    V_list = [[float(V_sb[j, c]) for c in range(rank_sb)] for j in range(mp)]
    W_list = [[float(W_sb[k, c]) for c in range(rank_sb)] for k in range(pn)]
    return {"U": U_list, "V": V_list, "W": W_list}
