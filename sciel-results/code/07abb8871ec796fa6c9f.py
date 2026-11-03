"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i, j, k, l] for (n x m) * (m x p).
    T[i, j, k, l] = 1 if j == k, else 0.
    The tensor represents: C[i,l] = sum_j A[i,j] * B[j,l]
    """
    T = np.zeros((n, m, p, n, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for l in range(n):
                    if j == k:
                        T[i, j, k, l, 0] = 0  # placeholder
    # Actually, let's think about this more carefully.
    # The matrix multiplication tensor has indices (i, j, k) where:
    # A is n x m, B is m x p, C is n x p
    # C[i, l] = sum_j A[i, j] * B[j, l]
    # So the tensor T[i, j, l] = 1 (it's a 3-way tensor)
    # But the task says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Wait, let me re-read: "U, V, W': factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # And "W[k*n+i, r]" - so W has p*n rows, indexed by (k, i) where k is column of B / row of C? 
    # Let me think: A is (n x m), B is (m x p), C is (n x p)
    # U has n*m rows -> indexed by (i, j) for A
    # V has m*p rows -> indexed by (j, l) for B  
    # W has p*n rows -> indexed by (l, i) for C? But C is (n x p)...
    # W[k*n+i, r] where k ranges over p and i ranges over n, so k is column index of C, i is row index
    # So C[i, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * ... no wait.
    # Actually for tensor decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where T is the multiplication tensor.
    # T[i,j,k] should be 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C.
    # So T[i,j,k] = 1 for all valid i,j,k (it's the identity in the middle index).
    # Wait no. The multiplication tensor T satisfies:
    # T[i, j, k] = 1 if j is the shared index, i.e., always 1 for the "diagonal" in j.
    # Actually the tensor is: T[i, j, k] = delta_{j,k} ... no that's not right either.
    # Let me reconsider. For matrix mult (n x m) * (m x p):
    # The tensor T has three modes: (i, j) for A, (j, k) for B, (i, k) for C
    # T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k'
    # So it's the identity tensor in the sense that it picks out matching indices.
    # Flattening: U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    # The decomposition is: T = sum_r U_r ⊗ V_r ⊗ W_r (outer product)
    # Where T is viewed as a 3-tensor with dimensions (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) are consistent, i.e., the j in a matches j in b, and k in b matches k in c, and i in a matches i in c.
    
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _rank_of_flattening(T, mode):
    """Compute the rank of a specific flattening of the tensor T.
    mode=0: flatten modes 1,2 and keep mode 0 -> matrix of shape (d0, d1*d2)
    mode=1: flatten modes 0,2 and keep mode 1 -> matrix of shape (d1, d0*d2)
    mode=2: flatten modes 0,1 and keep mode 2 -> matrix of shape (d2, d0*d1)
    """
    d0, d1, d2 = T.shape
    if mode == 0:
        M = T.reshape(d0, d1 * d2)
    elif mode == 1:
        M = T.transpose(1, 0, 2).reshape(d1, d0 * d2)
    else:
        M = T.transpose(2, 0, 1).reshape(d2, d0 * d1)
    return np.linalg.matrix_rank(M)


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Reconstruct T from decomposition
    rank = U.shape[1]
    T_recon = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T_recon += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
    
    # True tensor
    T_true = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_true[a, b, c] = 1.0
    
    diff = np.abs(T_recon - T_true)
    return np.max(diff) < tol


def _schoolbook(n, m, p):
    """Return the schoolbook (trivial) decomposition."""
    rank = n * m * p
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
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin
    
    # Step 1: Compute the multiplication tensor and its flattening ranks
    T = _build_tensor(n, m, p)
    
    # Compute ranks of all three flattenings to get lower bounds
    r0 = _rank_of_flattening(T, 0)  # rank of (n*m) x (m*p * p*n) flattening
    r1 = _rank_of_flattening(T, 1)  # rank of (m*p) x (n*m * p*n) flattening
    r2 = _rank_of_flattening(T, 2)  # rank of (p*n) x (n*m * m*p) flattening
    
    lower_bound = max(r0, r1, r2)
    
    # The schoolbook rank is n*m*p = 72
    schoolbook_rank = n * m * p
    
    # For 3x3x8 matrix multiplication, we need to find the minimum rank.
    # Known results: for 3x3x3, min rank is 23 (Landsberg/Strassen type).
    # For 3x3x8, we can potentially do better than 72.
    
    # Strategy: Try to find a decomposition with rank < 72 using randomized search
    # and structured approaches.
    
    # First, let's try a structured approach exploiting the structure.
    # The tensor T[i,j,k] = 1 only when indices are consistent.
    # We can try to find a rank-r decomposition where r < n*m*p.
    
    # Approach: Use the fact that we can combine terms.
    # For each (i, k) pair, we need sum_j A[i,j]*B[j,k] = C[i,k].
    # The schoolbook uses one term per (i,j,k). We can potentially share terms.
    
    # Let's try a randomized approach with gradient-free search.
    # We'll parameterize U, V, W with random initializations and try to minimize
    # ||T - sum_r U_r ⊗ V_r ⊗ W_r||^2
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = _schoolbook(n, m, p)
    
    # Try ranks from lower_bound up to schoolbook_rank - 1
    # But we need to be careful about time.
    
    # For a 3x3x8 problem, let's think about what's achievable.
    # The tensor has n*m*p = 72 nonzero entries.
    # Each rank-1 term can cover at most one "diagonal" slice.
    # Actually, a rank-1 term U_r ⊗ V_r ⊗ W_r contributes to T[a,b,c] 
    # only if U_r[a]*V_r[b]*W_r[c] != 0.
    
    # Let's try a more systematic approach:
    # We know that for matrix multiplication (n x m) * (m x p),
    # the border rank and tensor rank are related to the dimensions.
    
    # For 3x3x3, the minimum rank is known to be 23.
    # For 3x3x8, we can think of it as 8 copies of 3x3x1 or 3x1x3, etc.
    
    # Let's try to use a block structure. 
    # Actually, let's just do a focused search.
    
    # Try rank values from a reasonable lower bound up to schoolbook_rank
    # We'll use a randomized tensor decomposition approach.
    
    # Key insight: We can try to find a decomposition where some terms are shared
    # across different (i,j,k) tuples.
    
    # Let's try a structured decomposition first.
    # For each (i, k) pair, we need to compute sum_j A[i,j]*B[j,k].
    # If we could use the same "rank-1 component" for multiple (i,j,k) triples,
    # we'd reduce the rank.
    
    # A rank-1 term (u, v, w) contributes u[a]*v[b]*w[c] to T[a,b,c].
    # For it to be nonzero at T[a,b,c] where a=(i,j), b=(j',k), c=(k',i'),
    # we need u[i*m+j] * v[j'*p+k] * w[k'*n+i'] != 0.
    # And it should equal 1 at the correct positions and 0 elsewhere.
    
    # This is a very constrained problem. Let's try a different approach.
    
    # Approach: Start from schoolbook and try to merge terms.
    # Two schoolbook terms (i1,j1,k1) and (i2,j2,k2) can potentially be merged
    # if there's a linear combination that works.
    
    # Actually, let's try a simpler approach: use the known result that for
    # matrix multiplication, we can sometimes do better using Strassen-like ideas.
    
    # For 3x3 matrices, Strassen gives rank 23 (vs 27 schoolbook).
    # For 3x3x8, we can potentially apply similar ideas.
    
    # Let's try a randomized search for a lower rank decomposition.
    
    def _try_rank(r, n, m, p, T, max_iters=500, seed=42):
        """Try to find a rank-r decomposition using random initialization and alternating least squares."""
        rng = np.random.RandomState(seed)
        
        # Initialize U, V, W randomly
        U = rng.randn(n * m, r) * 0.1
        V = rng.randn(m * p, r) * 0.1
        W = rng.ravel(p * n, r) if False else rng.randn(p * n, r) * 0.1
        
        # We want to minimize ||T - sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]||^2
        # This is a non-convex problem. We'll use alternating minimization.
        
        # Flatten T for computation
        T_flat = T.reshape(n * m, m * p * p * n)
        
        for it in range(max_iters):
            # Update U: for each column a of U, minimize ||T[:,a,:] - u_a * (V ⊗ W)||^2
            # This is equivalent to: U[:,r] = (V_r ⊗ W_r)^T T_col_r / ||V_r ⊗ W_r||^2
            # where T_col_r is the slice of T corresponding to rank component r.
            # Actually, the objective is sum_r ||T - sum_s u_s v_s w_s||^2
            # The gradient w.r.t. u_r is: -2 (T - sum_s u_s v_s w_s) * (v_r ⊗ w_r)
            # Setting to zero: (T - sum_s u_s v_s w_s) * (v_r ⊗ w_r) = 0
            
            # Simpler: use ALS (Alternating Least Squares)
            # Fix V, W, solve for U:
            # min_U sum_r ||T - sum_s u_s ⊗ v_s ⊗ w_s||^2
            # = min_U sum_r ||T - u_r ⊗ (v_r ⊗ w_r) - sum_{s≠r} u_s ⊗ v_s ⊗ w_s||^2
            # For each r: u_r = argmin ||T - u_r ⊗ (v_r ⊗ w_r) - residual_r||^2
            # where residual_r = sum_{s≠r} u_s ⊗ v_s ⊗ w_s
            # u_r = <T - residual_r, v_r ⊗ w_r> / ||v_r ⊗ w_r||^2
            
            # This is getting complex. Let's use a simpler approach.
            
            # Actually, let's just check if the current decomposition is good enough.
            T_recon = np.zeros((n * m, m * p, p * n))
            for rr in range(r):
                T_recon += np.outer(np.outer(U[:, rr], V[:, rr]), W[:, rr])
            
            err = np.linalg.norm(T_recon - T)
            if err < 1e-6:
                return U, V, W, True
            
            # Simple gradient step (not ideal but let's try)
            # Gradient of ||T - sum_r u_r v_r w_r||^2 w.r.t. u_r:
            # -2 (T - sum_s u_s v_s w_s) * (v_r ⊗ w_r)
            
            # This is too slow for Python. Let's use a different strategy.
            break
        
        return None, None, None, False
    
    # The ALS approach above is too slow in pure Python for this size.
    # Let's think of a better approach.
    
    # Key realization: The tensor T for 3x3x8 matrix multiplication has a very specific structure.
    # T[(i,j), (j',k), (k',i')] = 1 iff i=i', j=j', k=k'.
    # So T is essentially the "identity" tensor connecting the three matrices.
    
    # The number of nonzero entries is n*m*p = 72.
    # Each rank-1 term can potentially explain multiple nonzero entries if the 
    # corresponding vectors have overlapping support.
    
    # For a rank-1 term (u, v, w):
    # u has n*m entries, v has m*p entries, w has p*n entries.
    # The term contributes u[a]*v[b]*w[c] to T[a,b,c].
    # For this to match T, we need:
    # - At positions where T=1: u[a]*v[b]*w[c] = 1
    # - At positions where T=0: u[a]*v[b]*w[c] = 0
    # (for the sum over all rank terms to work out)
    
    # This is a very hard constraint satisfaction problem.
    
    # Let's try a completely different approach: exploit the fact that
    # 3x3x8 = 3x3x3 + 3x3x3 + 3x3x2 (splitting the 8 columns).
    # But this doesn't directly help.
    
    # Actually, for the purpose of this task, let's see if we can find ANY
    # decomposition with rank < 72.
    
    # Let's try a structured approach:
    # For each pair (i, k), we need sum_j A[i,j]*B[j,k] = C[i,k].
    # There are n*p = 24 such (i,k) pairs, each requiring m=3 multiplications.
    # Total: 24*3 = 72 scalar multiplications (schoolbook).
    
    # Can we share multiplications? For example, if we compute A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k],
    # can we reuse any intermediate products across different (i,k) pairs?
    
    # In general, no, because each product A[i,j]*B[j,k] involves specific i and k.
    # But with clever linear combinations, we might reduce the count.
    
    # For 3x3x3, Strassen's algorithm shows rank 23 < 27 is possible.
    # The key is to use identities like:
    # (a+b)(c+d) = ac + ad + bc + bd
    # which lets you compute 4 products with 3 multiplications.
    
    # For 3x3x8, we might be able to apply similar ideas.
    
    # Let's try to construct a decomposition with rank < 72 using a heuristic.
    
    # Approach: Start with schoolbook and try to find linear dependencies.
    # If we can find that some combination of schoolbook terms can be replaced
    # by fewer terms, we reduce the rank.
    
    # Actually, let's just try to verify if rank 72 is indeed minimal or if we
    # can do better. Given the time constraint, let's try a few specific ranks.
    
    # For now, let's return the schoolbook decomposition as a safe baseline,
    # but also try to find something better.
    
    # Let's try rank values: 71, 70, 69, ... down to lower_bound
    # But we need an efficient search method.
    
    # Given the complexity, let's try a randomized approach with a time limit.
    
    def _randomized_search(n, m, p, target_rank, T, max_time=10):
        """Randomized search for a rank-r decomposition."""
        start = time.time()
        rng = np.random.RandomState(123)
        
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for attempt in range(200):
            if time.time() - start > max_time:
                break
            
            U = rng.randn(n * m, target_rank) * 0.5
            V = rng.randn(m * p, target_rank) * 0.5
            W = rng.randn(p * n, target_rank) * 0.5
            
            # Alternating least squares
            for it in range(100):
                # Update U
                for r in range(target_rank):
                    # Compute the "effective" target for this rank component
                    # T_r = T - sum_{s≠r} u_s v_s w_s
                    # u_r = <T_r, v_r ⊗ w_r> / ||v_r ⊗ w_r||^2
                    
                    # This is expensive. Let's use a simpler update.
                    pass
                
                # This is too slow. Break.
                break
            
            # Just compute the error for this random initialization
            T_recon = np.zeros((n * m, m * p, p * n))
            for r in range(target_rank):
                T_recon += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
            err = np.linalg.norm(T_recon - T)
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
        
        if best_err < 1e-6 and best_U is not None:
            return best_U, best_V, best_W, True
        return None, None, None, False
    
    # The randomized search above is unlikely to succeed quickly for this problem.
    # Let's think about this differently.
    
    # For the 3x3x8 matrix multiplication tensor, let's compute the actual
    # lower bounds from the flattening ranks.
    
    # r0 = rank of T flattened as (n*m) x (m*p * p*n) = (9) x (24*24) = (9) x (576)
    # r1 = rank of T flattened as (m*p) x (n*m * p*n) = (24) x (9*24) = (24) x (216)
    # r2 = rank of T flattened as (p*n) x (n*m * m*p) = (24) x (9*24) = (24) x (216)
    
    # Let's compute these.
    # Actually, we already computed them above. Let's use them.
    
    # For the schoolbook decomposition, the rank is 72.
    # The lower bounds from flattenings are r0, r1, r2.
    
    # Let's see what these are.
    # r0: The matrix is 9 x 576. Each row i (for fixed i) has nonzeros only where
    # the first index of (i,j) is i. So row i has nonzeros in columns corresponding to
    # (j, k, i') where i'=i. There are m*p = 24 such columns, all with value 1.
    # So each row has 24 nonzeros. The rows for different i are orthogonal (no overlap).
    # So r0 = 3 (one per i value).
    
    # r1: The matrix is 24 x 216. Each row (j, k) has nonzeros where the second index
    # of (i, j') is j and the third index (k', i') has k'=k. 
    # So for row (j,k), nonzeros are at positions (i, i') where the first mode has j'=j 
    # and the third mode has k'=k. That's n*n = 9 positions.
    # Rows with different j are disjoint in the first mode. Rows with same j but different k
    # are disjoint in the third mode.
    # So we have m*p = 24 rows, each with 9 nonzeros, and they're all orthogonal.
    # So r1 = 24.
    
    # r2: Similarly, r2 = 24.
    
    # So lower_bound = max(3, 24, 24) = 24.
    
    # This means the rank is at least 24. The schoolbook gives 72.
    # Can we achieve rank 24? Probably not, but maybe something in between.
    
    # For 3x3x3, the minimum rank is 23, and the flattening ranks are:
    # r0 = 3, r1 = 9, r2 = 9. So lower bound is 9, but actual min rank is 23.
    
    # For 3x3x8, let's try to find a good decomposition.
    
    # Given the time constraints and the difficulty of this problem, let's try
    # a few specific structured decompositions.
    
    # Idea: Use the fact that we can group the 8 columns.
    # For each (i, k), we need sum_j A[i,j]*B[j,k].
    # If we use the same "pattern" for different k values, we might save rank.
    
    # Actually, let's just try to find if there's a known result for 3x3x8.
    # I don't have specific knowledge, so let's try a heuristic search.
    
    # Given the time constraints, let's return the schoolbook decomposition
    # unless we can quickly find something better.
    
    # Let's try one more thing: a structured decomposition that exploits
    # the linearity in the third dimension (the 8 columns).
    
    # For fixed i and j, as k varies, we have A[i,j]*B[j,k] for k=0,...,7.
    # This is just A[i,j] times the k-th row of B's j-th column... 
    # Actually, for fixed (i,j), the contribution to C[i,k] for all k is:
    # A[i,j] * B[j,:] (a row vector of length 8)
    
    # So C[i,:] = sum_j A[i,j] * B[j,:]
    
    # This is just matrix multiplication. The "rank" here refers to the tensor rank.
    
    # I think for this problem, finding a rank < 72 decomposition is non-trivial
    # and may require sophisticated techniques. Given the time constraint,
    # let's return the schoolbook decomposition.
    
    # But wait - let's try one more idea. What if we use a decomposition
    # that's not based on individual (i,j,k) terms but on linear combinations?
    
    # For example, for 3x3x3, Strassen uses 23 terms instead of 27.
    # The key terms involve products like (A[i,0]+A[i,1])*(B[0,k]+B[1,k]).
    
    # Let's try to apply a similar idea for 3x3x8.
    
    # Actually, I realize that for this specific task format, the evaluator
    # checks if the decomposition is exact. So we need to be very careful.
    
    # Let's just return the schoolbook decomposition for now, as it's guaranteed
    # to be correct.
    
    U, V, W = _schoolbook(n, m, p)
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
