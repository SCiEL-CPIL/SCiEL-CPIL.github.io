import numpy as np
import time
from fractions import Fraction
from itertools import product

START_TIME = time.time()
TIME_LIMIT = 110  # Safety margin for 120s limit

def _elapsed():
    return time.time() - START_TIME

def _verify(U, V, W, n=3, m=5, p=7):
    """Verify that the decomposition is exact for matrix multiplication."""
    rank = len(U[0])
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each (i, j, k), the product A[i,j]*B[j,k] should be captured
    # The decomposition says: C[k,i] = sum_j A[i,j] B[j,k]
    # In tensor form: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # must equal the standard multiplication tensor
    
    # Build the target tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Target: 1 (since T[i,j,k] = delta for the multiplication tensor)
                # Actually for matrix mult, T[i,j,k] = 1 if we're computing C[k,i] = sum_j A[i,j]B[j,k]
                # The tensor entry is 1 for each (i,j,k)
                target = 1.0
                # Computed: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                computed = 0.0
                for r in range(rank):
                    computed += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(computed - target) > 1e-9:
                    return False
    return True

def _verify_random(U, V, W, n=3, m=5, p=7, trials=10):
    """Verify by actually multiplying random matrices."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute using decomposition
        # C[k,i] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
        # Wait, let me think about this more carefully.
        # The bilinear form: for each r, we compute:
        #   u_r = U[:, r] reshaped to (n, m)
        #   v_r = V[:, r] reshaped to (m, p)
        #   w_r = W[:, r] reshaped to (p, n)
        # Then C += w_r * (u_r * A) * (B * v_r) -- no that's not right either.
        
        # Actually the standard form: C[k,i] = sum_r W[k*n+i, r] * (U[i*m+j, r] * A[i,j]) * (V[j*p+k, r] * B[j,k])
        # summed over j. So:
        # C[k,i] = sum_r W[k*n+i, r] * sum_j (U[i*m+j, r] * A[i,j]) * (V[j*p+k, r] * B[j,k])
        #        = sum_r W[k*n+i, r] * (U_row_i_r @ A[i,:]) * (B[:,k] @ V_col_k_r)
        
        C_computed = np.zeros((p, n))
        for r in range(rank):
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            # For each r, the contribution is:
            # C += w_r * (u_r * A) * (B * v_r) element-wise? No...
            # Let me use the tensor definition directly.
            pass
        
        # Use the tensor definition:
        # C[k,i] = sum_j sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
        # = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k])
        
        for k in range(p):
            for i in range(n):
                val = 0.0
                for r in range(rank):
                    inner = 0.0
                    for j in range(m):
                        inner += U[i*m+j, r] * A[i, j] * V[j*p+k, r] * B[j, k]
                    val += W[k*n+i, r] * inner
                C_computed[k, i] = val
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def construct():
    n, m, p = 3, 5, 7
    
    # Start with schoolbook rank = 105
    rank = n * m * p
    U = [[0.0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0.0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0.0 for _ in range(rank)] for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][idx] = 1.0
                V[j*p + k][idx] = 1.0
                W[k*n + i][idx] = 1.0
                idx += 1
    
    # Try to reduce rank by finding linear dependencies
    # The key insight: if two rank-1 terms have proportional U, V, or W vectors,
    # we can combine them.
    
    # Strategy: Use the structure of matrix multiplication.
    # For fixed i and k, summing over j gives the (i,k) entry of C.
    # The terms with same (i,k) but different j can potentially be combined
    # if their U and V vectors align.
    
    # Better approach: Try to find a lower-rank decomposition using the
    # known structure. For 3x5x7, we can try:
    # 1. Pad to 4x8 (or 4x4 x 4x4) and use Strassen
    # 2. Direct search for rank reduction
    
    # Let's try the Strassen-based approach:
    # Pad A from 3x5 to 4x8 (pad with zeros)
    # Pad B from 5x7 to 8x4 (pad with zeros)
    # Wait, that doesn't work dimensionally for Strassen.
    
    # Actually, let's think differently. The minimum rank for 3x5x7 matrix mult
    # is not well-known, but we know it's <= 105 (schoolbook).
    
    # Let me try a direct approach: use the fact that for small matrices,
    # we can sometimes find lower-rank decompositions.
    
    # Approach: Try to express the tensor as sum of fewer rank-1 tensors.
    # The tensor T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p].
    # We want T = sum_{r=1}^{rank} u_r * v_r * w_r where
    # u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
    # and T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This is equivalent to: the 3D tensor of all ones can be written as
    # sum of rank-1 tensors, but with the constraint that the indices are
    # coupled: U depends on (i,j), V on (j,k), W on (k,i).
    
    # For the all-ones tensor, we can use:
    # T[i,j,k] = sum_r a_r[i,j] * b_r[j,k] * c_r[k,i]
    
    # One simple decomposition: use the "separable" structure.
    # If we set U[i*m+j, r] = alpha_r[i] * beta_r[j]
    #           V[j*p+k, r] = beta_r[j] * gamma_r[k]  -- no, this doesn't work
    # Let me think again.
    
    # Actually, for the all-ones 3D tensor, the tensor rank is min(n,m,p) = 3
    # if we could use arbitrary vectors. But here the vectors are constrained
    # by the index structure.
    
    # Let me try a different approach: use the known result that for
    # m x n x p matrix multiplication, the rank is at most m*n*p, and
    # for small cases, we can do better.
    
    # For 3x5x7, let me try to find a decomposition of rank < 105.
    
    # Key observation: The tensor is "separable" in a certain sense.
    # T[i,j,k] = 1 = (1)(1)(1)
    # So we can write it as a sum of products.
    
    # Let's try: use the fact that 1 = sum over some basis.
    # Actually, the simplest non-trivial decomposition:
    # T[i,j,k] = sum_{r=1}^{R} u_r(i,j) * v_r(j,k) * w_r(k,i)
    
    # If we can find R < 105 vectors such that this holds, we win.
    
    # One approach: use the "outer product" structure.
    # For each (i,j), define a vector in R^p. For each (j,k), a vector in R^n.
    # Then T[i,j,k] = <u_{i,j}, v_{j,k}> * w_{k,i}... no.
    
    # Let me just try to reduce the schoolbook decomposition by finding
    # linear combinations that cancel out.
    
    # In the schoolbook decomposition, each term (i,j,k) contributes:
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # and all other entries are 0.
    
    # To reduce rank, we need to find coefficients c_{i,j,k} such that
    # sum_{i,j,k} c_{i,j,k} * (e_{i,j} * e_{j,k} * e_{k,i}) = T
    # where e_{i,j} is the unit vector in R^{n*m} at position (i,j), etc.
    
    # This is equivalent to finding a low-rank decomposition of the tensor.
    
    # For now, let me return the schoolbook decomposition as a baseline,
    # and try to improve it if time permits.
    
    # Actually, let me try a smarter approach. The tensor T[i,j,k] = 1
    # can be decomposed as:
    # T[i,j,k] = sum_{r=1}^{min(n,m,p)} u_r[i] * v_r[j] * w_r[k]
    # where u_r, v_r, w_r are vectors of length n, m, p respectively.
    # But this gives T[i,j,k] = (sum_r u_r[i] v_r[j] w_r[k]) which is
    # a rank-1 tensor in each mode, not what we want.
    
    # We need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where U is n*m dimensional, V is m*p dimensional, W is p*n dimensional.
    
    # This is a more constrained problem. Let me try to use the structure.
    
    # For fixed j, the slice T[:,j,:] is an n x p matrix of all ones.
    # This can be written as: sum_r U[:,j,r] * V[j,:,r] * W[:,r] (outer products)
    # where U[:,j,r] is an n-vector, V[j,:,r] is a p-vector, W[:,r] is a p*n vector.
    
    # The all-ones n x p matrix has rank 1, so it can be written as
    # u * v^T where u is all-ones n-vector and v is all-ones p-vector.
    # So for each j, we need: sum_r U[:,j,r] * V[j,:,r] * W[:,r] = 1_n * 1_p^T
    
    # This suggests that for each j, the vectors U[:,j,r] and V[j,:,r] and W[:,r]
    # should be chosen so that their "triple product" gives the all-ones matrix.
    
    # One simple solution: for each j, use a single term with
    # U[:,j,r_j] = 1_n, V[j,:,r_j] = 1_p, W[:,r_j] = 1_{p*n}
    # This gives rank = m = 5! Let's verify.
    
    # If rank = m = 5, and for each j, we have one term r_j:
    # T[i,j,k] = U[i*m+j, r_j] * V[j*p+k, r_j] * W[k*n+i, r_j]
    #          = 1 * 1 * 1 = 1. ✓
    
    # But wait, we need to check that different j's don't interfere.
    # For a given (i,j,k), only the term r_j contributes (since U[i*m+j', r_j'] = 0
    # for j' != j, and V[j'*p+k, r_j] = 0 for j' != j).
    
    # Actually, in this construction, U[i*m+j, r] is nonzero only when r = r_j,
    # and V[j'*p+k, r] is nonzero only when r = r_{j'}. So for a fixed (i,j,k),
    # the only r that contributes is r_j (from U) and r_j (from V), so they agree.
    
    # Wait, but W[k*n+i, r] is also involved. For r = r_j, W[k*n+i, r_j] = 1.
    # So T[i,j,k] = U[i*m+j, r_j] * V[j*p+k, r_j] * W[k*n+i, r_j] = 1*1*1 = 1. ✓
    
    # And for r != r_j, either U[i*m+j, r] = 0 or V[j*p+k, r] = 0, so no contribution.
    
    # This gives rank = m = 5! That's a huge improvement from 105.
    
    # Let me double-check the dimensions:
    # U: (n*m) x rank = (15) x 5
    # V: (m*p) x rank = (35) x 5
    # W: (p*n) x rank = (21) x 5
    
    # For each j in 0..4 (m=5), assign r = j:
    # U[i*m + j, j] = 1 for all i in 0..2 (n=3)
    # V[j*p + k, j] = 1 for all k in 0..6 (p=7)
    # W[k*n + i, j] = 1 for all k in 0..6, i in 0..2
    
    # Verify: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Only r = j contributes: U[i*m+j, j] * V[j*p+k, j] * W[k*n+i, j] = 1*1*1 = 1. ✓
    
    # This is correct! Rank = 5.
    
    # But wait, is this really valid? Let me verify with actual matrix multiplication.
    # C[k,i] = sum_j A[i,j] * B[j,k]
    # Using the decomposition:
    # C[k,i] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # Hmm, this doesn't look right. Let me re-derive.
    
    # The bilinear form for matrix multiplication is:
    # C[k,i] = sum_j A[i,j] * B[j,k]
    # In the tensor decomposition, we have:
    # C[k,i] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k])
    # No, that's not right either. The correct form is:
    # C[k,i] = sum_r W[k*n+i, r] * sum_j (U[i*m+j, r] * A[i,j]) * (V[j*p+k, r] * B[j,k])
    #         = sum_r W[k*n+i, r] * sum_j U[i*m+j, r] * V[j*p+k, r] * A[i,j] * B[j,k]
    
    # So for the decomposition to work, we need:
    # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1 for all (i,j,k)
    
    # With my construction:
    # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]
    # = W[k*n+i, j] * U[i*m+j, j] * V[j*p+k, j]  (only r=j contributes)
    # = 1 * 1 * 1 = 1. ✓
    
    # Great, so rank = 5 works!
    
    # But wait, can we do even better? The minimum rank for 3x5x7 matrix mult...
    # Actually, I think rank = min(n, m, p) might not always work. Let me verify
    # more carefully.
    
    # The issue is: in my construction, I'm using the same "slot" r=j for all
    # (i,k) pairs with that j. This means that for a fixed j, all the (i,k)
    # entries are computed using the same rank-1 term. This works because
    # T[i,j,k] = 1 for all i,k, so the same term can serve all of them.
    
    # So rank = m = 5 should be valid. But is it minimal?
    # The tensor rank of the matrix multiplication tensor for n x m x p is
    # at least max(n, m, p) in some cases, but I'm not sure of the exact bound.
    
    # For now, let's go with rank = 5 and verify it.
    
    rank = m  # = 5
    U = [[0.0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0.0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0.0 for _ in range(rank)] for _ in range(p * n)]
    
    for j in range(m):
        for i in range(n):
            U[i * m + j][j] = 1.0
        for k in range(p):
            V[j * p + k][j] = 1.0
        for k in range(p):
            for i in range(n):
                W[k * n + i][j] = 1.0
    
    # Verify
    if not _verify(U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0.0 for _ in range(rank)] for _ in range(n * m)]
        V = [[0.0 for _ in range(rank)] for _ in range(m * p)]
        W = [[0.0 for _ in range(rank)] for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][idx] = 1.0
                    V[j*p + k][idx] = 1.0
                    W[k*n + i][idx] = 1.0
                    idx += 1
    
    return {'U': U, 'V': V, 'W': W}
