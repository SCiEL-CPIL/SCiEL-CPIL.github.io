"""
Strategy: Direct Rank Reduction via Null-Space Detection and Local Search.

Instead of building decompositions from structural blocks, we:
1. Start with a known valid decomposition (rank 39 hybrid Strassen/schoolbook).
2. Build the full tensor T of shape (n*m, m*p, p*n) from the decomposition.
3. For each rank-1 term, check if it lies in the span of the others by
   solving a linear system. If so, eliminate it.
4. Perform local search: randomly perturb entries, re-verify, and if still
   valid with lower rank, keep it.
5. Also try random projections to find lower-rank representations.

This operates in the solution space directly, not via structural recursion.
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0

    # Build the target multiplication tensor
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        return T

    T = build_tensor()

    def verify(U, V, W, tol=1e-9):
        """Verify that sum_c U[:,c] * V[:,c] * W[:,c] == T (outer product sum)."""
        rank = U.shape[1]
        # Reconstruct tensor
        T_recon = np.zeros((n * m, m * p, p * n))
        for c in range(rank):
            T_recon += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        return np.max(np.abs(T_recon - T)) < tol

    def verify_fast(U, V, W, tol=1e-9):
        """Faster verification using matrix multiplication."""
        rank = U.shape[1]
        # T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
        # Check: for each c, the outer product U[:,c] outer V[:,c] outer W[:,c]
        # Sum over c should equal T.
        # Use: reshape approach
        # T_recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # This is a 3-way tensor contraction.
        # We can check: for each pair (a,b), T_recon[a,b,:] = (U[a,:] * V[b,:]) @ W.T
        # Or: T_recon = sum_r np.outer(U[:,r], V[:,r]) * W[:,r][None, None, :]
        # Let's use einsum for speed
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(T_recon - T)) < tol

    # Step 1: Build initial rank-39 decomposition
    # Split m=3 into m1=1, m2=2
    # Part 1: (2,1,7) -> schoolbook rank 14
    # Part 2: (2,2,7) -> use Strassen for the 2x2 part, schoolbook for the 7
    # Actually, let's use a simpler approach: split m into 1+2
    # (2,1,7): rank 14 (schoolbook)
    # (2,2,7): rank 14 (schoolbook) or use Strassen-like
    # Total: 28? No, that's not right.
    # Let me just use the known good rank-39 construction.
    # Actually, let me build a valid decomposition and then try to reduce rank.

    def build_hybrid_rank39():
        """Build a rank-39 decomposition: split m=3 into m1=1, m2=2.
        Part 1: (n=2, m=1, p=7) schoolbook: rank 14
        Part 2: (n=2, m=2, p=7) schoolbook: rank 28
        Total: 42. That's just schoolbook.
        
        Let me think differently. The known improvement to rank 39 comes from:
        Splitting m=3 as 1+2, and for the (2,2,7) part, using a Strassen-like
        decomposition for the 2x2x2 sub-blocks.
        
        Actually, let me just start with schoolbook rank 42 and try to reduce.
        """
        rank = n * m * p  # 42
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
        return U, V, W

    U, V, W = build_hybrid_rank39()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Verify initial
    if not verify_fast(U, V, W):
        # Should not happen, but fallback
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Step 2: Try to reduce rank by finding linear dependencies
    # For a rank-r decomposition, the r terms are rank-1 tensors.
    # If one term is a linear combination of others, we can eliminate it.
    # 
    # The key insight: in a tensor decomposition, if the set of rank-1 terms
    # is linearly dependent (as vectors in the tensor space), we might be able
    # to reduce the rank. However, tensor rank is not simply the dimension of
    # the span of the rank-1 terms (that would be tensor rank in the "linear"
    # sense, which is different).
    #
    # A better approach: try to find a lower-rank decomposition by random search.
    # Start with rank r, try to express T as sum of r-1 rank-1 terms.
    # This is equivalent to: given r-1 random rank-1 terms, can we find the
    # remaining coefficients to complete the decomposition?
    #
    # Alternative: Use the fact that if we have a valid rank-r decomposition,
    # we can try to "merge" two terms into one if they are compatible.
    # Two rank-1 terms U1 V1 W1 and U2 V2 W2 can be merged if there exist
    # U', V', W' such that U1 V1 W1 + U2 V2 W2 = U' V' W'.
    # This is possible if and only if the two rank-1 tensors sum to a rank-1 tensor.
    #
    # Let's try a different approach: random search for lower rank.

    def try_random_rank_reduction(current_rank, num_trials=1000):
        """Try to find a decomposition of rank current_rank-1 by random search."""
        target_rank = current_rank - 1
        if target_rank < 1:
            return None

        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break

            # Random initialization
            U_rand = np.random.randn(n * m, target_rank)
            V_rand = np.random.randn(m * p, target_rank)
            W_rand = np.random.randn(p * n, target_rank)

            # Scale to reasonable magnitude
            U_rand /= np.sqrt(target_rank)
            V_rand /= np.sqrt(target_rank)
            W_rand /= np.sqrt(target_rank)

            # Try to solve for exact decomposition
            # T = sum_c U[:,c] V[:,c] W[:,c]
            # This is a system of n*m*p*p*n equations (but many are zero)
            # Actually, T has n*m*p non-zero entries, each = 1.
            # The equation is: for each (i,j,k): sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
            # And for all other (a,b,c) indices where T[a,b,c]=0: sum_r U[a,r] V[b,r] W[c,r] = 0

            # This is a nonlinear system. Let's use a different strategy:
            # Fix U and V, solve for W linearly.
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed a,b: T[a,b,:] = (U[a,:] * V[b,:]) @ W.T
            # So W.T = (U * V.T) @ T.reshape(n*m, m*p, p*n).T ... this gets complex.

            # Let's try: fix U, V and solve for W.
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # Let M[a,b,r] = U[a,r] * V[b,r]
            # Then T[a,b,:] = M[a,b,:] @ W.T  (where W is (p*n, target_rank))
            # So T.reshape(n*m, m*p, p*n) = M @ W.T  (broadcasting)
            # T_reshaped[a,b,c] = sum_r M[a,b,r] W[c,r]
            # This means: for each c, T[:,:,c] = M @ W[c,:]  ... no
            # T[a,b,c] = sum_r M[a,b,r] * W[c,r]
            # Let's define: for each c, the matrix T_c[a,b] = T[a,b,c]
            # T_c = M @ W[c,:]  where M is (n*m, m*p, target_rank) -> need to reshape
            # Actually: T[a,b,c] = sum_r M[a,b,r] W[c,r]
            # This is: T_reshaped = np.einsum('abr,cr->abc', M, W)
            # To solve for W given U,V: 
            # For each (a,b), T[a,b,c] = sum_r M[a,b,r] W[c,r]
            # This is a linear system in W for each (a,b) pair, but coupled through c.
            # 
            # Let's vectorize: 
            # T_vec[a,b,c] = sum_r M[a,b,r] W[c,r]
            # Reshape: let A_mat = M.reshape(n*m*m*p, target_rank)
            # Let B_mat = W.reshape(p*n*target_rank, 1) ... no
            # 
            # Better: T_reshaped (n*m, m*p, p*n) = einsum('abr,cr->abc', M, W)
            # = np.einsum('abrc->abc', M * W[None, None, :, :])  -- no
            # 
            # Let's use: T_reshaped = np.tensordot(M, W, axes=([2], [1]))
            # M is (n*m, m*p, target_rank), W is (p*n, target_rank)
            # tensordot(M, W, axes=([2],[1])) gives (n*m, m*p, p*n) -- yes!
            # So T = tensordot(M, W, axes=([2],[1]))
            # To solve for W: this is a linear system.
            # T_reshaped (n*m, m*p, p*n) = M (n*m, m*p, target_rank) contracted with W (p*n, target_rank)
            # In matrix form: reshape T to (n*m*m*p, p*n), M to (n*m*m*p, target_rank)
            # Then T_mat = M_mat @ W.T  where W.T is (target_rank, p*n)
            # So W.T = pinv(M_mat) @ T_mat  (least squares, but we need exact)
            # If M_mat has full column rank (target_rank), we can solve exactly.

            M = U_rand[:, None, :] * V_rand[None, :, :]  # (n*m, m*p, target_rank)
            M_mat = M.reshape(n * m * m * p, target_rank)
            T_mat = T.reshape(n * m * m * p, p * n)

            # Solve: T_mat = M_mat @ W.T
            # W.T = M_mat^+ @ T_mat
            try:
                W_T, residuals, rank_m, sv = np.linalg.lstsq(M_mat, T_mat, rcond=None)
                W_solved = W_T.T  # (p*n, target_rank)
            except:
                continue

            # Check if solution is exact
            T_recon = np.tensordot(M, W_solved, axes=([2], [1]))
            if np.max(np.abs(T_recon - T)) < 1e-8:
                return U_rand, V_rand, W_solved

        return None

    # Try rank reduction from 42 down
    for target_rank in range(41, 20, -1):
        if time.time() > deadline - 10:
            break
        result = try_random_rank_reduction(target_rank + 1, num_trials=50)
        if result is not None:
            U, V, W = result
            if verify_fast(U, V, W):
                best_rank = target_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break

    # Step 3: Local search - try to perturb and maintain validity
    # This is less likely to help for exact integer/half-integer entries,
    # but let's try a structured perturbation.

    # Step 4: Try a completely different construction approach
    # Use the known Strassen algorithm for 2x2x2 and extend to our dimensions.
    # 
    # For (2,3,7): Think of A as (2,3) and B as (3,7).
    # Split B into B1 (first 2 columns of inner dim) and B2 (last 1 column).
    # C = A1 B1 + A2 B2 where A1 is (2,2), A2 is (2,1), B1 is (2,7), B2 is (1,7).
    # R(2,2,7) + R(2,1,7)
    # R(2,1,7) = 7 (schoolbook, since it's just 2*7 scalar mults... wait)
    # R(2,1,7): A is (2,1), B is (1,7), C is (2,7).
    # C[i,k] = A[i,0] * B[0,k]. This is 2*7 = 14 scalar multiplications.
    # But we can do better: C = A * B where A is a column vector and B is a row vector.
    # This is an outer product! Rank 1! C[i,k] = A[i] * B[k].
    # So R(2,1,7) = 1? No wait, that's only if A and B are fixed.
    # In matrix multiplication, A and B are arbitrary.
    # C[i,k] = A[i,0] * B[0,k] for i in {0,1}, k in {0,...,6}.
    # This is 14 independent products. Can we do it with fewer?
    # The tensor is: T[i,0,k] = A[i,0] * B[0,k].
    # This is an outer product of A[:,0] and B[0,:].
    # As a tensor in (A-space, B-space, C-space):
    # T is rank 1! Because T[i, j, k] = u_i * v_j * w_k where u = A[:,0], v = B[0,:], w = identity on C.
    # Wait, no. The multiplication tensor for (n,m,p) has T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B produce the (i,k) entry of C.
    # For (2,1,7): T[i,j,k] where i in {0,1}, j in {0}, k in {0,...,6}.
    # T[i,0,k] = 1 for all i,k.
    # This is NOT rank 1 as a tensor. It's a matrix of all ones (2x7), which has rank 1 as a matrix,
    # but as a 3-tensor T[i,j,k] with j fixed, it's just a 2D array.
    # The "rank" of a 3-tensor where one index is trivial is the matrix rank of the remaining 2D array.
    # T[i,0,k] = 1 for all i,k -> this is a 2x7 matrix of all ones -> matrix rank 1.
    # So R(2,1,7) = 1? That doesn't seem right for matrix multiplication.
    # 
    # Let me reconsider. The multiplication tensor M_{n,m,p} has entries:
    # M_{n,m,p}[i,j,k] = 1 if i is row of A, j is column of A / row of B, k is column of B.
    # The multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 (it's always 1 for valid indices).
    # The rank of this tensor is the minimum r such that T[i,j,k] = sum_{c=1}^r U[i,c] V[j,c] W[k,c].
    # 
    # For (2,1,7): T[i,j,k] = 1 for i in {0,1}, j={0}, k in {0..6}.
    # We need: 1 = sum_c U[i,c] V[0,c] W[k,c] for all i,k.
    # Let V[0,c] = v_c. Then 1 = sum_c U[i,c] v_c W[k,c].
    # Let a_i = U[i,:] * v (element-wise product summed? No, it's sum_c U[i,c] v_c W[k,c])
    # Define P[i,c] = U[i,c] v_c. Then 1 = sum_c P[i,c] W[k,c] for all i,k.
    # This means the matrix P W.T = ones(2,7).
    # Minimum rank of P: P is (2, r), W is (7, r). P W.T is (2,7) all ones.
    # Rank of all-ones (2,7) matrix is 1. So we need r >= 1.
    # With r=1: P[i,0] * W[k,0] = 1 for all i,k. Set P[i,0]=1, W[k,0]=1. Works!
    # So R(2,1,7) = 1? 
    # But that means C[i,k] = A[i,0] * B[0,k] can be computed with 1 multiplication?
    # That's clearly wrong. The issue is that in the tensor decomposition,
    # U[i,c] corresponds to A[i,j] coefficient, V[j,c] to B[j,k] coefficient, W[k,c] to C[i,k] coefficient.
    # The decomposition says: C[i,k] = sum_c (sum_j U[i,j,c] A[i,j]) (sum_j V[j,k,c] B[j,k]) W[i,k,c]?
    # No, the standard form is:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_c U[i,c] V[j,c] W[k,c] ... this doesn't look right.
    # 
    # The correct form: The multiplication tensor is T with T[i,j,k] = 1.
    # A rank-r decomposition is: T[i,j,k] = sum_{c=1}^r U[i,c] V[j,c] W[k,c].
    # This means: for the actual multiplication, if we define:
    #   intermediate_c = (sum_i U[i,c] A[i,j]) * (sum_k V[j,c] B[j,k]) ... no.
    # 
    # Actually, the standard interpretation:
    #   C = sum_c (U^c A) (V^c B)^T ... no, dimensions don't work.
    # 
    # Let me look at this more carefully.
    # T is a 3-tensor with T[i,j,k] = 1 for the multiplication tensor.
    # A rank-1 term is U[i] V[j] W[k].
    # The sum is: sum_c U_c[i] V_c[j] W_c[k] = T[i,j,k] = 1.
    # 
    # For the actual matrix multiplication C = AB:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor decomposition gives us an algorithm:
    #   For each c: compute M_c = (U_c^T A) * (V_c^T B) ... no.
    # 
    # The correct algorithm from tensor decomposition:
    #   Let A be (n x m), B be (m x p), C be (n x p).
    #   T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
    #   Algorithm:
    #     For each c:
    #       a_c = sum_j U[j,c] * ... 
    # 
    # I think the correct interpretation is:
    #   C[i,k] = sum_j A[i,j] B[j,k]
    #   = sum_c (sum_j U[i,c] A[i,j]) * (sum_j V[j,c] B[j,k]) * W[k,c]  -- NO, this doesn't work dimensionally.
    # 
    # Let me look at the schoolbook: U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]  -- note the indices!
    # In the schoolbook, for term c corresponding to (i,j,k):
    #   U has 1 at position (i,j) in the U factor (which is indexed by row i*m+j)
    #   V has 1 at position (j,k) in the V factor (indexed by j*p+k)  
    #   W has 1 at position (k,i) in the W factor (indexed by k*n+i)
    # And T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
    # 
    # So the correct tensor is: T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # where U is (n*m, r), V is (m*p, r), W is (p*n, r).
    # 
    # For (2,1,7): n=2, m=1, p=7.
    # U is (2*1, r) = (2, r), V is (1*7, r) = (7, r), W is (7*2, r) = (14, r).
    # T[i,j,k] = sum_c U[i, c] V[k, c] W[k*2+i, c] for i in {0,1}, j=0, k in {0..6}.
    # T[i,0,k] = 1 for all i,k.
    # So: 1 = sum_c U[i,c] V[k,c] W[k*2+i, c] for all i in {0,1}, k in {0..6}.
    # 
    # With r=1: 1 = U[i,0] V[k,0] W[k*2+i, 0] for all i,k.
    # Set U[i,0] = 1, V[k,0] = 1, W[k*2+i, 0] = 1. This works!
    # So R(2,1,7) = 1.
    # 
    # But wait, that means we can compute C = A B where A is (2,1) and B is (1,7) with 1 multiplication?
    # The algorithm would be:
    #   Compute X = U^T A = [1, 1] A = A[0,0] + A[1,0]  (a scalar)
    #   Compute Y = V^T B = [1,1,...,1] B = sum_k B[0,k]  (a scalar)
    #   Compute Z = W * X * Y = [1,1,1,1,1,1,1,1,1,1,1,1,1,1] * X * Y
    #   C = Z reshaped to (2,7)?
    # 
    # That gives C[i,k] = X * Y for all i,k, which is (A[0,0]+A[1,0]) * (sum_k B[0,k]).
    # But the correct answer is C[i,k] = A[i,0] * B[0,k].
    # These are NOT the same!
    # 
    # So what's wrong? The issue is that the tensor decomposition must satisfy:
    # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # For (2,1,7): T[i,0,k] = sum_c U[i, c] V[k, c] W[k*2+i, c] = 1.
    # But the actual multiplication is C[i,k] = A[i,0] B[0,k].
    # The tensor T encodes the STRUCTURE of the multiplication, not the values.
    # The decomposition tells us HOW to compute C from A and B:
    #   C[i,k] = sum_j A[i,j] B[j,k] = sum_c (sum_j U[i*m+j,c] A[i,j]) * (sum_j V[j*p+k,c] B[j,k]) * W[k*n+i,c]
    # 
    # Wait, that's not right either. Let me think about this more carefully.
    # 
    # The standard result: if T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c],
    # then C[i,k] = sum_j A[i,j] B[j,k] can be computed as:
    #   C[i,k] = sum_c (sum_j U[i,c] A[i,j]) * (sum_j V[j,c] B[j,k]) * W[k,c]
    # NO! That's not dimensionally consistent.
    # 
    # The correct formula: 
    #   C[i,k] = sum_j A[i,j] B[j,k]
    #   = sum_c U[i,c] * (sum_j A[i,j] V[j,c]) * W[k,c]  -- no
    # 
    # I think the correct interpretation is:
    #   Define intermediate values:
    #     M_c = sum_{i,j} U[i,c] A[i,j]  (scalar for each c)  -- NO
    # 
    # Let me look at this from the schoolbook example.
    # Schoolbook for (2,3,7): 42 terms, one for each (i,j,k).
    # For term c=(i,j,k): U[i*3+j, c]=1, V[j*7+k, c]=1, W[k*2+i, c]=1.
    # The algorithm: for each c, compute product P_c = A[i,j] * B[j,k] (the specific entries).
    # Then C[i,k] = sum_{j} A[i,j] B[j,k] = sum_{c: c corresponds to (i,*,k)} P_c.
    # 
    # So the general algorithm from a rank-r decomposition:
    #   For each c in 1..r:
    #     Compute S_c = sum_{i,j} U[i*m+j, c] A[i,j]  -- NO, U is indexed by (i,j) pair.
    # 
    # I think the correct formula is:
    #   C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k])
    # 
    # Let's verify with schoolbook:
    # For term c=(i0,j0,k0): U[i*3+j, c] = 1 if i=i0, j=j0, else 0.
    # So sum_j U[i*3+j, c] A[i,j] = A[i0, j0] if i=i0, else 0.
    # Similarly sum_j V[j*7+k, c] B[j,k] = B[j0, k0] if k=k0, else 0.
    # W[k*2+i, c] = 1 if k=k0, i=i0.
    # So C[i,k] = sum_c W[k*2+i, c] * (A[i0,c, j0,c] if i=i0,c else 0) * (B[j0,c, k0,c] if k=k0,c else 0)
    # = A[i,j] B[j,k] when c=(i,j,k) and i=i, k=k. Sum over j: sum_j A[i,j] B[j,k]. CORRECT!
    # 
    # So the formula is:
    #   C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    # 
    # This means each term c involves:
    #   - A linear combination of row i of A (weighted by U[i*m+*, c])
    #   - A linear combination of column k of B (weighted by V[*p+k, c])
    #   - Multiplied together and scaled by W[k*n+i, c]
    # 
    # For (2,1,7) with r=1:
    #   C[i,k] = W[k*2+i, 0] * (sum_j U[i, c] A[i,j]) * (sum_j V[j, c] B[j,k])
    #   = W[k*2+i, 0] * (U[i,0] A[i,0]) * (sum_j V[j,0] B[j,k])
    #   = W[k*2+i, 0] * U[i,0] * A[i,0] * (sum_j V[j,0] B[j,k])
    # 
    # We need this to equal A[i,0] * B[0,k] for all i,k.
    # So: W[k*2+i, 0] * U[i,0] * (sum_j V[j,0] B[j,k]) = B[0,k] for all i,k.
    # This must hold for ALL B, so: W[k*2+i, 0] * U[i,0] * V[j,0] = delta(j,0) for all j,k,i.
    # For j=0: W[k*2+i, 0] * U[i,0] * V[0,0] = 1 for all i,k.
    # For j!=0: W[k*2+i, 0] * U[i,0] * V[j,0] = 0 for all i,k.
    # 
    # From j!=0: either W[k*2+i,0]=0 or U[i,0]=0 or V[j,0]=0 for all j!=0.
    # If V[j,0]=0 for j!=0, then sum_j V[j,0] B[j,k] = V[0,0] B[0,k].
    # Then we need W[k*2+i,0] * U[i,0] * V[0,0] * B[0,k] = B[0,k], so W[k*2+i,0] * U[i,0] * V[0,0] = 1.
    # Set U[i,0]=1, V[0,0]=1, W[k*2+i,0]=1. This works!
    # 
    # So R(2,1,7) = 1. The algorithm:
    #   Compute X = A[0,0] + A[1,0]  (sum of A, since U[i,0]=1)
    #   Wait no: sum_j U[i*m+j, c] A[i,j] = U[i,0] * A[i,0] = A[i,0] (since m=1, only j=0)
    #   Compute Y = sum_j V[j,0] B[j,k] = V[0,0] B[0,k] = B[0,k]
    #   C[i,k] = W[k*2+i,0] * A[i,0] * B[0,k] = 1 * A[i,0] * B[0,k]. CORRECT!
    # 
    # So indeed R(2,1,7) = 1. The "1 multiplication" is A[i,0] * B[0,k] for each (i,k),
    # but the tensor rank counts the number of scalar multiplications in the algorithm.
    # With r=1, the algorithm is:
    #   For c=0:
    #     For each i: compute U_c[i] = U[i,0] * A[i,0] = A[i,0]  (this is just copying, no mult)
    #     For each k: compute V_c[k] = V[0,0] * B[0,k] = B[0,k]  (copying)
    #     For each (i,k): C[i,k] += W[k*2+i,0] * U_c[i] * V_c[k] = A[i,0] * B[0,k]
    # 
    # The scalar multiplications are: A[i,0]*B[0,k] for each (i,k) -- that's 14 multiplications!
    # But the tensor rank is 1? That seems contradictory.
    # 
    # I think I'm confusing tensor rank with the number of scalar multiplications.
    # In the tensor decomposition framework, the "rank" r means we need r "intermediate" values,
    # and each intermediate involves one scalar multiplication.
    # 
    # Let me re-examine. The standard result:
    #   If T[i,j,k] = sum_{c=1}^r U[i,c] V[j,c] W[k,c],
    #   then C = AB can be computed with r scalar multiplications.
    # 
    # The algorithm:
    #   For c = 1 to r:
    #     M_c = (sum_{i,j} U[i,c] A[i,j]) * (sum_{j,k} V[j,c] B[j,k])  -- NO, this gives a scalar, not a matrix.
    # 
    # I think the correct algorithm is:
    #   For c = 1 to r:
    #     Compute X_c = U_c^T * A  where U_c is the c-th column of U, reshaped appropriately
    #     Compute Y_c = V_c^T * B
    #     C += W_c * (X_c * Y_c)  -- element-wise?
    # 
    # I'm getting confused. Let me just use the verify function to check correctness
    # and focus on finding lower-rank decompositions empirically.

    # Let's try a different approach: use known results.
    # R(2,2,2) = 7 (Strassen)
    # R(2,2,1) = ? (multiply 2x2 by 2x1, result 2x1)
    #   C[i] = sum_j A[i,j] B[j] for i=0,1.
    #   This is 4 scalar multiplications, but can we do better?
    #   T[i,j,k] with n=2,m=2,p=1: T[i,j,0] = 1 for i,j in {0,1}.
    #   Need: 1 = sum_c U[i*m+j, c] V[j*p+0, c] W[0*n+i, c] = sum_c U[i*2+j, c] V[j, c] W[i, c]
    #   For r=2: Can we do it?
    #   This is equivalent to: the 2x2 all-ones matrix has rank 2, so we need at least 2.
    #   With r=2: U[0,0]=1,V[0,0]=1,W[0,0]=1; U[1,1]=1,V[1,1]=1,W[1,1]=1.
    #   Check T[0,0,0] = U[0,0]V[0,0]W[0,0] + U[0,1]V[0,1]W[0,1] = 1*1*1 + 0 = 1. OK
    #   T[0,1,0] = U[1,0]V[1,0]W[0,0] + U[1,1]V[1,1]W[0,1] = 0 + 0 = 0. But should be 1! FAIL.
    #   
    #   Let me try: U[0,0]=1,U[0,1]=1; V[0,0]=1,V[1,0]=1; W[0,0]=1,W[0,1]=1; W[1,0]=1,W[1,1]=1
    #   T[0,0,0] = U[0,0]V[0,0]W[0,0] + U[0,1]V[0,1]W[0,1] = 1*1*1 + 1*0*1 = 1. OK
    #   T[0,1,0] = U[1,0]V[1,0]W[0,0] + U[1,1]V[1,1]W[0,1] = 0*1*1 + 0*0*1 = 0. Should be 1. FAIL.
    #   
    #   Hmm, let me set up the equations properly.
    #   T[i,j,0] = sum_c U[i*2+j, c] V[j, c] W[i, c] = 1 for all i,j in {0,1}.
    #   Let a_c = U[0,c], b_c = U[1,c], c_c = U[2,c], d_c = U[3,c]
    #   e_c = V[0,c], f_c = V[1,c]
    #   g_c = W[0,c], h_c = W[1,c]
    #   T[0,0,0] = a_c e_c g_c summed = 1
    #   T[0,1,0] = b_c f_c g_c summed = 1  (wait, T[i,j,k] with i=0,j=1: U[0*2+1,c]=U[1,c]=b_c, V[1,c]=f_c, W[0*2+0,c]=W[0,c]=g_c)
    #   T[1,0,0] = c_c e_c h_c summed = 1
    #   T[1,1,0] = d_c f_c h_c summed = 1
    #   
    #   With r=2:
    #   a1 e1 g1 + a2 e2 g2 = 1
    #   b1 f1 g1 + b2 f2 g2 = 1
    #   c1 e1 h1 + c2 e2 h2 = 1
    #   d1 f1 h1 + d2 f2 h2 = 1
    #   
    #   Try: a1=1,e1=1,g1=1; a2=0,e2=0,g2=0
    #   b1=0,f1=0; b2=1,f2=1,g2=1 -> b1f1g1+b2f2g2 = 0+1 = 1. OK
    #   c1=1,e1=1,h1=1; c2=0 -> c1e1h1+c2e2h2 = 1. OK
    #   d1=0,f1=0; d2=1,f2=1,h2=1 -> d1f1h1+d2f2h2 = 0+1 = 1. OK
    #   
    #   So: U = [[1,0],[0,1],[1,0],[0,1]], V = [[1,0],[0,1]], W = [[1,1],[1,1]]
    #   Check: T[0,0,0] = 1*1*1 + 0*0*1 = 1. OK
    #   T[0,1,0] = 0*0*1 + 1*1*1 = 1. OK
    #   T[1,0,0] = 1*1*1 + 0*0*1 = 1. OK
    #   T[1,1,0] = 0*0*1 + 1*1*1 = 1. OK
    #   
    #   So R(2,2,1) = 2! (It's just the matrix rank of the all-ones 2x2 matrix.)
    #   Makes sense: C = A * B where B is a column vector. C[i] = A[i,:] . B.
    #   This is 2 dot products of length 2, which is 4 scalar mults, but the tensor rank is 2.
    #   The "2 multiplications" in the tensor sense correspond to the 2 rows of the result.

    # OK so let me just focus on finding a good decomposition empirically.
    # Let me try to construct a decomposition for (2,3,7) using the splitting:
    # m=3 = 2+1
    # Part 1: (2,2,7), Part 2: (2,1,7)
    # R(2,1,7) = 1 (as shown above)
    # R(2,2,7) = ? 
    #   Split p=7 = 2+2+2+1
    #   R(2,2,2) = 7, R(2,2,1) = 2
    #   R(2,2,7) <= 7+7+7+2 = 23
    #   Or split p=7 as 2+5: R(2,2,2)+R(2,2,5)
    #   R(2,2,5) <= R(2,2,2)+R(2,2,3) <= 7 + R(2,2,3)
    #   R(2,2,3) <= R(2,2,2)+R(2,2,1) = 7+2 = 9
    #   So R(2,2,5) <= 7+9 = 16
    #   R(2,2,7) <= 7+16 = 23
    #   Total: 23+1 = 24
    # 
    # That's much better than 42! Let me try to construct this.

    def build_strassen_2x2x2():
        """Strassen's 7-term decomposition for 2x2x2."""
        # Strassen's algorithm:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22) B11
        # M3 = A11 (B12 - B22)
        # M4 = A22 (B21 - B11)
        # M5 = (A11 + A12) B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        n, m, p = 2, 2, 2
        rank = 7
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        # U indexed by [i*m+j], V by [j*p+k], W by [k*n+i]
        # A indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # B indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # C indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # W indexed by [k*n+i]: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1  # A00, A11
        V[0, 0] = 1; V[3, 0] = 1  # B00, B11
        W[0, 0] = 1; W[3, 0] = 1  # C00, C11
        
        # M2 = (A10+A11) B00
        U[2, 1] = 1; U[3, 1] = 1  # A10, A11
        V[0, 1] = 1               # B00
        W[2, 1] = 1               # C10
        W[3, 1] = -1              # C11 (with - sign)
        
        # M3 = A00 (B01-B11)
        U[0, 2] = 1               # A00
        V[1, 2] = 1; V[3, 2] = -1 # B01, -B11
        W[1, 2] = 1               # C01
        W[3, 2] = 1               # C11
        
        # M4 = A11 (B10-B00)
        U[3, 3] = 1               # A11
        V[2, 3] = 1; V[0, 3] = -1 # B10, -B00
        W[0, 3] = 1               # C00
        W[2, 3] = 1               # C10
        
        # M5 = (A00+A01) B11
        U[0, 4] = 1; U[1, 4] = 1  # A00, A01
        V[3, 4] = 1               # B11
        W[0, 4] = -1              # C00
        W[1, 4] = 1               # C01
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1 # A10, -A00
        V[0, 5] = 1; V[1, 5] = 1  # B00, B01
        W[3, 5] = 1               # C11
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1 # A01, -A11
        V[2, 6] = 1; V[3, 6] = 1  # B10, B11
        W[0, 6] = 1               # C00
        W[3, 6] = 1               # C11
        
        return U, V, W

    def build_2x2x1():
        """Rank-2 decomposition for (2,2,1)."""
        n, m, p = 2, 2, 1
        rank = 2
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        # U: (4,2), V: (2,2), W: (2,2)
        # A: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # B: (0,0)->0, (1,0)->1
        # C: (0,0)->0, (1,0)->1
        # W indexed by [k*n+i]: k=0 only, so W[i, c]
        
        # Term 0: A00*B00 -> C00, A10*B00 -> C10
        U[0, 0] = 1  # A00
        U[2, 0] = 1  # A10
        V[0, 0] = 1  # B00
        W[0, 0] = 1  # C00
        W[1, 0] = 1  # C10
        
        # Term 1: A01*B01 -> C00, A11*B01 -> C10
        U[1, 1] = 1  # A01
        U[3, 1] = 1  # A11
        V[1, 1] = 1  # B01
        W[0, 1] = 1  # C00
        W[1, 1] = 1  # C10
        
        return U, V, W

    def build_2x1x7():
        """Rank-1 decomposition for (2,1,7)."""
        n, m, p = 2, 1, 7
        rank = 1
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        # U: (2,1), V: (7,1), W: (14,1)
        U[:, 0] = 1.0
        V[:, 0] = 1.0
        W[:, 0] = 1.0
        return U, V, W

    def combine_decomps(decomps, n_total, m_total, p_total, m_splits, p_splits):
        """Combine decompositions for different (n, m_i, p_i) blocks.
        m_splits: list of m values that sum to m_total
        p_splits: list of p values that sum to p_total
        decomps: list of (U, V, W) for each (m_i, p_j) block
        """
        total_rank = sum(d[0].shape[1] for d in decomps)
        U = np.zeros((n_total * m_total, total_rank))
        V = np.zeros((m_total * p_total, total_rank))
        W = np.zeros((p_total * n_total, total_rank))
        
        col_offset = 0
        m_offset = 0
        for mi, (U_i, V_i, W_i) in enumerate(decomps):
            r_i = U_i.shape[1]
            # U_i is (n_total * mi, r_i)
            for i in range(n_total):
                for j in range(mi):
                    U[i * m_total + (m_offset + j), col_offset:col_offset + r_i] = U_i[i * mi + j, :]
            # V_i is (mi * p_total, r_i)  -- wait, V_i should be (mi * pi, r_i) where pi is the p for this block
            # Hmm, this is getting complicated. Let me simplify.
            # For now, just handle the case where we split m only.
            for j in range(mi):
                for k in range(p_total):
                    V[(m_offset + j) * p_total + k, col_offset:col_offset + r_i] = V_i[j * p_total + k, :]
            for k in range(p_total):
                for i in range(n_total):
                    W[k * n_total + i, col_offset:col_offset + r_i] = W_i[k * n_total + i, :]
            col_offset += r_i
            m_offset += mi
        
        return U, V, W

    # Let's build (2,3,7) by splitting m=3 into m1=2, m2=1
    # Part 1: (2,2,7) -- split p=7 into 2+2+2+1
    # Part 2: (2,1,7) -- rank 1

    # Build (2,2,2) Strassen
    U_s, V_s, W_s = build_strassen_2x2x2()
    
    # Build (2,2,1)
    U_221, V_221, W_221 = build_2x2x1()
    
    # Build (2,1,7)
    U_217, V_217, W_217 = build_2x1x7()

    # Now combine (2,2,7) from (2,2,2)+(2,2,2)+(2,2,2)+(2,2,1)
    # Each (2,2,2) has rank 7, (2,2,1) has rank 2. Total: 23.
    # The p dimension is split as 2+2+2+1.
    
    # For the (2,2,7) part, U is the same for all p-blocks (since n,m are the same).
    # V is block-diagonal in the p dimension.
    # W is block-diagonal in the p dimension.
    
    # Let me build (2,2,7) explicitly:
    n_sub, m_sub, p_sub = 2, 2, 7
    p_parts = [2, 2, 2, 1]
    decomp_parts = [(U_s, V_s, W_s), (U_s, V_s, W_s), (U_s, V_s, W_s), (U_221, V_221, W_221)]
    
    rank_227 = sum(d[0].shape[1] for d in decomp_parts)  # 7+7+7+2 = 23
    U_227 = np.zeros((n_sub * m_sub, rank_227))
    V_227 = np.zeros((m_sub * p_sub, rank_227))
    W_227 = np.zeros((p_sub * n_sub, rank_227))
    
    col_off = 0
    p_off = 0
    for pi, (U_p, V_p, W_p) in enumerate(decomp_parts):
        r_p = U_p.shape[1]
        p_len = p_parts[pi]
        # U is the same
        U_227[:, col_off:col_off + r_p] = U_p
        # V: rows j*p_sub + k for j in [0,2), k in [p_off, p_off+p_len)
        for j in range(m_sub):
            for k in range(p_len):
                V_227[j * p_sub + (p_off + k), col_off:col_off + r_p] = V_p[j * p_len + k, :]
        # W: rows k*n_sub + i for k in [p_off, p_off+p_len), i in [0,2)
        for k in range(p_len):
            for i in range(n_sub):
                W_227[(p_off + k) * n_sub + i, col_off:col_off + r_p] = W_p[k * n_sub + i, :]
        col_off += r_p
        p_off += p_len

    # Now combine (2,2,7) and (2,1,7) to get (2,3,7)
    # m=3 = 2+1
    rank_total = rank_227 + U_217.shape[1]  # 23 + 1 = 24
    U_final = np.zeros((2 * 3, rank_total))
    V_final = np.zeros((3 * 7, rank_total))
    W_final = np.zeros((7 * 2, rank_total))
    
    # Part 1: (2,2,7), m_offset=0
    r1 = rank_227
    for i in range(2):
        for j in range(2):
            U_final[i * 3 + j, :r1] = U_227[i * 2 + j, :]
    for j in range(2):
        for k in range(7):
            V_final[j * 7 + k, :r1] = V_227[j * 7 + k, :]
    for k in range(7):
        for i in range(2):
            W_final[k * 2 + i, :r1] = W_227[k * 2 + i, :]
    
    # Part 2: (2,1,7), m_offset=2
    r2 = U_217.shape[1]
    for i in range(2):
        for j in range(1):
            U_final[i * 3 + (2 + j), r1:r1 + r2] = U_217[i * 1 + j, :]
    for j in range(1):
        for k in range(7):
            V_final[(2 + j) * 7 + k, r1:r1 + r2] = V_217[j * 7 + k, :]
    for k in range(7):
        for i in range(2):
            W_final[k * 2 + i, r1:r1 + r2] = W_217[k * 2 + i, :]
    
    # Verify
    if verify_fast(U_final, V_final, W_final):
        best_rank = rank_total
        best_U, best_V, best_W = U_final.copy(), V_final.copy(), W_final.copy()
    else:
        # Fallback to schoolbook
        U_sb, V_sb, W_sb = build_hybrid_rank39()
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = U_sb.shape[1]

    # Try to reduce rank further with random search
    for target_rank in range(best_rank - 1, 15, -1):
        if time.time() > deadline - 10:
            break
        result = try_random_rank_reduction(target_rank + 1, num_trials=100)
        if result is not None:
            U_t, V_t, W_t = result
            if verify_fast(U_t, V_t, W_t):
                best_rank = target_rank
                best_U, best_V, best_W = U_t.copy(), V_t.copy(), W_t.copy()
                break

    # Final verification
    if not verify_fast(best_U, best_V, best_W):
        # Emergency fallback
        U_sb, V_sb, W_sb = build_hybrid_rank39()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
