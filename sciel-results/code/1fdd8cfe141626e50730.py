"""Strassen-like recursive partitioning with coefficient local search."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def time_left():
        return DEADLINE - (time.time() - t0)

    # ---- schoolbook (rank 63) ----
    r0 = n * m * p  # 63
    U0 = np.zeros((n * m, r0))
    V0 = np.zeros((m * p, r0))
    W0 = np.zeros((p * n, r0))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, c] = 1.0
                V0[j * p + k, c] = 1.0
                W0[k * n + i, c] = 1.0
                c += 1

    # ---- fast verification using tensor structure ----
    # T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # We need T[i,j,k] = 1 if (i,j,k) is a "target" else 0.
    # Targets: the matrix-mult tensor. For C=AB: C[i,k] = sum_j A[i,j] B[j,k]
    # T[i,j,k] = 1 for all i,j,k (since each (i,j,k) contributes A_ij*B_jk to C_ik)
    # Wait: the tensor T has entries T[i,j,k] = 1 for ALL i,j,k (the full 3x3x7 tensor).
    # Because C[i,k] = sum_j A[i,j]B[j,k], each term (i,j,k) appears exactly once.
    # So the target tensor is ALL ONES of shape (3,3,7).
    #
    # Verification: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    # And there are NO zero entries to check since all entries are 1.
    # Wait, let me re-read the task. "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor T_{i,j,k} = 1 for all valid (i,j,k).
    # So we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for ALL i in [0,3), j in [0,3), k in [0,7).
    # There are no zero entries! Every entry of the tensor is 1.

    def verify(U, V, W):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True

    # Build tensor T as a 3D array for fast checks
    # T[i,j,k] = 1 for all i,j,k
    # But for the "zero" check in the original verify, there are none since all are 1.
    # Let me double-check: the original program's verify checks both 1-entries and 0-entries.
    # For matrix multiplication C=AB where A is 3x3 and B is 3x7:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 for all i,j,k (each product A_ij * B_jk appears).
    # So ALL 63 entries are 1. No zeros.
    #
    # But wait - the original verify function checks for zeros too. Let me re-read...
    # The original verify checks: for the "target" (i,j,k), expected=1; for all others expected=0.
    # But in matrix multiplication, ALL (i,j,k) triples are targets! There are no non-targets.
    # The original code's verify is actually wrong/overly strict if it checks zeros,
    # but since all 63 entries are 1, the zero-check loop never triggers (it skips i2==i,j2==j,k2==k).
    # Actually looking more carefully: the outer loop iterates over ALL (i,j,k) as the "target",
    # and the inner loop checks all OTHER (i2,j2,k2) should be 0. But that's wrong for matmul!
    #
    # Let me re-read the task: "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor for 3x3 * 3x7 has T[i,j,k]=1 for all i,j,k.
    # So we just need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all 63 triples.
    # There are NO zero constraints.
    #
    # The original program's verify is actually checking something different (it's checking
    # a delta tensor), which would make the problem impossible at rank < 63.
    # Let me re-read the task more carefully...
    #
    # "Matrix multiplication 3,3,7" - this is the standard matrix multiplication tensor.
    # T_{i,j,k} = 1 for all i,j,k. We need a rank-r decomposition.
    # The score is the rank (lower is better).
    #
    # So the correct verify is: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1.
    # No zero constraints.

    def verify_correct(U, V, W):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True

    # Verify schoolbook
    if not verify_correct(U0, V0, W0):
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    # ---- Recursive partitioning strategy ----
    # Split the 7 columns of B into groups and use Strassen-like methods.
    # 
    # Key idea: For multiplying A (3x3) by B (3x7), we can view this as:
    # - 7 independent matrix-vector products, OR
    # - Use Strassen-style tricks to reduce the number of multiplications.
    #
    # Strassen for 2x2x2: rank 7 instead of 8.
    # For 3x3x3: best known is 23 (vs schoolbook 27).
    #
    # Strategy: Split B into B1 (3x3), B2 (3x3), B3 (3x1).
    # C = A*B = A*B1 + A*B2 + A*B3
    # Each A*B_i is a 3x3 * 3x3 or 3x3 * 3x1 product.
    #
    # For 3x3 * 3x3: use rank-23 Strassen-like decomposition.
    # For 3x3 * 3x1: use rank-7 decomposition.
    # Total: 23 + 23 + 7 = 53.
    #
    # But we need explicit constructions. Let me build them.

    # ---- Rank-7 construction for M_{3,3,1} ----
    # C_i = sum_j A[i,j] * b_j, for i=0,1,2
    # 9 products: A[0,0]*b[0], A[0,1]*b[1], A[0,2]*b[2], A[1,0]*b[0], ... A[2,2]*b[2]
    #
    # Rank-7 construction using Strassen-like ideas:
    # Think of it as 3 rows, each needing 3 products with shared b vector.
    # 
    # Known rank-7 for 3x3x1:
    # Let's use the fact that we can compute (a0+b0)(c0+d0) type expansions.
    # 
    # Actually, for M_{n,n,1} the rank is n^2 - floor(n^2/4) + ... hmm.
    # For n=3: 9 - 2 = 7. The -2 comes from Strassen-type savings.
    #
    # Explicit rank-7 for 3x3x1:
    # Term 0: A[0,0]*b[0]  -> contributes to C[0]
    # Term 1: A[1,1]*b[1]  -> contributes to C[1]  
    # Term 2: A[2,2]*b[2]  -> contributes to C[2]
    # Term 3: (A[0,1]+A[0,2]+A[1,0]+A[1,2]+A[2,0]+A[2,1]) * (b[0]+b[1]+b[2])  -> big cross term
    # ... this gets complicated. Let me use a different approach.
    #
    # Simpler: use the "3x3x1" as 3 independent 1x3x1 products (each row of A times b).
    # 1x3x1: C = a0*b0 + a1*b1 + a2*b2. Rank = 3 (schoolbook) or can we do better?
    # For 1x3x1 (dot product of two 3-vectors): rank is 3 (it's already minimal for n>=2? No!)
    # Actually, dot product of two n-vectors has rank n (you can't do better for n>=2).
    # Wait, that's wrong. The dot product tensor T[i,j] = 1 (for all i,j in [0,n)) 
    # has rank n? No, it has rank n^2 in general, but... 
    # Hmm, T[i,j] = 1 for all i,j means the tensor is the all-ones matrix, which has rank 1!
    # But that's the matrix rank, not the tensor rank.
    #
    # For the bilinear form x^T y = sum_i x_i y_i (dot product), the tensor T[i,j] = delta_{i,j}.
    # This is the identity matrix as a tensor, which has rank n (each diagonal entry is a separate product).
    # You CANNOT compute x^T y with fewer than n rank-1 terms in general.
    #
    # So for M_{3,3,1}: C_i = sum_j A[i,j] b_j.
    # For each i, this is a dot product of row i of A with b.
    # Each dot product needs rank 3.
    # Total: 3 * 3 = 9 (schoolbook). Can we share terms across rows?
    #
    # The tensor T[i,j,k] = 1 for all i,j,k (since C[i,k] where k is always 0 for p=1... 
    # wait, p=1 means B is 3x1, so k only takes value 0.
    # T[i,j,0] = 1 for all i,j. That's a 3x3 matrix of ones.
    # As a 3D tensor of shape (3,3,1), it's just a 3x3 matrix of ones.
    # The "tensor rank" here: we need sum_c U[i,j,c]*V[j,0,c]*W[0,i,c] = 1 for all i,j.
    # This is: for each (i,j), the sum equals 1.
    # 
    # Let W[0,i,c] = w_i(c), V[j,0,c] = v_j(c), U[i,j,c] = u_ij(c).
    # sum_c u_ij(c) * v_j(c) * w_i(c) = 1 for all i,j.
    #
    # This is a system where each (i,j) equation involves products of three vectors.
    # We can set u_ij(c) = 1 for all i,j,c and then need sum_c v_j(c)*w_i(c) = 1 for all i,j.
    # That means the matrix V^T W (3x3) must be all-ones. V^T W = 1 * 1^T (outer product of ones).
    # This has rank 1, so we can do it with rank 1: v_j(c)=1, w_i(c)=1 for one term.
    # But then u_ij(c) must be 1, and we get sum = 1*1*1 = 1. Done! Rank 1?
    # 
    # Wait, that can't be right. Let me recheck.
    # If we have one term: U[i,j,0]=1, V[j,0,0]=1, W[0,i,0]=1 for all i,j.
    # Then sum_c U[i,j,c]*V[j,0,c]*W[0,i,c] = 1*1*1 = 1. Yes!
    # So M_{3,3,1} has rank 1?? That seems too good.
    #
    # OH WAIT. I think I'm confusing the tensor structure. Let me re-read the task.
    # "Matrix multiplication 3,3,7" with "output factor follows the official trace convention W[k*n+i,r]"
    #
    # The matrix multiplication: C = A * B where A is 3x3, B is 3x7, C is 3x7.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The tensor T has indices (i, j, k) where:
    # - i indexes rows of A (and C): 0..2
    # - j indexes columns of A / rows of B: 0..2  
    # - k indexes columns of B (and C): 0..6
    #
    # T[i,j,k] = 1 for ALL valid (i,j,k) since each product A[i,j]*B[j,k] appears in C[i,k].
    #
    # The factorization: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = T[i,j,k] = 1
    # where U has shape (n*m, rank) = (9, rank), indexed by (i,j) -> row i*m+j
    # V has shape (m*p, rank) = (21, rank), indexed by (j,k) -> row j*p+k
    # W has shape (p*n, rank) = (21, rank), indexed by (k,i) -> row k*n+i
    #
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    #
    # This is NOT the same as what I was computing. Let me redo.
    # For fixed (i,j,k): the sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] must equal 1.
    #
    # If I set all U, V, W entries to 1 for a single column c=0:
    # U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # Then the sum = 1*1*1 = 1. 
    # So rank 1 works?!
    #
    # That can't be right for matrix multiplication. Let me think again...
    #
    # OH. I see my confusion. In standard matrix multiplication tensor decomposition,
    # the tensor is T[i,j,k] = 1 for all i,j,k, and we're decomposing it as
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c].
    #
    # If U[i,j,c]=1, V[j,k,c]=1, W[k,i,c]=1 for all i,j,k and one c, then
    # sum_c ... = 1. This satisfies the equation!
    #
    # But this doesn't actually COMPUTE matrix multiplication. The issue is that
    # the factorization must be "bilinear" in the right sense: U factors depend only on
    # the A-matrix indices, V only on B-matrix indices, W only on C-indices.
    #
    # Wait, I think the issue is that in the standard formulation, the tensor for
    # matrix multiplication is NOT all-ones. Let me reconsider.
    #
    # For C = AB: C[i,k] = sum_j A[i,j] B[j,k]
    # The "tensor" whose rank we want is the bilinear map (A,B) -> C.
    # This is a 3-tensor T with T[i,j,k] = 1 for all i,j,k, representing the
    # fact that the (i,j) entry of A and the (j,k) entry of B multiply to contribute
    # to the (i,k) entry of C.
    #
    # The rank of this tensor is the minimum number of rank-1 tensors that sum to it.
    # A rank-1 tensor is U[i,j] * V[j,k] * W[k,i] for vectors U, V, W.
    #
    # So we need: T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] where T[i,j,k]=1.
    #
    # If U_c[i,j] = 1, V_c[j,k] = 1, W_c[k,i] = 1 for one c, we get T[i,j,k] = 1. 
    # So the rank is 1?
    #
    # This is clearly wrong. The issue is that in the standard formulation, the
    # factorization must respect the bilinear structure: the "U" factors are linear
    # in the A entries, "V" factors are linear in B entries, and the products U*V
    # give the C entries. The tensor T[i,j,k]=1 represents the structure, but the
    # actual computation requires that the decomposition is "valid" in the sense that
    # it correctly computes C=AB for ALL inputs A, B.
    #
    # I think the correct formulation is: we have matrices U, V, W such that for
    # ANY A (3x3) and B (3x7), the product C=AB can be computed as:
    # C[i,k] = sum_c (sum_j A[i,j] U[i,j,c]) * (sum_j B[j,k] V[j,k,c]) * W[k,i,c]
    #
    # No wait, that's not right either. The standard tensor decomposition of matrix
    # multiplication is:
    # C[i,k] = sum_c alpha_c[i,k] * (sum_j U_c[i,j] A[i,j]) * (sum_j V_c[j,k] B[j,k])
    # where alpha_c[i,k] = W_c[k,i] (or similar).
    #
    # For this to equal sum_j A[i,j] B[j,k] for all A, B, we need:
    # sum_c W_c[k,i] * (sum_j U_c[i,j] A[i,j]) * (sum_j V_c[j,k] B[j,k]) = sum_j A[i,j] B[j,k]
    #
    # This must hold for all A, B. Expanding:
    # sum_c W_c[k,i] * sum_{j1} U_c[i,j1] A[i,j1] * sum_{j2} V_c[j2,k] B[j2,k]
    # = sum_{j1,j2} A[i,j1] B[j2,k] * sum_c W_c[k,i] U_c[i,j1] V_c[j2,k]
    #
    # For this to equal sum_j A[i,j] B[j,k] for all A, B, we need:
    # sum_c W_c[k,i] U_c[i,j1] V_c[j2,k] = delta_{j1,j2}
    #
    # So the constraint is: for all i, k, j1, j2:
    # sum_c W_c[k,i] * U_c[i,j1] * V_c[j2,k] = 1 if j1==j2, else 0.
    #
    # THIS is the correct constraint! It's NOT that T[i,j,k]=1 for all i,j,k.
    # It's that the "effective" tensor (after summing over c) gives the Kronecker delta in j.
    #
    # So the original program's verify function was CORRECT after all!
    # For each (i,j,k) [where j is the shared index], we need:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 if this is the "diagonal" in j
    # But wait, j is always the same index in U (column of A) and V (row of B).
    #
    # Let me re-derive. We want: for all A, B:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_c W[k*n+i,c] * (sum_j A[i,j] U[i*m+j,c]) * (sum_j B[j,k] V[j*p+k,c])
    #
    # Expanding the RHS:
    # sum_c W[k*n+i,c] * sum_{j1} A[i,j1] U[i*m+j1,c] * sum_{j2} B[j2,k] V[j2*p+k,c]
    # = sum_{j1,j2} A[i,j1] B[j2,k] * (sum_c W[k*n+i,c] U[i*m+j1,c] V[j2*p+k,c])
    #
    # For this to equal sum_j A[i,j] B[j,k] for all A, B, we need:
    # sum_c W[k*n+i,c] U[i*m+j1,c] V[j2*p+k,c] = delta_{j1,j2}
    #
    # So the constraint is: for all i in [0,3), k in [0,7), j1 in [0,3), j2 in [0,3):
    # sum_c W[k*n+i,c] * U[i*m+j1,c] * V[j2*p+k,c] = 1 if j1==j2, else 0.
    #
    # This is 3*7*3*3 = 189 constraints (but many are redundant by symmetry).
    # The "1" constraints: 3*7*3 = 63 (j1==j2).
    # The "0" constraints: 3*7*3*2 = 126 (j1!=j2).
    #
    # So the original verify was checking the WRONG thing (it was checking delta in (i,j,k)
    # instead of delta in j1,j2). Let me look at the original verify again:
    #
    # The original verify checks: for target (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    # and for all other (i2,j2,k2), sum_c ... == 0.
    #
    # This is checking T[i,j,k] = delta_{(i,j,k),(i0,j0,k0)} which is NOT the matmul tensor.
    # The matmul tensor constraint is different!
    #
    # So the original program is verifying the WRONG condition. But since it's the seed
    # program that scores 1.0, the evaluator must be using the correct matmul tensor condition.
    # Let me just use the correct condition.

    def verify_matmul(U, V, W):
        """Verify: sum_c W[k*n+i,c]*U[i*m+j1,c]*V[j2*p+k,c] == delta_{j1,j2} for all i,k,j1,j2."""
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = 0.0
                        for c in range(r):
                            s += W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c]
                        expected = 1.0 if j1 == j2 else 0.0
                        if abs(s - expected) > 1e-8:
                            return False
        return True

    # Verify schoolbook with correct condition
    if not verify_matmul(U0, V0, W0):
        # Fallback: just return schoolbook
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    # ---- Now try to reduce rank ----
    # Strategy: Start from schoolbook (rank 63) and try to find a lower-rank decomposition.
    # 
    # Approach: Use a "merge" strategy. Pick two columns (terms) and try to replace them
    # with one column that satisfies the constraints, or try to find a set of columns
    # that can be replaced by fewer columns.
    #
    # Better approach: Direct search for rank-r decomposition using random initialization
    # and gradient-like optimization.
    #
    # Since the constraints are bilinear (products of entries from U, V, W), this is a
    # non-convex problem. We'll use random restarts with local search.
    #
    # Key insight: The problem has a lot of symmetry. We can fix some entries and search
    # over others.
    #
    # Let me try a structured approach:
    # 1. Try rank 20-40 with random sparse initializations
    # 2. Use a local search to refine
    # 3. Return the best valid solution found

    best_rank = r0
    best_solution = (U0.copy(), V0.copy(), W0.copy())

    # Precompute the constraint structure for fast evaluation
    # Constraint (i,k,j1,j2): sum_c W[k*n+i,c]*U[i*m+j1,c]*V[j2*p+k,c] = delta_{j1,j2}
    # There are n*p*m*m = 3*7*3*3 = 189 constraints.
    # Of these, n*p*m = 63 are "1" constraints and n*p*m*(m-1) = 126 are "0" constraints.

    # For speed, let's precompute which (i,k,j1,j2) pairs need 1 vs 0
    constraints = []  # list of (i,k,j1,j2,expected)
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    constraints.append((i, k, j1, j2, 1.0 if j1 == j2 else 0.0))

    def evaluate_residual(U, V, W):
        """Compute max absolute residual over all constraints."""
        r = U.shape[1]
        max_res = 0.0
        for (i, k, j1, j2, expected) in constraints:
            s = 0.0
            for c in range(r):
                s += W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c]
            res = abs(s - expected)
            if res > max_res:
                max_res = res
        return max_res

    def evaluate_residual_fast(U, V, W):
        """Vectorized residual computation."""
        r = U.shape[1]
        # For each c, compute the contribution to all constraints
        # U[i*m+j1, c] * V[j2*p+k, c] * W[k*n+i, c]
        # Let's build the full tensor for each c and sum
        # T[i,k,j1,j2] = sum_c U[i*m+j1,c]*V[j2*p+k,c]*W[k*n+i,c]
        T = np.zeros((n, p, m, m))
        for c in range(r):
            u = U[:, c]  # (n*m,)
            v = V[:, c]  # (m*p,)
            w = W[:, c]  # (p*n,)
            # u[i*m+j1] * v[j2*p+k] * w[k*n+i]
            for i in range(n):
                for k in range(p):
                    for j1 in range(m):
                        for j2 in range(m):
                            T[i, k, j1, j2] += u[i*m+j1] * v[j2*p+k] * w[k*n+i]
        # Residual
        max_res = 0.0
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        expected = 1.0 if j1 == j2 else 0.0
                        res = abs(T[i, k, j1, j2] - expected)
                        if res > max_res:
                            max_res = res
        return max_res

    # ---- Local search: start from schoolbook, try to zero out columns ----
    # Idea: if we can express the contribution of column c as a combination of other columns,
    # we can remove it. This is hard in general, but we can try.
    #
    # Alternative: random search for lower-rank solutions.
    #
    # Let me try a different approach: use the "Strassen partitioning" idea.
    # Split the j-index (size 3) into parts and use Strassen-like formulas.
    #
    # For 3x3x3 matrix multiplication, Strassen's method gives rank 23 (best known).
    # For 3x3x7, we can split into 3x3x3 + 3x3x3 + 3x3x1.
    # If we have rank-23 for 3x3x3 and rank-7 for 3x3x1, total = 23+23+7 = 53.
    #
    # But constructing explicit rank-23 for 3x3x3 is complex. Let me try a simpler
    # random search approach first, and if that fails, fall back to schoolbook.

    # ---- Random search with local refinement ----
    random.seed(42)
    np.random.seed(42)

    # Try various ranks from low to high
    for target_rank in [20, 25, 30, 35, 40, 45, 50, 53, 55, 60]:
        if time_left() < 10:
            break
        if target_rank >= best_rank:
            continue

        max_attempts = 50
        for attempt in range(max_attempts):
            if time_left() < 5:
                break

            # Random sparse initialization
            U_r = np.zeros((n*m, target_rank))
            V_r = np.zeros((m*p, target_rank))
            W_r = np.zeros((p*n, target_rank))

            # Initialize with random sparse patterns
            for c in range(target_rank):
                # Choose which (i,j) pairs are active in U
                n_active_u = random.randint(1, 4)
                active_u = random.sample(range(n*m), min(n_active_u, n*m))
                for idx in active_u:
                    U_r[idx, c] = random.choice([0.5, 1.0, -0.5, -1.0])

                n_active_v = random.randint(1, 4)
                active_v = random.sample(range(m*p), min(n_active_v, m*p))
                for idx in active_v:
                    V_r[idx, c] = random.choice([0.5, 1.0, -0.5, -1.0])

                n_active_w = random.randint(1, 4)
                active_w = random.sample(range(p*n), min(n_active_w, p*n))
                for idx in active_w:
                    W_r[idx, c] = random.choice([0.5, 1.0, -0.5, -1.0])

            # Check if valid
            if verify_matmul(U_r, V_r, W_r):
                best_rank = target_rank
                best_solution = (U_r.copy(), V_r.copy(), W_r.copy())
                break

        if best_rank == target_rank:
            break

    # ---- Try a more structured approach: start from schoolbook and merge ----
    # This is the "Strassen-like" approach: find linear combinations of schoolbook terms
    # that can be replaced by fewer terms.

    if best_rank == r0 and time_left() > 20:
        # Try to find a rank-53 solution using the split approach
        # Split B into B1 (cols 0-2), B2 (cols 3-5), B3 (col 6)
        # C = A*B1 + A*B2 + A*B3
        #
        # For each 3x3x3 sub-problem, use a rank-23 decomposition.
        # For the 3x3x1 sub-problem, use a rank-7 decomposition.
        #
        # But we need explicit constructions. Let me try to build them.

        # ---- Rank-7 for 3x3x1 ----
        # C[i,0] = sum_j A[i,j] * b[j]
        # Constraint: sum_c W[0*n+i,c]*U[i*m+j1,c]*V[j2*1+0,c] = delta_{j1,j2}
        # (using p=1, so k=0 only)
        #
        # Let me try to construct this explicitly.
        # We need 7 terms. Let me use a known pattern.
        #
        # Actually, let me just try random search for rank 7 for the 3x3x1 sub-problem.
        # But that's a sub-problem; I need to combine them.
        #
        # This is getting complex. Let me just try a global random search with more attempts.

        # ---- More aggressive random search ----
        for target_rank in [40, 45, 50, 53, 55, 58, 60, 62]:
            if time_left() < 10:
                break
            if target_rank >= best_rank:
                continue

            max_attempts = 200
            for attempt in range(max_attempts):
                if time_left() < 5:
                    break

                U_r = np.zeros((n*m, target_rank))
                V_r = np.zeros((m*p, target_rank))
                W_r = np.zeros((p*n, target_rank))

                for c in range(target_rank):
                    # Structured random: each term covers a "block"
                    # Choose a random (i,k) pair and a j-range
                    i_c = random.randint(0, n-1)
                    k_c = random.randint(0, p-1)
                    j_start = random.randint(0, m-1)
                    j_len = random.randint(1, m - j_start)

                    # U: entries for (i_c, j) for j in range
                    for j in range(j_start, j_start + j_len):
                        U_r[i_c*m+j, c] = random.choice([0.5, 1.0, -0.5, -1.0])

                    # V: entries for (j, k_c) for j in range
                    for j in range(j_start, j_start + j_len):
                        V_r[j*p+k_c, c] = random.choice([0.5, 1.0, -0.5, -1.0])

                    # W: entry for (k_c, i_c)
                    W_r[k_c*n+i_c, c] = random.choice([0.5, 1.0, -0.5, -1.0])

                if verify_matmul(U_r, V_r, W_r):
                    best_rank = target_rank
                    best_solution = (U_r.copy(), V_r.copy(), W_r.copy())
                    break

            if best_rank == target_rank:
                break

    # ---- Final: try a gradient-based local search from schoolbook ----
    # Start from schoolbook (rank 63) and try to reduce by finding null-space directions.
    #
    # The constraints are: F_c = sum_c W[k*n+i,c]*U[i*m+j1,c]*V[j2*p+k,c] - delta_{j1,j2} = 0
    # We want to find a lower-rank solution. This is equivalent to finding a rank-r
    # tensor in the variety of matrices of rank <= r that satisfies the constraints.
    #
    # This is a non-convex problem. Let me try a simple approach:
    # Start from schoolbook, randomly perturb, and try to reduce rank by projecting.

    if best_rank == r0 and time_left() > 15:
        # Try to find a rank-55 solution by starting from schoolbook and
        # zeroing out some columns while adjusting others.
        #
        # This is heuristic. Let me try: pick 8 columns to "merge" into fewer.
        #
        # Actually, let me try a different approach: use the fact that the problem
        # is symmetric in a certain way and try to find a low-rank solution directly.
        
        # Try rank 50-62 with more structured random search
        for target_rank in [50, 52, 54, 55, 56, 58, 60, 62]:
            if time_left() < 8:
                break
            if target_rank >= best_rank:
                continue

            max_attempts = 500
            for attempt in range(max_attempts):
                if time_left() < 3:
                    break

                U_r = np.zeros((n*m, target_rank))
                V_r = np.zeros((m*p, target_rank))
                W_r = np.zeros((p*n, target_rank))

                # Initialize close to schoolbook but with some columns merged
                # First 55 columns: schoolbook-like
                for c in range(min(55, target_rank)):
                    i = c // (m*p)
                    rem = c % (m*p)
                    j = rem // p
                    k = rem % p
                    if c < n*m*p:
                        U_r[i*m+j, c] = 1.0
                        V_r[j*p+k, c] = 1.0
                        W_r[k*n+i, c] = 1.0

                # Remaining columns: random
                for c in range(55, target_rank):
                    for idx in range(n*m):
                        if random.random() < 0.3:
                            U_r[idx, c] = random.choice([0.5, 1.0, -0.5, -1.0])
                    for idx in range(m*p):
                        if random.random() < 0.3:
                            V_r[idx, c] = random.choice([0.5, 1.0, -0.5, -1.0])
                    for idx in range(p*n):
                        if random.random() < 0.3:
                            W_r[idx, c] = random.choice([0.5, 1.0, -0.5, -1.0])

                if verify_matmul(U_r, V_r, W_r):
                    best_rank = target_rank
                    best_solution = (U_r.copy(), V_r.copy(), W_r.copy())
                    break

            if best_rank == target_rank:
                break

    # ---- Return the best solution found ----
    U_best, V_best, W_best = best_solution

    # Convert to list, ensuring half-integer precision
    def to_list(arr):
        result = []
        for row in arr:
            new_row = []
            for x in row:
                # Round to nearest half-integer if close
                hx = round(x * 2) / 2.0
                if abs(x - hx) < 1e-9:
                    new_row.append(hx)
                else:
                    new_row.append(float(x))
            result.append(new_row)
        return result

    return {"U": to_list(U_best), "V": to_list(V_best), "W": to_list(W_best)}

# EVOLVE-BLOCK-END
