import numpy as np
import time
from itertools import product

def construct():
    n, m, p = 2, 4, 7

    # Schoolbook decomposition as fallback (rank 56)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1.0
                    V[j*p + k, idx] = 1.0
                    W[k*n + i, idx] = 1.0
                    idx += 1
        return U, V, W

    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def time_left():
        return time.time() < deadline

    # The tensor T_{i,j,k} = delta_{j,jp} for matrix multiplication
    # We need: sum_r U[i,j,r] * V[jp,k,r] * W[k,i,r] = delta_{j,jp}
    # 
    # Each rank-1 term r defines:
    #   A_support: set of (i,j) pairs where U[i,j,r] != 0
    #   B_support: set of (j,k) pairs where V[j,k,r] != 0  
    #   C_support: set of (k,i) pairs where W[k,i,r] != 0
    #
    # The product contributes to output monomial (i, j, jp, k) iff
    # (i,j) in A_support AND (j,k) in B_support AND (k,i) in C_support
    # with coefficient U[i,j,r]*V[j,k,r]*W[k,i,r]
    #
    # For exact decomposition, we need the sum over all terms to give
    # delta_{j,jp} for each (i,j,jp,k).
    #
    # Strategy: Start from Strassen block decomposition (rank 50), then
    # attempt to merge/replace terms using a greedy set-cover approach.

    # First, build the known Strassen-based decomposition (rank 50)
    def build_strassen_decomp():
        rank = 50
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        col = 0

        def add_strassen_block(i0, j0, k0):
            nonlocal col
            if not time_left():
                return

            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)

            e_idx = j0 * p + k0
            f_idx = j0 * p + (k0 + 1)
            g_idx = (j0 + 1) * p + k0
            h_idx = (j0 + 1) * p + (k0 + 1)

            c00_idx = k0 * n + i0
            c01_idx = (k0 + 1) * n + i0
            c10_idx = k0 * n + (i0 + 1)
            c11_idx = (k0 + 1) * n + (i0 + 1)

            # M1 = (a+d)(e+h)
            U[a_idx, col] += 1; U[d_idx, col] += 1
            V[e_idx, col] += 1; V[h_idx, col] += 1
            W[c00_idx, col] += 1; W[c11_idx, col] += 1
            col += 1

            # M2 = (c+d)e
            U[c_idx, col] += 1; U[d_idx, col] += 1
            V[e_idx, col] += 1
            W[c10_idx, col] += 1; W[c11_idx, col] -= 1
            col += 1

            # M3 = a(f-h)
            U[a_idx, col] += 1
            V[f_idx, col] += 1; V[h_idx, col] -= 1
            W[c01_idx, col] += 1; W[c11_idx, col] += 1
            col += 1

            # M4 = d(g-e)
            U[d_idx, col] += 1
            V[g_idx, col] += 1; V[e_idx, col] -= 1
            W[c00_idx, col] += 1; W[c10_idx, col] += 1
            col += 1

            # M5 = (a+b)h
            U[a_idx, col] += 1; U[b_idx, col] += 1
            V[h_idx, col] += 1
            W[c00_idx, col] -= 1; W[c01_idx, col] += 1
            col += 1

            # M6 = (c-a)(e+f)
            U[c_idx, col] += 1; U[a_idx, col] -= 1
            V[e_idx, col] += 1; V[f_idx, col] += 1
            W[c11_idx, col] += 1
            col += 1

            # M7 = (b-d)(g+h)
            U[b_idx, col] += 1; U[d_idx, col] -= 1
            V[g_idx, col] += 1; V[h_idx, col] += 1
            W[c00_idx, col] += 1
            col += 1

        def add_schoolbook_221_block(i0, j0, k0):
            nonlocal col
            if not time_left():
                return

            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)

            e_idx = j0 * p + k0
            g_idx = (j0 + 1) * p + k0

            c00_idx = k0 * n + i0
            c10_idx = k0 * n + (i0 + 1)

            U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1
            col += 1
            U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1
            col += 1
            U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1
            col += 1
            U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1
            col += 1

        for j_start in [0, 2]:
            for k_start in [0, 2, 4]:
                add_strassen_block(0, j_start, k_start)
            add_schoolbook_221_block(0, j_start, 6)

        return U[:n*m, :col], V[:m*p, :col], W[:p*n, :col]

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        if abs(s - (1.0 if j == jp else 0.0)) > tol:
                            return False
        return True

    # Build initial decomposition
    U0, V0, W0 = build_strassen_decomp()
    if U0.shape[1] != 50:
        # Incomplete due to timeout, use schoolbook
        U0, V0, W0 = schoolbook()

    best_U, best_V, best_W = U0, V0, W0
    best_rank = U0.shape[1]

    if not verify(U0, V0, W0):
        U0, V0, W0 = schoolbook()
        best_U, best_V, best_W = U0, V0, W0
        best_rank = U0.shape[1]

    # Now attempt improvement via greedy term merging
    # Idea: Try to replace pairs of schoolbook terms in the 2x2x1 blocks
    # with shared terms that also help adjacent Strassen blocks.
    #
    # More concretely: Look for rank-1 terms that can be "factored out" 
    # from multiple existing terms. If two terms share structure, we might
    # combine them into fewer terms.
    #
    # Alternative approach: Try different block partitions.
    # Instead of fixed 2x2x2 Strassen + 2x2x1 schoolbook, try:
    # - 2x3x2 Strassen-like blocks (if they exist with rank < 12)
    # - Overlapping blocks where a single term serves multiple sub-blocks
    
    # Try: Can we use a 2x4x2 block with rank < 14 (vs 2*7=14 for two 2x2x2)?
    # Or a 2x4x3 block with rank < 21 (vs 7+14=21)?
    #
    # For 2x4x2: We have A(2x4), B(4x2), C(2x2).
    # Schoolbook rank = 16. Two Strassen 2x2x2 = 14.
    # Can we do better than 14? Unlikely for this size.
    #
    # For 2x4x3: A(2x4), B(4x3), C(2x3).
    # Schoolbook rank = 24. One 2x2x2 + one 2x2x1 = 7+4=11 for first 2 cols,
    # then 2x2x1 for 3rd col = 4. Total = 15. Or two 2x2x2 + one 2x2x1 = 7+7+4=18.
    # Actually: split B into cols (0,1) and (2): 2x2x2 (rank 7) + 2x2x1 (rank 4) = 11.
    # Or split as (0) and (1,2): 2x2x1 (rank 4) + 2x2x2 (rank 7) = 11.
    # So 2x4x3 with rank 11 is already what we get from the block approach.
    
    # The real opportunity: Can terms from different 2x2x2 Strassen blocks
    # be shared? In the current approach, each Strassen block uses 7 independent
    # terms. If a term from block (j0=0,k0=0) could also contribute to block
    # (j0=0,k0=2), we'd save rank.
    #
    # For this, we need a term r where:
    #   U[i,j,r] nonzero for j in {0,1} (from first block) AND j in {2,3} (second)
    #   V[j,k,r] nonzero for k in {0,1} AND k in {2,3}
    #   W[k,i,r] nonzero for k in {0,1} AND k in {2,3}
    #
    # This is a "cross-block" term. Let's try to find such terms.
    
    # Greedy set-cover approach:
    # 1. Enumerate candidate rank-1 terms (with small integer/half-integer coefficients)
    # 2. For each candidate, determine which output monomials it "covers"
    # 3. Use greedy set cover to select minimal set of terms
    
    # Output monomials: (i, j, jp, k) where j==jp gives 1, else 0.
    # There are n*m*p = 56 "positive" monomials: (i, j, j, k) for all i,j,k.
    # And n*m*(m-1)*p = 2*4*3*7 = 168 "negative" monomials that must sum to 0.
    
    # A rank-1 term r with U_r, V_r, W_r contributes to monomial (i,j,jp,k):
    #   U_r[i*m+j] * V_r[jp*p+k] * W_r[k*n+i]
    #
    # For the term to be useful, it should have small support.
    
    # Let's try a different strategy: local search on the 2x2x1 blocks.
    # Replace the 4 schoolbook terms in each 2x2x1 block with potentially
    # fewer terms that also interact with neighboring Strassen blocks.
    
    # Actually, let's try a more direct approach: enumerate all possible
    # "patterns" for the 2x2x1 blocks and see if any pattern with rank < 4
    # exists that's compatible with the adjacent Strassen blocks.
    
    # For a 2x2x1 block (rows i=0,1; cols j=0,1; col k=0):
    # We need: sum_r U[0,j,r]*V[j,0,r]*W[0,0,r] = delta_{j,jp}
    #          sum_r U[1,j,r]*V[j,0,r]*W[1,0,r] = delta_{j,jp}
    # for j,jp in {0,1}.
    #
    # With rank 4 (schoolbook): U[0,0]=U[1,1]=1 on term 1,2; etc.
    # Can we do rank 3? We need 4 equations satisfied by 3 terms.
    # Each term contributes to at most 4 monomials (2x2 in the j-jp plane).
    # With 3 terms, max coverage is 12 monomial-contributions, but we need
    # exactly 4 ones and 4 zeros. This is tight but might work with cancellations.
    
    # Let's try: Can a 2x2x1 block be done in rank 3?
    # T = [[A, B], [C, D]] * [E; F] = [AE+BF, CE+DF]
    # We need: AE+BF = A, CE+DF = C (for the diagonal)
    #          AE+BF contributions to off-diagonal must cancel.
    
    # Actually for 2x2x1: output is 2x2 matrix C = A(2x2) * b(2x1)
    # C[0,0] = A[0,0]*b[0] + A[0,1]*b[1]
    # C[1,0] = A[1,0]*b[0] + A[1,1]*b[1]
    # This is just a matrix-vector product. Rank of this tensor...
    # It's the tensor for (2x2)*(2x1) -> (2x1). 
    # Schoolbook rank = 4. Can we do better?
    # This is a 2x2x1 tensor. The rank is at most 4 (schoolbook).
    # Actually, this tensor has rank exactly 4 because it's the Kronecker
    # product structure. Let me verify: T_{i,j,k} = delta_{j,k} for i in {0,1}, j,k in {0,1}.
    # Wait no. For multiplying A(2x2) by b(2x1):
    # C_i = sum_j A_{i,j} b_j
    # The tensor is T_{i,j,b} = delta_{j,b}... no.
    # T_{i, j, k} where i=output row, j=A column, k=b index.
    # T_{i,j,k} = 1 if j==k, else 0. (The i index just selects which row of A.)
    # So T = sum_i (e_i ⊗ delta) where delta is the 2x2 identity tensor.
    # Rank = 2 * rank(identity tensor) = 2 * 2 = 4. So rank 4 is minimal.
    
    # So the 2x2x1 blocks cannot be improved below rank 4.
    # The total rank is locked at 6*7 + 2*4 = 50 with this block structure.
    
    # To go below 50, we need a fundamentally different partition.
    # Options:
    # 1. Use overlapping blocks (terms shared between blocks)
    # 2. Use different block sizes (e.g., 2x3x2, 2x2x3)
    # 3. Use a non-block decomposition entirely
    
    # Let's try option 1: Can we share terms between adjacent 2x2x2 Strassen blocks?
    # Block 1: j in {0,1}, k in {0,1}
    # Block 2: j in {0,1}, k in {2,3}
    # A shared term would need V support in k={0,1,2,3} and W support in k={0,1,2,3}.
    # This is possible but the term would contribute to both blocks' outputs.
    # We'd need the contributions to be consistent.
    
    # Let's try a concrete approach: modify the Strassen decomposition
    # to allow cross-block terms.
    
    # For now, let's try a simple improvement: check if any of the 50 terms
    # can be eliminated (zero column) or if two terms can be merged.
    
    # Check for zero columns
    for c in range(U0.shape[1]):
        if np.all(U0[:, c] == 0) and np.all(V0[:, c] == 0) and np.all(W0[:, c] == 0):
            # Remove this column
            U0 = np.delete(U0, c, axis=1)
            V0 = np.delete(V0, c, axis=1)
            W0 = np.delete(W0, c, axis=1)
            c -= 1
            break  # Restart after deletion
    
    # Check for duplicate columns (same U, V, W up to sign)
    r = U0.shape[1]
    for c1 in range(r):
        if c1 >= U0.shape[1]:
            break
        for c2 in range(c1 + 1, U0.shape[1]):
            if np.array_equal(U0[:, c1], U0[:, c2]) and \
               np.array_equal(V0[:, c1], V0[:, c2]) and \
               np.array_equal(W0[:, c1], W0[:, c2]):
                # Duplicate - remove c2
                U0 = np.delete(U0, c2, axis=1)
                V0 = np.delete(V0, c2, axis=1)
                W0 = np.delete(W0, c2, axis=1)
                break
            elif np.array_equal(U0[:, c1], -U0[:, c2]) and \
                 np.array_equal(V0[:, c1], -V0[:, c2]) and \
                 np.array_equal(W0[:, c1], W0[:, c2]):
                # Opposite sign - remove c2
                U0 = np.delete(U0, c2, axis=1)
                V0 = np.delete(V0, c2, axis=1)
                W0 = np.delete(W0, c2, axis=1)
                break
            elif not time_left():
                break
        if not time_left():
            break
    
    if U0.shape[1] < best_rank and verify(U0, V0, W0):
        best_U, best_V, best_W = U0, V0, W0
        best_rank = U0.shape[1]
    
    # Try: Can we replace pairs of terms with a single term?
    # If U[:,c1]*V[:,c1]*W[:,c1] + U[:,c2]*V[:,c2]*W[:,c2] can be
    # expressed as a single rank-1 term, we save 1 rank.
    # This is unlikely for arbitrary terms, but let's check for structured cases.
    
    # For the Strassen terms, the products have specific structures.
    # Let's just verify and return the best we have.
    
    if not verify(U0, V0, W0):
        U0, V0, W0 = schoolbook()
        best_U, best_V, best_W = U0, V0, W0
        best_rank = U0.shape[1]
    
    # Ensure integer/half-integer entries
    # Check if entries are close to integers or half-integers
    def make_integer_or_half(X):
        X = X.copy()
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                v = X[i, j]
                # Round to nearest integer or half-integer
                candidates = [round(v), round(v - 0.5) + 0.5, round(v + 0.5) - 0.5]
                best_c = min(candidates, key=lambda c: abs(c - v))
                if abs(best_c - v) < 1e-6:
                    X[i, j] = best_c
        return X
    
    best_U = make_integer_or_half(best_U)
    best_V = make_integer_or_half(best_V)
    best_W = make_integer_or_half(best_W)
    
    # Final verification
    if not verify(best_U, best_V, best_W):
        U0, V0, W0 = schoolbook()
        best_U, best_V, best_W = U0, V0, W0
        best_rank = U0.shape[1]
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
