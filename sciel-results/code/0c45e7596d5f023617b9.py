"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify exact tensor decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(total - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W

def _try_strassen_blocks(n, m, p):
    """
    Try to build a decomposition using Strassen 2x2 blocks where possible,
    plus schoolbook for remaining terms. Returns (U, V, W) or None.
    """
    # For 3x4x6, we can place 2x2 Strassen blocks in the (i,j) and (j,k) dimensions.
    # A is 3x4, B is 4x6, C is 3x6.
    # We can cover i=0,1 with 2 rows and j=0,1 with 2 cols for a 2x2 Strassen block on C[0:2, 0:2].
    # But B is 4x6, so k goes 0..5. We need blocks in k too.
    # Let's try: for each pair of i's, pair of j's, pair of k's, apply Strassen.
    # This is complex. Instead, let's use a simpler approach:
    # Try rank reduction via linear algebra on the schoolbook terms.
    return None

def _rank_reduce_by_elimination(U, V, W, n, m, p, max_iter=50):
    """
    Try to reduce rank by finding linear dependencies among the rank-1 terms.
    Each term c has vectors u_c (in R^{n*m}), v_c (in R^{m*p}), w_c (in R^{p*n}).
    The constraint is: for all (i,j,k), sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1.
    
    We can try: if two terms c1, c2 have proportional u or v vectors, we might merge.
    More generally, we can set up a system and try to find a lower-rank solution.
    
    Simplified approach: try to remove terms one by one and solve for the remaining coefficients.
    """
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    
    # Try removing terms from the end (last added = least "essential" in some sense)
    # Actually, all schoolbook terms are symmetric. Let's try a different approach.
    # We'll try to find if the tensor can be expressed with fewer terms by
    # solving a linear system.
    
    # The tensor has n*m*p = 72 equations (one per (i,j,k) triple).
    # Each rank-1 term (u, v, w) contributes to all 72 equations.
    # We need: for each (i,j,k), sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1
    
    # This is a system of 72 equations in the "products" u_c[i*m+j]*v_c[j*p+k]*w_c[k*n+i].
    # It's nonlinear in the individual u, v, w entries.
    
    # Alternative: fix U and V, solve for W linearly? No, it's still nonlinear.
    
    # Let's try a heuristic: randomly perturb and use gradient descent? Too slow.
    
    # Instead, let's try a specific structured reduction.
    # For 3x4x6, note that m=4 is even. We can split j into two groups of 2.
    # C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    # Each half is a 3x2 @ 2x6 multiplication.
    # Rank of MM_{3,2,6} schoolbook = 3*2*6 = 36. Two halves = 72.
    # Can we do better for MM_{3,2,6}? Lower bound = max(6,12,18) = 18.
    # This is getting complex. Let's just return the input unchanged if we can't find a reduction.
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    t_start = time.time()
    deadline = t_start + 130  # 20s safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U, V, W)
    
    # Verify schoolbook
    if not _verify(U, V, W, n, m, p):
        # Should not happen, but just in case
        pass
    
    # Try Strassen-based block construction
    if time.time() < deadline:
        # For 3x4x6, let's try a manual construction.
        # Split the 4 columns of A into two groups of 2: j0={0,1}, j1={2,3}
        # Split the 6 rows of B into three groups of 2: k0={0,1}, k1={2,3}, k2={4,5}
        # For each (i-pair, j-group, k-group), we have a 2x2 @ 2x2 subproblem (Strassen rank 7)
        # But i only has 3 values, so we can do i-pairs (0,1) and handle i=2 separately.
        
        # Let's try: use Strassen for the (i=0,1) x (j=0,1) x (k=0,1) block,
        # and similarly for other combinations, with schoolbook for the rest.
        
        # This is complex to implement correctly. Let's try a simpler approach:
        # Use the fact that we can sometimes share terms.
        
        # For now, try a random search with verification for small rank reductions.
        # We'll try ranks from best_rank-1 down to some lower bound.
        # But random search in high dimensions is unlikely to work.
        
        # Instead, let's try a specific algebraic construction.
        # The key insight: if we can find a set of rank-1 tensors that span the
        # multiplication tensor, we can use that.
        
        # For MM_{n,m,p}, one known approach is to use the "Laderman" or 
        # "Strassen" algorithms for specific sizes.
        
        # For 3x4x6, let's try to use 2x2 Strassen blocks:
        # We have 3 rows of A, 4 cols of A = 2 rows of B, 6 cols of B = 3 rows of C.
        
        # Construction:
        # For i in {0,1}, j in {0,1}, k in {0,1}: Strassen (7 terms)
        # For i in {0,1}, j in {0,1}, k in {2,3}: Strassen (7 terms)  
        # For i in {0,1}, j in {0,1}, k in {4,5}: Strassen (7 terms)
        # For i in {0,1}, j in {2,3}, k in {0,1}: Strassen (7 terms)
        # For i in {0,1}, j in {2,3}, k in {2,3}: Strassen (7 terms)
        # For i in {0,1}, j in {2,3}, k in {4,5}: Strassen (7 terms)
        # For i = 2, j in {0,1}, k in {0,1}: schoolbook (2*1*2=4 terms)
        # For i = 2, j in {0,1}, k in {2,3}: schoolbook (4 terms)
        # For i = 2, j in {0,1}, k in {4,5}: schoolbook (4 terms)
        # For i = 2, j in {2,3}, k in {0,1}: schoolbook (4 terms)
        # For i = 2, j in {2,3}, k in {2,3}: schoolbook (4 terms)
        # For i = 2, j in {2,3}, k in {4,5}: schoolbook (4 terms)
        
        # Total: 6*7 + 6*4 = 42 + 24 = 66 < 72. Let's verify this!
        
        def strassen_2x2(i0, j0, k0):
            """Return list of 7 (U_contrib, V_contrib, W_contrib) dicts for Strassen 2x2 block.
            U index: i*m+j, V index: j*p+k, W index: k*n+i
            """
            terms = []
            # M1 = (A[i0,j0]+A[i0+1,j0]) * (B[j0,k0]+B[j0+1,k0])
            # M2 = (A[i0,j0+1]+A[i0+1,j0+1]) * B[j0,k0]
            # M3 = A[i0,j0] * (B[j0+1,k0]-B[j0,k0])
            # M4 = A[i0+1,j0] * (B[j0+1,k0+1]-B[j0,k0])
            # M5 = (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
            # M6 = (A[i0+1,j0+1]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
            # M7 = (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0+1]-B[j0+1,k0])
            
            # C00 = M1 - M4 + M6 - M7
            # C01 = M2 + M4
            # C10 = M3 + M5
            # C11 = M1 - M3 + M5 - M6
            
            # Each M_r is one rank-1 term.
            
            # M1
            U1 = {(i0, j0): 1, (i0+1, j0): 1}
            V1 = {(j0, k0): 1, (j0+1, k0): 1}
            W1 = {(k0, i0): 1, (k0+1, i0+1): 1}
            terms.append((U1, V1, W1))
            
            # M2
            U2 = {(i0, j0+1): 1, (i0+1, j0+1): 1}
            V2 = {(j0, k0): 1}
            W2 = {(k0, i0+1): 1}
            terms.append((U2, V2, W2))
            
            # M3
            U3 = {(i0, j0): 1}
            V3 = {(j0+1, k0): 1, (j0, k0): -1}
            W3 = {(k0+1, i0): 1, (k0, i0+1): -1}
            terms.append((U3, V3, W3))
            
            # M4
            U4 = {(i0+1, j0): 1}
            V4 = {(j0+1, k0+1): 1, (j0, k0): -1}
            W4 = {(k0+1, i0): -1, (k0+1, i0+1): 1}
            terms.append((U4, V4, W4))
            
            # M5
            U5 = {(i0, j0): 1, (i0, j0+1): 1}
            V5 = {(j0+1, k0+1): 1}
            W5 = {(k0+1, i0): 1, (k0+1, i0+1): 1}
            terms.append((U5, V5, W5))
            
            # M6
            U6 = {(i0+1, j0+1): 1, (i0, j0): -1}
            V6 = {(j0, k0): 1, (j0, k0+1): 1}
            W6 = {(k0, i0): 1, (k0, i0+1): -1}
            terms.append((U6, V6, W6))
            
            # M7
            U7 = {(i0, j0+1): 1, (i0+1, j0+1): -1}
            V7 = {(j0+1, k0+1): 1, (j0+1, k0): -1}
            W7 = {(k0, i0): -1}
            terms.append((U7, V7, W7))
            
            return terms
        
        def schoolbook_1x2x2(i, j0, k0):
            """Schoolbook for 1x2 @ 2x2 = 4 terms."""
            terms = []
            for j in range(2):
                for k in range(2):
                    U_t = {(i, j0+j): 1}
                    V_t = {(j0+j, k0+k): 1}
                    W_t = {(k0+k, i): 1}
                    terms.append((U_t, V_t, W_t))
            return terms
        
        # Build the combined decomposition
        all_terms = []
        
        # Strassen blocks for i in {0,1}
        for i0 in [0]:  # Only one pair: (0,1)
            for j0 in [0, 2]:  # j groups: {0,1} and {2,3}
                for k0 in [0, 2, 4]:  # k groups: {0,1}, {2,3}, {4,5}
                    all_terms.extend(strassen_2x2(i0, j0, k0))
        
        # Schoolbook for i = 2
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                all_terms.extend(schoolbook_1x2x2(2, j0, k0))
        
        # Now assemble U, V, W from all_terms
        rank = len(all_terms)
        U_new = [[0.0] * rank for _ in range(n * m)]
        V_new = [[0.0] * rank for _ in range(m * p)]
        W_new = [[0.0] * rank for _ in range(p * n)]
        
        for c, (U_dict, V_dict, W_dict) in enumerate(all_terms):
            for (i, j), val in U_dict.items():
                U_new[i * m + j][c] = val
            for (j, k), val in V_dict.items():
                V_new[j * p + k][c] = val
            for (k, i), val in W_dict.items():
                W_new[k * n + i][c] = val
        
        # Verify
        if _verify(U_new, V_new, W_new, n, m, p):
            if rank < best_rank:
                best_rank = rank
                best_result = (U_new, V_new, W_new)
        
        if time.time() > deadline:
            break
    
    # Return the best result found
    U_best, V_best, W_best = best_result
    # Convert to lists of ints if all values are integers
    def to_clean_list(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                if abs(val - round(val)) < 1e-12:
                    new_row.append(int(round(val)))
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    return {"U": to_clean_list(U_best), "V": to_clean_list(V_best), "W": to_clean_list(W_best)}
# EVOLVE-BLOCK-END
