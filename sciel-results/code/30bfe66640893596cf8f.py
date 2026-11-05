"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 7

    # Strategy: Use a more efficient decomposition
    # Split into 2x4x4 (first 4 columns of B) + 2x4x3 (last 3 columns of B)
    # 
    # For 2x4x4: We can use a rank-12 decomposition
    # The key insight: C[i,k] = sum_j A[i,j]*B[j,k] for k=0..3
    # We can group columns and use structured decompositions
    #
    # Actually, let me try a different structural approach:
    # Use the fact that for 2 rows, we can exploit the 2x2 structure differently
    #
    # Approach: Use "column pairing" for B
    # Pair B columns: (0,1), (2,3), (4,5), and handle 6 separately
    # For each pair (2j, 2j+1), compute both C[i,2j] and C[i,2j+1]
    # This is equivalent to computing A @ B_pair where B_pair is 4x2
    # 
    # A better structural idea:
    # The 2x4x7 tensor can be seen as 7 slices of 2x4
    # Each slice k: C[:,k] = A @ B[:,k]
    # The rank of the full tensor is at most the sum of ranks of slices
    # but we can share structure across slices
    #
    # Let me try: rank-1 terms with shared U vectors
    # For 2x4, A has 8 entries. If we use rank-1: U[i,j]*V[j,k]*W[k,i]
    # Each rank-1 term covers one (i,j,k) triple
    #
    # New structural approach: Use a "block" decomposition where
    # we group the 7 output columns into blocks and use optimized
    # sub-decompositions for each block size.
    #
    # Block sizes: 4 + 3 (or 2+2+2+1, or 3+2+2, etc.)
    #
    # For a 2x4xq block, the minimum rank is known for small q:
    # 2x4x1: rank 4 (each of 4 columns needs 1 term, but they share A)
    #   Actually: C[i,k] = sum_j A[i,j]*B[j,k]
    #   Rank-1 term: U[i,j]=1, V[j,k]=1, W[k,i]=1 for fixed (i,j,k)
    #   For fixed k, we need sum_j A[i,j]*B[j,k] for i=0,1
    #   This is 2 linear forms in A and B
    #   Minimum rank for 2x4x1: we need at least 4 terms (one per j)
    #   But we can do better: use 2 terms with U having both i values
    #   Term 1: U[0,j]=1 for j in S1, U[1,j]=1 for j in S2, V[j,k]=1 for j in S1∪S2
    #   This gets complicated. Let me just use the known optimal for small cases.
    #
    # Known results for small matrix multiplication ranks:
    # 2x2x2: rank 7 (Strassen)
    # 2x2x1: rank 3 (C00=A00*B00+A01*B10, etc. - actually rank 2 for single output)
    #   Wait, for 2x2x1: C0=A00*B00+A01*B10, C1=A10*B00+A11*B10
    #   This is 2 outputs, 4 inputs each. Rank is 3 (not 2) because...
    #   Actually rank(2x2x1) = 3 is known.
    #
    # Let me just try to construct a good decomposition directly.
    #
    # Approach: Build a rank-r decomposition by solving the system
    # T[i,j,k] = sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] = 1
    #
    # For small enough rank, we can use iterative methods.
    # But first, let me try a hand-optimized decomposition.
    #
    # Insight: The tensor T[i,j,k]=1 for all i,j,k is the "all-ones" tensor
    # in the matmul format. Its rank is exactly the rank of matrix multiplication
    # for these dimensions.
    #
    # Lower bound: rank >= max(n*m, m*p, p*n) / something... 
    # Actually rank >= n*m*p / (n*m*p) ... that's trivial.
    # Better: rank >= max(n*m, m*p, p*n) is NOT true.
    # The trivial rank is n*m*p = 56.
    #
    # Let me try a constructive approach with rank ~ 25-30:
    # Use the fact that we can "share" computation across k values.
    #
    # For each pair of output columns (k1, k2), we can compute both
    # C[:,k1] and C[:,k2] using shared intermediate products.
    #
    # Let me try: 3 pairs (0,1), (2,3), (4,5) + 1 single (6)
    # For each pair, use a 2x4x2 sub-decomposition
    # Rank of 2x4x2: at most 2*4 = 8 (trivial), but can be less
    # 
    # Actually, let me just try to find a good decomposition computationally.
    # Use a greedy/iterative approach.

    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0

    # Try to find a low-rank decomposition
    # Use an iterative approach: start with random init, then optimize
    
    best_rank = n * m * p  # trivial upper bound: 56
    best_U = None
    best_V = None
    best_W = None

    # First, establish a good baseline using Strassen-style blocks
    # 6 blocks of 2x2x2 (rank 7 each) + 8 rank-1 = 50
    rank_baseline = 50
    U_b = np.zeros((n*m, rank_baseline))
    V_b = np.zeros((m*p, rank_baseline))
    W_b = np.zeros((p*n, rank_baseline))
    
    col = 0
    def add_strassen(i0, j0, k0):
        nonlocal col
        U_b[i0*m + j0, col] += 1
        U_b[(i0+1)*m + (j0+1), col] += 1
        V_b[j0*p + k0, col] += 1
        V_b[(j0+1)*p + (k0+1), col] += 1
        W_b[k0*n + i0, col] += 1
        W_b[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        U_b[(i0+1)*m + j0, col] += 1
        U_b[(i0+1)*m + (j0+1), col] += 1
        V_b[j0*p + k0, col] += 1
        W_b[k0*n + (i0+1), col] += 1
        W_b[(k0+1)*n + (i0+1), col] -= 1
        col += 1
        
        U_b[i0*m + j0, col] += 1
        V_b[j0*p + (k0+1), col] += 1
        V_b[(j0+1)*p + (k0+1), col] -= 1
        W_b[(k0+1)*n + i0, col] += 1
        W_b[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        U_b[(i0+1)*m + (j0+1), col] += 1
        V_b[(j0+1)*p + k0, col] += 1
        V_b[j0*p + k0, col] -= 1
        W_b[k0*n + i0, col] += 1
        W_b[k0*n + (i0+1), col] += 1
        col += 1
        
        U_b[i0*m + j0, col] += 1
        U_b[i0*m + (j0+1), col] += 1
        V_b[(j0+1)*p + (k0+1), col] += 1
        W_b[k0*n + i0, col] -= 1
        W_b[(k0+1)*n + i0, col] += 1
        col += 1
        
        U_b[(i0+1)*m + j0, col] += 1
        U_b[i0*m + j0, col] -= 1
        V_b[j0*p + k0, col] += 1
        V_b[j0*p + (k0+1), col] += 1
        W_b[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        U_b[i0*m + (j0+1), col] += 1
        U_b[(i0+1)*m + (j0+1), col] -= 1
        V_b[(j0+1)*p + k0, col] += 1
        V_b[(j0+1)*p + (k0+1), col] += 1
        W_b[k0*n + i0, col] += 1
        col += 1

    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    
    for i in range(2):
        for j in range(4):
            U_b[i*m + j, col] = 1
            V_b[j*p + 6, col] = 1
            W_b[6*n + i, col] = 1
            col += 1

    # Verify baseline
    T_recon = np.einsum('ar,br,cr->abc', U_b, V_b, W_b)
    if np.allclose(T_recon, T, atol=1e-9):
        best_rank = rank_baseline
        best_U, best_V, best_W = U_b.copy(), V_b.copy(), W_b.copy()

    # Now try to find a better decomposition using optimization
    # Use a structured random initialization and gradient-free optimization
    
    if time.time() - start_time < DEADLINE - 30:
        # Try various ranks from low to high
        # Use a simple iterative refinement approach
        
        for target_rank in range(20, best_rank):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Try to find a rank-target_rank decomposition
            # Initialize with random values
            U_try = np.random.randn(n*m, target_rank) * 0.5
            V_try = np.random.randn(m*p, target_rank) * 0.5
            W_try = np.random.randn(p*n, target_rank) * 0.5
            
            # Iteratively refine: for each term, adjust to better match T
            # Use a simple coordinate descent approach
            converged = False
            for iteration in range(50):
                if time.time() - start_time > DEADLINE - 5:
                    break
                
                # Compute current reconstruction
                T_curr = np.einsum('ar,br,cr->abc', U_try, V_try, W_try)
                error = np.max(np.abs(T_curr - T))
                
                if error < 1e-8:
                    converged = True
                    break
                
                # Try to improve each term
                # For term r, we want to minimize ||T - sum_s U_s*V_s*W_s||
                # This is complex; use a simpler heuristic
                # 
                # Alternative: use the fact that if we fix two factors,
                # the third is determined linearly
                # 
                # For now, just check if this random init is close enough
                # (unlikely, but let's try a few random inits)
            
            if converged:
                best_rank = target_rank
                best_U, best_V, best_W = U_try, V_try, W_try
                break
    
    # Try a more systematic approach: use known structure
    # The key structural insight: for 2x4x7, we can use a "row-pair" approach
    # where we exploit that there are only 2 rows of A and C
    
    if time.time() - start_time < DEADLINE - 20 and best_rank > 30:
        # Try a different structural decomposition
        # Use the fact that C[i,k] = sum_j A[i,j]*B[j,k]
        # For i=0: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
        # For i=1: C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
        #
        # Define intermediate variables:
        # X_j[k] = B[j,k] for j=0..3, k=0..6  (these are the B entries)
        # Y_i[j] = A[i,j] for i=0..1, j=0..3  (these are the A entries)
        #
        # C[i,k] = sum_j Y_i[j] * X_j[k]
        #
        # This is a 2x4x7 bilinear form. We can use "sum-product" decompositions.
        #
        # Try: use 4 "column groups" of B, each with a 2x1xq sub-decomp
        # Group 1: j=0, B[0,0:7] -> contributes A[0,0]*B[0,k] and A[1,0]*B[0,k]
        #   This is rank 2 (one term for each i)
        #   Term: U[i,0]=1 for i=0,1; V[0,k]=1 for all k; W[k,i]=1
        #   Wait, that's 2 terms total for j=0 (one per i)
        #   Actually: for fixed j, we need C[i,k] += A[i,j]*B[j,k]
        #   Rank-1 term: U[i,j]=1, V[j,k]=1, W[k,i]=1 for fixed (i,j,k)
        #   For fixed j, varying i and k: we need 2*7 = 14 rank-1 terms (trivial)
        #   But we can do better: for fixed j, C[i,k] = A[i,j]*B[j,k]
        #   This is a rank-1 matrix in (i,k) space! So 1 term suffices per j.
        #   Term: U[i,j]=A[i,j]... but U must be constant, not depend on A.
        #
        # Right, U, V, W are fixed coefficients. The decomposition is:
        # sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = 1 for all i,j,k
        #
        # So each term t defines a "pattern" that contributes to multiple (i,j,k).
        #
        # For a single j: we need sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] = 1 for all i,k
        # If we set U[i,j,t] = u_i, V[j,k,t] = v_k, W[k,i,t] = w_ki
        # Then sum_t u_i * v_k * w_ki = 1 for all i,k
        # This is a matrix equation: sum_t u_i v_k w_ki = 1
        # The matrix M[i,k] = sum_t u_i v_k w_ki must be all-ones
        # M = (sum_t u_i w_ki) * v_k... no, that's not separable.
        # M[i,k] = sum_t u_i * v_k * w_ki
        # Let S[i,k] = sum_t u_i * w_ki. Then M[i,k] = S[i,k] * v_k
        # We need S[i,k] * v_k = 1 for all i,k, so S[i,k] = 1/v_k
        # S[i,k] = sum_t u_i * w_ki = 1/v_k
        # This means for each k, sum_t u_i * w_ki = 1/v_k
        # And u_i, w_ki are shared across k.
        #
        # This is getting complex. Let me just try a direct construction.
        #
        # Simple approach: use rank-1 terms but with sparsity
        # Each rank-1 term (i,j,k) has U[i,j]=1, V[j,k]=1, W[k,i]=1
        # Total: 2*4*7 = 56 terms
        #
        # Can we merge some? If two terms share the same (i,j) but different k,
        # they have the same U vector. We can't merge them directly.
        #
        # The Strassen approach merges by using sums/differences.
        #
        # Let me try: for each pair of k values (k1,k2), use a 2x2 Strassen-like
        # decomposition on the (j,k) plane for fixed i... but i is only 2 values.
        #
        # Actually, here's a cleaner structural idea:
        # Treat the problem as: for each k, compute C[:,k] = A @ B[:,k]
        # This is 7 separate 2x4x1 problems, each of rank at most 4 (one per j)
        # Total: 7*4 = 28 if we don't share anything
        # But we CAN share: the A matrix is the same for all k
        #
        # For 2x4x1: C[i,k] = sum_j A[i,j]*B[j,k]
        # Rank of this 2x4x1 tensor: we need sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] = 1
        # For fixed k (say k=0), this is a 2x4 tensor.
        # The minimum rank of 2x4x1 matmul is known to be 3 (I think)
        # Let me verify: C0 = A00*B0 + A01*B1 + A02*B2 + A03*B3
        #              C1 = A10*B0 + A11*B1 + A12*B2 + A13*B3
        # Can we do this with 2 terms? 
        # Term 1: U[i,j,1]*V[j,0,1]*W[0,i,1]
        # Term 2: U[i,j,2]*V[j,0,2]*W[0,i,2]
        # We need: sum_{t=1,2} U[i,j,t]*V[j,0,t]*W[0,i,t] = 1 for all i,j
        # This is a 2x4 matrix that must be all-ones, expressed as sum of 2 rank-1 matrices
        # (after absorbing V and W). A 2x4 all-ones matrix has rank 1, so 1 term suffices!
        # Wait: M[i,j] = sum_t U[i,j,t]*V[j,t]*W[i,t]
        # Let a_t[i] = U[i,j,t]... no, U depends on j too.
        # M[i,j] = sum_t U[i,j,t] * V[j,t] * W[i,t]
        # This is not simply a sum of rank-1 matrices in (i,j) because U[i,j,t] is not separable.
        # If we require U[i,j,t] = u_i * v_j (separable), then M = sum_t (u_i*v_j)*(V[j,t]*W[i,t])
        # = sum_t u_i * (v_j * V[j,t]) * W[i,t]... still not clean.
        #
        # I think the rank of 2x4x1 is actually 4 (one per j column).
        # Let me just use the baseline and try small improvements.

        # Return the best decomposition found
        if best_U is not None:
            U_list = best_U.tolist()
            V_list = best_V.tolist()
            W_list = best_W.tolist()
            
            # Round to nearest 0.5 for cleanliness (should already be integers/half-integers)
            U_list = [[round(x*2)/2 for x in row] for row in U_list]
            V_list = [[round(x*2)/2 for x in row] for row in V_list]
            W_list = [[round(x*2)/2 for x in row] for row in W_list]
        
        # Verify after rounding
        U_np = np.array(U_list, dtype=float)
        V_np = np.array(V_list, dtype=float)
        W_np = np.array(W_list, dtype=float)
        T_check = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        if not np.allclose(T_check, T, atol=1e-9):
            # Rounding broke it; use original
            U_list = best_U.tolist()
            V_list = best_V.tolist()
            W_list = best_W.tolist()
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Should not reach here, but fallback
        return {"U": U_b.tolist(), "V": V_b.tolist(), "W": W_b.tolist()}
# EVOLVE-BLOCK-END
