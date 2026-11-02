"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    Returns {'U', 'V', 'W'} where:
      U has shape (n*m, rank)
      V has shape (m*p, rank)
      W has shape (p*n, rank)
    The decomposition satisfies: for all i,j,k, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    # Build a rank-23 decomposition for 3x3x3 using Strassen's method
    # Strassen's 3x3x3: 7 terms for 2x2x2, extended to 3x3x3
    # We use a known 3x3x3 rank-23 decomposition
    def build_3x3x3_strassen():
        """Return U,V,W for 3x3x3 with rank 23."""
        r = 23
        U = np.zeros((9, r))
        V = np.zeros((9, r))
        W = np.zeros((9, r))
        
        # Strassen's algorithm for 3x3x3: use 7-term 2x2x2 twice plus corrections
        # Actually, let's use a simpler approach: Kronecker product structure
        # 3x3x3 = (2+1)x(2+1)x(2+1)
        # Use rank-7 for the 2x2x2 part, and handle the +1 dimensions
        
        # Better: use the fact that 3x3x3 can be done with rank 23 via
        # 2x2x3 (rank 10) + 1x3x3 (rank 9) + 3x1x3 (rank 9) - overlaps
        # Let me just use a direct construction
        
        # Direct construction using polynomial interpolation / Reed-Muller
        # For 3x3x3, we can use the following:
        # The tensor T = sum_{i,j,k} e_i^U e_j^V e_k^W * delta_{i,j,k} (no, that's wrong)
        # T_{i,j,k} = delta_{i,k} for matrix mult: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
        
        # So we need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta_{i,k}... no
        # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        # where the product gives (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
        
        # So the constraint is: for each (i,j,k), the sum equals 1
        # This means the decomposition must satisfy the matrix multiplication identity
        
        # Let's use a known rank-23 for 3x3x3 from the literature
        # One approach: use 3 copies of 2x2x3 Strassen-like
        
        # Simpler: just use schoolbook for 3x3x3 (rank 27) and then try to reduce
        # But we want rank 23. Let me construct it explicitly.
        
        # Actually, for the purpose of this task, let me just build a valid decomposition
        # and then try to minimize rank. Start with a known valid one.
        
        # Use the "block" construction:
        # Split indices: i = i2 + i1 where i2 in {0,1}, i1 in {0,1,2}
        # This gets complicated. Let me just use a simpler valid construction.
        
        # For now, use rank-27 schoolbook for 3x3x3, then combine for 3x3x7
        # and then try to reduce the total rank.
        
        # Actually, let me just build the full 3x3x7 directly with a structured approach.
        return None  # placeholder
    
    # Build 3x3x7 directly:
    # For each k in 0..6, we need the j-th row of A times the k-th column of B
    # (A*B)_{i,k} = sum_{j=0}^{2} A_{i,j} * B_{j,k}
    
    # Simple construction: for each (i,k), use m=3 terms
    # Total rank = n*p*m = 3*7*3 = 63 (schoolbook)
    
    # Better: group by k. For each k, we have a 3x3 matrix mult (A @ B[:,k])
    # which is just a linear map from R^3 to R^3.
    # The tensor for fixed k is: T_k[i,j] = A_{i,j} * B_{j,k}
    # This is an outer product of the i-th row of A and the k-th column of B... no.
    # T_k is a 3x3 matrix where T_k[i,j] contributes to (AB)_{i,k}.
    # Actually (AB)_{i,k} = sum_j A_{i,j} B_{j,k}, so for fixed k, we have a bilinear form in (A_row_i, B_col_k).
    
    # The key insight: for fixed (i,k), (AB)_{i,k} = dot(A_row_i, B_col_k)
    # This is a rank-3 bilinear form (one term per j).
    # So naively we need 3 terms per (i,k) pair, giving 3*7*3 = 63.
    
    # To reduce: share terms across different (i,k) pairs.
    # If we use the same "j-index" structure for multiple (i,k) pairs, we can share.
    
    # Let me try a structured construction with rank < 63.
    
    def build_structured(r):
        """Try to build a rank-r decomposition with structure."""
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        
        # For each column k, we need to compute A @ B[:,k]
        # Let's assign r_k terms to column k, summing to r
        # For simplicity, distribute evenly or use a pattern
        
        # Try: for each k, use 3 terms (one per j), but share across k's
        # If we use the same U and W patterns for different k's, we save terms.
        
        # Construction: for c in 0..r-1, define which (i,j,k) it covers
        # Let's use a cyclic pattern
        
        for c in range(r):
            # Each term c covers a specific combination
            # Use: i = c % n, j = (c // n) % m, k = (c // (n*m)) % p
            # But this only gives n*m*p distinct terms
        
            # Better: allow each term to contribute to multiple (i,j,k)
            # Set U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] for a subset
        
            # For now, let's try: each term c is associated with (j, k_mod)
            # where it contributes to all i
            j = c % m
            k = c // m % p
            # This gives m*p = 21 distinct (j,k) pairs
            # For each, we can have multiple i's
            
            # Actually, let me try a different structure:
            # Group by j: for each j, we have m*p = 3*7 = 21 (j,k) entries in V
            # and n*p = 3*7 = 21 (k,i) entries in W, and n*m = 9 (i,j) in U
            
            # Let's just try random but with the constraint that entries are in {-1, 0, 1}
            # and use a greedy approach to satisfy the constraints
        
        return U, V, W
    
    # Start with schoolbook as best known
    best_U, best_V, best_W = schoolbook()
    best_rank = best_U.shape[1]
    
    # Try to find better decompositions using structured search
    # Key: use the fact that we can share terms across different k values
    
    def verify(U, V, W, tol=1e-9):
        """Check if the decomposition is valid."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Strategy: build from Strassen-like blocks
    # For 3x3x3, use rank 23 (known optimal is 23)
    # For the remaining 3x3x4, use rank 12 (schoolbook for 3x3x4 is 36, but we can do better)
    
    # Actually, let me try a direct optimization approach:
    # Start with a valid decomposition and try to reduce rank by setting some columns to zero
    # and adjusting others to compensate
    
    # Or: use a known 3x3x3 rank-23 and extend
    
    # Let me try: use 2x2x3 Strassen (rank 10) for the first 2 rows/cols, 
    # and handle the 3rd row/col separately
    
    # For now, let's try a systematic search for small improvements over schoolbook
    
    # Try rank 54, 51, 48, etc. by removing columns from schoolbook and checking
    # if the remaining columns still span (this is a heuristic)
    
    # Better approach: use the structure that (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # If we fix j, we have a 3x7 outer product structure
    
    # Let me try: for each j, use 7 terms (one per k), but share U across j's
    # Term (j, k): U[i*m+j, c] = delta_{i, i0} for some i0, V[j*p+k, c] = 1, W[k*n+i, c] = delta_{i, i0}
    # This doesn't help much.
    
    # I'll try a different tactic: use numpy to solve for a rank-r decomposition
    # by treating it as a tensor completion problem
    
    # For now, return schoolbook if we can't do better
    # But let's try at least one structured construction
    
    # Construction with rank = n * p * ceil(m/2) = 3 * 7 * 2 = 42
    # Idea: pair up j indices. For j=0,1 use shared terms, j=2 separate
    # This is getting too complex without a clear algorithm.
    
    # Let me just try to verify that schoolbook works and return it
    # The random search in the original never found anything better anyway
    
    # Try a few specific ranks with a smarter search
    for target_rank in [42, 39, 36, 33, 30, 27, 24, 21]:
        if target_rank >= best_rank:
            continue
        if time.time() > deadline - 10:
            break
        
        found = False
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            
            # Initialize with a structured pattern
            U = np.zeros((nm, target_rank))
            V = np.zeros((mp, target_rank))
            W = np.zeros((pn, target_rank))
            
            # Assign each term to a (j, k) pair
            # target_rank terms, m*p = 21 (j,k) pairs
            # Each (j,k) gets target_rank // 21 terms, remainder distributed
            base = target_rank // (m * p)
            rem = target_rank % (m * p)
            
            c = 0
            for j in range(m):
                for k in range(p):
                    num_terms = base + (1 if (j * p + k) < rem else 0)
                    for t in range(num_terms):
                        if c >= target_rank:
                            break
                        # For this (j,k) term, assign i values
                        # Use a pattern: term t for (j,k) covers i = (t + j) % n
                        i = (t + j) % n
                        U[i * m + j, c] = 1.0
                        V[j * p + k, c] = 1.0
                        W[k * n + i, c] = 1.0
                        c += 1
            
            if verify(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    # Convert to lists
    U_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else (2*x if abs(2*x - round(2*x)) < 1e-9 else x) for x in row] for row in best_U]
    V_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else (2*x if abs(2*x - round(2*x)) < 1e-9 else x) for x in row] for row in best_V]
    W_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else (2*x if abs(2*x - round(2*x)) < 1e-9 else x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
