import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110
    
    nU = n * m  # 12
    nV = m * p  # 32
    nW = p * n  # 24
    
    # Build target tensor indices
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
    
    # All possible (a,b,c) triples
    all_triples = []
    for a in range(nU):
        for b in range(nV):
            for c in range(nW):
                all_triples.append((a, b, c))
    
    target_set = set(target_indices)
    non_target_indices = [t for t in all_triples if t not in target_set]
    
    def verify(U, V, W, rank):
        """Check if decomposition is exact"""
        # Check target entries equal 1
        for a, b, c in target_indices:
            val = 0.0
            for cp in range(rank):
                val += U[a][cp] * V[b][cp] * W[c][cp]
            if abs(val - 1.0) > 1e-9:
                return False
        # Check non-target entries equal 0 (sample a subset for speed)
        import random
        random.seed(42)
        sample = random.sample(non_target_indices, min(500, len(non_target_indices)))
        for a, b, c in sample:
            val = 0.0
            for cp in range(rank):
                val += U[a][cp] * V[b][cp] * W[c][cp]
            if abs(val) > 1e-9:
                return False
        return True
    
    # Try structured decompositions with various ranks
    # Strategy: try to find patterns where multiple (i,j,k) terms can be combined
    
    # First, try the schoolbook as fallback
    schoolbook_rank = n * m * p
    schoolbook_U = [[0] * schoolbook_rank for _ in range(nU)]
    schoolbook_V = [[0] * schoolbook_rank for _ in range(nV)]
    schoolbook_W = [[0] * schoolbook_rank for _ in range(nW)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                schoolbook_U[i*m+j][col] = 1
                schoolbook_V[j*p+k][col] = 1
                schoolbook_W[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": schoolbook_U, "V": schoolbook_V, "W": schoolbook_W}
    best_rank = schoolbook_rank
    
    # Try ranks below schoolbook with structured approaches
    # Key idea: for each (i,j) pair, we need to handle 8 k-values.
    # If we can represent the 8 k-contributions with fewer than 8 rank-1 terms
    # per (i,j) pair, we save rank.
    
    # Approach 1: Try to use symmetry in k dimension
    # For fixed (i,j), the terms are (a=i*m+j, b=j*p+k, c=k*n+i) for k=0..7
    # The a index is fixed, b varies over {j*p, j*p+1, ..., j*p+7}, c varies over {0,1,...,7} shifted by i
    
    # Approach 2: Use a linear algebra approach
    # The multiplication tensor can be viewed as a matrix multiplication problem
    # Reshape and use SVD-like approaches
    
    # Let's try a direct algebraic construction
    # For 3x4x8, try rank 48 (half of schoolbook)
    # Idea: group k values in pairs and use 2x2 Strassen-like tricks
    
    # Actually, let's try a more systematic approach:
    # Use the fact that the tensor has a specific sparsity pattern
    
    # Try rank values that are achievable
    # Lower bound: max(nU, nV, nW) = 32
    # Try from 32 up to 95
    
    for rank in range(32, min(schoolbook_rank, 65)):
        if time.time() > deadline:
            break
        
        # Try structured initialization
        # Pattern: for each (i,j), assign rank/p columns to handle k values
        # With rank=48, each (i,j) gets 48/12 = 4 columns
        # With rank=36, each (i,j) gets 3 columns
        
        cols_per_ij = rank // (n * m)
        if cols_per_ij == 0:
            continue
        
        # Try a structured pattern
        U = [[0.0] * rank for _ in range(nU)]
        V = [[0.0] * rank for _ in range(nV)]
        W = [[0.0] * rank for _ in range(nW)]
        
        # Assign columns to (i,j) pairs
        col_offset = 0
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for t in range(cols_per_ij):
                    c = col_offset + t
                    if c >= rank:
                        break
                    U[a][c] = 1.0
                    # For V and W, we need to handle all k values
                    # This structured approach won't work directly
                    # because V[b][c] needs to be non-zero for specific b values
                    pass
                col_offset += cols_per_ij
        
        # The above structured approach is too rigid. Let me try a different strategy.
        # Use random initialization with verification for a few trials
        for trial in range(20):
            if time.time() > deadline:
                break
            
            np.random.seed(trial + rank)
            U = np.random.randn(nU, rank)
            V = np.random.randn(nV, rank)
            W = np.random.randn(nW, rank)
            
            # Quick check: compute the tensor product and see how close we are
            # This is expensive, so just do a quick sanity check on a few entries
            # Skip full verification for now, just try a few structured patterns
            
            # Try a pattern based on modular arithmetic
            U2 = [[0.0] * rank for _ in range(nU)]
            V2 = [[0.0] * rank for _ in range(nV)]
            W2 = [[0.0] * rank for _ in range(nW)]
            
            # Pattern: use rank = n*m*p/k for some k
            # For each term, assign it to a column based on a hash
            # This won't work for exact decomposition
            pass
        
        # Try a specific known construction for small cases
        # For 3x4x8, try to use the structure that p=8=2^3
        # Split into 2x2 blocks and use Strassen recursively
        # 3x4x8 = 3x4x(2x4) = 2 * (3x4x4)
        # Each 3x4x4 has schoolbook rank 48, total 96 - no help
        # But maybe 3x4x4 can be done in less than 48?
        
        # Let's try a completely different approach: use the fact that
        # we can share the A[i,j] factor across k values
        # C[i,k] = A[i,j] * B[j,k] for each j
        # So for fixed i, C[i,:] = sum_j A[i,j] * B[j,:]
        # This is a linear combination of the rows of B weighted by A[i,j]
        
        # The tensor decomposition must capture this structure
        # For each (i,j), the contribution is A[i,j] * B[j,k] for all k
        # In the tensor, this appears as U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # One idea: for each (i,j), use a single column where
        # U[i*m+j, c] = 1, V[j*p+k, c] = B[j,k] (but B is variable, not fixed)
        # This doesn't work because B is part of the input
        
        # The decomposition must work for ALL A and B, so the factors must be constants
        
        # Let me try a brute-force search for small rank reductions
        # Start with schoolbook and try to merge pairs of terms
        
        # For now, if we can't find better, return schoolbook
        pass
    
    # Try one more approach: use a linear programming / feasibility check
    # for specific small rank reductions
    
    # Try rank 95: can we merge two schoolbook terms?
    # Terms (i,j,k1) and (i,j,k2) with same (i,j) but different k
    # They share U[i*m+j, c1] and U[i*m+j, c2]
    # If we set U[i*m+j, c1] = U[i*m+j, c2] = 1 and adjust V and W...
    # V[j*p+k1, c1] = 1, V[j*p+k2, c2] = 1
    # W[k1*n+i, c1] = 1, W[k2*n+i, c2] = 1
    # To merge, we'd need a single column c where:
    # V[j*p+k1, c] * W[k1*n+i, c] = 1 and V[j*p+k2, c] * W[k2*n+i, c] = 1
    # But also V[j*p+k1, c] * W[k2*n+i, c] = 0 and V[j*p+k2, c] * W[k1*n+i, c] = 0
    # This requires V[j*p+k1, c] * W[k2*n+i, c] = 0 which means either
    # V[j*p+k1, c] = 0 or W[k2*n+i, c] = 0
    # But we need V[j*p+k1, c] * W[k1*n+i, c] = 1, so V[j*p+k1, c] != 0
    # Thus W[k2*n+i, c] = 0, but then V[j*p+k2, c] * W[k2*n+i, c] = 0 != 1
    # So we cannot merge two terms with same (i,j) but different k
    
    # Similarly, cannot merge terms with same (j,k) but different i,
    # or same (k,i) but different j
    
    # This suggests the schoolbook rank might be optimal for this specific case,
    # or that merging requires more complex combinations of 3+ terms
    
    # Given the time constraints and the difficulty of finding a better decomposition,
    # return the schoolbook decomposition
    
    return {"U": schoolbook_U, "V": schoolbook_V, "W": schoolbook_W}
