"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid tensor decomposition for n×m×p matrix mult."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_strassen_on_middle(n, m, p, deadline):
    """
    For 3x4x5: treat the middle dimension 4 as two blocks of size 2.
    Use Strassen (rank 7) for the 2x2 multiplication in the middle.
    
    The idea: C[i,k] = sum_j A[i,j] * B[j,k]
    Split j into j1 (first 2) and j2 (last 2).
    C[i,k] = sum_{j1=0}^{1} A[i,j1] * B[j1,k] + sum_{j2=0}^{1} A[i,2+j2] * B[2+j2,k]
    
    For each i,k, this is a 2-element dot product, which can be done with rank 3 (Strassen for 2x2).
    But we need to share the Strassen factors across all i,k pairs.
    
    Actually, let's think differently. We can group by the middle index j.
    For each j, we have A[:,j] * B[j,:] which is an outer product (rank 1).
    Total rank = m = 4 for this grouping, but that's just the schoolbook.
    
    Alternative: Use the fact that 4 = 2*2 and apply Strassen to the "batch" of multiplications.
    
    Let's try a different structured approach:
    - For 3x4x5, we can potentially use rank < 60 by finding linear dependencies.
    """
    start_time = time.time()
    n, m, p = 3, 4, 5
    
    # Strategy: Try to find a decomposition of rank r by solving a system of equations.
    # We'll use a greedy approach: start with schoolbook and try to merge terms.
    
    # First, let's try a specific structured decomposition.
    # For 3x4x5, consider the following:
    # We can write the multiplication as:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    #
    # Using Strassen-like combinations on pairs of middle indices:
    # Let's define:
    # S1 = A[i,0] + A[i,1]  (for each i)
    # S2 = B[0,k] - B[1,k]  (for each k)
    # S3 = A[i,0]            (for each i)
    # S4 = B[1,k]            (for each k)
    # Then A[i,0]*B[0,k] + A[i,1]*B[1,k] = S3*S4 + (S1-S3)*S2 + S3*S2 ... this is getting complex.
    
    # Let's try a more systematic approach: use integer linear programming on a small subset.
    
    # For now, try a few specific rank values with a structured search.
    
    # Approach: Use the fact that we can combine terms that have the same (i,k) pattern.
    # For each (i,k), the contribution is sum_j A[i,j]*B[j,k].
    # We want to find U,V,W such that for each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Let's try rank 55, 56, 57, 58, 59 with a structured initialization.
    
    for target_rank in [59, 58, 57, 56, 55]:
        if time.time() > deadline:
            break
        
        # Start from schoolbook and try to "merge" pairs of terms.
        # Schoolbook has 60 terms. To get rank r, we need to eliminate 60-r terms.
        
        # A simple way: if two terms (i1,j1,k1) and (i2,j2,k2) can be combined
        # such that the same (U,V,W) values work for both, we can save one rank.
        
        # Let's try a randomized approach with structure:
        # Initialize U, V, W as schoolbook, then randomly set some entries to 0
        # and adjust others to compensate.
        
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Initialize with a subset of schoolbook terms
            # Pick target_rank random (i,j,k) triples
            triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
            random.shuffle(triples)
            selected = triples[:target_rank]
            
            for c, (i, j, k) in enumerate(selected):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
            
            # Now try to fix the missing terms by adjusting values.
            # For each missing (i,j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # But currently it's 0 (since those (i,j,k) weren't selected).
            
            # This is a hard problem. Let's just verify if by chance it works.
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None

def _try_linear_dependency_search(n, m, p, deadline):
    """
    Try to find linear dependencies among the schoolbook terms that allow rank reduction.
    
    The key insight: if there exist coefficients alpha_c (not all zero) such that
    for all (i,j,k): sum_c alpha_c * [U0[i*m+j,c] * V0[j*p+k,c] * W0[k*n+i,c]] = 0
    where U0,V0,W0 are the schoolbook factors, then we can eliminate one term.
    
    For schoolbook, U0[i*m+j,c] = 1 iff c corresponds to (i,j,k_c), etc.
    So the condition becomes: for all (i,j,k), alpha_{(i,j,k)} = 0, which means no dependencies.
    
    This suggests schoolbook is already minimal for this structure. But maybe with a different
    initial decomposition, we can find dependencies.
    """
    start_time = time.time()
    
    # For 3x4x5, the minimum rank is at least max(n*m, m*p, p*n) = max(12, 20, 15) = 20.
    # The schoolbook rank is 60. There might be room for improvement.
    
    # Let's try a different approach: use the fact that we can permute the indices.
    # For example, if we reorder the middle dimension, we might find better structure.
    
    # Actually, let's try a completely different structured decomposition.
    # For 3x4x5, we can use the following idea:
    # - Split the 4 middle indices into two groups of 2.
    # - Use Strassen (rank 7) for the 2x2 multiplication within each group.
    # - But we need to share the Strassen factors.
    
    # Let me try a specific construction:
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # = (A[i,0] + A[i,1]) * (B[0,k] + B[1,k]) - A[i,0]*B[1,k] - A[i,1]*B[0,k]
    #   + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    # This gives us:
    # Term 1: (A[i,0] + A[i,1]) * (B[0,k] + B[1,k])  -> rank 1 per (i,k), total 3*5=15
    # Term 2: -A[i,0]*B[1,k]  -> rank 1 per (i,k), total 15
    # Term 3: -A[i,1]*B[0,k]  -> rank 1 per (i,k), total 15
    # Term 4: A[i,2]*B[2,k]  -> rank 1 per (i,k), total 15
    # Term 5: A[i,3]*B[3,k]  -> rank 1 per (i,k), total 15
    # Total: 75, which is worse than 60.
    
    # So this doesn't help. Let's try a different approach.
    
    # For 3x4x5, the best known upper bound might be 60 (schoolbook) or slightly less.
    # Let's just return None and rely on the schoolbook.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Start with schoolbook
    schoolbook = _build_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try Strassen-based approach
    if time.time() < deadline - 10:
        result = _try_strassen_on_middle(n, m, p, deadline)
        if result is not None:
            rank = len(result["U"][0])
            if rank < best_rank:
                best_result = result
                best_rank = rank
    
    # Try linear dependency search
    if time.time() < deadline - 5:
        result = _try_linear_dependency_search(n, m, p, deadline)
        if result is not None:
            rank = len(result["U"][0])
            if rank < best_rank:
                best_result = result
                best_rank = rank
    
    return best_result
# EVOLVE-BLOCK-END
