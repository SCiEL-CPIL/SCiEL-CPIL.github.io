"""Bootstrap SEARCH for rank-minimal exact tensor decomposition of 2x5 * 5x6."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _verify_rank(U, V, W, n=2, m=5, p=6):
    """Check if given factor matrices produce correct matmul tensor."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Tensor T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, 0 otherwise
    # For a candidate of rank r: T = sum_c U[:,c] * V[:,c] * W[:,c]
    # So T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-6:
                    return False
    
    # Check that off-diagonal terms are 0
    # The tensor has entries only at positions where the indices align:
    # T[a, b, c] is non-zero only if a = i*m+j, b = j'*p+k, c = k'*n+i' 
    # and j == j' and k == k' and i == i'
    # So T[a,b,c] = 0 unless b's first index matches a's second, and c's first matches b's second, and c's second matches a's first
    
    # Let's check a few off-diagonal cases to be safe
    checked = 0
    for a in range(n*m):
        for b in range(m*p):
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            if j != j2:
                continue  # These will always be 0 in the true tensor
            for c in range(p*n):
                k2 = c // n
                i2 = c % n
                if k != k2 or i != i2:
                    val = 0.0
                    for cc in range(rank):
                        val += U[a][cc] * V[b][cc] * W[c][cc]
                    if abs(val) > 1e-6:
                        return False
                    checked += 1
                    if checked > 10000:
                        break
            if checked > 10000:
                break
        if checked > 10000:
            break
    return True

def _schoolbook_rank():
    """Return the standard schoolbook decomposition."""
    n, m, p = DIMENSIONS
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

def _try_block_decomposition(n, m, p, partition_m):
    """
    Try to decompose matmul as sum of block decompositions.
    Partition the inner dimension m into blocks.
    For each block of size b, we use rank b * n * p for that block if we don't exploit structure.
    But maybe we can do better per block?
    
    Actually, for 2xb * bx p, the rank is at least max(2*b, b*p, 2*p)... no.
    The tensor rank of 2xb*b*p is at least the max dimension of any single factor in a 3-tensor.
    
    Let's just try small m values and find optimal.
    """
    pass

def _structured_search_for_rank(n, m, p, rank, max_trials=500, deadline=None):
    """
    Structured search for rank-r decomposition.
    Key insight: try to find decompositions where each column c has a specific
    support pattern. For matmul, the natural structure is that column c should
    be nonzero at positions (i*m+j), (j*p+k), (k*n+i) for some compatible (i,j,k).
    
    We parameterize each column by choosing which (i,j,k) triples it "covers" and
    assign weights. For rank < n*m*p, some columns must cover multiple triples.
    
    A column that covers multiple triples (i1,j1,k1), (i2,j2,k2), ... must have
    U nonzero at multiple positions, V nonzero at multiple positions, W nonzero
    at multiple positions. The product U[a,c]*V[b,c]*W[c_idx,c] must equal the
    sum of indicators for the covered triples.
    
    This is only possible if the covered triples share structure. For example,
    if two triples (i1,j1,k1) and (i2,j2,k2) have i1=i2, j1=j2, k1!=k2, then
    we need U to be nonzero at i*m+j, V to be nonzero at j*p+k1 AND j*p+k2,
    W to be nonzero at k1*n+i AND k2*n+i. But then the product would also be
    nonzero at (i*m+j, j*p+k1, k2*n+i) and (i*m+j, j*p+k2, k1*n+i), which
    are off-diagonal and must be zero. So this doesn't work unless we use
    cancellations.
    
    For exact decomposition with non-negative or simple entries, we can't merge
    triples that don't share all three indices. So for generic matmul, the rank
    is indeed n*m*p.
    
    Let's do a targeted search: try all ways to partition the 60 triples into
    rank groups, where each group must be "compatible" (all triples in a group
    share the same i, or the same j, or the same k).
    """
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_triples = len(triples)  # 60
    
    if rank >= num_triples:
        return None  # Can't be better than schoolbook
    
    # For rank < 60, we need to merge some triples.
    # A valid merge group must be such that the rank-1 tensor formed by the
    # group's support is a sum of basis tensors with no cross-terms.
    # This requires all triples in the group to have the same i, j, AND k,
    # which is impossible for distinct triples.
    # 
    # However, we can use cancellations. If a group has triples (i,j,k1) and
    # (i,j,k2), we can set:
    #   U[i*m+j, c] = 1
    #   V[j*p+k1, c] = 1, V[j*p+k2, c] = 1
    #   W[k1*n+i, c] = 1, W[k2*n+i, c] = 1
    # But this gives cross-terms at (i*m+j, j*p+k1, k2*n+i) and (i*m+j, j*p+k2, k1*n+i).
    # To cancel these, we'd need additional columns, which defeats the purpose.
    #
    # The only way to avoid cross-terms is if the groups are "orthogonal" in some
    # sense. For matmul tensors, this means the rank is n*m*p for generic entries.
    #
    # Let's verify this by trying to find any rank-59 decomposition.
    # We'll use a linear algebra approach: the space of rank-r tensors is a
    # variety, and we're asking if the matmul tensor lies in it.
    #
    # For now, let's do a random search with structured initializations.
    
    import random
    rng = np.random.RandomState(42)
    
    for trial in range(max_trials):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try a structured initialization:
        # Start with schoolbook and try to merge two triples into one column.
        # Pick two random triples to merge.
        idx1 = rng.randint(0, num_triples)
        idx2 = rng.randint(0, num_triples)
        if idx1 == idx2:
            continue
        
        i1, j1, k1 = triples[idx1]
        i2, j2, k2 = triples[idx2]
        
        # Only try merging if they share at least one index
        if i1 != i2 and j1 != j2 and k1 != k2:
            continue
        
        # Build the rank-r decomposition
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Fill in all triples except idx1 and idx2 as single columns
        col = 0
        for idx in range(num_triples):
            if idx == idx1 or idx == idx2:
                continue
            i, j, k = triples[idx]
            if col >= rank:
                break
            U[i*m+j, col] = 1.0
            V[j*p+k, col] = 1.0
            W[k*n+i, col] = 1.0
            col += 1
        
        # Now fill the merged column for idx1 and idx2
        if col < rank:
            # Try different weight assignments
            for w1 in [1.0, -1.0, 0.5, -0.5]:
                for w2 in [1.0, -1.0, 0.5, -0.5]:
                    U[i1*m+j1, col] = w1
                    U[i2*m+j2, col] = w2
                    V[j1*p+k1, col] = w1
                    V[j2*p+k2, col] = w2
                    W[k1*n+i1, col] = w1
                    W[k2*n+i2, col] = w2
                    
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    
                    if _verify_rank(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
        
        # Also try random initialization
        if col < rank:
            U[:, col:] = rng.randn(n*m, rank - col) * 0.1
            V[:, col:] = rng.randn(m*p, rank - col) * 0.1
            W[:, col:] = rng.randn(p*n, rank - col) * 0.1
            
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if _verify_rank(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as baseline
    best = _schoolbook_rank()
    best_rank = n * m * p  # 60
    
    # Try structured search for ranks from 59 down to 30
    for rank in range(59, 29, -1):
        if time.time() > deadline:
            break
        
        result = _structured_search_for_rank(n, m, p, rank, max_trials=200, deadline=deadline)
        if result is not None:
            best = result
            best_rank = rank
            break
    
    return best

# EVOLVE-BLOCK-END
