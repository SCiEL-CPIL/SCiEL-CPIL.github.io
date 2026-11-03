"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 2, 3, 4
    schoolbook_rank = n * m * p  # 24

    # Schoolbook solution as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = schoolbook_rank

    # Try ranks from schoolbook-1 down to 13
    for rank in range(schoolbook_rank - 1, 12, -1):
        if time.time() > deadline:
            break

        try:
            result = _try_rank_structured(n, m, p, rank, deadline)
            if result is not None:
                best_result = result
                best_rank = rank
                continue
        except Exception:
            continue

    return best_result

def _schoolbook(n, m, p):
    """Generate the trivial schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Reshape to 3D
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    # Compute S[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    S = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # Check if all entries are exactly 1
    max_err = np.max(np.abs(S - 1.0))
    return max_err < 1e-9

def _try_rank_structured(n, m, p, rank, deadline):
    """
    Try to find a rank decomposition with the given rank using structured search.
    
    Key insight: For 2x3x4 matrix multiplication, we can exploit the structure.
    Each rank-1 term corresponds to choosing:
    - A set of (i,j) indices for U (with values)
    - A set of (j,k) indices for V (with values)  
    - A set of (k,i) indices for W (with values)
    
    The product must sum to 1 for each (i,j,k) triple.
    
    We'll use a combinatorial approach with small integer/half-integer values.
    """
    
    # For small ranks, try structured patterns
    # The tensor is all-ones, so we need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    
    # Try using the fact that we can factor through shared indices
    # Strategy: try decompositions where each rank-1 term "covers" a subset
    
    # For rank r, we need r terms such that for each (i,j,k):
    # sum_{r'} U[i,j,r'] * V[j,k,r'] * W[k,i,r'] = 1
    
    # Let's try a different approach: use the structure of the problem
    # n=2 is small, so we can enumerate possibilities for W
    
    # W has shape (p, n, rank) = (4, 2, rank)
    # Each W[k,i,r] can be 0, 1, -1, 0.5, -0.5, etc.
    
    # For small ranks, do a focused search
    if rank <= 15:
        result = _focused_search(n, m, p, rank, deadline)
        if result is not None:
            return result
    
    return None

def _focused_search(n, m, p, rank, deadline):
    """
    Focused search for low-rank decompositions.
    
    Key structural insight: The multiplication tensor T[i,j,k]=1 can be written as
    sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    Since all entries are 1, we need the sum to be 1 for every (i,j,k).
    
    For small n=2, we can try to find patterns where:
    - U, V, W have sparse structure
    - Values are from {-1, 0, 1, 0.5, -0.5}
    """
    
    # Precompute all (i,j,k) triples
    triples = list(product(range(n), range(m), range(p)))
    
    # Try to build a decomposition incrementally
    # Start with the observation that if we can find r rank-1 tensors
    # whose product sums to 1 everywhere, we're done.
    
    # For this specific case, let's try a constructive approach:
    # The tensor is the "all-ones" tensor of shape (2,3,4).
    
    # One approach: use the fact that 1 = 1*1*1, and try to share terms.
    
    # Let's try: for each rank r, see if we can assign values such that
    # the constraint is satisfied.
    
    # Simplified: try to find U, V, W with entries in {-1, 0, 1} first
    # Then try {-1, -0.5, 0, 0.5, 1}
    
    value_sets = [
        [0, 1],
        [-1, 0, 1],
        [-1, -0.5, 0, 0.5, 1],
    ]
    
    for vals in value_sets:
        if time.time() > deadline:
            break
        
        result = _search_with_values(n, m, p, rank, vals, deadline)
        if result is not None:
            return result
    
    return None

def _search_with_values(n, m, p, rank, vals, deadline):
    """Search for a decomposition with entries from vals."""
    
    # This is still a huge search space, so use a smarter approach.
    # For the all-ones tensor, try to exploit symmetry.
    
    # Key idea: if we can find a "covering" where each (i,j,k) is hit
    # exactly once (or with appropriate weights), we're done.
    
    # For rank = n*m*p = 24, trivial solution works.
    # For lower rank, we need to "merge" some terms.
    
    # Let's try a constructive approach based on factorization:
    # 1 = a*b*c for various (a,b,c) in vals^3
    
    # For vals = {-1, 0, 1}: only (1,1,1) and (-1,-1,-1) work for product=1
    # For vals = {-1, -0.5, 0, 0.5, 1}: more options
    
    # Actually, let's try a different strategy: use the fact that the problem
    # is small enough for a more direct approach.
    
    # For n=2, m=3, p=4, the tensor has 24 entries.
    # We need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all 24 (i,j,k).
    
    # Let's try to find a solution where each "column" r of (U,V,W) 
    # contributes to a specific pattern.
    
    # Heuristic: try to make the decomposition "diagonal" in some sense.
    
    # For now, return None if we can't find a solution quickly.
    # The schoolbook solution at rank 24 is always available.
    
    # Try a few random structured initializations
    np.random.seed(rank * 7 + 42)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Create sparse random matrices with values from vals
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for r in range(rank):
            # Each column r: pick random non-zero entries
            for _ in range(np.random.randint(1, 4)):
                idx = np.random.randint(0, n * m)
                U[idx, r] = np.random.choice(vals)
            for _ in range(np.random.randint(1, 4)):
                idx = np.random.randint(0, m * p)
                V[idx, r] = np.random.choice(vals)
            for _ in range(np.random.randint(1, 4)):
                idx = np.random.randint(0, p * n)
                W[idx, r] = np.random.choice(vals)
        
        # Verify
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        S = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        
        if np.max(np.abs(S - 1.0)) < 1e-9:
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    return None

# EVOLVE-BLOCK-END
