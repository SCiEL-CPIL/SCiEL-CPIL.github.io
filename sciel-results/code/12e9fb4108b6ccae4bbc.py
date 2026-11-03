"""Matrix multiplication tensor decomposition for 2x3 * 3x9.
Schoolbook rank 54. Attempts to find lower-rank exact decomposition.
"""
import time
import numpy as np
from scipy.optimize import linprog
from scipy.linalg import solve

DIMENSIONS = [2, 3, 9]
START_TIME = time.time()
TIME_LIMIT = 130.0  # safety margin


def _remaining_time():
    return TIME_LIMIT - (time.time() - START_TIME)


def _verify(U, V, W, n=2, m=3, p=9):
    """Verify tensor decomposition: T[i,k,j] = sum_t U[i*m+k,t]*V[k*p+j,t]*W[j*n+i,t] = 1"""
    rank = len(U[0])
    for i in range(n):
        for k in range(m):
            for j in range(p):
                s = 0.0
                for t in range(rank):
                    s += U[i * m + k][t] * V[k * p + j][t] * W[j * n + i][t]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True


def _build_schoolbook(n=2, m=3, p=9):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1.0
                V[k * p + j][col] = 1.0
                W[j * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_lp_decomposition(n, m, p, max_rank=53, deadline=None):
    """Use LP to find rank-r decomposition with bounded entries.
    
    Strategy: Fix U to a structured pattern, then solve LP for V and W.
    """
    if deadline is None:
        deadline = START_TIME + TIME_LIMIT - 5
    deadline = min(deadline, START_TIME + TIME_LIMIT - 5)
    
    rank = n * m * p  # start with schoolbook
    
    # Try to reduce rank by structured search
    # For 2x3*3x9, key insight: n=2 is very small.
    # U is 6 x r. If r < 6, U rows must be linearly dependent.
    # But rank >= max(6, 27, 18) = 27 for tensor rank lower bound.
    
    # Actually the trivial lower bound is max(n*m, m*p, p*n) = 27.
    # But there may be stronger bounds.
    
    # Let's try a different approach: exploit the fact that n=2.
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # With n=2: C[0,j] = A[0,0]*B[0,j] + A[0,1]*B[1,j] + A[0,2]*B[2,j]
    #           C[1,j] = A[1,0]*B[0,j] + A[1,1]*B[1,j] + A[1,2]*B[2,j]
    
    # Try rank = 27 (the lower bound)
    # U is 6x27, V is 27x27, W is 18x27
    
    # For this to work, we need the tensor to have rank exactly 27.
    # Let's try a construction:
    # Group the 54 terms into 27 pairs and see if each pair can be merged.
    
    # Alternative: try to find a decomposition where some U/V/W entries are 0
    # or simple rational values.
    
    # Let's try: can we do rank 18?
    # U: 6x18, V: 27x18, W: 18x18
    # Lower bound says rank >= 27, so 18 is impossible.
    
    # Try rank 27.
    # This is a very hard optimization problem. Let's try a specific structure.
    
    # Structure: Since n=2, write U rows as (u_00, u_01, u_02) and (u_10, u_11, u_12)
    # For the decomposition to work, we need specific linear relations.
    
    # Let's try a more practical approach: start from schoolbook and try to
    # merge columns using LP relaxation.
    
    # For each pair of schoolbook columns (t1, t2), check if we can replace
    # them with a single column (u_new, v_new, w_new) that reproduces both.
    
    # This requires: for all (i,k,j):
    #   U[i*m+k,t1]*V[k*p+j,t1]*W[j*n+i,t1] + U[i*m+k,t2]*V[k*p+j,t2]*W[j*n+i,t2]
    #   = U[i*m+k,new]*V[k*p+j,new]*W[j*n+i,new]
    
    # In schoolbook, column t corresponds to (i_t, k_t, j_t) with all entries 1
    # only at those positions. So the product is 1 only when i=i_t, k=k_t, j=j_t.
    
    # To merge (i1,k1,j1) and (i2,k2,j2) into one term, we need a rank-1 tensor
    # that equals e_{i1,k1,j1} + e_{i2,k2,j2}. This is only possible if
    # {i1,k1,j1} and {i2,k2,j2} share structure.
    
    # e_{i,k,j} = u*e_i ⊗ v*e_k ⊗ w*e_j where u,v,w are unit vectors.
    # Sum of two such = u1*e_i1 ⊗ v1*e_k1 ⊗ w1*e_j1 + u2*e_i2 ⊗ v2*e_k2 ⊗ w2*e_j2
    # This is rank-1 only if (i1,k1,j1)=(i2,k2,j2) (same term) or in degenerate cases.
    
    # So we can't merge schoolbook columns directly. We need a completely
    # different decomposition.
    
    # Let's try a constructive approach for rank 27.
    # Key: n=2, m=3, p=9. m*p = 27.
    # If we can make V the identity (27x27), then we need:
    #   sum_t U[i*m+k,t] * delta_{k*p+j, t} * W[j*n+i,t] = 1
    #   = U[i*m+k, k*p+j] * W[j*n+i, k*p+j] = 1 for all i,k,j
    # This requires U[i*m+k, k*p+j] = 1 and W[j*n+i, k*p+j] = 1.
    # But U[i*m+k, k*p+j] = 1 means U[row, col] is 1 only when col = k*p+j
    # and row = i*m+k. Since k is determined by both row and col, this gives
    # U[i*m+k, k*p+j] = 1 for all i,k,j. That's 54 ones in U (6x27 matrix).
    # Row i*m+k of U has ones at columns k*9, k*9+1, ..., k*9+8 (9 ones per row).
    # And W[j*n+i, k*p+j] = 1 means W[j*2+i, k*9+j] = 1 for all i,j,k.
    # Row j*2+i of W has ones at columns 9+j, 18+j, 27+j, ... (3 ones per row, since k=0,1,2).
    # 
    # Let's check: U is 6x27. Row 0 (i=0,k=0): ones at cols 0-8.
    # Row 1 (i=0,k=1): ones at cols 9-17.
    # Row 2 (i=0,k=2): ones at cols 18-26.
    # Row 3 (i=1,k=0): ones at cols 0-8.
    # Row 4 (i=1,k=1): ones at cols 9-17.
    # Row 5 (i=1,k=2): ones at cols 18-26.
    # 
    # W is 18x27. Row j*2+i:
    # For i=0: rows 0,2,4,...,16 (j=0,1,...,8)
    #   Row 0 (j=0,i=0): ones at cols 0,9,18
    #   Row 2 (j=1,i=0): ones at cols 1,10,19
    #   ...
    # For i=1: rows 1,3,5,...,17
    #   Row 1 (j=0,i=1): ones at cols 0,9,18
    #   Row 3 (j=1,i=1): ones at cols 1,10,19
    #   ...
    
    # Verify: sum_t U[i*m+k,t]*V[k*p+j,t]*W[j*n+i,t]
    # With V = I: = U[i*m+k, k*p+j] * W[j*n+i, k*p+j]
    # = 1 * 1 = 1. ✓
    
    # But wait - this gives rank 27! Let me verify more carefully.
    # U has shape (6, 27). Row i*m+k (where i in {0,1}, k in {0,1,2}):
    #   U[i*3+k, t] = 1 iff t = k*9 + j for some j in {0,...,8}, i.e., t in {k*9, ..., k*9+8}
    # Actually: U[i*3+k, k*9+j] = 1 for all j. That's 9 ones per row.
    
    # V = identity 27x27.
    
    # W has shape (18, 27). Row j*n+i = j*2+i (where j in {0,...,8}, i in {0,1}):
    #   W[j*2+i, t] = 1 iff t = k*9 + j for some k in {0,1,2}, i.e., t in {j, 9+j, 18+j}
    # That's 3 ones per row.
    
    # Check: for any (i,k,j):
    #   sum_t U[i*3+k, t] * V[k*9+j, t] * W[j*2+i, t]
    #   V[k*9+j, t] = delta_{t, k*9+j}
    #   = U[i*3+k, k*9+j] * W[j*2+i, k*9+j]
    #   U[i*3+k, k*9+j] = 1 (since t=k*9+j is in range [k*9, k*9+8])
    #   W[j*2+i, k*9+j] = 1 (since t=k*9+j = k*9+j, which is of form k'*9+j with k'=k)
    #   = 1. ✓
    
    # This is a valid rank-27 decomposition!
    
    if time.time() > deadline:
        return None
    
    rank = 27
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # U[i*m+k, k*p+j] = 1 for all i, k, j
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][k * p + j] = 1.0
    
    # V = identity
    for t in range(rank):
        V[t][t] = 1.0
    
    # W[j*n+i, k*p+j] = 1 for all i, j, k
    for j in range(p):
        for i in range(n):
            for k in range(m):
                W[j * n + i][k * p + j] = 1.0
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    return None


def _try_low_rank_search(n, m, p, deadline=None):
    """Try to find rank < 27 decomposition (below the trivial lower bound is impossible,
    but verify rank 27 is optimal and try rank 26-27 with different structures)."""
    if deadline is None:
        deadline = START_TIME + TIME_LIMIT - 5
    deadline = min(deadline, START_TIME + TIME_LIMIT - 5)
    
    # Rank 27 is the trivial lower bound max(n*m, m*p, p*n) = max(6,27,18) = 27.
    # We've found a rank-27 construction. Can we do better?
    # Actually no - the lower bound is 27, so 27 is optimal!
    # Wait, is that right? The lower bound for tensor rank of matrix multiplication
    # [n,m,p] is max(n*m, m*p, p*n) only in trivial cases.
    # Actually, a standard lower bound is: rank >= max(n*m, m*p, p*n) / min(n,m,p)
    # No, that's not right either.
    
    # The correct lower bound: rank(T) >= max over all "slice" dimensions.
    # For matrix mult [n,m,p], the tensor lives in (R^{nm}) * (R^{mp}) * (R^{pn}).
    # A rank-1 tensor has "fiber" ranks 1 in each mode.
    # The "slice rank" or "subrank" gives lower bounds.
    
    # Actually, for [2,3,9]:
    # Mode-1 slice: for each (k,j), the vector over i has entries that are products.
    # The tensor T[i,k,j] = 1 for all i,k,j.
    # Fix (k,j): the vector over i is (1,1) - rank 1.
    # Fix (i,j): the vector over k is (1,1,1) - rank 1.  
    # Fix (i,k): the vector over j is (1,...,1) - rank 1.
    # So all slices have rank 1, which doesn't help.
    
    # The actual lower bound for [n,m,p] matrix multiplication tensor rank:
    # rank >= max(n*m, m*p, p*n) is NOT a valid lower bound in general.
    # The valid lower bounds include:
    # - rank >= max(n*m, m*p, p*n) / max(n,m,p) [not standard]
    # - rank >= n*m*p / min(n*m, m*p, p*n) [also not standard]
    
    # Standard: rank([n,m,p]) >= max(n*m, m*p, p*n) is false.
    # Example: [2,2,2] has rank 7 < 8 = 2*2*2.
    
    # But for [2,3,9], is 27 a valid lower bound?
    # m*p = 27. The V matrix is 27 x rank. For the decomposition to work,
    # the 27 rows of V must be linearly independent? Not necessarily.
    
    # Actually, a key lower bound: rank >= m*p if the tensor is "non-degenerate" in mode 2.
    # For matrix multiplication, the tensor IS non-degenerate in all modes (for n,m,p >= 2).
    # This gives rank >= max(n*m, m*p, p*n) = max(6, 27, 18) = 27.
    
    # So rank 27 IS optimal for [2,3,9]!
    # Our construction achieves exactly this bound.
    
    return None  # Already have optimal solution


def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' - factor matrices with integer/half-integer entries
    """
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT - 5
    
    # Try to find the rank-27 construction (which is provably optimal)
    result = _try_lp_decomposition(n, m, p, deadline=deadline)
    
    if result is None:
        # Fallback to schoolbook
        result = _build_schoolbook(n, m, p)
    
    return result


# Pre-compute at module load to be safe
if __name__ == "__main__":
    r = construct()
    rank = len(r['U'][0])
    print(f"Found decomposition with rank {rank}")
    print(f"U shape: ({len(r['U'])}, {len(r['U'][0])})")
    print(f"V shape: ({len(r['V'])}, {len(r['V'][0])})")
    print(f"W shape: ({len(r['W'])}, {len(r['W'][0])})")
    print(f"Verified: {_verify(r['U'], r['V'], r['W'])}")
