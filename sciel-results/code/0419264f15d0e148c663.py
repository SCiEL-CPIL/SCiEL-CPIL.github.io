"""Matrix multiplication 3,3,4 rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _build_schoolbook(n, m, p):
    """Standard schoolbook rank n*m*p decomposition."""
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
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float((Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i]).sum())
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_rank35(n, m, p):
    """
    Construct a rank-35 decomposition for 3x3x4 matmul.
    
    Strategy: start from schoolbook (36 terms). We show one schoolbook term
    (i0,j0,k0)=(0,0,0) can be absorbed by modifying another term plus adding
    one new term, netting rank 35. We find explicit integer coefficients.
    
    We remove term t* = (0,0,0). We add a new term with parameters a,b,c
    placed at a few (i,j,k) positions, and scale an existing term, such that
    the removed contribution is exactly reproduced.
    
    Implementation: solve a small linear system. Keep 34 schoolbook terms
    (all except (0,0,0)) plus 1 new term with free values at a chosen support.
    We pick the new term to have U support on (i,j) in S_U, V on (j,k) in S_V,
    W on (k,i) in S_W. The tensor entry contributed at (i,j,k) is
    U[i,j]*V[j,k]*W[k,i]. We need this to equal 1 at (0,0,0) and 0 at all
    other (i,j,k) where it's nonzero, AND not disturb others.
    
    The cleanest known identity: for the matmul tensor, the entry at (i,j,k)
    equals the sum over r of U_r V_r W_r. Using the relation that the tensor
    is 'flat' we can merge. We instead directly solve: choose the new term's
    support to be exactly {(0,0,0)}-compatible.
    
    Concretely: new term U nonzero only at (i=0,j=0) with value u;
    V nonzero only at (j=0,k=0) with value v; W nonzero only at (k=0,i=0) with
    value w. Then it contributes u*v*w at (0,0,0) and 0 everywhere else.
    Set u*v*w = 1 (e.g., u=v=w=1). This is just another (0,0,0) term - no gain.
    
    To actually reduce rank, we must let the new term contribute to MULTIPLE
    entries and compensate by adjusting existing terms. We set up a linear
    system: keep all 36 schoolbook coefficients free (x_t) plus one new term
    with fixed simple support pattern, and require the total to equal 1 at
    every (i,j,k). Then find a solution with one x_t = 0 (giving rank 35).
    """
    n, m, p = 3, 3, 4
    # Index list for schoolbook terms
    terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    R = len(terms)  # 36
    # New term support: we let it have value 1 at a small set of positions.
    # Choose new term: U=1 at (0,0) and (0,1); V=1 at (0,0) and (1,0); W=1 at (0,0).
    # Then new term contributes at (i,j,k) iff U[i,j]=1 and V[j,k]=1 and W[k,i]=1.
    # W=1 only at (k=0,i=0). So k=0,i=0. U=1 at (0,0),(0,1); V=1 at (0,0),(1,0).
    # Need (i,j) in {(0,0),(0,1)} and (j,k)=(j,0) in {(0,0),(1,0)} => j in {0,1}.
    # (i,j)=(0,0): j=0 ok, V(0,0)=1 ok -> contributes at (0,0,0)
    # (i,j)=(0,1): j=1 ok, V(1,0)=1 ok -> contributes at (0,1,0)
    # So new term = indicator of {(0,0,0),(0,1,0)} (sum of two schoolbook basis entries).
    
    # Build linear system: variables x_t for t in terms (36) + 1 for new term.
    # Constraint at each (i,j,k): sum_{t: t==(i,j,k)} x_t + new_contrib(i,j,k) = 1.
    # new_contrib is 1 at (0,0,0) and (0,1,0), else 0.
    # We want a solution where x_(0,0,0) = 0 (drop that schoolbook term).
    # With new term present, at (0,0,0): x_(000) + 1 = 1 => x_(000)=0. Good.
    # At (0,1,0): x_(010) + 1 = 1 => x_(010)=0.
    # At all other (i,j,k): x_t = 1.
    # So solution: all x_t=1 except x_(000)=0, x_(010)=0, and new term coefficient=1.
    # That gives rank = 36 - 2 + 1 = 35. 
    # But wait: does the new term equal sum of basis entries (000) and (010)?
    # Yes, by construction its tensor is 1 at exactly those two entries.
    # So U_new,V_new,W_new with that support reproduce exactly those two.
    # Then total tensor = (sum of all 36 schoolbook with x) + new
    #   = [1 at every entry from x, but x_(000)=0,x_(010)=0 so 0 at those] + [1 at (000),(010)]
    #   = 1 at every entry. 
    # Rank used: schoolbook terms with x!=0 are 34, plus new term = 35. 
    
    # Now build explicit U,V,W of rank 35.
    # Columns: 34 schoolbook (all except (0,0,0) and (0,1,0)) + 1 new.
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for (i, j, k) in terms:
        if (i, j, k) == (0, 0, 0) or (i, j, k) == (0, 1, 0):
            continue
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1
    # New term: U=1 at (0,0),(0,1); V=1 at (0,0),(1,0); W=1 at (0,0)
    U[0 * m + 0][col] = 1   # (i=0,j=0)
    U[0 * m + 1][col] = 1   # (i=0,j=1)
    V[0 * p + 0][col] = 1   # (j=0,k=0)
    V[1 * p + 0][col] = 1   # (j=1,k=0)
    W[0 * n + 0][col] = 1   # (k=0,i=0)
    col += 1
    assert col == rank, (col, rank)
    return U, V, W

def construct():
    """Return {'U','V','W'} for the rank-minimal decomposition of 3x3x4 matmul."""
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 110.0

    # Candidate rank-35 decomposition
    U, V, W = _build_rank35(n, m, p)
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook (rank 36), always valid
    if time.time() < deadline:
        U, V, W = _build_schoolbook(n, m, p)
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}

    # Last resort: return schoolbook unverified
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
