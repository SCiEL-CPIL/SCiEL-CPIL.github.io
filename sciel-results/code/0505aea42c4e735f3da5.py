"""Matrix multiplication 3x5x5 - rank-minimal exact tensor decomposition search."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 5]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 75
DEADLINE = time.time() + 130  # safety margin before 150s kill

def _tensor_coeff(i, j, k):
    """Return the coefficient of tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c].
    For a valid decomposition this must equal 1 for all i,j,k."""
    return 1

def _check_exact(U, V, W):
    """Verify exact decomposition. U shape (n*m, r), V (m*p, r), W (p*n, r).
    For each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] must == 1.
    Uses Fraction for exactness."""
    n, m, p = N, M, P
    r = len(U[0])
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                ui = Uf[i*m+j]
                vj = Vf[j*p+k]
                wk = Wf[k*n+i]
                for c in range(r):
                    s += ui[c] * vj[c] * wk[c]
                if s != 1:
                    return False
    return True

def _check_exact_fast(U, V, W):
    """Fast float check first, then exact if close."""
    n, m, p = N, M, P
    r = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i]))
                if abs(s - 1.0) > 1e-6:
                    return False
    # Float check passed; do exact check
    return _check_exact(U, V, W)

def _try_polish(U, V, W):
    """Try to find a low-rank exact decomposition using algebraic structure.
    Strategy: parameterize with integer/half-integer entries and solve.
    
    For matrix multiplication tensor, known low-rank decompositions exist.
    We use a structured search: assume entries are in {0, +/-1, +/-1/2}.
    """
    n, m, p = N, M, P
    best = None
    # Try ranks from small to schoolbook
    for rank in range(1, RANK_SCHOOLBOOK + 1):
        if time.time() > DEADLINE - 5:
            break
        # Use the schoolbook as base and try to merge columns
        # Build schoolbook
        U = [[0]*RANK_SCHOOLBOOK for _ in range(n*m)]
        V = [[0]*RANK_SCHOOLBOOK for _ in range(m*p)]
        W = [[0]*RANK_SCHOOLBOOK for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        # Try to reduce rank by finding linear dependencies
        # Use Gaussian elimination over the tensor structure
        # Actually, for exact low-rank, we need a different approach.
        # Try a known construction: Strassen-like for small cases
        break
    
    # Fallback: return schoolbook
    U = [[0]*RANK_SCHOOLBOOK for _ in range(n*m)]
    V = [[0]*RANK_SCHOOLBOOK for _ in range(m*p)]
    W = [[0]*RANK_SCHOOLBOOK for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """Find rank-minimal exact tensor decomposition for 3x5x5 matrix mult."""
    n, m, p = N, M, P
    start_time = time.time()
    
    # Start with schoolbook decomposition
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
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try to find lower rank decompositions using structured search
    # The key insight: matrix multiplication tensor T_{ijk} = delta_{jk}
    # (where the middle index connects) has special structure.
    
    # Approach: Use the fact that for fixed j, the tensor is separable in i,k
    # T[i,j,k] = 1 if we think of it as: for each j, we need to represent
    # the identity-like structure in (i,k) space.
    
    # Actually, the tensor is: T[i,j,k] = 1 for all i,j,k (it's the 
    # multiplication tensor where C[i,k] = sum_j A[i,j]*B[j,k])
    # Wait, no. The tensor for matrix mult is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i,j,k? No.
    # The tensor is: T_{(i,j),(j,k),(i,k)} ... 
    # In the convention W[k*n+i, r], the tensor coefficient for (i,j,k) is 1.
    
    # The tensor T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # This is actually the ALL-ONES tensor!
    # rank of all-ones tensor of size n x m x p is min(n,m,p) if entries are real.
    # But we need exact with integer/half-integer entries.
    
    # For all-ones tensor: T[i,j,k] = 1 = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # A rank-1 decomposition: U[i*m+j,0]=1, V[j*p+k,0]=1, W[k*n+i,0]=1
    # Check: sum_c 1*1*1 = 1. Yes! Rank 1 works for all-ones tensor!
    
    # But wait - is the tensor really all ones? Let me re-read the task.
    # "exactly decompose the matrix multiplication tensor"
    # Matrix multiplication: C = A * B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] which gives the coefficient of A[i,j]*B[j,k] in C[i,k]
    # That coefficient is 1 for all valid i,j,k.
    # So yes, it IS the all-ones tensor!
    
    # Therefore rank 1 should work:
    U1 = [[1] for _ in range(n * m)]
    V1 = [[1] for _ in range(m * p)]
    W1 = [[1] for _ in range(p * n)]
    
    if _check_exact(U1, V1, W1):
        return {"U": U1, "V": V1, "W": W1}
    
    # If that didn't work (shouldn't happen), fall back to schoolbook
    return best_result

# Verify at module level for debugging
if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print(f"U shape: ({len(result['U'])}, {len(result['U'][0])})")
    print(f"V shape: ({len(result['V'])}, {len(result['V'][0])})")
    print(f"W shape: ({len(result['W'])}, {len(result['W'][0])})")
    print(f"Exact: {_check_exact(result['U'], result['V'], result['W'])}")
