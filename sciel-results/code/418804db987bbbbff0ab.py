"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 3, 4]

def _verify_solution(U, V, W, n, m, p):
    """
    Verifies if U, V, W form a valid decomposition for M_{n,m,p}.
    U: (nm, rank), V: (mp, rank), W: (pn, rank)
    Condition: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] == delta(j, j') for all i, k, j, j'
    """
    rank = len(U[0])
    # We only need to check one i and one k for all j, j' pairs if the structure is uniform?
    # No, the condition must hold for ALL i, k, j, j'.
    # Due to symmetry of the tensor, checking one representative i, k might be sufficient if the matrices are constructed uniformly,
    # but here we must be strict.
    
    # To be fast, we check a few random instances of i, k.
    # Actually, for small n,m,p, checking all is fast.
    # n=2, m=3, p=4. Total checks: 2*4*3*3 = 72 checks.
    # Each check is a dot product of length 'rank'.
    # Rank 24 -> 24 multiplications. 72*24 = 1728 ops. Trivial.
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for jp in range(m):
                    s = 0.0
                    for c in range(rank):
                        s += W[k * n + i][c] * U[i * m + j][c] * V[jp * p + k][c]
                    target = 1.0 if j == jp else 0.0
                    if abs(s - target) > 1e-9:
                        return False
    return True

def _schoolbook():
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

def construct():
    # Return the known schoolbook solution which is valid.
    # Searching for rank 20 is too risky/slow without a hardcoded lookup.
    return _schoolbook()

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
