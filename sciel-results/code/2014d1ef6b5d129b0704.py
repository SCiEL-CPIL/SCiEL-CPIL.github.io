<<<<<<< SEARCH
def _verify(U, V, W, n, m, p):
    """Check exact decomposition: C[i,k] = sum_j A[i,j]B[j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    We check that this equals 1 if (i,j,k) is a valid index, else 0.
    But actually the tensor for matmul is T[i,j,k] = 1 for all valid i,j,k since
    C[i,k] = sum_j A[i,j] B[j,k]. So T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Check all indices exactly
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True
=======
def _verify(U, V, W, n, m, p):
    """Check exact decomposition using rational arithmetic for exactness."""
    from fractions import Fraction
    
    U_frac = [[Fraction(U[i][j]) for j in range(len(U[i]))] for i in range(len(U))]
    V_frac = [[Fraction(V[i][j]) for j in range(len(V[i]))] for i in range(len(V))]
    W_frac = [[Fraction(W[i][j]) for j in range(len(W[i]))] for i in range(len(W))]
    
    rank = len(U[0]) if U else 0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for r in range(rank):
                    val += U_frac[i * m + j][r] * V_frac[j * p + k][r] * W_frac[k * n + i][r]
                if val != Fraction(1):
                    return False
    return True
>>>>>>> REPLACE
