"""Matrix multiplication 4x4 * 4x9 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 9]
_START = time.time()
_DEADLINE = 138.0  # safety margin below the 150s kill


def _build_tensor(n, m, p):
    """T[io, u, v] = 1 iff output (i,k) uses a[i,j]*b[j,k]; io=k*n+i, u=i*m+j, v=j*p+k."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(k * n + i, i * m + j, j * p + k)] = 1
    return T


def _exact_check(U, V, W, T, n, m, p):
    """Verify sum_c U[u,c]*V[v,c]*W[io,c] == T[io,u,v] for all (io,u,v)."""
    rank = len(U[0]) if U else 0
    for io in range(p * n):
        for u in range(n * m):
            for v in range(m * p):
                s = 0
                for c in range(rank):
                    s += U[u][c] * V[v][c] * W[io][c]
                if s != T.get((io, u, v), 0):
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _try_reduce_rank(U, V, W, T, n, m, p):
    """
    Try to find linear dependencies among rank-1 terms.
    For each pair of terms, check if one can be expressed as a linear combination
    of others. If so, eliminate it.
    
    We look for: M_c = sum_{d != c} alpha_d * M_d where M_c = W_c ⊗ U_c ⊗ V_c
    This requires solving a linear system over the rationals.
    """
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    
    if rank == 0:
        return U, V, W
    
    # For each term c, build its sparse representation
    def get_term_support(c):
        """Get dict mapping (io,u,v) -> value for term c."""
        support = {}
        for io in range(p * n):
            wval = W[io][c]
            if wval == 0:
                continue
            for u in range(n * m):
                uval = U[u][c]
                if uval == 0:
                    continue
                for v in range(m * p):
                    val = wval * uval * V[v][c]
                    if val != 0:
                        support[(io, u, v)] = val
        return support
    
    # Try to eliminate terms by finding rational linear combinations
    # We'll iterate and try to remove one term at a time
    changed = True
    while changed:
        changed = False
        rank = len(U[0])
        for c in range(rank):
            if time.time() > _DEADLINE - 10:
                return U, V, W
            
            support_c = get_term_support(c)
            if not support_c:
                # Zero term, remove it
                U = [row[:c] + row[c+1:] for row in U]
                V = [row[:c] + row[c+1:] for row in V]
                W = [row[:c] + row[c+1:] for row in W]
                changed = True
                break
            
            # Try to express term c as linear combination of others
            # Build system: for each (io,u,v) in support_c,
            # sum_{d != c} alpha_d * W[io][d]*U[u][d]*V[v][d] = support_c[(io,u,v)]
            
            keys = list(support_c.keys())
            other_indices = [d for d in range(rank) if d != c]
            
            if not other_indices:
                continue
            
            # Build linear system A * alpha = b
            # A[i][j] = W[keys[i][0]][other_indices[j]] * U[keys[i][1]][other_indices[j]] * V[keys[i][2]][other_indices[j]]
            # b[i] = support_c[keys[i]]
            
            n_eq = len(keys)
            n_unknowns = len(other_indices)
            
            # Only try if the system is not too large
            if n_unknowns > 20 or n_eq > 30:
                continue
            
            A = []
            b = []
            for key in keys:
                io, u, v = key
                row = []
                for d in other_indices:
                    row.append(W[io][d] * U[u][d] * V[v][d])
                A.append(row)
                b.append(support_c[key])
            
            # Solve over rationals using Gaussian elimination with Fractions
            try:
                A_frac = [[Fraction(x) for x in row] for row in A]
                b_frac = [Fraction(x) for x in b]
                
                # Augmented matrix
                aug = [A_frac[i] + [b_frac[i]] for i in range(n_eq)]
                
                # Gaussian elimination
                r = 0
                for col in range(n_unknowns):
                    # Find pivot
                    piv = None
                    for i in range(r, n_eq):
                        if aug[i][col] != 0:
                            piv = i
                            break
                    if piv is None:
                        continue
                    
                    # Swap rows
                    aug[r], aug[piv] = aug[piv], aug[r]
                    
                    # Normalize pivot row
                    pv = aug[r][col]
                    aug[r] = [x / pv for x in aug[r]]
                    
                    # Eliminate other rows
                    for i in range(n_eq):
                        if i != r and aug[i][col] != 0:
                            f = aug[i][col]
                            aug[i] = [x - f * y for x, y in zip(aug[i], aug[r])]
                    r += 1
                
                # Check for inconsistency
                consistent = True
                for i in range(r, n_eq):
                    if any(aug[i][j] != 0 for j in range(n_unknowns)):
                        consistent = False
                        break
                    if aug[i][n_unknowns] != 0:
                        consistent = False
                        break
                
                if not consistent:
                    continue
                
                # Extract solution
                alpha = [Fraction(0)] * n_unknowns
                for i in range(r):
                    # Find leading column in row i
                    lead_col = None
                    for j in range(n_unknowns):
                        if aug[i][j] != 0:
                            lead_col = j
                            break
                    if lead_col is not None:
                        alpha[lead_col] = aug[i][n_unknowns]
                
                # Check if all alpha are integers (or at least rational with small denominators)
                # For exact decomposition, we need integer coefficients
                all_int = all(a.denominator == 1 for a in alpha)
                
                if all_int:
                    # We can eliminate term c
                    # New decomposition: replace term c with sum of alpha_d * term_d
                    # This means we just remove term c (the others already account for it)
                    U = [row[:c] + row[c+1:] for row in U]
                    V = [row[:c] + row[c+1:] for row in V]
                    W = [row[:c] + row[c+1:] for row in W]
                    changed = True
                    break
                    
            except (ZeroDivisionError, ValueError):
                continue
    
    return U, V, W


def _structured_decomposition(n, m, p):
    """
    Build a structured decomposition that may have exploitable symmetries.
    Uses the fact that C[i,k] = sum_j A[i,j]*B[j,k].
    
    We create rank-1 terms indexed by (i,j,k) but with potential for
    linear dependencies when n,m,p share factors.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Start with schoolbook
    U, V, W = _structured_decomposition(n, m, p)
    
    # Try to reduce rank
    U, V, W = _try_reduce_rank(U, V, W, T, n, m, p)
    
    # Verify
    if _exact_check(U, V, W, T, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    best = _schoolbook(n, m, p)
    return best
