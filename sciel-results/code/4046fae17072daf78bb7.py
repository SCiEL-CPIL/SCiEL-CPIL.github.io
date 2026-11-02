"""Matrix multiplication 4x4 * 4x9 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
from itertools import combinations

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
    Try to find linear dependencies among rank-1 terms using vectorized approach.
    For each term c, check if it can be expressed as a linear combination of others.
    If so, eliminate it.
    """
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    
    if rank == 0:
        return U, V, W
    
    changed = True
    while changed:
        changed = False
        rank = len(U[0])
        if rank == 0:
            break
            
        for c in range(rank):
            if time.time() > _DEADLINE - 15:
                return U, V, W
            
            # Check if term c is zero
            if all(U[u][c] == 0 for u in range(n * m)) or \
               all(V[v][c] == 0 for v in range(m * p)) or \
               all(W[io][c] == 0 for io in range(p * n)):
                U = [row[:c] + row[c+1:] for row in U]
                V = [row[:c] + row[c+1:] for row in V]
                W = [row[:c] + row[c+1:] for row in W]
                changed = True
                break
            
            # Try to express term c as linear combination of others
            other_indices = [d for d in range(rank) if d != c]
            n_unknowns = len(other_indices)
            
            if n_unknowns == 0:
                continue
            
            # Build the tensor for term c: T_c[io,u,v] = W[io][c]*U[u][c]*V[v][c]
            # And for each other term d: T_d[io,u,v] = W[io][d]*U[u][d]*V[v][d]
            # We need: sum_d alpha_d * T_d = T_c
            
            # Only consider non-zero entries of T_c
            keys = []
            for io in range(p * n):
                if W[io][c] == 0:
                    continue
                for u in range(n * m):
                    if U[u][c] == 0:
                        continue
                    for v in range(m * p):
                        if V[v][c] != 0:
                            keys.append((io, u, v))
            
            if not keys:
                continue
            
            # Build linear system
            n_eq = len(keys)
            
            # Skip if system too large
            if n_unknowns > 50 or n_eq > 100:
                continue
            
            A = np.zeros((n_eq, n_unknowns), dtype=object)
            b = np.zeros(n_eq, dtype=object)
            
            for i, (io, u, v) in enumerate(keys):
                b[i] = W[io][c] * U[u][c] * V[v][c]
                for j, d in enumerate(other_indices):
                    A[i, j] = W[io][d] * U[u][d] * V[v][d]
            
            # Solve using Fraction-based Gaussian elimination
            try:
                A_frac = [[Fraction(int(A[i, j])) for j in range(n_unknowns)] for i in range(n_eq)]
                b_frac = [Fraction(int(b[i])) for i in range(n_eq)]
                
                # Augmented matrix
                aug = [A_frac[i] + [b_frac[i]] for i in range(n_eq)]
                
                # Gaussian elimination
                r = 0
                for col in range(n_unknowns):
                    piv = None
                    for i in range(r, n_eq):
                        if aug[i][col] != 0:
                            piv = i
                            break
                    if piv is None:
                        continue
                    
                    aug[r], aug[piv] = aug[piv], aug[r]
                    pv = aug[r][col]
                    aug[r] = [x / pv for x in aug[r]]
                    
                    for i in range(n_eq):
                        if i != r and aug[i][col] != 0:
                            f = aug[i][col]
                            aug[i] = [x - f * y for x, y in zip(aug[i], aug[r])]
                    r += 1
                
                # Check consistency
                consistent = True
                for i in range(r, n_eq):
                    if any(aug[i][j] != 0 for j in range(n_unknowns)) or aug[i][n_unknowns] != 0:
                        consistent = False
                        break
                
                if not consistent:
                    continue
                
                # Extract solution
                alpha = [Fraction(0)] * n_unknowns
                for i in range(r):
                    for j in range(n_unknowns):
                        if aug[i][j] != 0:
                            alpha[j] = aug[i][n_unknowns]
                            break
                
                # Check if all coefficients are integers
                if all(a.denominator == 1 for a in alpha):
                    U = [row[:c] + row[c+1:] for row in U]
                    V = [row[:c] + row[c+1:] for row in V]
                    W = [row[:c] + row[c+1:] for row in W]
                    changed = True
                    break
                    
            except (ZeroDivisionError, ValueError, OverflowError):
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


def _try_pairwise_reduction(U, V, W, T, n, m, p):
    """
    Try to find pairs of terms that can be combined.
    For terms c1, c2, check if there exist alpha, beta such that:
    alpha * T_c1 + beta * T_c2 = 0 (or equals another term)
    This is a more targeted search than the full linear system.
    """
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    
    if rank < 2:
        return U, V, W
    
    # For each pair (c1, c2), check if T_c1 - T_c2 = 0 or T_c1 + T_c2 = 0
    # This would mean we can replace both with a single term or eliminate one
    
    for c1 in range(rank):
        if time.time() > _DEADLINE - 20:
            return U, V, W
            
        for c2 in range(c1 + 1, rank):
            # Check if T_c1 == T_c2 (then we can eliminate one)
            # Or T_c1 == -T_c2 (then sum is zero, eliminate both)
            
            same = True
            neg = True
            for io in range(p * n):
                for u in range(n * m):
                    for v in range(m * p):
                        v1 = W[io][c1] * U[u][c1] * V[v][c1]
                        v2 = W[io][c2] * U[u][c2] * V[v][c2]
                        if v1 != v2:
                            same = False
                        if v1 != -v2:
                            neg = False
                        if not same and not neg:
                            break
                    if not same and not neg:
                        break
                if not same and not neg:
                    break
            
            if same:
                # T_c1 == T_c2, eliminate c2
                U = [row[:c2] + row[c2+1:] for row in U]
                V = [row[:c2] + row[c2+1:] for row in V]
                W = [row[:c2] + row[c2+1:] for row in W]
                rank -= 1
                # Adjust indices if c2 < c1
                if c2 < c1:
                    c1 -= 1
                continue
            
            if neg:
                # T_c1 == -T_c2, sum is zero, eliminate both
                # Remove c2 first (higher index)
                U = [row[:c2] + row[c2+1:] for row in U]
                V = [row[:c2] + row[c2+1:] for row in V]
                W = [row[:c2] + row[c2+1:] for row in W]
                rank -= 1
                if c2 < c1:
                    c1 -= 1
                # Now remove c1
                U = [row[:c1] + row[c1+1:] for row in U]
                V = [row[:c1] + row[c1+1:] for row in V]
                W = [row[:c1] + row[c1+1:] for row in W]
                rank -= 1
                c1 = 0  # Restart from beginning
                break
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Start with schoolbook
    U, V, W = _structured_decomposition(n, m, p)
    
    # Try pairwise reduction first (fast)
    U, V, W = _try_pairwise_reduction(U, V, W, T, n, m, p)
    
    # Try full linear reduction
    U, V, W = _try_reduce_rank(U, V, W, T, n, m, p)
    
    # Verify
    if _exact_check(U, V, W, T, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    best = _schoolbook(n, m, p)
    return best
