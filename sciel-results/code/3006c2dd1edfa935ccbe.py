"""Matrix multiplication 4x4 * 4x9 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 9]
_START = time.time()
_DEADLINE = 138.0  # safety margin below the 150s kill


def _winograd_2x2x2():
    """
    Winograd rank-7 decomposition of the 2x2x2 matrix multiplication tensor.
    
    For C = A * B where A, B are 2x2:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Winograd's method uses 7 multiplications:
    Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    
    u0 = (a+d)*(e+h)
    u1 = (c+d)*e
    u2 = d*(f-e)
    u3 = (a+b)*h
    u4 = b*(g-h)
    u5 = (a+c)*(e+g)
    u6 = (b+d)*(f+h)
    
    C[0,0] = u0 - u1 - u4 + u5
    C[0,1] = u3 - u2 + u4 + u6
    C[1,0] = u1 - u3 + u5 - u6
    C[1,1] = u0 - u2 + u4 - u5
    """
    # 7 rank-1 terms
    # U has shape (n*m, rank) = (4, 7) for 2x2 input A
    # V has shape (m*p, rank) = (4, 7) for 2x2 input B  
    # W has shape (p*n, rank) = (4, 7) for 2x2 output C
    
    # Using indices: A[i,j] -> u_idx = i*2+j, B[j,k] -> v_idx = j*2+k, C[i,k] -> w_idx = i*2+k
    
    # Term 0: u0 = (a+d)*(e+h)
    # A part: a=1 (i=0,j=0), d=1 (i=1,j=1) -> U[0]=1, U[3]=1
    # B part: e=1 (j=0,k=0), h=1 (j=1,k=1) -> V[0]=1, V[3]=1
    # C part: contributes to all C[i,k] with coefficient 1
    # C[0,0]=1, C[0,1]=0, C[1,0]=0, C[1,1]=1? No...
    # Actually u0 contributes: C[0,0]+=u0, C[0,1]+=0, C[1,0]+=0, C[1,1]+=u0? 
    # Let me re-derive more carefully.
    
    # Standard Winograd 2x2x2:
    # C[0,0] = u0 - u1 - u4 + u5
    # C[0,1] = u3 - u2 + u4 + u6
    # C[1,0] = u1 - u3 + u5 - u6
    # C[1,1] = u0 - u2 + u4 - u5
    
    # u0 = (a+d)*(e+h): A factors a,d; B factors e,h
    # u1 = (c+d)*e: A factors c,d; B factor e
    # u2 = d*(f-e): A factor d; B factors f,e (with f=1, e=-1)
    # u3 = (a+b)*h: A factors a,b; B factor h
    # u4 = b*(g-h): A factor b; B factors g,h (with g=1, h=-1)
    # u5 = (a+c)*(e+g): A factors a,c; B factors e,g
    # u6 = (b+d)*(f+h): A factors b,d; B factors f,h
    
    # U matrix (4 rows for A[0,0],A[0,1],A[1,0],A[1,1], 7 cols for u0..u6)
    # A[0,0]=a: u0(+1), u3(+1), u5(+1) -> [1, 0, 0, 1, 0, 1, 0]
    # A[0,1]=b: u3(+1), u4(+1), u6(+1) -> [0, 0, 0, 1, 1, 0, 1]
    # A[1,0]=c: u1(+1), u5(+1) -> [0, 1, 0, 0, 0, 1, 0]
    # A[1,1]=d: u0(+1), u1(+1), u2(+1), u6(+1) -> [1, 1, 1, 0, 0, 0, 1]
    
    U2 = [
        [1, 0, 0, 1, 0, 1, 0],  # a
        [0, 0, 0, 1, 1, 0, 1],  # b
        [0, 1, 0, 0, 0, 1, 0],  # c
        [1, 1, 1, 0, 0, 0, 1],  # d
    ]
    
    # V matrix (4 rows for B[0,0],B[0,1],B[1,0],B[1,1], 7 cols)
    # B[0,0]=e: u0(+1), u1(+1), u2(-1), u5(+1) -> [1, 1, -1, 0, 0, 1, 0]
    # B[0,1]=f: u2(+1), u6(+1) -> [0, 0, 1, 0, 0, 0, 1]
    # B[1,0]=g: u4(+1), u5(+1) -> [0, 0, 0, 0, 1, 1, 0]
    # B[1,1]=h: u0(+1), u3(+1), u4(-1), u6(+1) -> [1, 0, 0, 1, -1, 0, 1]
    
    V2 = [
        [1, 1, -1, 0, 0, 1, 0],   # e
        [0, 0, 1, 0, 0, 0, 1],    # f
        [0, 0, 0, 0, 1, 1, 0],    # g
        [1, 0, 0, 1, -1, 0, 1],   # h
    ]
    
    # W matrix (4 rows for C[0,0],C[0,1],C[1,0],C[1,1], 7 cols)
    # C[0,0]: u0(+1), u1(-1), u4(-1), u5(+1) -> [1, -1, 0, 0, -1, 1, 0]
    # C[0,1]: u3(+1), u2(-1), u4(+1), u6(+1) -> [0, 0, -1, 1, 1, 0, 1]
    # C[1,0]: u1(+1), u3(-1), u5(+1), u6(-1) -> [0, 1, 0, -1, 0, 1, -1]
    # C[1,1]: u0(+1), u2(-1), u4(+1), u5(-1) -> [1, 0, -1, 0, 1, -1, 0]
    
    W2 = [
        [1, -1, 0, 0, -1, 1, 0],   # C[0,0]
        [0, 0, -1, 1, 1, 0, 1],    # C[0,1]
        [0, 1, 0, -1, 0, 1, -1],   # C[1,0]
        [1, 0, -1, 0, 1, -1, 0],   # C[1,1]
    ]
    
    return U2, V2, W2


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
