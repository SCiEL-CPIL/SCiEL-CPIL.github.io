"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1"""
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    return T

def _check_decomposition_exact(U, V, W, T):
    """Check if U, V, W exactly decompose tensor T using integer arithmetic."""
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    nm, mp, pn = T.shape
    r = U.shape[1]
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                s = 0
                for rr in range(r):
                    s += U[a, rr] * V[b, rr] * W[c, rr]
                if s != T[a, b, c]:
                    return False
    return True

def _try_rank_greedy(r, n, m, p, T, deadline):
    """Try to find a rank-r decomposition using a greedy approach."""
    if time.time() > deadline:
        return None
    
    nm, mp, pn = T.shape
    # Initialize with zeros
    U = np.zeros((nm, r), dtype=object)
    V = np.zeros((mp, r), dtype=object)
    W = np.zeros((pn, r), dtype=object)
    
    # Remaining tensor
    T_rem = T.astype(object).copy()
    
    for col in range(r):
        if time.time() > deadline - 1:
            return None
        
        # Find the maximum entry in T_rem to guide the choice
        max_val = 0
        max_pos = (0, 0, 0)
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if abs(T_rem[a, b, c]) > max_val:
                        max_val = abs(T_rem[a, b, c])
                        max_pos = (a, b, c)
        
        if max_val == 0:
            # All remaining entries are zero, we can stop early
            break
        
        a, b, c = max_pos
        # Set U[a, col] = 1, V[b, col] = 1, W[c, col] = T_rem[a, b, c]
        # Then subtract U[:, col] * V[:, col] * W[:, col] from T_rem
        U[a, col] = 1
        V[b, col] = 1
        W[c, col] = T_rem[a, b, c]
        
        for aa in range(nm):
            for bb in range(mp):
                for cc in range(pn):
                    T_rem[aa, bb, cc] -= U[aa, col] * V[bb, col] * W[cc, col]
    
    # Check if all columns are used or T_rem is zero
    if np.all(T_rem == 0):
        # Trim unused columns
        used_cols = [col for col in range(r) if not (np.all(U[:, col] == 0) and np.all(V[:, col] == 0) and np.all(W[:, col] == 0))]
        if used_cols:
            U = U[:, used_cols]
            V = V[:, used_cols]
            W = W[:, used_cols]
        else:
            U = U[:, :1]
            V = V[:, :1]
            W = W[:, :1]
        return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    deadline = time.time() + 115  # 115 second deadline (5s safety margin)
    
    # Schoolbook rank
    schoolbook = n * m * p
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    lower_bound = max(n*m, m*p, p*n)
    
    best_rank = schoolbook
    best_decomp = None
    
    # First, get the schoolbook decomposition as fallback
    U_school = [[0] * schoolbook for _ in range(n*m)]
    V_school = [[0] * schoolbook for _ in range(m*p)]
    W_school = [[0] * schoolbook for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_decomp = (U_school, V_school, W_school)
    
    # Try to find lower rank decompositions
    for r in range(lower_bound, schoolbook):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank_greedy(r, n, m, p, T, deadline)
        if result is not None:
            U, V, W = result
            # Verify the decomposition
            if _check_decomposition_exact(U, V, W, T):
                best_rank = r
                best_decomp = (U, V, W)
                break
    
    return {"U": best_decomp[0], "V": best_decomp[1], "W": best_decomp[2]}
# EVOLVE-BLOCK-END
