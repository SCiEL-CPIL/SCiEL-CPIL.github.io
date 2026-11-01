import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]

def _target_tensor(n, m, p):
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                key = (i*m+j, j*p+k, k*n+i)
                T[key] = T.get(key, 0) + 1
    return T

def _from_columns(cols, n, m, p, rank):
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for r, (Ac, Bc, Cc) in enumerate(cols):
        for (i, j), v in Ac.items():
            U[i*m+j][r] += v
        for (j, k), v in Bc.items():
            V[j*p+k][r] += v
        for (i, k), v in Cc.items():
            W[k*n+i][r] += v
    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    target = _target_tensor(n, m, p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j; b = j*p+k; c = k*n+i
                s = 0
                for r in range(rank):
                    s += U[a][r]*V[b][r]*W[c][r]
                if abs(s - target.get((a, b, c), 0)) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return U, V, W

def _build_low_rank(n, m, p):
    cols = []
    def strassen(ai, aj, bj, bk, ci, ck):
        def A(i, j): return (ai+i, aj+j)
        def B(j, k): return (bj+j, bk+k)
        def C(i, k): return (ci+i, ck+k)
        M = [
            ([(A(0,0),1),(A(1,1),1)], [(B(0,0),1),(B(1,1),1)], [(C(0,0),1),(C(1,1),1)]),
            ([(A(1,0),1),(A(1,1),1)], [(B(0,0),1)], [(C(1,0),1),(C(1,1),-1)]),
            ([(A(0,0),1)], [(B(1,0),1),(B(1,1),-1)], [(C(0,1),1),(C(1,1),1)]),
            ([(A(1,1),1)], [(B(0,1),1),(B(1,0),-1)], [(C(0,0),1),(C(1,0),1)]),
            ([(A(0,0),1),(A(0,1),1)], [(B(1,0),1)], [(C(0,0),-1),(C(0,1),1)]),
            ([(A(0,1),1),(A(0,0),-1)], [(B(0,0),1),(B(1,0),1)], [(C(1,1),1)]),
            ([(A(1,0),1),(A(1,1),-1)], [(B(1,1),1),(B(0,1),1)], [(C(0,0),1),(C(0,1),-1)]),
        ]
        for Ac, Bc, Cc in M:
            cols.append((dict(Ac), dict(Bc), dict(Cc)))
    strassen(0, 0, 0, 0, 0, 0)      # A11 B11 -> C11
    strassen(0, 0, 0, 2, 0, 2)      # A11 B12 -> C12
    cols.append(( {(0,2):1,(1,2):1}, {(2,0):1,(2,1):1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1} ))  # A12 B21 -> C11
    cols.append(( {(0,2):1,(1,2):1}, {(2,2):1,(2,3):1}, {(0,2):1,(0,3):1,(1,2):1,(1,3):1} ))  # A12 B22 -> C12
    for k in range(2):
        cols.append(( {(2,0):1,(2,1):1}, {(0,k):1,(1,k):1}, {(2,k):1} ))                       # A21 B11 -> C21
    for k in range(2):
        cols.append(( {(2,0):1,(2,1):1}, {(0,2+k):1,(1,2+k):1}, {(2,2+k):1} ))                 # A21 B12 -> C22
    cols.append(( {(2,2):1}, {(2,0):1,(2,1):1}, {(2,0):1,(2,1):1} ))                           # A22 B21 -> C21
    cols.append(( {(2,2):1}, {(2,2):1,(2,3):1}, {(2,2):1,(2,3):1} ))                           # A22 B22 -> C22
    rank = len(cols)
    return _from_columns(cols, n, m, p, rank)

def _try_lower_rank(n, m, p, max_rank, deadline):
    """Try to find a lower rank decomposition by searching for sparse patterns."""
    target = _target_tensor(n, m, p)
    target_keys = set(target.keys())
    
    # Try various rank values from low to high
    for rank in range(18, max_rank + 1):
        if time.time() > deadline:
            break
        
        # For each rank, try to find a valid decomposition
        # Use a greedy approach: pick terms that cover the most uncovered targets
        covered = set()
        cols = []
        
        for r in range(rank):
            if time.time() > deadline:
                break
            
            # Find the best term to add
            best_score = -1
            best_term = None
            
            # Try simple basis-like terms
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        key = (i*m+j, j*p+k, k*n+i)
                        if key in target and target[key] == 1:
                            # Check how many new targets this covers
                            # A single term only covers one target
                            if key not in covered:
                                score = 1
                                if score > best_score:
                                    best_score = score
                                    best_term = ({(i,j): 1}, {(j,k): 1}, {(i,k): 1})
            
            # Also try more complex terms with multiple entries
            # Try terms with 2 entries in each factor
            for i1 in range(n):
                for i2 in range(n):
                    if i1 == i2:
                        continue
                    for j in range(m):
                        for k1 in range(p):
                            for k2 in range(p):
                                if k1 == k2:
                                    continue
                                # Term: (i1,j)*(j,k1) -> (i1,k1) + (i2,j)*(j,k2) -> (i2,k2)
                                # This is getting complex, let's try a simpler approach
                                pass
            
            if best_term is None:
                # No more simple terms to add, try to complete with schoolbook
                break
            
            Ac, Bc, Cc = best_term
            cols.append((Ac, Bc, Cc))
            for key in target_keys:
                a, b, c = key
                if (a // m, a % m) in Ac and (b // p, b % p) in Bc and (c // n, c % n) in Cc:
                    # Check if this term contributes to this target
                    i, j = a // m, a % m
                    jj, k = b // p, b % p
                    kk, ii = c // n, c % n
                    if i == ii and j == jj and k == kk:
                        covered.add(key)
        
        if len(cols) == rank:
            # Verify this decomposition
            U, V, W = _from_columns(cols, n, m, p, rank)
            if _verify(U, V, W, n, m, p, rank):
                return U, V, W, rank
    
    return None

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    DEADLINE = 115.0
    
    # First try the Strassen-based approach (rank 22)
    U, V, W = _build_low_rank(n, m, p)
    current_rank = len(U[0])
    if _verify(U, V, W, n, m, p, current_rank) and (time.time() - t0) < DEADLINE:
        best_result = {"U": U, "V": V, "W": W}
        best_rank = current_rank
    else:
        best_result = None
        best_rank = n * m * p
    
    # Try to find a lower rank decomposition
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 5.0 and best_rank > 18:
        result = _try_lower_rank(n, m, p, best_rank - 1, DEADLINE - 5.0)
        if result is not None:
            U2, V2, W2, rank2 = result
            if _verify(U2, V2, W2, n, m, p, rank2):
                best_result = {"U": U2, "V": V2, "W": W2}
                best_rank = rank2
    
    # If we still have time, try even lower ranks
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 10.0 and best_rank > 16:
        result = _try_lower_rank(n, m, p, 17, DEADLINE - 5.0)
        if result is not None:
            U2, V2, W2, rank2 = result
            if _verify(U2, V2, W2, n, m, p, rank2):
                best_result = {"U": U2, "V": V2, "W": W2}
                best_rank = rank2
    
    if best_result is not None and (time.time() - t0) < DEADLINE:
        return best_result
    
    if time.time() - t0 < DEADLINE:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    return {"U": [[0]], "V": [[0]], "W": [[0]]}
