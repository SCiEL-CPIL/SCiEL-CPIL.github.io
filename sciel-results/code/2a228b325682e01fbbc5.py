"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import sys
import random
from fractions import Fraction

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

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U) or any(len(row) != rank for row in V) or any(len(row) != rank for row in W):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = Fraction(0)
                for c in range(rank):
                    actual += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if actual != 1:
                    return False
    return True

def _gauss_solve(A, b):
    """Solve A*x = b over rationals. A is list of lists, b is list. Returns x or None."""
    n = len(b)
    m = len(A[0]) if A else 0
    if n == 0:
        return [Fraction(0)] * m
    
    # Augmented matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    
    rank = 0
    pivot_cols = []
    for col in range(m):
        pivot_row = -1
        for row in range(rank, n):
            if M[row][col] != 0:
                pivot_row = row
                break
        if pivot_row == -1:
            continue
        M[rank], M[pivot_row] = M[pivot_row], M[rank]
        pivot_val = M[rank][col]
        for c in range(col, m + 1):
            M[rank][c] /= pivot_val
        for row in range(n):
            if row == rank:
                continue
            factor = M[row][col]
            if factor != 0:
                for c in range(col, m + 1):
                    M[row][c] -= factor * M[rank][c]
        pivot_cols.append(col)
        rank += 1
    
    for row in range(rank, n):
        if M[row][m] != 0:
            return None
    
    if rank < m:
        return None
    
    x = [Fraction(0)] * m
    for col in pivot_cols:
        row = pivot_cols.index(col)
        x[col] = M[row][m]
    return x

def _try_rank(r, deadline):
    n, m, p = DIMENSIONS
    num_i, num_j, num_k = n * m, m * p, p * n
    
    for attempt in range(50):
        if time.time() > deadline:
            return None
        
        random.seed(attempt * 12345 + r)
        
        U = [[Fraction(0)] * r for _ in range(num_i)]
        V = [[Fraction(0)] * r for _ in range(num_j)]
        
        for c in range(r):
            for idx in range(num_i):
                if random.random() < 0.4:
                    U[idx][c] = random.choice([Fraction(1), Fraction(-1)])
            for idx in range(num_j):
                if random.random() < 0.4:
                    V[idx][c] = random.choice([Fraction(1), Fraction(-1)])
        
        # Build linear system for W
        num_eqs = n * m * p
        num_unk = num_k * r
        
        A = [[Fraction(0)] * num_unk for _ in range(num_eqs)]
        b = [Fraction(1)] * num_eqs
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq = i * m * p + j * p + k
                    for c in range(r):
                        coeff = U[i * m + j][c] * V[j * p + k][c]
                        if coeff != 0:
                            A[eq][(k * n + i) * r + c] += coeff
        
        W_flat = _gauss_solve(A, b)
        if W_flat is None:
            continue
        
        W = [[W_flat[(k * n + i) * r + c] for c in range(r)] for i in range(n) for k in range(p)]
        
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    best_result = _schoolbook()
    best_rank = n * m * p
    
    lower_bound = max(n * m, m * p, p * n)
    
    for r in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        result = _try_rank(r, deadline)
        if result is not None:
            best_rank = r
            best_result = result
    
    return best_result

# EVOLVE-BLOCK-END
