import random
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]
DEADLINE = 145.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Schoolbook baseline (rank 84)
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        # T[i, k, j] = 1
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def is_half_integer(x, tol=1e-9):
        if abs(x - round(x)) < tol:
            return True
        if abs(2*x - round(2*x)) < tol:
            return True
        return False

    def find_half_integer_combination(coeffs, target=1.0, max_terms=2, search_range=10):
        nz = [(r, c) for r, c in enumerate(coeffs) if abs(c) > 1e-9]
        if not nz:
            return None
        
        # Try single term
        for r, c in nz:
            w = target / c
            if is_half_integer(w):
                return {r: w}
        
        if max_terms >= 2:
            for idx1 in range(len(nz)):
                for idx2 in range(idx1 + 1, len(nz)):
                    r1, c1 = nz[idx1]
                    r2, c2 = nz[idx2]
                    for a_num in range(-search_range*2, search_range*2+1):
                        a = a_num / 2.0
                        if a == 0:
                            continue
                        rem = target - a * c1
                        if abs(rem) < 1e-9:
                            continue
                        b = rem / c2
                        if is_half_integer(b):
                            return {r1: a, r2: b}
        return None

    def try_rank_r_structured(rank, time_limit):
        if time.time() - start_time > time_limit:
            return None
        
        # Try random sparse matrices with specific structure
        for trial in range(200):
            if time.time() - start_time > time_limit:
                return None
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            
            # Sparse initialization
            for idx in range(n*m):
                nz_count = random.choice([1, 1, 2])
                cols = random.sample(range(rank), min(nz_count, rank))
                for c in cols:
                    U[idx, c] = random.choice([-1.0, 1.0])
            
            for idx in range(m*p):
                nz_count = random.choice([1, 1, 2])
                cols = random.sample(range(rank), min(nz_count, rank))
                for c in cols:
                    V[idx, c] = random.choice([-1.0, 1.0])
            
            # Compute M
            M = np.zeros((p*n, rank))
            for i in range(n):
                for k in range(p):
                    for r in range(rank):
                        s = 0.0
                        for j in range(m):
                            s += U[i*m+j, r] * V[j*p+k, r]
                        M[k*n+i, r] = s
            
            # Solve for W
            W = np.zeros((p*n, rank))
            valid = True
            for idx in range(p*n):
                coeffs = M[idx, :]
                result = find_half_integer_combination(coeffs, 1.0, max_terms=2, search_range=8)
                if result is None:
                    valid = False
                    break
                for r, w_val in result.items():
                    W[idx, r] = w_val
            
            if valid and check_decomposition(U, V, W, rank):
                return (U, V, W)
        
        return None

    # Search ranks
    for rank in range(15, 84):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        result = try_rank_r_structured(rank, DEADLINE - 5)
        if result is not None:
            U, V, W = result
            if rank < best_rank:
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
