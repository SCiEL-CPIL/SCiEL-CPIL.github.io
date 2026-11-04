"""Matrix multiplication 3x3x5 rank-minimal exact tensor decomposition search."""
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import random
import numpy as np
from itertools import combinations

def _build_tensor(n, m, p):
    """Return dict T[(a,b,c)] = 1 for the multiplication tensor.
    a = i*m + j, b = j*p + k, c = k*n + i.
    """
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[(a, b, c)] = 1
    return T

def _verify(U, V, W, T, n, m, p):
    """Check that sum_c U[a,c]*V[b,c]*W[c] == T[(a,b,c)] for all a,b,c."""
    rank = len(U[0]) if U else 0
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                s = Fraction(0)
                for col in range(rank):
                    s += U[a][col] * V[b][col] * W[c][col]
                expected = T.get((a, b, c), 0)
                if s != expected:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    return U, V, W

def _to_output(U, V, W):
    """Convert Fraction matrices to int/half-int lists for output."""
    def conv(M):
        out = []
        for row in M:
            r = []
            for x in row:
                if x.denominator == 1:
                    r.append(int(x.numerator))
                elif x.denominator == 2:
                    r.append(Fraction(x.numerator, 2))
                else:
                    r.append(x)
            out.append(r)
        return out
    return {"U": conv(U), "V": conv(V), "W": conv(W)}

def _try_merge_two_terms(n, m, p, term1, term2, U, V, W):
    """
    Try to merge two schoolbook terms into one by finding a rank-1 tensor
    that equals the sum of the two original rank-1 tensors, OR by adjusting
    coefficients. Returns modified (U,V,W) with one fewer column, or None.
    """
    i1, j1, k1 = term1
    i2, j2, k2 = term2
    
    a1, b1, c1 = i1*m+j1, j1*p+k1, k1*n+i1
    a2, b2, c2 = i2*m+j2, j2*p+k2, k2*n+i2
    
    if a1 == a2 and b1 == b2 and c1 == c2:
        return None
    
    u_support = {a1} if a1 == a2 else {a1, a2}
    v_support = {b1} if b1 == b2 else {b1, b2}
    w_support = {c1} if c1 == c2 else {c1, c2}
    
    for a in u_support:
        for b in v_support:
            for c in w_support:
                if (a, b, c) != (a1, b1, c1) and (a, b, c) != (a2, b2, c2):
                    return None
    
    candidates = [Fraction(1), Fraction(-1), Fraction(2), Fraction(-2), 
                  Fraction(1,2), Fraction(-1,2)]
    
    for u_val1 in candidates:
        for u_val2 in candidates:
            for v_val1 in candidates:
                for v_val2 in candidates:
                    try:
                        w_c1 = Fraction(1) / (u_val1 * v_val1)
                        w_c2 = Fraction(1) / (u_val2 * v_val2)
                    except ZeroDivisionError:
                        continue
                    
                    if w_c1 not in candidates and w_c2 not in candidates:
                        continue
                    
                    U_new = [row[:] for row in U]
                    V_new = [row[:] for row in V]
                    W_new = [row[:] for row in W]
                    
                    col1 = None
                    col2 = None
                    for col in range(len(U[0])):
                        if U[a1][col] == 1 and V[b1][col] == 1 and W[c1][col] == 1:
                            if col1 is None:
                                col1 = col
                        if U[a2][col] == 1 and V[b2][col] == 1 and W[c2][col] == 1:
                            if col2 is None:
                                col2 = col
                    
                    if col1 is None or col2 is None or col1 == col2:
                        continue
                    
                    for a in range(n*m):
                        U_new[a][col1] = Fraction(0)
                    for b in range(m*p):
                        V_new[b][col1] = Fraction(0)
                    for c in range(p*n):
                        W_new[c][col1] = Fraction(0)
                    
                    for a in u_support:
                        if a == a1 and a == a2:
                            U_new[a][col1] = u_val1
                        elif a == a1:
                            U_new[a][col1] = u_val1
                        elif a == a2:
                            U_new[a][col1] = u_val2
                    
                    for b in v_support:
                        if b == b1 and b == b2:
                            V_new[b][col1] = v_val1
                        elif b == b1:
                            V_new[b][col1] = v_val1
                        elif b == b2:
                            V_new[b][col1] = v_val2
                    
                    for c in w_support:
                        if c == c1 and c == c2:
                            W_new[c][col1] = w_c1
                        elif c == c1:
                            W_new[c][col1] = w_c1
                        elif c == c2:
                            W_new[c][col1] = w_c2
                    
                    for a in range(n*m):
                        U_new[a][col2] = Fraction(0)
                    for b in range(m*p):
                        V_new[b][col2] = Fraction(0)
                    for c in range(p*n):
                        W_new[c][col2] = Fraction(0)
                    
                    return U_new, V_new, W_new, col2
    
    return None

def _try_strassen_like(n, m, p):
    """
    Try to use a Strassen-like decomposition for the 3x3x3 core
    and handle the extra 2 columns of the 5-column matrix.
    """
    return None

def _try_remove_column(n, m, p, U, V, W, col_idx, T):
    """Try removing a single column and see if the decomposition still works."""
    rank = len(U[0])
    if rank <= 1:
        return None
    
    U_new = [row[:col_idx] + row[col_idx+1:] for row in U]
    V_new = [row[:col_idx] + row[col_idx+1:] for row in V]
    W_new = [row[:col_idx] + row[col_idx+1:] for row in W]
    
    if _verify(U_new, V_new, W_new, T, n, m, p):
        return U_new, V_new, W_new
    
    return None

def _try_replace_column(n, m, p, U, V, W, col_idx, T, candidates):
    """Try replacing a column with a different rank-1 tensor."""
    rank = len(U[0])
    if rank <= 1:
        return None
    
    for cand in candidates:
        u_vals, v_vals, w_vals = cand
        U_new = [row[:] for row in U]
        V_new = [row[:] for row in V]
        W_new = [row[:] for row in W]
        
        for a in range(n*m):
            U_new[a][col_idx] = u_vals[a]
        for b in range(m*p):
            V_new[b][col_idx] = v_vals[b]
        for c in range(p*n):
            W_new[c][col_idx] = w_vals[c]
        
        if _verify(U_new, V_new, W_new, T, n, m, p):
            return U_new, V_new, W_new
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 130.0

    T = _build_tensor(n, m, p)
    rank0 = n * m * p

    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = rank0

    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append((i, j, k))
    
    candidate_pairs = []
    for idx1, t1 in enumerate(terms):
        for idx2 in range(idx1 + 1, len(terms)):
            t2 = terms[idx2]
            if t1[0] == t2[0] or t1[1] == t2[1] or t1[2] == t2[2]:
                candidate_pairs.append((t1, t2))
    
    random.seed(42)
    random.shuffle(candidate_pairs)
    
    # First pass: try merging pairs
    for t1, t2 in candidate_pairs:
        if time.time() > deadline:
            break
        result = _try_merge_two_terms(n, m, p, t1, t2, best_U, best_V, best_W)
        if result is not None:
            U_new, V_new, W_new, col_to_remove = result
            U_final = [row[:col_to_remove] + row[col_to_remove+1:] for row in U_new]
            V_final = [row[:col_to_remove] + row[col_to_remove+1:] for row in V_new]
            W_final = [row[:col_to_remove] + row[col_to_remove+1:] for row in W_new]
            
            if _verify(U_final, V_final, W_final, T, n, m, p):
                best_U, best_V, best_W = U_final, V_final, W_final
                best_rank = best_rank - 1
                print(f"Found rank {best_rank} decomposition!")
                break
    
    # Second pass: try removing any column
    if time.time() < deadline:
        for col in range(len(best_U[0])):
            if time.time() > deadline:
                break
            result = _try_remove_column(n, m, p, best_U, best_V, best_W, col, T)
            if result is not None:
                best_U, best_V, best_W = result
                best_rank -= 1
                print(f"Removed column, rank {best_rank}!")
                break
    
    # Third pass: try more pair merges with updated matrices
    if time.time() < deadline:
        for t1, t2 in candidate_pairs:
            if time.time() > deadline:
                break
            result = _try_merge_two_terms(n, m, p, t1, t2, best_U, best_V, best_W)
            if result is not None:
                U_new, V_new, W_new, col_to_remove = result
                U_final = [row[:col_to_remove] + row[col_to_remove+1:] for row in U_new]
                V_final = [row[:col_to_remove] + row[col_to_remove+1:] for row in V_new]
                W_final = [row[:col_to_remove] + row[col_to_remove+1:] for row in W_new]
                
                if _verify(U_final, V_final, W_final, T, n, m, p):
                    best_U, best_V, best_W = U_final, V_final, W_final
                    best_rank = best_rank - 1
                    print(f"Found rank {best_rank} decomposition in pass 3!")
                    break
    
    return _to_output(best_U, best_V, best_W)
# EVOLVE-BLOCK-END
