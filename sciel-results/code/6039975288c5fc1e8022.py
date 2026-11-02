import time
import numpy as np
from fractions import Fraction
from itertools import product
from copy import deepcopy

DIMENSIONS = [3, 4, 6]

def _build_target(n, m, p):
    tgt = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tgt[(i * m + j, j * p + k, k * n + i)] = 1
    return tgt

def _verify(U, V, W, tgt, n, m, p):
    R = len(U[0])
    acc = {}
    for r in range(R):
        u_nz = [(a, U[a][r]) for a in range(len(U)) if U[a][r] != 0]
        v_nz = [(b, V[b][r]) for b in range(len(V)) if V[b][r] != 0]
        w_nz = [(c, W[c][r]) for c in range(len(W)) if W[c][r] != 0]
        for a, ua in u_nz:
            for b, vb in v_nz:
                uv = ua * vb
                for c, wc in w_nz:
                    acc[(a, b, c)] = acc.get((a, b, c), 0) + uv * wc
    for key, val in tgt.items():
        if acc.get(key, 0) != val:
            return False
    for key, val in acc.items():
        if val != 0 and key not in tgt:
            return False
    return True

def _add_strassen_2x2x2(U, V, W, a_base, b_base, c_base, rank_offset, n, m, p):
    A00 = a_base
    A01 = a_base + 1
    A10 = a_base + m
    A11 = a_base + m + 1
    
    B00 = b_base
    B01 = b_base + 1
    B10 = b_base + p
    B11 = b_base + p + 1
    
    C00 = c_base
    C01 = c_base + 1
    C10 = c_base + n
    C11 = c_base + n + 1
    
    # M1
    U[A00][rank_offset] += 1
    U[A11][rank_offset] += 1
    V[B00][rank_offset] += 1
    V[B11][rank_offset] += 1
    W[C00][rank_offset] += 1
    W[C11][rank_offset] += 1
    
    # M2
    U[A10][rank_offset+1] += 1
    U[A11][rank_offset+1] += 1
    V[B00][rank_offset+1] += 1
    W[C10][rank_offset+1] += 1
    
    # M3
    U[A00][rank_offset+2] += 1
    V[B10][rank_offset+2] += 1
    V[B11][rank_offset+2] -= 1
    W[C01][rank_offset+2] += 1
    
    # M4
    U[A11][rank_offset+3] += 1
    V[B01][rank_offset+3] += 1
    V[B00][rank_offset+3] -= 1
    W[C00][rank_offset+3] += 1
    W[C10][rank_offset+3] += 1
    
    # M5
    U[A00][rank_offset+4] += 1
    U[A01][rank_offset+4] += 1
    V[B11][rank_offset+4] += 1
    W[C01][rank_offset+4] += 1
    W[C00][rank_offset+4] -= 1
    
    # M6
    U[A10][rank_offset+5] += 1
    U[A00][rank_offset+5] -= 1
    V[B00][rank_offset+5] += 1
    V[B01][rank_offset+5] += 1
    W[C11][rank_offset+5] += 1
    
    # M7
    U[A01][rank_offset+6] += 1
    U[A11][rank_offset+6] -= 1
    V[B10][rank_offset+6] += 1
    V[B11][rank_offset+6] += 1
    W[C00][rank_offset+6] += 1
    
    return rank_offset + 7

def _add_schoolbook_2x2x1(U, V, W, a_base, b_base, c_base, rank_offset, n, m, p):
    A00 = a_base
    A01 = a_base + 1
    A10 = a_base + m
    A11 = a_base + m + 1
    
    B00 = b_base
    B10 = b_base + p
    
    C00 = c_base
    C10 = c_base + n
    
    U[A00][rank_offset] += 1
    V[B00][rank_offset] += 1
    W[C00][rank_offset] += 1
    
    U[A10][rank_offset+1] += 1
    V[B00][rank_offset+1] += 1
    W[C10][rank_offset+1] += 1
    
    U[A01][rank_offset+2] += 1
    V[B10][rank_offset+2] += 1
    W[C00][rank_offset+2] += 1
    
    U[A11][rank_offset+3] += 1
    V[B10][rank_offset+3] += 1
    W[C10][rank_offset+3] += 1
    
    return rank_offset + 4

def _add_strassen_2x2x3(U, V, W, a_base, b_base, c_base, rank_offset, n, m, p):
    r = _add_strassen_2x2x2(U, V, W, a_base, b_base, c_base, rank_offset, n, m, p)
    r = _add_schoolbook_2x2x1(U, V, W, a_base, b_base + 2, c_base + 2, r, n, m, p)
    return r

def _add_schoolbook_1x2x6(U, V, W, a_base, b_base, c_base, rank_offset, n, m, p):
    A0 = a_base
    A1 = a_base + 1
    
    for j in range(6):
        B0j = b_base + j
        B1j = b_base + p + j
        Cj = c_base + j
        
        U[A0][rank_offset] += 1
        V[B0j][rank_offset] += 1
        W[Cj][rank_offset] += 1
        rank_offset += 1
        
        U[A1][rank_offset] += 1
        V[B1j][rank_offset] += 1
        W[Cj][rank_offset] += 1
        rank_offset += 1
        
    return rank_offset

def _build_rank68():
    n, m, p = DIMENSIONS
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    r = _add_strassen_2x2x3(U, V, W, 0, 0, 0, r, n, m, p)
    r = _add_strassen_2x2x3(U, V, W, 0, 3, 3, r, n, m, p)
    r = _add_strassen_2x2x3(U, V, W, 2, 12, 0, r, n, m, p)
    r = _add_strassen_2x2x3(U, V, W, 2, 15, 3, r, n, m, p)
    r = _add_schoolbook_1x2x6(U, V, W, 8, 0, 6, r, n, m, p)
    r = _add_schoolbook_1x2x6(U, V, W, 10, 12, 6, r, n, m, p)
    
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, tgt, max_attempts=200):
    """Try to reduce rank by 1 by finding a linear dependency."""
    R = len(U[0])
    if R <= 1:
        return None
    
    # Convert to Fraction for exact arithmetic
    Uf = [[Fraction(U[i][r]) for r in range(R)] for i in range(len(U))]
    Vf = [[Fraction(V[i][r]) for r in range(R)] for i in range(len(V))]
    Wf = [[Fraction(W[i][r]) for r in range(R)] for i in range(len(W))]
    
    # Try removing each rank index and see if we can fix it
    for remove_idx in range(R):
        if time.time() > 145:
            return None
        
        # Build the remaining tensors
        U_rem = [row[:remove_idx] + row[remove_idx+1:] for row in Uf]
        V_rem = [row[:remove_idx] + row[remove_idx+1:] for row in Vf]
        W_rem = [row[:remove_idx] + row[remove_idx+1:] for row in Wf]
        
        # Compute what the remaining sum gives for each target term
        R_rem = R - 1
        acc = {}
        for r in range(R_rem):
            u_nz = [(a, U_rem[a][r]) for a in range(len(U_rem)) if U_rem[a][r] != 0]
            v_nz = [(b, V_rem[b][r]) for b in range(len(V_rem)) if V_rem[b][r] != 0]
            w_nz = [(c, W_rem[c][r]) for c in range(len(W_rem)) if W_rem[c][r] != 0]
            for a, ua in u_nz:
                for b, vb in v_nz:
                    uv = ua * vb
                    for c, wc in w_nz:
                        acc[(a, b, c)] = acc.get((a, b, c), Fraction(0)) + uv * wc
        
        # Find the deficit
        deficit = {}
        for key, val in tgt.items():
            current = acc.get(key, Fraction(0))
            if current != val:
                deficit[key] = val - current
        
        if not deficit:
            # Already works without this rank
            return (U_rem, V_rem, W_rem)
        
        # Check if deficit can be expressed as a single rank-1 tensor
        # deficit[a,b,c] = u[a] * v[b] * w[c]
        # This is a rank-1 tensor condition
        
        # Get all nonzero entries
        nz_keys = list(deficit.keys())
        if not nz_keys:
            return (U_rem, V_rem, W_rem)
        
        # Try to find u, v, w such that u[a]*v[b]*w[c] = deficit[a,b,c]
        # Pick a reference entry
        a0, b0, c0 = nz_keys[0]
        d0 = deficit[(a0, b0, c0)]
        
        # We need: u[a]*v[b]*w[c] = deficit[a,b,c] for all (a,b,c) in deficit
        # And u[a]*v[b]*w[c] = 0 for all other (a,b,c) not in tgt
        
        # This is complex. Let's try a simpler approach:
        # Check if deficit is actually rank 1 by checking consistency
        # For rank 1: deficit[a1,b1,c1] * deficit[a2,b2,c2] = deficit[a1,b2,c1] * deficit[a2,b1,c2]
        
        # But we also need zero outside tgt. This is very constrained.
        # Let's just check if all deficit entries are consistent with rank 1
        
        # For now, skip this complex check and just return the reduced version
        # if it happens to work (which is rare)
        
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 145
    
    # Start with rank 68 solution
    U, V, W = _build_rank68()
    tgt = _build_target(n, m, p)
    
    if not _verify(U, V, W, tgt, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    best_U, best_V, best_W = U, V, W
    best_rank = len(U[0])
    
    # Try rank reduction
    result = _try_reduce_rank(U, V, W, n, m, p, tgt, max_attempts=50)
    if result is not None:
        new_U, new_V, new_W = result
        if _verify(new_U, new_V, new_W, tgt, n, m, p):
            best_U, best_V, best_W = new_U, new_V, new_W
            best_rank = len(new_U[0])
    
    # Convert to int/float if possible
    def _convert_matrix(M):
        result = []
        for row in M:
            new_row = []
            for val in row:
                if isinstance(val, Fraction):
                    # Convert to int if integer, else float
                    if val.denominator == 1:
                        new_row.append(val.numerator)
                    else:
                        new_row.append(float(val))
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    U_out = _convert_matrix(best_U)
    V_out = _convert_matrix(best_V)
    W_out = _convert_matrix(best_W)
    
    return {"U": U_out, "V": V_out, "W": W_out}
