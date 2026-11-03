import time
import numpy as np
from fractions import Fraction

N, M, P = 3, 4, 4
NM, MP, PN = N * M, M * P, P * N


def _verify(U, V, W, tol=1e-10):
    rank = len(U[0])
    U_arr = np.array(U, dtype=float).reshape(N, M, rank)
    V_arr = np.array(V, dtype=float).reshape(M, P, rank)
    W_arr = np.array(W, dtype=float).reshape(P, N, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err


def _verify_fast(U, V, W):
    rank = len(U[0])
    U_arr = np.array(U, dtype=float).reshape(N, M, rank)
    V_arr = np.array(V, dtype=float).reshape(M, P, rank)
    W_arr = np.array(W, dtype=float).reshape(P, N, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    return np.max(np.abs(T - 1.0))


def _schoolbook():
    rank = NM * MP * PN
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][c] = 1
                V[j * P + k][c] = 1
                W[k * N + i][c] = 1
                c += 1
    return {'U': U, 'V': V, 'W': W}


def _strassen_44():
    """Original Strassen-based rank-44 decomposition."""
    rank = 44
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    s_steps = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 1): 1, (0, 0): -1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1})
    ]
    
    blocks = [(0, 0, 0, 0, 0, 0), (0, 2, 2, 0, 0, 0), (0, 0, 0, 2, 0, 2), (0, 2, 2, 2, 0, 2)]
    
    col = 0
    for (xi, xj, yj, yk, zi, zk) in blocks:
        for (u_c, v_c, w_c) in s_steps:
            for (a, b), val in u_c.items():
                U[(a + xi) * M + (b + xj)][col] += val
            for (b, c), val in v_c.items():
                V[(b + yj) * P + (c + yk)][col] += val
            for (a, c), val in w_c.items():
                W[(c + zk) * N + (a + zi)][col] += val
            col += 1
    
    for j in range(M):
        for k in range(P):
            U[2 * M + j][col] = 1
            V[j * P + k][col] = 1
            W[k * N + 2][col] = 1
            col += 1
    
    assert col == rank
    return {'U': U, 'V': V, 'W': W}


def _try_reduce_by_merge(U, V, W, deadline):
    """
    Try to reduce rank by finding pairs of columns that can be merged.
    Two columns c1, c2 can be merged if U[:,c2]=a*U[:,c1], V[:,c2]=b*V[:,c1], W[:,c2]=c*W[:,c1] with a*b*c=1.
    """
    rank = len(U[0])
    best_U, best_V, best_W = U[:], V[:], W[:]
    best_rank = rank
    
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    deadline = deadline - 5.0
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        found_merge = False
        for c1 in range(best_rank - 1):
            if time.time() > deadline:
                break
            for c2 in range(c1 + 1, best_rank):
                u1 = [U_frac[i][c1] for i in range(NM)]
                u2 = [U_frac[i][c2] for i in range(NM)]
                v1 = [V_frac[i][c1] for i in range(MP)]
                v2 = [V_frac[i][c2] for i in range(MP)]
                w1 = [W_frac[i][c1] for i in range(PN)]
                w2 = [W_frac[i][c2] for i in range(PN)]
                
                a = None
                for i in range(NM):
                    if u1[i] != 0:
                        if u2[i] != 0 and a is None:
                            a = u2[i] / u1[i]
                        elif u2[i] == 0:
                            a = None
                            break
                        elif u2[i] / u1[i] != a:
                            a = None
                            break
                    elif u2[i] != 0:
                        a = None
                        break
                
                if a is None or a == 0:
                    continue
                
                b = None
                for i in range(MP):
                    if v1[i] != 0:
                        if v2[i] != 0 and b is None:
                            b = v2[i] / v1[i]
                        elif v2[i] == 0:
                            b = None
                            break
                        elif v2[i] / v1[i] != b:
                            b = None
                            break
                    elif v2[i] != 0:
                        b = None
                        break
                
                if b is None or b == 0:
                    continue
                
                c = None
                for i in range(PN):
                    if w1[i] != 0:
                        if w2[i] != 0 and c is None:
                            c = w2[i] / w1[i]
                        elif w2[i] == 0:
                            c = None
                            break
                        elif w2[i] / w1[i] != c:
                            c = None
                            break
                    elif w2[i] != 0:
                        c = None
                        break
                
                if c is None or c == 0:
                    continue
                
                if a * b * c != 1:
                    continue
                
                # Found a merge!
                for i in range(NM):
                    U_frac[i][c1] = U_frac[i][c1] + U_frac[i][c2]
                for i in range(MP):
                    V_frac[i][c1] = V_frac[i][c1] + V_frac[i][c2]
                for i in range(PN):
                    W_frac[i][c1] = W_frac[i][c1] + W_frac[i][c2]
                
                U_frac = [row[:c2] + row[c2+1:] for row in U_frac]
                V_frac = [row[:c2] + row[c2+1:] for row in V_frac]
                W_frac = [row[:c2] + row[c2+1:] for row in W_frac]
                
                best_rank = best_rank - 1
                best_U = [[float(x) for x in row] for row in U_frac]
                best_V = [[float(x) for x in row] for row in V_frac]
                best_W = [[float(x) for x in row] for row in W_frac]
                
                found_merge = True
                break
            if found_merge or time.time() > deadline:
                break
        
        if not found_merge:
            break
    
    return {'U': best_U, 'V': best_V, 'W': best_W}, best_rank


def _gaussian_reduce(U, V, W, deadline):
    """
    Try to reduce rank using Gaussian elimination over the rationals.
    We look for linear dependencies among the columns.
    
    The tensor is T = sum_c U_c ⊗ V_c ⊗ W_c.
    If sum_c alpha_c * U_c ⊗ V_c ⊗ W_c = 0, then we can reduce rank.
    
    This is a bilinear system, so it's tricky. But we can try:
    Fix two columns and see if their combination is in the span of others.
    """
    rank = len(U[0])
    best_U, best_V, best_W = U[:], V[:], W[:]
    best_rank = rank
    
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    deadline = deadline - 5.0
    
    # Check for duplicate or negated columns first (quick wins)
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        found = False
        for c1 in range(best_rank - 1):
            if time.time() > deadline:
                break
            for c2 in range(c1 + 1, best_rank):
                # Check if col c2 = -col c1 in U and V, but same in W
                u_match = all(U_frac[i][c1] + U_frac[i][c2] == 0 for i in range(NM))
                v_match = all(V_frac[i][c1] + V_frac[i][c2] == 0 for i in range(MP))
                w_match = all(W_frac[i][c1] - W_frac[i][c2] == 0 for i in range(PN))
                
                if u_match and v_match and w_match:
                    U_frac = [row[:c2] + row[c2+1:] for row in U_frac]
                    V_frac = [row[:c2] + row[c2+1:] for row in V_frac]
                    W_frac = [row[:c2] + row[c2+1:] for row in W_frac]
                    best_rank -= 1
                    best_U = [[float(x) for x in row] for row in U_frac]
                    best_V = [[float(x) for x in row] for row in V_frac]
                    best_W = [[float(x) for x in row] for row in W_frac]
                    found = True
                    break
                
                # Check if col c2 = col c1 exactly
                u_match2 = all(U_frac[i][c1] - U_frac[i][c2] == 0 for i in range(NM))
                v_match2 = all(V_frac[i][c1] - V_frac[i][c2] == 0 for i in range(MP))
                w_match2 = all(W_frac[i][c1] - W_frac[i][c2] == 0 for i in range(PN))
                
                if u_match2 and v_match2 and w_match2:
                    U_frac = [row[:c2] + row[c2+1:] for row in U_frac]
                    V_frac = [row[:c2] + row[c2+1:] for row in V_frac]
                    W_frac = [row[:c2] + row[c2+1:] for row in W_frac]
                    best_rank -= 1
                    best_U = [[float(x) for x in row] for row in U_frac]
                    best_V = [[float(x) for x in row] for row in V_frac]
                    best_W = [[float(x) for x in row] for row in W_frac]
                    found = True
                    break
            
            if found or time.time() > deadline:
                break
        
        if not found:
            break
    
    return {'U': best_U, 'V': best_V, 'W': best_W}, best_rank


def construct():
    start = time.time()
    
    # Start with the known Strassen-based decomposition with rank 44
    result = _strassen_44()
    
    # Verify correctness
    ok, err = _verify(result['U'], result['V'], result['W'])
    if not ok:
        result = _schoolbook()
        return result
    
    best_result = result
    best_rank = len(result['U'][0])
    
    # Deadline: 140 seconds (10s safety margin)
    deadline = start + 140.0
    
    # Try to reduce rank using merge heuristic
    if time.time() < deadline - 30:
        merged_result, merged_rank = _try_reduce_by_merge(result['U'], result['V'], result['W'], deadline)
        if merged_rank < best_rank:
            ok, err = _verify(merged_result['U'], merged_result['V'], merged_result['W'])
            if ok:
                best_result = merged_result
                best_rank = merged_rank
    
    # Try Gaussian reduction
    if time.time() < deadline - 10:
        reduced_result, reduced_rank = _gaussian_reduce(best_result['U'], best_result['V'], best_result['W'], deadline)
        if reduced_rank < best_rank:
            ok, err = _verify(reduced_result['U'], reduced_result['V'], reduced_result['W'])
            if ok:
                best_result = reduced_result
                best_rank = reduced_rank
    
    # Final verification
    ok, err = _verify(best_result['U'], best_result['V'], best_result['W'])
    if not ok:
        best_result = result
    
    return best_result
