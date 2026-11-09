import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 3]
TIME_LIMIT = 115.0
START_TIME = time.time()


def elapsed():
    return time.time() - START_TIME


def deadline_exceeded():
    return elapsed() > TIME_LIMIT


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


def _verify_exact(U, V, W, n, m, p):
    R = len(U[0])
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                ui = Uf[i * m + j]
                vk = Vf[j * p + k]
                wk = Wf[k * n + i]
                for r in range(R):
                    s += ui[r] * vk[r] * wk[r]
                if s != 1:
                    return False
    return True


def _construct_rank17(n, m, p):
    if n != 2 or m != 3 or p != 3:
        return None

    R = 17
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    U[u_idx(0,0)][0] = 1; U[u_idx(1,1)][0] = 1
    V[v_idx(0,0)][0] = 1; V[v_idx(1,1)][0] = 1
    W[w_idx(0,0)][0] = 1; W[w_idx(1,1)][0] = 1
    
    U[u_idx(1,0)][1] = 1; U[u_idx(1,1)][1] = 1
    V[v_idx(0,0)][1] = 1
    W[w_idx(0,1)][1] = 1; W[w_idx(1,1)][1] = -1
    
    U[u_idx(0,0)][2] = 1
    V[v_idx(0,1)][2] = 1; V[v_idx(1,1)][2] = -1
    W[w_idx(1,0)][2] = 1; W[w_idx(1,1)][2] = 1
    
    U[u_idx(1,1)][3] = 1
    V[v_idx(1,0)][3] = 1; V[v_idx(0,0)][3] = -1
    W[w_idx(0,0)][3] = 1; W[w_idx(0,1)][3] = 1
    
    U[u_idx(0,0)][4] = 1; U[u_idx(0,1)][4] = 1
    V[v_idx(1,1)][4] = 1
    W[w_idx(0,0)][4] = -1; W[w_idx(1,0)][4] = 1
    
    U[u_idx(1,0)][5] = 1; U[u_idx(0,0)][5] = -1
    V[v_idx(0,0)][5] = 1; V[v_idx(0,1)][5] = 1
    W[w_idx(1,1)][5] = 1
    
    U[u_idx(0,1)][6] = 1; U[u_idx(1,1)][6] = -1
    V[v_idx(1,0)][6] = 1; V[v_idx(1,1)][6] = 1
    W[w_idx(0,0)][6] = 1

    boundary_terms = [
        (0, 2, 0), (1, 2, 0),
        (0, 2, 1), (1, 2, 1),
        (0, 2, 2), (1, 2, 2),
        (0, 0, 2), (1, 0, 2),
        (0, 1, 2), (1, 1, 2)
    ]
    
    for idx, (i, j, k) in enumerate(boundary_terms):
        r = 7 + idx
        U[u_idx(i, j)][r] = 1
        V[v_idx(j, k)][r] = 1
        W[w_idx(k, i)][r] = 1

    return {"U": U, "V": V, "W": W}


def _gauss_solve(A, b):
    num_eqs = len(A)
    num_unknowns = len(b)
    
    aug = [A[i][:] + [b[i]] for i in range(num_eqs)]
    
    pivot_cols = []
    rank = 0
    
    for col in range(num_unknowns):
        if rank >= num_eqs:
            break
        pivot_row = None
        for row in range(rank, num_eqs):
            if aug[row][col] != 0:
                pivot_row = row
                break
        if pivot_row is None:
            continue
        aug[rank], aug[pivot_row] = aug[pivot_row], aug[rank]
        pivot_val = aug[rank][col]
        for row in range(num_eqs):
            if row != rank and aug[row][col] != 0:
                factor = aug[row][col] / pivot_val
                for c in range(col, num_unknowns + 1):
                    aug[row][c] -= factor * aug[rank][c]
        pivot_cols.append(col)
        rank += 1
    
    for row in range(rank, num_eqs):
        if aug[row][num_unknowns] != 0:
            return None
    
    x = [Fraction(0)] * num_unknowns
    for i, col in enumerate(pivot_cols):
        x[col] = aug[i][num_unknowns] / aug[i][col]
    
    return x


def _try_eliminate_tensor(U, V, W, n, m, p, elim_idx):
    """Try to eliminate tensor at elim_idx by expressing it as combo of others."""
    R = len(U[0])
    if R <= 1:
        return None
    
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    
    num_eqs = n * m * p
    num_unknowns = R - 1
    
    A = []
    b = []
    unknown_to_original = []
    
    for idx, (i, j, k) in enumerate(product(range(n), range(m), range(p))):
        row = []
        for r in range(R):
            if r == elim_idx:
                continue
            unknown_to_original.append(r)
            row.append(Uf[i*m+j][r] * Vf[j*p+k][r] * Wf[k*n+i][r])
        A.append(row)
        b.append(Uf[i*m+j][elim_idx] * Vf[j*p+k][elim_idx] * Wf[k*n+i][elim_idx])
    
    try:
        coeffs = _gauss_solve(A, b)
    except (ZeroDivisionError, IndexError):
        return None
    
    if coeffs is None:
        return None
    
    # Verify
    for idx, (i, j, k) in enumerate(product(range(n), range(m), range(p))):
        lhs = Fraction(0)
        for r in range(num_unknowns):
            orig_r = unknown_to_original[r]
            lhs += coeffs[r] * Uf[i*m+j][orig_r] * Vf[j*p+k][orig_r] * Wf[k*n+i][orig_r]
        rhs = Uf[i*m+j][elim_idx] * Vf[j*p+k][elim_idx] * Wf[k*n+i][elim_idx]
        if lhs != rhs:
            return None
    
    # Apply elimination
    new_U = [row[:] for row in Uf]
    new_V = [row[:] for row in Vf]
    new_W = [row[:] for row in Wf]
    
    for r in range(num_unknowns):
        if coeffs[r] != 0:
            orig_r = unknown_to_original[r]
            for i in range(n * m):
                new_U[i][orig_r] -= coeffs[r] * new_U[i][elim_idx]
            for j in range(m * p):
                new_V[j][orig_r] -= coeffs[r] * new_V[j][elim_idx]
            for k in range(p * n):
                new_W[k][orig_r] -= coeffs[r] * new_W[k][elim_idx]
    
    # Remove eliminated tensor
    for i in range(n * m):
        new_U[i].pop(elim_idx)
    for j in range(m * p):
        new_V[j].pop(elim_idx)
    for k in range(p * n):
        new_W[k].pop(elim_idx)
    
    # Convert back
    U_out = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in new_U]
    V_out = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in new_V]
    W_out = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in new_W]
    
    return {"U": U_out, "V": V_out, "W": W_out}


def _search_rank_reduction(U, V, W, n, m, p, max_time=30.0):
    """Try to reduce rank by eliminating tensors one at a time."""
    start_time = time.time()
    current_U = [row[:] for row in U]
    current_V = [row[:] for row in V]
    current_W = [row[:] for row in W]
    current_R = len(current_U[0])
    
    while current_R > 1:
        if time.time() - start_time > max_time:
            break
        if deadline_exceeded():
            break
        
        eliminated = False
        # Try eliminating from the end (boundary terms first)
        for elim_idx in range(current_R - 1, -1, -1):
            if deadline_exceeded():
                break
            
            result = _try_eliminate_tensor(current_U, current_V, current_W, n, m, p, elim_idx)
            if result is not None:
                if _verify_exact(result["U"], result["V"], result["W"], n, m, p):
                    current_U = result["U"]
                    current_V = result["V"]
                    current_W = result["W"]
                    current_R = len(current_U[0])
                    eliminated = True
                    break
        
        if not eliminated:
            break
    
    if current_R < len(U[0]):
        return {"U": current_U, "V": current_V, "W": current_W}
    return None


def construct():
    n, m, p = DIMENSIONS
    
    result = _construct_rank17(n, m, p)
    if result is not None:
        if _verify_exact(result["U"], result["V"], result["W"], n, m, p):
            # Try to reduce rank
            reduced = _search_rank_reduction(result["U"], result["V"], result["W"], n, m, p, max_time=40.0)
            if reduced is not None:
                result = reduced
    
    if result is None:
        result = _schoolbook(n, m, p)
    
    return result


def main():
    return construct()


if __name__ == '__main__':
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
