"""Search for rank-minimal exact tensor decomposition of 3x5 x 5x5 matrix multiplication."""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 5]
n, m, p = DIMENSIONS
N = n * m  # 15
M = m * p  # 25
P = p * n  # 15
NUM_EQUATIONS = n * m * p  # 75

def _verify(U, V, W, rank):
    """Check if the decomposition is exact. U,V,W are lists of lists of Fraction."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                s = Fraction(0)
                for r in range(rank):
                    s += U[u_row][r] * V[v_row][r] * W[w_row][r]
                if s != 1:
                    return False
    return True

def _all_int_half(U, V, W):
    for mat in (U, V, W):
        for row in mat:
            for v in row:
                if v.denominator not in (1, 2):
                    return False
    return True

def _solve_for_W(U, V, rank):
    """Given U and V, solve for W such that tensor = 1. Returns W or None if no solution."""
    W_unknowns = P * rank
    A = [[Fraction(0) for _ in range(W_unknowns)] for _ in range(NUM_EQUATIONS)]
    b = [Fraction(1) for _ in range(NUM_EQUATIONS)]
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                w_row = k * n + i
                for r in range(rank):
                    coeff = U[i * m + j][r] * V[j * p + k][r]
                    if coeff != 0:
                        A[eq_idx][w_row * rank + r] = coeff
                eq_idx += 1
    aug = [A[e] + [b[e]] for e in range(NUM_EQUATIONS)]
    rows = NUM_EQUATIONS
    cols = W_unknowns
    row = 0
    for col in range(cols):
        if row >= rows:
            break
        pivot = -1
        for r in range(row, rows):
            if aug[r][col] != 0:
                pivot = r
                break
        if pivot == -1:
            continue
        aug[row], aug[pivot] = aug[pivot], aug[row]
        pv = aug[row][col]
        for c in range(col, cols + 1):
            aug[row][c] /= pv
        for r in range(rows):
            if r != row and aug[r][col] != 0:
                factor = aug[r][col]
                for c in range(col, cols + 1):
                    aug[r][c] -= factor * aug[row][c]
        row += 1
    for r in range(row, rows):
        all_zero = all(aug[r][c] == 0 for c in range(cols))
        if not all_zero:
            return None
    x = [Fraction(0) for _ in range(cols)]
    for r in range(row):
        for c in range(cols):
            if aug[r][c] != 0:
                x[c] = aug[r][cols]
                break
    W = [[x[w_row * rank + r] for r in range(rank)] for w_row in range(P)]
    return W

def _solve_for_V(U, W, rank):
    """Given U and W, solve for V."""
    V_unknowns = M * rank
    A = [[Fraction(0) for _ in range(V_unknowns)] for _ in range(NUM_EQUATIONS)]
    b = [Fraction(1) for _ in range(NUM_EQUATIONS)]
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                v_row = j * p + k
                for r in range(rank):
                    coeff = U[i * m + j][r] * W[k * n + i][r]
                    if coeff != 0:
                        A[eq_idx][v_row * rank + r] = coeff
                eq_idx += 1
    aug = [A[e] + [b[e]] for e in range(NUM_EQUATIONS)]
    rows = NUM_EQUATIONS
    cols = V_unknowns
    row = 0
    for col in range(cols):
        if row >= rows:
            break
        pivot = -1
        for r in range(row, rows):
            if aug[r][col] != 0:
                pivot = r
                break
        if pivot == -1:
            continue
        aug[row], aug[pivot] = aug[pivot], aug[row]
        pv = aug[row][col]
        for c in range(col, cols + 1):
            aug[row][c] /= pv
        for r in range(rows):
            if r != row and aug[r][col] != 0:
                factor = aug[r][col]
                for c in range(col, cols + 1):
                    aug[r][c] -= factor * aug[row][c]
        row += 1
    for r in range(row, rows):
        all_zero = all(aug[r][c] == 0 for c in range(cols))
        if not all_zero:
            return None
    x = [Fraction(0) for _ in range(cols)]
    for r in range(row):
        for c in range(cols):
            if aug[r][c] != 0:
                x[c] = aug[r][cols]
                break
    V = [[x[v_row * rank + r] for r in range(rank)] for v_row in range(M)]
    return V

def _solve_for_U(V, W, rank):
    """Given V and W, solve for U."""
    U_unknowns = N * rank
    A = [[Fraction(0) for _ in range(U_unknowns)] for _ in range(NUM_EQUATIONS)]
    b = [Fraction(1) for _ in range(NUM_EQUATIONS)]
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i * m + j
                for r in range(rank):
                    coeff = V[j * p + k][r] * W[k * n + i][r]
                    if coeff != 0:
                        A[eq_idx][u_row * rank + r] = coeff
                eq_idx += 1
    aug = [A[e] + [b[e]] for e in range(NUM_EQUATIONS)]
    rows = NUM_EQUATIONS
    cols = U_unknowns
    row = 0
    for col in range(cols):
        if row >= rows:
            break
        pivot = -1
        for r in range(row, rows):
            if aug[r][col] != 0:
                pivot = r
                break
        if pivot == -1:
            continue
        aug[row], aug[pivot] = aug[pivot], aug[row]
        pv = aug[row][col]
        for c in range(col, cols + 1):
            aug[row][c] /= pv
        for r in range(rows):
            if r != row and aug[r][col] != 0:
                factor = aug[r][col]
                for c in range(col, cols + 1):
                    aug[r][c] -= factor * aug[row][c]
        row += 1
    for r in range(row, rows):
        all_zero = all(aug[r][c] == 0 for c in range(cols))
        if not all_zero:
            return None
    x = [Fraction(0) for _ in range(cols)]
    for r in range(row):
        for c in range(cols):
            if aug[r][c] != 0:
                x[c] = aug[r][cols]
                break
    U = [[x[u_row * rank + r] for r in range(rank)] for u_row in range(N)]
    return U

def _random_int_half():
    val = random.randint(-3, 3)
    if random.random() < 0.5:
        val = Fraction(val, 2)
    else:
        val = Fraction(val)
    return val

def _construct_random(rank):
    U = [[_random_int_half() for _ in range(rank)] for _ in range(N)]
    V = [[_random_int_half() for _ in range(rank)] for _ in range(M)]
    W = [[_random_int_half() for _ in range(rank)] for _ in range(P)]
    return U, V, W

def _refine(U, V, W, rank, max_iters=40):
    for it in range(max_iters):
        W_new = _solve_for_W(U, V, rank)
        if W_new is None:
            return None
        V_new = _solve_for_V(U, W_new, rank)
        if V_new is None:
            return None
        U_new = _solve_for_U(V_new, W_new, rank)
        if U_new is None:
            return None
        U, V, W = U_new, V_new, W_new
        if _verify(U, V, W, rank):
            return U, V, W
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145

    best_rank = n * m * p
    best_U = best_V = best_W = None

    # Schoolbook fallback
    rank_sb = best_rank
    U_sb = [[Fraction(0) for _ in range(rank_sb)] for _ in range(N)]
    V_sb = [[Fraction(0) for _ in range(rank_sb)] for _ in range(M)]
    W_sb = [[Fraction(0) for _ in range(rank_sb)] for _ in range(P)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = Fraction(1)
                V_sb[j * p + k][col] = Fraction(1)
                W_sb[k * n + i][col] = Fraction(1)
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Search for lower rank
    for target_rank in range(35, 75):
        if time.time() > deadline:
            break
        found = False
        for attempt in range(15):
            if time.time() > deadline:
                break
            U, V, W = _construct_random(target_rank)
            result = _refine(U, V, W, target_rank, max_iters=30)
            if result is not None:
                U_r, V_r, W_r = result
                if _verify(U_r, V_r, W_r, target_rank):
                    if _all_int_half(U_r, V_r, W_r):
                        if target_rank < best_rank:
                            best_rank = target_rank
                            best_U, best_V, best_W = U_r, V_r, W_r
                        found = True
                        break
        if not found:
            continue

    U_out = [[float(v) for v in row] for row in best_U]
    V_out = [[float(v) for v in row] for row in best_V]
    W_out = [[float(v) for v in row] for row in best_W]
    return {"U": U_out, "V": V_out, "W": W_out}
