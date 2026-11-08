"""Search for rank-minimal exact tensor decomposition of 2x3 @ 3x8 matrix multiply."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 130  # safety margin under 150s

    n, m, p = DIMENSIONS
    # tensor: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # W index: k*n + i

    # Schoolbook rank = n*m*p = 48
    def schoolbook():
        rank = n * m * p
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
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    # Entry set for search: small half-integers
    # We'll search for rank r from lower bound upward.
    # Lower bound: max(n*m, m*p, p*n) = max(6,24,16)=24, but also need rank >= max dimension of tensor support considerations.
    # Actually a simple lower bound: rank >= max(n*m, m*p, p*n) is not always valid for tensor rank.
    # Use rank >= max(n*m, m*p, p*n) = 24 as a heuristic lower bound to start search.
    # But true rank could be lower. Let's start from a modest value and go up.
    # To be safe, start search from rank 24 upward, but also keep schoolbook as fallback.

    # We'll try to find a decomposition with rank r using random initialization + local search.
    # Represent entries as integers (multiply by 2 to use half-integers: entries are multiples of 1/2).
    # Work in integer space: entries are integers, final values divided by 2.
    # Then U[i,j,c]*V[j,k,c]*W[k,i,c] summed over c, each product is integer, sum must equal 2 (since T=1, scaled by 2^3=8? No.)
    # Let's think: if U,V,W entries are multiples of 1/2, write U = U'/2 etc. Then sum U' V' W' / 8 = 1 => sum U'V'W' = 8.
    # So work with integer matrices U',V',W' and require sum_c U'[i,j,c]*V'[j,k,c]*W'[k,i,c] = 8 for all (i,j,k).

    # Search space for integer entries: small integers, e.g. -4..4.
    # This is a huge search. Use random local search.

    def verify(U, V, W):
        """U,V,W are lists of lists of ints (scaled by 2). Check sum = 8 for all (i,j,k)."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 8:
                        return False
        return True

    def rand_int(lo, hi):
        return random.randint(lo, hi)

    # Local search: random init, then try to fix errors by changing single entries.
    # Error for (i,j,k) = sum - 8. We want all errors 0.
    # Objective: sum of squares of errors (or sum of abs).

    def objective(U, V, W):
        rank = len(U[0])
        total = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    e = s - 8
                    total += e * e
        return total

    # Try to find decomposition for a given rank r.
    def try_rank(r, max_iters=5000, entry_range=(-3, 3)):
        U = [[rand_int(entry_range[0], entry_range[1]) for _ in range(r)] for _ in range(n*m)]
        V = [[rand_int(entry_range[0], entry_range[1]) for _ in range(r)] for _ in range(m*p)]
        W = [[rand_int(entry_range[0], entry_range[1]) for _ in range(r)] for _ in range(p*n)]
        cur = objective(U, V, W)
        if cur == 0:
            return U, V, W
        best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
        best_obj = cur
        it = 0
        while it < max_iters:
            it += 1
            # Pick a random matrix, row, col
            mat = random.choice([0, 1, 2])
            if mat == 0:
                row = random.randrange(n*m)
                col = random.randrange(r)
                old = U[row][col]
                new = rand_int(entry_range[0], entry_range[1])
                if new == old:
                    continue
                U[row][col] = new
                new_obj = objective(U, V, W)
                if new_obj <= best_obj:
                    best_obj = new_obj
                    best_U, best_V, best_W = [row2[:] for row2 in U], [row2[:] for row2 in V], [row2[:] for row2 in W]
                    if new_obj == 0:
                        return U, V, W
                else:
                    U[row][col] = old
            elif mat == 1:
                row = random.randrange(m*p)
                col = random.randrange(r)
                old = V[row][col]
                new = rand_int(entry_range[0], entry_range[1])
                if new == old:
                    continue
                V[row][col] = new
                new_obj = objective(U, V, W)
                if new_obj <= best_obj:
                    best_obj = new_obj
                    best_U, best_V, best_W = [row2[:] for row2 in U], [row2[:] for row2 in V], [row2[:] for row2 in W]
                    if new_obj == 0:
                        return U, V, W
                else:
                    V[row][col] = old
            else:
                row = random.randrange(p*n)
                col = random.randrange(r)
                old = W[row][col]
                new = rand_int(entry_range[0], entry_range[1])
                if new == old:
                    continue
                W[row][col] = new
                new_obj = objective(U, V, W)
                if new_obj <= best_obj:
                    best_obj = new_obj
                    best_U, best_V, best_W = [row2[:] for row2 in U], [row2[:] for row2 in V], [row2[:] for row2 in W]
                    if new_obj == 0:
                        return U, V, W
                else:
                    W[row][col] = old
            if it % 500 == 0 and time.time() > deadline:
                break
        return None

    # Try ranks from a lower bound up to schoolbook
    # Lower bound heuristic: rank >= max(n*m, m*p, p*n) = 24, but let's try from 20 to be safe.
    # Actually, tensor rank lower bound: rank >= max(n*m, m*p, p*n) is NOT generally true.
    # For matrix multiplication, rank >= max(n*m, m*p, p*n) is a known lower bound? No, that's for bilinear complexity which is tensor rank.
    # Actually bilinear complexity = tensor rank, and lower bound is max(n*m, m*p, p*n) is not standard.
    # Let's just try from a modest value. To save time, try ranks 24..47 and also some lower.
    # Given time constraints, try a few ranks.

    for r in range(24, best_rank):
        if time.time() > deadline:
            break
        result = try_rank(r, max_iters=3000)
        if result is not None:
            U, V, W = result
            # Convert from integer (scaled by 2) to half-integer
            Uf = [[x / 2 for x in row] for row in U]
            Vf = [[x / 2 for x in row] for row in V]
            Wf = [[x / 2 for x in row] for row in W]
            # Verify with fractions
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = Fraction(0)
                        for c in range(r):
                            s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c]) / 8
                        if s != 1:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                best = {"U": Uf, "V": Vf, "W": Wf}
                best_rank = r
                break  # found minimal in this range; stop

    return best
# EVOLVE-BLOCK-END
