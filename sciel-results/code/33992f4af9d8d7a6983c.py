"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import fractions

DIMENSIONS = [3, 3, 4]
Frac = fractions.Fraction

# EVOLVE-BLOCK-START
def _validate(U, V, W, n, m, p, rank):
    """Exact check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    Uf = [[Frac(U[i][c]) for c in range(rank)] for i in range(n * m)]
    Vf = [[Frac(V[j][c]) for c in range(rank)] for j in range(m * p)]
    Wf = [[Frac(W[i][c]) for c in range(rank)] for i in range(p * n)]
    for i in range(n):
        for j in range(m):
            ui = Uf[i * m + j]
            for k in range(p):
                vj = Vf[j * p + k]
                wk = Wf[k * n + i]
                s = Frac(0)
                for c in range(rank):
                    s += ui[c] * vj[c] * wk[c]
                if s != Frac(1):
                    return False
    return True

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

def _gauss_solve(rows, rhs, rank):
    """
    Solve the system rows x W = rhs using exact Gaussian elimination.
    rows: list of lists of Fractions, each of length rank
    rhs: list of Fractions
    Returns W (list of Fractions of length rank) if consistent, else None.
    """
    aug = [row[:] + [rhs[i]] for i, row in enumerate(rows)]
    num_rows = len(aug)
    
    pivot_row = 0
    for col in range(rank):
        pivot = -1
        for r in range(pivot_row, num_rows):
            if aug[r][col] != 0:
                pivot = r
                break
        if pivot == -1:
            continue
        
        aug[pivot_row], aug[pivot] = aug[pivot], aug[pivot_row]
        
        pv = aug[pivot_row][col]
        for c in range(col, rank + 1):
            aug[pivot_row][c] /= pv
        
        for r in range(num_rows):
            if r != pivot_row and aug[r][col] != 0:
                factor = aug[r][col]
                for c in range(col, rank + 1):
                    aug[r][c] -= factor * aug[pivot_row][c]
        
        pivot_row += 1
        if pivot_row == num_rows:
            break
    
    for r in range(num_rows):
        is_zero_row = all(aug[r][c] == 0 for c in range(rank))
        if is_zero_row and aug[r][rank] != 0:
            return None
    
    W = [Frac(0)] * rank
    for r in range(pivot_row):
        for c in range(rank):
            if aug[r][c] != 0:
                W[c] = aug[r][rank]
                break
    return W

def _search_rank(rank, n, m, p, deadline, rng):
    """
    Try to find an exact rank-`rank` decomposition.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    while time.time() < deadline:
        U = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(nm)]
        V = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(mp)]
        
        W = [[0] * rank for _ in range(pn)]
        ok = True
        
        for i in range(n):
            for k in range(p):
                rows = []
                rhs = []
                for j in range(m):
                    for l in range(p):
                        row = [Frac(U[i * m + j][c] * V[l * p + k][c]) for c in range(rank)]
                        rows.append(row)
                        rhs.append(Frac(1 if j == l else 0))
                
                w_vec = _gauss_solve(rows, rhs, rank)
                if w_vec is None:
                    ok = False
                    break
                W[k * n + i] = [w if isinstance(w, int) else (w.numerator // w.denominator if w.denominator != 1 and w % 1 == 0 else w) for w in w_vec]
                # Simplify conversion: just keep as Fraction or convert to int if possible
                W[k * n + i] = [int(w) if w.denominator == 1 else w for w in w_vec]
            
            if not ok:
                break
        
        if ok:
            if _validate(U, V, W, n, m, p, rank):
                return U, V, W
        
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    schoolbook = _schoolbook()
    
    rng = random.Random(42)
    
    # Try rank 30
    result = _search_rank(30, n, m, p, start_time + 5, rng)
    if result:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Try rank 25
    result = _search_rank(25, n, m, p, start_time + 10, rng)
    if result:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    return schoolbook

# EVOLVE-BLOCK-END
