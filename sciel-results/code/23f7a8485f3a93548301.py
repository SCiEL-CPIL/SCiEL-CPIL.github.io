import time
import random
import fractions
import numpy as np

DIMENSIONS = [3, 3, 4]
Frac = fractions.Fraction

def _validate(U, V, W, n, m, p, rank):
    """Exact check that the tensor decomposition holds."""
    Uf = [[Frac(U[i][c]) for c in range(rank)] for i in range(n * m)]
    Vf = [[Frac(V[j][c]) for c in range(rank)] for j in range(m * p)]
    Wf = [[Frac(W[k][c]) for c in range(rank)] for k in range(p * n)]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    s = Frac(0)
                    for c in range(rank):
                        s += Uf[i * m + j][c] * Vf[l * p + k][c] * Wf[k * n + i][c]
                    expected = Frac(1) if j == l else Frac(0)
                    if s != expected:
                        return False
    return True

def _solve_w_system_exact(U, V, i, k, n, m, p, rank):
    """Solve for W[k*n+i] given U, V. Returns list of Fractions or None if inconsistent."""
    rows = []
    rhs = []
    for j in range(m):
        for l in range(m):
            row = [Frac(U[i * m + j][c]) * Frac(V[l * p + k][c]) for c in range(rank)]
            rows.append(row)
            rhs.append(Frac(1) if j == l else Frac(0))
    
    R = len(rows)
    C = rank
    aug = [rows[r][:] + [rhs[r]] for r in range(R)]
    
    pivots = []
    col = 0
    for r in range(R):
        if col >= C:
            break
        pr = None
        for rr in range(r, R):
            if aug[rr][col] != 0:
                pr = rr
                break
        if pr is None:
            col += 1
            continue
        if pr != r:
            aug[r], aug[pr] = aug[pr], aug[r]
        pivots.append((r, col))
        pv = aug[r][col]
        if pv != 1:
            aug[r] = [x / pv for x in aug[r]]
        for rr in range(R):
            if rr != r and aug[rr][col] != 0:
                f = aug[rr][col]
                aug[rr] = [a - f * b for a, b in zip(aug[rr], aug[r])]
        col += 1
    
    for rr in range(R):
        coeff_zero = all(aug[rr][c] == 0 for c in range(C))
        if coeff_zero and aug[rr][C] != 0:
            return None
    
    Wvec = [Frac(0)] * C
    for (r, c) in pivots:
        Wvec[c] = aug[r][C]
    
    return Wvec

def _get_strassen_2x2():
    """
    Returns U, V, W for 2x2 * 2x2 Strassen decomposition.
    U is 4x7, V is 4x7, W is 4x7.
    Indices:
    U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (c-d)(f+g)
    U[2][5] = 1; U[3][5] = -1
    V[1][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # M7 = (b-c)(e+g)
    U[1][6] = 1; U[2][6] = -1
    V[0][6] = 1; V[2][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _search_with_strassen_base(n, m, p, deadline, start_rank=20, end_rank=35):
    """
    Search for decomposition by fixing the 2x2 block to Strassen
    and randomizing the rest.
    """
    sU, sV, sW = _get_strassen_2x2()
    
    for rank in range(start_rank, end_rank + 1):
        if time.time() > deadline:
            break
        
        # Number of random trials per rank
        # We can afford more trials for lower ranks if we are fast, but let's balance
        max_trials = 2000 if rank < 25 else 1000
        
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            
            # 1. Fill Strassen part for indices corresponding to 2x2 block
            # A[0:2, 0:2] corresponds to U indices 0,1,2,3 (i*m+j)
            # B[0:2, 0:2] corresponds to V indices 0,1,2,3 (j*p+k)
            # Note: p=4, so V indices for j=0 are 0,1,2,3. For j=1 are 4,5,6,7.
            # Strassen V is defined for k in 0,1. So V[j*4 + k] for k in 0,1.
            
            for i in range(2):
                for j in range(2):
                    u_idx = i * m + j
                    for c in range(7):
                        U[u_idx][c] = sU[u_idx][c]
            
            for j in range(2):
                for k in range(2):
                    v_idx = j * p + k
                    for c in range(7):
                        V[v_idx][c] = sV[v_idx][c]
            
            # 2. Randomize the rest of U and V
            # U has indices for i in 0..2, j in 0..2
            # Fixed: i in 0,1 and j in 0,1
            # Free: (i=0,j=2), (i=1,j=2), (i=2,j=0), (i=2,j=1), (i=2,j=2)
            # V has indices for j in 0..2, k in 0..3
            # Fixed: j in 0,1 and k in 0,1
            # Free: (j=0,k=2), (j=0,k=3), (j=1,k=2), (j=1,k=3), (j=2,k=0), (j=2,k=1), (j=2,k=2), (j=2,k=3)
            
            # We will randomize all free entries for columns 0..rank-1
            # For the fixed entries, we keep the Strassen values in cols 0..6 and 0 in cols 7..rank-1
            # For free entries, we randomize all cols 0..rank-1
            
            rng = random.Random(trial * 1000 + rank)
            
            # Randomize U free parts
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    if i < 2 and j < 2:
                        # Already set for c<7, set rest to 0
                        for c in range(7, rank):
                            U[u_idx][c] = 0
                    else:
                        for c in range(rank):
                            # Small integers to keep numbers manageable
                            U[u_idx][c] = rng.randint(-2, 2)
            
            # Randomize V free parts
            for j in range(m):
                for k in range(p):
                    v_idx = j * p + k
                    if j < 2 and k < 2:
                        # Already set for c<7, set rest to 0
                        for c in range(7, rank):
                            V[v_idx][c] = 0
                    else:
                        for c in range(rank):
                            V[v_idx][c] = rng.randint(-2, 2)
            
            # Check for trivial zero rows in U or V which might cause issues or be invalid
            # (Actually, zero rows in U or V are allowed as long as the tensor equation holds, 
            # but they reduce effective rank. We want exact rank, so we should avoid 
            # dependencies that allow reduction, but for now we just solve.)
            
            # 3. Solve for W
            W = [[0] * rank for _ in range(p * n)]
            ok = True
            for i in range(n):
                for k in range(p):
                    Wvec = _solve_w_system_exact(U, V, i, k, n, m, p, rank)
                    if Wvec is None:
                        ok = False
                        break
                    # Convert to int if possible, else keep Fraction
                    # The problem allows half-integers. 
                    # We will store as Fraction or Int.
                    W[k * n + i] = [int(x) if x.denominator == 1 else x for x in Wvec]
                if not ok:
                    break
            
            if not ok:
                continue
            
            # 4. Validate
            if _validate(U, V, W, n, m, p, rank):
                return {"U": U, "V": V, "W": W}
                
    return None

def _schoolbook(n, m, p):
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_result = _schoolbook(n, m, p)
    
    # Try the structured search based on Strassen
    try:
        result = _search_with_strassen_base(n, m, p, deadline)
        if result is not None:
            best_result = result
    except Exception:
        pass
    
    return best_result
