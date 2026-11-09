"""Matrix multiplication 2x5x5 with Strassen-based rank-46 decomposition and enhanced rank reduction."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

DEADLINE = 135.0

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if total != 1:
                    return False
    return True

def _strassen_decomposition(n, m, p):
    """Build the Strassen-based decomposition starting from rank 46."""
    rank = 46
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    c = 0
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal c
        x00 = i_off * m + j_off
        x01 = i_off * m + j_off + 1
        x10 = (i_off+1) * m + j_off
        x11 = (i_off+1) * m + j_off + 1
        
        y00 = j_off * p + k_off
        y01 = j_off * p + k_off + 1
        y10 = (j_off+1) * p + k_off
        y11 = (j_off+1) * p + k_off + 1
        
        z00 = k_off * n + i_off
        z01 = (k_off+1) * n + i_off
        z10 = k_off * n + (i_off+1)
        z11 = (k_off+1) * n + (i_off+1)
        
        U[x00][c] += 1; U[x11][c] += 1
        V[y00][c] += 1; V[y11][c] += 1
        W[z00][c] += 1; W[z11][c] += 1
        c += 1
        
        U[x10][c] += 1; U[x11][c] += 1
        V[y00][c] += 1
        W[z10][c] += 1; W[z11][c] -= 1
        c += 1
        
        U[x00][c] += 1
        V[y01][c] += 1; V[y11][c] -= 1
        W[z01][c] += 1; W[z11][c] += 1
        c += 1
        
        U[x11][c] += 1
        V[y10][c] += 1; V[y00][c] -= 1
        W[z00][c] += 1; W[z10][c] += 1
        c += 1
        
        U[x00][c] += 1; U[x01][c] += 1
        V[y11][c] += 1
        W[z00][c] -= 1; W[z01][c] += 1
        c += 1
        
        U[x10][c] += 1; U[x00][c] -= 1
        V[y00][c] += 1; V[y01][c] += 1
        W[z11][c] += 1
        c += 1
        
        U[x01][c] += 1; U[x11][c] -= 1
        V[y10][c] += 1; V[y11][c] += 1
        W[z00][c] += 1
        c += 1
    
    def add_schoolbook_2x2_x_2x1(i_off, j_off, k_off):
        nonlocal c
        for ii in range(2):
            for jj in range(2):
                U[(i_off+ii)*m + (j_off+jj)][c] += 1
                V[(j_off+jj)*p + k_off][c] += 1
                W[k_off*n + (i_off+ii)][c] += 1
                c += 1
    
    def add_schoolbook_2x1_x_1x2(i_off, j_off, k_off):
        nonlocal c
        for ii in range(2):
            for kk in range(2):
                U[(i_off+ii)*m + j_off][c] += 1
                V[j_off*p + (k_off+kk)][c] += 1
                W[(k_off+kk)*n + (i_off+ii)][c] += 1
                c += 1
    
    def add_schoolbook_2x1_x_1x1(i_off, j_off, k_off):
        nonlocal c
        for ii in range(2):
            U[(i_off+ii)*m + j_off][c] += 1
            V[j_off*p + k_off][c] += 1
            W[k_off*n + (i_off+ii)][c] += 1
            c += 1
    
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_schoolbook_2x2_x_2x1(0, 0, 4)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_schoolbook_2x2_x_2x1(0, 2, 4)
    add_schoolbook_2x1_x_1x2(0, 4, 0)
    add_schoolbook_2x1_x_1x2(0, 4, 2)
    add_schoolbook_2x1_x_1x1(0, 4, 4)
    
    return U, V, W

def _try_merge_columns(U, V, W, n, m, p, c1, c2, factor):
    """Try to merge columns c1 and c2 by adding factor * c2 to c1 and removing c2."""
    rank = len(U[0])
    if c1 == c2:
        return None
    
    new_rank = rank - 1
    new_U = [[0] * new_rank for _ in range(n * m)]
    new_V = [[0] * new_rank for _ in range(m * p)]
    new_W = [[0] * new_rank for _ in range(p * n)]
    
    new_col = 0
    for c in range(rank):
        if c == c2:
            continue
        if c == c1:
            for i in range(n * m):
                new_U[i][new_col] = U[i][c1] + factor * U[i][c2]
            for j in range(m * p):
                new_V[j][new_col] = V[j][c1] + factor * V[j][c2]
            for k in range(p * n):
                new_W[k][new_col] = W[k][c1] + factor * W[k][c2]
        else:
            for i in range(n * m):
                new_U[i][new_col] = U[i][c]
            for j in range(m * p):
                new_V[j][new_col] = V[j][c]
            for k in range(p * n):
                new_W[k][new_col] = W[k][c]
        new_col += 1
    
    if _verify(new_U, new_V, new_W, n, m, p):
        return new_U, new_V, new_W
    return None

def _search_lower_rank_fast(U, V, W, n, m, p, target_rank, max_time=30.0):
    """Try to find a decomposition of rank target_rank using greedy merging."""
    current_rank = len(U[0])
    if current_rank < target_rank:
        return U, V, W
    
    start_time = time.time()
    
    # Expanded set of factors to try
    factors_to_try = [
        Fraction(1), Fraction(-1), 
        Fraction(1,2), Fraction(-1,2), 
        Fraction(1,3), Fraction(-1,3),
        Fraction(2), Fraction(-2),
        Fraction(1,4), Fraction(-1,4),
        Fraction(3), Fraction(-3)
    ]
    
    while current_rank > target_rank:
        if time.time() - start_time > max_time:
            break
            
        merged = False
        # Try all pairs
        for c1 in range(current_rank):
            if merged or time.time() - start_time > max_time:
                break
            for c2 in range(c1 + 1, current_rank):
                if merged or time.time() - start_time > max_time:
                    break
                for factor in factors_to_try:
                    if time.time() - start_time > max_time:
                        break
                    result = _try_merge_columns(U, V, W, n, m, p, c1, c2, factor)
                    if result is not None:
                        U, V, W = result
                        current_rank -= 1
                        merged = True
                        break
        
        if not merged:
            break
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    U, V, W = _strassen_decomposition(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = len(U[0])
    
    start_time = time.time()
    
    for target_rank in range(best_rank - 1, 0, -1):
        if time.time() > DEADLINE:
            break
        
        time_budget = min(25.0, (DEADLINE - time.time()) * 0.5)
        if time_budget < 5.0:
            break
            
        result = _search_lower_rank_fast(U, V, W, n, m, p, target_rank, max_time=time_budget)
        if result is not None:
            new_rank = len(result[0][0])
            if new_rank < best_rank:
                best_U, best_V, best_W = result
                best_rank = new_rank
                U, V, W = result
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
