"""Combined program: Strassen-based decomposition with systematic rank reduction."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import itertools
from fractions import Fraction

DEADLINE = 135.0  # safety margin before 150s kill

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    # Check dimensions
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matrix multiplication, T[i,j,k] should be 1 if we're computing (AB)[i,k] = sum_j A[i,j]*B[j,k]
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
    U = [[0]*rank for _ in range(n*m)]  # 10 x 46
    V = [[0]*rank for _ in range(m*p)]  # 25 x 46
    W = [[0]*rank for _ in range(p*n)]  # 10 x 46
    
    c = 0  # current column index
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal c
        # X = A[i_off:i_off+2, j_off:j_off+2]
        # Y = B[j_off:j_off+2, k_off:k_off+2]
        # Z = C[i_off:i_off+2, k_off:k_off+2]
        
        # Strassen factors:
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y01-Y11)
        # M4 = X11(Y10-Y00)
        # M5 = (X00+X01)Y11
        # M6 = (X10-X00)(Y00+Y01)
        # M7 = (X01-X11)(Y10+Y11)
        
        # Output:
        # Z00 = M1+M4-M5+M7
        # Z01 = M3+M5
        # Z10 = M2+M4
        # Z11 = M1-M2+M3+M6
        
        # U indices for X:
        x00 = i_off * m + j_off
        x01 = i_off * m + j_off + 1
        x10 = (i_off+1) * m + j_off
        x11 = (i_off+1) * m + j_off + 1
        
        # V indices for Y:
        y00 = j_off * p + k_off
        y01 = j_off * p + k_off + 1
        y10 = (j_off+1) * p + k_off
        y11 = (j_off+1) * p + k_off + 1
        
        # W indices for Z:
        z00 = k_off * n + i_off
        z01 = (k_off+1) * n + i_off
        z10 = k_off * n + (i_off+1)
        z11 = (k_off+1) * n + (i_off+1)
        
        # M1 = (X00+X11)(Y00+Y11)
        U[x00][c] += 1; U[x11][c] += 1
        V[y00][c] += 1; V[y11][c] += 1
        W[z00][c] += 1; W[z11][c] += 1  # Z00 gets +M1, Z11 gets +M1
        c += 1
        
        # M2 = (X10+X11)Y00
        U[x10][c] += 1; U[x11][c] += 1
        V[y00][c] += 1
        W[z10][c] += 1; W[z11][c] -= 1  # Z10 gets +M2, Z11 gets -M2
        c += 1
        
        # M3 = X00(Y01-Y11)
        U[x00][c] += 1
        V[y01][c] += 1; V[y11][c] -= 1
        W[z01][c] += 1; W[z11][c] += 1  # Z01 gets +M3, Z11 gets +M3
        c += 1
        
        # M4 = X11(Y10-Y00)
        U[x11][c] += 1
        V[y10][c] += 1; V[y00][c] -= 1
        W[z00][c] += 1; W[z10][c] += 1  # Z00 gets +M4, Z10 gets +M4
        c += 1
        
        # M5 = (X00+X01)Y11
        U[x00][c] += 1; U[x01][c] += 1
        V[y11][c] += 1
        W[z00][c] -= 1; W[z01][c] += 1  # Z00 gets -M5, Z01 gets +M5
        c += 1
        
        # M6 = (X10-X00)(Y00+Y01)
        U[x10][c] += 1; U[x00][c] -= 1
        V[y00][c] += 1; V[y01][c] += 1
        W[z11][c] += 1  # Z11 gets +M6
        c += 1
        
        # M7 = (X01-X11)(Y10+Y11)
        U[x01][c] += 1; U[x11][c] -= 1
        V[y10][c] += 1; V[y11][c] += 1
        W[z00][c] += 1  # Z00 gets +M7
        c += 1
    
    def add_schoolbook_2x2_x_2x1(i_off, j_off, k_off):
        """A[i_off:i_off+2, j_off:j_off+2] (2x2) × B[j_off:j_off+2, k_off] (2x1) → C[i_off:i_off+2, k_off] (2x1)"""
        nonlocal c
        for ii in range(2):
            for jj in range(2):
                U[(i_off+ii)*m + (j_off+jj)][c] += 1
                V[(j_off+jj)*p + k_off][c] += 1
                W[k_off*n + (i_off+ii)][c] += 1
                c += 1
    
    def add_schoolbook_2x1_x_1x2(i_off, j_off, k_off):
        """A[i_off:i_off+2, j_off] (2x1) × B[j_off, k_off:k_off+2] (1x2) → C[i_off:i_off+2, k_off:k_off+2] (2x2)"""
        nonlocal c
        for ii in range(2):
            for kk in range(2):
                U[(i_off+ii)*m + j_off][c] += 1
                V[j_off*p + (k_off+kk)][c] += 1
                W[(k_off+kk)*n + (i_off+ii)][c] += 1
                c += 1
    
    def add_schoolbook_2x1_x_1x1(i_off, j_off, k_off):
        """A[i_off:i_off+2, j_off] (2x1) × B[j_off, k_off] (1x1) → C[i_off:i_off+2, k_off] (2x1)"""
        nonlocal c
        for ii in range(2):
            U[(i_off+ii)*m + j_off][c] += 1
            V[j_off*p + k_off][c] += 1
            W[k_off*n + (i_off+ii)][c] += 1
            c += 1
    
    # Block 1: A[:,{0,1}] × B[{0,1},{0,1}] → C[{0,1},{0,1}]
    add_strassen_block(0, 0, 0)
    
    # Block 2: A[:,{0,1}] × B[{0,1},{2,3}] → C[{0,1},{2,3}]
    add_strassen_block(0, 0, 2)
    
    # Block 3: A[:,{0,1}] × B[{0,1},{4}] → C[{0,1},{4}]
    add_schoolbook_2x2_x_2x1(0, 0, 4)
    
    # Block 4: A[:,{2,3}] × B[{2,3},{0,1}] → C[{0,1},{0,1}]
    add_strassen_block(0, 2, 0)
    
    # Block 5: A[:,{2,3}] × B[{2,3},{2,3}] → C[{0,1},{2,3}]
    add_strassen_block(0, 2, 2)
    
    # Block 6: A[:,{2,3}] × B[{2,3},{4}] → C[{0,1},{4}]
    add_schoolbook_2x2_x_2x1(0, 2, 4)
    
    # Block 7: A[:,{4}] × B[{4},{0,1}] → C[{0,1},{0,1}]
    add_schoolbook_2x1_x_1x2(0, 4, 0)
    
    # Block 8: A[:,{4}] × B[{4},{2,3}] → C[{0,1},{2,3}]
    add_schoolbook_2x1_x_1x2(0, 4, 2)
    
    # Block 9: A[:,{4}] × B[{4},{4}] → C[{0,1},{4}]
    add_schoolbook_2x1_x_1x1(0, 4, 4)
    
    return U, V, W

def _try_merge(U, V, W, n, m, p, c1, c2, factor):
    """Try to merge columns c1 and c2 by adding factor * c2 to c1 and removing c2."""
    rank = len(U[0])
    if c1 == c2:
        return None
    
    # Create new matrices without column c2
    new_rank = rank - 1
    new_U = [[0] * new_rank for _ in range(n * m)]
    new_V = [[0] * new_rank for _ in range(m * p)]
    new_W = [[0] * new_rank for _ in range(p * n)]
    
    new_col = 0
    for c in range(rank):
        if c == c2:
            continue
        if c == c1:
            # Merge c2 into c1
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
    
    # Verify
    if _verify(new_U, new_V, new_W, n, m, p):
        return new_U, new_V, new_W
    return None

def _search_lower_rank(U, V, W, n, m, p, target_rank, max_time=30.0):
    """Try to find a decomposition of rank target_rank starting from current rank."""
    current_rank = len(U[0])
    if current_rank < target_rank:
        return U, V, W
    
    start_time = time.time()
    
    # Try to reduce rank by merging columns
    # We'll try all pairs and a set of candidate factors
    factors_to_try = [Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2), 
                      Fraction(2), Fraction(-2), Fraction(1,3), Fraction(-1,3),
                      Fraction(3), Fraction(-3)]
    
    while current_rank > target_rank:
        if time.time() - start_time > max_time:
            break
            
        merged = False
        for c1 in range(current_rank):
            for c2 in range(c1 + 1, current_rank):
                for factor in factors_to_try:
                    result = _try_merge(U, V, W, n, m, p, c1, c2, factor)
                    if result is not None:
                        U, V, W = result
                        current_rank -= 1
                        merged = True
                        break
                if merged:
                    break
            if merged:
                break
        
        if not merged:
            break
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with Strassen-based decomposition (rank 46)
    U, V, W = _strassen_decomposition(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = len(U[0])
    
    start_time = time.time()
    
    # Try to find lower rank decompositions
    for target_rank in range(best_rank - 1, 0, -1):
        if time.time() > DEADLINE:
            break
        
        # Give more time for higher targets (closer to current rank)
        time_budget = min(20.0, (DEADLINE - time.time()) * 0.5)
        if time_budget < 5.0:
            break
            
        result = _search_lower_rank(U, V, W, n, m, p, target_rank, max_time=time_budget)
        if result is not None:
            new_rank = len(result[0][0])
            if new_rank < best_rank:
                best_U, best_V, best_W = result
                best_rank = new_rank
                U, V, W = result
                # Continue trying to go lower
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
